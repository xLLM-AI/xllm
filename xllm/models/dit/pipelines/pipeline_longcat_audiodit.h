/* Copyright 2026 The xLLM Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

// LongCat-AudioDiT pipeline for xLLM.
// Ref:
// https://github.com/meituan-longcat/LongCat-AudioDiT/blob/main/audiodit/modeling_audiodit.py
// Architecture:
//   - AudioDiTVae: WAV-VAE encoder/decoder with SnakeBeta activations and
//     weight_norm convolutions.
//   - AudioDiTTransformer: DiT backbone (24 blocks) with:
//       timestep embedding, input/text embedders, ConvNeXtV2 text processing,
//       rotary position embedding, self-attn + cross-attn + FFN per block,
//       global AdaLN, long-skip connection.
//   - UMT5EncoderModel: text encoder (google/umt5-base) from umt5_encoder.h.
//   - ODE solver: inline Euler method.
//   - CFG and APG guidance.
//
// Usage (registered as "audiodit"):
//   DiTForwardInput input;
//   input.prompts = {"a jazz piano piece"};
//   input.generation_params.audio_duration_frames = 500;
//   input.generation_params.audio_steps = 16;
//   input.generation_params.guidance_scale = 4.0f;
//   auto output = model->forward(input);
//   // output.tensors[0] = waveform (1, num_samples)

#pragma once

#include "core/runtime/dit_forward_params.h"
#include "core/util/json_reader.h"
#include "models/dit/transformers/transformer_longcat_audiodit.h"
#include "models/model_registry.h"

namespace xllm {

// ============================================================================
// LongCatAudioDiTPipeline - main pipeline implementation
// ============================================================================

class LongCatAudioDiTPipelineImpl final : public torch::nn::Module {
 public:
  explicit LongCatAudioDiTPipelineImpl(const DiTModelContext& context)
      : context_(context) {
    options_ = context.get_tensor_options();

    LOG(INFO) << "Initializing LongCat-AudioDiT pipeline...";

    // Build VAE with default config matching Python AudioDiTVaeConfig
    AudioDiTVaeConfig vae_cfg;
    vae_cfg.encoder_cfg.in_channels = 1;
    vae_cfg.encoder_cfg.channels = 128;
    vae_cfg.encoder_cfg.c_mults = {1, 2, 4, 8, 16};
    vae_cfg.encoder_cfg.strides = {2, 4, 4, 8, 8};
    vae_cfg.encoder_cfg.encoder_latent_dim = 128;  // 2 * latent_dim
    vae_cfg.encoder_cfg.use_snake = true;
    vae_cfg.encoder_cfg.use_downsample_shortcut =
        true;  // config.downsample_shortcut="averaging"
    vae_cfg.decoder_cfg.in_channels = 1;
    vae_cfg.decoder_cfg.channels = 128;
    vae_cfg.decoder_cfg.c_mults = {1, 2, 4, 8, 16};
    vae_cfg.decoder_cfg.strides = {2, 4, 4, 8, 8};
    vae_cfg.decoder_cfg.latent_dim = 64;
    vae_cfg.decoder_cfg.use_snake = true;
    vae_cfg.decoder_cfg.use_upsample_shortcut =
        true;  // config.upsample_shortcut="duplicating"
    vae_cfg.decoder_cfg.use_in_shortcut =
        true;  // config.in_shortcut="duplicating"
    vae_cfg.decoder_cfg.final_tanh = false;
    vae_cfg.scale = 0.71f;
    vae_cfg.downsampling_ratio = 2048;
    vae_cfg.latent_dim = 64;
    vae_ = register_module("vae", AudioDiTVae(vae_cfg));

    // Build Transformer with default config
    AudioDiTTransformerConfig tr_cfg;
    tr_cfg.dim = 1536;
    tr_cfg.depth = 24;
    tr_cfg.heads = 24;
    tr_cfg.ff_mult = 4.0f;
    tr_cfg.latent_dim = 64;
    tr_cfg.text_dim = 768;
    tr_cfg.long_skip = true;
    tr_cfg.text_conv = true;
    tr_cfg.use_latent_condition = true;
    tr_cfg.adaln_type = "global";
    transformer_ = register_module("transformer", AudioDiTTransformer(tr_cfg));

    // UMT5 text encoder.
    // In the flat layout the context has no per-component args; build ModelArgs
    // from the known UMT5 config embedded in config.json.
    torch::TensorOptions t5_opts =
        context.get_tensor_options().dtype(torch::kFloat32);
    ModelContext t5_ctx;
    if (context.has_component("text_encoder")) {
      t5_ctx = ModelContext(context.get_parallel_args(),
                            context.get_model_args("text_encoder"),
                            context.get_quant_args("text_encoder"),
                            t5_opts);
    } else {
      ModelArgs t5_args;
      t5_args.vocab_size() = 256384;
      t5_args.d_model() = 768;
      t5_args.num_layers() = 12;
      t5_args.n_heads() = 12;
      t5_args.d_kv() = 64;
      t5_args.d_ff() = 2048;
      t5_args.act_fn() = "gelu_new";
      t5_args.is_gated_act() = true;
      t5_args.relative_attention_num_buckets() = 32;
      t5_args.relative_attention_max_distance() = 128;
      t5_args.layer_norm_eps() = 1e-6f;
      t5_ctx = ModelContext(
          context.get_parallel_args(), t5_args, QuantArgs(), t5_opts);
    }
    text_encoder_ = register_module("text_encoder", UMT5TextEncoder(t5_ctx));
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
    const auto& gp = input.generation_params;

    // Batch size
    int64_t batch_size = static_cast<int64_t>(input.prompts.size());
    if (batch_size == 0) {
      batch_size = 1;
    }

    auto device = options_.device();
    auto dtype = options_.dtype().toScalarType();

    // ── 1. Tokenize and encode text ─────────────────────────────────────────
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded for AudioDiT";
    int64_t max_seq_len = gp.max_sequence_length;

    // Tokenize prompts
    std::vector<std::vector<int32_t>> batch_tokens;
    std::vector<int32_t> attention_mask_flat;
    batch_tokens.reserve(batch_size);

    for (int64_t i = 0; i < batch_size; ++i) {
      const std::string& prompt =
          (i < static_cast<int64_t>(input.prompts.size())) ? input.prompts[i]
                                                           : "";
      std::vector<int32_t> tokens;
      if (!tokenizer_->encode(prompt, &tokens, /*add_bos=*/false)) {
        tokens.clear();
      }
      // HuggingFace AutoTokenizer (used in official Python) appends EOS (id=1)
      // after encoding. SentencePiece does not, causing a 1-token difference.
      // Add EOS here to match official tokenization.
      if (!tokens.empty()) {
        tokens.push_back(1);  // EOS token id for UMT5 / SentencePiece
      }
      if (static_cast<int64_t>(tokens.size()) > max_seq_len) {
        tokens.resize(max_seq_len);
      }
      batch_tokens.push_back(std::move(tokens));
    }

    // Pad to max_seq_len and build attention mask
    std::vector<int32_t> input_ids_flat;
    input_ids_flat.reserve(batch_size * max_seq_len);
    attention_mask_flat.reserve(batch_size * max_seq_len);

    for (int64_t i = 0; i < batch_size; ++i) {
      auto& tokens = batch_tokens[i];
      int64_t real_len = static_cast<int64_t>(tokens.size());
      int64_t pad_len = max_seq_len - real_len;
      // pad token id: try to get from tokenizer, fallback to 0
      int32_t pad_id = 0;
      input_ids_flat.insert(input_ids_flat.end(), tokens.begin(), tokens.end());
      input_ids_flat.insert(input_ids_flat.end(), pad_len, pad_id);
      for (int64_t j = 0; j < max_seq_len; ++j) {
        attention_mask_flat.push_back(j < real_len ? 1 : 0);
      }
    }

    torch::Tensor input_ids =
        torch::tensor(input_ids_flat, torch::dtype(torch::kLong))
            .view({batch_size, max_seq_len})
            .to(device);
    torch::Tensor attention_mask =
        torch::tensor(attention_mask_flat, torch::dtype(torch::kBool))
            .view({batch_size, max_seq_len})
            .to(device);
    torch::Tensor text_len = attention_mask.to(torch::kLong).sum(1);  // (B,)

    // Get text embeddings from T5/UMT5
    torch::Tensor text_condition = encode_text(input_ids, attention_mask);
    // (B, max_seq_len, 768) float32

    // Truncate text_condition and attention_mask to actual max token length,
    // matching official Python which never pads text to max_seq_len.
    // Padding zeros corrupt text_embed, ConvNeXtV2 and cross-attention K/V
    // because: (1) Linear bias adds non-zero values on pad positions before the
    // second masked_fill; (2) ConvNeXtV2 depthwise conv leaks bias across pad
    // positions; (3) cross-attention attends over all 512 positions including
    // dead ones.  Truncating removes all this noise.
    // encode_text already slices to actual_len internally, so text_condition
    // is (B, actual_len, 768). Always truncate attention_mask to match.
    int64_t actual_text_len = text_condition.size(1);
    attention_mask = attention_mask.slice(1, 0, actual_text_len);
    // Null text for CFG/APG unconditional pass.
    // Python reference uses neg_text=zeros with cond_mask=text_mask (the real
    // mask, all-True for actual tokens).  AudioDiTEmbedder's two masked_fills
    // only zero out *padding* positions, so the linear-bias output on real
    // positions remains non-zero (text_std≈0.66 per reference log).
    // Using an all-False mask would zero the entire text embedding and
    // diverges.
    torch::Tensor neg_text = torch::zeros_like(text_condition);
    torch::Tensor neg_text_mask =
        attention_mask;  // same real mask as cond pass
    torch::Tensor neg_text_len = text_len.clone();  // keeps divisor non-zero

    // ── 2. Encode prompt audio if provided ──────────────────────────────────
    torch::Tensor prompt_latent;  // (B, prompt_frames, latent_dim)
    int64_t prompt_dur = 0;

    if (input.prompt_audio.defined() && input.prompt_audio.numel() > 0) {
      std::tie(prompt_latent, prompt_dur) =
          encode_prompt_audio(input.prompt_audio);
    } else {
      prompt_latent =
          torch::empty({batch_size, 0, vae_->latent_dim_},
                       torch::dtype(torch::kFloat32).device(device));
    }

    // ── 3. Duration ─────────────────────────────────────────────────────────
    // audio_duration_frames: total latent frames (prompt + gen)
    // 0 means auto-estimate from text, matching official Python:
    //   approx_duration_from_text(text,
    //   max_duration=max_wav_duration-prompt_time)
    //   + ratio adjustment when prompt audio is present.
    int64_t max_dur;
    if (gp.audio_duration_frames > 0) {
      max_dur = static_cast<int64_t>(gp.audio_duration_frames);
    } else {
      const int64_t kSampleRate = sampling_rate_;
      constexpr float kMaxDuration = 30.0f;
      constexpr float kEnDurPerChar = 0.082f;
      constexpr float kZhDurPerChar = 0.21f;

      // Count CJK / EN chars and estimate duration in seconds.
      // Matches official Python utils.approx_duration_from_text.
      auto approx_dur = [&](const std::string& text,
                            float max_dur_sec) -> float {
        int nzh = 0, nen = 0, nother = 0;
        for (size_t ci = 0; ci < text.size();) {
          unsigned char c = static_cast<unsigned char>(text[ci]);
          if (c >= 0xE0 && ci + 2 < text.size()) {
            uint32_t cp =
                ((c & 0x0F) << 12) |
                ((static_cast<unsigned char>(text[ci + 1]) & 0x3F) << 6) |
                (static_cast<unsigned char>(text[ci + 2]) & 0x3F);
            if (cp >= 0x4E00 && cp <= 0x9FFF)
              ++nzh;
            else
              ++nother;
            ci += 3;
          } else if (c < 0x80) {
            if (std::isalpha(c))
              ++nen;
            else if (c != ' ' && c != '\t' && c != '\n')
              ++nother;
            ++ci;
          } else {
            ++nother;
            ++ci;
          }
        }
        if (nzh > nen)
          nzh += nother;
        else
          nen += nother;
        float d = nzh * kZhDurPerChar + nen * kEnDurPerChar;
        return std::min(d, max_dur_sec);
      };

      // The synthesis text is everything after "prompt_text " prefix.
      // input.prompts[0] is full_text = prompt_text + " " + text (when voice
      // cloning) or just text (TTS). audio_prompt_text holds the prompt_text
      // portion so we can split them.
      const std::string& full_text =
          input.prompts.empty() ? "" : input.prompts[0];
      const std::string& prompt_text_part = input.audio_prompt_text;

      // Extract synthesis-only text
      std::string synth_text = full_text;
      if (!prompt_text_part.empty() &&
          full_text.size() > prompt_text_part.size()) {
        // full_text = prompt_text + " " + text  →  skip prefix + space
        synth_text = full_text.substr(prompt_text_part.size() + 1);
      }

      float prompt_time = static_cast<float>(prompt_dur) *
                          vae_->downsampling_ratio_ / kSampleRate;
      float max_synth_dur = kMaxDuration - prompt_time;
      float dur_sec = approx_dur(synth_text, max_synth_dur);
      dur_sec = std::max(dur_sec, 1.0f);

      // Voice cloning: apply ratio = clip(prompt_time / approx_pd, 1.0, 1.5)
      if (prompt_dur > 0 && !prompt_text_part.empty()) {
        float approx_pd = approx_dur(prompt_text_part, kMaxDuration);
        if (approx_pd > 0.0f) {
          float ratio = std::min(1.5f, std::max(1.0f, prompt_time / approx_pd));
          dur_sec *= ratio;
        }
      }

      int64_t gen_frames = static_cast<int64_t>(dur_sec * kSampleRate /
                                                vae_->downsampling_ratio_);
      max_dur = gen_frames + prompt_dur;
      max_dur = std::min(max_dur,
                         static_cast<int64_t>(kMaxDuration * kSampleRate /
                                              vae_->downsampling_ratio_));
    }
    max_dur = std::max(max_dur, prompt_dur + 1);

    // ── 4. Build masks ───────────────────────────────────────────────────────
    // mask: (B, max_dur) all valid (True)
    torch::Tensor mask = torch::ones({batch_size, max_dur},
                                     torch::dtype(torch::kBool).device(device));
    torch::Tensor text_mask = attention_mask;  // (B, max_seq_len)

    // ── 5. Latent conditioning ───────────────────────────────────────────────
    int64_t gen_len = max_dur - prompt_dur;
    torch::Tensor latent_cond;
    torch::Tensor empty_latent_cond;
    if (prompt_dur > 0) {
      // Pad prompt_latent to full duration
      latent_cond = torch::nn::functional::pad(
          prompt_latent,
          torch::nn::functional::PadFuncOptions({0, 0, 0, gen_len}));
      empty_latent_cond = torch::zeros_like(latent_cond);
    } else {
      latent_cond = torch::zeros({batch_size, max_dur, vae_->latent_dim_},
                                 torch::dtype(torch::kFloat32).device(device));
      empty_latent_cond = latent_cond;
    }

    // ── 6. Initialize noise ──────────────────────────────────────────────────
    int64_t seed = gp.seed;
    torch::TensorOptions noise_opts =
        torch::dtype(torch::kFloat32).device(device);
    std::vector<int64_t> noise_shape = {batch_size, max_dur, vae_->latent_dim_};
    torch::Tensor y0 = xllm::dit::randn_tensor(noise_shape, seed, noise_opts);

    // ── 7. ODE Euler solve ───────────────────────────────────────────────────
    int64_t steps = (gp.audio_steps > 0) ? gp.audio_steps : 16;
    float cfg_strength = gp.guidance_scale;
    const std::string& guidance_method = gp.audio_guidance_method;

    // t: [0, 1] with `steps` points (inclusive)
    torch::Tensor t_schedule = torch::linspace(
        0.0f, 1.0f, steps, torch::dtype(torch::kFloat32).device(device));

    // Snapshot of prompt frames' noise (for blending at each step)
    torch::Tensor prompt_noise;
    if (prompt_dur > 0) {
      prompt_noise = y0.slice(1, 0, prompt_dur).clone();
    }

    // APG momentum buffer
    MomentumBuffer apg_buf;
    apg_buf.momentum = -0.3f;

    // Euler integration
    torch::Tensor y = y0;
    for (int64_t i = 0; i < steps - 1; ++i) {
      float t_val = t_schedule[i].item<float>();
      float dt = t_schedule[i + 1].item<float>() - t_val;
      torch::Tensor t_tensor = torch::full({batch_size}, t_val, noise_opts);

      // Blend prompt frames
      if (prompt_dur > 0) {
        torch::Tensor blended = prompt_noise * (1.0f - t_val) +
                                latent_cond.slice(1, 0, prompt_dur) * t_val;
        y = y.clone();
        y.slice(1, 0, prompt_dur) = blended;
      }

      // Conditional forward
      torch::Tensor pred = transformer_->forward(y.to(torch::kFloat32),
                                                 text_condition,
                                                 text_len,
                                                 t_tensor,
                                                 mask,
                                                 text_mask,
                                                 latent_cond);
      if (cfg_strength < 1e-5f) {
        // No guidance
        y = (y + pred * dt).detach();
        continue;
      }

      // Zero out prompt frames for unconditional pass
      torch::Tensor y_null = y.clone();
      if (prompt_dur > 0) {
        y_null.slice(1, 0, prompt_dur).zero_();
      }
      torch::Tensor null_pred =
          transformer_->forward(y_null.to(torch::kFloat32),
                                neg_text,
                                neg_text_len,
                                t_tensor,
                                mask,
                                neg_text_mask,
                                empty_latent_cond);
      torch::Tensor guided_pred;
      if (guidance_method == "apg") {
        // APG guidance on the generated frames only
        torch::Tensor x_s = y.slice(1, prompt_dur);
        torch::Tensor pred_s = pred.slice(1, prompt_dur);
        torch::Tensor null_s = null_pred.slice(1, prompt_dur);
        torch::Tensor pred_sample = x_s + (1.0f - t_val) * pred_s;
        torch::Tensor null_sample = x_s + (1.0f - t_val) * null_s;
        torch::Tensor apg_out = apg_forward(pred_sample,
                                            null_sample,
                                            cfg_strength,
                                            &apg_buf,
                                            /*eta=*/0.5f,
                                            /*norm_threshold=*/0.0f);
        torch::Tensor apg_velocity = (apg_out - x_s) / (1.0f - t_val + 1e-9f);
        // Pad back prompt frames with zeros
        guided_pred = torch::nn::functional::pad(
            apg_velocity,
            torch::nn::functional::PadFuncOptions({0, 0, prompt_dur, 0}));
      } else {
        // CFG: pred + (pred - null_pred) * cfg_strength
        guided_pred =
            pred.to(torch::kFloat32) +
            (pred.to(torch::kFloat32) - null_pred.to(torch::kFloat32)) *
                cfg_strength;
      }

      y = (y.to(torch::kFloat32) + guided_pred * dt).detach();
    }

    // ── 8. Decode latent -> waveform ─────────────────────────────────────────
    // Use only the generated frames (skip prompt frames)
    torch::Tensor pred_latent = y.slice(1, prompt_dur);  // (B, gen_len, 64)
    pred_latent = pred_latent.permute({0, 2, 1}).to(torch::kFloat32);
    // (B, 64, gen_len)
    torch::Tensor waveform;
    try {
      waveform = vae_->decode(pred_latent).squeeze(1);  // (B, num_samples)
    } catch (const std::exception& e) {
      LOG(ERROR) << "[LongCat-AudioDiT] VAE decode failed: " << e.what();
      waveform = torch::zeros({batch_size, vae_->downsampling_ratio_ * gen_len},
                              noise_opts);
    }

    waveform = waveform.cpu().to(torch::kFloat32).contiguous();
    DiTForwardOutput out;
    out.tensors = torch::chunk(waveform, waveform.size(0), 0);
    return out;
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
    LOG(INFO) << "LongCat-AudioDiT pipeline loading model from "
              << loader->model_root_path();

    // Read sampling_rate from config.json, matching official Python's
    // model.config.sampling_rate. Falls back to the default (24000) if absent.
    {
      const std::string config_path =
          loader->model_root_path() + "/config.json";
      JsonReader cfg;
      if (cfg.parse(config_path)) {
        if (auto v = cfg.value<int32_t>("sampling_rate")) {
          sampling_rate_ = v.value();
        }
      }
    }

    if (loader->has_component("transformer")) {
      // Component layout: separate subdirectories per component.
      auto transformer_loader = loader->take_component_loader("transformer");
      auto vae_loader = loader->take_component_loader("vae");
      auto text_encoder_loader = loader->take_component_loader("text_encoder");
      auto tokenizer_loader = loader->take_component_loader("tokenizer");

      load_module_from_state_dicts(*transformer_loader,
                                   transformer_.ptr().get());
      transformer_->to(options_.device());
      // Official inference runs transformer in float32, VAE in float16.
      transformer_->to(torch::kFloat32);

      load_module_from_state_dicts(*vae_loader, vae_.ptr().get());
      vae_->to(options_.device());
      vae_->to_half();

      text_encoder_->load_model(std::move(text_encoder_loader));
      text_encoder_->to(options_.device());
      // Official runs text encoder in float32 (same as transformer).
      text_encoder_->to(torch::kFloat32);

      tokenizer_ = tokenizer_loader->tokenizer();
    } else {
      // Flat layout: single model.safetensors at root contains all weights.
      // Keys are prefixed: "transformer.*", "vae.*", "text_encoder.*".
      auto flat_loader = loader->take_component_loader("model");

      load_module_from_state_dicts(
          *flat_loader, transformer_.ptr().get(), "transformer.");
      transformer_->to(options_.device());
      // Official inference runs transformer in float32, VAE in float16.
      transformer_->to(torch::kFloat32);

      load_module_from_state_dicts(*flat_loader, vae_.ptr().get(), "vae.");
      vae_->to(options_.device());
      vae_->to_half();

      text_encoder_->load_model_from_state_dicts(
          flat_loader->get_state_dicts());
      text_encoder_->to(options_.device());
      // Official runs text encoder in float32 (same as transformer).
      text_encoder_->to(torch::kFloat32);

      tokenizer_ = flat_loader->tokenizer();
    }
  }

 private:
  // Encode text using T5/UMT5 encoder
  // input_ids: (B, S) int64, attention_mask: (B, S) bool
  // Returns: (B, S, 768) float32
  torch::Tensor encode_text(const torch::Tensor& input_ids,
                            const torch::Tensor& attention_mask) {
    torch::NoGradGuard no_grad;
    // Truncate to actual token length before T5 encoding to match official
    // Python which never pads (tokenizer output has exact sequence length).
    // Processing 512 tokens when only 25 are real pollutes T5 self-attention
    // with 487 padding positions and changes the output for real tokens.
    int64_t actual_len =
        attention_mask.to(torch::kLong).sum(1).max().item<int64_t>();
    torch::Tensor ids = input_ids.slice(1, 0, actual_len);
    return text_encoder_->forward(ids);
  }

  // Encode prompt audio (B, 1, T) -> latent (B, frames, 64) + prompt_frames
  std::pair<torch::Tensor, int64_t> encode_prompt_audio(
      const torch::Tensor& prompt_audio) {
    auto device = options_.device();
    int64_t full_hop = vae_->downsampling_ratio_;
    constexpr int64_t kOffset = 3;  // extra frames offset

    torch::Tensor wav = prompt_audio.to(device);
    if (wav.dim() == 2) {
      wav = wav.unsqueeze(1);  // (B, 1, T)
    }

    // Pad to multiple of full_hop
    int64_t T = wav.size(2);
    if (T % full_hop != 0) {
      int64_t pad_amt = full_hop - (T % full_hop);
      wav = torch::nn::functional::pad(
          wav, torch::nn::functional::PadFuncOptions({0, pad_amt}));
    }
    // Extra offset padding
    wav = torch::nn::functional::pad(
        wav, torch::nn::functional::PadFuncOptions({0, full_hop * kOffset}));

    torch::Tensor latent = vae_->encode(wav);  // (B, 64, T')
    // Remove offset frames
    if (kOffset != 0) {
      latent = latent.slice(2, 0, latent.size(2) - kOffset);
    }
    int64_t prompt_frames = latent.size(2);
    return {latent.permute({0, 2, 1}).to(torch::kFloat32), prompt_frames};
    // Returns (B, prompt_frames, 64)
  }

  // Member variables
  DiTModelContext context_;
  torch::TensorOptions options_;

  // Audio config read from config.json (matches official
  // model.config.sampling_rate)
  int32_t sampling_rate_ = 24000;

  // Model components
  AudioDiTVae vae_{nullptr};
  AudioDiTTransformer transformer_{nullptr};
  UMT5TextEncoder text_encoder_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
};
TORCH_MODULE(LongCatAudioDiTPipeline);

// ============================================================================
// Model registration
// ============================================================================
// Register under "audiodit" to match the model_type field in config.json.
namespace {
const bool longcat_audio_dit_registered = []() {
  ModelRegistry::register_dit_model_factory(
      "audiodit", [](const DiTModelContext& context) {
        LongCatAudioDiTPipeline model(context);
        model->eval();
        return std::make_unique<DiTModelImpl<LongCatAudioDiTPipeline>>(
            std::move(model), context.get_tensor_options());
      });
  return true;
}();
}  // namespace

}  // namespace xllm
