/* Copyright 2025-2026 The xLLM Authors.

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

#pragma once

#include <torch/torch.h>

#if defined(USE_CUDA)
#include <ATen/autocast_mode.h>
#include <ATen/cuda/CUDAGeneratorImpl.h>
#endif

#include <atomic>
#include <filesystem>
#include <limits>
#include <string>
#include <vector>

#include "core/framework/dit_model_loader.h"
#include "core/framework/model/dit_model.h"
#include "core/framework/model_context.h"
#include "core/framework/tokenizer/fast_tokenizer.h"
#include "core/framework/tokenizer/tokenizer_args.h"
#include "core/util/json_reader.h"
#include "models/dit/autoencoders/autoencoder_text_vae_cola.h"
#include "models/dit/transformers/transformer_cola_dit.h"
#include "models/model_registry.h"

namespace xllm {

// ---------------------------------------------------------------------------
// ColaDLMPipeline — main pipeline for Cola-DLM text diffusion inference
// ---------------------------------------------------------------------------
// Reference: cola_dlm/inference.py generate_task_repaint_inference()
//
// Implements the three-step inference algorithm:
// 1. Prefix encode: z^pre = VAE.encode(x^pre)
// 2. Block-wise latent prior transport with CFG + Euler integration
// 3. Conditional decode: x^res = VAE.decode(z)

class ColaDLMPipelineImpl final : public torch::nn::Module {
 public:
  explicit ColaDLMPipelineImpl(const DiTModelContext& context) {
    options_ = context.get_tensor_options();

    // Create DiT transformer
    dit_ = register_module(
        "dit", ColaDiTTransformer(context.get_model_context("cola_dit")));

    // Create VAE
    vae_ = register_module(
        "vae", ColaTextVAEModel(context.get_model_context("cola_vae")));
  }

  void load_model(std::unique_ptr<DiTModelLoader> loader) {
#if defined(USE_CUDA)
    CHECK(options_.device().is_cuda())
        << "Cola-DLM is only supported on CUDA devices.";
#else
    LOG(FATAL) << "Cola-DLM is only supported in CUDA builds.";
#endif

    if (loader->has_component("cola_dit")) {
      // Component layout: separate subdirectories per component.
      auto dit_loader = loader->take_component_loader("cola_dit");
      auto vae_loader = loader->take_component_loader("cola_vae");

      dit_->load_model(std::move(dit_loader));
      dit_->to(options_.device());

      vae_->load_model(std::move(vae_loader));
      vae_->to(options_.device());

      // Load tokenizer: either from component or from tokenizer.json in root
      if (loader->has_component("tokenizer")) {
        auto tokenizer_loader = loader->take_component_loader("tokenizer");
        tokenizer_ = tokenizer_loader->tokenizer();
      } else {
        // Look for tokenizer.json in the model root directory
        const std::string root_path = loader->model_root_path();
        std::filesystem::path tokenizer_path =
            std::filesystem::path(root_path) / "tokenizer.json";
        if (!std::filesystem::exists(tokenizer_path)) {
          // Also check parent directory (for nested layouts like
          // model_root/cola_dlm/ where tokenizer.json is in model_root/)
          tokenizer_path =
              std::filesystem::path(root_path).parent_path() / "tokenizer.json";
        }
        CHECK(std::filesystem::exists(tokenizer_path))
            << "tokenizer.json not found in model root or parent: "
            << root_path;
        TokenizerArgs tokenizer_args;
        tokenizer_args.vocab_file(tokenizer_path.string());
        tokenizer_args.tokenizer_type("fast");
        tokenizer_ = std::make_unique<FastTokenizer>(tokenizer_args);
      }
    } else {
      LOG(FATAL) << "Cola-DLM: required component 'cola_dit' not found under "
                 << loader->model_root_path()
                 << ". Expected cola_dit/ and cola_vae/ component "
                    "subdirectories (from model_index.json or auto-discovery), "
                    "plus tokenizer.json in the model root or a tokenizer "
                    "component.";
    }

    // Read block_size from DiT config.json
    // Try multiple possible locations for nested directory layouts
    std::vector<std::string> config_candidates = {
        loader->model_root_path() + "/cola_dit/config.json",
        loader->model_root_path() + "/cola_dlm/cola_dit/config.json",
    };
    // Also check parent directory for nested layouts
    std::filesystem::path root(loader->model_root_path());
    if (root.has_parent_path()) {
      config_candidates.push_back(root.parent_path().string() +
                                  "/cola_dit/config.json");
    }
    for (const auto& config_path : config_candidates) {
      JsonReader cfg;
      if (cfg.parse(config_path)) {
        if (auto v = cfg.value<int32_t>("block_size")) {
          block_size_ = v.value();
          break;
        }
      }
    }
  }

  DiTForwardOutput forward(const DiTForwardInput& input) {
    torch::NoGradGuard no_grad;
#if defined(USE_CUDA)
    CHECK(options_.device().is_cuda())
        << "Cola-DLM is only supported on CUDA devices.";
#else
    LOG(FATAL) << "Cola-DLM is only supported in CUDA builds.";
#endif

    CHECK_LE(input.batch_size, 1)
        << "Cola-DLM text generation supports batch_size=1 only.";
    CHECK(!input.prompts.empty()) << "Cola-DLM: missing prompt.";

    const auto& gp = input.generation_params;
    auto device = options_.device();

    // Generation parameters from TextGenerationRequest.
    int64_t max_new_tokens = gp.max_new_tokens;
    int32_t diffusion_steps = gp.diffusion_steps;
    float temperature = gp.temperature;
    int32_t top_k = gp.top_k;
    float top_p = gp.top_p;
    float repetition_penalty = gp.repetition_penalty;
    float guidance_scale = gp.guidance_scale > 0 ? gp.guidance_scale : 7.0f;
    int64_t seed = gp.seed;
    const bool seed_is_set = gp.seed_is_set;

    // RNG policy (implemented below, Step 5):
    // - seed_is_set && seed >= 0: per-block CUDAGenerator with COLA formula
    //   (COLA_INFER_PER_SAMPLE_NOISE_SEED).
    // - otherwise: advance forward_counter_ and call torch::cuda::manual_seed
    //   once per forward, then draw block noise via global torch::randn
    //   (matches official run_cola.py stochastic path).

    int64_t block_size = block_size_;
    int64_t patch_size = vae_->patch_size();
    int64_t chunk = patch_size * block_size;

    constexpr float kTimestepScale = 1000.0f;

    // -----------------------------------------------------------------
    // Step 1: Tokenize + block-align pad
    // -----------------------------------------------------------------
    CHECK(tokenizer_ != nullptr) << "Tokenizer not loaded";

    std::string prompt_text = input.prompts.empty() ? "" : input.prompts[0];
    std::vector<int32_t> tokens;
    CHECK(tokenizer_->encode(prompt_text, &tokens, /*add_bos=*/false))
        << "Cola-DLM: tokenizer encode failed for prompt: " << prompt_text;
    CHECK(!tokens.empty())
        << "Cola-DLM: empty tokens after encoding for prompt: " << prompt_text;

    // Pad to multiple of chunk (patch_size * block_size)
    int64_t orig_len = static_cast<int64_t>(tokens.size());
    int64_t pad_len = (chunk - orig_len % chunk) % chunk;
    constexpr const char* kPadToken = "<|pad|>";
    const std::optional<int32_t> pad_token_id =
        tokenizer_->token_to_id(kPadToken);
    CHECK(pad_token_id.has_value())
        << "Cola-DLM: pad token '" << kPadToken << "' not found in tokenizer";
    const int32_t pad_token_id_value = pad_token_id.value();
    for (int64_t i = 0; i < pad_len; ++i) {
      tokens.push_back(pad_token_id_value);
    }

    auto input_ids =
        torch::tensor(tokens, torch::dtype(torch::kLong)).to(device);

    // Token labels: 1 = real, 3 = padding
    std::vector<int64_t> token_labels_vec(orig_len, 1);
    token_labels_vec.resize(orig_len + pad_len, 3);
    auto token_labels =
        torch::tensor(token_labels_vec, torch::dtype(torch::kLong)).to(device);

    // -----------------------------------------------------------------
    // Step 2: VAE encode
    // -----------------------------------------------------------------
    std::vector<torch::Tensor> input_ids_list = {input_ids};
#if defined(USE_CUDA)
    torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
    torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
    auto [latents, sample_lens] = vae_->encode(input_ids_list);
#if defined(USE_CUDA)
    torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif
    latents = latents.to(torch::kFloat32);  // match Python .float()

    // -----------------------------------------------------------------
    // Step 3: Latent labels + prefix / first-block split
    // (matches official Python generate_task_repaint_inference)
    // -----------------------------------------------------------------
    int64_t n_patches = latents.size(0);
    int64_t latent_dim = latents.size(1);

    torch::Tensor latent_labels;
    if (patch_size > 1) {
      auto reshaped = token_labels.reshape({n_patches, patch_size});
      auto c1 = reshaped.eq(1).any(/*dim=*/-1);
      auto c2 = reshaped.eq(2).any(/*dim=*/-1);
      latent_labels = torch::full(
          {n_patches},
          3,
          torch::TensorOptions().dtype(torch::kLong).device(device));
      latent_labels.masked_fill_(c2, 2);
      latent_labels.masked_fill_(c1, 1);
    } else {
      latent_labels = token_labels;
    }

    int64_t num_real = latent_labels.eq(1).sum().item<int64_t>();

    torch::Tensor prefix_latents;
    torch::Tensor first_block_latents;
    int64_t first_block_prompt_tokens = 0;

    if (num_real % block_size != 0) {
      int64_t start_idx = (num_real / block_size) * block_size;
      prefix_latents = latents.slice(0, 0, start_idx);

      if (start_idx + block_size <= n_patches) {
        first_block_latents =
            latents.slice(0, start_idx, start_idx + block_size).clone();
        // Prompt token count inside the first gen block.
        int64_t token_start = start_idx * patch_size;
        int64_t token_end = std::min(token_start + block_size * patch_size,
                                     token_labels.size(0));
        first_block_prompt_tokens =
            token_labels.slice(0, token_start, token_end)
                .eq(1)
                .sum()
                .item<int64_t>();
      } else {
        prefix_latents = latents.slice(0, 0, num_real);
        first_block_latents =
            latents.slice(0, n_patches - block_size, n_patches).clone();
      }
    } else {
      prefix_latents = latents.slice(0, 0, num_real);
      first_block_latents =
          latents.slice(0, n_patches - block_size, n_patches).clone();
    }

    int64_t prefix_len = prefix_latents.size(0);

    int64_t max_blocks = (max_new_tokens + block_size * patch_size - 1) /
                         (block_size * patch_size);

    // Per-sample CFG scale for block 0: when prefix is empty, cond == uncond
    // so CFG just amplifies bf16 noise; fall back to scale=1.
    float cfg_scale_block0 = (prefix_len == 0) ? 1.0f : guidance_scale;

    // -----------------------------------------------------------------
    // Step 4: Enable per-layer KV caches on DiT and VAE decoder.
    // -----------------------------------------------------------------
    dit_->set_kv_cache(true);
    vae_->set_kv_cache(true);

    // -----------------------------------------------------------------
    // Step 4a: Prefix prefetch — write prefix K/V into DiT and VAE caches.
    // (Matches official Python inference.py lines 499-517.)
    // -----------------------------------------------------------------
    if (prefix_len > 0) {
      std::vector<int64_t> prefix_k_lens = {prefix_len};

      // DiT prefix prefetch at timestep=0.
      auto ts_prefix = torch::zeros(
          {prefix_len},
          torch::TensorOptions().dtype(torch::kBFloat16).device(device));
#if defined(USE_CUDA)
      torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
      torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
      dit_->forward(prefix_latents.to(torch::kBFloat16),
                    prefix_k_lens,
                    prefix_k_lens,
                    ts_prefix,
                    /*update_kv=*/true,
                    /*use_kv_cache=*/true);
#if defined(USE_CUDA)
      torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif

      // VAE decoder prefix prefetch.
#if defined(USE_CUDA)
      torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
      torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
      vae_->decode(prefix_latents,
                   prefix_k_lens,
                   prefix_k_lens,
                   /*update_kv=*/true);
#if defined(USE_CUDA)
      torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif
    }

    // -----------------------------------------------------------------
    // Step 5: Block-wise generation loop
    // (Matches official Python inference.py lines 571-706.)
    // -----------------------------------------------------------------
    // Per-sample cumulative K length, initially = prefix_len.
    int64_t k_len_cum = prefix_len;

    std::vector<int64_t> all_token_ids;
    const int64_t sample_id = 0;

    // Global diffusion schedule shared by every generation block, matching
    // inference.py: linspace(T, 0, timestep_num + 1). Block 0 still uses
    // t=T for to-generate positions on the first Euler step; prompt positions
    // inside that block are pinned to t=0 via clean-guidance (so the per-token
    // ts mean can look like 312.5 when 11/16 positions are prompt — that is not
    // a reduced t_start).
    auto timesteps = torch::linspace(
        kTimestepScale, 0.0f, diffusion_steps + 1, torch::kFloat32);

    // When seed_is_set && seed >= 0: COLA deterministic per-block noise
    // (matches COLA_INFER_PER_SAMPLE_NOISE_SEED).  When seed is omitted or
    // seed < 0: global torch.randn (matches official run_cola.py default).
    const bool use_deterministic_noise = seed_is_set && (seed >= 0);
    if (!use_deterministic_noise) {
      const uint64_t request_seed =
          (++forward_counter_) * 6364136223846793005ULL +
          1442695040888963407ULL;
#if defined(USE_CUDA)
      torch::cuda::manual_seed(request_seed);
#endif
    }

    for (int64_t block_idx = 0; block_idx < max_blocks; ++block_idx) {
      // After adding current block, total K = k_len_cum + block_size.
      k_len_cum += block_size;
      std::vector<int64_t> k_lens_cond = {k_len_cum};
      std::vector<int64_t> q_lens = {block_size};
      std::vector<int64_t> k_lens_uncond = {block_size};

      float block_cfg_scale =
          (block_idx == 0) ? cfg_scale_block0 : guidance_scale;

      // Draw initial block noise.
      //   seed_is_set && seed >= 0 → COLA formula
      //   (COLA_INFER_PER_SAMPLE_NOISE_SEED) otherwise → global torch.randn
      //   (run_cola.py / cola-log.txt style)
      torch::Tensor txt;
      if (use_deterministic_noise) {
#if defined(USE_CUDA)
        torch::Generator noise_gen =
            torch::make_generator<torch::CUDAGeneratorImpl>(
                static_cast<torch::DeviceIndex>(device.index()));
        const uint64_t effective_seed = static_cast<uint64_t>(
            seed + sample_id * 1000LL + block_idx * 10'000'000LL);
        noise_gen.set_current_seed(effective_seed);
        txt = torch::randn(
            {block_size, latent_dim}, noise_gen, latents.options());
#else
        txt = torch::randn({block_size, latent_dim}, latents.options());
#endif
      } else {
        txt = torch::randn({block_size, latent_dim}, latents.options());
      }

      // --- Euler integration loop ------------------------------------------
      for (int64_t t_idx = 0; t_idx < diffusion_steps; ++t_idx) {
        float t_curr = timesteps[t_idx].item<float>();
        float t_next = timesteps[t_idx + 1].item<float>();
        float dt = (t_curr - t_next) / kTimestepScale;

        // Block 0 clean-guidance: pin prompt latent positions to ground truth
        // and fix their timestep to 0 throughout ALL diffusion steps.
        // flat_mask has 'first_block_prompt_tokens' True entries at positions
        // [0..first_block_prompt_tokens-1] (all prompt latents are contiguous
        // because padding comes after the prompt).
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }

        // Timestep tensor for current block only (bf16, matching official
        // inference.py ts_bf16 passed into ColaDiTModel.forward).
        auto ts_tensor = torch::full(
            {block_size},
            t_curr,
            torch::TensorOptions().dtype(torch::kBFloat16).device(device));
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          // Zero timestep for prompt positions (already clean).
          ts_tensor.slice(0, 0, first_block_prompt_tokens).fill_(0.0f);
        }

        // --- Conditional + Unconditional DiT passes (bf16 autocast) ---
        torch::Tensor drift_cond;
        torch::Tensor drift_uncond;
        {
#if defined(USE_CUDA)
          torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
          torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
          drift_cond = dit_->forward(txt.to(torch::kBFloat16),
                                     k_lens_cond,
                                     q_lens,
                                     ts_tensor,
                                     /*update_kv=*/false,
                                     /*use_kv_cache=*/true);
          drift_uncond = dit_->forward(txt.to(torch::kBFloat16),
                                       k_lens_uncond,
                                       q_lens,
                                       ts_tensor,
                                       /*update_kv=*/false,
                                       /*use_kv_cache=*/false);
#if defined(USE_CUDA)
          torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif
        }

        // CFG combination in bf16 (matches official inference.py), then
        // Euler update in fp32 because txt latents are fp32.
        auto drift_bf16 =
            block_cfg_scale * (drift_cond.to(torch::kBFloat16) -
                               drift_uncond.to(torch::kBFloat16)) +
            drift_uncond.to(torch::kBFloat16);
        txt = txt - drift_bf16.to(torch::kFloat32) * dt;

        // Re-pin prompt positions after Euler step.
        if (block_idx == 0 && first_block_prompt_tokens > 0) {
          txt.slice(0, 0, first_block_prompt_tokens) =
              first_block_latents.slice(0, 0, first_block_prompt_tokens);
        }
      }

      // --- VAE decode current block (update_kv=True commits to VAE cache) ---
#if defined(USE_CUDA)
      torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
      torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
      auto decoded = vae_->decode(txt,
                                  k_lens_cond,
                                  q_lens,
                                  /*update_kv=*/true);
#if defined(USE_CUDA)
      torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif

      // decoded: (1, block_size*patch_size, vocab)
      auto block_logits = decoded.squeeze(0);  // (block_tokens, vocab)
      int64_t block_tokens = block_size * patch_size;

      // Greedy or sampling decode — matches official sample_with_strategies().
      //
      // Apply repetition_penalty first (before temperature/top_k/top_p),
      // exactly as the official sample_with_strategies() does even for greedy
      // (temperature=0) decoding.
      // repetition_penalty=1.0 is a no-op; values > 1.0 penalise reuse.
      auto logits_rep = block_logits.clone();
      if (repetition_penalty != 1.0f && !all_token_ids.empty()) {
        // Build a (block_tokens, n_prev) index tensor of previously generated
        // token IDs, then gather their current logit scores, apply the penalty,
        // and scatter back.
        auto prev_ids = torch::tensor(
            std::vector<int64_t>(all_token_ids.begin(), all_token_ids.end()),
            torch::TensorOptions().dtype(torch::kLong).device(device));
        // Broadcast prev_ids to (block_tokens, n_prev)
        int64_t n_prev = prev_ids.size(0);
        auto prev_ids_exp =
            prev_ids.unsqueeze(0).expand({block_tokens, n_prev});
        // Gather logit for each prev token at every generated position
        auto scores = torch::gather(logits_rep, /*dim=*/1, prev_ids_exp);
        // Penalty: if logit < 0 → multiply by penalty; else → divide
        scores = torch::where(scores < 0,
                              scores * repetition_penalty,
                              scores / repetition_penalty);
        logits_rep.scatter_(/*dim=*/1, prev_ids_exp, scores);
      }

      torch::Tensor block_ids;
      if (temperature < 1e-5f) {
        block_ids = logits_rep.argmax(/*dim=*/-1);  // (block_tokens,)
      } else {
        // Apply temperature scaling in logit space (before softmax).
        // Use logits_rep (with repetition penalty already applied).
        auto logits = logits_rep / temperature;

        // top_k: keep only the top-k logits, set the rest to -inf.
        // Must be done in LOGIT space so that softmax re-normalises correctly.
        if (top_k > 0) {
          int64_t k = std::min(top_k, static_cast<int32_t>(logits.size(-1)));
          auto [topk_values, topk_indices] = torch::topk(logits, k, /*dim=*/-1);
          // threshold = k-th largest logit per row
          auto min_topk = topk_values.select(-1, k - 1).unsqueeze(-1);
          logits = torch::where(
              logits < min_topk,
              torch::full_like(logits, -std::numeric_limits<float>::infinity()),
              logits);
        }

        // top_p (nucleus sampling): keep the smallest set whose cumulative
        // softmax probability >= top_p, set the rest to -inf.
        if (top_p > 0.0f && top_p < 1.0f) {
          auto [sorted_logits, sorted_indices] =
              torch::sort(logits, /*dim=*/-1, /*descending=*/true);
          auto cumulative_probs =
              torch::softmax(sorted_logits, /*dim=*/-1).cumsum(/*dim=*/-1);
          // Remove tokens with cumulative probability above the threshold
          // (shift right so the first token above threshold is kept)
          auto remove_mask = cumulative_probs > top_p;
          remove_mask.narrow(-1, 1, remove_mask.size(-1) - 1)
              .copy_(remove_mask.narrow(-1, 0, remove_mask.size(-1) - 1));
          remove_mask.narrow(-1, 0, 1).fill_(false);
          auto indices_to_remove =
              remove_mask.scatter(-1, sorted_indices, remove_mask);
          logits = logits.masked_fill(indices_to_remove,
                                      -std::numeric_limits<float>::infinity());
        }

        // Softmax over the filtered logit distribution, then sample.
        auto probs = torch::softmax(logits, /*dim=*/-1);
        block_ids = torch::multinomial(probs, /*num_samples=*/1).squeeze(-1);
      }

      // Collect tokens.
      auto ids_cpu = block_ids.cpu().contiguous();
      for (int64_t i = 0; i < ids_cpu.size(0); ++i) {
        all_token_ids.push_back(ids_cpu[i].item<int64_t>());
      }

      // --- Commit denoised block to DiT KV cache at timestep=0 ---
      {
        auto ts_zero = torch::zeros(
            {block_size},
            torch::TensorOptions().dtype(torch::kBFloat16).device(device));
#if defined(USE_CUDA)
        torch::autocast::set_autocast_dtype(torch::kCUDA, torch::kBFloat16);
        torch::autocast::set_autocast_enabled(torch::kCUDA, true);
#endif
        dit_->forward(txt.to(torch::kBFloat16),
                      k_lens_cond,
                      q_lens,
                      ts_zero,
                      /*update_kv=*/true,
                      /*use_kv_cache=*/true);
#if defined(USE_CUDA)
        torch::autocast::set_autocast_enabled(torch::kCUDA, false);
#endif
      }
    }

    // -----------------------------------------------------------------
    // Step 6: Clean up KV caches.
    // -----------------------------------------------------------------
    dit_->set_kv_cache(false);
    vae_->set_kv_cache(false);

    // -----------------------------------------------------------------
    // Step 7: Trim leading prompt tokens from first block, detokenize.
    // -----------------------------------------------------------------
    int64_t trim_count =
        std::max(int64_t{0},
                 std::min(first_block_prompt_tokens,
                          static_cast<int64_t>(all_token_ids.size())));

    std::vector<int32_t> output_tokens(all_token_ids.begin() + trim_count,
                                       all_token_ids.end());

    std::string output_text;
    if (tokenizer_) {
      // Match official inference.py decode_batch(...,
      // skip_special_tokens=False).
      output_text =
          tokenizer_->decode(output_tokens, /*skip_special_tokens=*/false);
    }

    DiTForwardOutput output;
    output.text_output.push_back(output_text);
    return output;
  }

 private:
  torch::TensorOptions options_;
  ColaDiTTransformer dit_{nullptr};
  ColaTextVAEModel vae_{nullptr};
  std::unique_ptr<Tokenizer> tokenizer_;
  int64_t block_size_ = 4;
  // Monotonic per-forward counter used to derive unique block noise seeds
  // when the caller has not specified an explicit seed (seed_is_set == false).
  std::atomic<uint64_t> forward_counter_{0};
};
TORCH_MODULE(ColaDLMPipeline);

// ---------------------------------------------------------------------------
// Register Cola-DLM pipeline (CUDA only).
// ---------------------------------------------------------------------------

#if defined(USE_CUDA)
REGISTER_DIT_MODEL(cola_dlm, ColaDLMPipeline);
#endif

}  // namespace xllm
