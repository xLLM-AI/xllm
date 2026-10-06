/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <memory>
#include <optional>
#include <unordered_set>
#include <utility>
#include <vector>

#include "models/llm/npu/qwen3.h"
#include "models/model_registry.h"
#include "models/vlm/mposition/mposition.h"
#include "models/vlm/npu/qwen3_audio_encoder.h"
#include "processors/qwen3_asr_processor.h"
#include "processors/qwen3_audio_common.h"

namespace xllm::npu::model {

inline void load_qwen3_asr_model_args(const JsonReader& json, ModelArgs* args) {
  LOAD_ARG_OR(model_type, "model_type", "qwen3_asr");
  LOAD_ARG_OR(has_feature_extractor, "has_feature_extractor", true);
  LOAD_ARG_OR(mm_audio_feature_size, "feature_size", 128);
  LOAD_ARG_OR(mm_audio_sampling_rate, "sampling_rate", 16000);
  LOAD_ARG_OR(mm_audio_hop_length, "hop_length", 160);
  LOAD_ARG_OR(mm_audio_chunk_length, "chunk_length", 30);
  LOAD_ARG_OR(mm_audio_n_fft, "n_fft", 400);
  LOAD_ARG_OR(mm_audio_dither, "dither", 0.0);
  LOAD_ARG_OR(mm_audio_truncation, "truncation", false);
  LOAD_ARG_OR(mm_audio_do_normalize, "do_normalize", false);

  LOAD_ARG_OR(audio_token_id, "thinker_config.audio_token_id", 151676);
  LOAD_ARG_OR(
      audio_start_token_id, "thinker_config.audio_start_token_id", 151669);
  LOAD_ARG_OR(audio_end_token_id, "thinker_config.audio_end_token_id", 151670);
  LOAD_ARG_OR(dtype, "thinker_config.dtype", "bfloat16");

  LOAD_ARG_OR(
      attention_bias, "thinker_config.text_config.attention_bias", false);
  LOAD_ARG_OR(
      attention_dropout, "thinker_config.text_config.attention_dropout", 0.0);
  LOAD_ARG_OR(bos_token_id, "thinker_config.text_config.bos_token_id", 151643);
  LOAD_ARG_OR(eos_token_id, "thinker_config.text_config.eos_token_id", 151645);
  LOAD_ARG_OR(hidden_act, "thinker_config.text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "thinker_config.text_config.hidden_size", 2048);
  LOAD_ARG_OR(
      intermediate_size, "thinker_config.text_config.intermediate_size", 6144);
  LOAD_ARG_OR(max_position_embeddings,
              "thinker_config.text_config.max_position_embeddings",
              65536);
  LOAD_ARG_OR(
      max_window_layers, "thinker_config.text_config.max_window_layers", 28);
  LOAD_ARG_OR(n_heads, "thinker_config.text_config.num_attention_heads", 16);
  LOAD_ARG_OR(n_layers, "thinker_config.text_config.num_hidden_layers", 28);
  LOAD_ARG_OR(n_kv_heads, "thinker_config.text_config.num_key_value_heads", 8);
  LOAD_ARG_OR(rms_norm_eps, "thinker_config.text_config.rms_norm_eps", 1e-06);
  LOAD_ARG_OR(
      sliding_window, "thinker_config.text_config.sliding_window", 32768);
  LOAD_ARG_OR(tie_word_embeddings,
              "thinker_config.text_config.tie_word_embeddings",
              true);
  LOAD_ARG_OR(
      initializer_range, "thinker_config.text_config.initializer_range", 0.02);
  LOAD_ARG_OR(use_sliding_window,
              "thinker_config.text_config.use_sliding_window",
              false);
  LOAD_ARG_OR_FUNC(head_dim, "thinker_config.text_config.head_dim", [args] {
    return args->hidden_size() / args->n_heads();
  });
  LOAD_ARG_OR(rope_scaling_rope_type,
              "thinker_config.text_config.rope_scaling.type",
              "mrope");
  LOAD_ARG(rope_scaling_mrope_section,
           "thinker_config.text_config.rope_scaling.mrope_section");
  LOAD_ARG_OR(rope_theta, "thinker_config.text_config.rope_theta", 1000000.0f);
  LOAD_ARG_OR(vocab_size, "thinker_config.text_config.vocab_size", 151936);

  if (args->rope_scaling_rope_type() == "default") {
    args->rope_scaling_rope_type() = "mrope";
  }
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({151643, 151645}));

  LOAD_ARG_OR(mm_audio_num_attention_heads,
              "thinker_config.audio_config.encoder_attention_heads",
              16);
  LOAD_ARG_OR(
      mm_audio_hidden_size, "thinker_config.audio_config.d_model", 1024);
  LOAD_ARG_OR(mm_audio_layer_norm_eps,
              "thinker_config.audio_config.layer_norm_eps",
              1e-5);
  LOAD_ARG_OR(mm_audio_downsample_hidden_size,
              "thinker_config.audio_config.downsample_hidden_size",
              480);
  LOAD_ARG_OR(
      mm_audio_num_mel_bins, "thinker_config.audio_config.num_mel_bins", 128);
  LOAD_ARG_OR(mm_audio_max_source_positions,
              "thinker_config.audio_config.max_source_positions",
              1500);
  LOAD_ARG_OR(mm_audio_n_window, "thinker_config.audio_config.n_window", 50);
  LOAD_ARG_OR(mm_audio_n_window_infer,
              "thinker_config.audio_config.n_window_infer",
              800);
  LOAD_ARG_OR(mm_audio_conv_chunksize,
              "thinker_config.audio_config.conv_chunksize",
              500);
  LOAD_ARG_OR(mm_audio_encoder_layers,
              "thinker_config.audio_config.encoder_layers",
              24);
  LOAD_ARG_OR(
      mm_audio_output_dim, "thinker_config.audio_config.output_dim", 2048);
}

class Qwen3ASRForConditionalGenerationImpl final : public torch::nn::Module {
 public:
  explicit Qwen3ASRForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    audio_tower_ = register_module("audio_tower", Qwen3AudioEncoder(context));
    language_model_ =
        register_module("language_model", QWen3ForCausalLM(context));
  }

  void prepare_encoder_input(const ModelInputParams& input_params,
                             std::optional<Qwen3AudioInputs>& audio_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor input_features;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kInputFeaturesKey)) {
      input_features = res.value();
    }

    torch::Tensor feature_lengths;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kFeatureLengthKey)) {
      feature_lengths = res.value();
    }

    torch::Tensor feature_origin_lengths;
    if (std::optional<torch::Tensor> res =
            mm_data.get<torch::Tensor>(qwen3_audio::kFeatureOriginLengthsKey)) {
      feature_origin_lengths = res.value();
    }

    if (input_features.defined() && feature_lengths.defined() &&
        feature_origin_lengths.defined()) {
      audio_inputs = Qwen3AudioInputs{
          input_features, feature_lengths, feature_origin_lengths};
    }
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<Qwen3AudioInputs> audio_input;
    prepare_encoder_input(input_params, audio_input);
    MMDict multimodal_embeds;
    if (audio_input) {
      const torch::Tensor feature_origin_lengths =
          audio_input->feature_origin_lengths.to(options_.device(),
                                                 torch::kLong);

      const torch::Tensor input_features =
          audio_input->input_features.permute({1, 0}).to(options_);

      const torch::Tensor audio_embeds =
          audio_tower_->forward(input_features, feature_origin_lengths);

      const torch::Tensor audio_tokens =
          audio_input->feature_lengths.cpu().contiguous().to(torch::kLong);

      std::vector<int64_t> feature_lens_vec(
          audio_tokens.data_ptr<int64_t>(),
          audio_tokens.data_ptr<int64_t>() + audio_tokens.numel());

      multimodal_embeds[get_embedding_key(MMType::AUDIO)] =
          audio_embeds.split(feature_lens_vec, /*dim=*/0);
    }
    return multimodal_embeds;
  }

  torch::Tensor merge_multimodal_embeddings(
      torch::Tensor inputs_embeds,
      const torch::Tensor& multimodal_embeds,
      const torch::Tensor& is_multimodal) {
    inputs_embeds.index_put_({is_multimodal}, multimodal_embeds);
    return inputs_embeds;
  }

  torch::Tensor get_input_embeddings(const torch::Tensor input_ids,
                                     const ModelInputParams& input_params) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor inputs_embeds =
        language_model_->get_input_embeddings(input_ids);
    std::optional<torch::Tensor> audio_embeds =
        mm_data.get<torch::Tensor>(get_embedding_key(MMType::AUDIO));
    std::optional<torch::Tensor> audio_mask =
        mm_data.get<torch::Tensor>(qwen3_audio::kMaskKey);
    if (audio_embeds.has_value() && audio_mask.has_value()) {
      inputs_embeds = merge_multimodal_embeddings(
          inputs_embeds, audio_embeds.value(), audio_mask.value());
    }
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& selected_indices) {
    return language_model_->logits(hidden_states, selected_indices);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    for (const auto& state_dict : loader->get_state_dicts()) {
      audio_tower_->load_state_dict(
          state_dict->get_dict_with_prefix("thinker.audio_tower."));
    }
    audio_tower_->verify_loaded_weights("thinker.audio_tower.");
    audio_tower_->merge_loaded_weights();
    audio_tower_->to(options_.device(),
                     torch::typeMetaToScalarType(options_.dtype()));

    if (!model_args_.encoder_embedding_mode()) {
      language_model_->load_model(
          std::move(loader), "thinker.model.", "thinker.lm_head.");
    }
  }

  layer::NpuLmHead get_npu_lm_head() {
    return language_model_->get_npu_lm_head();
  }

  void set_npu_lm_head(layer::NpuLmHead& head) {
    language_model_->set_npu_lm_head(head);
  }

  layer::NpuWordEmbedding get_npu_word_embedding() {
    return language_model_->get_npu_word_embedding();
  }

  void set_npu_word_embedding(layer::NpuWordEmbedding& npu_word_embedding) {
    language_model_->set_npu_word_embedding(npu_word_embedding);
  }

 private:
  ModelArgs model_args_;
  torch::TensorOptions options_;
  Qwen3AudioEncoder audio_tower_{nullptr};
  QWen3ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(Qwen3ASRForConditionalGeneration);

REGISTER_MULTIMODAL_PROCESSOR(qwen3_asr, Qwen3ASRMultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(qwen3_asr, Qwen3ASRForConditionalGeneration);
REGISTER_MPOSITION_GENERATOR(qwen3_asr, xllm::Qwen3VLMPositionGenerator);

REGISTER_MODEL_ARGS(qwen3_asr, [&] { load_qwen3_asr_model_args(json, args); });

}  // namespace xllm::npu::model
