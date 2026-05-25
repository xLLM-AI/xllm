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

#include <torch/nn/functional/normalization.h>

#include <filesystem>
#include <optional>
#include <unordered_set>
#include <vector>

#include "core/common/global_flags.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/npu/npu_qwen3_decoder_layer_impl.h"
#include "llm_model_base.h"

namespace xllm::npu::model {

class QWen3DecoderLayerImpl
    : public LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer> {
 public:
  QWen3DecoderLayerImpl(const ModelContext& context, const int32_t layer_id)
      : LlmDecoderLayerImplBase<layer::NpuQwen3DecoderLayer>(context,
                                                             layer_id) {}
};
TORCH_MODULE(QWen3DecoderLayer);

class QWen3ModelImpl : public LlmModelImplBase<QWen3DecoderLayer> {
 public:
  QWen3ModelImpl(const ModelContext& context)
      : LlmModelImplBase<QWen3DecoderLayer>("qwen3", context.get_model_args()) {
    // register submodules
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();
    auto parallel_args = context.get_parallel_args();
    auto dp_local_tp_size =
        parallel_args.world_size() / parallel_args.dp_size();
    dp_rank_ = parallel_args.rank() / dp_local_tp_size;

    blocks_ = register_module("layers", torch::nn::ModuleList());
    layers_.reserve(model_args.n_layers());
    norm_ = register_module("norm", layer::NpuRMSNorm(context));
    npu_embed_tokens_ =
        register_module("npu_embed_tokens", layer::NpuWordEmbedding(context));
    atb_pos_emb_ = layer::NpuPosEmbedding(context);
    cos_sin_ = layer::rotary::get_concat_rotary_embedding(
        128,
        model_args.max_position_embeddings(),
        model_args.rope_theta(),
        options);
    int32_t mask_value =
        ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill() ? -9984
                                                                         : 1;
    // encode_attn_mask_ =
    //   layer::AttentionMask(options.device(),
    //   options.dtype()).get_attn_mask(2048, options.device(),
    //   options.dtype());
    attn_mask_ = layer::AttentionMask(options.device(),
                                      options.dtype().toScalarType(),
                                      /*mask_value=*/mask_value);

    for (int32_t i = 0; i < model_args.n_layers(); i++) {
      auto block = QWen3DecoderLayer(context, i);
      layers_.push_back(block);
      blocks_->push_back(block);
    }

    // Eagle3: layer ids to capture (can be read from layers_to_capture in
    // config.json)
    if (::xllm::SpeculativeConfig::get_instance().speculative_algorithm() ==
        "Eagle3") {
      const auto& layer_ids_from_config = model_args.layers_to_capture();
      if (!layer_ids_from_config.empty()) {
        set_eagle3_layers_to_capture(
            std::make_optional<std::vector<int32_t>>(layer_ids_from_config));
      } else {
        set_eagle3_layers_to_capture();
      }
      // Pre-allocate aux output buffer [max_tokens_per_batch, hidden_size *
      // num_captured]
      const int64_t num_captured = layers_to_capture_set_.size();
      const int64_t aux_dim = model_args.hidden_size() * num_captured;
      aux_output_buffer_ = torch::empty(
          {::xllm::SchedulerConfig::get_instance().max_tokens_per_batch(),
           aux_dim},
          options);
    }
  }

  void set_eagle3_layers_to_capture(
      const std::optional<std::vector<int32_t>>& layer_ids = std::nullopt) {
    capture_aux_hidden_states_ = true;
    layers_to_capture_set_.clear();
    if (!layer_ids.has_value()) {
      int32_t num_layers = layers_.size();
      layers_to_capture_set_.insert(2);
      layers_to_capture_set_.insert(num_layers / 2);
      layers_to_capture_set_.insert(num_layers - 3);
    } else {
      // Config uses 0-based layer indices, same as default {2, n/2, n-3}
      for (int32_t val : layer_ids.value()) {
        layers_to_capture_set_.insert(val);
      }
    }
    LOG(INFO) << "layers_to_capture_set_ size: "
              << layers_to_capture_set_.size();
  }

  void set_quarot_global_rotation(torch::Tensor global_rotation) {
    quarot_global_rotation_t_ = global_rotation.transpose(0, 1).contiguous();
  }

  virtual ModelOutput forward(torch::Tensor tokens,
                              torch::Tensor positions,
                              std::vector<KVCache>& kv_caches,
                              const ModelInputParams& input_params) {
    bool use_deepstack = input_params.multimodal.deep_stacks.size() > 0;
    std::vector<torch::Tensor> deep_stacks;

    if (tokens.numel() == 0) {
      tokens = torch::tensor({1}).to(torch::kInt32).to(tokens.device());
      positions = torch::tensor({0}).to(torch::kInt32).to(tokens.device());
    }
    auto inputs_embeds = input_params.embedding.input_embedding;
    torch::Tensor h;
    if (inputs_embeds.defined()) {
      h = inputs_embeds;
    } else {
      h = npu_embed_tokens_(tokens, 0);
    }

    // This residual tensor would be shared by all the layers, as the
    // current layer would use the output residual from previous layer,
    // the layer could use the residual through local variable
    // without passing the residual tensor to the layer.
    torch::Tensor residual;
    if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm()) {
      residual = torch::zeros_like(h, h.options());
      set_residual(residual);
    }

    if (use_deepstack) {
      deep_stacks =
          input_params.multimodal.deep_stacks;  // [num_deepstack, hidden_size]
    }
    auto target_cos_sin = atb_pos_emb_(cos_sin_, positions, 0);
    auto target_cos_sin_chunks = target_cos_sin.chunk(/*chunks=*/2, /*dim=*/-1);
    auto cos_pos = target_cos_sin_chunks[0].contiguous();
    auto sin_pos = target_cos_sin_chunks[1].contiguous();
    if (positions.dim() == 2) {  // mrope
      auto apply = [this](torch::Tensor x) {
        auto freqs_t = x[0].clone();
        // mrop_length == freqs_length == head_dim / 2
        int64_t mrop_length = freqs_t.size(-1) / 2;

        for (int dim_idx = 1; dim_idx <= 2; ++dim_idx) {
          int64_t offset = dim_idx;
          int64_t section_len = mrope_section_[dim_idx];
          int64_t length = section_len * 3;

          // Since the last dim of freqs is repeated to 2*mrop_length
          // idx_first_half: [offset, offset+3, offset+6, ... < mrop_length]
          // idx_second_half: [mrop_length+offset, mrop_length+offset+3,
          //     mrop_length+offset+6, ... < 2*mrop_length]
          torch::TensorOptions options =
              torch::TensorOptions().dtype(torch::kLong).device(x.device());
          auto idx_first_half = torch::arange(offset, length, 3, options);
          auto idx_second_half = torch::arange(
              offset + mrop_length, length + mrop_length, 3, options);

          auto idx_tensor = torch::cat({idx_first_half, idx_second_half}, 0);
          // freqs_t[..., idx] = freqs[dim_idx][..., idx]
          auto src = x[dim_idx].index_select(-1, idx_tensor);
          freqs_t.index_copy_(-1, idx_tensor, src);
        }
        return freqs_t;
      };
      cos_pos = apply(cos_pos.reshape(
          {positions.sizes().front(), -1, cos_pos.sizes().back()}));
      sin_pos = apply(sin_pos.reshape(
          {positions.sizes().front(), -1, sin_pos.sizes().back()}));
    }

    torch::Tensor attn_mask;
    // for chunked prefill, generate the attn mask.
    if (!input_params.meta.batch_forward_type.is_decode()) {
      if (::xllm::SchedulerConfig::get_instance().enable_chunked_prefill()) {
        int max_kv_seq = input_params.meta.kv_max_seq_len;
        int num_sequences = input_params.meta.num_sequences;
        if (num_sequences > 0) {
          std::vector<torch::Tensor> req_mask_vec;
          req_mask_vec.reserve(num_sequences);

          for (int j = 0; j < num_sequences; j++) {
            auto mask = attn_mask_.gen_append_mask(
                input_params.attention.host.q_seq_lens[j],
                input_params.attention.host.kv_seq_lens[j],
                max_kv_seq,
                cos_pos.dtype().toScalarType(),
                cos_pos.device());
            req_mask_vec.emplace_back(mask);
          }
          attn_mask = torch::cat(req_mask_vec, 0);
        }
      } else {
        attn_mask = attn_mask_.get_attn_mask(
            128, cos_pos.dtype().toScalarType(), cos_pos.device());
      }
    }

    ModelInputParams& input_params_new =
        const_cast<ModelInputParams&>(input_params);
    const int64_t num_tokens = h.size(0);
    const int64_t hidden_size = h.size(-1);
    int64_t capture_idx = 0;
    RollingLayerGuard rolling_guard(rolling_mgr_);
    for (size_t i = 0; i < layers_.size(); i++) {
      aclrtEvent* event{nullptr};
      std::atomic<bool>* event_flag{nullptr};

      if (input_params.parallel.layer_synchronizer != nullptr) {
        event = input_params.parallel.layer_synchronizer->get_event(i);
        event_flag =
            input_params.parallel.layer_synchronizer->get_event_flag(i);
      }
      if (!input_params.synchronize_layer(i)) {
        return ModelOutput();
      }

      auto& layer = layers_[i];
      const int32_t layer_index = i;
      if (capture_aux_hidden_states_ &&
          layers_to_capture_set_.count(layer_index) != 0) {
        torch::Tensor aux_h = h;
        if (::xllm::KernelConfig::get_instance().enable_interlayer_addnorm() &&
            residual.defined()) {
          aux_h = h + residual;
        }
        aux_h = restore_quarot_hidden(
            aux_h.reshape({num_tokens, hidden_size}), hidden_size);
        aux_output_buffer_.slice(0, 0, num_tokens)
            .slice(
                1, capture_idx * hidden_size, (capture_idx + 1) * hidden_size)
            .copy_(aux_h);
        capture_idx++;
      }

      if (layer_forward_interrupted_) {
        LOG(INFO) << "Forward interrupted at layer: " << i;
        return ModelOutput();
      }
      rolling_guard.before_layer(layer_index);

      layer(h,
            cos_pos,
            sin_pos,
            attn_mask,
            kv_caches[i],
            input_params_new,
            event,
            event_flag);

      rolling_guard.after_layer(layer_index);
      if (use_deepstack) {
        if (deep_stacks.size() > 0 && i < deep_stacks.size()) {
          h = h + deep_stacks[i];
        }
      }
    }
    auto hidden_states = norm_(h, 0);
    if (capture_aux_hidden_states_) {
      torch::Tensor aux_hidden_states =
          aux_output_buffer_.slice(0, 0, num_tokens);
      return ModelOutput(hidden_states, torch::Tensor(), aux_hidden_states);
    }
    return ModelOutput(hidden_states);
  }

 private:
  torch::Tensor restore_quarot_hidden(torch::Tensor aux_h,
                                      int64_t hidden_size) {
    if (!quarot_global_rotation_t_.defined() ||
        quarot_global_rotation_t_.numel() == 0) {
      return aux_h;
    }

    return torch::matmul(aux_h, quarot_global_rotation_t_);
  }

  torch::Tensor viusal_pos_mask_;
  std::unordered_set<int32_t> layers_to_capture_set_;
  bool capture_aux_hidden_states_ = false;
  torch::Tensor aux_output_buffer_;
  torch::Tensor quarot_global_rotation_t_;
};
TORCH_MODULE(QWen3Model);

class QWen3ForCausalLMImpl : public LlmForCausalLMImplBase<QWen3Model> {
 public:
  QWen3ForCausalLMImpl(const ModelContext& context)
      : LlmForCausalLMImplBase<QWen3Model>(context),
        device_(context.get_tensor_options().device()),
        dtype_(context.get_tensor_options().dtype().toScalarType()) {}

  torch::Tensor pooler(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    auto h = hidden_states;
    if (seleted_idxes.defined()) {
      h = h.index_select(/*dim=*/0, seleted_idxes);
    }
    return torch::nn::functional::normalize(
        h, torch::nn::functional::NormalizeFuncOptions().p(2).dim(1));
  }

  void load_model(std::unique_ptr<ModelLoader> loader,
                  std::string prefix = "model.") override {
    const std::filesystem::path model_path(loader->model_weights_path());
    LlmForCausalLMImplBase<QWen3Model>::load_model(std::move(loader), prefix);
    load_optional_quarot_rotation(model_path);
  }

  void lazy_load_model(std::unique_ptr<ModelLoader> loader,
                       std::string prefix = "model.") override {
    const std::filesystem::path model_path(loader->model_weights_path());
    LlmForCausalLMImplBase<QWen3Model>::lazy_load_model(std::move(loader),
                                                        prefix);
    load_optional_quarot_rotation(model_path);
  }

 private:
  void load_optional_quarot_rotation(const std::filesystem::path& model_path) {
    const std::filesystem::path quarot_path =
        model_path / "optional" / "quarot.safetensors";
    if (!std::filesystem::exists(quarot_path)) {
      return;
    }

    auto state_dict = StateDictFromSafeTensor::load(quarot_path.string());
    torch::Tensor global_rotation = state_dict->get_tensor("global_rotation");
    if (!global_rotation.defined()) {
      LOG(WARNING) << "Optional QuaRot file exists but global_rotation is "
                      "missing: "
                   << quarot_path.string();
      return;
    }
    CHECK_EQ(global_rotation.dim(), 2)
        << "QuaRot global_rotation must be a 2D tensor";
    CHECK_EQ(global_rotation.size(0), global_rotation.size(1))
        << "QuaRot global_rotation must be square";

    global_rotation =
        global_rotation
            .to(torch::TensorOptions()
                    .dtype(dtype_)
                    .device(device_),
                /*non_blocking=*/false,
                /*copy=*/true)
            .contiguous();
    model_->set_quarot_global_rotation(global_rotation);
    LOG(INFO) << "Loaded optional QuaRot global_rotation from "
              << quarot_path.string() << ", shape=" << global_rotation.sizes();
  }

  torch::Device device_;
  torch::Dtype dtype_;
};
TORCH_MODULE(QWen3ForCausalLM);

// register the causal model
REGISTER_CAUSAL_MODEL_WITH_VARNAME(qwen3_atb, qwen3_atb, QWen3ForCausalLM);

// register the model args
REGISTER_MODEL_ARGS_WITH_VARNAME(qwen3_atb, qwen3_atb, [&] {
  LOAD_ARG_OR(model_type, "model_type", "qwen3");
  LOAD_ARG_OR(dtype, "torch_dtype", "");
  LOAD_ARG_OR(vocab_size, "vocab_size", 152064);
  LOAD_ARG_OR(hidden_size, "hidden_size", 3584);
  LOAD_ARG_OR(hidden_act, "hidden_act", "silu");
  LOAD_ARG_OR(n_layers, "num_hidden_layers", 28);
  LOAD_ARG_OR(n_heads, "num_attention_heads", 28);
  LOAD_ARG(n_kv_heads, "num_key_value_heads");
  // LOAD_ARG_OR(no_bias, "no_bias", true);
  LOAD_ARG_OR(intermediate_size, "intermediate_size", 18944);
  LOAD_ARG_OR(max_position_embeddings, "max_position_embeddings", 32768);
  LOAD_ARG_OR(rms_norm_eps, "rms_norm_eps", 1e-6);
  LOAD_ARG_OR(eos_token_id, "eos_token_id", 151643);
  LOAD_ARG_OR(rope_theta, "rope_theta", 1000000.0f);

  // For qwen3/2.5 model < 7B,  tie_word_embeddings = true
  LOAD_ARG_OR(tie_word_embeddings, "tie_word_embeddings", false);

  LOAD_ARG_OR(use_sliding_window, "use_sliding_window", false);
  LOAD_ARG_OR(max_window_layers, "max_window_layers", 28);

  // Eagle3: layer ids (0-based) to capture from config, e.g.
  // "layers_to_capture": [2, 14, 25]; defaults to empty if missing
  LOAD_ARG_OR(layers_to_capture, "layers_to_capture", std::vector<int32_t>{});

  LOAD_ARG_OR_FUNC(head_dim, "head_dim", [&] {
    return args->hidden_size() / args->n_heads();
  });

  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({args->eos_token_id()}));
});

}  // namespace xllm::npu::model
