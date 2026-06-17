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

#include <atb/atb_infer.h>
#include <c10/core/ScalarType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstdint>
#include <unordered_map>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/layers/npu/npu_kimik25_vision_encoder_layer_impl.h"
#include "core/layers/npu/npu_lm_head_impl.h"
#include "core/layers/npu/npu_qwen2_decoder_layer_impl.h"
#include "core/layers/npu/npu_rms_norm_impl.h"
#include "models/llm/npu/deepseek_v3.h"
#include "models/model_registry.h"
#include "processors/kimi25_image_processor.h"
#include "processors/multimodal_processor.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"

namespace xllm {
const int32_t KIMIV_VT_INFER_MAX_PATCH_NUM = 16328;
#define PrintTensor(tensor) print_tensor(tensor, #tensor, 10, true, false);

namespace {
StateDict get_dict_with_prefix_fallback(
    const StateDict& state_dict,
    const std::vector<std::string>& prefixes) {
  CHECK(!prefixes.empty()) << "prefixes should not be empty";
  auto dict = state_dict.get_dict_with_prefix(prefixes[0]);
  for (size_t idx = 1; idx < prefixes.size() && dict.size() == 0; ++idx) {
    dict = state_dict.get_dict_with_prefix(prefixes[idx]);
  }
  return dict;
}

void load_tensor_if_defined(const StateDict& state_dict,
                            const char* key,
                            torch::Tensor& dst,
                            bool& loaded,
                            const std::string& module_name,
                            const std::string& tensor_name) {
  auto src = state_dict.get_tensor(key);
  if (!src.defined()) {
    return;
  }
  CHECK_EQ(dst.sizes(), src.sizes())
      << tensor_name << " size mismatch for " << module_name;
  dst.data().copy_(src);
  loaded = true;
}

void load_linear_if_defined(const StateDict& state_dict,
                            torch::nn::Linear& linear,
                            bool& weight_loaded,
                            bool& bias_loaded,
                            const std::string& module_name,
                            const std::string& linear_name) {
  load_tensor_if_defined(state_dict,
                         "weight",
                         linear->weight,
                         weight_loaded,
                         module_name,
                         linear_name + ".weight");
  load_tensor_if_defined(state_dict,
                         "bias",
                         linear->bias,
                         bias_loaded,
                         module_name,
                         linear_name + ".bias");
}

void load_layernorm_if_defined(const StateDict& state_dict,
                               torch::nn::LayerNorm& layer_norm,
                               bool& weight_loaded,
                               bool& bias_loaded,
                               const std::string& module_name,
                               const std::string& layernorm_name) {
  load_tensor_if_defined(state_dict,
                         "weight",
                         layer_norm->weight,
                         weight_loaded,
                         module_name,
                         layernorm_name + ".weight");
  load_tensor_if_defined(state_dict,
                         "bias",
                         layer_norm->bias,
                         bias_loaded,
                         module_name,
                         layernorm_name + ".bias");
}
}  // namespace

class KimiK2_5_VisionBlockImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionBlockImpl(const ModelContext& context) {
    // register submodules
    encoder_layer_ = register_module(
        "encoder_layer", layer::NpuKimik25VisionEncoderLayer(context));
  }

  struct BlockInput {
    torch::Tensor hidden_states;
    torch::Tensor cu_seqlens;  // cumulative seqlens, shape [batch + 1]
    int64_t max_seqlen = 0;
    torch::Tensor cos_pos;
    torch::Tensor sin_pos;
  };

  torch::Tensor forward(BlockInput& block_input, int32_t node_id) {
    auto seqlens = torch::diff(block_input.cu_seqlens);
    auto seqlens_cpu = seqlens.cpu().to(torch::kInt32).contiguous();
    std::vector<int> seqlens_vec(
        seqlens_cpu.data_ptr<int>(),
        seqlens_cpu.data_ptr<int>() + seqlens_cpu.numel());

    auto token_num = block_input.hidden_states.size(0);
    CHECK(block_input.cos_pos.defined())
        << "cos_pos is undefined for " << name();
    CHECK(block_input.sin_pos.defined())
        << "sin_pos is undefined for " << name();
    CHECK_EQ(block_input.cos_pos.size(0), token_num)
        << "cos_pos token count mismatch for " << name();
    CHECK_EQ(block_input.sin_pos.size(0), token_num)
        << "sin_pos token count mismatch for " << name();

    return encoder_layer_(block_input.hidden_states,
                          block_input.cos_pos,
                          block_input.sin_pos,
                          seqlens,
                          seqlens_vec,
                          node_id);
  }

  // load the weight from the checkpoint
  void load_state_dict(const StateDict& state_dict) {
    // call each submodule's load_state_dict function
    encoder_layer_->load_state_dict(state_dict);
  }

  void verify_loaded_weights(const std::string& prefix) const {
    encoder_layer_->verify_loaded_weights();
  }
  void merge_loaded_weights() { encoder_layer_->merge_loaded_weights(); }

 private:
  layer::NpuKimik25VisionEncoderLayer encoder_layer_{nullptr};
};
TORCH_MODULE(KimiK2_5_VisionBlock);

class KimiK2_5_VisionPosEmbDividedImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionPosEmbDividedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    pos_emb_height_ = model_args.mm_init_pos_emb_height();
    pos_emb_width_ = model_args.mm_init_pos_emb_width();
    pos_emb_time_ = model_args.mm_init_pos_emb_time();
    dim_ = model_args.mm_hidden_size();

    weight_ = register_parameter(
        "weight",
        torch::empty({pos_emb_height_, pos_emb_width_, dim_}, options));
    torch::nn::init::normal_(weight_);

    time_weight_ = register_buffer(
        "time_weight", build_time_weight(pos_emb_time_, dim_, options));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor grid_thws) {
    std::vector<torch::Tensor> pos_embs;
    auto count = grid_thws.size(0);
    pos_embs.reserve(count);
    auto grid_thws_cpu = grid_thws.cpu().to(torch::kLong).contiguous();

    for (int64_t i = 0; i < count; ++i) {
      auto t = grid_thws_cpu[i][0].item<int64_t>();
      auto h = grid_thws_cpu[i][1].item<int64_t>();
      auto w = grid_thws_cpu[i][2].item<int64_t>();

      CHECK_LE(t, pos_emb_time_) << "grid_thws t larger than init_pos_emb_time";

      torch::Tensor pos_emb_2d;
      if (h == pos_emb_height_ && w == pos_emb_width_) {
        pos_emb_2d = weight_.flatten(0, 1);
      } else {
        namespace F = torch::nn::functional;
        pos_emb_2d = F::interpolate(weight_.permute({2, 0, 1}).unsqueeze(0),
                                    F::InterpolateFuncOptions()
                                        .size(std::vector<int64_t>({h, w}))
                                        .mode(torch::kBicubic)
                                        .align_corners(false))
                         .squeeze(0)
                         .permute({1, 2, 0})
                         .flatten(0, 1);
      }

      torch::Tensor pos_emb_3d;
      if (t == 1) {
        pos_emb_3d = pos_emb_2d;
      } else {
        pos_emb_3d = pos_emb_2d.unsqueeze(0).repeat({t, 1, 1}) +
                     time_weight_.index({torch::indexing::Slice(0, t)});
      }
      pos_embs.emplace_back(pos_emb_3d.reshape({-1, pos_emb_3d.size(-1)}));
    }

    auto pos_emb = torch::cat(pos_embs, 0).to(x.options());
    return x + pos_emb;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("weight");
    if (weight.defined()) {
      if (weight.sizes() != weight_.sizes()) {
        CHECK_EQ(weight.numel(), weight_.numel())
            << "pos_emb weight numel mismatch for " << name();
        weight = weight.view(weight_.sizes());
      }
      DCHECK_EQ(weight_.sizes(), weight.sizes())
          << "pos_emb weight size mismatch for " << name();
      weight_.data().copy_(weight.to(weight_.device()).to(weight_.dtype()));
      weight_loaded_ = true;
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(weight_loaded_) << "weight is not loaded for " << prefix + "weight";
  }

 private:
  torch::Tensor build_time_weight(int64_t t_size,
                                  int64_t embed_dim,
                                  const torch::TensorOptions& options) {
    CHECK_EQ(embed_dim % 2, 0) << "embed_dim must be even";
    auto float_opts =
        torch::TensorOptions().dtype(torch::kFloat32).device(options.device());
    auto omega = torch::arange(embed_dim / 2, float_opts);
    omega = 1.0 / torch::pow(10000.0, omega / (embed_dim / 2.0));
    auto pos = torch::arange(t_size, float_opts).reshape({-1});
    auto out = torch::einsum("m,d->md", {pos, omega});
    auto emb = torch::cat({torch::sin(out), torch::cos(out)}, 1);
    return emb.unsqueeze(1).to(options.dtype());
  }

 private:
  int64_t pos_emb_height_ = 0;
  int64_t pos_emb_width_ = 0;
  int64_t pos_emb_time_ = 0;
  int64_t dim_ = 0;
  torch::Tensor weight_;
  torch::Tensor time_weight_;
  bool weight_loaded_ = false;
};
TORCH_MODULE(KimiK2_5_VisionPosEmbDivided);

class KimiK2_5_VisionPatchEmbedImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionPatchEmbedImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    auto in_features = model_args.mm_num_channels() *
                       model_args.mm_patch_size() * model_args.mm_patch_size();
    auto out_features = model_args.mm_hidden_size();

    proj_ = register_module(
        "proj",
        torch::nn::Linear(
            torch::nn::LinearOptions(in_features, out_features).bias(true)));
    proj_->weight.set_data(proj_->weight.to(options));
    proj_->bias.set_data(proj_->bias.to(options));

    pos_emb_ =
        register_module("pos_emb", KimiK2_5_VisionPosEmbDivided(context));
  }

  torch::Tensor forward(torch::Tensor x, torch::Tensor grid_thws) {
    x = x.view({x.size(0), -1});
    x = proj_(x);
    x = pos_emb_(x, grid_thws);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    auto weight = state_dict.get_tensor("proj.weight");
    if (weight.defined()) {
      weight = weight.reshape({weight.size(0), -1});
      DCHECK_EQ(proj_->weight.sizes(), weight.sizes())
          << "proj weight size mismatch for " << name();
      proj_->weight.data().copy_(weight);
      proj_weight_loaded_ = true;
    }

    auto bias = state_dict.get_tensor("proj.bias");
    if (bias.defined()) {
      DCHECK_EQ(proj_->bias.sizes(), bias.sizes())
          << "proj bias size mismatch for " << name();
      proj_->bias.data().copy_(bias);
      proj_bias_loaded_ = true;
    }

    pos_emb_->load_state_dict(state_dict.get_dict_with_prefix("pos_emb."));
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(proj_weight_loaded_)
        << "weight is not loaded for " << prefix + "proj.weight";
    CHECK(proj_bias_loaded_)
        << "bias is not loaded for " << prefix + "proj.bias";
    pos_emb_->verify_loaded_weights(prefix + "pos_emb.");
  }

 private:
  bool proj_weight_loaded_ = false;
  bool proj_bias_loaded_ = false;
  torch::nn::Linear proj_{nullptr};
  KimiK2_5_VisionPosEmbDivided pos_emb_{nullptr};
};
TORCH_MODULE(KimiK2_5_VisionPatchEmbed);

class KimiK2_5_VisionRotaryEmbeddingImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionRotaryEmbeddingImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    dim_ = model_args.mm_head_dim();
    if (dim_ <= 0 && model_args.mm_num_attention_heads() > 0) {
      dim_ = model_args.mm_hidden_size() / model_args.mm_num_attention_heads();
    }
    CHECK_GT(dim_, 0) << "invalid vision head dim";
    CHECK_EQ(dim_ % 4, 0) << "rope_2d head dim must be divisible by 4";
  }

  torch::Tensor get_freqs_cis(torch::Tensor grid_thws,
                              const torch::Device& device) {
    if (!freqs_cis_cache_.defined() || freqs_cis_cache_.device() != device) {
      freqs_cis_cache_ = precompute_freqs_cis(device);
    }

    std::vector<torch::Tensor> freqs_cis;
    auto count = grid_thws.size(0);
    freqs_cis.reserve(count);

    auto grid_thws_cpu = grid_thws.cpu().to(torch::kLong).contiguous();
    for (int64_t idx = 0; idx < count; ++idx) {
      auto t = grid_thws_cpu[idx][0].item<int64_t>();
      auto h = grid_thws_cpu[idx][1].item<int64_t>();
      auto w = grid_thws_cpu[idx][2].item<int64_t>();

      CHECK_GE(h, 1) << "grid_thws h must be >= 1";
      CHECK_GE(w, 1) << "grid_thws w must be >= 1";
      CHECK_LE(h, max_height_) << "grid_thws h exceeds rope_2d max_height";
      CHECK_LE(w, max_width_) << "grid_thws w exceeds rope_2d max_width";

      auto freq = freqs_cis_cache_
                      .index({torch::indexing::Slice(0, h),
                              torch::indexing::Slice(0, w)})
                      .reshape({-1, dim_ / 2})
                      .repeat({t, 1});
      freqs_cis.emplace_back(freq);
    }

    if (freqs_cis.empty()) {
      return torch::empty({0, dim_ / 2}, freqs_cis_cache_.options());
    }
    return torch::cat(freqs_cis, 0);
  }

 private:
  torch::Tensor precompute_freqs_cis(const torch::Device& device) const {
    auto options = torch::TensorOptions().dtype(torch::kFloat32).device(device);

    auto n = max_height_ * max_width_;
    auto flat_pos = torch::arange(0, n, options);
    auto x_pos = torch::remainder(flat_pos, max_width_);
    auto y_pos = torch::floor_divide(flat_pos, max_width_);
    auto dim_range = torch::arange(0, dim_, 4, options);

    auto freqs = 1.0 / torch::pow(theta_base_, dim_range / dim_);
    auto x_freqs = torch::outer(x_pos, freqs).to(torch::kFloat32);
    auto y_freqs = torch::outer(y_pos, freqs).to(torch::kFloat32);
    auto x_cis = torch::polar(torch::ones_like(x_freqs), x_freqs);
    auto y_cis = torch::polar(torch::ones_like(y_freqs), y_freqs);

    auto freqs_cis = torch::cat({x_cis.unsqueeze(-1), y_cis.unsqueeze(-1)}, -1);
    return freqs_cis.reshape({max_height_, max_width_, dim_ / 2});
  }

 private:
  int64_t dim_ = 0;
  int64_t max_height_ = 512;
  int64_t max_width_ = 512;
  float theta_base_ = 10000.0f;
  torch::Tensor freqs_cis_cache_;
};
TORCH_MODULE(KimiK2_5_VisionRotaryEmbedding);

class KimiK2_5_VisionPatchMergerImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionPatchMergerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    int64_t d_model = model_args.mm_projection_dim();  // out_hidden_size
    context_dim_ = model_args.mm_hidden_size();
    int spatial_merge_size = model_args.mm_spatial_merge_size();
    auto ln_eps = model_args.mm_layer_norm_eps() > 0
                      ? model_args.mm_layer_norm_eps()
                      : 1e-5f;

    hidden_size_ =
        context_dim_ * static_cast<int>(std::pow(spatial_merge_size, 2));

    pre_norm_ = register_module(
        "pre_norm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({context_dim_})
                                 .eps(ln_eps)
                                 .elementwise_affine(true)));
    pre_norm_->weight.set_data(pre_norm_->weight.to(options));
    pre_norm_->bias.set_data(pre_norm_->bias.to(options));

    linear_1_ = register_module(
        "linear_1",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, hidden_size_).bias(true)));
    linear_1_->weight.set_data(linear_1_->weight.to(options));
    linear_1_->bias.set_data(linear_1_->bias.to(options));

    linear_2_ = register_module(
        "linear_2",
        torch::nn::Linear(
            torch::nn::LinearOptions(hidden_size_, d_model).bias(true)));
    linear_2_->weight.set_data(linear_2_->weight.to(options));
    linear_2_->bias.set_data(linear_2_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor x) {
    x = pre_norm_(x);
    x = x.view({-1, hidden_size_});
    x = linear_1_(x);
    x = torch::gelu(x);
    x = linear_2_(x);
    return x;
  }

  void load_state_dict(const StateDict& state_dict) {
    // prefer sglang/HF keys while keeping legacy fallback.
    auto ln_dict = state_dict.get_dict_with_prefix("pre_norm.");

    load_layernorm_if_defined(ln_dict,
                              pre_norm_,
                              is_pre_norm_weight_loaded,
                              is_pre_norm_bias_loaded,
                              name(),
                              "pre_norm");

    auto linear_1_dict = get_dict_with_prefix_fallback(
        state_dict, {"proj.0.", "linear_1.", "mlp.0."});
    load_linear_if_defined(linear_1_dict,
                           linear_1_,
                           is_linear_1_weight_loaded_,
                           is_linear_1_bias_loaded_,
                           name(),
                           "linear_1");

    auto linear_2_dict = get_dict_with_prefix_fallback(
        state_dict, {"proj.2.", "linear_2.", "mlp.2."});
    load_linear_if_defined(linear_2_dict,
                           linear_2_,
                           is_linear_2_weight_loaded_,
                           is_linear_2_bias_loaded_,
                           name(),
                           "linear_2");
  }

  void verify_loaded_weights(const std::string& prefix) const {
    CHECK(is_pre_norm_weight_loaded)
        << "weight is not loaded for " << prefix + "pre_norm.weight";
    CHECK(is_pre_norm_bias_loaded)
        << "bias is not loaded for " << prefix + "pre_norm.bias";
    CHECK(is_linear_1_weight_loaded_)
        << "weight is not loaded for " << prefix + "linear_1.weight";
    CHECK(is_linear_1_bias_loaded_)
        << "bias is not loaded for " << prefix + "linear_1.bias";
    CHECK(is_linear_2_weight_loaded_)
        << "weight is not loaded for " << prefix + "linear_2.weight";
    CHECK(is_linear_2_bias_loaded_)
        << "bias is not loaded for " << prefix + "linear_2.bias";
  }

  void merge_loaded_weights() {}

 private:
  int64_t hidden_size_;
  int64_t context_dim_;

  torch::nn::LayerNorm pre_norm_{nullptr};
  torch::nn::Linear linear_1_{nullptr};
  torch::nn::Linear linear_2_{nullptr};
  bool is_pre_norm_weight_loaded = false;
  bool is_pre_norm_bias_loaded = false;
  bool is_linear_1_weight_loaded_ = false;
  bool is_linear_1_bias_loaded_ = false;
  bool is_linear_2_weight_loaded_ = false;
  bool is_linear_2_bias_loaded_ = false;
};
TORCH_MODULE(KimiK2_5_VisionPatchMerger);

class KimiK2_5_VisionEncoderImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionEncoderImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();
    auto options = context.get_tensor_options();

    hidden_size_ = model_args.mm_hidden_size();
    rope_2d_ =
        register_module("rope_2d", KimiK2_5_VisionRotaryEmbedding(context));
    blocks_ = register_module("blocks", torch::nn::ModuleList());

    for (int32_t idx = 0; idx < model_args.mm_num_hidden_layers(); idx++) {
      auto block = KimiK2_5_VisionBlock(context);
      blocks_->push_back(block);
      layers_.push_back(block);
    }
    auto ln_eps = model_args.mm_layer_norm_eps() > 0
                      ? model_args.mm_layer_norm_eps()
                      : 1e-5f;
    final_layernorm_ = register_module(
        "final_layernorm",
        torch::nn::LayerNorm(torch::nn::LayerNormOptions({hidden_size_})
                                 .eps(ln_eps)
                                 .elementwise_affine(true)));
    final_layernorm_->weight.set_data(final_layernorm_->weight.to(options));
    final_layernorm_->bias.set_data(final_layernorm_->bias.to(options));
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    // Align with MoonViT3dEncoder:
    // rope_freqs_cis + cu_seqlens + max_seqlen are prepared once and reused
    // across all encoder blocks.
    auto lengths = (grid_thw.index({torch::indexing::Slice(), 0}) *
                    grid_thw.index({torch::indexing::Slice(), 1}) *
                    grid_thw.index({torch::indexing::Slice(), 2}))
                       .to(torch::kInt32)
                       .to(hidden_states.device());
    auto rope_freqs_cis =
        rope_2d_->get_freqs_cis(grid_thw, hidden_states.device());
    auto max_seqlen = lengths.max().item<int64_t>();
    auto zero = torch::zeros({1}, lengths.options());
    auto cu_seqlens = torch::cat({zero, lengths.cumsum(0, torch::kInt32)}, 0);

    CHECK_EQ(rope_freqs_cis.size(0), hidden_states.size(0))
        << "rope_freqs_cis and hidden_states token count mismatch";
    CHECK_EQ(cu_seqlens.size(0), grid_thw.size(0) + 1)
        << "cu_seqlens length mismatch";
    CHECK_EQ(cu_seqlens.index({-1}).item<int32_t>(), hidden_states.size(0))
        << "cu_seqlens last value mismatch with token count";

    // Convert complex cis(freqs) to real cos/sin tensors for NPU vision block.
    auto rope_freqs_cis_ri = torch::view_as_real(rope_freqs_cis);
    auto cos_pos =
        rope_freqs_cis_ri
            .index({torch::indexing::Slice(), torch::indexing::Slice(), 0})
            .unsqueeze(-1)
            .repeat({1, 1, 2})
            .view({hidden_states.size(0), -1})
            .to(hidden_states.options());
    auto sin_pos =
        rope_freqs_cis_ri
            .index({torch::indexing::Slice(), torch::indexing::Slice(), 1})
            .unsqueeze(-1)
            .repeat({1, 1, 2})
            .view({hidden_states.size(0), -1})
            .to(hidden_states.options());

    CHECK_EQ(cos_pos.size(0), hidden_states.size(0))
        << "cos_pos and hidden_states token count mismatch";
    CHECK_EQ(sin_pos.size(0), hidden_states.size(0))
        << "sin_pos and hidden_states token count mismatch";

    KimiK2_5_VisionBlockImpl::BlockInput block_input{
        hidden_states, cu_seqlens, max_seqlen, cos_pos, sin_pos};

    for (int idx = 0; idx < blocks_->size(); ++idx) {
      block_input.hidden_states = layers_[idx](block_input, idx);
    }
    return final_layernorm_(block_input.hidden_states);
  }

  void load_state_dict(const StateDict& state_dict) {
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->load_state_dict(state_dict.get_dict_with_prefix(
          "blocks." + std::to_string(idx) + "."));
    }

    auto ln_dict = state_dict.get_dict_with_prefix("final_layernorm.");
    load_layernorm_if_defined(ln_dict,
                              final_layernorm_,
                              final_ln_weight_loaded_,
                              final_ln_bias_loaded_,
                              name(),
                              "final_layernorm");
  }

  void verify_loaded_weights(const std::string& prefix) const {
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->verify_loaded_weights(prefix + "blocks." +
                                          std::to_string(idx) + ".");
    }
    CHECK(final_ln_weight_loaded_)
        << "weight is not loaded for " << prefix + "final_layernorm.weight";
    CHECK(final_ln_bias_loaded_)
        << "bias is not loaded for " << prefix + "final_layernorm.bias";
  }

  void merge_loaded_weights() {
    for (int idx = 0; idx < blocks_->size(); ++idx) {
      layers_[idx]->merge_loaded_weights();
    }
  }

 private:
  int hidden_size_ = 0;
  bool final_ln_weight_loaded_ = false;
  bool final_ln_bias_loaded_ = false;

  KimiK2_5_VisionRotaryEmbedding rope_2d_{nullptr};
  torch::nn::ModuleList blocks_{nullptr};
  std::vector<KimiK2_5_VisionBlock> layers_;
  torch::nn::LayerNorm final_layernorm_{nullptr};
};
TORCH_MODULE(KimiK2_5_VisionEncoder);

class KimiK2_5_VisionTransformerImpl : public torch::nn::Module {
 public:
  KimiK2_5_VisionTransformerImpl(const ModelContext& context) {
    auto model_args = context.get_model_args();

    spatial_merge_size_ = model_args.mm_spatial_merge_size();
    spatial_merge_unit_ = static_cast<int>(std::pow(spatial_merge_size_, 2));

    patch_embed_ =
        register_module("patch_embed", KimiK2_5_VisionPatchEmbed(context));
    encoder_ = register_module("encoder", KimiK2_5_VisionEncoder(context));
  }

  torch::Tensor tpool_patch_merger(torch::Tensor hidden_states,
                                   torch::Tensor grid_thw) {
    std::vector<torch::Tensor> outputs;
    auto count = grid_thw.sizes()[0];
    outputs.reserve(count);

    int64_t offset = 0;
    auto grid_thw_cpu = grid_thw.cpu().to(torch::kLong).contiguous();
    for (int64_t idx = 0; idx < count; ++idx) {
      auto t = grid_thw_cpu[idx][0].item<int64_t>();
      auto h = grid_thw_cpu[idx][1].item<int64_t>();
      auto w = grid_thw_cpu[idx][2].item<int64_t>();

      CHECK_EQ(h % spatial_merge_size_, 0)
          << "height must be divisible by spatial_merge_size";
      CHECK_EQ(w % spatial_merge_size_, 0)
          << "width must be divisible by spatial_merge_size";

      auto token_num = t * h * w;
      auto seq = hidden_states.slice(0, offset, offset + token_num);
      offset += token_num;

      auto new_h = h / spatial_merge_size_;
      auto new_w = w / spatial_merge_size_;
      seq = seq.view({t,
                      new_h,
                      spatial_merge_size_,
                      new_w,
                      spatial_merge_size_,
                      hidden_states.size(-1)});
      seq = seq.permute({0, 1, 3, 2, 4, 5}).contiguous().mean(0);
      seq = seq.view(
          {new_h * new_w, spatial_merge_unit_, hidden_states.size(-1)});
      outputs.emplace_back(seq);
    }

    if (outputs.empty()) {
      return hidden_states.view(
          {0, spatial_merge_unit_, hidden_states.size(-1)});
    }
    return torch::cat(outputs, 0);
  }

  torch::Tensor forward(torch::Tensor hidden_states,
                        torch::Tensor grid_thw) {  // [batch,thw]
    hidden_states = patch_embed_(hidden_states, grid_thw);
    hidden_states = encoder_(hidden_states, grid_thw);
    // Align with MoonViT3dPretrainedModel:
    // return vision tower output after tpool patch merge.
    return tpool_patch_merger(hidden_states, grid_thw);
  }

  void load_state_dict(const StateDict& state_dict) {
    patch_embed_->load_state_dict(
        state_dict.get_dict_with_prefix("patch_embed."));

    auto encoder_state_dict = state_dict.get_dict_with_prefix("encoder.");
    if (encoder_state_dict.size() > 0) {
      encoder_->load_state_dict(encoder_state_dict);
    } else {
      // fallback for checkpoints without explicit encoder prefix
      encoder_->load_state_dict(state_dict);
    }
  }

  void verify_loaded_weights(const std::string& prefix) const {
    patch_embed_->verify_loaded_weights(prefix + "patch_embed.");
    encoder_->verify_loaded_weights(prefix + "encoder.");
  }

  void merge_loaded_weights() { encoder_->merge_loaded_weights(); }

 private:
  int spatial_merge_size_ = 0;
  int spatial_merge_unit_ = 0;

  KimiK2_5_VisionPatchEmbed patch_embed_{nullptr};
  KimiK2_5_VisionEncoder encoder_{nullptr};
};
TORCH_MODULE(KimiK2_5_VisionTransformer);

struct KimiK2_5_VLImageInputs {
  torch::Tensor pixel_values;
  torch::Tensor image_grid_thw;
};

struct KimiK2_5_VLVideoInputs {
  torch::Tensor pixel_values_videos;
  torch::Tensor video_grid_thw;
  torch::Tensor second_per_grid_ts;
};

class KimiK2_5_VLForConditionalGenerationImpl : public torch::nn::Module {
 public:
  KimiK2_5_VLForConditionalGenerationImpl(const ModelContext& context)
      : model_args_(context.get_model_args()),
        options_(context.get_tensor_options()) {
    auto parallel_args = context.get_parallel_args();
    const int32_t dp_size =
        parallel_args.dp_size() > 0 ? parallel_args.dp_size() : 1;
    const int32_t tp_size = parallel_args.world_size() / dp_size;
    CHECK_EQ(parallel_args.world_size(), tp_size * dp_size)
        << "invalid parallel config for kimi_k25: world_size("
        << parallel_args.world_size() << ") must be divisible by dp_size("
        << dp_size << ")";
    CHECK_LE(tp_size, 8) << "kimi_k25 only supports tp_size <= 8, got tp_size="
                         << tp_size
                         << " (world_size=" << parallel_args.world_size()
                         << ", dp_size=" << dp_size << ")";
    visual_ =
        register_module("vision_tower", KimiK2_5_VisionTransformer(context));
    auto mm_ptype = model_args_.mm_projector_type();
    if (mm_ptype == "patchmerger") {
      mm_projector_ =
          register_module("mm_projector", KimiK2_5_VisionPatchMerger(context));
    } else if (mm_ptype.empty() || mm_ptype == "none" ||
               mm_ptype == "identity") {
      // keep mm_projector_ as nullptr
    } else {
      CHECK(false) << "unsupported mm_projector_type for kimi_k25: "
                   << mm_ptype;
    }

    language_model_ = register_module(
        "language_model", npu::model::DeepseekV2ForCausalLM(context));
  }

  void prepare_encoder_input(
      const ModelInputParams& input_params,
      std::optional<KimiK2_5_VLImageInputs>& image_inputs,
      std::optional<KimiK2_5_VLVideoInputs>& video_inputs) {
    const auto& mm_data = input_params.multimodal.mm_data;
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values"))
      pixel_values = res.value();

    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw"))
      image_grid_thw = res.value();

    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos"))
      pixel_values_videos = res.value();

    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw"))
      video_grid_thw = res.value();

    torch::Tensor second_per_grid_ts;
    if (const auto& res = mm_data.get<torch::Tensor>("second_per_grid_ts"))
      second_per_grid_ts = res.value();

    if (pixel_values.defined() && image_grid_thw.defined())
      image_inputs = KimiK2_5_VLImageInputs{pixel_values, image_grid_thw};

    if (pixel_values_videos.defined() && video_grid_thw.defined() &&
        second_per_grid_ts.defined())
      video_inputs = KimiK2_5_VLVideoInputs{
          pixel_values_videos, video_grid_thw, second_per_grid_ts};
  }

  std::vector<torch::Tensor> process_vision_features(torch::Tensor pixel_values,
                                                     torch::Tensor grid_thws) {
    int n = grid_thws.size(0);
    auto n_patches_each_media = grid_thws.prod(-1);
    int max_infer_batch = std::max(n_patches_each_media.max().item<int>(),
                                   KIMIV_VT_INFER_MAX_PATCH_NUM);
    auto n_patches_tensor =
        n_patches_each_media.cpu().to(torch::kInt).contiguous();
    std::vector<int> n_patches_vec(
        n_patches_tensor.data_ptr<int>(),
        n_patches_tensor.data_ptr<int>() + n_patches_tensor.numel());

    std::vector<torch::Tensor> features;
    int pre_sum = 0;
    int current_group_start = 0;
    int current_group_patches = 0;

    for (int i = 0; i < n; i++) {
      int current_media_patches = n_patches_vec[i];
      if (current_group_patches + current_media_patches <= max_infer_batch) {
        current_group_patches += current_media_patches;
        continue;
      }

      if (current_group_start < i) {
        auto group_grid_thw = grid_thws.slice(0, current_group_start, i);
        int group_n_patches = 0;
        for (int j = current_group_start; j < i; j++) {
          group_n_patches += n_patches_vec[j];
        }
        auto group_input =
            pixel_values.slice(0, pre_sum, pre_sum + group_n_patches);
        auto group_output = visual_(group_input, group_grid_thw);
        features.push_back(mm_projector_ ? mm_projector_(group_output)
                                         : group_output);
        pre_sum += group_n_patches;
      }
      current_group_start = i;
      current_group_patches = current_media_patches;
    }

    if (current_group_start < n) {
      auto group_grid_thw = grid_thws.slice(0, current_group_start, n);
      int group_n_patches = 0;
      for (int j = current_group_start; j < n; j++) {
        group_n_patches += n_patches_vec[j];
      }
      auto group_input =
          pixel_values.slice(0, pre_sum, pre_sum + group_n_patches);
      auto group_output = visual_(group_input, group_grid_thw);
      features.push_back(mm_projector_ ? mm_projector_(group_output)
                                       : group_output);
    }

    return features;
  }

  MMDict get_multimodal_embeddings(const ModelInputParams& input_params) {
    std::optional<KimiK2_5_VLImageInputs> image_input;
    std::optional<KimiK2_5_VLVideoInputs> video_input;
    prepare_encoder_input(input_params, image_input, video_input);
    auto merge_size =
        model_args_.mm_image_merge_size() > 0
            ? model_args_.mm_image_merge_size()
            : std::max<int64_t>(model_args_.mm_spatial_merge_size(),
                                int64_t(2));
    MMDict multimodal_embeds;
    if (image_input) {
      // visual
      auto pixel_values = image_input->pixel_values.to(options_);
      auto grid_thw = image_input->image_grid_thw.to(pixel_values.device());
      CHECK(grid_thw.scalar_type() == torch::kInt32 ||
            grid_thw.scalar_type() == torch::kInt64)
          << "image_grid_thw must be int tensor, got dtype="
          << grid_thw.scalar_type();
      auto image_features = process_vision_features(pixel_values, grid_thw);
      auto image_embeds = torch::cat(image_features, 0);
      auto image_tokens =
          (image_input->image_grid_thw.prod(-1) / merge_size / merge_size)
              .cpu()
              .contiguous()
              .to(torch::kLong);
      std::vector<int64_t> image_tokens_vec(
          image_tokens.data_ptr<int64_t>(),
          image_tokens.data_ptr<int64_t>() + image_tokens.numel());
      multimodal_embeds["image|embedding"] =
          image_embeds.split(image_tokens_vec, 0 /*dim*/);
    }
    return multimodal_embeds;
  }

  torch::Tensor apply_mm_projector(torch::Tensor vision_embeddings) {
    if (!mm_projector_) {
      return vision_embeddings;
    }
    CHECK_EQ(vision_embeddings.dim(), 3)
        << "expect vision embeddings with shape [N, merge_unit, hidden]";
    return mm_projector_(vision_embeddings);
  }

  torch::Tensor generate_multimodal_mask(torch::Tensor input_ids) {
    auto special_token_ids = torch::tensor(
        {model_args_.image_token_id(), model_args_.video_token_id()},
        input_ids.options().dtype(torch::kInt64));
    auto is_multimodal = torch::isin(input_ids, special_token_ids);
    return is_multimodal;
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
    torch::Tensor multimodal_embeds;
    if (const auto& emb = mm_data.get<torch::Tensor>("embedding")) {
      multimodal_embeds = emb.value();
    }
    auto inputs_embeds =
        language_model_->get_npu_word_embedding()(input_ids, 0);
    if (!multimodal_embeds.defined()) {
      return inputs_embeds;
    }
    auto is_multimodal = generate_multimodal_mask(input_ids);
    inputs_embeds = merge_multimodal_embeddings(
        inputs_embeds, multimodal_embeds, is_multimodal);
    return inputs_embeds;
  }

  ModelOutput forward(const torch::Tensor& tokens,
                      const torch::Tensor& positions,
                      std::vector<KVCache>& kv_caches,
                      const ModelInputParams& input_params) {
    return language_model_(tokens, positions, kv_caches, input_params);
  }

  torch::Tensor logits(const torch::Tensor& hidden_states,
                       const torch::Tensor& seleted_idxes) {
    return language_model_->logits(hidden_states, seleted_idxes);
  }

  void load_model(std::unique_ptr<ModelLoader> loader) {
    LOG(INFO) << "loading vit / projector weight...";
    for (const auto& state_dict : loader->get_state_dicts()) {
      auto vision_dict = state_dict->get_dict_with_prefix("vision_tower.");
      if (vision_dict.size() == 0) {
        vision_dict = state_dict->get_dict_with_prefix("visual.");
      }
      visual_->load_state_dict(vision_dict);
      if (mm_projector_) {
        auto mm_projector_dict =
            state_dict->get_dict_with_prefix("mm_projector.");
        if (mm_projector_dict.size() > 0) {
          mm_projector_->load_state_dict(mm_projector_dict);
        }
      }
    }
    LOG(INFO) << "verifying vit weight...";
    // verify
    visual_->verify_loaded_weights("vision_tower.");
    visual_->merge_loaded_weights();

    if (mm_projector_) {
      LOG(INFO) << "verifying projector weight...";
      mm_projector_->verify_loaded_weights("mm_projector.");
      mm_projector_->merge_loaded_weights();
    }

    if (!model_args_.encoder_embedding_mode()) {
      LOG(INFO) << "loading llm weight...";
      language_model_->load_model_with_prefixes(std::move(loader),
                                                "language_model.model.",
                                                "language_model.lm_head.");
      LOG(INFO) << "loaded all weight.";
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

  KimiK2_5_VisionTransformer visual_{nullptr};
  KimiK2_5_VisionPatchMerger mm_projector_{nullptr};
  npu::model::DeepseekV2ForCausalLM language_model_{nullptr};
};
TORCH_MODULE(KimiK2_5_VLForConditionalGeneration);

using KimiK25MultimodalProcessor = MultimodalProcessor<KimiK25PromptProcessor,
                                                       KimiK25ImageProcessor,
                                                       KimiK25VideoProcessor>;
REGISTER_MULTIMODAL_PROCESSOR(kimi_k25, KimiK25MultimodalProcessor);
REGISTER_CAUSAL_VLM_MODEL(kimi_k25, KimiK2_5_VLForConditionalGeneration);

REGISTER_MODEL_ARGS(kimi_k25, [&] {
  // text config (Kimi-K2.5): args are under text_config.* in HF config.
  LOAD_ARG_OR(model_type, "model_type", "kimi_k25");
  LOAD_ARG_OR_FUNC(dtype, "dtype", [&] {
    return json.value_or<std::string>("torch_dtype", "bfloat16");
  });
  LOAD_ARG_OR(vocab_size, "text_config.vocab_size", 163840);
  LOAD_ARG_OR(hidden_act, "text_config.hidden_act", "silu");
  LOAD_ARG_OR(hidden_size, "text_config.hidden_size", 7168);
  LOAD_ARG_OR(initializer_range, "text_config.initializer_range", 0.02f);
  LOAD_ARG_OR(intermediate_size, "text_config.intermediate_size", 18432);
  LOAD_ARG_OR(n_layers, "text_config.num_hidden_layers", 61);
  LOAD_ARG_OR(n_heads, "text_config.num_attention_heads", 64);
  LOAD_ARG_OR(n_kv_heads, "text_config.num_key_value_heads", 64);
  LOAD_ARG_OR(
      max_position_embeddings, "text_config.max_position_embeddings", 262144);
  LOAD_ARG_OR(rms_norm_eps, "text_config.rms_norm_eps", 1e-05);
  LOAD_ARG_OR(rope_theta, "text_config.rope_theta", 50000.0f);

  LOAD_ARG_OR(attention_bias, "text_config.attention_bias", false);
  LOAD_ARG_OR(attention_dropout, "text_config.attention_dropout", 0.0f);

  // [Kimi-K2.5 config missing, keep DeepSeek-V3-compatible default]
  LOAD_ARG_OR(max_window_layers, "text_config.max_window_layers", 61);
  LOAD_ARG_OR(tie_word_embeddings, "text_config.tie_word_embeddings", false);

  // DeepSeek-V3 style MoE / MLA args.
  LOAD_ARG_OR(first_k_dense_replace, "text_config.first_k_dense_replace", 1);
  LOAD_ARG_OR(moe_layer_freq, "text_config.moe_layer_freq", 1);
  LOAD_ARG_OR(topk_method, "text_config.topk_method", "noaux_tc");
  LOAD_ARG_OR(n_routed_experts, "text_config.n_routed_experts", 384);
  LOAD_ARG_OR(n_shared_experts, "text_config.n_shared_experts", 1);
  LOAD_ARG_OR(num_experts_per_tok, "text_config.num_experts_per_tok", 8);
  LOAD_ARG_OR(moe_intermediate_size, "text_config.moe_intermediate_size", 2048);
  LOAD_ARG_OR(
      routed_scaling_factor, "text_config.routed_scaling_factor", 2.827f);
  LOAD_ARG_OR(norm_topk_prob, "text_config.norm_topk_prob", true);
  LOAD_ARG_OR(n_group, "text_config.n_group", 1);
  LOAD_ARG_OR(topk_group, "text_config.topk_group", 1);
  LOAD_ARG_OR(qk_nope_head_dim, "text_config.qk_nope_head_dim", 128);
  LOAD_ARG_OR(qk_rope_head_dim, "text_config.qk_rope_head_dim", 64);
  LOAD_ARG_OR(v_head_dim, "text_config.v_head_dim", 128);
  LOAD_ARG_OR(q_lora_rank, "text_config.q_lora_rank", 1536);
  LOAD_ARG_OR(kv_lora_rank, "text_config.kv_lora_rank", 512);
  LOAD_ARG_OR(scoring_func, "text_config.scoring_func", "sigmoid");

  SET_ARG(head_dim, args->qk_nope_head_dim() + args->qk_rope_head_dim());
  LOAD_ARG_OR_FUNC(
      rotary_dim, "rotary_dim", [&] { return args->qk_rope_head_dim(); });

  LOAD_ARG(rope_scaling_rope_type, "text_config.rope_scaling.type");
  if (args->rope_scaling_rope_type().empty() ||
      args->rope_scaling_rope_type() == "default" ||
      args->rope_scaling_rope_type() == "yarn") {
    // Kimi-K2.5 text config uses HF rope_scaling.type="yarn", while the
    // DeepSeek MLA implementation in xLLM keys off "deepseek_yarn".
    args->rope_scaling_rope_type() = "deepseek_yarn";
  }
  LOAD_ARG(rope_scaling_beta_fast, "text_config.rope_scaling.beta_fast");
  LOAD_ARG(rope_scaling_beta_slow, "text_config.rope_scaling.beta_slow");
  LOAD_ARG(rope_scaling_factor, "text_config.rope_scaling.factor");
  LOAD_ARG_OR(rope_extrapolation_factor,
              "text_config.rope_scaling.extrapolation_factor",
              1.0f);
  LOAD_ARG(rope_scaling_mscale, "text_config.rope_scaling.mscale");
  LOAD_ARG(rope_scaling_mscale_all_dim,
           "text_config.rope_scaling.mscale_all_dim");
  LOAD_ARG(rope_scaling_original_max_position_embeddings,
           "text_config.rope_scaling.original_max_position_embeddings");
  LOAD_ARG_OR(
      rope_scaling_attn_factor, "text_config.rope_scaling.attn_factor", 1.0f);
  LOAD_ARG_OR(
      num_nextn_predict_layers, "text_config.num_nextn_predict_layers", 1);

  LOAD_ARG_OR(bos_token_id, "text_config.bos_token_id", 163584);
  LOAD_ARG_OR(eos_token_id, "text_config.eos_token_id", 163585);
  LOAD_ARG_OR(pad_token_id, "text_config.pad_token_id", 163839);

  // Kimi-K2.5 uses media_* special tokens instead of Qwen's vision/image/video
  // token family. The official config only exposes media_placeholder_token_id;
  // the begin/content/end ids come from the tokenizer special token table
  // below.

  LOAD_ARG_OR_FUNC(vision_start_token_id, "vision_start_token_id", [&] {
    return int32_t(163602);  // <|media_begin|>
  });
  LOAD_ARG_OR_FUNC(vision_end_token_id, "vision_end_token_id", [&] {
    return int32_t(163604);  // <|media_end|>
  });
  LOAD_ARG_OR_FUNC(vision_token_id, "vision_token_id", [&] {
    return int32_t(163603);  // <|media_content|>
  });
  LOAD_ARG_OR_FUNC(image_token_id, "image_token_id", [&] {
    return int32_t(163605);  // <|media_pad|>
  });
  LOAD_ARG_OR_FUNC(video_token_id, "video_token_id", [&] {
    return int32_t(163605);  // <|media_pad|>
  });

  // vision_config inferred mapping (Kimi-K2.5):
  LOAD_ARG_OR(mm_num_hidden_layers, "vision_config.vt_num_hidden_layers", 27);
  // vt_hidden_act is not provided in Kimi-K2.5 config, keep default "silu".
  LOAD_ARG_OR(mm_hidden_act, "vision_config.vt_hidden_act", "silu");
  LOAD_ARG_OR(mm_hidden_size, "vision_config.mm_hidden_size", 1152);
  LOAD_ARG_OR(mm_intermediate_size, "vision_config.vt_intermediate_size", 4304);
  LOAD_ARG_OR(
      mm_num_attention_heads, "vision_config.vt_num_attention_heads", 16);

  // Projector-related args from Kimi-K2.5 vision_config.
  LOAD_ARG_OR(mm_projection_dim, "vision_config.text_hidden_size", 7168);
  LOAD_ARG_OR(
      mm_projector_type, "vision_config.mm_projector_type", "patchmerger");
  LOAD_ARG_OR(
      mm_projector_hidden_act, "vision_config.projector_hidden_act", "gelu");
  // NOTE: projector_ln_eps is mapped to mm_layer_norm_eps by inference.
  LOAD_ARG_OR(mm_layer_norm_eps, "vision_config.projector_ln_eps", 1e-05f);

  LOAD_ARG_OR(mm_patch_size, "vision_config.patch_size", 14);
  // Kimi-K2.5 uses merge_kernel_size (e.g. [2,2]); map first dim by inference.
  LOAD_ARG_OR_FUNC(
      mm_spatial_merge_size, "vision_config.spatial_merge_size", [&] {
        if (auto merge_kernel_size = json.value<std::vector<int64_t>>(
                "vision_config.merge_kernel_size");
            merge_kernel_size.has_value() && !merge_kernel_size->empty()) {
          return (*merge_kernel_size)[0];
        }
        return int64_t(2);
      });
  LOAD_ARG_OR_FUNC(mm_image_merge_size, "vision_config.image_merge_size", [&] {
    return args->mm_spatial_merge_size() > 0 ? args->mm_spatial_merge_size()
                                             : int64_t(2);
  });

  LOAD_ARG_OR_FUNC(mm_head_dim, "vision_config.head_dim", [&] {
    return args->mm_hidden_size() / args->mm_num_attention_heads();
  });
  // No explicit spatial_patch_size in Kimi-K2.5; fallback to patch_size.
  LOAD_ARG_OR_FUNC(mm_spatial_patch_size,
                   "vision_config.spatial_patch_size",
                   [&] { return args->mm_patch_size(); });

  // Original qwen2_5_vl vision args not found in Kimi-K2.5 vision_config.
  // Keep defaults and mark as unmapped:
  // mm_num_channels is required by patch_embed in_features (in_dim * patch^2).
  LOAD_ARG_OR(mm_num_channels, "vision_config.in_chans", 3);
  // LOAD_ARG_OR(mm_window_size, "vision_config.window_size", 112);
  // LOAD_ARG_OR(mm_fullatt_block_indexes,
  //             "vision_config.fullatt_block_indexes",
  //             std::vector<int64_t>({7, 15, 23, 31}));
  // LOAD_ARG_OR(mm_tokens_per_second, "vision_config.tokens_per_second", 2);
  // LOAD_ARG_OR(mm_temporal_patch_size, "vision_config.temporal_patch_size",
  // 2);

  // [Compared with qwen2_5_vl] qwen mrope args are absent in Kimi-K2.5 config.
  // LOAD_ARG(rope_scaling_mrope_section, "rope_scaling.mrope_section");

  // Add Args For Kimi-K2.5
  LOAD_ARG_OR(mm_init_pos_emb_time, "vision_config.init_pos_emb_time", 4);
  LOAD_ARG_OR(mm_init_pos_emb_width, "vision_config.init_pos_emb_width", 64);
  LOAD_ARG_OR(mm_init_pos_emb_height, "vision_config.init_pos_emb_height", 64);

  // New Kimi-K2.5 vision_config keys currently not mapped to ModelArgs:
  // - _attn_implementation
  // - merge_type
  // - pos_emb_type
  // - video_attn_type
  // New Kimi-K2.5 config keys not registered in ModelArgs:
  // - architectures
  // - aux_loss_alpha
  // - seq_aux
  // - transformers_version
  // - use_cache
  SET_ARG(stop_token_ids, std::unordered_set<int32_t>({163585, 163586}));
});

REGISTER_TOKENIZER_ARGS(kimi_k25, [&] {
  // Kimi-K2.5 uses TikTokenTokenizer with tiktoken.model.
  // HF files:
  // - tokenizer_config.json (tokenizer_class=TikTokenTokenizer)
  // - tiktoken.model
  SET_ARG(tokenizer_type, "tiktoken");
  SET_ARG(vocab_file, "tiktoken.model");

  // ref:
  // https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenizer_config.json
  const std::vector<SpecialToken> special_tokens(
      {{"[BOS]", 163584},
       {"[EOS]", 163585},
       {"<|im_end|>", 163586},
       {"<|im_user|>", 163587},
       {"<|im_assistant|>", 163588},
       {"<|start_header_id|>", 163590},
       {"<|end_header_id|>", 163591},
       {"[EOT]", 163593},
       {"<|im_system|>", 163594},
       {"<|tool_calls_section_begin|>", 163595},
       {"<|tool_calls_section_end|>", 163596},
       {"<|tool_call_begin|>", 163597},
       {"<|tool_call_argument_begin|>", 163598},
       {"<|tool_call_end|>", 163599},
       {"<|im_middle|>", 163601},
       {"<|media_begin|>", 163602},
       {"<|media_content|>", 163603},
       {"<|media_end|>", 163604},
       {"<|media_pad|>", 163605},
       {"<think>", 163606},
       {"</think>", 163607},
       {"[UNK]", 163838},
       {"[PAD]", 163839}});
  SET_ARG(special_tokens, special_tokens);

  // Keep parser tokens visible in decoded output instead of stripping them.
  const std::vector<std::string> parser_visible_special_tokens(
      {"<|tool_calls_section_begin|>",
       "<|tool_calls_section_end|>",
       "<|tool_call_begin|>",
       "<|tool_call_argument_begin|>",
       "<|tool_call_end|>",
       "<think>",
       "</think>"});
  SET_ARG(visible_special_tokens, parser_visible_special_tokens);

  // ref:
  // https://huggingface.co/moonshotai/Kimi-K2.5/blob/main/tokenization_kimi.py#L53-L62
  // N.B. re2 doesn't support character class intersection (&&) or subtraction
  // (--). Since [\p{Han}]+ is the first branch and matches all Han characters
  // first, we can safely remove the '&&[^\p{Han}]' part from subsequent
  // branches. N.B. replaced '\s+(?!\S)' with '\s+[^\s]' - re2 doesn't support
  // negative lookahead.
  const std::string pattern_str =
      R"([\p{Han}]+|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]*[\p{Ll}\p{Lm}\p{Lo}\p{M}]+(?i:'s|'t|'re|'ve|'m|'ll|'d)?|[^\r\n\p{L}\p{N}]?[\p{Lu}\p{Lt}\p{Lm}\p{Lo}\p{M}]+[\p{Ll}\p{Lm}\p{Lo}\p{M}]*(?i:'s|'t|'re|'ve|'m|'ll|'d)?|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+[^\s]|\s+)";
  SET_ARG(pattern, pattern_str);
});
}  // namespace xllm
