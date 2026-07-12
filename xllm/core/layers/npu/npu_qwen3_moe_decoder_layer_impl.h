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

#include <glog/logging.h>
#include <torch/torch.h>
#include <torch_npu/torch_npu.h>

#include <nlohmann/json.hpp>

#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/npu_dp_ep_padding.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "loader/qwen3_moe_decoder_loader.h"
#include "npu_base_layer.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/models/qwen3/layer/moe_decoder_layer.h"

namespace xllm {
namespace layer {

class NpuQwen3MoeDecoderLayerImpl : public BaseLayer {
 public:
  explicit NpuQwen3MoeDecoderLayerImpl(const ModelContext& context,
                                       const int32_t layer_id);

  ~NpuQwen3MoeDecoderLayerImpl() override = default;

  void merge_loaded_weights() override;

  int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        std::optional<torch::Tensor>& residual,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

 private:
  struct ShardingConfig {
    bool is_sharded;
    int index;
    bool use_dp_sharding = false;
  };

  void initialize_tensors(const torch::TensorOptions& options);

  void param_from_args(atb_speed::qwen::MoeDecoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill);

  void initialize_basic_parameters(atb_speed::qwen::MoeDecoderLayerParam& param,
                                   const ModelArgs& args,
                                   const ParallelArgs& parallel_args,
                                   bool is_prefill);

  void initialize_attention_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_mlp_parameters(atb_speed::qwen::MoeDecoderLayerParam& param,
                                 const ModelArgs& args,
                                 const ParallelArgs& parallel_args);

  void initialize_parallel_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param,
      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::qwen::MoeDecoderLayerParam& param);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::qwen::MoeDecoderLayerParam& param);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               std::optional<torch::Tensor>& residual,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               const ModelInputParams& input_params,
                               bool is_prefill,
                               bool use_graph_decode_input);

  torch::Tensor block_tables_placeholder_;
  std::string model_name_;

  int32_t device_id_;
  int32_t layer_id_;

  int32_t ep_size_;
  int32_t num_experts_;
  int32_t num_experts_per_partition_;
  int32_t ep_local_tp_size_;
  int32_t ep_local_tp_rank_;
  int32_t start_expert_id_;
  int32_t end_expert_id_;
  int32_t ep_rank_;

  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;

  int32_t num_speculative_tokens_ = 0;
  atb_speed::qwen::MoeDecoderLayerParam prefill_param_;
  atb_speed::qwen::MoeDecoderLayerParam decode_graph_param_;
  atb_speed::qwen::MoeDecoderLayerParam decode_eager_param_;

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_graph_node_;
  atb_speed::Model::Node decode_eager_node_;

  atb::Tensor internal_tensor_;

  torch::Tensor tensor_placeholder_;
  torch::Tensor slot_tensor_placeholder_;
  torch::Tensor int_tensor_placeholder_;
  torch::Tensor decode_attn_mask_;
  torch::Tensor expert_group_;
  torch::Tensor quant_add_norm_scaling_;
  torch::Tensor quant_add_norm_offset_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
  torch::Tensor final_hidden_states_;

  std::vector<int32_t> int_placeholder_;

  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;

  std::mutex shared_experts_mutex_;
  std::mutex experts_mutex_;
};
TORCH_MODULE(NpuQwen3MoeDecoderLayer);

std::vector<torch::Tensor> get_dtp_inputs(torch::Tensor token_size_per_dp_group,
                                          int32_t dp_local_tp_size,
                                          int32_t dp_rank,
                                          int32_t dp_size,
                                          int32_t rank,
                                          at::Device device);

}  // namespace layer
}  // namespace xllm
