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

#pragma once

#include <torch/torch.h>

#include <atomic>
#include <cstdint>
#include <mutex>
#include <string>
#include <unordered_map>
#include <vector>

#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/state_dict/state_dict.h"
#include "npu_base_layer.h"
#include "xllm_atb_layers/core/include/atb_speed/base/hosttensor_binder.h"
#include "xllm_atb_layers/core/include/atb_speed/base/model.h"
#include "xllm_atb_layers/core/include/atb_speed/log.h"
#include "xllm_atb_layers/core/include/atb_speed/utils/model_factory.h"
#include "xllm_atb_layers/models/onerec/layer/block_layer.h"
#include "xllm_atb_layers/operations/fusion/utils.h"

namespace xllm {
namespace layer {

class NpuOneRecBlockLayerImpl final : public BaseLayer {
 public:
  explicit NpuOneRecBlockLayerImpl(const ModelContext& context,
                                   bool is_decoder = false,
                                   int32_t layer_id = 0);

  ~NpuOneRecBlockLayerImpl() override = default;

  void load_state_dict(const StateDict& state_dict) override;

  void verify_loaded_weights(const std::string& prefix) const;

  void merge_loaded_weights() override;

  int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        ModelInputParams& input_params,
                        torch::Tensor* encoder_output = nullptr,
                        int32_t node_id = 0,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        const torch::Tensor& expert_array = torch::Tensor());

  void set_cross_kv_cache_for_graph(const torch::Tensor& k_cache,
                                    const torch::Tensor& v_cache);

 private:
  void param_from_args(atb_speed::onerec::BlockLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill,
                       const ModelInputParams* input_params = nullptr);

  void build_encoder_node_variant_pack(atb_speed::Model::Node& node,
                                       torch::Tensor& x,
                                       at::Tensor& attn_mask,
                                       ModelInputParams& input_params,
                                       bool is_prefill,
                                       int32_t layer_id = 0);

  void build_decoder_node_variant_pack(atb_speed::Model::Node& node,
                                       torch::Tensor& x,
                                       at::Tensor& attn_mask,
                                       KVCache& kv_cache,
                                       ModelInputParams& input_params,
                                       bool is_prefill,
                                       bool is_first_prefill,
                                       torch::Tensor* encoder_output = nullptr,
                                       int32_t layer_id = 0);

  void build_decoder_moe_node_variant_pack(
      atb_speed::Model::Node& node,
      torch::Tensor& x,
      at::Tensor& attn_mask,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      bool is_prefill,
      bool is_first_prefill,
      torch::Tensor* encoder_output = nullptr,
      int32_t layer_id = 0,
      const torch::Tensor& expert_array = torch::Tensor());

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::onerec::BlockLayerParam& param);

  int64_t init_attn_mask();

  int32_t setup_common_decoder_tensors(
      atb_speed::Model::Node& node,
      torch::Tensor& x,
      at::Tensor& attn_mask,
      KVCache& kv_cache,
      ModelInputParams& input_params,
      const atb_speed::onerec::BlockLayerParam& param,
      bool is_first_prefill,
      torch::Tensor* encoder_output = nullptr,
      int32_t start_tensor_idx = 0);

  void resize_experts_weights(int32_t num_of_device_experts);
  void process_expert_weights(const StateDict& state_dict,
                              const std::string& state_key,
                              const torch::Tensor& tensor);
  void process_shared_expert_weights(const StateDict& state_dict,
                                     const std::string& name,
                                     const torch::Tensor& tensor);
  void merge_experts_weights();
  void merge_shared_experts_weights();
  bool validate_decoder_moe_weights(const std::string& prefix) const;
  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      bool transpose = false);
  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_gate,
                                      std::vector<torch::Tensor>& experts_up,
                                      bool transpose = false);
  int32_t extract_expert_index(const std::string& name);
  std::string extract_endswith(const std::string& input);

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node prefill_node_atb_;
  atb_speed::Model::Node decode_node_;
  atb_speed::Model::Node decoder_prefill_only_decode_node_;
  atb_speed::Model::Node decoder_prefill_only_decode_node_atb_;
  std::string model_name_;
  atb_speed::onerec::BlockLayerParam prefill_param_;
  atb_speed::onerec::BlockLayerParam prefill_param_atb_;
  atb_speed::onerec::BlockLayerParam decode_param_;
  atb_speed::onerec::BlockLayerParam decoder_prefill_only_decode_param_;
  atb_speed::onerec::BlockLayerParam decoder_prefill_only_decode_param_atb_;

  atb::Tensor internal_tensors_;
  atb::Tensor placeholder_;

  at::Tensor encoder_output_contiguous_;
  at::Tensor decode_attn_mask_;
  at::Tensor prefill_attn_mask_;
  at::Tensor at_placeholder_;
  std::vector<int32_t> placeholder_vec_;
  torch::Tensor cross_k_cache_;
  torch::Tensor cross_v_cache_;
  std::vector<int32_t> seq_lens_vec_;

  int32_t device_id_ = 0;
  bool is_decoder_ = false;
  int32_t layer_id_ = 0;

  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;
  std::mutex experts_mutex_;
  int32_t start_expert_id_ = 0;
  int32_t end_expert_id_ = 0;
  int32_t num_experts_per_partition_ = 0;
  int32_t ep_size_ = 1;
  int32_t ep_local_tp_rank_ = 0;
  int32_t ep_local_tp_size_ = 1;

  std::vector<torch::Tensor> shared_expert_gate_weights_;
  std::vector<torch::Tensor> shared_expert_up_weights_;
  std::vector<torch::Tensor> shared_expert_down_weights_;
  std::unordered_map<std::string, torch::Tensor> shared_expert_weights_map_;

  torch::Tensor expert_group_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
};
TORCH_MODULE(NpuOneRecBlockLayer);

}  // namespace layer
}  // namespace xllm
