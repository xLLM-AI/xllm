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

#include "framework/eplb/expert_buffer_manager.h"
#include "framework/eplb/expert_weight_buffer_shm.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/npu_dp_ep_padding.h"
#include "framework/state_dict/state_dict.h"
#include "loader/deepseek_v32_decoder_loader.h"
#include "npu_base_layer.h"
#include "npu_deepseek_v2_decoder_layer_impl.h"
#include "xllm_atb_layers/models/deepseekv2/layer/decoder_layer.h"

namespace xllm {
namespace layer {

class NpuDeepseekV32DecoderLayerImpl : public BaseLayer {
 public:
  explicit NpuDeepseekV32DecoderLayerImpl(const ModelContext& context,
                                          const int32_t layer_id);

  ~NpuDeepseekV32DecoderLayerImpl() {};

  virtual void merge_loaded_weights() override;

  torch::Tensor build_expert_routing_map(std::vector<int32_t> expert_lists);

  void prepare_expert_weight(const std::vector<int32_t>& expert_list);

  void update_expert_weight();

  virtual int64_t init_layer() override;

  torch::Tensor forward(torch::Tensor& x,
                        torch::Tensor& cos_pos,
                        torch::Tensor& sin_pos,
                        torch::Tensor& attn_mask,
                        KVCache& kv_cache,
                        const ModelInputParams& input_params,
                        aclrtEvent* event = nullptr,
                        std::atomic<bool>* event_flag = nullptr,
                        int node_id = 0);

  torch::Tensor forward_with_topk(torch::Tensor& x,
                                  torch::Tensor& cos_pos,
                                  torch::Tensor& sin_pos,
                                  torch::Tensor& attn_mask,
                                  KVCache& kv_cache,
                                  const ModelInputParams& input_params,
                                  const torch::Tensor& shared_topk_indices,
                                  torch::Tensor* output_topk_indices,
                                  aclrtEvent* event = nullptr,
                                  std::atomic<bool>* event_flag = nullptr,
                                  int node_id = 0);

  torch::Tensor forward_with_mtp_topk_fallback(
      torch::Tensor& x,
      torch::Tensor& cos_pos,
      torch::Tensor& sin_pos,
      torch::Tensor& attn_mask,
      KVCache& kv_cache,
      const ModelInputParams& input_params,
      torch::Tensor* output_topk_indices,
      aclrtEvent* event = nullptr,
      std::atomic<bool>* event_flag = nullptr,
      int node_id = 0);

  bool is_topk_sharing_enabled() const { return skip_topk_ || output_topk_; }
  bool skip_topk() const { return skip_topk_; }
  bool output_topk() const { return output_topk_; }
  bool has_mtp_topk_fallback() const { return has_mtp_topk_fallback_; }

 private:
  struct ShardingConfig {
    bool is_sharded;
    int index;
    bool use_dp_sharding = false;
  };

  std::string get_expert_shm_key(int32_t layer_id,
                                 int32_t expert_index,
                                 const std::string& suffix);

  void initialize_tensors(const torch::TensorOptions& options);

  void param_from_args(atb_speed::deepseekV2::DecoderLayerParam& param,
                       const ModelArgs& args,
                       const ParallelArgs& parallel_args,
                       bool is_prefill);

  void reserve_experts_weights(int num_of_device_experts);

  void initialize_device_expert_list(int numdevice, int num_layers);

  void initialize_basic_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args,
      bool is_prefill);

  void initialize_attention_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_mlp_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ModelArgs& args,
      const ParallelArgs& parallel_args);

  void initialize_parallel_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param,
      const ParallelArgs& parallel_args);

  void initialize_quantization_parameters(
      atb_speed::deepseekV2::DecoderLayerParam& param);

  void merge_and_copy_gate_up_weights(
      torch::Tensor& target_buffer,
      const std::vector<torch::Tensor>& experts_gate,
      const std::vector<torch::Tensor>& experts_up,
      bool do_transpose = false);
  void merge_and_copy_down_weights(
      torch::Tensor& target_buffer,
      const std::vector<torch::Tensor>& experts_down);

  int64_t init_node(atb_speed::Model::Node& node,
                    atb_speed::deepseekV2::DecoderLayerParam& param);

  void build_node_variant_pack(atb_speed::Model::Node& node,
                               torch::Tensor& x,
                               torch::Tensor& cos_pos,
                               torch::Tensor& sin_pos,
                               torch::Tensor& attn_mask,
                               KVCache& kv_cache,
                               ModelInputParams& input_params,
                               bool is_prefill,
                               const torch::Tensor& shared_topk_indices,
                               torch::Tensor* output_topk_indices,
                               bool skip_topk,
                               bool output_topk);

  torch::Tensor block_tables_placeholder_;
  std::string model_name_;

  int32_t device_id_;
  int32_t layer_id_;
  int32_t num_key_value_heads_;
  int32_t qk_nope_head_dim_;
  int32_t v_head_dim_;
  int32_t kv_lora_rank_;
  int32_t qk_rope_head_dim_;
  bool skip_topk_ = false;
  bool output_topk_ = false;
  bool has_mtp_topk_fallback_ = false;

  int32_t rank_;
  int32_t first_k_dense_replace_;
  int32_t n_layers_;
  int32_t localWorldSize_;
  int32_t ep_size_;
  int32_t num_experts_;
  int32_t num_experts_per_partition_;
  int32_t ep_local_tp_size_;
  int32_t ep_local_tp_rank_;
  int32_t start_expert_id_;
  int32_t end_expert_id_;
  int32_t ep_rank_;
  int32_t redundant_experts_num_;

  int32_t dp_size_;
  int32_t dp_local_tp_size_;
  int32_t dp_rank_;
  int32_t dp_local_tp_rank_;
  int32_t cp_size_;

  float sm_scale_;
  int32_t quant_group_size_ = 0;
  int32_t num_speculative_tokens_ = 0;
  // Compatibility vector: entries may be legacy LinearType or new LinearDesc.
  std::vector<int32_t> attn_linear_quant_types_;

  atb_speed::deepseekV2::DecoderLayerParam prefill_param_;
  atb_speed::deepseekV2::DecoderLayerParam decode_param_;
  atb_speed::deepseekV2::DecoderLayerParam mtp_prefill_fallback_param_;
  atb_speed::deepseekV2::DecoderLayerParam mtp_decode_fallback_param_;

  atb_speed::Model::Node prefill_node_;
  atb_speed::Model::Node decode_node_;
  atb_speed::Model::Node mtp_prefill_fallback_node_;
  atb_speed::Model::Node mtp_decode_fallback_node_;

  atb::Tensor internal_tensor_;

  torch::Tensor at_cumsum_;
  torch::Tensor tensor_placeholder_;
  torch::Tensor slot_tensor_placeholder_;
  torch::Tensor int_tensor_placeholder_;
  torch::Tensor decode_attn_mask_;
  torch::Tensor expert_group_;
  torch::Tensor one_hot_;
  torch::Tensor zero_hot_;
  torch::Tensor final_hidden_states_;
  torch::Tensor at_start_expert_id_;
  torch::Tensor at_in_device_expert_count_;

  std::vector<int32_t> int_placeholder_;

  torch::Tensor expert_routing_map_;
  torch::Tensor expert_routing_map_buffer_;
};
TORCH_MODULE(NpuDeepseekV32DecoderLayer);

std::vector<torch::Tensor> get_dtp_inputs(torch::Tensor token_size_per_dp_group,
                                          int32_t dp_local_tp_size,
                                          int32_t dp_rank,
                                          int32_t dp_size,
                                          int32_t rank,
                                          at::Device device);

}  // namespace layer
}  // namespace xllm
