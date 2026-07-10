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

#include "base_loader.h"

namespace xllm {
namespace layer {

class DeekseekV32DecoderLoader : public BaseLoader {
 public:
  DeekseekV32DecoderLoader(uint64_t weight_count,
                           const ModelContext& context,
                           int32_t layer_id,
                           int32_t prefill_firstKDenseReplace,
                           int32_t prefill_numOfDeviceExperts,
                           int32_t prefill_qkRopeHeadDim,
                           int32_t prefill_numAttentionHeadsPerRank,
                           int32_t decode_worldSize,
                           int32_t qk_nope_head_dim_,
                           int32_t kv_lora_rank,
                           int32_t num_key_value_heads,
                           int32_t v_head_dim,
                           bool prefill_isBF16,
                           bool decode_isBF16,
                           const std::vector<int32_t>& attn_linear_quant_types,
                           bool skip_topk,
                           LoadMode mode = LoadMode::kEager);

  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights(const std::string& prefix) const override;

 protected:
  void merge_host_at_weights() override;

  void initialize_device_expert_list(int num_device, int num_device_expert);

  int extract_expert_index(const std::string& name);

  std::string get_expert_shm_key(int32_t layer_id,
                                 int32_t expert_index,
                                 const std::string& suffix);

  int get_mapped_index(const std::string& name,
                       const std::unordered_map<std::string, int>& mapping);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim);

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim,
                                   int local_tp_rank,
                                   int local_tp_size);

  std::string extract_endswith(const std::string& input);

  void set_kv_weight(const StateDict& state_dict,
                     const std::string& tensor_name,
                     int weight_position,
                     int dim);

  void preprocess_linear_for_rope();

  void process_expert_weights(const StateDict& state_dict,
                              const std::string& name,
                              const torch::Tensor& tensor);

  void process_shared_expert_weights(const StateDict& state_dict,
                                     const std::string& name,
                                     const torch::Tensor& tensor);

  void process_mlp_common_weights(const StateDict& state_dict,
                                  const std::string& name,
                                  const torch::Tensor& tensor);

  void process_general_weights(const StateDict& state_dict,
                               const std::string& name,
                               const torch::Tensor& tensor);

  void convert_descaled_weights_to_float();

  void convert_offsets_to_int8();

  torch::Tensor convert_fp16_to_int64(const torch::Tensor& fp16_tensor);

  void handle_device_specific_bias();

  void merge_shared_experts_weights();

  void merge_experts_weights();

  void preprocess_w4a8_dynamic_experts_weights();

  bool use_quant_weight_mapping() const;
  bool is_attn_dynamic_desc(int32_t index) const;
  bool is_attn_quant_desc(int32_t index) const;
  bool should_skip_indexer_weight(const std::string& name) const;
  void reset_skipped_indexer_weights();

  int get_w4a8_expert_shard_dim(const std::string& suffix) const;

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      bool transpose = false);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_up,
                                      std::vector<torch::Tensor>& experts_gate,
                                      bool transpose = false);

  void squeeze_experts_weights();

  void update_expert_weight();

  void prepare_expert_weight(const std::vector<int32_t>& expert_list);

  void initialize_weight_tensors(const torch::TensorOptions& options);

  void initialize_tensors(const torch::TensorOptions& options);

  torch::Tensor view_tensor(torch::Tensor weight,
                            const std::string& name,
                            bool pre_view);

  torch::Tensor view_indexer_tensor(torch::Tensor weight,
                                    const std::string& name,
                                    bool pre_view);

  void reserve_experts_weights(int num_of_device_experts);

  torch::Tensor trans_rope_weight(torch::Tensor weight);
  torch::Tensor trans_front_rope_weight(torch::Tensor weight);

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
  int32_t quant_group_size_ = 0;

  int32_t layer_id_;
  int32_t qk_nope_head_dim_;
  int32_t kv_lora_rank_;
  int32_t v_head_dim_;
  int32_t index_n_heads_;
  int32_t index_head_dim_;
  int32_t num_key_value_heads_;
  int32_t prefill_firstKDenseReplace_;
  int32_t prefill_numOfDeviceExperts_;
  int32_t prefill_qkRopeHeadDim_;
  int32_t prefill_numAttentionHeadsPerRank_;
  int32_t decode_worldSize_;
  bool prefill_isBF16_;
  bool decode_isBF16_;
  bool skip_topk_;
  bool indexer_rope_interleave_;
  // Compatibility vector: entries may be legacy LinearType or new LinearDesc.
  std::vector<int32_t> attn_linear_quant_types_;
  std::mutex shared_experts_mutex_;
  std::mutex experts_mutex_;

  torch::Tensor tensor_placeholder_;
};
}  // namespace layer
}  // namespace xllm
