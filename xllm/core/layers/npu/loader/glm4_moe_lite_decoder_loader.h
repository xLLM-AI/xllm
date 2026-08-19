/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#pragma once

#include "base_loader.h"

static const uint64_t WEIGHT_COUNT_PER_LAYER = 84;

namespace xllm {
namespace layer {

class Glm4MoeDecoderLiteLoader : public BaseLoader {
 public:
  Glm4MoeDecoderLiteLoader(uint64_t weight_count,
                           const ModelContext& context,
                           int32_t layer_id,
                           int32_t prefill_param_firstKDenseReplace,
                           LoadMode mode = LoadMode::kEager);
  void load_state_dict(const StateDict& state_dict) override;
  void verify_loaded_weights() const override;
  void resize_experts_weights(int num_of_device_experts) override;

 protected:
  void merge_host_at_weights() override;

  int32_t layer_id_;
  int32_t qk_nope_head_dim_;
  int32_t qk_rope_head_dim_;
  int32_t kv_lora_rank_;
  int32_t v_head_dim_;

  int32_t num_heads_;
  int32_t actual_n_heads_;
  int32_t num_key_value_heads_;
  int32_t prefill_param_firstKDenseReplace_;

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

  torch::Tensor tensor_placeholder_;

  std::unordered_map<std::string, torch::Tensor> shared_experts_weights_;
  std::unordered_map<std::string, std::vector<torch::Tensor>> experts_weights_;

  std::mutex shared_experts_mutex_;
  std::mutex experts_mutex_;

  torch::ScalarType dtype_;

  void set_kv_weight(const StateDict& state_dict,
                     const std::string& tensor_name);

  void padding_qb_weight(const StateDict& state_dict,
                         const std::string& name,
                         const torch::Tensor& tensor);

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

  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim);
  torch::Tensor get_sharded_tensor(const StateDict& state_dict,
                                   const std::string& name,
                                   int dim,
                                   int local_tp_rank,
                                   int local_tp_size);

  std::string extract_endswith(const std::string& input);

  int extract_expert_index(const std::string& name);

  void merge_shared_experts_weights();

  void merge_experts_weights();

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts,
                                      bool transpose = false);

  torch::Tensor merge_experts_weights(std::vector<torch::Tensor>& experts_up,
                                      std::vector<torch::Tensor>& experts_gate,
                                      bool transpose = false);

  int get_mapped_index(const std::string& name,
                       const std::unordered_map<std::string, int>& mapping);
};

}  // namespace layer
}  // namespace xllm
