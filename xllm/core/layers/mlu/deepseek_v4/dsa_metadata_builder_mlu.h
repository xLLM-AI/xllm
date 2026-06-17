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

#include <cstdint>
#include <vector>

namespace xllm {
struct ModelInputParams;
struct DSACacheInfo;
struct DSAGroupInfo;

namespace layer {

struct AttentionMetadata;
struct DSAMetadata;

// MLU DeepSeek V4 builder for DSAMetadata.
class DSAMetadataBuilderMlu final {
 public:
  static AttentionMetadata build(
      const ModelInputParams& params,
      const torch::Tensor& positions,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
      int64_t window_size);

 private:
  // Build DSA-specific fields into dsa_metadata.
  static void build_dsa_fields(
      const ModelInputParams& params,
      const AttentionMetadata& attn_metadata,
      const torch::Tensor& positions,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
      int64_t window_size,
      DSAMetadata& dsa_metadata);

  // Step 1: expand block_table to slot array for one manager.
  static torch::Tensor expand_blocks_to_slots(
      const torch::Tensor& block_table,
      const DSAGroupInfo& gi,
      const std::vector<int32_t>& ctx_lens,
      int32_t batch_size,
      int64_t total_tokens);

  // Compute how many slots a single seq needs for this group.
  static int64_t compute_slot_num(const DSAGroupInfo& gi, int64_t token_len);

  // Per-group processing.
  static void process_group(const torch::Tensor& raw_bt,
                            const DSAGroupInfo& gi,
                            const std::vector<int32_t>& ctx_lens,
                            const std::vector<int32_t>& q_lens,
                            int32_t batch_size,
                            int64_t total_tokens,
                            torch::Tensor& out_bt,
                            torch::Tensor& out_slots);

  static void process_token_group(const torch::Tensor& raw_bt,
                                  int32_t ratio,
                                  int32_t block_size,
                                  const std::vector<int32_t>& ctx_lens,
                                  const std::vector<int32_t>& q_lens,
                                  int32_t batch_size,
                                  int64_t total_tokens,
                                  torch::Tensor& out_bt,
                                  torch::Tensor& out_slots);

  static void process_swa_group(const torch::Tensor& raw_bt,
                                int32_t block_size,
                                const std::vector<int32_t>& ctx_lens,
                                const std::vector<int32_t>& q_lens,
                                int32_t batch_size,
                                torch::Tensor& out_bt,
                                torch::Tensor& out_slots);

  static void build_c128_meta(DSAMetadata& dsa_metadata,
                              const std::vector<torch::Tensor>& proc_bt,
                              const std::vector<DSAGroupInfo>& group_infos,
                              int32_t batch_size);

  // Build sequence length tensors and host-side MLU batch metadata.
  static void build_seq_lengths(const AttentionMetadata& attn_metadata,
                                int32_t batch_size,
                                DSAMetadata& dsa_metadata,
                                std::vector<int32_t>& q_lens_vec,
                                std::vector<int32_t>& kv_lens_vec);

  static void build_swa_plan(DSAMetadata& dsa_metadata,
                             const std::vector<int32_t>& q_lens_vec,
                             int64_t window_size);

  // Build input_positions, c4_pad_positions, c128_pad_positions.
  static void build_positions(const ModelInputParams& params,
                              int32_t batch_size,
                              DSAMetadata& dsa_metadata);
};
}  // namespace layer
}  // namespace xllm
