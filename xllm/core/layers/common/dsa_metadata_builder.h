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

#include <vector>

namespace xllm {
struct ModelInputParams;
struct DSACacheInfo;
struct DSAGroupInfo;

namespace layer {

struct AttentionMetadata;
struct DSAMetadata;
struct DSACPMetadata;

// Builder class for DSAMetadata.
// Builds a complete AttentionMetadata (with dsa_metadata populated) from
// ModelInputParams and model-specific data.  This replaces the need for a
// separate AttentionMetadataBuilder::build() call in DeepSeek V4.
class DSAMetadataBuilder {
 public:
  // Build a complete AttentionMetadata with DSAMetadata populated inside.
  // Internally constructs the base AttentionMetadata fields (q_cu_seq_lens,
  // block_table, slot_mapping, etc.) and the DSA-specific fields (RoPE,
  // block tables, slot mappings, sequence lengths, compressed positions).
  //
  //   params: batch-level model input params
  //   positions: token position IDs tensor
  //   dsa_cos_sin: precomputed RoPE cos/sin table (optional, can be undefined)
  //   caches_info: per-layer cache specs [layer_id][cache_idx]
  //   group_infos: per-group info [group_id]
  static AttentionMetadata build(
      const ModelInputParams& params,
      const torch::Tensor& positions,
      const torch::Tensor& dsa_cos_sin,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
      const torch::Tensor& dsa_c4_cos_sin = torch::Tensor(),
      const torch::Tensor& dsa_c128_cos_sin = torch::Tensor());

  // Build per-rank DSA-CP local token metadata (M0 of the DSA-CP plan).
  //
  // Splits the flattened query-token stream evenly across cp_size ranks and
  // returns the CURRENT rank's (cp_rank) local view. This mirrors vllm-ascend
  // _build_local_token_metadata and is intentionally a pure function over
  // plain tensors so it can be unit-tested on CPU without a full
  // ModelInputParams / RoPE pipeline.
  //
  //   query_start_loc : (num_reqs+1,) int32/int64, global cumulative query
  //                     starts with a leading 0 (i.e.
  //                     DSAMetadata::actual_seq_lengths_query).
  //                     query_start_loc.back() == num_input_tokens.
  //   seq_lens        : (num_reqs,) int32/int64, per-request KV context
  //                     length (DSAMetadata::actual_seq_lengths_kv).
  //   cp_size         : CP group width (>= 1).
  //   cp_rank         : this rank's index within the CP group ([0, cp_size)).
  //   full_cos/full_sin : optional RoPE tables over the PADDED global token
  //                     stream (length == num_tokens_pad). When defined, the
  //                     result's local_cos/local_sin are sliced to
  //                     [local_start, local_end). Pass undefined tensors to
  //                     skip (e.g. in pure metadata unit tests).
  //
  // When cp_size == 1 the returned metadata is the identity view (local ==
  // global) so callers can use a single code path.
  static DSACPMetadata build_cp_local_metadata(
      const torch::Tensor& query_start_loc,
      const torch::Tensor& seq_lens,
      int32_t cp_size,
      int32_t cp_rank,
      const torch::Tensor& full_cos = torch::Tensor(),
      const torch::Tensor& full_sin = torch::Tensor());

 private:
  // Build DSA-specific fields into dsa_metadata.
  static void build_dsa_fields(
      const ModelInputParams& params,
      const torch::Tensor& positions,
      const torch::Tensor& dsa_cos_sin,
      const torch::Tensor& dsa_c4_cos_sin,
      const torch::Tensor& dsa_c128_cos_sin,
      const std::vector<std::vector<DSACacheInfo>>& caches_info,
      const std::vector<DSAGroupInfo>& group_infos,
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

  // Step 2: per-group processing.
  static void process_group(const torch::Tensor& raw_bt,
                            const DSAGroupInfo& gi,
                            const std::vector<int32_t>& ctx_lens,
                            const std::vector<int32_t>& q_lens,
                            int32_t batch_size,
                            int64_t total_tokens,
                            int64_t graph_slot_capacity,
                            int32_t block_table_capacity_cols,
                            torch::Tensor& out_bt,
                            torch::Tensor& out_slots);

  static void process_token_group(const torch::Tensor& raw_bt,
                                  int32_t ratio,
                                  int32_t block_size,
                                  const std::vector<int32_t>& ctx_lens,
                                  const std::vector<int32_t>& q_lens,
                                  int32_t batch_size,
                                  int64_t total_tokens,
                                  int64_t graph_slot_capacity,
                                  int32_t block_table_capacity_cols,
                                  torch::Tensor& out_bt,
                                  torch::Tensor& out_slots);

  static void process_swa_group(const torch::Tensor& raw_bt,
                                int32_t block_size,
                                const std::vector<int32_t>& ctx_lens,
                                const std::vector<int32_t>& q_lens,
                                int32_t batch_size,
                                int64_t graph_slot_capacity,
                                int32_t block_table_capacity_cols,
                                torch::Tensor& out_bt,
                                torch::Tensor& out_slots);

  // Build actual_seq_lengths_kv and actual_seq_lengths_query.
  static void build_seq_lengths(const ModelInputParams& params,
                                const torch::Device& target_device,
                                int32_t batch_size,
                                DSAMetadata& dsa_metadata);

  // Build input_positions, c4_pad_positions, c128_pad_positions.
  static void build_positions(const ModelInputParams& params,
                              int32_t batch_size,
                              DSAMetadata& dsa_metadata);
};

}  // namespace layer
}  // namespace xllm
