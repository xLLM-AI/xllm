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

#include <string>

#include "framework/parallel_state/parallel_args.h"

namespace xllm {

enum class RowParallelReduceMode : int8_t {
  NONE = 0,
  ALL_REDUCE = 1,
  REDUCE_SCATTER = 2,
  MATMUL_REDUCE_SCATTER = 3,
};

struct FlashComm1Context {
  bool enabled = false;
  int32_t tp_rank = 0;
  int32_t tp_world_size = 1;
  int32_t original_num_tokens = 0;
  int32_t padded_num_tokens = 0;
  int32_t padded_local_num_tokens = 0;
  int32_t pad_size = 0;
  bool enable_mmrs_fusion = false;
  std::string mmrs_comm_mode = "aiv";
  ProcessGroup* tp_group = nullptr;
};

struct FlashComm1Options {
  bool enable_flashcomm1 = false;
  int32_t min_prefill_tokens = 8192;
  bool enable_mmrs_fusion = false;
  std::string mmrs_comm_mode = "aiv";
};

// Token geometry FC1 needs when it runs inside an outer CP sharding.
//
// FC1 and model-side CP both shard the token axis, but over different groups
// (tp_group vs cp_group) which are orthogonal in the rank layout
// dp_rank * (cp_size * tp_size) + cp_rank * tp_size + tp_rank. They compose as
// long as CP is the OUTER shard and FC1 the inner one: CP splits the batch
// across cp_group first, then FC1 splits each CP rank's rows across tp_group.
// That requires separating the two token counts FC1 used to conflate:
//
//   global_num_tokens - the DP-local batch before CP shards it. Identical on
//       every rank of the CP group, so it is what the min_prefill_tokens
//       threshold must be tested against to keep the on/off decision uniform.
//   local_num_tokens  - this rank's row count after the CP shard. This is the
//       sequence FC1 actually pads and splits, so it drives all geometry
//       (padded_num_tokens, padded_local_num_tokens, pad_size) and becomes
//       ctx.original_num_tokens.
//   min_local_num_tokens - the smallest local_num_tokens across the CP group.
//       V4's CP split is per-sequence and contiguous, so segments are uneven
//       and short batches leave high cp_ranks with few or zero rows. Every rank
//       can derive this from the global q_seq_lens without communicating, so
//       gating on it keeps FC1 uniformly on or off across the whole CP group
//       instead of leaving some ranks sharded and others not.
//
// With cp_size == 1 all three are the same value and the geometry collapses to
// the pre-CP behaviour.
struct FlashComm1TokenGeometry {
  int32_t global_num_tokens = 0;
  int32_t local_num_tokens = 0;
  int32_t min_local_num_tokens = 0;

  static FlashComm1TokenGeometry without_cp(int32_t num_tokens) {
    return FlashComm1TokenGeometry{num_tokens, num_tokens, num_tokens};
  }
};

class FlashComm1ContextScope {
 public:
  explicit FlashComm1ContextScope(const FlashComm1Context* ctx);
  ~FlashComm1ContextScope();

  FlashComm1ContextScope(const FlashComm1ContextScope&) = delete;
  FlashComm1ContextScope& operator=(const FlashComm1ContextScope&) = delete;

 private:
  const FlashComm1Context* previous_;
};

const FlashComm1Context* get_current_flash_comm1_context();

bool is_sequence_sharded(const FlashComm1Context& ctx);

torch::Tensor pad_rows_by_copy(const torch::Tensor& input, int64_t padded_rows);

// Topology/config gate for FC1, independent of the process group and platform.
// FC1 shards the sequence over the TP group, so it only needs a consistent
// token count within that group: DP is fine (each DP rank owns a whole batch),
// and CP is fine too because every rank of a TP group shares one cp_rank and
// therefore one local row count.
//
// This CP-aware overload is for callers that apply the CP shard BEFORE building
// the FC1 context and can describe both token counts (see
// FlashComm1TokenGeometry). Composition is rejected unless the smallest CP
// segment still leaves one full alignment unit per TP rank, so no rank ends up
// reduce-scattering an empty or degenerate shard.
bool is_flash_comm1_eligible(const FlashComm1TokenGeometry& geometry,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options);

// Legacy gate for callers that have NOT been adapted to shard CP first. Such a
// caller would hand FC1 the pre-CP token count and shard the sequence twice, so
// this overload keeps refusing to compose with CP.
bool is_flash_comm1_eligible(int32_t num_tokens,
                             bool is_prefill,
                             const ParallelArgs& parallel_args,
                             const FlashComm1Options& options);

FlashComm1Context build_flash_comm1_context(
    const FlashComm1TokenGeometry& geometry,
    bool is_prefill,
    const ParallelArgs& parallel_args,
    const FlashComm1Options& options);

FlashComm1Context build_flash_comm1_context(int32_t num_tokens,
                                            bool is_prefill,
                                            const ParallelArgs& parallel_args,
                                            const FlashComm1Options& options);

torch::Tensor shard_sequence(const torch::Tensor& input,
                             const FlashComm1Context& ctx);

torch::Tensor gather_sequence(const torch::Tensor& input,
                              const FlashComm1Context& ctx);

torch::Tensor maybe_pad_and_reduce(torch::Tensor input,
                                   const FlashComm1Context& ctx,
                                   RowParallelReduceMode mode);

RowParallelReduceMode row_parallel_reduce_mode_for_fc1(
    const FlashComm1Context& ctx);

torch::Tensor maybe_shard_residual(const torch::Tensor& residual,
                                   const FlashComm1Context& ctx);

}  // namespace xllm
