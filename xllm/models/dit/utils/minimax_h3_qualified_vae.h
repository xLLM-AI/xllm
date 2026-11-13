/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <torch/types.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <vector>

namespace xllm {

inline constexpr int32_t kMiniMaxH3QualifiedWorldSize = 16;
inline constexpr int32_t kMiniMaxH3QualifiedSerialVaeSize = 1;
inline constexpr int32_t kMiniMaxH3QualifiedParallelVaeSize = 16;

// Owner (rank0) builds the full video; binding peak = decode() final
// cat(decoded_chunks, 2) = 3,507,259,392 B = 3.2664 GiB (HBM-B peak B; B-3
// in-place affine drops peak C to 2.8741 GiB so B binds). reserve = peak x 1.35
// + 256 MiB rounded up conservatively for unmodeled npu_fusion_attention
// workspace, HCCL staging, and allocator fragmentation.
inline constexpr int64_t kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes =
    int64_t{4864} * 1024 * 1024;  // 4.75 GiB

// Nonowner releases `gathered` right after the symmetric allgather and never
// builds rows/clips/full video; binding peak = local_stack + gathered + latent
// = 763,004,928 B = 0.7106 GiB. reserve = peak x 1.35 + 256 MiB rounded up
// conservatively for HCCL allgather staging (672 MiB x 16 ranks).
inline constexpr int64_t kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes =
    int64_t{1536} * 1024 * 1024;  // 1.5 GiB

inline int64_t minimax_h3_parallel_vae_decode_reserve_bytes(bool decode_owner) {
  return decode_owner ? kMiniMaxH3ParallelVaeDecodeOwnerReserveBytes
                      : kMiniMaxH3ParallelVaeDecodeNonownerReserveBytes;
}

// Role-aware decode-entry admission (architect review of the HBM-C C-1
// re-check): only a nonowner hard-fails when aclrt raw free drops below its
// reserve. Owner raw free UNDERSTATES usable memory: the formal CO rank0
// sampled peak 62383/65536 MiB leaves ~3.1 GiB raw free, below the 4.75 GiB
// analytic owner reserve, yet the serial owner path survives the 7.1654 GiB
// out-of-place affine peak because raw free excludes torch allocator pool
// reuse. B-3 strictly lowers owner demand (peak C 7.1654 -> 2.8741 GiB;
// binding peak B = 3.2664 GiB), so an owner hard gate would false-reject a
// proven-survivable config; the owner logs available vs reserve instead.
// Falsification: if the owner OOMs on NPU despite B-3, this log-only owner
// admission is invalid and must be replaced using measured allocator data.
inline bool minimax_h3_parallel_vae_decode_entry_hard_fails(
    bool decode_owner,
    int64_t available_bytes,
    int64_t decode_reserve_bytes) {
  return !decode_owner && available_bytes < decode_reserve_bytes;
}

// In-place pixel affine for the qualified decode output (HBM-B finding B-3).
// `decoded` is a consumed sink: the FP32 cast aliases when already FP32 (the
// qualified weights are cast to FP32 at load) or replaces a local with no
// other references, so mul_/add_/clamp_ are bit-identical to the out-of-place
// chain clamp(decoded.to(f32) * std + mean, 0, 1) while removing three
// full-video temporaries. permute({0,2,1,3,4}) + contiguous is unchanged. The
// caller must not read `decoded` after this call.
inline torch::Tensor minimax_h3_vae_affine_to_pixels(
    torch::Tensor decoded,
    const torch::Tensor& pixel_mean,
    const torch::Tensor& pixel_std) {
  decoded = decoded.to(torch::kFloat32);
  decoded.mul_(pixel_std.to(decoded.device()))
      .add_(pixel_mean.to(decoded.device()))
      .clamp_(0.0, 1.0);
  return decoded.permute({0, 2, 1, 3, 4}).contiguous();
}

struct MiniMaxH3QualifiedVaePlan {
  bool reference_encode_owner = false;
  bool video_vae_worker = false;
  bool collaborative_video_decode = false;
  bool audio_decode_owner = false;
  bool output_owner = false;
};

inline bool minimax_h3_qualified_vae_size_is_valid(int32_t world_size,
                                                   int32_t vae_size) {
  return world_size == kMiniMaxH3QualifiedWorldSize &&
         (vae_size == kMiniMaxH3QualifiedSerialVaeSize ||
          vae_size == kMiniMaxH3QualifiedParallelVaeSize);
}

inline MiniMaxH3QualifiedVaePlan minimax_h3_qualified_vae_plan(
    int32_t world_size,
    int32_t rank,
    int32_t vae_size) {
  if (!minimax_h3_qualified_vae_size_is_valid(world_size, vae_size) ||
      rank < 0 || rank >= world_size) {
    throw std::invalid_argument(
        "MiniMax-H3 qualified VAE requires world16 with vae_size 1 or 16");
  }
  const bool owner = rank == 0;
  const bool collaborative = vae_size == kMiniMaxH3QualifiedParallelVaeSize;
  return {/*reference_encode_owner=*/owner,
          /*video_vae_worker=*/owner || collaborative,
          /*collaborative_video_decode=*/collaborative,
          /*audio_decode_owner=*/owner,
          /*output_owner=*/owner};
}

inline std::vector<size_t> minimax_h3_vae_local_tile_indices(size_t tile_count,
                                                             size_t world_size,
                                                             size_t rank) {
  if (world_size == 0 || rank >= world_size) {
    throw std::invalid_argument("MiniMax-H3 VAE tile rank is invalid");
  }
  std::vector<size_t> indices;
  indices.reserve((tile_count + world_size - 1) / world_size);
  for (size_t tile = rank; tile < tile_count; tile += world_size) {
    indices.emplace_back(tile);
  }
  return indices;
}

struct MiniMaxH3VaeGatherCoordinate {
  size_t rank = 0;
  size_t local_slot = 0;
};

inline MiniMaxH3VaeGatherCoordinate minimax_h3_vae_gather_coordinate(
    size_t tile_index,
    size_t world_size) {
  if (world_size == 0) {
    throw std::invalid_argument("MiniMax-H3 VAE gather world size is zero");
  }
  return {tile_index % world_size, tile_index / world_size};
}

enum class MiniMaxH3VaeCollectivePhase {
  LOCAL_PREPARATION,
  AGREEMENT,
  ALLGATHER,
  REORDER,
  COMPLETE,
  FAILED,
};

class MiniMaxH3VaeCollectiveState final {
 public:
  MiniMaxH3VaeCollectivePhase phase() const { return phase_; }

  void local_preparation_finished(bool local_succeeded) {
    require_phase(MiniMaxH3VaeCollectivePhase::LOCAL_PREPARATION);
    local_succeeded_ = local_succeeded;
    phase_ = MiniMaxH3VaeCollectivePhase::AGREEMENT;
  }

  void agreement_finished(bool all_succeeded) {
    require_phase(MiniMaxH3VaeCollectivePhase::AGREEMENT);
    phase_ = local_succeeded_ && all_succeeded
                 ? MiniMaxH3VaeCollectivePhase::ALLGATHER
                 : MiniMaxH3VaeCollectivePhase::FAILED;
  }

  void allgather_finished() {
    require_phase(MiniMaxH3VaeCollectivePhase::ALLGATHER);
    phase_ = MiniMaxH3VaeCollectivePhase::REORDER;
  }

  void reorder_finished() {
    require_phase(MiniMaxH3VaeCollectivePhase::REORDER);
    phase_ = MiniMaxH3VaeCollectivePhase::COMPLETE;
  }

 private:
  void require_phase(MiniMaxH3VaeCollectivePhase expected) const {
    if (phase_ != expected) {
      throw std::logic_error("MiniMax-H3 VAE collective phase is out of order");
    }
  }

  MiniMaxH3VaeCollectivePhase phase_ =
      MiniMaxH3VaeCollectivePhase::LOCAL_PREPARATION;
  bool local_succeeded_ = false;
};

inline bool minimax_h3_parallel_vae_hbm_preflight_passes(
    int64_t available_bytes,
    int64_t resident_weight_bytes,
    int64_t decode_reserve_bytes) {
  if (available_bytes < 0 || resident_weight_bytes <= 0 ||
      decode_reserve_bytes <= 0 ||
      resident_weight_bytes >
          std::numeric_limits<int64_t>::max() - decode_reserve_bytes) {
    return false;
  }
  return available_bytes >= resident_weight_bytes + decode_reserve_bytes;
}

}  // namespace xllm
