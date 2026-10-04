/* Copyright 2026 The xLLM Authors.

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

#include "kernels/mlu/chunk_kda.h"

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <vector>

#include "triton_jit/include/jit_kernel.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int64_t kHeadDim = 128;
constexpr int64_t kBlockC = 16;
constexpr int64_t kWorkspaceGroupChunks = 128;
constexpr int64_t kWorkspaceLimitBytes = 2LL * 1024 * 1024 * 1024;
constexpr char kKernelPath[] =
    "xllm.core.kernels.mlu.triton_kernel.chunk_kda_fwd";

uint32_t grid_size(int64_t job_count, int64_t core_count) {
  CHECK_GT(job_count, 0);
  CHECK_GT(core_count, 0);
  return static_cast<uint32_t>(std::min(job_count, core_count));
}

}  // namespace

int64_t kda_prefill_chunk_size() {
  static const int64_t chunk_size = [] {
    const char* env = std::getenv("XLLM_MLU_KDA_CHUNK_SIZE");
    if (env == nullptr) {
      return ChunkKDAImpl::kDefaultChunkSize;
    }
    char* parse_end = nullptr;
    const int64_t parsed =
        static_cast<int64_t>(std::strtoll(env, &parse_end, 10));
    CHECK(parse_end != env && *parse_end == '\0')
        << "XLLM_MLU_KDA_CHUNK_SIZE must be 16 or 64";
    CHECK(parsed == 16 || parsed == 64)
        << "XLLM_MLU_KDA_CHUNK_SIZE must be 16 or 64";
    return parsed;
  }();
  return chunk_size;
}

using xllm::triton_jit::JITKernel;

namespace {

class ChunkKDAWorkspace final {
 public:
  ChunkKDAWorkspace(int64_t slots,
                    int64_t heads,
                    int64_t chunk_size,
                    const torch::TensorOptions& options) {
    const torch::TensorOptions fp32_options = options.dtype(torch::kFloat32);
    lower_inverse_ =
        torch::empty({slots, heads, chunk_size, chunk_size}, fp32_options);
    aq_ = torch::empty_like(lower_inverse_);
    const std::vector<int64_t> token_shape = {
        slots, heads, chunk_size, kHeadDim};
    w_ = torch::empty(token_shape, fp32_options);
    // State multiplies value-major U with the transposed chunk inverse.
    u_ = torch::empty({slots, heads, kHeadDim, chunk_size}, fp32_options);
    qg_ = torch::empty(token_shape, fp32_options);
    kg_ = torch::empty(token_shape, fp32_options);
    normalized_q_ = torch::empty(token_shape, fp32_options);
    normalized_k_ = torch::empty(token_shape, fp32_options);
    cumulative_gate_ = torch::empty(token_shape, fp32_options);
  }

  torch::Tensor lower_inverse_;
  torch::Tensor aq_;
  torch::Tensor w_;
  torch::Tensor u_;
  torch::Tensor qg_;
  torch::Tensor kg_;
  torch::Tensor normalized_q_;
  torch::Tensor normalized_k_;
  torch::Tensor cumulative_gate_;
};

void launch_group(const torch::Tensor& q,
                  const torch::Tensor& k,
                  const torch::Tensor& v,
                  const torch::Tensor& log_gate,
                  const torch::Tensor& beta,
                  const torch::Tensor& input_state,
                  const torch::Tensor& final_state,
                  const torch::Tensor& output,
                  const torch::Tensor& cu_seqlens,
                  const torch::Tensor& chunk_indices,
                  const ChunkKDAWorkspace& workspace,
                  int64_t chunk_base,
                  int64_t slots,
                  int64_t num_sequences,
                  int64_t num_heads,
                  int64_t chunk_size,
                  bool use_qk_l2norm,
                  int64_t core_count) {
  cnrtQueue_t queue = torch_mlu::getCurMLUStream();
  const int64_t chunk_head_jobs = slots * num_heads;
  JITKernel::get(kKernelPath, "tmo_chunk_kda_gate_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{grid_size(chunk_head_jobs, core_count), 1, 1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              v,
              beta,
              workspace.w_,
              workspace.u_,
              workspace.qg_,
              workspace.kg_,
              log_gate,
              q,
              k,
              workspace.normalized_q_,
              workspace.normalized_k_,
              workspace.cumulative_gate_,
              cu_seqlens,
              chunk_indices,
              chunk_base,
              slots,
              /*H=*/static_cast<int32_t>(num_heads),
              /*BT=*/static_cast<int32_t>(chunk_size),
              /*D=*/static_cast<int32_t>(kHeadDim),
              /*BK=*/static_cast<int32_t>(kHeadDim),
              /*USE_QK_L2NORM=*/use_qk_l2norm ? 1 : 0);

  JITKernel::get(kKernelPath, "tmo_chunk_kda_kkt_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{grid_size(chunk_head_jobs, core_count), 1, 1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/5},
              workspace.normalized_q_,
              workspace.normalized_k_,
              workspace.cumulative_gate_,
              beta,
              workspace.lower_inverse_,
              workspace.aq_,
              cu_seqlens,
              chunk_indices,
              chunk_base,
              slots,
              /*H=*/static_cast<int32_t>(num_heads),
              /*BT=*/static_cast<int32_t>(chunk_size),
              /*D=*/static_cast<int32_t>(kHeadDim),
              /*BC=*/static_cast<int32_t>(kBlockC),
              /*BN=*/static_cast<int32_t>(chunk_size),
              /*BK=*/static_cast<int32_t>(kHeadDim));

  JITKernel::get(kKernelPath, "tmo_chunk_kda_inverse_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{grid_size(chunk_head_jobs, core_count), 1, 1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              workspace.lower_inverse_,
              slots,
              /*H=*/static_cast<int32_t>(num_heads),
              /*BT=*/static_cast<int32_t>(chunk_size),
              /*B0=*/static_cast<int32_t>(kBlockC));

  constexpr int64_t kValueBlock = 32;
  const int64_t state_jobs =
      num_sequences * num_heads * (kHeadDim / kValueBlock);
  JITKernel::get(kKernelPath, "tmo_chunk_kda_state_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{grid_size(state_jobs, core_count), 1, 1},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/99},
              workspace.lower_inverse_,
              workspace.w_,
              workspace.u_,
              workspace.qg_,
              workspace.kg_,
              workspace.aq_,
              workspace.cumulative_gate_,
              input_state,
              final_state,
              output,
              cu_seqlens,
              chunk_indices,
              chunk_base,
              slots,
              num_sequences,
              /*H=*/static_cast<int32_t>(num_heads),
              /*BT=*/static_cast<int32_t>(chunk_size),
              /*D=*/static_cast<int32_t>(kHeadDim),
              /*BK=*/static_cast<int32_t>(kHeadDim),
              /*BV=*/static_cast<int32_t>(kValueBlock));
}

std::tuple<torch::Tensor, torch::Tensor> forward_chunks(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& log_gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& chunk_indices,
    bool output_final_state,
    int64_t num_heads,
    int64_t chunk_size,
    bool use_qk_l2norm,
    int64_t core_count) {
  const int64_t total_chunks = chunk_indices.size(0);
  // Seven token tiles and two chunk matrices.
  // Budget padded chunks; packed token count excludes tail padding.
  const int64_t elements_per_chunk_head =
      7 * chunk_size * kHeadDim + 2 * chunk_size * chunk_size;
  const int64_t bytes_per_chunk_head =
      elements_per_chunk_head * static_cast<int64_t>(sizeof(float));
  CHECK_LE(num_heads, kWorkspaceLimitBytes / bytes_per_chunk_head)
      << "A single KDA workspace chunk exceeds the 2 GiB budget";
  const int64_t budget_slots =
      kWorkspaceLimitBytes / (num_heads * bytes_per_chunk_head);
  const int64_t workspace_slots =
      total_chunks <= budget_slots
          ? total_chunks
          : std::min(kWorkspaceGroupChunks, budget_slots);

  ChunkKDAWorkspace workspace(
      workspace_slots, num_heads, chunk_size, q.options());
  torch::Tensor output = torch::empty_like(v);
  torch::Tensor final_state = torch::empty_like(initial_state);
  torch::Tensor state_source = initial_state;
  for (int64_t chunk_base = 0; chunk_base < total_chunks;
       chunk_base += workspace_slots) {
    const int64_t slots = std::min(workspace_slots, total_chunks - chunk_base);
    launch_group(q,
                 k,
                 v,
                 log_gate,
                 beta,
                 state_source,
                 final_state,
                 output,
                 cu_seqlens,
                 chunk_indices,
                 workspace,
                 chunk_base,
                 slots,
                 initial_state.size(0),
                 num_heads,
                 chunk_size,
                 use_qk_l2norm,
                 core_count);
    state_source = final_state;
  }
  return {output, output_final_state ? final_state : torch::Tensor()};
}

}  // namespace

ChunkKDAImpl::ChunkKDAImpl(int64_t num_heads) : num_heads_(num_heads) {
  CHECK_GT(num_heads_, 0) << "Chunk KDA requires at least one head";
  torch_mlu::DeviceProp* properties =
      torch_mlu::getDeviceProperties(torch_mlu::current_device());
  CHECK(properties != nullptr);
  total_core_num_ =
      properties->cluster_count * properties->core_num_per_cluster;
}

std::tuple<torch::Tensor, torch::Tensor> ChunkKDAImpl::forward(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& log_gate,
    const torch::Tensor& beta,
    const torch::Tensor& initial_state,
    const torch::Tensor& cu_seqlens,
    const torch::Tensor& chunk_indices,
    bool output_final_state,
    bool use_qk_l2norm) {
  CHECK_EQ(q.dim(), 4) << "Chunk KDA q must be [1,T,H,K]";
  CHECK_EQ(k.sizes(), q.sizes()) << "Chunk KDA q/k shape mismatch";
  CHECK_EQ(v.dim(), 4) << "Chunk KDA v must be [1,T,H,V]";
  CHECK_EQ(q.size(0), 1) << "Chunk KDA only supports packed batch size 1";
  CHECK_EQ(v.size(0), 1) << "Chunk KDA v batch size mismatch";
  CHECK_EQ(v.size(1), q.size(1)) << "Chunk KDA v token count mismatch";
  CHECK_EQ(q.size(2), num_heads_) << "Chunk KDA q head count mismatch";
  CHECK_EQ(v.size(2), num_heads_) << "Chunk KDA v head count mismatch";
  CHECK_EQ(q.size(3), kHeadDim) << "Chunk KDA requires key dimension 128";
  CHECK_EQ(v.size(3), kHeadDim) << "Chunk KDA requires value dimension 128";
  CHECK_EQ(q.scalar_type(), torch::kBFloat16) << "Chunk KDA q must be bfloat16";
  CHECK_EQ(k.scalar_type(), torch::kBFloat16) << "Chunk KDA k must be bfloat16";
  CHECK_EQ(v.scalar_type(), torch::kBFloat16) << "Chunk KDA v must be bfloat16";
  CHECK_EQ(initial_state.scalar_type(), torch::kFloat32)
      << "Chunk KDA state must be float32";
  CHECK_EQ(initial_state.dim(), 4) << "Chunk KDA state must be [N,H,V,K]";
  CHECK_EQ(initial_state.size(0), cu_seqlens.size(0) - 1)
      << "Chunk KDA requires one initial state per sequence";
  CHECK_EQ(initial_state.size(1), num_heads_)
      << "Chunk KDA state head count mismatch";
  CHECK_EQ(initial_state.size(2), v.size(3))
      << "Chunk KDA state value dimension mismatch";
  CHECK_EQ(initial_state.size(3), q.size(3))
      << "Chunk KDA state key dimension mismatch";
  CHECK_EQ(cu_seqlens.dim(), 1) << "Chunk KDA cu_seqlens must be 1D";
  CHECK_EQ(chunk_indices.dim(), 2)
      << "Chunk KDA chunk_indices must be [chunks,2]";
  CHECK_EQ(chunk_indices.size(1), 2)
      << "Chunk KDA chunk_indices second dimension must be 2";
  CHECK_GT(q.size(1), 0) << "Chunk KDA requires at least one token";
  CHECK_GT(chunk_indices.size(0), 0) << "Chunk KDA requires at least one chunk";

  torch::Tensor gate =
      (log_gate.dim() == 3 ? log_gate.unsqueeze(/*dim=*/0) : log_gate)
          .contiguous()
          .to(torch::kFloat32);
  torch::Tensor activated_beta =
      (beta.dim() == 2 ? beta.unsqueeze(/*dim=*/0) : beta)
          .contiguous()
          .to(torch::kFloat32);
  CHECK_EQ(gate.sizes(), q.sizes()) << "Chunk KDA gate shape mismatch";
  CHECK_EQ(activated_beta.size(0), 1) << "Chunk KDA beta batch size mismatch";
  CHECK_EQ(activated_beta.size(1), q.size(1))
      << "Chunk KDA beta token count mismatch";
  CHECK_EQ(activated_beta.size(2), num_heads_)
      << "Chunk KDA beta head count mismatch";

  torch::Tensor q_work = q.contiguous();
  torch::Tensor k_work = k.contiguous();
  torch::Tensor v_work = v.contiguous();
  torch::Tensor cu_seqlens_work = cu_seqlens.contiguous().to(torch::kInt32);
  torch::Tensor chunk_indices_work =
      chunk_indices.contiguous().to(torch::kInt32);
  return forward_chunks(q_work,
                        k_work,
                        v_work,
                        gate,
                        activated_beta,
                        initial_state.contiguous(),
                        cu_seqlens_work,
                        chunk_indices_work,
                        output_final_state,
                        num_heads_,
                        kda_prefill_chunk_size(),
                        use_qk_l2norm,
                        total_core_num_);
}

}  // namespace xllm::kernel::mlu
