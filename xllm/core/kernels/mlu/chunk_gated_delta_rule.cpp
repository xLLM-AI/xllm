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

#include "chunk_gated_delta_rule.h"

#include <cnrt.h>
#include <framework/core/MLUStream.h>
#include <framework/core/device.h>

#include <cmath>
#include <optional>
#include <unordered_map>

#include "kernels/mlu/mlu_ops_api.h"
#include "triton_jit/include/jit_kernel.h"

namespace xllm {
namespace kernel {
namespace mlu {

namespace {
std::optional<torch::Tensor> opt_tensor(const torch::Tensor& t) {
  return t.numel() > 0 ? std::optional<torch::Tensor>(t) : std::nullopt;
}
}  // namespace

ChunkGatedDeltaRuleImpl::ChunkGatedDeltaRuleImpl(int64_t num_k_heads,
                                                 int64_t num_v_heads)
    : total_core_num_(0), num_k_heads_(num_k_heads), num_v_heads_(num_v_heads) {
  auto device = torch_mlu::current_device();
  auto device_prop = torch_mlu::getDeviceProperties(device);
  total_core_num_ =
      device_prop->cluster_count * device_prop->core_num_per_cluster;
}

std::tuple<torch::Tensor, torch::Tensor> ChunkGatedDeltaRuleImpl::forward(
    torch::Tensor& q,
    torch::Tensor& k,
    torch::Tensor& v,
    torch::Tensor& g,
    torch::Tensor& beta,
    torch::Tensor& initial_state,
    torch::Tensor& cu_seqlens,
    torch::Tensor& chunk_indices,
    bool output_final_state,
    bool use_qk_l2norm_in_kernel) {
  // All input tensors should be contiguous
  q = q.contiguous();
  k = k.contiguous();
  v = v.contiguous();
  g = g.contiguous().to(torch::kFloat32);
  beta = beta.contiguous();
  initial_state = initial_state.contiguous();
  cu_seqlens = cu_seqlens.contiguous().to(torch::kInt32);

  if (use_qk_l2norm_in_kernel) {
    q = l2norm(q, -1);
    k = l2norm(k, -1);
  }
  float scale = 1.0f / std::sqrt(static_cast<float>(k.size(-1)));
  constexpr int64_t chunk_size = kDefaultChunkSize;

  // Step 1: Chunk preparation
  torch::Tensor g_cumsum = chunk_local_cumsum(
      g, chunk_size, false, cu_seqlens, false, chunk_indices);

  // Step 2: Permute g for downstream operations: [B, T, H] -> [H, B, T]
  torch::Tensor g_trans = g_cumsum.permute({2, 0, 1}).contiguous();

  // Step 3: Compute attention matrix A
  torch::Tensor A = chunk_scaled_dot_kkt_fwd(
      k, beta, g_cumsum, cu_seqlens, chunk_size, chunk_indices);

  // Step 4: Solve triangular system
  torch::Tensor A_tril = solve_tril(A, cu_seqlens, torch::kBFloat16);

  // Step 5: Recompute w and u
  auto [w, u] =
      recompute_w_fwd(k, v, beta, g_trans, A_tril, cu_seqlens, chunk_indices);

  // Step 6: Compute h and v_new (and final_state if needed)
  auto [h, v_new, final_state] =
      chunk_gated_delta_rule_fwd_h(k,
                                   w,
                                   u,
                                   g_trans,
                                   std::nullopt,
                                   initial_state,
                                   output_final_state,
                                   chunk_size,
                                   true,
                                   cu_seqlens,
                                   chunk_indices);

  // Step 7: Compute final output
  torch::Tensor o = chunk_fwd_o(
      q, k, v_new, h, g_trans, scale, cu_seqlens, chunk_size, chunk_indices);

  return std::make_tuple(o, final_state);
}

// ============================================================================
// Helper Methods
// ============================================================================

cnrtDim3_t ChunkGatedDeltaRuleImpl::compute_grid_dim(
    int64_t total_chunks) const {
  int64_t grid_num = std::min(total_core_num_, total_chunks);
  return cnrtDim3_t{static_cast<uint32_t>(grid_num), 1, 1};
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_lens(
    const torch::Tensor& cu_seqlens) const {
  return cu_seqlens.narrow(0, 1, cu_seqlens.size(0) - 1) -
         cu_seqlens.narrow(0, 0, cu_seqlens.size(0) - 1);
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_chunk_offsets(
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size) const {
  torch::Tensor lengths = prepare_lens(cu_seqlens);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor zero = cu_seqlens.new_zeros({1});
  torch::Tensor combined = torch::cat({zero, num_chunks}, 0);
  return combined.cumsum(-1).to(torch::kInt32);
}

torch::Tensor ChunkGatedDeltaRuleImpl::prepare_chunk_indices(
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size) const {
  torch::Tensor lengths = prepare_lens(cu_seqlens);
  torch::Tensor num_chunks = (lengths + chunk_size - 1) / chunk_size;
  num_chunks = num_chunks.to(torch::kLong);
  torch::Tensor cumsum = torch::cumsum(num_chunks, 0);
  int64_t total = cumsum[-1].item<int64_t>();
  torch::Tensor arange_total = torch::arange(total, cu_seqlens.options());

  torch::Tensor zeros = torch::zeros({1}, cumsum.options());
  torch::Tensor prefix = torch::cat({zeros, cumsum.slice(0, 0, -1)});
  torch::Tensor repeats_prefix = torch::repeat_interleave(prefix, num_chunks);

  torch::Tensor indices = arange_total - repeats_prefix;
  torch::Tensor mask = indices == 0;
  torch::Tensor col0 = mask.cumsum(0) - 1;
  auto out = torch::stack({col0, indices}, 1).to(cu_seqlens);
  return out.to(torch::kInt32);
}

// ============================================================================
// Core Computation Methods
// ============================================================================

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_local_cumsum(
    const torch::Tensor& g,
    int64_t chunk_size,
    bool reverse,
    const torch::Tensor& cu_seqlens,
    bool head_first,
    const torch::Tensor& chunk_indices) {
  int64_t B = g.size(0);
  int64_t T = head_first ? g.size(2) : g.size(1);
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, chunk_size);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * NT;

  auto output =
      torch::empty_like(g, torch::dtype(torch::kFloat32).device(g.device()));
  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  // cumsum.py: *fp32, *fp32, *i32, *i32, i32, i32, [H], 64, i32, i32, 0, 1, 0
  triton_jit::JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.cumsum",
      /*fn_name=*/"tmo_chunk_local_cumsum_scalar_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{dim.x, dim.y, dim.z},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/1},
              g,
              output,
              cu_seqlens,
              chunk_indices,
              static_cast<int32_t>(T),
              static_cast<int32_t>(B),
              static_cast<int32_t>(num_v_heads_),
              /*BT=*/64,
              static_cast<int32_t>(NT),
              static_cast<int32_t>(chunk_num),
              /*flags=*/0,
              1,
              0);
  return output;
}

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_scaled_dot_kkt_fwd(
    const torch::Tensor& k,
    const torch::Tensor& beta,
    const torch::Tensor& g,
    const torch::Tensor& cu_seqlens,
    int64_t chunk_size,
    const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t K = k.size(3);
  int64_t H = beta.size(-1);
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, chunk_size);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * NT;

  auto A = torch::empty({B, T, H, chunk_size},
                        torch::dtype(torch::kFloat32).device(k.device()));
  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  triton_jit::JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.chunk_scaled_dot_kkt",
      /*fn_name=*/"tmo_chunk_scaled_dot_kkt_fwd_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{dim.x, dim.y, dim.z},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              k,
              beta,
              g,
              A,
              cu_seqlens,
              chunk_indices,
              static_cast<int32_t>(T),
              static_cast<int32_t>(B),
              static_cast<int32_t>(num_v_heads_),
              static_cast<int32_t>(num_k_heads_),
              static_cast<int32_t>(K),
              /*BT=*/64,
              /*BK=*/128,
              static_cast<int32_t>(NT),
              static_cast<int32_t>(chunk_num),
              /*ALLOW_TF32=*/1,
              /*IS_VARLEN=*/cu_seqlens.numel() > 0 ? 1 : 0,
              /*USE_G=*/1);
  return A;
}

torch::Tensor ChunkGatedDeltaRuleImpl::solve_tril(
    const torch::Tensor& A,
    const std::optional<torch::Tensor>& cu_seqlens,
    const std::optional<torch::ScalarType>& output_dtype) {
  torch::ScalarType dtype =
      output_dtype.has_value() ? output_dtype.value() : A.scalar_type();

  torch::Tensor A_float =
      (A.scalar_type() != at::kFloat) ? A.to(at::kFloat) : A;

  torch::Tensor output = torch::empty(
      A_float.sizes(),
      torch::TensorOptions().dtype(dtype).device(A_float.device()));

  if (cu_seqlens.has_value()) {
    tmo::torch_api::solve_tril(
        A_float, output, cu_seqlens.value().to(torch::kInt32));
  } else {
    tmo::torch_api::solve_tril(A_float, output, std::nullopt);
  }

  return output;
}

std::pair<torch::Tensor, torch::Tensor>
ChunkGatedDeltaRuleImpl::recompute_w_fwd(const torch::Tensor& k,
                                         const torch::Tensor& v,
                                         const torch::Tensor& beta,
                                         const torch::Tensor& g_cumsum,
                                         const torch::Tensor& A,
                                         const torch::Tensor& cu_seqlens,
                                         const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t K = k.size(3);
  int64_t V = v.size(-1);
  int64_t H = v.size(-2);
  int64_t BT = A.size(-1);
  int64_t BK = ceil_div(K, 64) * 64;
  int64_t BV = ceil_div(V, 64) * 64;
  int64_t NT;
  if (cu_seqlens.numel() == 0) {
    NT = ceil_div(T, BT);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * H * NT;

  auto u = torch::empty_like(v, k.device());
  auto w = k.new_empty({B, T, H, K}, k.device());
  auto trans_beta = beta.permute({2, 0, 1}).contiguous();

  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  triton_jit::JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.wy_fast",
      /*fn_name=*/"tmo_recompute_w_u_fwd_kernel")
      .launch(static_cast<void*>(queue),
              /*grid=*/{dim.x, dim.y, dim.z},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              k,
              v,
              trans_beta,
              w,
              u,
              A,
              opt_tensor(g_cumsum),
              opt_tensor(cu_seqlens),
              opt_tensor(chunk_indices),
              static_cast<int32_t>(T),
              static_cast<int32_t>(B),
              static_cast<int32_t>(num_v_heads_),
              static_cast<int32_t>(num_k_heads_),
              static_cast<int32_t>(K),
              static_cast<int32_t>(V),
              static_cast<int32_t>(BT),
              static_cast<int32_t>(BK),
              static_cast<int32_t>(BV),
              static_cast<int32_t>(NT),
              static_cast<int32_t>(chunk_num),
              /*ALLOW_TF32=*/1,
              /*IS_VARLEN=*/cu_seqlens.numel() > 0 ? 1 : 0);
  return std::make_pair(w, u);
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor>
ChunkGatedDeltaRuleImpl::chunk_gated_delta_rule_fwd_h(
    const torch::Tensor& k,
    const torch::Tensor& w,
    const torch::Tensor& u,
    const std::optional<torch::Tensor>& g,
    const std::optional<torch::Tensor>& gk,
    const std::optional<torch::Tensor>& initial_state,
    bool output_final_state,
    int64_t chunk_size,
    bool save_new_value,
    const std::optional<torch::Tensor>& cu_seqlens,
    const torch::Tensor& chunk_indices) {
  int64_t B = k.size(0);
  int64_t T = k.size(1);
  int64_t K = k.size(3);
  int64_t V = u.size(-1);
  int64_t H = u.size(-2);
  int64_t BK = ceil_div(K, 64) * 64;
  int64_t BV = kBv;

  int64_t N, NT;
  torch::Tensor cu_seqlens_tensor, chunk_offsets;

  if (!cu_seqlens.has_value()) {
    N = B;
    NT = ceil_div(T, chunk_size);
  } else {
    cu_seqlens_tensor = cu_seqlens.value();
    N = cu_seqlens_tensor.size(0) - 1;
    NT = chunk_indices.size(0);
    chunk_offsets = prepare_chunk_offsets(cu_seqlens_tensor, chunk_size);
  }

  auto h = k.new_empty({B, NT, H, V, K}, k.device());
  torch::Tensor final_state, v_new;
  if (output_final_state) {
    final_state = k.new_empty({N, H, V, K}, torch::kFloat32);
  }
  if (save_new_value) {
    v_new = torch::empty_like(u);
  }

  int64_t NV = ceil_div(V, kBv);
  cnrtDim3_t dim_block = {
      static_cast<uint32_t>(NV), static_cast<uint32_t>(N * H), 1};
  auto queue = torch_mlu::getCurMLUStream();
  int32_t use_g = g.has_value() ? 1 : 0;
  int32_t use_gk = gk.has_value() ? 1 : 0;
  int32_t use_initial_state = initial_state.has_value() ? 1 : 0;
  int32_t store_final_state = output_final_state ? 1 : 0;
  int32_t save_new_value_flag = save_new_value ? 1 : 0;
  int32_t is_varlen = cu_seqlens.has_value() ? 1 : 0;

  triton_jit::JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.chunk_delta_h",
      /*fn_name=*/"tmo_chunk_gated_delta_rule_fwd_kernel_h_blockdim64")
      .launch(static_cast<void*>(queue),
              /*grid=*/{dim_block.x, dim_block.y, dim_block.z},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              k,
              u,
              w,
              opt_tensor(v_new),
              g,
              gk,
              h,
              initial_state,
              opt_tensor(final_state),
              cu_seqlens,
              opt_tensor(chunk_offsets),
              static_cast<int32_t>(T),
              static_cast<int32_t>(B),
              static_cast<int32_t>(num_v_heads_),
              static_cast<int32_t>(num_k_heads_),
              static_cast<int32_t>(K),
              static_cast<int32_t>(V),
              /*BT=*/64,
              static_cast<int32_t>(BK),
              static_cast<int32_t>(BV),
              /*USE_G=*/use_g,
              /*USE_GK=*/use_gk,
              /*USE_INITIAL_STATE=*/use_initial_state,
              /*STORE_FINAL_STATE=*/store_final_state,
              /*SAVE_NEW_VALUE=*/save_new_value_flag,
              /*IS_VARLEN=*/is_varlen,
              /*ALLOW_TF32=*/1,
              /*STATE_V_FIRST=*/1);
  return std::make_tuple(h, v_new, final_state);
}

torch::Tensor ChunkGatedDeltaRuleImpl::chunk_fwd_o(
    const torch::Tensor& q,
    const torch::Tensor& k,
    const torch::Tensor& v,
    const torch::Tensor& h,
    const std::optional<torch::Tensor>& g,
    const std::optional<float>& scale,
    const std::optional<torch::Tensor>& cu_seqlens,
    int64_t chunk_size,
    const torch::Tensor& chunk_indices) {
  int64_t B = q.size(0);
  int64_t T = q.size(1);
  int64_t K = q.size(3);
  int64_t V = v.size(-1);
  int64_t H = v.size(-2);
  int64_t BT = kDefaultChunkSize;
  int64_t BK = ceil_div(K, 64) * 64;
  int64_t BV = ceil_div(V, 64) * 64;
  int64_t NT;
  if (!cu_seqlens.has_value()) {
    NT = ceil_div(T, BT);
  } else {
    NT = chunk_indices.size(0);
  }
  int64_t chunk_num = B * H * NT;

  float scale_value = scale.has_value()
                          ? scale.value()
                          : 1.0f / std::sqrt(static_cast<float>(k.size(-1)));

  torch::Tensor o = torch::empty_like(v);
  torch::Tensor g_internal = g.has_value() ? g.value() : torch::Tensor();
  torch::Tensor cu_seqlens_internal =
      cu_seqlens.has_value() ? cu_seqlens.value() : torch::Tensor();

  auto queue = torch_mlu::getCurMLUStream();
  auto dim = compute_grid_dim(chunk_num);

  triton_jit::JITKernel::get(
      /*py_path=*/"torch_mlu_ops.triton.fla.chunk_o",
      /*fn_name=*/"tmo_chunk_fwd_kernel_o")
      .launch(static_cast<void*>(queue),
              /*grid=*/{dim.x, dim.y, dim.z},
              /*cfg=*/{/*num_warps=*/1, /*num_stages=*/4},
              q,
              k,
              v,
              h,
              opt_tensor(g_internal),
              o,
              opt_tensor(cu_seqlens_internal),
              chunk_indices,
              scale_value,
              static_cast<int32_t>(T),
              static_cast<int32_t>(B),
              static_cast<int32_t>(num_v_heads_),
              static_cast<int32_t>(num_k_heads_),
              static_cast<int32_t>(K),
              static_cast<int32_t>(V),
              static_cast<int32_t>(BT),
              static_cast<int32_t>(BK),
              static_cast<int32_t>(BV),
              static_cast<int32_t>(NT),
              static_cast<int32_t>(chunk_num),
              /*USE_G=*/g.has_value() ? 1 : 0,
              /*IS_VARLEN=*/cu_seqlens.has_value() ? 1 : 0,
              /*ALLOW_TF32=*/1,
              /*STATE_V_FIRST=*/1);
  return o;
}
}  // namespace mlu
}  // namespace kernel
}  // namespace xllm
