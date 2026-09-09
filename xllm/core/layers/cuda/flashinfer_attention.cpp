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

#include "flashinfer_attention.h"

#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"

namespace xllm {
namespace layer {

namespace {

// Eager causal + padding attention fallback when custom mask is used (e.g.
// LongCat text encoder). FlashInfer's custom mask path gives wrong token-0
// output; this path matches diffusers.
std::tuple<torch::Tensor, std::optional<torch::Tensor>>
run_eager_causal_padded_attention(const torch::Tensor& query,
                                  const torch::Tensor& key,
                                  const torch::Tensor& value,
                                  const torch::Tensor& attn_mask_1d,
                                  float scale,
                                  int64_t num_heads,
                                  int64_t num_kv_heads,
                                  int64_t head_size) {
  torch::Tensor m = attn_mask_1d;
  if (m.device() != query.device()) {
    m = m.to(query.device());
  }
  if (!m.is_floating_point()) {
    m = m.to(torch::kFloat32);
  }
  int64_t T = query.size(0);
  CHECK_EQ(m.size(0), T) << "[eager attention] mask length " << m.size(0)
                         << " != query seq len " << T;
  auto device = query.device();
  auto causal = torch::tril(torch::ones(
      {T, T}, torch::TensorOptions().dtype(torch::kFloat32).device(device)));
  auto pad2d = m.unsqueeze(0).expand({T, T});
  auto combined = (causal * pad2d).to(torch::kFloat32);
  const float mask_val = -std::numeric_limits<float>::infinity();
  auto add_mask = torch::where(combined > 0.5f,
                               torch::zeros_like(combined),
                               torch::full_like(combined, mask_val));
  int64_t g = num_heads / num_kv_heads;
  // [T,K,D] -> [T,K,D,1] -> [T,K,D,g] -> permute to [T,K,g,D] -> [T,K*g,D].
  // Head h = kv_head k, replicate r: h = k*g + r; each head gets full D
  // dims.
  auto Kg = key.unsqueeze(3).expand({-1, -1, -1, g});
  auto Vg = value.unsqueeze(3).expand({-1, -1, -1, g});
  torch::Tensor Kr =
      Kg.permute({0, 1, 3, 2}).reshape({-1, num_heads, head_size});
  torch::Tensor Vr =
      Vg.permute({0, 1, 3, 2}).reshape({-1, num_heads, head_size});
  auto Qf = query.to(torch::kFloat32);
  auto Kf = Kr.to(torch::kFloat32);
  // Optimized: use bmm to compute Q @ K^T directly, avoiding O(T^2 * H * D)
  // intermediate tensor. Memory: O(T^2 * H) instead of O(T^2 * H * D).
  // Q: [T, H, D] -> [H, T, D], K: [T, H, D] -> [H, D, T]
  // scores = Q @ K^T = [H, T, T] -> permute to [T, H, T]
  auto Qf_HTD = Qf.permute({1, 0, 2});               // [H, T, D]
  auto Kf_HDT = Kf.permute({1, 2, 0});               // [H, D, T]
  auto scores = torch::bmm(Qf_HTD, Kf_HDT) * scale;  // [H, T, T]
  scores = scores.permute({1, 0, 2});                // [T, H, T]
  scores = scores + add_mask.unsqueeze(1);
  // Match diffusers: softmax in float32, cast attn to query dtype; attn @ V
  // in bf16.
  auto attn =
      torch::softmax(scores.to(torch::kFloat32), -1).to(query.scalar_type());
  auto out = torch::einsum("thj,jhd->thd", {attn, Vr}).contiguous();
  auto result = out.view({-1, num_heads * head_size});
  return {result, std::nullopt};
}

}  // namespace

FlashInferAttentionImpl::FlashInferAttentionImpl(int64_t num_heads,
                                                 int64_t head_size,
                                                 float scale,
                                                 int64_t num_kv_heads,
                                                 int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window - 1) {
  float_workspace_buffer_ = flashinfer::FlashinferWorkspace::get_instance()
                                .get_float_workspace_buffer();
  int_workspace_buffer_ = flashinfer::FlashinferWorkspace::get_instance()
                              .get_int_workspace_buffer();
  page_locked_int_workspace_buffer_ =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>>
FlashInferAttentionImpl::forward(const AttentionMetadata& attn_metadata,
                                 torch::Tensor& query,
                                 torch::Tensor& key,
                                 torch::Tensor& value,
                                 torch::Tensor& output,
                                 KVCache& kv_cache) {
  std::optional<at::Tensor> output_lse = std::nullopt;
  if (attn_metadata.max_seq_len == 0) {
    output = output.view({-1, num_heads_ * head_size_});
    return std::make_tuple(output, output_lse);
  }

  query = query.view({-1, num_heads_, head_size_});
  key = key.view({-1, num_kv_heads_, head_size_});
  value = value.view({-1, num_kv_heads_, head_size_});
  output = output.view({-1, num_heads_, head_size_});

  torch::Tensor k_cache = kv_cache.get_k_cache();
  torch::Tensor v_cache = kv_cache.get_v_cache();

  // Only reshape and store to cache if k_cache is properly initialized
  // For prefill without KV cache (e.g., LongCat text encoding), skip this step
  if (k_cache.defined() && k_cache.dim() >= 2) {
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  if (attn_metadata.is_prefill) {
    prefill_forward(attn_metadata, query, key, value, output, output_lse);
  } else if (attn_metadata.is_chunked_prefill) {
    chunked_prefill_forward(
        attn_metadata, query, key, output, output_lse, k_cache, v_cache);
  } else {
    decoder_forward(
        attn_metadata, query, key, output, output_lse, k_cache, v_cache);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void FlashInferAttentionImpl::prefill_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& key,
    torch::Tensor& value,
    torch::Tensor& output,
    std::optional<at::Tensor>& output_lse) {
  bool use_custom_mask = attn_metadata.attn_mask.defined();

  std::string backend = xllm::kernel::cuda::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      use_custom_mask);

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_prefill_plan_info(attn_metadata.plan_info,
                                         backend,
                                         attn_metadata,
                                         query.scalar_type(),
                                         key.scalar_type(),
                                         output.scalar_type(),
                                         head_size_,
                                         head_size_,
                                         num_heads_,
                                         num_kv_heads_,
                                         attn_metadata.enable_cuda_graph);
  }

  if (use_custom_mask) {
    auto [result, _] =
        run_eager_causal_padded_attention(query,
                                          key,
                                          value,
                                          attn_metadata.attn_mask,
                                          scale_,
                                          num_heads_,
                                          num_kv_heads_,
                                          head_size_);
    output = result;
    return;
  }

  xllm::kernel::cuda::batch_prefill_with_optional_piecewise_capture(
      attn_metadata.plan_info->uri,
      attn_metadata.plan_info->plan_info,
      float_workspace_buffer_,
      int_workspace_buffer_,
      page_locked_int_workspace_buffer_,
      query,
      key,
      value,
      attn_metadata.q_cu_seq_lens,
      attn_metadata.kv_cu_seq_lens,
      sliding_window_,
      scale_,
      output,
      output_lse);
}

void FlashInferAttentionImpl::chunked_prefill_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    const torch::Tensor& key,
    torch::Tensor& output,
    std::optional<at::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  // Get block_size from k_cache if defined and has proper dimensions,
  // otherwise use a default value (for prefill without KV cache, e.g., LongCat)
  int64_t block_size = 1;
  if (k_cache.defined() && k_cache.dim() >= 2) {
    block_size = k_cache.size(1);
  }

  // NOTE: we only support "fa2" backend for BatchPrefillWithPagedKvcacheKernel
  // for flashinfer v0.6.2, because it would cause performance degradation if
  // using "fa3" backend.
  std::string backend = "fa2";

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_chunked_prefill_plan_info(
        attn_metadata.plan_info,
        backend,
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        block_size,
        sliding_window_,
        attn_metadata.enable_cuda_graph);
  }

  xllm::kernel::cuda::batch_chunked_prefill(
      attn_metadata.plan_info->uri,
      attn_metadata.plan_info->plan_info,
      float_workspace_buffer_,
      int_workspace_buffer_,
      page_locked_int_workspace_buffer_,
      query,
      k_cache,
      v_cache,
      attn_metadata.paged_kv_indptr,
      attn_metadata.paged_kv_indices,
      attn_metadata.paged_kv_last_page_len,
      sliding_window_,
      scale_,
      output,
      output_lse,
      attn_metadata.qo_indptr);
}

void FlashInferAttentionImpl::decoder_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    const torch::Tensor& key,
    torch::Tensor& output,
    std::optional<at::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  // Get block_size from k_cache if defined and has proper dimensions,
  // otherwise use a default value (for prefill without KV cache, e.g., LongCat)
  int64_t block_size = 1;
  if (k_cache.defined() && k_cache.dim() >= 2) {
    block_size = k_cache.size(1);
  }

  // NOTE: we only support "fa2" backend for BatchPrefillWithPagedKvcacheKernel
  // for flashinfer v0.6.2, because it would cause performance degradation if
  // using "fa3" backend.
  std::string backend = "fa2";

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    flashinfer::update_decode_plan_info(attn_metadata.plan_info,
                                        backend,
                                        attn_metadata,
                                        query.scalar_type(),
                                        key.scalar_type(),
                                        output.scalar_type(),
                                        head_size_,
                                        head_size_,
                                        num_heads_,
                                        num_kv_heads_,
                                        block_size,
                                        sliding_window_,
                                        attn_metadata.enable_cuda_graph,
                                        decode_use_tensor_core_);
  }

  xllm::kernel::cuda::batch_decode(attn_metadata.plan_info->uri,
                                   attn_metadata.plan_info->plan_info,
                                   float_workspace_buffer_,
                                   int_workspace_buffer_,
                                   page_locked_int_workspace_buffer_,
                                   query,
                                   k_cache,
                                   v_cache,
                                   attn_metadata.paged_kv_indptr,
                                   attn_metadata.paged_kv_indices,
                                   attn_metadata.paged_kv_last_page_len,
                                   sliding_window_,
                                   scale_,
                                   output,
                                   output_lse,
                                   decode_use_tensor_core_,
                                   attn_metadata.qo_indptr);
}

}  // namespace layer
}  // namespace xllm
