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

#include "xattention.h"

#include <algorithm>

#include "core/common/global_flags.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/platform/device.h"
#include "flashinfer_planinfo.h"
#include "flashinfer_workspace.h"
#include "framework/kv_cache/kv_cache.h"
#include "kernels/cuda/cuda_ops_api.h"
#include "kernels/cuda/xattention/xattention_ops_api.h"
#include "layers/common/attention_metadata.h"
#include "xattention_planinfo.h"
#include "xattention_workspace.h"

namespace xllm {
namespace layer {

XAttentionImpl::XAttentionImpl(int64_t num_heads,
                               int64_t head_size,
                               float scale,
                               int64_t num_kv_heads,
                               int64_t sliding_window)
    : BaseAttentionImpl(num_heads,
                        head_size,
                        scale,
                        num_kv_heads,
                        sliding_window) {}

void XAttentionImpl::run_single_stage_decode(
    const AttentionMetadata& attn_metadata,
    const torch::Tensor& key,
    torch::Tensor& query,
    torch::Tensor& output) {
  torch::Tensor full_k_cache = attn_metadata.full_k_cache.unsqueeze(1);
  torch::Tensor full_v_cache = attn_metadata.full_v_cache.unsqueeze(1);

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    std::string backend = xllm::kernel::cuda::determine_attention_backend(
        /*pos_encoding_mode=*/0,
        /*use_fp16_qk_reduction=*/false,
        /*use_custom_mask=*/false);
    flashinfer::update_decode_plan_info(
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
        /*block_size=*/full_k_cache.size(1),
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/false,
        /*use_tensor_core=*/decode_use_tensor_core_);
  }

  std::optional<at::Tensor> unshared_lse = std::nullopt;

  torch::Tensor float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  torch::Tensor int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  torch::Tensor page_locked_int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  xllm::kernel::cuda::batch_decode(attn_metadata.plan_info->uri,
                                   attn_metadata.plan_info->plan_info,
                                   float_workspace_buffer,
                                   int_workspace_buffer,
                                   page_locked_int_workspace_buffer,
                                   query,
                                   full_k_cache,
                                   full_v_cache,
                                   attn_metadata.paged_kv_indptr,
                                   attn_metadata.paged_kv_indices,
                                   attn_metadata.paged_kv_last_page_len,
                                   sliding_window_,
                                   scale_,
                                   output,
                                   unshared_lse,
                                   decode_use_tensor_core_,
                                   attn_metadata.qo_indptr);
}

void XAttentionImpl::run_two_stage_decode(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    torch::Tensor& output) {
  CHECK(attn_metadata.shared_plan_info != nullptr)
      << "shared_plan_info must be initialized for two-stage decode.";
  CHECK(attn_metadata.unshared_plan_info != nullptr)
      << "unshared_plan_info must be initialized for two-stage decode.";

  const int64_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
  CHECK_GT(batch_size, 0) << "invalid batch_size";

  const int64_t total_beam = query.size(0);
  CHECK_EQ(total_beam % batch_size, 0)
      << "total_beam must be divisible by batch_size";

  const auto& cache = attn_metadata.xattention_two_stage_decode_cache.value();
  CHECK(cache.shared_lse.defined() && cache.shared_o.defined() &&
        cache.unshared_lse.defined() && cache.unshared_o.defined())
      << "two-stage output cache tensors are not initialized.";
  CHECK(cache.q_cu_seq_lens_shared.defined() &&
        cache.qo_indptr_expanded.defined() &&
        cache.paged_kv_indptr_expanded.defined() &&
        cache.paged_kv_indices_expanded.defined() &&
        cache.paged_kv_last_page_len_expanded.defined())
      << "two-stage index cache tensors are not initialized.";

  CHECK_EQ(cache.shared_o.size(0), total_beam)
      << "shared_o first dim mismatch: expected total_beam=" << total_beam
      << ", got " << cache.shared_o.size(0);
  CHECK_EQ(cache.unshared_o.size(0), total_beam)
      << "unshared_o first dim mismatch: expected total_beam=" << total_beam
      << ", got " << cache.unshared_o.size(0);
  CHECK_EQ(cache.q_cu_seq_lens_shared.numel(), batch_size + 1)
      << "q_cu_seq_lens_shared size mismatch: expected " << (batch_size + 1)
      << ", got " << cache.q_cu_seq_lens_shared.numel();
  CHECK_EQ(cache.qo_indptr_expanded.numel(), total_beam + 1)
      << "qo_indptr_expanded size mismatch: expected " << (total_beam + 1)
      << ", got " << cache.qo_indptr_expanded.numel();
  CHECK_EQ(cache.paged_kv_indptr_expanded.numel(), total_beam + 1)
      << "paged_kv_indptr_expanded size mismatch: expected " << (total_beam + 1)
      << ", got " << cache.paged_kv_indptr_expanded.numel();
  CHECK_EQ(cache.paged_kv_indices_expanded.numel(), total_beam)
      << "paged_kv_indices_expanded size mismatch: expected " << total_beam
      << ", got " << cache.paged_kv_indices_expanded.numel();
  CHECK_EQ(cache.paged_kv_last_page_len_expanded.numel(), total_beam)
      << "paged_kv_last_page_len_expanded size mismatch: expected "
      << total_beam << ", got "
      << cache.paged_kv_last_page_len_expanded.numel();

  std::string backend = xllm::kernel::cuda::determine_attention_backend(
      /*pos_encoding_mode=*/0,
      /*use_fp16_qk_reduction=*/false,
      /*use_custom_mask=*/false);
  // ===== Shared stage: attend to shared (prompt) KV =====
  const int64_t unshared_offset = static_cast<int64_t>(
      ::xllm::SchedulerConfig::get_instance().max_tokens_per_batch());
  const int64_t shared_kv_capacity =
      std::min(attn_metadata.full_k_cache.size(0), unshared_offset);

  torch::Tensor shared_k_cache =
      attn_metadata.full_k_cache.slice(0, 0, shared_kv_capacity);
  torch::Tensor shared_v_cache =
      attn_metadata.full_v_cache.slice(0, 0, shared_kv_capacity);

  AttentionMetadata shared_attn_meta = attn_metadata;

  shared_attn_meta.plan_info = attn_metadata.shared_plan_info;
  shared_attn_meta.q_cu_seq_lens = cache.q_cu_seq_lens_shared;
  VLOG(kGraphExecutorLogVerboseLevel)
      << "shared_attn_meta.q_cu_seq_lens:" << shared_attn_meta.q_cu_seq_lens;

  auto& workspace = flashinfer::FlashinferWorkspace::get_instance();
  torch::Tensor shared_int_workspace_buffer =
      workspace.get_int_workspace_buffer();
  torch::Tensor shared_page_locked_int_workspace_buffer =
      workspace.get_page_locked_int_workspace_buffer();
  CHECK(shared_int_workspace_buffer.defined() &&
        shared_page_locked_int_workspace_buffer.defined())
      << "workspace buffers must be initialized.";

  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.shared_plan_info->plan_info.defined())
        << "shared stage plan_info should not be null when enable_cuda_graph "
           "is true";
    CHECK(!attn_metadata.shared_plan_info->uri.empty())
        << "shared stage plan_info uri should not be empty when "
           "enable_cuda_graph is true";
  } else {
    xattention::update_xattention_plan_info(attn_metadata.shared_plan_info,
                                            backend,
                                            shared_attn_meta,
                                            query.scalar_type(),
                                            shared_k_cache.scalar_type(),
                                            cache.shared_o.scalar_type(),
                                            head_size_,
                                            head_size_,
                                            num_heads_,
                                            num_kv_heads_,
                                            /*block_size=*/1,
                                            /*window_size_left=*/-1,
                                            /*enable_cuda_graph=*/false,
                                            /*causal=*/false,
                                            /*use_tensor_core=*/true,
                                            /*is_shared_stage_plan=*/true);
  }

  torch::Tensor float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  std::optional<torch::Tensor> shared_lse = cache.shared_lse;
  xllm::kernel::cuda::batch_prefill_non_causal(
      attn_metadata.shared_plan_info->uri,
      attn_metadata.shared_plan_info->plan_info,
      float_workspace_buffer,
      shared_int_workspace_buffer,
      shared_page_locked_int_workspace_buffer,
      query,
      shared_k_cache,
      shared_v_cache,
      shared_attn_meta.q_cu_seq_lens,
      shared_attn_meta.kv_cu_seq_lens,
      /*window_left=*/-1,
      /*sm_scale=*/scale_,
      cache.shared_o,
      shared_lse);

  // ===== Unshared stage: attend to unshared (per-beam) KV =====
  AttentionMetadata unshared_attn_meta = attn_metadata;
  unshared_attn_meta.plan_info = attn_metadata.unshared_plan_info;
  unshared_attn_meta.paged_kv_indptr = cache.paged_kv_indptr_expanded;
  unshared_attn_meta.paged_kv_indices = cache.paged_kv_indices_expanded;
  unshared_attn_meta.paged_kv_last_page_len =
      cache.paged_kv_last_page_len_expanded;
  unshared_attn_meta.qo_indptr = cache.qo_indptr_expanded;

  auto& xattention_workspace =
      xllm::layer::xattention::XAttentionWorkspace::get_instance();
  torch::Tensor unshared_int_workspace_buffer =
      xattention_workspace.get_int_workspace_buffer();
  torch::Tensor unshared_page_locked_int_workspace_buffer =
      xattention_workspace.get_page_locked_int_workspace_buffer();
  CHECK(unshared_int_workspace_buffer.defined() &&
        unshared_page_locked_int_workspace_buffer.defined())
      << "two-stage unshared workspace buffers must be initialized.";

  const int64_t max_decode_step = attn_metadata.unshared_k_cache.size(2);
  torch::Tensor unshared_k_cache = attn_metadata.unshared_k_cache.view(
      {total_beam, max_decode_step, num_kv_heads_, head_size_});
  torch::Tensor unshared_v_cache = attn_metadata.unshared_v_cache.view(
      {total_beam, max_decode_step, num_kv_heads_, head_size_});
  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.unshared_plan_info->plan_info.defined())
        << "unshared stage plan_info should not be null when "
           "enable_cuda_graph is true";
    CHECK(!attn_metadata.unshared_plan_info->uri.empty())
        << "unshared stage plan_info uri should not be empty when "
           "enable_cuda_graph is true";
  } else {
    xattention::update_xattention_plan_info(
        attn_metadata.unshared_plan_info,
        backend,
        unshared_attn_meta,
        query.scalar_type(),
        unshared_k_cache.scalar_type(),
        cache.unshared_o.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*block_size=*/max_decode_step,
        /*window_size_left=*/sliding_window_,
        /*enable_cuda_graph=*/false,
        /*causal=*/false,
        /*use_tensor_core=*/decode_use_tensor_core_,
        /*is_shared_stage_plan=*/false);
  }

  std::optional<torch::Tensor> unshared_lse = cache.unshared_lse;
  xllm::kernel::cuda::batch_decode(
      attn_metadata.unshared_plan_info->uri,
      attn_metadata.unshared_plan_info->plan_info,
      float_workspace_buffer,
      unshared_int_workspace_buffer,
      unshared_page_locked_int_workspace_buffer,
      query,
      unshared_k_cache,
      unshared_v_cache,
      unshared_attn_meta.paged_kv_indptr,
      unshared_attn_meta.paged_kv_indices,
      unshared_attn_meta.paged_kv_last_page_len,
      sliding_window_,
      scale_,
      cache.unshared_o,
      unshared_lse,
      // Keep execution path consistent with the
      // kernel URI chosen in update_xattention_plan_info.
      /*use_tensor_core=*/decode_use_tensor_core_,
      unshared_attn_meta.qo_indptr);

  // ===== Combine shared/unshared results =====
  xllm::kernel::cuda::lse_combine(output,
                                  cache.shared_o,
                                  cache.shared_lse,
                                  cache.unshared_o,
                                  cache.unshared_lse);
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>> XAttentionImpl::forward(
    const AttentionMetadata& attn_metadata,
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

  CHECK(!attn_metadata.is_chunked_prefill)
      << "chunked prefill is not supported";

  if (attn_metadata.is_prefill) {
    prefill_forward(attn_metadata, query, key, value, output, output_lse);
  } else {
    decoder_forward(attn_metadata, query, key, value, output);
  }

  output = output.view({-1, num_heads_ * head_size_});
  return {output, output_lse};
}

void XAttentionImpl::prefill_forward(const AttentionMetadata& attn_metadata,
                                     torch::Tensor& query,
                                     torch::Tensor& key,
                                     torch::Tensor& value,
                                     torch::Tensor& output,
                                     std::optional<at::Tensor>& output_lse) {
  if (attn_metadata.enable_cuda_graph) {
    CHECK(attn_metadata.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    CHECK(!attn_metadata.plan_info->uri.empty())
        << "plan_info uri should not be empty when enable_cuda_graph is true";
  } else {
    flashinfer::update_prefill_plan_info(
        attn_metadata.plan_info,
        xllm::kernel::cuda::determine_attention_backend(
            /*pos_encoding_mode=*/0,
            /*use_fp16_qk_reduction=*/false,
            /*use_custom_mask=*/false),
        attn_metadata,
        query.scalar_type(),
        key.scalar_type(),
        output.scalar_type(),
        head_size_,
        head_size_,
        num_heads_,
        num_kv_heads_,
        /*enable_cuda_graph=*/false);
  }

  xllm::kernel::cuda::prefill_reshape_and_cache(
      key, value, attn_metadata.full_k_cache, attn_metadata.full_v_cache);

  torch::Tensor float_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_float_workspace_buffer();
  torch::Tensor int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_int_workspace_buffer();
  torch::Tensor page_locked_int_workspace_buffer =
      flashinfer::FlashinferWorkspace::get_instance()
          .get_page_locked_int_workspace_buffer();

  xllm::kernel::cuda::batch_prefill_with_optional_piecewise_capture(
      attn_metadata.plan_info->uri,
      attn_metadata.plan_info->plan_info,
      float_workspace_buffer,
      int_workspace_buffer,
      page_locked_int_workspace_buffer,
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

void XAttentionImpl::decoder_forward(const AttentionMetadata& attn_metadata,
                                     torch::Tensor& query,
                                     torch::Tensor& key,
                                     torch::Tensor& value,
                                     torch::Tensor& output) {
  uint32_t batch_size = attn_metadata.kv_cu_seq_lens.size(0) - 1;
  uint32_t total_beam = query.size(0);
  uint32_t beam_size = total_beam / batch_size;

  key = key.view({batch_size, beam_size, num_kv_heads_, head_size_});
  value = value.view({batch_size, beam_size, num_kv_heads_, head_size_});

  xllm::kernel::cuda::decoder_reshape_and_cache(key,
                                                value,
                                                attn_metadata.unshared_k_cache,
                                                attn_metadata.unshared_v_cache,
                                                attn_metadata.step_tensor);
  if (::xllm::RecConfig::get_instance().enable_xattention_one_stage()) {
    run_single_stage_decode(attn_metadata, key, query, output);
  } else {
    run_two_stage_decode(attn_metadata, query, output);
  }
}

}  // namespace layer
}  // namespace xllm
