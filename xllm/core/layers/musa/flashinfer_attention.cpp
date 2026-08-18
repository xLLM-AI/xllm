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

#include "layers/musa/flashinfer_attention.h"

#include <algorithm>
#include <cstdlib>
#include <limits>
#include <string>
#include <tuple>

#include "framework/kv_cache/kv_cache.h"
#include "kernels/musa/musa_ops_api.h"
#include "kernels/ops_api.h"
#include "layers/common/attention_metadata.h"
#include "layers/cuda/flashinfer_workspace.h"
#include "layers/musa/flashinfer_planinfo.h"

namespace xllm {
namespace layer {

namespace {

bool is_fa3_shape_supported(int64_t head_size,
                            int64_t num_heads,
                            int64_t num_kv_heads) {
  if (num_kv_heads <= 0 || num_heads % num_kv_heads != 0) {
    return false;
  }
  const int64_t gqa_ratio = num_heads / num_kv_heads;
  return head_size == 256 && (gqa_ratio == 6 || gqa_ratio == 8);
}

bool use_expanded_spec_decode_attention(
    const AttentionMetadata& attn_metadata) {
  const ExpandedDecodeMetadata& expanded = attn_metadata.expanded_decode;
  return expanded.enabled && expanded.paged_kv_indptr.defined() &&
         expanded.paged_kv_indptr.numel() >= 2;
}

AttentionMetadata build_expanded_decode_metadata(
    const AttentionMetadata& attn_metadata) {
  AttentionMetadata decode_meta = attn_metadata;
  const ExpandedDecodeMetadata& expanded = attn_metadata.expanded_decode;
  const int64_t expanded_batch = expanded.paged_kv_indptr.size(0) - 1;
  const torch::Device expanded_device = expanded.paged_kv_indptr.device();
  const torch::TensorOptions expanded_int_options =
      torch::TensorOptions().dtype(torch::kInt32).device(expanded_device);
  decode_meta.paged_kv_indptr = expanded.paged_kv_indptr;
  decode_meta.paged_kv_indices = expanded.paged_kv_indices;
  decode_meta.paged_kv_last_page_len = expanded.paged_kv_last_page_len;
  decode_meta.block_table = expanded.block_table;
  decode_meta.kv_seq_lens = expanded.kv_seq_lens;
  decode_meta.qo_indptr =
      torch::arange(0, expanded_batch + 1, expanded_int_options);
  decode_meta.max_query_len = 1;
  // Host mirrors live under fa3_metadata after the common metadata move.
  decode_meta.fa3_metadata.paged_kv_indptr_host = torch::Tensor();
  decode_meta.fa3_metadata.paged_kv_indices_host = torch::Tensor();
  decode_meta.fa3_metadata.paged_kv_last_page_len_host = torch::Tensor();
  return decode_meta;
}

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
                        sliding_window > 0 ? sliding_window - 1 : -1) {
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
  std::optional<torch::Tensor> output_lse = std::nullopt;
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
    CHECK(attn_metadata.slot_mapping.defined())
        << "FlashInferAttention requires slot_mapping when KV cache is defined";
    CHECK_EQ(attn_metadata.slot_mapping.numel(), key.size(0))
        << "slot_mapping token count mismatch: slot_mapping="
        << attn_metadata.slot_mapping.sizes() << ", key=" << key.sizes()
        << ", value=" << value.sizes() << ", k_cache=" << k_cache.sizes()
        << ", v_cache=" << v_cache.sizes()
        << ", is_prefill=" << attn_metadata.is_prefill
        << ", is_chunked_prefill=" << attn_metadata.is_chunked_prefill;
    xllm::kernel::ReshapePagedCacheParams reshape_paged_cache_params;
    reshape_paged_cache_params.key = key;
    reshape_paged_cache_params.value = value;
    reshape_paged_cache_params.k_cache = k_cache;
    reshape_paged_cache_params.v_cache = v_cache;
    reshape_paged_cache_params.slot_mapping = attn_metadata.slot_mapping;
    xllm::kernel::reshape_paged_cache(reshape_paged_cache_params);
  }

  const bool spec_verify_expanded_decode =
      attn_metadata.is_chunked_prefill &&
      (attn_metadata.expanded_decode.enabled ||
       use_expanded_spec_decode_attention(attn_metadata) ||
       attn_metadata.is_spec_verify);
  if (attn_metadata.is_prefill) {
    prefill_forward(
        attn_metadata, query, key, value, output, output_lse, k_cache, v_cache);
  } else if (spec_verify_expanded_decode) {
    decoder_forward(
        attn_metadata, query, key, output, output_lse, k_cache, v_cache);
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
    std::optional<torch::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  const Fa3AttentionMetadata& fa3_metadata = attn_metadata.fa3_metadata;
  bool use_custom_mask = attn_metadata.attn_mask.defined();

  static const int32_t fa3_setting = [] {
    const char* env = std::getenv("XLLM_USE_FA3");
    if (env == nullptr) {
      return int32_t{-1};
    }
    return std::string(env) == "1" ? int32_t{1} : int32_t{0};
  }();
  const int64_t gqa_ratio = num_kv_heads_ > 0 ? num_heads_ / num_kv_heads_ : 0;
  const bool fa3_shape_supported =
      query.scalar_type() == torch::kBFloat16 &&
      is_fa3_shape_supported(head_size_, num_heads_, num_kv_heads_);
  const bool default_to_fa3 = fa3_shape_supported;
  const bool use_fa3 = fa3_setting < 0 ? default_to_fa3 : fa3_setting == 1;

  // Supported BF16 shapes use FA3 by default.
  if (use_fa3) {
    CHECK(!use_custom_mask)
        << "XLLM_USE_FA3=1 does not support custom attention masks";
    CHECK_EQ(head_size_, 256)
        << "XLLM_USE_FA3=1 requires head_dim=256, got " << head_size_;
    CHECK_GT(num_kv_heads_, 0)
        << "XLLM_USE_FA3=1 requires at least one KV head";
    CHECK(num_heads_ == num_kv_heads_ * 6 || num_heads_ == num_kv_heads_ * 8)
        << "XLLM_USE_FA3=1 requires GQA ratio 6 or 8 (nq=" << num_heads_
        << ", nkv=" << num_kv_heads_ << ", ratio=" << gqa_ratio << ")";
    CHECK_EQ(query.scalar_type(), torch::kBFloat16)
        << "XLLM_USE_FA3=1 requires bf16 query";
    CHECK(attn_metadata.q_cu_seq_lens.defined())
        << "XLLM_USE_FA3=1 requires q_cu_seq_lens";
    CHECK(attn_metadata.kv_cu_seq_lens.defined())
        << "XLLM_USE_FA3=1 requires kv_cu_seq_lens";
    CHECK_GT(attn_metadata.max_query_len, 0)
        << "XLLM_USE_FA3=1 requires max_query_len > 0";
    CHECK_GT(attn_metadata.max_seq_len, 0)
        << "XLLM_USE_FA3=1 requires max_seq_len > 0";

    // MUSA FA3 path uses Mate's paged flash_attn_with_kvcache for
    // extend/prefill. xLLM has already written the projected K/V into the
    // same [block, page, kv_head, head_dim] cache before entering this
    // function, so use that path for regular paged prefill instead of
    // rereading dense K/V. Default on when metadata is available;
    // set XLLM_FA3_PREFILL_PAGED=0 to force the dense path.
    static const bool use_paged_fa3_prefill = [] {
      const char* env = std::getenv("XLLM_FA3_PREFILL_PAGED");
      return env == nullptr || std::string(env) != "0";
    }();
    // Ordinary-prefill dispatch: a fresh request has no
    // existing prefix and uses the dense ragged FA3 kernel; paged
    // flash_attn_with_kvcache is reserved for an actual cached prefix (or a
    // later chunk). Merely having a block table is not evidence of a prefix --
    // xLLM allocates one for every request before the first forward.
    bool has_cached_prefix = true;
    if (attn_metadata.q_seq_lens_vec.size() ==
            attn_metadata.kv_seq_lens_vec.size() &&
        !attn_metadata.q_seq_lens_vec.empty()) {
      has_cached_prefix = false;
      for (size_t i = 0; i < attn_metadata.q_seq_lens_vec.size(); ++i) {
        if (attn_metadata.kv_seq_lens_vec[i] >
            attn_metadata.q_seq_lens_vec[i]) {
          has_cached_prefix = true;
          break;
        }
      }
    }
    const bool paged_prefill_supported =
        use_paged_fa3_prefill && k_cache.defined() && v_cache.defined() &&
        attn_metadata.block_table.defined() &&
        attn_metadata.kv_seq_lens.defined() &&
        attn_metadata.kv_cu_seq_lens.defined() && has_cached_prefix;
    if (paged_prefill_supported) {
      CHECK_EQ(k_cache.dim(), 4) << "FA3 paged prefill requires a 4D key cache";
      CHECK_EQ(v_cache.dim(), 4)
          << "FA3 paged prefill requires a 4D value cache";
      CHECK_EQ(attn_metadata.block_table.scalar_type(), torch::kInt32)
          << "FA3 paged prefill requires int32 block_table";
      CHECK(attn_metadata.block_table.is_contiguous())
          << "FA3 paged prefill requires contiguous block_table";
      CHECK_EQ(attn_metadata.block_table.size(0),
               attn_metadata.kv_seq_lens.numel())
          << "FA3 paged prefill batch metadata mismatch";

      torch::Tensor seqused_k = attn_metadata.kv_seq_lens;
      if (seqused_k.scalar_type() != torch::kInt32) {
        seqused_k = seqused_k.to(torch::kInt32);
      }
      seqused_k = seqused_k.contiguous();
      const torch::Tensor cu_seqlens_q =
          attn_metadata.q_cu_seq_lens.contiguous();
      const torch::Tensor cu_seqlens_k_new =
          attn_metadata.kv_cu_seq_lens.contiguous();
      const int32_t batch_size =
          static_cast<int32_t>(attn_metadata.block_table.size(0));
      torch::Tensor scheduler_metadata;
      if (fa3_metadata.share_fa3_scheduler_metadata &&
          fa3_metadata.fa3_scheduler_metadata.defined()) {
        CHECK_EQ(fa3_metadata.fa3_scheduler_metadata.numel(),
                 static_cast<int64_t>(batch_size) * 4)
            << "FA3 prefill scheduler metadata shape changed within one "
               "forward";
        scheduler_metadata = fa3_metadata.fa3_scheduler_metadata;
      }
      if (!scheduler_metadata.defined()) {
        scheduler_metadata = xllm::kernel::musa::fa3_prefill_scheduler_metadata(
            query.device(),
            batch_size,
            static_cast<int32_t>(num_heads_),
            static_cast<int32_t>(num_kv_heads_),
            static_cast<int32_t>(head_size_),
            static_cast<int32_t>(head_size_),
            static_cast<int32_t>(attn_metadata.max_query_len),
            static_cast<int32_t>(attn_metadata.max_seq_len),
            static_cast<int32_t>(sliding_window_),
            /*window_size_right=*/0,
            cu_seqlens_q,
            cu_seqlens_k_new,
            seqused_k);
        if (fa3_metadata.share_fa3_scheduler_metadata) {
          fa3_metadata.fa3_scheduler_metadata = scheduler_metadata;
        }
      }

      torch::Tensor lse_tensor;
      if (output_lse.has_value() && output_lse->defined()) {
        lse_tensor = *output_lse;
      } else {
        const int64_t required = num_heads_ * query.size(0);
        const auto lse_options = torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(query.device());
        const bool need_realloc =
            !prefill_lse_buf_.defined() ||
            prefill_lse_buf_.dtype() != lse_options.dtype() ||
            prefill_lse_buf_.device() != lse_options.device() ||
            prefill_lse_buf_.numel() < required;
        if (need_realloc) {
          prefill_lse_buf_ = torch::empty({required}, lse_options);
        }
        lse_tensor = prefill_lse_buf_.narrow(0, 0, required)
                         .view({num_heads_, query.size(0)});
      }

      torch::Tensor query_contiguous = query.contiguous();
      xllm::kernel::musa::fa3_prefill_paged(
          query_contiguous,
          k_cache,
          v_cache,
          cu_seqlens_q,
          cu_seqlens_k_new,
          seqused_k,
          attn_metadata.block_table,
          scheduler_metadata,
          /*max_seqlen_q=*/attn_metadata.max_query_len,
          /*window_left=*/sliding_window_,
          /*window_right=*/0,
          scale_,
          output,
          lse_tensor);
      if (output_lse.has_value()) {
        *output_lse = lse_tensor;
      }
      return;
    }

    const int64_t total_q = query.size(0);
    torch::Tensor lse_tensor;
    if (output_lse.has_value() && output_lse->defined()) {
      lse_tensor = *output_lse;
    } else {
      const int64_t required = num_heads_ * total_q;
      const auto lse_options =
          torch::TensorOptions().dtype(torch::kFloat32).device(query.device());
      const bool need_realloc =
          !prefill_lse_buf_.defined() ||
          prefill_lse_buf_.dtype() != lse_options.dtype() ||
          prefill_lse_buf_.device() != lse_options.device() ||
          prefill_lse_buf_.numel() < required;
      if (need_realloc) {
        prefill_lse_buf_ = torch::empty({required}, lse_options);
      }
      lse_tensor =
          prefill_lse_buf_.narrow(0, 0, required).view({num_heads_, total_q});
    }

    torch::Tensor query_contiguous = query.contiguous();
    torch::Tensor key_contiguous = key.contiguous();
    torch::Tensor value_contiguous = value.contiguous();
    torch::Tensor q_cu_seq_lens = attn_metadata.q_cu_seq_lens.contiguous();
    torch::Tensor kv_cu_seq_lens = attn_metadata.kv_cu_seq_lens.contiguous();
    xllm::kernel::musa::fa3_prefill(
        query_contiguous,
        key_contiguous,
        value_contiguous,
        q_cu_seq_lens,
        kv_cu_seq_lens,
        /*max_seqlen_q=*/attn_metadata.max_query_len,
        /*max_seqlen_k=*/attn_metadata.max_seq_len,
        /*window_left=*/-1,
        /*window_right=*/-1,
        scale_,
        output,
        lse_tensor);
    if (output_lse.has_value()) {
      *output_lse = lse_tensor;
    }
    return;
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

  std::string backend = xllm::kernel::musa::determine_attention_backend(
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
    musa::flashinfer::update_prefill_plan_info(attn_metadata.plan_info,
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

  xllm::kernel::musa::batch_prefill(attn_metadata.plan_info->uri,
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
    std::optional<torch::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  // Graph capture plans expanded spec-verify as batch_decode (exports "run").
  // batch_chunked_prefill looks up "paged_run" on the same URI and fatals.
  if (attn_metadata.plan_info &&
      (attn_metadata.plan_info->uri.find("batch_decode") != std::string::npos ||
       attn_metadata.is_spec_verify)) {
    decoder_forward(
        attn_metadata, query, key, output, output_lse, k_cache, v_cache);
    return;
  }
  const Fa3AttentionMetadata& fa3_metadata = attn_metadata.fa3_metadata;
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
    musa::flashinfer::update_chunked_prefill_plan_info(
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

  std::optional<torch::Tensor> qo_indptr_arg;
  if (attn_metadata.qo_indptr.has_value() &&
      attn_metadata.qo_indptr->defined()) {
    qo_indptr_arg = attn_metadata.qo_indptr;
  }

  xllm::kernel::musa::batch_chunked_prefill(
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
      qo_indptr_arg,
      /*causal=*/true,
      fa3_metadata.paged_kv_indptr_host,
      fa3_metadata.paged_kv_indices_host,
      fa3_metadata.paged_kv_last_page_len_host);
}

void FlashInferAttentionImpl::decoder_forward(
    const AttentionMetadata& attn_metadata,
    torch::Tensor& query,
    const torch::Tensor& key,
    torch::Tensor& output,
    std::optional<torch::Tensor>& output_lse,
    const torch::Tensor& k_cache,
    const torch::Tensor& v_cache) {
  std::optional<AttentionMetadata> expanded_decode_meta;
  if (use_expanded_spec_decode_attention(attn_metadata)) {
    expanded_decode_meta = build_expanded_decode_metadata(attn_metadata);
  }
  const AttentionMetadata& decode_attn =
      expanded_decode_meta.has_value() ? *expanded_decode_meta : attn_metadata;
  // Keep FA3 scheduler sharing on the original metadata so layer-to-layer
  // reuse still works when decode_attn is an expanded copy.
  const Fa3AttentionMetadata& fa3_metadata = attn_metadata.fa3_metadata;
  // Match the graph executor default while allowing an explicit FA3 override.
  {
    static const int32_t fa3_setting = [] {
      const char* decode_env = std::getenv("XLLM_USE_FA3_DECODE");
      if (decode_env != nullptr) {
        return std::string(decode_env) == "1" ? int32_t{1} : int32_t{0};
      }
      const char* env = std::getenv("XLLM_USE_FA3");
      if (env == nullptr) {
        return int32_t{-1};
      }
      return std::string(env) == "1" ? int32_t{1} : int32_t{0};
    }();
    const int64_t gqa_ratio =
        num_kv_heads_ > 0 ? num_heads_ / num_kv_heads_ : 0;
    const bool fa3_shape_supported =
        query.scalar_type() == torch::kBFloat16 &&
        is_fa3_shape_supported(head_size_, num_heads_, num_kv_heads_);
    const bool default_to_fa3 = fa3_shape_supported && gqa_ratio == 8;
    const bool use_fa3 = fa3_setting < 0 ? default_to_fa3 : fa3_setting == 1;
    if (use_fa3 && fa3_shape_supported) {
      CHECK(decode_attn.block_table.defined())
          << "FA3 decode requires block_table (rectangular page_table)";
      const int64_t batch_size = decode_attn.block_table.size(0);

      // seqused_k = per-seq kv length (NOT cumulative). attn_metadata
      // already keeps it under `kv_seq_lens`; if undefined fall back to
      // torch::diff of the cumulative form.
      torch::Tensor seqused_k = decode_attn.kv_seq_lens;
      if (!seqused_k.defined() || seqused_k.numel() == 0) {
        CHECK(decode_attn.kv_cu_seq_lens.defined())
            << "FA3 decode requires kv_seq_lens or kv_cu_seq_lens";
        seqused_k = torch::diff(decode_attn.kv_cu_seq_lens).to(torch::kInt32);
      } else if (seqused_k.scalar_type() != torch::kInt32) {
        seqused_k = seqused_k.to(torch::kInt32);
      }
      seqused_k = seqused_k.contiguous();

      // page_table: native rectangular block_table built by the input builder
      // from allocated KV blocks. Unused slots are
      // -1; graph mode reuses persistent_block_tables_ updated each step.
      const torch::Tensor page_table = decode_attn.block_table;

      // Use the exact host-side KV length for scheduler partitioning.
      const int32_t max_seqlen_k =
          static_cast<int32_t>(decode_attn.max_seq_len);
      const int32_t max_seqlen_q =
          static_cast<int32_t>(std::max<int64_t>(decode_attn.max_query_len, 1));
      CHECK_GT(max_seqlen_k, 0)
          << "FA3 decode requires attn_metadata.max_seq_len > 0";

      // Qwen hybrid explicitly scopes this cache to one model forward. Other
      // models retain per-layer generation so shared graph-capture metadata
      // cannot accidentally keep scheduler values from a previous step.
      torch::Tensor scheduler_metadata;
      if (fa3_metadata.share_fa3_scheduler_metadata &&
          fa3_metadata.fa3_scheduler_metadata.defined()) {
        CHECK_EQ(fa3_metadata.fa3_scheduler_metadata.numel(), batch_size * 4)
            << "FA3 scheduler metadata shape changed within one forward";
        scheduler_metadata = fa3_metadata.fa3_scheduler_metadata;
      }
      const torch::Tensor cu_seqlens_q =
          decode_attn.qo_indptr.has_value() && decode_attn.qo_indptr->defined()
              ? *decode_attn.qo_indptr
              : decode_attn.q_cu_seq_lens;
      if (!scheduler_metadata.defined()) {
        scheduler_metadata = xllm::kernel::musa::fa3_decode_scheduler_metadata(
            query.device(),
            /*batch_size=*/static_cast<int32_t>(batch_size),
            /*num_heads_q=*/static_cast<int32_t>(num_heads_),
            /*num_heads_kv=*/static_cast<int32_t>(num_kv_heads_),
            /*head_dim_qk=*/static_cast<int32_t>(head_size_),
            /*head_dim_vo=*/static_cast<int32_t>(head_size_),
            /*max_seqlen_q=*/max_seqlen_q,
            /*max_seqlen_k=*/max_seqlen_k,
            /*window_size_left=*/static_cast<int32_t>(sliding_window_),
            /*window_size_right=*/0,
            /*cu_seqlens_q=*/cu_seqlens_q,
            /*seqused_k=*/seqused_k);
        if (fa3_metadata.share_fa3_scheduler_metadata) {
          fa3_metadata.fa3_scheduler_metadata = scheduler_metadata;
        }
      }

      // Keep the FA3 LSE address stable across graph capture and replay.
      const int64_t total_q = query.size(0);
      torch::Tensor lse_tensor;
      if (output_lse.has_value() && output_lse->defined()) {
        lse_tensor = *output_lse;
      } else {
        const int64_t required = num_heads_ * total_q;
        const auto lse_options = torch::TensorOptions()
                                     .dtype(torch::kFloat32)
                                     .device(query.device());
        const bool need_realloc =
            !decode_lse_buf_.defined() ||
            decode_lse_buf_.dtype() != lse_options.dtype() ||
            decode_lse_buf_.device() != lse_options.device() ||
            decode_lse_buf_.numel() < required;
        if (need_realloc) {
          constexpr int64_t kDefaultMaxGraphBatchSize = 256;
          decode_lse_buf_ = torch::empty(
              {std::max(required, kDefaultMaxGraphBatchSize * num_heads_)},
              lse_options);
        }
        lse_tensor =
            decode_lse_buf_.narrow(0, 0, required).view({num_heads_, total_q});
      }

      xllm::kernel::musa::fa3_decode(
          query,
          k_cache,
          v_cache,
          cu_seqlens_q,
          seqused_k,
          page_table,
          scheduler_metadata,
          /*max_seqlen_q=*/max_seqlen_q,
          /*window_left=*/static_cast<int64_t>(sliding_window_),
          /*window_right=*/0,
          scale_,
          output,
          lse_tensor);
      if (output_lse.has_value()) {
        *output_lse = lse_tensor;
      }
      return;
    }
  }

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

  if (decode_attn.enable_cuda_graph) {
    CHECK(decode_attn.plan_info->plan_info.defined())
        << "plan_info plan_info should not be null when enable_cuda_graph is "
           "true";
    VLOG(kGraphExecutorLogVerboseLevel)
        << "no need to update plan_info for CUDA graph";
  } else {
    musa::flashinfer::update_decode_plan_info(decode_attn.plan_info,
                                              backend,
                                              decode_attn,
                                              query.scalar_type(),
                                              key.scalar_type(),
                                              output.scalar_type(),
                                              head_size_,
                                              head_size_,
                                              num_heads_,
                                              num_kv_heads_,
                                              block_size,
                                              sliding_window_,
                                              decode_attn.enable_cuda_graph,
                                              decode_use_tensor_core_);
  }

  xllm::kernel::musa::batch_decode(
      decode_attn.plan_info->uri,
      decode_attn.plan_info->plan_info,
      float_workspace_buffer_,
      int_workspace_buffer_,
      page_locked_int_workspace_buffer_,
      query,
      k_cache,
      v_cache,
      decode_attn.paged_kv_indptr,
      decode_attn.paged_kv_indices,
      decode_attn.paged_kv_last_page_len,
      sliding_window_,
      scale_,
      output,
      output_lse,
      decode_use_tensor_core_,
      decode_attn.qo_indptr,
      decode_attn.fa3_metadata.paged_kv_indptr_host,
      decode_attn.fa3_metadata.paged_kv_indices_host,
      decode_attn.fa3_metadata.paged_kv_last_page_len_host);
}

}  // namespace layer
}  // namespace xllm
