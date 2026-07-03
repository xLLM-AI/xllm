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

#include <glog/logging.h>

#include <cstdint>
#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::cuda {

namespace {

constexpr const char* kFa3MetadataUri =
    "fmha_get_metadata_6x1_ragged_q_padded_k_causal_packgqa";

constexpr int32_t kFa3TileM = 32;
constexpr int32_t kFa3TileN = 64;

constexpr const char* kFa3FwdUriHash =
    "9e4f4b2e6574a7a45a93fef39cf9b0485651e39052d9dfd88c2e1439137a9374";

std::string fa3_fwd_uri() { return std::string("fmha_fwd_") + kFa3FwdUriHash; }

ffi::Optional<ffi::Tensor> none_tensor() {
  return ffi::Optional<ffi::Tensor>();
}

ffi::Optional<int64_t> none_int() { return ffi::Optional<int64_t>(); }

}  // namespace

torch::Tensor fa3_decode_scheduler_metadata(const torch::Device& device,
                                            int32_t batch_size,
                                            int32_t num_heads_q,
                                            int32_t num_heads_kv,
                                            int32_t head_dim_qk,
                                            int32_t head_dim_vo,
                                            int32_t max_seqlen_q,
                                            int32_t max_seqlen_k,
                                            int32_t window_size_left,
                                            int32_t window_size_right,
                                            const torch::Tensor& cu_seqlens_q,
                                            const torch::Tensor& seqused_k) {
  CHECK_GT(batch_size, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::empty({static_cast<int64_t>(batch_size) * 4}, options);

  MusaTvmffiStreamGuard stream_guard(device);

  const int64_t b = batch_size;
  auto num_splits_dynamic = metadata.slice(/*dim=*/0, /*start=*/0, /*end=*/b);
  auto batch_table = metadata.slice(/*dim=*/0, /*start=*/b, /*end=*/2 * b);
  auto num_m_blocks = metadata.slice(/*dim=*/0, /*start=*/2 * b, /*end=*/3 * b);
  auto num_nheads_in_l2 =
      metadata.slice(/*dim=*/0, /*start=*/3 * b, /*end=*/4 * b);

  const std::string uri = kFa3MetadataUri;

  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         ffi::Optional<ffi::Tensor>(),
                         ffi::Optional<ffi::Tensor>(),
                         to_ffi_tensor(seqused_k),
                         ffi::Optional<ffi::Tensor>(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         ffi::Optional<ffi::Tensor>(),
                         to_ffi_tensor(num_splits_dynamic),
                         to_ffi_tensor(batch_table),
                         to_ffi_tensor(num_m_blocks),
                         to_ffi_tensor(num_nheads_in_l2),
                         /*num_splits=*/static_cast<int64_t>(1),
                         static_cast<int64_t>(kFa3TileM),
                         static_cast<int64_t>(kFa3TileN),
                         /*mp_margin=*/static_cast<int64_t>(0));

  return metadata;
}

void fa3_decode(const torch::Tensor& query,
                const torch::Tensor& k_cache,
                const torch::Tensor& v_cache,
                const torch::Tensor& cu_seqlens_q,
                const torch::Tensor& seqused_k,
                const torch::Tensor& page_table,
                const torch::Tensor& scheduler_metadata,
                int64_t max_seqlen_q,
                int64_t window_left,
                int64_t window_right,
                double sm_scale,
                torch::Tensor& output,
                torch::Tensor& output_lse) {
  CHECK(scheduler_metadata.defined() && scheduler_metadata.numel() >= 3)
      << "fa3_decode: scheduler_metadata must be precomputed (size >= 3*B)";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK(page_table.defined() && page_table.scalar_type() == torch::kInt32);

  const std::string uri = fa3_fwd_uri();
  MusaTvmffiStreamGuard stream_guard(query.device());

  const int64_t b = scheduler_metadata.numel() / 4;
  CHECK_GT(b, 0) << "fa3_decode: scheduler_metadata size must be 4*batch_size";
  auto num_splits_dynamic = scheduler_metadata.slice(0, 0, b);
  auto batch_table = scheduler_metadata.slice(0, b, 2 * b);
  auto num_m_blocks = scheduler_metadata.slice(0, 2 * b, 3 * b);

  get_function(uri, uri)(to_ffi_tensor(query),
                         to_ffi_tensor(k_cache),
                         to_ffi_tensor(v_cache),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         to_ffi_tensor(seqused_k),
                         ffi::Optional<int64_t>(max_seqlen_q),
                         none_int(),
                         to_ffi_tensor(page_table),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         none_tensor(),
                         sm_scale,
                         /*is_causal=*/true,
                         window_left,
                         window_right,
                         /*attention_chunk=*/static_cast<int64_t>(0),
                         /*softcap=*/0.0,
                         /*mp_margin=*/static_cast<int64_t>(0),
                         /*num_splits=*/static_cast<int64_t>(0),
                         to_ffi_tensor(num_splits_dynamic),
                         to_ffi_tensor(batch_table),
                         to_ffi_tensor(num_m_blocks),
                         none_tensor(),
                         to_ffi_tensor(output),
                         to_ffi_tensor(output_lse),
                         /*cp_world_size=*/static_cast<int64_t>(1),
                         /*cp_rank=*/static_cast<int64_t>(0),
                         none_tensor());
}

void batch_decode(const std::string& uri,
                  ffi::Array<int64_t> plan_info,
                  torch::Tensor float_workspace_buffer,
                  torch::Tensor int_workspace_buffer,
                  torch::Tensor page_locked_int_workspace_buffer,
                  torch::Tensor query,
                  torch::Tensor k_cache,
                  torch::Tensor v_cache,
                  torch::Tensor paged_kv_indptr,
                  torch::Tensor paged_kv_indices,
                  torch::Tensor paged_kv_last_page_len,
                  int64_t window_left,
                  double sm_scale,
                  torch::Tensor output,
                  std::optional<torch::Tensor>& output_lse,
                  bool use_tensor_core,
                  std::optional<torch::Tensor> qo_indptr,
                  const torch::Tensor& paged_kv_indptr_host,
                  const torch::Tensor& paged_kv_indices_host,
                  const torch::Tensor& paged_kv_last_page_len_host) {
  (void)use_tensor_core;
  {
    VLOG(kGraphExecutorLogVerboseLevel) << "plan_info: " << plan_info;

    (void)paged_kv_indptr_host;
    (void)paged_kv_indices_host;
    (void)paged_kv_last_page_len_host;

    MusaTvmffiStreamGuard stream_guard(query.device());
    get_function(uri, "run")(
        to_ffi_tensor(float_workspace_buffer),
        to_ffi_tensor(int_workspace_buffer),
        plan_info,
        to_ffi_tensor(query),
        to_ffi_tensor(k_cache),
        to_ffi_tensor(v_cache),
        to_ffi_tensor(paged_kv_indptr),
        to_ffi_tensor(paged_kv_indices),
        to_ffi_tensor(paged_kv_last_page_len),
        to_ffi_tensor(output),
        output_lse.has_value() ? to_ffi_tensor(output_lse.value())
                               : ffi::Optional<ffi::Tensor>(),
        /*kv_layout_code=*/0,
        window_left,
        support_pdl(),
        /*maybe_alibi_slopes=*/ffi::Optional<ffi::Tensor>(),
        /*logits_soft_cap=*/0.0,
        sm_scale,
        /*rope_rcp_scale=*/1.0,
        /*rope_rcp_theta=*/1.0 / 10000.0);
  }
}

}  // namespace xllm::kernel::cuda
