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
#include <cstdlib>
#include <optional>
#include <string>

#include "core/common/global_flags.h"
#include "core/kernels/musa/musa_ops_api.h"
#include "core/kernels/musa/musa_tvmffi_stream.h"

namespace xllm::kernel::musa {

namespace {

constexpr const char* kFa3MetadataUriGqa6 =
    "fmha_get_metadata_6x1_ragged_q_padded_k_causal_packgqa";
constexpr const char* kFa3MetadataUriGqa8 =
    "fmha_get_metadata_8x1_ragged_q_padded_k_causal_packgqa";

constexpr int32_t kFa3TileM = 32;
constexpr int32_t kFa3TileN = 64;
constexpr int32_t kFa3Gqa6MultiBatchDynamicSplitMinKvLength = 8192;

// Paged-KV FA3 (decode / with_kvcache). head_dim=256, GQA=6/8,
// packgqa+metadata.
constexpr const char* kFa3FwdUriHashGqa6 =
    "9e4f4b2e6574a7a45a93fef39cf9b0485651e39052d9dfd88c2e1439137a9374";
constexpr const char* kFa3FwdUriHashGqa8 =
    "94150355c74bdc57b0ec3f0a18926ec238aa401b7a6506ec460120ca8726277b";
constexpr const char* kFa3FwdCombine16Uri =
    "fmha_fwd_combine_bf16_16x64x16_ragged_q_metadata";
constexpr const char* kFa3FwdCombine32Uri =
    "fmha_fwd_combine_bf16_16x64x32_ragged_q_metadata";
constexpr const char* kFa3FwdCombine64Uri =
    "fmha_fwd_combine_bf16_16x64x64_ragged_q_metadata";

// Dense ragged FA3 prefill (Mate flash_attn_varlen). bf16, head_dim=256,
// GQA ratio=6, causal, no packgqa/metadata. Built into FLASHINFER_OPS_PATH.
constexpr const char* kFa3PrefillFwdUriHashGqa6 =
    "7ee83f6c1e99c1e66180d62c666ae3683127d3e048aeda12e77ee4569f9912c9";
// Qwen3.5-35B-A3B uses GQA=8. This artifact is the M=192, pack_gqa=false
// configuration selected by Mate for ordinary dense prefill. The older
// 556a1a artifact is the decode/pack-GQA M=32 configuration and is roughly
// 3-4x slower for a 20k-token ragged prefill.
constexpr const char* kFa3PrefillFwdUriHashGqa8 =
    "f950a279e338c0aa62c4d285c73cbedc8da55a148c172855ea03b6c08978d029";

const char* fa3_metadata_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 metadata supports GQA ratios 6 and 8, got " << gqa_ratio;
  return gqa_ratio == 6 ? kFa3MetadataUriGqa6 : kFa3MetadataUriGqa8;
}

std::string fa3_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 decode supports GQA ratios 6 and 8, got " << gqa_ratio;
  const char* hash = gqa_ratio == 6 ? kFa3FwdUriHashGqa6 : kFa3FwdUriHashGqa8;
  return std::string("fmha_fwd_") + hash;
}

std::string fa3_prefill_fwd_uri(int64_t gqa_ratio) {
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 prefill supports GQA ratios 6 and 8, got " << gqa_ratio;
  const char* hash =
      gqa_ratio == 6 ? kFa3PrefillFwdUriHashGqa6 : kFa3PrefillFwdUriHashGqa8;
  return std::string("fmha_fwd_") + hash;
}

bool enable_fa3_gqa6_dynamic_split() {
  static const bool enabled = [] {
    const char* env = std::getenv("XLLM_FA3_GQA6_DYNAMIC_SPLIT");
    return env == nullptr || std::string(env) != "0";
  }();
  return enabled;
}

int64_t fa3_gqa6_num_splits(int32_t batch_size, int32_t max_seqlen_k) {
  if (!enable_fa3_gqa6_dynamic_split()) {
    return 1;
  }

  // Split-K exposes useful parallelism for isolated decode. With a larger
  // batch, the heads and sequences already fill the device; defer the extra
  // reduction until the KV length is large enough to amortize it.
  if (batch_size <= 2 ||
      max_seqlen_k >= kFa3Gqa6MultiBatchDynamicSplitMinKvLength) {
    return 0;
  }
  return 1;
}

ffi::Optional<ffi::Tensor> none_tensor() {
  return ffi::Optional<ffi::Tensor>();
}

ffi::Optional<int64_t> none_int() { return ffi::Optional<int64_t>(); }

void combine_fa3_split_k(const ffi::Any& fwd_result,
                         const torch::Tensor& cu_seqlens_q,
                         int64_t max_seqlen_q,
                         const torch::Tensor& scheduler_metadata,
                         torch::Tensor& output,
                         torch::Tensor& output_lse) {
  auto result_values = fwd_result.cast<ffi::Array<ffi::Any>>();
  CHECK_EQ(result_values.size(), 2)
      << "FA3 forward must return [accumulators, num_splits]";
  auto accumulators = result_values[0].cast<ffi::Array<ffi::Tensor>>();
  CHECK_EQ(accumulators.size(), 2)
      << "FA3 forward must return output and LSE accumulators";
  const int64_t num_splits = result_values[1].cast<int64_t>();
  CHECK_GT(num_splits, 0) << "FA3 forward returned invalid split count";
  CHECK_LE(num_splits, 64)
      << "deployed FA3 combine artifacts support at most 64 splits";
  const char* combine_uri =
      num_splits <= 16
          ? kFa3FwdCombine16Uri
          : (num_splits <= 32 ? kFa3FwdCombine32Uri : kFa3FwdCombine64Uri);

  get_function(combine_uri, combine_uri)(to_ffi_tensor(cu_seqlens_q),
                                         none_tensor(),
                                         ffi::Optional<int64_t>(max_seqlen_q),
                                         to_ffi_tensor(output),
                                         to_ffi_tensor(output_lse),
                                         accumulators[0],
                                         accumulators[1],
                                         to_ffi_tensor(scheduler_metadata),
                                         num_splits);
}

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
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);

  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::empty({static_cast<int64_t>(batch_size) * 4}, options);

  TvmffiStreamGuard stream_guard(device);

  const int64_t b = batch_size;
  auto num_splits_dynamic = metadata.slice(/*dim=*/0, /*start=*/0, /*end=*/b);
  auto batch_table = metadata.slice(/*dim=*/0, /*start=*/b, /*end=*/2 * b);
  auto num_m_blocks = metadata.slice(/*dim=*/0, /*start=*/2 * b, /*end=*/3 * b);
  auto num_nheads_in_l2 =
      metadata.slice(/*dim=*/0, /*start=*/3 * b, /*end=*/4 * b);

  const int64_t gqa_ratio = num_heads_q / num_heads_kv;
  const std::string uri = fa3_metadata_uri(gqa_ratio);

  // The GQA=8 kernel comes from the current Mate cache. Its
  // output contract is one contiguous [4 * batch] metadata tensor. The older
  // GQA=6 artifact below predates that ABI and accepts four output views.
  if (gqa_ratio == 8) {
    get_function(uri, uri)(
        static_cast<int64_t>(batch_size),
        static_cast<int64_t>(num_heads_q),
        static_cast<int64_t>(num_heads_kv),
        static_cast<int64_t>(head_dim_qk),
        static_cast<int64_t>(head_dim_vo),
        static_cast<int64_t>(max_seqlen_q),
        static_cast<int64_t>(max_seqlen_k),
        /*max_seqlen_k_new=*/static_cast<int64_t>(0),
        to_ffi_tensor(cu_seqlens_q),
        none_tensor(),
        none_tensor(),
        to_ffi_tensor(seqused_k),
        none_tensor(),
        static_cast<int64_t>(window_size_left),
        static_cast<int64_t>(window_size_right),
        none_tensor(),
        to_ffi_tensor(metadata),
        // FA3 decode policy: let Mate choose the split count
        // from the current KV length. A fixed single split leaves long-context
        // decode severely under-parallelized.
        /*num_splits=*/static_cast<int64_t>(0),
        static_cast<int64_t>(kFa3TileM),
        static_cast<int64_t>(kFa3TileN),
        /*mp_margin=*/static_cast<int64_t>(0));
    return metadata;
  }

  get_function(uri, uri)(
      static_cast<int64_t>(batch_size),
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
      // This older four-output metadata ABI supports the same Mate split
      // heuristic as the current contiguous ABI. Keep a runtime rollback for
      // A/B validation and deployments with different cached artifacts.
      /*num_splits=*/fa3_gqa6_num_splits(batch_size, max_seqlen_k),
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
  CHECK(scheduler_metadata.defined())
      << "fa3_decode: scheduler_metadata must be precomputed";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK(page_table.defined() && page_table.scalar_type() == torch::kInt32);
  const int64_t batch_size = seqused_k.numel();
  CHECK_GT(batch_size, 0);
  CHECK_EQ(cu_seqlens_q.numel(), batch_size + 1);
  CHECK_EQ(page_table.size(0), batch_size);
  CHECK_EQ(scheduler_metadata.numel(), batch_size * 4)
      << "fa3_decode: scheduler_metadata size must be 4*batch_size";

  CHECK_GT(k_cache.size(-2), 0);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // Match the current Mate ABI used by the GQA=8 artifact: one
  // scheduler_metadata tensor followed by the optional learnable sink.
  if (gqa_ratio == 8) {
    auto fwd_result =
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
                               to_ffi_tensor(scheduler_metadata),
                               none_tensor(),
                               to_ffi_tensor(output),
                               to_ffi_tensor(output_lse),
                               /*cp_world_size=*/static_cast<int64_t>(1),
                               /*cp_rank=*/static_cast<int64_t>(0),
                               none_tensor());

    // Metadata-enabled FA3 writes split-K partials. Keep the returned tensors
    // alive through combine; graph capture's FFI allocation replay owns their
    // backing storage across subsequent graph replays.
    combine_fa3_split_k(fwd_result,
                        cu_seqlens_q,
                        max_seqlen_q,
                        scheduler_metadata,
                        output,
                        output_lse);
    return;
  }

  const int64_t b = batch_size;
  auto num_splits_dynamic = scheduler_metadata.slice(0, 0, b);
  auto batch_table = scheduler_metadata.slice(0, b, 2 * b);
  auto num_m_blocks = scheduler_metadata.slice(0, 2 * b, 3 * b);

  auto fwd_result =
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

  // The older GQA=6 forward ABI exposes scheduler metadata as tensor views,
  // but its split-K result contract matches GQA=8.
  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
}

void fa3_prefill(const torch::Tensor& query,
                 const torch::Tensor& key,
                 const torch::Tensor& value,
                 const torch::Tensor& cu_seqlens_q,
                 const torch::Tensor& cu_seqlens_k,
                 int64_t max_seqlen_q,
                 int64_t max_seqlen_k,
                 int64_t window_left,
                 int64_t window_right,
                 double sm_scale,
                 torch::Tensor& output,
                 torch::Tensor& output_lse) {
  CHECK(query.defined() && key.defined() && value.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3) << "fa3_prefill: query must be [tokens, heads, dim]";
  CHECK_EQ(key.dim(), 3) << "fa3_prefill: key must be [tokens, heads, dim]";
  CHECK_EQ(value.dim(), 3) << "fa3_prefill: value must be [tokens, heads, dim]";
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(cu_seqlens_k.defined() && cu_seqlens_k.scalar_type() == torch::kInt32);
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: only bf16 is supported for the cached dense FA3 URI";
  CHECK_EQ(key.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: key must be bf16";
  CHECK_EQ(value.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: value must be bf16";
  CHECK_EQ(output.scalar_type(), torch::kBFloat16)
      << "fa3_prefill: output must be bf16";
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32)
      << "fa3_prefill: output_lse must be fp32";
  CHECK(query.is_contiguous() && key.is_contiguous() && value.is_contiguous())
      << "fa3_prefill: query, key, and value must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k.is_contiguous())
      << "fa3_prefill: cumulative sequence lengths must be contiguous";
  CHECK(output.is_contiguous() && output_lse.is_contiguous())
      << "fa3_prefill: output tensors must be contiguous";
  CHECK_EQ(query.size(-1), 256) << "fa3_prefill: head_dim must be 256";
  CHECK_EQ(key.size(-1), 256) << "fa3_prefill: key head_dim must be 256";
  CHECK_EQ(value.sizes(), key.sizes())
      << "fa3_prefill: key and value shapes must match";
  CHECK_GT(key.size(-2), 0);
  CHECK_EQ(query.size(-2) % key.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / key.size(-2);
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "fa3_prefill: requires GQA ratio 6 or 8 (nq=" << query.size(-2)
      << ", nkv=" << key.size(-2) << ")";
  CHECK_EQ(output.sizes(), query.sizes())
      << "fa3_prefill: output shape must match query";
  CHECK_EQ(output_lse.dim(), 2)
      << "fa3_prefill: output_lse must be [heads, tokens]";
  CHECK_EQ(output_lse.size(0), query.size(1))
      << "fa3_prefill: output_lse head count must match query";
  CHECK_GE(output_lse.numel(), query.size(0) * query.size(1))
      << "fa3_prefill: output_lse is too small";
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  const std::string uri = fa3_prefill_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // Arg order matches mate jit `_fmha_fwd` / flash_attn_varlen_func mutlass
  // path for dense ragged Q/K (has_metadata=False).
  get_function(uri, uri)(to_ffi_tensor(query),
                         to_ffi_tensor(key),
                         to_ffi_tensor(value),
                         none_tensor(),  // k_new
                         none_tensor(),  // v_new
                         none_tensor(),  // q_v
                         to_ffi_tensor(cu_seqlens_q),
                         to_ffi_tensor(cu_seqlens_k),
                         none_tensor(),  // cu_seqlens_k_new
                         none_tensor(),  // seqused_q
                         none_tensor(),  // seqused_k
                         max_seqlen_q,
                         max_seqlen_k,
                         none_tensor(),  // page_table
                         none_tensor(),  // kv_batch_idx
                         none_tensor(),  // leftpad_k
                         none_tensor(),  // rotary_cos
                         none_tensor(),  // rotary_sin
                         none_tensor(),  // seqlens_rotary
                         none_tensor(),  // q_descale
                         none_tensor(),  // k_descale
                         none_tensor(),  // v_descale
                         sm_scale,
                         /*is_causal=*/true,
                         window_left,
                         window_right,
                         /*attention_chunk=*/static_cast<int64_t>(0),
                         /*softcap=*/0.0,
                         /*mp_margin=*/static_cast<int64_t>(0),
                         /*num_splits=*/static_cast<int64_t>(0),
                         none_tensor(),  // scheduler_metadata
                         none_tensor(),  // learnable_sink
                         to_ffi_tensor(output),
                         to_ffi_tensor(output_lse),
                         /*cp_world_size=*/static_cast<int64_t>(1),
                         /*cp_rank=*/static_cast<int64_t>(0),
                         none_tensor());
}

torch::Tensor fa3_prefill_scheduler_metadata(
    const torch::Device& device,
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
    const torch::Tensor& cu_seqlens_k_new,
    const torch::Tensor& seqused_k) {
  CHECK_GT(batch_size, 0);
  CHECK_GT(num_heads_kv, 0);
  CHECK_EQ(num_heads_q % num_heads_kv, 0);
  CHECK(cu_seqlens_q.defined() && cu_seqlens_q.scalar_type() == torch::kInt32);
  CHECK(cu_seqlens_k_new.defined() &&
        cu_seqlens_k_new.scalar_type() == torch::kInt32);
  CHECK(seqused_k.defined() && seqused_k.scalar_type() == torch::kInt32);
  CHECK_EQ(cu_seqlens_q.numel(), static_cast<int64_t>(batch_size) + 1);
  CHECK_EQ(cu_seqlens_k_new.numel(), static_cast<int64_t>(batch_size) + 1);
  CHECK_EQ(seqused_k.numel(), static_cast<int64_t>(batch_size));
  CHECK_GT(max_seqlen_q, 0);
  CHECK_GT(max_seqlen_k, 0);

  const int64_t gqa_ratio = num_heads_q / num_heads_kv;
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "MUSA FA3 prefill metadata supports GQA ratios 6 and 8, got "
      << gqa_ratio;
  auto options = torch::TensorOptions().dtype(torch::kInt32).device(device);
  torch::Tensor metadata =
      torch::empty({static_cast<int64_t>(batch_size) * 4}, options);

  TvmffiStreamGuard stream_guard(device);
  const std::string uri = fa3_metadata_uri(gqa_ratio);
  const int64_t b = batch_size;
  if (gqa_ratio == 8) {
    // The current GQA=8 Mate artifact returns one contiguous [4*B] tensor.
    get_function(uri, uri)(static_cast<int64_t>(batch_size),
                           static_cast<int64_t>(num_heads_q),
                           static_cast<int64_t>(num_heads_kv),
                           static_cast<int64_t>(head_dim_qk),
                           static_cast<int64_t>(head_dim_vo),
                           static_cast<int64_t>(max_seqlen_q),
                           static_cast<int64_t>(max_seqlen_k),
                           /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                           to_ffi_tensor(cu_seqlens_q),
                           none_tensor(),
                           to_ffi_tensor(cu_seqlens_k_new),
                           to_ffi_tensor(seqused_k),
                           none_tensor(),
                           static_cast<int64_t>(window_size_left),
                           static_cast<int64_t>(window_size_right),
                           none_tensor(),
                           to_ffi_tensor(metadata),
                           /*num_splits=*/static_cast<int64_t>(1),
                           static_cast<int64_t>(kFa3TileM),
                           static_cast<int64_t>(kFa3TileN),
                           /*mp_margin=*/static_cast<int64_t>(0));
    return metadata;
  }

  // Keep compatibility with the older GQA=6 artifact, whose output ABI is
  // four separate views rather than one packed tensor.
  auto num_splits_dynamic = metadata.slice(0, 0, b);
  auto batch_table = metadata.slice(0, b, 2 * b);
  auto num_m_blocks = metadata.slice(0, 2 * b, 3 * b);
  auto num_nheads_in_l2 = metadata.slice(0, 3 * b, 4 * b);
  get_function(uri, uri)(static_cast<int64_t>(batch_size),
                         static_cast<int64_t>(num_heads_q),
                         static_cast<int64_t>(num_heads_kv),
                         static_cast<int64_t>(head_dim_qk),
                         static_cast<int64_t>(head_dim_vo),
                         static_cast<int64_t>(max_seqlen_q),
                         static_cast<int64_t>(max_seqlen_k),
                         /*max_seqlen_k_new=*/static_cast<int64_t>(0),
                         to_ffi_tensor(cu_seqlens_q),
                         none_tensor(),
                         to_ffi_tensor(cu_seqlens_k_new),
                         to_ffi_tensor(seqused_k),
                         none_tensor(),
                         static_cast<int64_t>(window_size_left),
                         static_cast<int64_t>(window_size_right),
                         none_tensor(),
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

void fa3_prefill_paged(const torch::Tensor& query,
                       const torch::Tensor& k_cache,
                       const torch::Tensor& v_cache,
                       const torch::Tensor& cu_seqlens_q,
                       const torch::Tensor& cu_seqlens_k_new,
                       const torch::Tensor& seqused_k,
                       const torch::Tensor& page_table,
                       const torch::Tensor& scheduler_metadata,
                       int64_t max_seqlen_q,
                       int64_t window_left,
                       int64_t window_right,
                       double sm_scale,
                       torch::Tensor& output,
                       torch::Tensor& output_lse) {
  CHECK(query.defined() && k_cache.defined() && v_cache.defined());
  CHECK(cu_seqlens_q.defined() && cu_seqlens_k_new.defined());
  CHECK(seqused_k.defined() && page_table.defined());
  CHECK(scheduler_metadata.defined());
  CHECK(output.defined() && output_lse.defined());
  CHECK_EQ(query.dim(), 3)
      << "fa3_prefill_paged: query must be [tokens, heads, dim]";
  CHECK_EQ(k_cache.dim(), 4)
      << "fa3_prefill_paged: k_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(v_cache.dim(), 4)
      << "fa3_prefill_paged: v_cache must be [blocks, page, heads, dim]";
  CHECK_EQ(k_cache.sizes(), v_cache.sizes())
      << "fa3_prefill_paged: key/value cache shapes must match";
  CHECK_EQ(query.scalar_type(), torch::kBFloat16)
      << "fa3_prefill_paged: only bf16 is supported";
  CHECK_EQ(k_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(v_cache.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output.scalar_type(), torch::kBFloat16);
  CHECK_EQ(output_lse.scalar_type(), torch::kFloat32);
  CHECK(query.is_contiguous() && k_cache.is_contiguous() &&
        v_cache.is_contiguous() && output.is_contiguous() &&
        output_lse.is_contiguous())
      << "fa3_prefill_paged: tensors must be contiguous";
  CHECK(cu_seqlens_q.is_contiguous() && cu_seqlens_k_new.is_contiguous() &&
        seqused_k.is_contiguous() && page_table.is_contiguous() &&
        scheduler_metadata.is_contiguous())
      << "fa3_prefill_paged: metadata tensors must be contiguous";
  CHECK_EQ(cu_seqlens_q.scalar_type(), torch::kInt32);
  CHECK_EQ(cu_seqlens_k_new.scalar_type(), torch::kInt32);
  CHECK_EQ(seqused_k.scalar_type(), torch::kInt32);
  CHECK_EQ(page_table.scalar_type(), torch::kInt32);
  CHECK_EQ(scheduler_metadata.scalar_type(), torch::kInt32);
  CHECK_EQ(query.size(-1), 256);
  CHECK_EQ(k_cache.size(-1), 256);
  CHECK_EQ(v_cache.size(-1), 256);
  CHECK_EQ(query.size(-2) % k_cache.size(-2), 0);
  const int64_t gqa_ratio = query.size(-2) / k_cache.size(-2);
  CHECK(gqa_ratio == 6 || gqa_ratio == 8)
      << "fa3_prefill_paged: requires GQA ratio 6 or 8, got " << gqa_ratio;
  CHECK_EQ(page_table.size(0), seqused_k.numel());
  CHECK_EQ(scheduler_metadata.numel(), seqused_k.numel() * 4);
  CHECK_GT(max_seqlen_q, 0);

  const std::string uri = fa3_fwd_uri(gqa_ratio);
  TvmffiStreamGuard stream_guard(query.device());

  // This is the Mate ABI used by MUSA FA3
  // flash_attn_with_kvcache call.  The KV values are already in
  // k_cache/v_cache; cu_seqlens_k_new supplies the causal position of each
  // query token.
  auto fwd_result = get_function(uri, uri)(
      to_ffi_tensor(query),
      to_ffi_tensor(k_cache),
      to_ffi_tensor(v_cache),
      none_tensor(),  // k_new (cache was populated by reshape_paged_cache)
      none_tensor(),  // v_new
      none_tensor(),  // q_v
      to_ffi_tensor(cu_seqlens_q),
      none_tensor(),  // cu_seqlens_k
      to_ffi_tensor(cu_seqlens_k_new),
      none_tensor(),  // seqused_q
      to_ffi_tensor(seqused_k),
      static_cast<int64_t>(max_seqlen_q),
      none_int(),  // max_seqlen_k
      to_ffi_tensor(page_table),
      none_tensor(),  // kv_batch_idx
      none_tensor(),  // leftpad_k
      none_tensor(),  // rotary_cos
      none_tensor(),  // rotary_sin
      none_tensor(),  // seqlens_rotary
      none_tensor(),  // q_descale
      none_tensor(),  // k_descale
      none_tensor(),  // v_descale
      sm_scale,
      /*is_causal=*/true,
      window_left,
      window_right,
      /*attention_chunk=*/static_cast<int64_t>(0),
      /*softcap=*/0.0,
      /*mp_margin=*/static_cast<int64_t>(0),
      /*num_splits=*/static_cast<int64_t>(0),
      to_ffi_tensor(scheduler_metadata),
      none_tensor(),  // learnable_sink
      to_ffi_tensor(output),
      to_ffi_tensor(output_lse),
      /*cp_world_size=*/static_cast<int64_t>(1),
      /*cp_rank=*/static_cast<int64_t>(0),
      none_tensor());
  combine_fa3_split_k(fwd_result,
                      cu_seqlens_q,
                      max_seqlen_q,
                      scheduler_metadata,
                      output,
                      output_lse);
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

    TvmffiStreamGuard stream_guard(query.device());
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

}  // namespace xllm::kernel::musa
