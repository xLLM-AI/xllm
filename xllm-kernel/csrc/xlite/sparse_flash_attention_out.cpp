/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include <cmath>

#include "csrc/xlite/attention_common.h"
extern "C" uint32_t aclrtlaunch_gather_sparse_kv_cache_bfloat16_t(uint32_t,
                                                                  aclrtStream,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  void*,
                                                                  uint32_t,
                                                                  uint32_t,
                                                                  uint32_t,
                                                                  uint32_t,
                                                                  uint32_t,
                                                                  uint32_t);
extern "C" uint32_t aclrtlaunch_mla_v3_bfloat16_t(uint32_t,
                                                  aclrtStream,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  void*,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  uint32_t,
                                                  float);
namespace {
torch::Tensor run(const torch::Tensor& query,
                  const torch::Tensor& cache,
                  const torch::Tensor& indices,
                  const torch::Tensor& query_lengths,
                  const torch::Tensor& key_lengths,
                  const torch::Tensor& block_table,
                  const torch::Tensor& query_rope,
                  const torch::Tensor& key_rope,
                  double scale,
                  torch::Tensor output) {
  using namespace xllm_kernel::xlite;
  check_tensor(query);
  check_pair(query, cache);
  check_pair(query, query_rope);
  check_pair(query, key_rope);
  check_pair(query, output);
  CHECK(query.scalar_type() == torch::kBFloat16 && query.dim() == 3 &&
        query.size(0) == 1 && query.size(2) == 512);
  CHECK(query.size(1) > 0 && query.size(1) <= 32);
  CHECK(cache.dim() == 4 && cache.size(1) == 128 && cache.size(2) == 1 &&
        cache.size(3) == 512);
  CHECK(query_rope.dim() == 3 && query_rope.size(0) == 1 &&
        query_rope.size(1) == query.size(1) && query_rope.size(2) == 64);
  CHECK(key_rope.dim() == 4 && key_rope.size(0) == cache.size(0) &&
        key_rope.size(1) == 128 && key_rope.size(2) == 1 &&
        key_rope.size(3) == 64);
  check_metadata(indices, query, 3);
  check_metadata(query_lengths, query, 1);
  check_metadata(key_lengths, query, 1);
  check_metadata(block_table, query, 2);
  CHECK(indices.size(0) == 1 && indices.size(1) == 1 &&
        indices.size(2) == 2048);
  CHECK(query_lengths.numel() == 1 && key_lengths.numel() == 1 &&
        block_table.size(0) == 1 && block_table.size(1) > 0 &&
        block_table.size(1) <= cache.size(0));
  CHECK(output.sizes() == query.sizes() && output.is_contiguous());
  CHECK(std::isfinite(scale) && scale > 0 &&
        scale <= std::numeric_limits<float>::max());
  const c10::DeviceGuard guard(query.device());
  torch::Tensor q = query.contiguous(), k = cache.contiguous(),
                qr = query_rope.contiguous(), kr = key_rope.contiguous();
  torch::Tensor flat = indices.reshape({1, 2048});
  torch::Tensor valid = torch::logical_and(flat >= 0, flat < key_lengths[0]);
  torch::Tensor counts = valid.sum(1, false, torch::kInt32);
  torch::Tensor sorted = std::get<0>(
      torch::where(valid,
                   flat,
                   torch::full_like(flat, std::numeric_limits<int32_t>::max()))
          .sort(1));
  torch::Tensor compact =
      torch::where(sorted == std::numeric_limits<int32_t>::max(),
                   torch::zeros_like(sorted),
                   sorted);
  torch::Tensor lens = torch::ones({1}, indices.options()),
                starts = torch::zeros({1}, indices.options());
  torch::Tensor cached = counts.clamp_min(1) - 1;
  torch::Tensor dense = torch::zeros({1, 2048, 1, 512}, q.options()),
                dense_rope = torch::zeros({1, 2048, 1, 64}, q.options());
  const uint32_t cube = cores(q, true), vectors = cores(q);
  torch::Tensor qk =
      torch::empty({static_cast<int64_t>(cube) * 128 * 2, 2048}, q.options());
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(q.device().index()).stream();
  at_npu::native::OpCommand::RunOpApi(
      "XliteSparseDecode",
      [q,
       k,
       qr,
       kr,
       compact,
       lens,
       starts,
       cached,
       block_table,
       dense,
       dense_rope,
       qk,
       output,
       cube,
       vectors,
       stream,
       scale]() -> int {
        int status =
            checked_status(aclrtlaunch_gather_sparse_kv_cache_bfloat16_t(
                vectors,
                stream,
                k.data_ptr(),
                kr.data_ptr(),
                block_table.data_ptr(),
                compact.data_ptr(),
                lens.data_ptr(),
                cached.data_ptr(),
                dense.data_ptr(),
                dense_rope.data_ptr(),
                1,
                2048,
                128,
                static_cast<uint32_t>(block_table.size(1)),
                512,
                64));
        CHECK(status == 0);
        return checked_status(
            aclrtlaunch_mla_v3_bfloat16_t(cube,
                                          stream,
                                          q.data_ptr(),
                                          qr.data_ptr(),
                                          dense.data_ptr(),
                                          dense_rope.data_ptr(),
                                          qk.data_ptr(),
                                          output.data_ptr(),
                                          starts.data_ptr(),
                                          lens.data_ptr(),
                                          cached.data_ptr(),
                                          static_cast<uint32_t>(q.size(1)),
                                          64,
                                          512,
                                          1,
                                          2048,
                                          static_cast<float>(scale)));
      });
  // A fully masked query has zero output; the one-token scratch computation
  // above keeps the raw kernel's positive-length precondition satisfied.
  output.mul_(counts.gt(0).to(output.scalar_type()).reshape({1, 1, 1}));
  return output;
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_sparse_flash_attention_out(Tensor query, Tensor cache, Tensor "
      "indices, Tensor query_lengths, Tensor key_lengths, Tensor block_table, "
      "Tensor query_rope, Tensor key_rope, float scale, Tensor(a!) output) -> "
      "Tensor(a!)");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_sparse_flash_attention_out", &run);
}
