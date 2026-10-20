/* Copyright 2026 The xLLM Authors. Licensed under Apache-2.0. */

#include "csrc/xlite/attention_common.h"
using Launcher = uint32_t(uint32_t,
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
                          void*,
                          void*,
                          void*,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t,
                          uint32_t);
extern "C" Launcher aclrtlaunch_indexer_topk_bfloat16_t;
extern "C" Launcher aclrtlaunch_indexer_topk_float16_t;
namespace {
using namespace xllm_kernel::xlite;
torch::Tensor run_out(const torch::Tensor& query,
                      const torch::Tensor& key_cache,
                      const torch::Tensor& weights,
                      const torch::Tensor& query_lengths,
                      const torch::Tensor& key_lengths,
                      const torch::Tensor& block_table,
                      int64_t topk,
                      torch::Tensor output) {
  check_tensor(query);
  check_pair(query, key_cache);
  check_pair(query, weights);
  CHECK(query.dim() == 3 && query.size(0) > 0 && query.size(0) <= 128 &&
        query.size(1) == 32 && query.size(2) == 128);
  CHECK(key_cache.dim() == 4 && key_cache.size(1) == 128 &&
        key_cache.size(2) == 1 && key_cache.size(3) == 128);
  CHECK(weights.dim() == 2 && weights.size(0) == query.size(0) &&
        weights.size(1) == 32);
  check_metadata(query_lengths, query, 1);
  check_metadata(key_lengths, query, 1);
  check_metadata(block_table, query, 2);
  CHECK(query_lengths.numel() == 1 && key_lengths.numel() == 1 &&
        block_table.size(0) == 1);
  CHECK(block_table.size(1) > 0 && block_table.size(1) <= key_cache.size(0));
  CHECK(topk == 2048 && block_table.size(1) * 128 >= topk);
  check_metadata(output, query, 3);
  CHECK(output.size(0) == query.size(0) && output.size(1) == 1 &&
        output.size(2) == topk);
  const c10::DeviceGuard guard(query.device());
  torch::Tensor q = query.contiguous(), k = key_cache.contiguous();
  torch::Tensor w =
      torch::cat({torch::zeros({q.size(0), 128}, q.options()), weights}, 1)
          .contiguous();
  torch::Tensor starts = torch::zeros({1}, query_lengths.options());
  torch::Tensor cached = key_lengths - query_lengths;
  torch::Tensor ids =
      torch::arange(block_table.size(1) * 128, query_lengths.options());
  const uint32_t count = cores(q, true), vectors = cores(q);
  // Workspace constants are from pinned GVirt kernel_param.h, not runtime
  // guesses.
  torch::Tensor scores =
      torch::empty({2 * static_cast<int64_t>(count) * 128, 4096}, q.options());
  torch::Tensor last =
      torch::empty({q.size(0), 2 * topk}, query_lengths.options());
  torch::Tensor sync =
      torch::zeros({static_cast<int64_t>(vectors)}, query_lengths.options());
  const aclrtStream stream =
      c10_npu::getCurrentNPUStream(q.device().index()).stream();
  Launcher* launcher = q.scalar_type() == torch::kBFloat16
                           ? aclrtlaunch_indexer_topk_bfloat16_t
                           : aclrtlaunch_indexer_topk_float16_t;
  at_npu::native::OpCommand::RunOpApi(
      "XliteLightningIndexer",
      [q,
       k,
       w,
       starts,
       query_lengths,
       cached,
       block_table,
       ids,
       scores,
       last,
       sync,
       output,
       count,
       stream,
       launcher]() -> int {
        return checked_status(
            launcher(count,
                     stream,
                     q.data_ptr(),
                     k.data_ptr(),
                     w.data_ptr(),
                     starts.data_ptr(),
                     query_lengths.data_ptr(),
                     cached.data_ptr(),
                     block_table.data_ptr(),
                     scores.data_ptr(),
                     last.data_ptr(),
                     ids.data_ptr(),
                     output.data_ptr(),
                     sync.data_ptr(),
                     32,
                     128,
                     128,
                     1,
                     static_cast<uint32_t>(block_table.size(1)),
                     2048));
      });
  return output;
}
torch::Tensor run(const torch::Tensor& query,
                  const torch::Tensor& key_cache,
                  const torch::Tensor& weights,
                  const torch::Tensor& query_lengths,
                  const torch::Tensor& key_lengths,
                  const torch::Tensor& block_table,
                  int64_t topk) {
  CHECK(query.dim() == 3 && topk == 2048);
  torch::Tensor output = torch::empty({query.size(0), 1, topk},
                                      query.options().dtype(torch::kInt32));
  return run_out(query,
                 key_cache,
                 weights,
                 query_lengths,
                 key_lengths,
                 block_table,
                 topk,
                 output);
}
}  // namespace
TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "xlite_lightning_indexer(Tensor query, Tensor key_cache, Tensor weights, "
      "Tensor query_lengths, Tensor key_lengths, Tensor block_table, int topk) "
      "-> Tensor");
  m.def(
      "xlite_lightning_indexer_out(Tensor query, Tensor key_cache, Tensor "
      "weights, Tensor query_lengths, Tensor key_lengths, Tensor block_table, "
      "int topk, Tensor(a!) output) -> Tensor(a!)");
}
TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("xlite_lightning_indexer", &run);
  m.impl("xlite_lightning_indexer_out", &run_out);
}
