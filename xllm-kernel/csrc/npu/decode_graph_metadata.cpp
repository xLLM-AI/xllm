// Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

#include <torch/library.h>
#include <torch/types.h>

#include <algorithm>
#include <cstdint>

namespace xllm_kernel {
namespace {

torch::Tensor update_decode_graph_metadata(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& slot_mapping,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& paged_kv_indptr,
    const torch::Tensor& paged_kv_indices,
    const torch::Tensor& paged_kv_last_page_len,
    torch::Tensor& dst_tokens,
    torch::Tensor& dst_positions,
    torch::Tensor& dst_slot_mapping,
    torch::Tensor& dst_kv_seq_lens,
    torch::Tensor& dst_kv_seq_lens_delta,
    torch::Tensor& dst_paged_kv_indptr,
    torch::Tensor& dst_paged_kv_indices,
    torch::Tensor& dst_paged_kv_last_page_len,
    int64_t padded_num_tokens) {
  const int64_t n = tokens.size(0);
  const int64_t p = padded_num_tokens;

  dst_tokens.slice(0, 0, n).copy_(tokens);
  dst_positions.slice(0, 0, n).copy_(positions);
  dst_slot_mapping.slice(0, 0, n).copy_(slot_mapping);
  if (p > n) {
    dst_tokens.slice(0, n, p).zero_();
    dst_positions.slice(0, n, p).zero_();
    dst_slot_mapping.slice(0, n, p).fill_(-1);
  }

  const int64_t src_len = std::min<int64_t>(kv_seq_lens.size(0), n + 1);
  dst_kv_seq_lens.slice(0, 0, src_len).copy_(kv_seq_lens.slice(0, 0, src_len));
  if (p >= n) {
    dst_kv_seq_lens.slice(0, src_len, p + 1)
        .copy_(kv_seq_lens.slice(0, src_len - 1, src_len));
  }
  dst_kv_seq_lens_delta.slice(0, 0, p).copy_(
      dst_kv_seq_lens.slice(0, 1, p + 1) - dst_kv_seq_lens.slice(0, 0, p));

  const int64_t indptr_len = std::min<int64_t>(paged_kv_indptr.size(0), n + 1);
  dst_paged_kv_indptr.slice(0, 0, indptr_len)
      .copy_(paged_kv_indptr.slice(0, 0, indptr_len));
  if (p >= n) {
    dst_paged_kv_indptr.slice(0, indptr_len, p + 1)
        .copy_(paged_kv_indptr.slice(0, indptr_len - 1, indptr_len));
  }

  dst_paged_kv_last_page_len.slice(0, 0, n).copy_(
      paged_kv_last_page_len.slice(0, 0, n));
  if (p > n) {
    dst_paged_kv_last_page_len.slice(0, n, p).fill_(1);
  }

  const int64_t num_pages = paged_kv_indices.size(0);
  dst_paged_kv_indices.slice(0, 0, num_pages).copy_(paged_kv_indices);

  return dst_tokens;
}

TORCH_LIBRARY_FRAGMENT(xllm_kernel, m) {
  m.def(
      "update_decode_graph_metadata(Tensor tokens, Tensor positions, Tensor "
      "slot_mapping, Tensor kv_seq_lens, Tensor paged_kv_indptr, Tensor "
      "paged_kv_indices, Tensor paged_kv_last_page_len, Tensor(a!) dst_tokens, "
      "Tensor(b!) dst_positions, Tensor(c!) dst_slot_mapping, Tensor(d!) "
      "dst_kv_seq_lens, Tensor(e!) dst_kv_seq_lens_delta, Tensor(f!) "
      "dst_paged_kv_indptr, Tensor(g!) dst_paged_kv_indices, Tensor(h!) "
      "dst_paged_kv_last_page_len, int padded_num_tokens) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(xllm_kernel, PrivateUse1, m) {
  m.impl("update_decode_graph_metadata",
         TORCH_FN(update_decode_graph_metadata));
}

}  // namespace
}  // namespace xllm_kernel
