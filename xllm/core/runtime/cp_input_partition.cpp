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

#include "runtime/cp_input_partition.h"

#include <glog/logging.h>

#include <algorithm>
#include <cstdint>
#include <numeric>
#include <vector>

namespace xllm::cp {

namespace {

torch::TensorOptions cpu_int32_options() {
  return torch::TensorOptions()
      .dtype(torch::kInt)
      .device(torch::kCPU)
      .pinned_memory(true);
}

std::vector<int32_t> to_seq_lens(const std::vector<int32_t>& lens,
                                 int32_t num_sequences) {
  if (lens.empty()) {
    return std::vector<int32_t>(num_sequences, 0);
  }
  const bool is_cumsum =
      lens.size() == static_cast<size_t>(num_sequences + 1) &&
      lens.front() == 0;
  std::vector<int32_t> seq_lens;
  seq_lens.reserve(num_sequences);
  if (is_cumsum) {
    for (int32_t i = 0; i < num_sequences; ++i) {
      seq_lens.push_back(std::max(0, lens[i + 1] - lens[i]));
    }
  } else {
    CHECK_GE(lens.size(), static_cast<size_t>(num_sequences));
    for (int32_t i = 0; i < num_sequences; ++i) {
      seq_lens.push_back(std::max(0, lens[i]));
    }
  }
  return seq_lens;
}

std::vector<int32_t> build_seq_lens(const std::vector<int32_t>& original,
                                    const std::vector<int32_t>& lengths,
                                    int32_t num_sequences) {
  const bool is_cumsum =
      original.size() == static_cast<size_t>(num_sequences + 1) &&
      !original.empty() && original.front() == 0;
  std::vector<int32_t> result;
  if (is_cumsum) {
    result.reserve(num_sequences + 1);
    result.push_back(0);
    for (const int32_t len : lengths) {
      result.push_back(result.back() + len);
    }
  } else {
    result.assign(lengths.begin(), lengths.end());
  }
  return result;
}

torch::Tensor gather_token_level_tensor(const torch::Tensor& src,
                                        const torch::Tensor& gather_indices,
                                        int64_t expected_token_num) {
  if (!src.defined() || src.numel() == 0) {
    return src;
  }
  if (src.numel() != expected_token_num) {
    return src;
  }
  return src.index_select(0, gather_indices);
}

// CP partition uses CPU gather indices and host-side seq-len vectors. Inputs
// that already went through ForwardInput::to(NPU) (e.g. MTP prefill with
// device_tensors_ready) must be brought back to CPU here; otherwise
// index_select triggers "Expected NPU tensor" on torch_npu.
void ensure_cpu_for_cp_partition(ForwardInput& input) {
  auto to_cpu_if_needed = [](const char* /*name*/, torch::Tensor& tensor) {
    if (tensor.defined() && !tensor.device().is_cpu()) {
      tensor = tensor.to(torch::kCPU);
    }
  };

  to_cpu_if_needed("token_ids", input.token_ids);
  to_cpu_if_needed("positions", input.positions);
  to_cpu_if_needed("mtp_shifted_token_ids",
                   input.input_params.embedding.mtp_shifted_token_ids);
  to_cpu_if_needed("selected_token_idxes",
                   input.sampling_params.selected_token_idxes);

  auto& attn_dev = input.input_params.attention.device;
  to_cpu_if_needed("attention.device.q_seq_lens", attn_dev.q_seq_lens);
  to_cpu_if_needed("attention.device.kv_seq_lens", attn_dev.kv_seq_lens);
  to_cpu_if_needed("attention.device.q_cu_seq_lens", attn_dev.q_cu_seq_lens);

  input.token_ids_host = input.token_ids;
  input.positions_host = input.positions;
  input.device_tensors_ready = false;
}

}  // namespace

void cp_partition_inplace(ForwardInput& input,
                          int32_t cp_rank,
                          int32_t cp_size) {
  if (cp_size <= 1) {
    return;
  }
  // MIXED (chunked prefill + decode) still runs the prefill ATB node and needs
  // per-CP-rank token slices; only pure DECODE batches skip partition.
  if (input.input_params.meta.batch_forward_type.is_decode()) {
    return;
  }
  const int32_t num_sequences = input.input_params.meta.num_sequences;
  if (num_sequences <= 0) {
    return;
  }
  if (!input.token_ids.defined() || input.token_ids.numel() == 0) {
    LOG(ERROR) << "[CP_PARTITION] cp_partition_inplace skipped: token_ids "
                  "empty/undefined cp_rank="
               << cp_rank << " cp_size=" << cp_size
               << " host_buffer_has_layout="
               << input.input_host_buffer_has_layout
               << " device_tensors_ready=" << input.device_tensors_ready
               << " host_token_ids_defined=" << input.host_token_ids().defined()
               << " host_token_ids_numel="
               << (input.host_token_ids().defined()
                       ? input.host_token_ids().numel()
                       : 0);
    return;
  }

  CHECK_GT(cp_size, 0);
  CHECK_GE(cp_rank, 0);
  CHECK_LT(cp_rank, cp_size);

  ensure_cpu_for_cp_partition(input);

  const int64_t token_num = input.token_ids.numel();
  const int32_t num_chunks = cp_size * 2;

  auto& input_params = input.input_params;
  auto& attention = input_params.attention;

  const std::vector<int32_t> input_lens =
      !attention.host.q_seq_lens.empty()
          ? to_seq_lens(attention.host.q_seq_lens, num_sequences)
          : to_seq_lens(attention.host.kv_seq_lens, num_sequences);

  std::vector<int32_t> cp_q_lens;
  cp_q_lens.reserve(num_sequences);
  std::vector<int64_t> gather_vec;
  gather_vec.reserve(token_num);
  int32_t cp_global_max_seq_len = 0;

  std::vector<int64_t> old_seq_offsets;
  old_seq_offsets.reserve(num_sequences + 1);
  old_seq_offsets.push_back(0);

  // Per-sequence chunk length used by the token gather below. The
  // selected_token_idxes remap MUST reuse the exact same chunk lengths so its
  // per-rank token layout matches the gathered / CP all-gather hidden states.
  std::vector<int64_t> seq_chunk_lens;
  seq_chunk_lens.reserve(num_sequences);

  for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
    const int32_t input_len = std::max(0, input_lens[seq_idx]);
    const int64_t seq_start = old_seq_offsets.back();
    const int64_t chunk_len =
        (input_len + num_chunks - 1) / static_cast<int64_t>(num_chunks);
    seq_chunk_lens.push_back(chunk_len);

    auto range_len = [&](int64_t local_start, int64_t local_end) -> int64_t {
      local_start = std::max<int64_t>(0, local_start);
      local_end = std::max<int64_t>(0, local_end);
      local_start = std::min<int64_t>(local_start, input_len);
      local_end = std::min<int64_t>(local_end, input_len);
      return std::max<int64_t>(0, local_end - local_start);
    };

    int64_t local_len = 0;
    auto append_range = [&](int64_t local_start, int64_t local_end) {
      const int64_t valid_len = range_len(local_start, local_end);
      if (valid_len <= 0) {
        return;
      }
      const int64_t start =
          std::max<int64_t>(0, std::min<int64_t>(local_start, input_len));
      for (int64_t i = 0; i < valid_len; ++i) {
        gather_vec.push_back(seq_start + start + i);
      }
      local_len += valid_len;
    };

    append_range(chunk_len * cp_rank, chunk_len * (cp_rank + 1));
    append_range(chunk_len * (num_chunks - 1 - cp_rank),
                 chunk_len * (num_chunks - cp_rank));

    cp_q_lens.push_back(static_cast<int32_t>(local_len));
    old_seq_offsets.push_back(seq_start + input_len);

    int64_t seq_cp_max = 0;
    for (int32_t rank = 0; rank < cp_size; ++rank) {
      const int64_t former_len =
          range_len(chunk_len * rank, chunk_len * (rank + 1));
      const int64_t latter_len = range_len(chunk_len * (num_chunks - 1 - rank),
                                           chunk_len * (num_chunks - rank));
      seq_cp_max = std::max(seq_cp_max, former_len + latter_len);
    }
    cp_global_max_seq_len =
        std::max(cp_global_max_seq_len, static_cast<int32_t>(seq_cp_max));
  }
  CHECK_EQ(old_seq_offsets.back(), token_num);

  auto gather_indices = torch::tensor(
      gather_vec,
      torch::TensorOptions().dtype(torch::kInt64).device(torch::kCPU));

  input.token_ids =
      gather_token_level_tensor(input.token_ids, gather_indices, token_num);
  input.positions =
      gather_token_level_tensor(input.positions, gather_indices, token_num);
  input.token_ids_host = input.token_ids;
  input.positions_host = input.positions;
  input_params.embedding.mtp_shifted_token_ids = gather_token_level_tensor(
      input_params.embedding.mtp_shifted_token_ids, gather_indices, token_num);

  const std::vector<int32_t> new_q_lens =
      build_seq_lens(attention.host.q_seq_lens, cp_q_lens, num_sequences);
  const std::vector<int32_t> new_kv_lens =
      build_seq_lens(attention.host.kv_seq_lens, cp_q_lens, num_sequences);

  attention.host.q_seq_lens = new_q_lens;
  attention.host.kv_seq_lens = new_kv_lens;
  attention.device.q_seq_lens = torch::tensor(new_q_lens, cpu_int32_options());
  attention.device.kv_seq_lens =
      torch::tensor(new_kv_lens, cpu_int32_options());

  std::vector<int32_t> cu;
  cu.reserve(cp_q_lens.size());
  std::partial_sum(cp_q_lens.begin(), cp_q_lens.end(), std::back_inserter(cu));
  attention.host.q_cu_seq_lens = cu;
  attention.device.q_cu_seq_lens = torch::tensor(cu, cpu_int32_options());

  input_params.meta.q_max_seq_len = cp_global_max_seq_len;
  input_params.meta.kv_max_seq_len = cp_global_max_seq_len;

  auto& selected = input.sampling_params.selected_token_idxes;
  if (selected.defined() && selected.numel() > 0) {
    auto selected_cpu = selected.to(torch::kCPU).to(torch::kInt32).contiguous();
    const int32_t* selected_data = selected_cpu.data_ptr<int32_t>();
    const int64_t selected_num = selected_cpu.numel();

    const int64_t num_chunks_i64 = static_cast<int64_t>(cp_size) * 2;

    // Build the per-rank token layout from `seq_chunk_lens` (the padded
    // q_seq_len based chunk sizes used by the token gather), NOT from the
    // selected token positions. Deriving chunk_len from the selected positions
    // assigned chunk_len=1 to sequences without a selected token, which made
    // token_num_per_rank / seq_prefix_per_rank diverge from the gathered and
    // CP all-gather hidden-states layout and produced out-of-range
    // selected_token_idxes in the draft LmHead gather.
    std::vector<int64_t> seq_prefix_per_rank(num_sequences, 0);
    int64_t token_num_per_rank = 0;
    for (int32_t seq_idx = 0; seq_idx < num_sequences; ++seq_idx) {
      seq_prefix_per_rank[seq_idx] = token_num_per_rank;
      token_num_per_rank +=
          (seq_chunk_lens[seq_idx] * num_chunks_i64) / cp_size;
    }

    std::vector<int32_t> remapped;
    remapped.reserve(selected_num);
    for (int64_t i = 0; i < selected_num; ++i) {
      const int64_t old_idx = static_cast<int64_t>(selected_data[i]);
      auto upper = std::upper_bound(
          old_seq_offsets.begin(), old_seq_offsets.end(), old_idx);
      int64_t seq_idx =
          static_cast<int64_t>(upper - old_seq_offsets.begin()) - 1;
      seq_idx = std::max<int64_t>(
          0,
          std::min<int64_t>(seq_idx, static_cast<int64_t>(num_sequences) - 1));

      const int64_t seq_start = old_seq_offsets[seq_idx];
      const int64_t seq_len =
          std::max<int64_t>(0, old_seq_offsets[seq_idx + 1] - seq_start);
      const int64_t chunk_len = std::max<int64_t>(1, seq_chunk_lens[seq_idx]);

      int64_t token_pos = old_idx - seq_start;
      token_pos =
          std::max<int64_t>(0, std::min<int64_t>(token_pos, seq_len - 1));
      const int64_t chunk_id = token_pos / chunk_len;
      const int64_t offset = token_pos % chunk_len;
      const int64_t rank_id =
          chunk_id >= cp_size ? static_cast<int64_t>(2 * cp_size) - chunk_id - 1
                              : chunk_id;
      const int64_t remap_idx = token_num_per_rank * rank_id +
                                seq_prefix_per_rank[seq_idx] +
                                (chunk_id / cp_size) * chunk_len + offset;
      remapped.push_back(static_cast<int32_t>(remap_idx));
    }

    input.sampling_params.selected_token_idxes =
        torch::tensor(remapped, cpu_int32_options());
  }
}

}  // namespace xllm::cp
