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

#include "batch_input_builder.h"

#include <c10/core/DeviceType.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <thread>
#include <utility>
#include <vector>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/platform/platform.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/request/sequence.h"
#include "framework/sampling/sampling_params.h"
#include "models/vlm/mposition/mposition.h"
#include "runtime/params_utils.h"
#include "util/blocking_counter.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/utils.h"

namespace xllm {
namespace {

// Minimum estimated total query tokens in a batch before process_sequences
// takes the multithreaded path. Below this, the fixed thread-dispatch plus
// O(total-tokens) serial-merge overhead outweighs the parallel savings, so the
// single-threaded path is faster (measured: decode batches of thousands of
// 1-token sequences and small prefill batches all regress under threading;
// only large prefill workloads above this range benefit). Chosen conservatively
// so decode always stays single-threaded while large prefill still fans out.
constexpr size_t kMultithreadTokenThreshold = 65536;

uint32_t get_sample_source_position(const SampleSlot& sample_slot) {
  if (sample_slot.token_position == 0) {
    return 0;
  }
  return static_cast<uint32_t>(sample_slot.token_position - 1);
}

void append_xtensor_offsets(TransferKVInfo* info,
                            const TransferKVInfo& full_info,
                            const std::vector<size_t>& remote_idxs) {
  if (full_info.dst_xtensor_layer_offsets.empty()) {
    return;
  }

  info->dst_xtensor_layer_offsets.reserve(
      full_info.dst_xtensor_layer_offsets.size());
  for (const XTensorLayerOffsets& full_layer :
       full_info.dst_xtensor_layer_offsets) {
    CHECK_EQ(full_layer.k_offsets.size(), full_info.remote_blocks_ids.size());
    CHECK_EQ(full_layer.v_offsets.size(), full_info.remote_blocks_ids.size());
    XTensorLayerOffsets layer;
    layer.k_offsets.reserve(remote_idxs.size());
    layer.v_offsets.reserve(remote_idxs.size());
    for (size_t remote_idx : remote_idxs) {
      CHECK_LT(remote_idx, full_layer.k_offsets.size());
      CHECK_LT(remote_idx, full_layer.v_offsets.size());
      layer.k_offsets.emplace_back(full_layer.k_offsets[remote_idx]);
      layer.v_offsets.emplace_back(full_layer.v_offsets[remote_idx]);
    }
    info->dst_xtensor_layer_offsets.emplace_back(std::move(layer));
  }
}

std::vector<int32_t> build_q_cu_seq_lens_vec(
    const std::vector<int32_t>& q_seq_lens) {
  std::vector<int32_t> q_cu_seq_lens;
  if (q_seq_lens.empty()) {
    return q_cu_seq_lens;
  }
#if defined(USE_NPU) || defined(USE_MUSA)
  q_cu_seq_lens.reserve(q_seq_lens.size());
  int32_t cum_seq_len = 0;
  for (int32_t q_len : q_seq_lens) {
    cum_seq_len += q_len;
    q_cu_seq_lens.emplace_back(cum_seq_len);
  }
#else
  CHECK(q_seq_lens.front() == 0)
      << "q_seq_lens must be cumulative with leading zero";
  q_cu_seq_lens.assign(q_seq_lens.begin() + 1, q_seq_lens.end());
#endif
  return q_cu_seq_lens;
}

struct BlockCopyKernelInputData {
  std::vector<int32_t> src_indices;
  std::vector<int32_t> dst_indices;
  std::vector<int32_t> cum_sum;
  bool has_overlap = false;
};

BlockCopyKernelInputData build_block_copy_kernel_input_data(
    const std::vector<BlockTransferInfo>& swap_blocks,
    bool detect_overlap) {
  BlockCopyKernelInputData input_data;
  if (swap_blocks.empty()) {
    return input_data;
  }

  int32_t current_src = swap_blocks[0].src_block_id;
  input_data.src_indices.reserve(swap_blocks.size());
  input_data.dst_indices.reserve(swap_blocks.size());
  input_data.cum_sum.reserve(swap_blocks.size());

  std::unordered_set<int32_t> src_set;
  std::unordered_map<int32_t, int32_t> dst_to_src;
  if (detect_overlap) {
    for (const auto& block : swap_blocks) {
      src_set.insert(block.src_block_id);
    }
  }

  input_data.src_indices.push_back(swap_blocks[0].src_block_id);
  input_data.dst_indices.push_back(swap_blocks[0].dst_block_id);
  if (detect_overlap) {
    dst_to_src.emplace(swap_blocks[0].dst_block_id,
                       swap_blocks[0].src_block_id);
    if (src_set.count(swap_blocks[0].dst_block_id) > 0 &&
        swap_blocks[0].dst_block_id != swap_blocks[0].src_block_id) {
      input_data.has_overlap = true;
    }
  }

  for (size_t i = 1; i < swap_blocks.size(); ++i) {
    input_data.dst_indices.push_back(swap_blocks[i].dst_block_id);
    if (detect_overlap) {
      auto [it, inserted] = dst_to_src.emplace(swap_blocks[i].dst_block_id,
                                               swap_blocks[i].src_block_id);
      if (!inserted && it->second != swap_blocks[i].src_block_id) {
        input_data.has_overlap = true;
      }
      if (src_set.count(swap_blocks[i].dst_block_id) > 0 &&
          swap_blocks[i].dst_block_id != swap_blocks[i].src_block_id) {
        input_data.has_overlap = true;
      }
    }
    if (swap_blocks[i].src_block_id != current_src) {
      input_data.src_indices.push_back(swap_blocks[i].src_block_id);
      input_data.cum_sum.push_back(static_cast<int32_t>(i));
      current_src = swap_blocks[i].src_block_id;
    }
  }
  input_data.cum_sum.emplace_back(static_cast<int32_t>(swap_blocks.size()));
  return input_data;
}

torch::Tensor build_pinned_int_tensor(const std::vector<int32_t>& values) {
  return torch::tensor(values,
                       torch::TensorOptions()
                           .dtype(torch::kInt)
                           .device(torch::kCPU)
                           .pinned_memory(true));
}

// Whether the current prefill step end should hold a linear-state checkpoint.
// Checkpoints are saved at prefill step ends that land on a chunk-end boundary
// (stride = max_tokens_per_chunk_for_prefill). The linear-state cache is a
// sparse per-chunk overlay: KV may cache every block boundary while
// linear-state saves only at chunk ends.
bool should_save_linear_checkpoint(Sequence* sequence,
                                   uint32_t boundary_tokens,
                                   uint32_t chunk_stride) {
  if (sequence == nullptr || !sequence->is_prefill_stage()) {
    return false;
  }
  if (boundary_tokens == 0 || chunk_stride == 0) {
    return false;
  }
  return boundary_tokens % chunk_stride == 0;
}

}  // namespace

BatchInputBuilder::BatchInputBuilder(
    const std::vector<Sequence*>& sequences,
    const std::vector<uint32_t>& allowed_max_tokens,
    const std::vector<torch::Tensor>& input_embeddings_vec,
    const std::vector<MMData>& mm_data_vec,
    std::vector<BlockTransferInfo>* swap_block_transfer_infos,
    const uint64_t batch_id,
    const ModelArgs* args,
    BatchForwardType batch_forward_type,
    int32_t cp_size,
    ThreadPool* thread_pool)
    : sequences_(sequences),
      allowed_max_tokens_(allowed_max_tokens),
      input_embeddings_vec_(input_embeddings_vec),
      mm_data_vec_(mm_data_vec),
      args_(args),
      thread_pool_(thread_pool),
      num_sequences_(sequences.size()),
      swap_block_transfer_infos_(swap_block_transfer_infos),
      batch_id_(batch_id),
      cp_size_(Platform::uses_model_cp_partition() ? 1 : std::max(1, cp_size)) {
  // Reserve space for better performance
  const size_t reserve_size = 1024;
  state_.flatten_tokens_vec.reserve(reserve_size);
  state_.flatten_positions_vec.reserve(reserve_size);
  state_.mrope_positions_vec.reserve(sequences.size());
  state_.block_tables_vec.reserve(sequences.size());
  state_.acc_logprob_vec.reserve(sequences.size());
  state_.mtp_shifted_token_ids.reserve(reserve_size);
  state_.mtp_bootstrap_embeddings.reserve(sequences.size());
  state_.mtp_bootstrap_row_idxes.reserve(sequences.size());
  if (args_ != nullptr) {
    use_mrope_ = (args_->rope_scaling_rope_type() == "mrope");
  }
  write_block_ids_.clear();
  state_.batch_forward_type = batch_forward_type;
}

TransferKVInfo BatchInputBuilder::build_step_transfer_info(
    const TransferKVInfo& full_info,
    const std::vector<uint64_t>& local_block_ids,
    size_t next_transfer_block_idx,
    uint32_t seq_len,
    uint32_t block_size,
    size_t* advanced_transfer_block_idx) {
  CHECK(advanced_transfer_block_idx != nullptr);
  *advanced_transfer_block_idx = next_transfer_block_idx;

  TransferKVInfo info;
  info.request_id = full_info.request_id;
  info.dp_rank = full_info.dp_rank;
  info.remote_instance_info = full_info.remote_instance_info;
  info.local_linear_state_ids = full_info.local_linear_state_ids;
  info.remote_linear_state_ids = full_info.remote_linear_state_ids;

  if (block_size == 0 || local_block_ids.empty()) {
    return info;
  }

  const size_t local_size = local_block_ids.size();
  const size_t remote_size = full_info.remote_blocks_ids.size();
  const size_t win_begin = next_transfer_block_idx;
  const size_t win_end =
      static_cast<size_t>(util::ceil_div(seq_len, block_size));
  const size_t map_end = std::min(win_end, local_size);
  const size_t remote_stride =
      static_cast<size_t>(util::kv_split_size_effective());
  const size_t remote_end = map_end * remote_stride;
  CHECK_GE(util::align_up(remote_size, remote_stride), remote_end)
      << "remote block coverage shortage, request_id=" << full_info.request_id
      << ", remote_size=" << remote_size << ", remote_end=" << remote_end
      << ", remote_stride=" << remote_stride;

  const size_t stable_end = static_cast<size_t>(seq_len / block_size);
  *advanced_transfer_block_idx =
      std::max(next_transfer_block_idx, std::min(stable_end, map_end));

  if (win_begin >= map_end) {
    return info;
  }

  std::vector<size_t> remote_idxs;
  const size_t block_cnt = map_end - win_begin;
  info.local_blocks_ids.reserve(block_cnt);
  info.remote_blocks_ids.reserve(block_cnt * remote_stride);
  remote_idxs.reserve(block_cnt * remote_stride);
  for (size_t local_idx = win_begin; local_idx < map_end; ++local_idx) {
    info.local_blocks_ids.emplace_back(local_block_ids[local_idx]);
    for (size_t offset = 0; offset < remote_stride; ++offset) {
      const size_t remote_idx = local_idx * remote_stride + offset;
      if (remote_idx >= full_info.remote_blocks_ids.size()) {
        if (remote_stride > 1) {
          break;
        }
        LOG(FATAL) << "Out of bound access: remote_idx=" << remote_idx
                   << ", remote_blocks_ids.size()="
                   << full_info.remote_blocks_ids.size();
      }
      info.remote_blocks_ids.emplace_back(
          full_info.remote_blocks_ids[remote_idx]);
      remote_idxs.emplace_back(remote_idx);
    }
  }

  append_xtensor_offsets(&info, full_info, remote_idxs);
  return info;
}

KVBlockTransferGroup BatchInputBuilder::build_group_step_transfer(
    const KVBlockTransferGroup& full_group,
    const std::vector<int32_t>& local_block_ids,
    size_t next_transfer_block_idx,
    uint32_t seq_len,
    uint32_t block_size,
    size_t* advanced_transfer_block_idx) {
  CHECK(advanced_transfer_block_idx != nullptr);
  *advanced_transfer_block_idx = next_transfer_block_idx;

  KVBlockTransferGroup step_group;
  step_group.group_id = full_group.group_id;
  if (block_size == 0 || local_block_ids.empty()) {
    return step_group;
  }

  const size_t local_size = local_block_ids.size();
  const size_t remote_size = full_group.remote_blocks_ids.size();
  const size_t win_begin = next_transfer_block_idx;
  const size_t win_end =
      static_cast<size_t>(util::ceil_div(seq_len, block_size));
  const size_t map_end = std::min(win_end, local_size);
  CHECK_GE(remote_size, map_end)
      << "Grouped KV remote block coverage shortage, group_id="
      << full_group.group_id << ", remote_size=" << remote_size
      << ", remote_end=" << map_end;

  const size_t stable_end = static_cast<size_t>(seq_len / block_size);
  *advanced_transfer_block_idx =
      std::max(next_transfer_block_idx, std::min(stable_end, map_end));
  if (win_begin >= map_end) {
    return step_group;
  }

  const size_t block_count = map_end - win_begin;
  step_group.local_blocks_ids.reserve(block_count);
  step_group.remote_blocks_ids.reserve(block_count);
  for (size_t local_idx = win_begin; local_idx < map_end; ++local_idx) {
    const int32_t local_block_id = local_block_ids[local_idx];
    if (local_block_id < 0) {
      continue;
    }
    step_group.local_blocks_ids.emplace_back(
        static_cast<uint64_t>(local_block_id));
    step_group.remote_blocks_ids.emplace_back(
        full_group.remote_blocks_ids[local_idx]);
  }
  return step_group;
}

ForwardInput BatchInputBuilder::build_forward_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  process_sequences();
  padding_decode_batch_size(num_decoding_tokens, min_decoding_batch_size);

  return state_to_forward_input();
}

void BatchInputBuilder::process_sequences() {
  // Multithreading only helps when the parallelized per-sequence work is large
  // enough to amortize the fixed thread-dispatch cost plus the serial merge of
  // per-thread states (which is O(total tokens)). Decode batches carry ~1 query
  // token per sequence, so even thousands of sequences stay far below that
  // break-even and run faster single-threaded. Gate on the estimated total
  // query-token workload, not just the sequence count. The estimate mirrors the
  // q_seq_len computed in process_single_sequence and uses only O(1) accessors.
  bool use_multithread = thread_pool_ && num_sequences_ >= thread_pool_->size();
  if (use_multithread) {
    size_t total_query_tokens = 0;
    for (int32_t i = 0; i < num_sequences_; ++i) {
      const Sequence* sequence = sequences_[i];
      const size_t need_compute = sequence->num_need_compute_tokens();
      total_query_tokens +=
          std::min(need_compute, static_cast<size_t>(allowed_max_tokens_[i]));
    }
    use_multithread = total_query_tokens >= kMultithreadTokenThreshold;
  }

  if (use_multithread) {
    process_sequences_multithreaded();
  } else {
    for (int32_t i = 0; i < num_sequences_; ++i) {
      process_single_sequence(i);
    }
  }
}

void BatchInputBuilder::process_sequences_multithreaded() {
  const size_t threads_num = thread_pool_->size();
  const size_t sequences_per_thread =
      (num_sequences_ + threads_num - 1) / threads_num;

  BlockingCounter counter(threads_num);

  // safe state for each thread
  std::vector<BuilderState> thread_builder_states;
  std::vector<std::unordered_set<int32_t>> thread_write_block_ids;
  thread_builder_states.resize(threads_num);
  thread_write_block_ids.resize(threads_num);

  for (auto& thread_state : thread_builder_states) {
    thread_state.batch_forward_type = state_.batch_forward_type;
    // Reserve per-thread scratch so parallel processing does not repeatedly
    // reallocate (which serializes on the allocator and erodes the speedup).
    thread_state.block_tables_vec.reserve(sequences_per_thread);
    thread_state.new_token_slot_ids.reserve(sequences_per_thread);
    thread_state.kv_cache_tokens_nums.reserve(sequences_per_thread);
#if defined(USE_NPU) || defined(USE_MUSA)
    thread_state.seq_lens.reserve(sequences_per_thread);
    thread_state.q_seq_lens.reserve(sequences_per_thread);
#endif
    thread_state.embedding_ids.reserve(sequences_per_thread);
    thread_state.linear_state_ids.reserve(sequences_per_thread);
    thread_state.request_ids.reserve(sequences_per_thread);
    thread_state.extra_token_ids.reserve(sequences_per_thread);
  }

  // parallel processing function
  auto process_sequences_range =
      [&](size_t thread_start_idx,
          size_t thread_end_idx,
          BuilderState& state,
          std::unordered_set<int32_t>& write_block_ids) {
        for (size_t i = thread_start_idx;
             i < thread_end_idx && i < static_cast<size_t>(num_sequences_);
             ++i) {
          process_single_sequence(i, &state, &write_block_ids);
        }
      };

  // Start parallel tasks
  for (size_t thread_idx = 0; thread_idx < threads_num; ++thread_idx) {
    size_t thread_start_idx = thread_idx * sequences_per_thread;
    size_t thread_end_idx = std::min(thread_start_idx + sequences_per_thread,
                                     static_cast<size_t>(num_sequences_));

    thread_pool_->schedule([process_sequences_range,
                            thread_start_idx,
                            thread_end_idx,
                            &thread_builder_states,
                            &thread_write_block_ids,
                            thread_idx,
                            &counter]() mutable {
      process_sequences_range(thread_start_idx,
                              thread_end_idx,
                              thread_builder_states[thread_idx],
                              thread_write_block_ids[thread_idx]);
      counter.decrement_count();
    });
  }

  // Wait for all tasks to complete
  counter.wait();

  // Pre-reserve the destination vectors to their exact final sizes so the
  // serial merge does a single allocation per field instead of growing
  // geometrically across thread states. The merge is O(total work) and runs
  // single-threaded, so realloc churn here directly caps the achievable
  // multithreaded speedup.
  size_t total_tokens = 0;
  size_t total_seqs = 0;
  size_t total_slots = 0;
  size_t total_paged_indices = 0;
  for (const auto& state : thread_builder_states) {
    total_tokens += state.flatten_tokens_vec.size();
    total_seqs += state.block_tables_vec.size();
    total_slots += state.new_token_slot_ids.size();
    total_paged_indices += state.paged_kv_indices.size();
  }
  state_.flatten_tokens_vec.reserve(total_tokens);
  if (!use_mrope_) {
    state_.flatten_positions_vec.reserve(total_tokens);
  } else {
    state_.mrope_positions_vec.reserve(total_seqs);
  }
  state_.block_tables_vec.reserve(total_seqs);
  state_.new_token_slot_ids.reserve(total_slots);
  state_.kv_cache_tokens_nums.reserve(total_seqs);
#if defined(USE_NPU) || defined(USE_MUSA)
  state_.seq_lens.reserve(total_seqs);
  state_.q_seq_lens.reserve(total_seqs);
#endif
  state_.embedding_ids.reserve(total_seqs);
  state_.linear_state_ids.reserve(total_seqs);
  state_.request_ids.reserve(total_seqs);
  state_.extra_token_ids.reserve(total_seqs);
  state_.paged_kv_indices.reserve(total_paged_indices);
  state_.paged_kv_indptr.reserve(total_seqs + 1);
  state_.paged_kv_last_page_len.reserve(total_seqs);

  // Merge results from all threads
  for (const auto& state : thread_builder_states) {
    state_.flatten_tokens_vec.insert(state_.flatten_tokens_vec.end(),
                                     state.flatten_tokens_vec.begin(),
                                     state.flatten_tokens_vec.end());
    if (!use_mrope_) {
      state_.flatten_positions_vec.insert(state_.flatten_positions_vec.end(),
                                          state.flatten_positions_vec.begin(),
                                          state.flatten_positions_vec.end());
    } else {
      state_.mrope_positions_vec.insert(state_.mrope_positions_vec.end(),
                                        state.mrope_positions_vec.begin(),
                                        state.mrope_positions_vec.end());
    }
    state_.block_tables_vec.insert(state_.block_tables_vec.end(),
                                   state.block_tables_vec.begin(),
                                   state.block_tables_vec.end());
    state_.acc_logprob_vec.insert(state_.acc_logprob_vec.end(),
                                  state.acc_logprob_vec.begin(),
                                  state.acc_logprob_vec.end());
    // selected_token_idxes and sample_idxes need offset
    int32_t selected_token_idxes_offset =
        static_cast<int32_t>(state_.flatten_tokens_vec.size()) -
        static_cast<int32_t>(state.flatten_tokens_vec.size());
    for (const auto& idx : state.selected_token_idxes) {
      state_.selected_token_idxes.emplace_back(idx +
                                               selected_token_idxes_offset);
    }
    state_.sampling_params.insert(state_.sampling_params.end(),
                                  state.sampling_params.begin(),
                                  state.sampling_params.end());
    int32_t sample_idxes_offset =
        static_cast<int32_t>(state_.sample_idxes.size());
    for (const auto& idx : state.sample_idxes) {
      state_.sample_idxes.emplace_back(idx + sample_idxes_offset);
    }
    state_.unique_token_ids_vec.insert(state_.unique_token_ids_vec.end(),
                                       state.unique_token_ids_vec.begin(),
                                       state.unique_token_ids_vec.end());
    state_.unique_token_counts_vec.insert(state_.unique_token_counts_vec.end(),
                                          state.unique_token_counts_vec.begin(),
                                          state.unique_token_counts_vec.end());
    state_.unique_token_lens_vec.insert(state_.unique_token_lens_vec.end(),
                                        state.unique_token_lens_vec.begin(),
                                        state.unique_token_lens_vec.end());
    state_.max_seq_len = std::max(state_.max_seq_len, state.max_seq_len);
    state_.q_max_seq_len = std::max(state_.q_max_seq_len, state.q_max_seq_len);
#if defined(USE_NPU) || defined(USE_MUSA)
    state_.seq_lens.insert(
        state_.seq_lens.end(), state.seq_lens.begin(), state.seq_lens.end());
    state_.q_seq_lens.insert(state_.q_seq_lens.end(),
                             state.q_seq_lens.begin(),
                             state.q_seq_lens.end());
    state_.kv_cache_tokens_nums.insert(state_.kv_cache_tokens_nums.end(),
                                       state.kv_cache_tokens_nums.begin(),
                                       state.kv_cache_tokens_nums.end());
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU)
    int32_t seq_len_offset = state_.seq_lens.back();
    // skip the first element which is 0
    for (size_t i = 1; i < state.seq_lens.size(); ++i) {
      state_.seq_lens.emplace_back(state.seq_lens[i] + seq_len_offset);
    }
    int32_t q_seq_len_offset = state_.q_seq_lens.back();
    for (size_t i = 1; i < state.q_seq_lens.size(); ++i) {
      state_.q_seq_lens.emplace_back(state.q_seq_lens[i] + q_seq_len_offset);
    }
#endif
    state_.new_token_slot_ids.insert(state_.new_token_slot_ids.end(),
                                     state.new_token_slot_ids.begin(),
                                     state.new_token_slot_ids.end());
    const int32_t row_offset =
        static_cast<int32_t>(state_.embedding_ids.size());
    state_.embedding_ids.insert(state_.embedding_ids.end(),
                                state.embedding_ids.begin(),
                                state.embedding_ids.end());
    state_.linear_state_ids.insert(state_.linear_state_ids.end(),
                                   state.linear_state_ids.begin(),
                                   state.linear_state_ids.end());
    state_.linear_state_cache_ops.insert(state_.linear_state_cache_ops.end(),
                                         state.linear_state_cache_ops.begin(),
                                         state.linear_state_cache_ops.end());
    state_.request_ids.insert(state_.request_ids.end(),
                              state.request_ids.begin(),
                              state.request_ids.end());
    state_.extra_token_ids.insert(state_.extra_token_ids.end(),
                                  state.extra_token_ids.begin(),
                                  state.extra_token_ids.end());
    state_.mtp_shifted_token_ids.insert(state_.mtp_shifted_token_ids.end(),
                                        state.mtp_shifted_token_ids.begin(),
                                        state.mtp_shifted_token_ids.end());
    for (int32_t row_idx : state.mtp_bootstrap_row_idxes) {
      state_.mtp_bootstrap_row_idxes.emplace_back(row_offset + row_idx);
    }
    state_.mtp_bootstrap_embeddings.insert(
        state_.mtp_bootstrap_embeddings.end(),
        state.mtp_bootstrap_embeddings.begin(),
        state.mtp_bootstrap_embeddings.end());
    state_.transfer_kv_infos.insert(state_.transfer_kv_infos.end(),
                                    state.transfer_kv_infos.begin(),
                                    state.transfer_kv_infos.end());

    // for flashinfer
    // we skip the first '0' element
    int32_t paged_kv_indptr_offset = state_.paged_kv_indptr.back();
    for (size_t i = 1; i < state.paged_kv_indptr.size(); ++i) {
      state_.paged_kv_indptr.emplace_back(state.paged_kv_indptr[i] +
                                          paged_kv_indptr_offset);
    }
    state_.paged_kv_indices.insert(state_.paged_kv_indices.end(),
                                   state.paged_kv_indices.begin(),
                                   state.paged_kv_indices.end());
    state_.paged_kv_last_page_len.insert(state_.paged_kv_last_page_len.end(),
                                         state.paged_kv_last_page_len.begin(),
                                         state.paged_kv_last_page_len.end());

    if (!state.multi_block_tables.empty()) {
      if (state_.multi_block_tables.empty()) {
        state_.multi_block_tables.resize(state.multi_block_tables.size());
      }
      CHECK_EQ(state_.multi_block_tables.size(),
               state.multi_block_tables.size())
          << "multi_block_tables manager count mismatch while merging thread "
             "states. dst_manager_num="
          << state_.multi_block_tables.size()
          << ", src_manager_num=" << state.multi_block_tables.size();
      for (size_t m = 0; m < state.multi_block_tables.size(); ++m) {
        auto& dst_mgr_tables = state_.multi_block_tables[m];
        const auto& src_mgr_tables = state.multi_block_tables[m];
        dst_mgr_tables.insert(
            dst_mgr_tables.end(), src_mgr_tables.begin(), src_mgr_tables.end());
      }
    }
  }
  for (const auto& write_block_ids : thread_write_block_ids) {
    write_block_ids_.insert(write_block_ids.begin(), write_block_ids.end());
  }
}

void BatchInputBuilder::process_single_sequence(
    int32_t seq_index,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  BuilderState& state = state_ptr ? *state_ptr : state_;

  auto* sequence = sequences_[seq_index];
  const auto token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const uint32_t n_kv_cache_tokens = sequence->kv_state().kv_cache_tokens_num();

  // Validate and calculate sequence lengths
  CHECK(allowed_max_tokens_[seq_index] > 0);
  const uint32_t q_seq_len =
      std::min(n_tokens - n_kv_cache_tokens, allowed_max_tokens_[seq_index]);
  uint32_t padded_q_seq_len = q_seq_len;
  // Continuous scheduler can enlarge token budget for CP prefill padding.
  // Keep physical q_len aligned to 2 * cp_size to match later cp_partition.
  if (cp_size_ > 1 && state.batch_forward_type.no_decode()) {
    const uint32_t aligned_q_seq_len =
        xllm::util::align_up(q_seq_len, cp_size_ * 2);
    padded_q_seq_len = aligned_q_seq_len;
  }
  const uint32_t logical_seq_len = q_seq_len + n_kv_cache_tokens;
  const uint32_t seq_len = padded_q_seq_len + n_kv_cache_tokens;

  // Validation
  CHECK_GE(sequence->kv_state().current_max_tokens_capacity(), seq_len);
  CHECK_GT(q_seq_len, 0) << "at least one token should be processed. "
                         << "n_tokens: " << n_tokens
                         << ", n_kv_cache_tokens: " << n_kv_cache_tokens
                         << ", current_max_tokens_capacity: "
                         << sequence->kv_state().current_max_tokens_capacity()
                         << ", allowed_max_tokens: "
                         << allowed_max_tokens_[seq_index];

  // Update state
  state.max_seq_len = std::max(state.max_seq_len, seq_len);
  state.q_max_seq_len = std::max(state.q_max_seq_len, padded_q_seq_len);
  state.kv_cache_tokens_nums.emplace_back(n_kv_cache_tokens);
#if defined(USE_NPU)
  state.seq_lens.push_back(seq_len);
  state.q_seq_lens.push_back(padded_q_seq_len);
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU)
  state.seq_lens.push_back(state.seq_lens.back() + seq_len);
  state.q_seq_lens.push_back(state.q_seq_lens.back() + padded_q_seq_len);
#endif
  // Process multi-modal input
  process_multi_modal_inputs(sequence, n_kv_cache_tokens, q_seq_len, seq_index);
  // Process tokens and positions
  extract_tokens_and_positions(
      sequence, n_kv_cache_tokens, logical_seq_len, seq_len, state_ptr);

  // Setup KV cache
  setup_kv_cache_info(sequence,
                      n_kv_cache_tokens,
                      seq_len,
                      padded_q_seq_len,
                      state_ptr,
                      write_block_ids_ptr);

  // Input for beam search kernel
  if (::xllm::BeamSearchConfig::get_instance().enable_beam_search_kernel() &&
      sequence->check_beam_search() && sequence->num_generated_tokens() > 0) {
    state.acc_logprob_vec.emplace_back(sequence->get_acc_logprob());
  }
}

void BatchInputBuilder::extract_tokens_and_positions(Sequence* sequence,
                                                     uint32_t n_kv_cache_tokens,
                                                     uint32_t seq_len,
                                                     uint32_t padded_seq_len,
                                                     BuilderState* state_ptr) {
  BuilderState& state = state_ptr ? *state_ptr : state_;
  const size_t seq_token_begin = state.flatten_tokens_vec.size();

  const auto& token_ids = sequence->tokens();
  const uint32_t n_tokens = token_ids.size();
  const auto& sample_slots = sequence->sample_slots();
  size_t sample_slot_idx = 0;

  // Handle MRope positions
  if (use_mrope_) {
    state.mrope_positions_vec.emplace_back(
        get_mrope_positions(sequence, n_kv_cache_tokens, seq_len));
  }

  // Process real tokens
  for (uint32_t j = n_kv_cache_tokens; j < seq_len; ++j) {
    state.flatten_tokens_vec.emplace_back(token_ids[j]);

    if (!use_mrope_) {
      state.flatten_positions_vec.push_back(static_cast<int32_t>(j));
    }

    if (sample_slots.empty()) {
      // Non-sample requests only select the last prompt token.
      if (j + 1 < n_tokens) continue;
      handle_sampling_parameters(sequence, state_ptr);
      continue;
    }

    // Sample requests need one sampling entry per selector hit. The logits for
    // selector start position come from the preceding token's hidden state.
    while (sample_slot_idx < sample_slots.size()) {
      const uint32_t sample_source_position =
          get_sample_source_position(sample_slots[sample_slot_idx]);
      if (sample_source_position < j) {
        ++sample_slot_idx;
        continue;
      }
      if (sample_source_position > j) {
        break;
      }
      handle_sampling_parameters(sequence, state_ptr);
      ++sample_slot_idx;
    }
  }

  // Right padding for CP prefill: append physical tokens only for cache/layout.
  if (padded_seq_len > seq_len) {
    const int32_t pad_token_id = args_ ? args_->pad_token_id() : 0;
    for (uint32_t j = seq_len; j < padded_seq_len; ++j) {
      state.flatten_tokens_vec.emplace_back(pad_token_id);
      if (!use_mrope_) {
        state.flatten_positions_vec.push_back(static_cast<int32_t>(j));
      }
    }
  }

  append_linear_state_row(sequence, n_kv_cache_tokens, seq_len, state);

  // Add extra token id
  int32_t extra_token_id = -1;
  if (n_tokens == seq_len) {
    // last chunk of prefill and decode
    // add -1 as extra token id
    state.extra_token_ids.emplace_back(-1);
    state.embedding_ids.emplace_back(sequence->get_single_block_id());
    state.request_ids.emplace_back(sequence->request_id());
    torch::Tensor mtp_bootstrap = sequence->get_mtp_bootstrap_embedding();
    if (state.batch_forward_type.is_decode() && mtp_bootstrap.defined()) {
      CHECK_LT(n_kv_cache_tokens, seq_len)
          << "MTP bootstrap decode input must contain current token";
      const int32_t token_id = token_ids[n_kv_cache_tokens];
      const bool is_fake_token = token_id < 0;
      const bool allow_fake_token =
          ::xllm::SchedulerConfig::get_instance().enable_schedule_overlap();
      if (is_fake_token) {
        CHECK(allow_fake_token)
            << "MTP bootstrap fake token is only allowed with schedule "
            << "overlap";
      } else {
        state.mtp_bootstrap_row_idxes.emplace_back(
            static_cast<int32_t>(state.embedding_ids.size() - 1));
        if (mtp_bootstrap.dim() == 1) {
          state.mtp_bootstrap_embeddings.emplace_back(
              mtp_bootstrap.unsqueeze(0));
        } else {
          CHECK(mtp_bootstrap.dim() == 2 && mtp_bootstrap.size(0) == 1)
              << "MTP bootstrap embedding should be [hidden] or [1, hidden]";
          state.mtp_bootstrap_embeddings.emplace_back(mtp_bootstrap);
        }
      }
    }
  } else {
    extra_token_id = token_ids[seq_len];
    state.extra_token_ids.emplace_back(extra_token_id);
  }

  // Build MTP shifted token ids for both pure prefill and chunked-prefill
  // (no_decode) when CP is enabled. The shift-by-1 layout is required by the
  // CP-aware MTP loss computation across both layouts; using is_prefill()
  // here (upstream default) would miss chunked-prefill chunks and produce
  // mis-aligned logits when CP > 1.
  if (cp_size_ > 1 && state.batch_forward_type.no_decode()) {
    const uint32_t q_len = seq_len - n_kv_cache_tokens;
    if (q_len > 1) {
      state.mtp_shifted_token_ids.insert(
          state.mtp_shifted_token_ids.end(),
          state.flatten_tokens_vec.begin() + seq_token_begin + 1,
          state.flatten_tokens_vec.begin() + seq_token_begin + q_len);
    }
    state.mtp_shifted_token_ids.emplace_back(extra_token_id);
    if (padded_seq_len > seq_len) {
      const int32_t pad_token_id = args_ ? args_->pad_token_id() : 0;
      state.mtp_shifted_token_ids.insert(state.mtp_shifted_token_ids.end(),
                                         padded_seq_len - seq_len,
                                         pad_token_id);
    }
  }
}

void BatchInputBuilder::append_linear_state_row(Sequence* sequence,
                                                uint32_t n_kv_cache_tokens,
                                                uint32_t seq_len,
                                                BuilderState& state) {
  // linear_state_ids must stay aligned with logical batch rows even when the
  // model has no linear-attention layers, because downstream consumers index by
  // batch row. Prefer the dedicated linear-state slot. Linear-attention decode
  // paths that only carry a scheduler-side single block still share that live
  // slot value across embedding and linear-state transport fields.
  const bool has_linear_attention =
      args_ && has_linear_attention_layers(*args_);
  int32_t linear_state_id = sequence->get_linear_state_slot_id();
  if (linear_state_id < 0 && has_linear_attention) {
    linear_state_id = sequence->get_single_block_id();
  }
  state.linear_state_ids.emplace_back(linear_state_id);
  if (!has_linear_attention) {
    return;
  }

  LinearStateCacheOp linear_state_cache_op;
  linear_state_cache_op.linear_state_id = state.linear_state_ids.back();
  // Linear-state checkpoints live on chunk-end boundaries, so the prefix hash
  // is chained per chunk (stride = max_tokens_per_chunk_for_prefill), not per
  // KV block. The engine enforces this stride is a positive multiple of
  // block_size when linear prefix cache is on (llm_engine.cpp); guard against
  // an unset (<= 0) stride so a misconfigured run simply skips cache ops.
  const int32_t chunk_stride = ::xllm::SchedulerConfig::get_instance()
                                   .max_tokens_per_chunk_for_prefill();
  // Cold-start restore: emit a restore hash only when a restore source
  // checkpoint is mounted on this sequence -- class A at admission
  // (allocate_shared_for_sequence) or class B at the previous step's
  // save-rotation (allocate_for_sequence) -- AND the reused prefix lands
  // on a chunk-end boundary, where the recurrent state lives in a checkpoint.
  // A mounted source is present exactly on a slot that is cold and needs
  // copy-in; continued forwards keep their live slot warm with no source
  // mounted, so they emit no restore and are not reset to cold by the worker.
  // The source slot id is taken from that mounted block below.
  const bool needs_restore_hash = sequence->has_linear_restore_src_block() &&
                                  n_kv_cache_tokens > 0 && chunk_stride > 0 &&
                                  n_kv_cache_tokens % chunk_stride == 0;
  // Exit-boundary save: persist the live state only when this prefill step
  // lands on a chunk-end boundary, so the linear-state cache stays a sparse
  // per-chunk overlay on top of the per-block KV cache.
  const bool needs_save_hash =
      should_save_linear_checkpoint(sequence, seq_len, chunk_stride);
  // Refresh the sequence's cached chunk hashes to cover this step's deepest
  // boundary, then read them back. The cache is chained and incremental, so
  // this only hashes chunks not seen on a previous step; the match probe and
  // this builder now share the one hash source instead of each recomputing.
  Slice<XXH3Key> linear_state_hashes;
  if (needs_restore_hash || needs_save_hash) {
    sequence->update_linear_state_hashes(static_cast<uint32_t>(chunk_stride));
    linear_state_hashes = sequence->linear_state_hashes();
  }
  // Restore source (block-carried): allocate_shared_for_sequence mounts the
  // deepest-hit checkpoint at admission (class A); allocate_for_sequence
  // mounts the slot it just checkpointed at the previous step's save-rotation
  // (class B). Take it unconditionally so the pin never outlives the step
  // that consumed the match; its id is used below only when a restore hash is
  // actually emitted, otherwise the handle drops here and releases the pin.
  std::optional<Block> mounted_restore_src =
      sequence->take_linear_restore_src_block();
  if (needs_restore_hash) {
    const size_t restore_chunk_idx =
        static_cast<size_t>(n_kv_cache_tokens) / chunk_stride - 1;
    if (restore_chunk_idx < linear_state_hashes.size()) {
      linear_state_cache_op.restore_requested = true;
      if (mounted_restore_src.has_value()) {
        linear_state_cache_op.restore_src_slot_id = mounted_restore_src->id();
      }
    }
  }
  if (needs_save_hash) {
    const size_t save_chunk_idx =
        static_cast<size_t>(seq_len) / chunk_stride - 1;
    if (save_chunk_idx < linear_state_hashes.size()) {
      // Record the boundary hash on the sequence. The LINEAR leaf executes
      // the save at the next step's allocate_for_sequence, after this step's
      // forward writes the boundary state into the live slot. Writing only
      // the sequence's own pending-save field keeps this safe inside the
      // parallel build loop.
      sequence->set_pending_linear_save(linear_state_hashes[save_chunk_idx]);
    }
  }
  state.linear_state_cache_ops.emplace_back(std::move(linear_state_cache_op));
}

void BatchInputBuilder::handle_sampling_parameters(Sequence* sequence,
                                                   BuilderState* state_ptr) {
  BuilderState& state = state_ptr ? *state_ptr : state_;

  // Select token for sampling
  state.selected_token_idxes.push_back(
      static_cast<int32_t>(state.flatten_tokens_vec.size() - 1));
  state.sampling_params.push_back(sequence->sampling_param());
  state.sample_idxes.push_back(
      static_cast<int32_t>(state.selected_token_idxes.size() - 1));

  // Process unique tokens
  if (need_unique_tokens_) {
    const auto& seq_token_counts = sequence->token_to_count_map();
    auto& ids = state.unique_token_ids_vec.emplace_back();
    auto& counts = state.unique_token_counts_vec.emplace_back();

    ids.reserve(seq_token_counts.size());
    counts.reserve(seq_token_counts.size());

    for (const auto& [token_id, count] : seq_token_counts) {
      CHECK(count >= 0) << "token count should be greater than 0";
      ids.push_back(token_id);
      counts.push_back(count);
    }

    state.unique_token_lens_vec.push_back(static_cast<int32_t>(ids.size()));
  }
}

torch::Tensor BatchInputBuilder::get_mrope_positions(Sequence* sequence,
                                                     uint32_t start,
                                                     uint32_t end) {
  if (sequence->stage() == SequenceStage::DECODE) {
    const int32_t mrope_position_delta = sequence->get_mrope_position_delta();
    const size_t num_tokens = sequence->num_tokens();
    return torch::arange(
               static_cast<int32_t>(mrope_position_delta + num_tokens - 1),
               static_cast<int32_t>(mrope_position_delta + num_tokens),
               torch::kInt32)
        .expand({3, -1});
  } else {
    std::unique_ptr<MPositionGenerator> generator =
        MPositionGeneratorFactory::get_instance().create_mposition_generator(
            args_->model_type());
    std::tuple<torch::Tensor, int32_t> result =
        generator->generate(sequence->tokens(), sequence->mm_data(), *args_);
    sequence->set_mrope_position_delta(std::get<1>(result));
    return std::get<0>(result).slice(/*dim=*/1, start, end);
  }
}

void BatchInputBuilder::setup_kv_cache_info(
    Sequence* sequence,
    uint32_t n_kv_cache_tokens,
    uint32_t seq_len,
    uint32_t q_seq_len,
    BuilderState* state_ptr,
    std::unordered_set<int32_t>* write_block_ids_ptr) {
  BuilderState& state = state_ptr ? *state_ptr : state_;
  std::unordered_set<int32_t>& write_block_ids =
      write_block_ids_ptr ? *write_block_ids_ptr : write_block_ids_;

  sequence->kv_state().incr_kv_cache_tokens_num(/*size=*/q_seq_len);

  const auto blocks = sequence->kv_state().blocks(BlockType::KV);
  if (sequence->kv_state().has_multi_block_export()) {
    const auto export_view = sequence->kv_state().multi_block_export_view();
    if (state.multi_block_tables.empty()) {
      state.multi_block_tables.resize(export_view.size());
    }
    CHECK_EQ(state.multi_block_tables.size(), export_view.size())
        << "composite block manager count mismatch. existing_manager_num="
        << state.multi_block_tables.size()
        << ", current_manager_num=" << export_view.size();
    for (size_t m = 0; m < export_view.size(); ++m) {
      const auto& composite_block = *export_view[m].second;
      std::vector<int32_t> block_ids;
      block_ids.reserve(composite_block.size());
      for (const auto& block : composite_block) {
        block_ids.push_back(block.id());
      }
      state.multi_block_tables[m].emplace_back(std::move(block_ids));
    }
    const auto& transfer_kv_info = sequence->kv_state().transfer_kv_info();
    if (transfer_kv_info.has_value() &&
        !transfer_kv_info->block_transfer_groups.empty()) {
      TransferKVInfo step_info;
      step_info.request_id = transfer_kv_info->request_id;
      step_info.dp_rank = transfer_kv_info->dp_rank;
      step_info.remote_instance_info = transfer_kv_info->remote_instance_info;
      step_info.local_linear_state_ids =
          transfer_kv_info->local_linear_state_ids;
      step_info.remote_linear_state_ids =
          transfer_kv_info->remote_linear_state_ids;
      for (const auto& full_group : transfer_kv_info->block_transfer_groups) {
        const auto block_type =
            block_type_from_cache_group_id(full_group.group_id);
        if (!block_type.has_value()) {
          LOG(ERROR) << "Unknown KV cache transfer group: "
                     << full_group.group_id;
          continue;
        }
        const BlockType group_block_type = block_type.value();
        const auto blocks = sequence->kv_state().blocks(group_block_type);
        if (blocks.empty() || full_group.remote_blocks_ids.empty()) {
          continue;
        }

        uint32_t block_size = 0;
        std::vector<int32_t> local_block_ids;
        local_block_ids.reserve(blocks.size());
        for (const auto& block : blocks) {
          local_block_ids.emplace_back(block.id());
          if (block.is_valid()) {
            block_size = block.size();
          }
        }
        if (block_size == 0) {
          continue;
        }

        const size_t next_transfer_block_idx =
            sequence->kv_state().next_group_transfer_block_idx(
                group_block_type);
        size_t advanced_transfer_block_idx = next_transfer_block_idx;
        KVBlockTransferGroup step_group =
            build_group_step_transfer(full_group,
                                      local_block_ids,
                                      next_transfer_block_idx,
                                      seq_len,
                                      block_size,
                                      &advanced_transfer_block_idx);
        sequence->kv_state().advance_group_transfer_block_idx(
            group_block_type, advanced_transfer_block_idx);
        if (!step_group.local_blocks_ids.empty()) {
          step_info.block_transfer_groups.emplace_back(std::move(step_group));
        }
      }
      if (!step_info.block_transfer_groups.empty()) {
        state.transfer_kv_infos.emplace_back(std::move(step_info));
      }
    }
    return;
  }

  // Keep [manager][batch][block_ids] row-aligned even if a sequence has no
  // composite blocks.
  if (!state.multi_block_tables.empty()) {
    for (auto& mgr_tables : state.multi_block_tables) {
      mgr_tables.emplace_back(std::vector<int32_t>{});
    }
  }

  const auto slot_ids = sequence->kv_state().cache_slots(
      BlockType::KV, n_kv_cache_tokens, seq_len);
  state.new_token_slot_ids.insert(
      state.new_token_slot_ids.end(), slot_ids.begin(), slot_ids.end());

  std::vector<int32_t> block_ids;
  std::vector<uint64_t> local_block_ids;
  block_ids.reserve(blocks.size());
  local_block_ids.reserve(blocks.size());
  int32_t block_size = 0;
  for (const auto& block : blocks) {
    block_size = block.size();
    block_ids.push_back(block.id());
    local_block_ids.emplace_back(static_cast<uint64_t>(block.id()));
    state.paged_kv_indices.push_back(block.id());
  }
  state.paged_kv_indptr.push_back(state.paged_kv_indptr.back() + blocks.size());
  int32_t last_page_len =
      (seq_len % block_size == 0) ? block_size : seq_len % block_size;
  state.paged_kv_last_page_len.push_back(last_page_len);

  // calculate the block ids that need to be written
  int32_t kv_cache_block_idx = n_kv_cache_tokens / block_size;
  for (auto iter = block_ids.cbegin() + kv_cache_block_idx;
       iter != block_ids.cend();
       ++iter) {
    write_block_ids.insert(*iter);
  }

  auto& transfer_kv_info = sequence->kv_state().transfer_kv_info();
  if (transfer_kv_info.has_value()) {
    const size_t next_transfer_block_idx =
        sequence->kv_state().next_transfer_block_idx();
    size_t advanced_transfer_block_idx = next_transfer_block_idx;
    TransferKVInfo step_info = BatchInputBuilder::build_step_transfer_info(
        transfer_kv_info.value(),
        local_block_ids,
        next_transfer_block_idx,
        seq_len,
        static_cast<uint32_t>(block_size),
        &advanced_transfer_block_idx);
    sequence->kv_state().advance_transfer_block_idx(
        advanced_transfer_block_idx);
    if (!step_info.local_blocks_ids.empty()) {
      state.transfer_kv_infos.emplace_back(std::move(step_info));
    }
  }

  state.block_tables_vec.emplace_back(std::move(block_ids));
}

void BatchInputBuilder::padding_decode_batch_size(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size) {
  if (num_sequences_ < min_decoding_batch_size) {
    const uint32_t n_tokens = state_.flatten_tokens_vec.size();
    // kv_cache is not empty in decoding phase
    const bool in_decoding_phase = !state_.batch_forward_type.is_prefill();
    const bool same_num_decoding_tokens =
        state_.q_max_seq_len == num_decoding_tokens &&
        n_tokens == num_sequences_ * num_decoding_tokens;
    if (in_decoding_phase && same_num_decoding_tokens) {
      // add padding tokens to the batch
      for (int32_t i = num_sequences_; i < min_decoding_batch_size; ++i) {
        for (int32_t k = 0; k < num_decoding_tokens; ++k) {
          state_.flatten_tokens_vec.emplace_back(0);
          if (!use_mrope_) {
            state_.flatten_positions_vec.emplace_back(0);
          } else {
            state_.mrope_positions_vec.emplace_back(
                torch::zeros({3, 1}, torch::kInt));
          }
          state_.new_token_slot_ids.emplace_back(0);
        }
#if defined(USE_NPU) || defined(USE_MUSA)
        state_.seq_lens.push_back(num_decoding_tokens);
        state_.q_seq_lens.push_back(num_decoding_tokens);
#elif defined(USE_MLU) || defined(USE_CUDA) || defined(USE_ILU) || \
    defined(USE_DCU)
        state_.seq_lens.push_back(state_.seq_lens.back() + num_decoding_tokens);
        state_.q_seq_lens.push_back(state_.q_seq_lens.back() +
                                    num_decoding_tokens);
#endif
        state_.block_tables_vec.emplace_back();
        if (!state_.multi_block_tables.empty()) {
          for (auto& mgr_tables : state_.multi_block_tables) {
            mgr_tables.emplace_back(std::vector<int32_t>{});
          }
        }
        state_.paged_kv_indices.push_back(0);
        state_.paged_kv_indptr.push_back(state_.paged_kv_indptr.back() + 1);
        state_.paged_kv_last_page_len.push_back(1);
      }
    }
  }
}

ForwardInput BatchInputBuilder::state_to_forward_input() {
  if (state_.flatten_tokens_vec.empty()) {
    return {};
  }

  ForwardInput forward_input;

  // Create tensors
  forward_input.token_ids =
      torch::tensor(state_.flatten_tokens_vec, torch::kInt);
  forward_input.token_ids_host = forward_input.token_ids;

  if (!use_mrope_) {
    forward_input.positions =
        torch::tensor(state_.flatten_positions_vec, torch::kInt);
  } else {
    forward_input.positions = torch::cat(state_.mrope_positions_vec, 1);
  }
  forward_input.positions_host = forward_input.positions;

  auto& input_params = forward_input.input_params;
  input_params.meta.batch_forward_type = state_.batch_forward_type;
  input_params.meta.num_sequences = static_cast<int32_t>(num_sequences_);
  input_params.meta.kv_max_seq_len = state_.max_seq_len;
  input_params.meta.q_max_seq_len = state_.q_max_seq_len;
  input_params.attention.device.kv_seq_lens =
      torch::tensor(state_.seq_lens, torch::kInt);
  input_params.attention.device.kv_cache_tokens_nums =
      torch::tensor(state_.kv_cache_tokens_nums, torch::kInt);
  input_params.attention.device.q_seq_lens =
      torch::tensor(state_.q_seq_lens, torch::kInt);
  std::vector<int32_t> q_cu_seq_lens =
      build_q_cu_seq_lens_vec(state_.q_seq_lens);
  input_params.attention.device.q_cu_seq_lens =
      torch::tensor(q_cu_seq_lens, torch::kInt);
  input_params.attention.host.kv_cache_tokens_nums =
      std::move(state_.kv_cache_tokens_nums);
  input_params.attention.host.kv_seq_lens = std::move(state_.seq_lens);
  input_params.attention.host.q_cu_seq_lens = std::move(q_cu_seq_lens);
  input_params.attention.host.q_seq_lens = std::move(state_.q_seq_lens);
  input_params.attention.device.new_cache_slots =
      torch::tensor(state_.new_token_slot_ids, torch::kInt);

  // for flashinfer
  input_params.attention.device.paged_kv_indptr =
      torch::tensor(state_.paged_kv_indptr, torch::kInt);
  input_params.attention.device.paged_kv_indices =
      torch::tensor(state_.paged_kv_indices, torch::kInt);
  input_params.attention.device.paged_kv_last_page_len =
      torch::tensor(state_.paged_kv_last_page_len, torch::kInt);

  // Setup multimodal data
  input_params.multimodal.mm_data.batch(mm_data_vec_);

  // Setup block tables
  util::pad_2d_vector(state_.block_tables_vec, /*pad_value=*/0);
  input_params.attention.device.block_tables =
      create_2d_tensor(state_.block_tables_vec, torch::kInt);
  input_params.attention.host.block_tables =
      input_params.attention.device.block_tables;

  // Setup grouped cache block tables.
  for (auto& mgr_tables : state_.multi_block_tables) {
    util::pad_2d_vector(mgr_tables, /*pad_value=*/-1);
    input_params.multi_block_tables.push_back(
        create_2d_tensor(mgr_tables, torch::kInt));
  }

  if (input_embeddings_vec_.size() != 0) {
    input_params.embedding.input_embedding = torch::cat(input_embeddings_vec_);
  }

  input_params.embedding.embedding_ids = std::move(state_.embedding_ids);
  input_params.embedding.linear_state_ids = std::move(state_.linear_state_ids);
  input_params.linear_state_cache_ops =
      std::move(state_.linear_state_cache_ops);
  if (!input_params.embedding.linear_state_ids.empty()) {
    input_params.embedding.linear_state_indices =
        torch::tensor(input_params.embedding.linear_state_ids, torch::kInt);
  }
  input_params.embedding.request_ids = std::move(state_.request_ids);
  input_params.embedding.extra_token_ids = std::move(state_.extra_token_ids);
  if (!state_.mtp_shifted_token_ids.empty()) {
    // Write both the upstream "root" path (consumed by non-CP MTP code paths
    // and by the existing shm serializer) and the CP-specific embedding path
    // (consumed by cp_input_partition + mtp_worker_impl). Both tensors share
    // storage via from_blob; the cost is one extra tensor handle, not a copy.
    auto mtp_tensor = torch::tensor(state_.mtp_shifted_token_ids, torch::kInt);
    input_params.embedding.mtp_shifted_token_ids = mtp_tensor;
    input_params.mtp_shifted_token_ids = mtp_tensor;
  }
  if (!state_.mtp_bootstrap_embeddings.empty()) {
    CHECK_EQ(state_.mtp_bootstrap_row_idxes.size(),
             state_.mtp_bootstrap_embeddings.size());
    input_params.embedding.mtp_bootstrap_row_idxes =
        std::move(state_.mtp_bootstrap_row_idxes);
    input_params.embedding.mtp_bootstrap_embeddings =
        torch::cat(state_.mtp_bootstrap_embeddings, /*dim=*/0);
    for (Sequence* sequence : sequences_) {
      sequence->clear_mtp_bootstrap_embedding();
    }
  }
  input_params.meta.batch_id = batch_id_;

  forward_input.transfer_kv_infos = std::move(state_.transfer_kv_infos);
  process_swap_block_infos(forward_input);

  CHECK_EQ(state_.sampling_params.size(), state_.selected_token_idxes.size());
  // Setup sampling parameters
  if (!state_.selected_token_idxes.empty()) {
    util::pad_2d_vector<int64_t>(state_.unique_token_ids_vec, /*pad_value=*/0);
    util::pad_2d_vector(state_.unique_token_counts_vec, /*pad_value=*/0);

    forward_input.sampling_params.init(state_.sampling_params,
                                       state_.selected_token_idxes,
                                       state_.sample_idxes,
                                       state_.unique_token_ids_vec,
                                       state_.unique_token_counts_vec,
                                       state_.unique_token_lens_vec);
  }

  return forward_input;
}

void BatchInputBuilder::process_swap_block_infos(ForwardInput& forward_input) {
  if (swap_block_transfer_infos_ == nullptr ||
      swap_block_transfer_infos_->empty()) {
    return;
  }

  auto& input_params = forward_input.input_params;
  auto& swap_blocks = *swap_block_transfer_infos_;
  if (::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel()) {
    std::sort(swap_blocks.begin(),
              swap_blocks.end(),
              [](const BlockTransferInfo& a, const BlockTransferInfo& b) {
                return a.src_block_id < b.src_block_id;
              });
#if defined(USE_CUDA)
    input_params.block_copy.swap_blocks.insert(
        input_params.block_copy.swap_blocks.end(),
        swap_blocks.begin(),
        swap_blocks.end());
    const BlockCopyKernelInputData kernel_input =
        build_block_copy_kernel_input_data(swap_blocks,
                                           /*detect_overlap=*/true);
    if (!kernel_input.has_overlap) {
      input_params.block_copy.src_block_indices =
          build_pinned_int_tensor(kernel_input.src_indices);
      input_params.block_copy.dst_block_indices =
          build_pinned_int_tensor(kernel_input.dst_indices);
      input_params.block_copy.cum_sum =
          build_pinned_int_tensor(kernel_input.cum_sum);
    }
#else
    const BlockCopyKernelInputData kernel_input =
        build_block_copy_kernel_input_data(swap_blocks,
                                           /*detect_overlap=*/false);
    input_params.block_copy.src_block_indices =
        build_pinned_int_tensor(kernel_input.src_indices);
    input_params.block_copy.dst_block_indices =
        build_pinned_int_tensor(kernel_input.dst_indices);
    input_params.block_copy.cum_sum =
        build_pinned_int_tensor(kernel_input.cum_sum);
#endif
  } else {
    input_params.block_copy.swap_blocks.insert(
        input_params.block_copy.swap_blocks.end(),
        swap_blocks.begin(),
        swap_blocks.end());
  }
}

void BatchInputBuilder::process_multi_modal_inputs(Sequence* sequence,
                                                   uint32_t n_kv_cache_tokens,
                                                   uint32_t q_seq_len,
                                                   int32_t seq_index) {
  MMData& mm_data = sequence->mutable_mm_data();
  if ((sequence->stage() != SequenceStage::DECODE) && mm_data.valid()) {
    UpdateMMItemScheduleStateVisitor visitor(
        n_kv_cache_tokens, q_seq_len, seq_index);
    mm_data.foreach (visitor);
    if (visitor.mm_data_items_.empty()) {
      return;
    }
    MMData scheduled_mm_data(visitor.scheduled_type_,
                             std::move(visitor.mm_data_items_));
    mm_data_vec_.emplace_back(std::move(scheduled_mm_data));
  }
}
}  // namespace xllm
