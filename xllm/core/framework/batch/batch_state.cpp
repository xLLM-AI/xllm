/* Copyright 2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#include "core/framework/batch/batch_state.h"

#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#include <atomic>
#include <numeric>
#include <utility>

#include "core/framework/batch/forward_input_builder.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/model/model_args.h"

namespace xllm {

void BatchState::add(Sequence* sequence, uint32_t allowed_max_token) {
  CHECK(sequence != nullptr);
  CHECK(!sequence->finished());
  CHECK_GT(allowed_max_token, 0);

  set_batch_id();
  sequence_plan_.add(sequence, allowed_max_token);

  const auto& input_embedding = sequence->get_input_embedding();
  if (input_embedding.defined()) {
    input_embeddings_vec_.emplace_back(input_embedding);
  }

  update_forward_type(sequence);
}

void BatchState::update_forward_type(Sequence* sequence) {
  auto stage = sequence->stage();
  switch (batch_forward_type_.value()) {
    case BatchForwardType::PREFILL:
      if (stage == SequenceStage::CHUNKED_PREFILL) {
        batch_forward_type_ = BatchForwardType::CHUNKED_PREFILL;
      } else if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::CHUNKED_PREFILL:
      if (stage == SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::DECODE:
      if (stage != SequenceStage::DECODE) {
        batch_forward_type_ = BatchForwardType::MIXED;
      }
      break;
    case BatchForwardType::MIXED:
      break;
    case BatchForwardType::EMPTY:
      batch_forward_type_ = BatchForwardType(static_cast<int32_t>(stage));
      break;
  }
}

void BatchState::set_batch_id() {
  static std::atomic<uint64_t> next_batch_id{1};
  while (batch_id_ == UNINITIALIZED_BATCH_ID) {
    batch_id_ = next_batch_id.fetch_add(1, std::memory_order_relaxed);
  }
}

void BatchState::reserve(size_t sequence_count, size_t group_count) {
  sequence_plan_.reserve(sequence_count);
  input_embeddings_vec_.reserve(sequence_count);
  sequence_groups_.reserve(group_count);
}

void BatchState::add(SequencesGroup* sequence_group) {
  CHECK(sequence_group != nullptr);
  CHECK(!sequence_group->sequences().empty());
  set_batch_id();
  sequence_groups_.emplace_back(sequence_group);
}

bool BatchState::has_partial_finished_beam_group() const {
  if (sequence_groups_.empty()) {
    return false;
  }

  for (auto* seq_group : sequence_groups_) {
    if (!seq_group->check_beam_search()) {
      continue;
    }

    const auto& sequences = seq_group->sequences();
    if (sequences.empty()) {
      continue;
    }

    const size_t finished_cnt = static_cast<size_t>(
        std::count_if(sequences.begin(), sequences.end(), [](const auto& seq) {
          return seq->finished();
        }));
    if (finished_cnt > 0 && finished_cnt < sequences.size()) {
      return true;
    }
  }
  return false;
}

void BatchState::refresh_sequences_from_groups() {
  if (sequence_groups_.empty()) {
    return;
  }
  BatchSequencePlan next_plan;
  size_t sequence_count = 0;
  for (const auto* group : sequence_groups_) {
    sequence_count += group->sequences().size();
  }
  next_plan.reserve(sequence_count);
  for (const auto* group : sequence_groups_) {
    for (const auto& sequence : group->sequences()) {
      next_plan.add(sequence.get(), std::numeric_limits<uint32_t>::max());
    }
  }
  sequence_plan_ = std::move(next_plan);
}

void BatchState::dp_balance_shuffle_seqs() {
#if defined(USE_NPU)
  // this shuffle operation is mainly used for npu with 24 cores
  // and specific mla op implementation
  constexpr size_t kNumNpuCores = 24;
  if (::xllm::KernelConfig::get_instance().enable_customize_mla_kernel() &&
      ::xllm::ParallelConfig::get_instance().enable_dp_balance() &&
      sequence_plan_.sequences().size() > kNumNpuCores) {
    std::vector<uint32_t> kv_cache_tokens_num;
    kv_cache_tokens_num.reserve(sequence_plan_.sequences().size());
    for (auto& seq : sequence_plan_.sequences()) {
      kv_cache_tokens_num.push_back(seq->kv_state().kv_cache_tokens_num());
    }
    auto seq_index_shift = cal_seq_exchange_index(kv_cache_tokens_num);

    std::vector<size_t> source_indices(sequence_plan_.size());
    for (const auto& [source, target] : seq_index_shift) {
      source_indices[target] = source;
    }
    sequence_plan_.reorder(source_indices);
  }
#else
  // TODO: implement dp_balance_shuffle_seqs for non-npu devices
  static bool warning = true;
  if (warning) {
    LOG(WARNING)
        << "dp_balance_shuffle_seqs is not implemented for current device";
    warning = false;
  }
#endif
}

std::unordered_map<uint32_t, uint32_t> BatchState::cal_seq_exchange_index(
    std::vector<uint32_t>& kv_cache_tokens_num) {
  constexpr size_t kNumNpuCores = 24;
  const size_t num_seqs = kv_cache_tokens_num.size();
  const size_t base_per_core = num_seqs / kNumNpuCores;
  const size_t remainder = num_seqs % kNumNpuCores;

  // find the indices of the remainder biggest elements
  std::vector<uint32_t> indices(num_seqs);
  std::iota(indices.begin(), indices.end(), 0);
  if (remainder > 0) {
    std::nth_element(indices.begin(),
                     indices.end() - remainder,
                     indices.end(),
                     [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
                       return kv_cache_tokens_num[a] < kv_cache_tokens_num[b];
                     });
  }

  std::vector<uint32_t> base_indices(indices.begin(),
                                     indices.end() - remainder);
  std::vector<uint32_t> remainder_indices(indices.end() - remainder,
                                          indices.end());

  // sort base_indices in descending order
  std::sort(base_indices.begin(),
            base_indices.end(),
            [&kv_cache_tokens_num](uint32_t a, uint32_t b) {
              return kv_cache_tokens_num[a] > kv_cache_tokens_num[b];
            });

  // allocate a long and a short request to each core, to ensuring
  // load balance among all cores
  std::vector<std::vector<uint32_t>> base_assignment(
      kNumNpuCores, std::vector<uint32_t>(base_per_core));
  for (size_t i = 0; i < base_indices.size(); ++i) {
    const size_t col = i / kNumNpuCores;
    const size_t row = (col % 2 == 0) ? (i % kNumNpuCores)
                                      : (kNumNpuCores - 1 - (i % kNumNpuCores));
    base_assignment[row][col] = base_indices[i];
  }

  // record the index map, first one is original index,
  // second one is the target index to be exchanged to
  std::unordered_map<uint32_t, uint32_t> index_shift;
  // add base part data
  for (size_t i = 0; i < kNumNpuCores; ++i) {
    for (size_t j = 0; j < base_per_core; ++j) {
      const uint32_t idx = base_assignment[i][j];
      index_shift[idx] = static_cast<uint32_t>(i + j * kNumNpuCores);
    }
  }
  // add remainder part data
  for (size_t i = 0; i < remainder; ++i) {
    index_shift[remainder_indices[i]] =
        static_cast<uint32_t>(i + kNumNpuCores * base_per_core);
  }

  return index_shift;
}

void BatchState::refresh_forward_type(const std::vector<Sequence*>& sequences) {
  batch_forward_type_ = BatchForwardType();
  for (auto* sequence : sequences) {
    update_forward_type(sequence);
  }
}

size_t BatchState::num_group_sequences() const {
  size_t count = 0;
  for (const auto* group : sequence_groups_) {
    count += group->sequences().size();
  }
  return count;
}

std::vector<Sequence*> BatchState::group_sequences() const {
  std::vector<Sequence*> sequences;
  sequences.reserve(num_group_sequences());
  for (const auto* group : sequence_groups_) {
    for (const auto& sequence : group->sequences()) {
      sequences.emplace_back(sequence.get());
    }
  }
  return sequences;
}

Sequence* BatchState::group_sequence(size_t index) const {
  for (const auto* group : sequence_groups_) {
    if (index < group->sequences().size()) {
      return group->sequences()[index].get();
    }
    index -= group->sequences().size();
  }
  LOG(FATAL) << "Group sequence index out of range";
  return nullptr;
}

BatchInputData BatchState::input_data(const BatchSequencePlan& plan) {
  return {plan.sequences(),
          sequence_groups_,
          plan.budgets(),
          input_embeddings_vec_,
          mm_data_vec_,
          &swap_block_transfer_infos_,
          batch_id_,
          batch_forward_type_};
}

ForwardInput BatchState::prepare_sequence_input(
    uint32_t num_decoding_tokens,
    uint32_t min_decoding_batch_size,
    const ModelArgs& args,
    int32_t cp_size) {
  CHECK(sequence_groups_.empty() || !sequence_plan_.empty())
      << "Sequence input requires scheduled sequences; group-only input "
         "requires a domain-specific input builder";
  const auto data = input_data(sequence_plan_);
  output_handler_.prepare(data);
  ForwardInputBuilder builder(data, &args, cp_size);
  auto input =
      builder.build_forward_input(num_decoding_tokens, min_decoding_batch_size);
  linear_restore_src_blocks_ = builder.take_linear_restore_src_blocks();
  return input;
}

ForwardInput BatchState::prepare_distributed_input(const ModelArgs& args,
                                                   ThreadPool* thread_pool,
                                                   int32_t cp_size) {
  CHECK(sequence_groups_.empty() || !sequence_plan_.empty())
      << "Sequence input requires scheduled sequences";
  dp_balance_shuffle_seqs();
  const auto data = input_data(sequence_plan_);
  output_handler_.prepare(data);
  ForwardInputBuilder builder(data, &args, cp_size, thread_pool);
  auto input = builder.build_forward_input(/*num_decoding_tokens=*/0,
                                           /*min_decoding_batch_size=*/0);
  linear_restore_src_blocks_ = builder.take_linear_restore_src_blocks();
  if (has_partial_finished_beam_group()) {
    input.sampling_params.acc_logprob = torch::Tensor();
  }
  return input;
}

}  // namespace xllm
