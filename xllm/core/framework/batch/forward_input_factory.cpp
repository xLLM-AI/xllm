/* Copyright 2026 The xLLM Authors.

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

#include "core/framework/batch/forward_input_factory.h"

#include <glog/logging.h>

#include <cstddef>
#include <limits>
#include <utility>

#include "core/framework/batch/batch_group.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/eplb/eplb_controller.h"
#include "core/framework/model/model_args.h"
#include "core/util/threadpool.h"
#include "core/util/utils.h"

namespace xllm {

ForwardInputFactory::ForwardInputFactory(ForwardInputFactoryOptions options)
    : options_(options) {
  CHECK_GT(options_.dp_size, 0);
  CHECK_GT(options_.cp_size, 0);
  CHECK_LE(options_.cp_size, std::numeric_limits<int32_t>::max());
  dp_batch_embedding_ids_.resize(options_.dp_size);
  dp_batch_request_ids_.resize(options_.dp_size);
  dp_batch_generations_.resize(options_.dp_size, 0);
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"ForwardInputFactory.forward_input");
}

ForwardInputFactory::~ForwardInputFactory() = default;

void ForwardInputFactory::create_inputs(BatchGroup& batches,
                                        const ModelArgs& model_args,
                                        std::vector<LlmForwardInput>& inputs,
                                        bool& is_graph_warmup) {
  CHECK_EQ(batches.size(), options_.dp_size)
      << "Split DP batch failed with dp_size as " << options_.dp_size
      << " and actual batch size as " << batches.size() << ".";

  inputs.clear();
  PreparationState state;
  state.inputs.swap(inputs);
  state.inputs.reserve(options_.dp_size);
  state.dp_token_counts.resize(options_.dp_size);
  state.dp_sequence_counts.resize(options_.dp_size);
  state.dp_kv_max_seq_lens.resize(options_.dp_size);
  if (options_.enable_dp_global_json_object_active) {
    state.dp_global_json_object_active.resize(options_.dp_size);
  }
  state.dp_is_decode.resize(options_.dp_size, 0);
  prepare_rank_inputs(batches, model_args, state);
  finalize_inputs(state);

  inputs = std::move(state.inputs);
  is_graph_warmup = state.is_graph_warmup;
}

void ForwardInputFactory::set_eplb_controller(EplbController* controller) {
  eplb_controller_ = controller;
}

void ForwardInputFactory::prepare_rank_inputs(BatchGroup& batches,
                                              const ModelArgs& model_args,
                                              PreparationState& state) {
  for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
    state.inputs.emplace_back(batches[dp_rank].prepare_forward_input(
        model_args, threadpool_.get(), static_cast<int32_t>(options_.cp_size)));
    const auto& input = state.inputs[dp_rank];
    const auto& meta = input.input_params.meta;
    const BatchForwardType& current_forward_type = meta.batch_forward_type;
    state.dp_token_counts[dp_rank] =
        static_cast<int32_t>(input.host_token_ids().numel());
    state.dp_sequence_counts[dp_rank] = meta.num_sequences;
    state.dp_kv_max_seq_lens[dp_rank] = meta.kv_max_seq_len;
    if (options_.enable_dp_global_json_object_active) {
      state.dp_global_json_object_active[dp_rank] =
          !input.json_object_states.empty() ||
          !input.json_object_state_snapshots.empty();
    }
    if (util::is_deepseek_v4_model_type(model_args.model_type())) {
      const int64_t actual_scheduled_tokens =
          static_cast<int64_t>(input.host_token_ids().numel());
      CHECK_LE(actual_scheduled_tokens, options_.max_tokens_per_batch)
          << "DSV4 actual scheduled tokens exceed max_tokens_per_batch used "
             "for SWA cache allocation. This can make the shared SWA burst "
             "pool smaller than the block/table consumer needs and may cause "
             "SWA KV rows to be overwritten or read from the wrong position. "
             "Please increase --max_tokens_per_batch, reduce scheduler token "
             "load, or check chunked-prefill padding. Details: dp_rank="
          << dp_rank << ", actual_scheduled_tokens=" << actual_scheduled_tokens
          << ", max_tokens_per_batch=" << options_.max_tokens_per_batch
          << ", q_max_seq_len=" << meta.q_max_seq_len
          << ", kv_max_seq_len=" << meta.kv_max_seq_len
          << ", batch_forward_type=" << current_forward_type.to_string();
    }
    if (state.batch_forward_type.is_empty() &&
        !current_forward_type.is_empty()) {
      state.batch_forward_type = current_forward_type;
    }
    if (!current_forward_type.is_empty()) {
      state.has_non_empty_batch = true;
      state.all_non_empty_batches_are_decode =
          state.all_non_empty_batches_are_decode &&
          current_forward_type.is_decode();
    }
    state.is_graph_warmup = state.is_graph_warmup || meta.is_graph_warmup;
    state.dp_is_decode[dp_rank] =
        current_forward_type.is_decode() && meta.q_max_seq_len == 1;

    const auto& embedding = input.input_params.embedding;
    if (dp_batch_embedding_ids_[dp_rank] != embedding.embedding_ids ||
        dp_batch_request_ids_[dp_rank] != embedding.request_ids) {
      dp_batch_embedding_ids_[dp_rank] = embedding.embedding_ids;
      dp_batch_request_ids_[dp_rank] = embedding.request_ids;
      ++dp_batch_generations_[dp_rank];
    }
  }
}

void ForwardInputFactory::finalize_inputs(PreparationState& state) {
  // Graph decode requires empty ranks to participate using Worker fake inputs.
  if (::xllm::ExecutionConfig::get_instance().enable_graph() &&
      state.batch_forward_type.is_decode()) {
    for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
      if (state.inputs[dp_rank]
              .input_params.meta.batch_forward_type.is_empty() &&
          state.dp_token_counts[dp_rank] == 0) {
        state.dp_is_decode[dp_rank] = 1;
      }
    }
  }

  annotate_eplb_inputs(state);

  for (auto& input : state.inputs) {
    input.input_params.meta.is_graph_warmup = state.is_graph_warmup;
    auto& parallel = input.input_params.parallel;
    parallel.dp_global_token_nums = state.dp_token_counts;
    parallel.dp_global_batch_generations = dp_batch_generations_;
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      input.input_params.meta.batch_forward_type = state.batch_forward_type;
    }
  }

  // Every rank owns independent metadata. Once the preceding ranks have their
  // copies, the last rank can retain the temporary buffers without allocating
  // another set. Keep its mutable and raw token counts independent as well.
  for (size_t dp_rank = 0; dp_rank + 1 < state.inputs.size(); ++dp_rank) {
    auto& parallel = state.inputs[dp_rank].input_params.parallel;
    parallel.dp_global_sequence_nums = state.dp_sequence_counts;
    parallel.raw_dp_global_token_nums = state.dp_token_counts;
    parallel.dp_global_kv_max_seq_lens = state.dp_kv_max_seq_lens;
    parallel.dp_global_json_object_active = state.dp_global_json_object_active;
    parallel.dp_is_decode = state.dp_is_decode;
  }
  auto& last_parallel = state.inputs.back().input_params.parallel;
  last_parallel.dp_global_sequence_nums = std::move(state.dp_sequence_counts);
  last_parallel.raw_dp_global_token_nums = std::move(state.dp_token_counts);
  last_parallel.dp_global_kv_max_seq_lens = std::move(state.dp_kv_max_seq_lens);
  last_parallel.dp_global_json_object_active =
      std::move(state.dp_global_json_object_active);
  last_parallel.dp_is_decode = std::move(state.dp_is_decode);
}

void ForwardInputFactory::annotate_eplb_inputs(PreparationState& state) {
  if (eplb_controller_ == nullptr) {
    return;
  }
  eplb_controller_->annotate_inputs(
      state.inputs,
      state.dp_token_counts,
      /*allow_eplb_command=*/state.has_non_empty_batch &&
          state.all_non_empty_batches_are_decode && !state.is_graph_warmup);
}

}  // namespace xllm
