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

#include "core/framework/batch/vlm_forward_input_factory.h"

#include <glog/logging.h>

#include <utility>

#include "core/framework/batch/batch_group.h"
#include "core/framework/config/execution_config.h"
#include "core/util/threadpool.h"

namespace xllm {

VlmForwardInputFactory::VlmForwardInputFactory(
    VlmForwardInputFactoryOptions options)
    : options_(options) {
  CHECK_GT(options_.dp_size, 0);
  threadpool_ = std::make_unique<ThreadPool>(
      /*num_threads=*/16,
      /*cpu_binding=*/false,
      /*pool_name=*/"VlmForwardInputFactory.forward_input");
}

VlmForwardInputFactory::~VlmForwardInputFactory() = default;

void VlmForwardInputFactory::create_inputs(
    BatchGroup& batches,
    const ModelArgs& model_args,
    std::vector<VlmForwardInput>& inputs) {
  CHECK_EQ(batches.size(), options_.dp_size);
  PreparationState state;
  state.inputs.reserve(options_.dp_size);
  state.dp_token_counts.resize(options_.dp_size);
  state.dp_sequence_counts.resize(options_.dp_size);
  state.dp_kv_max_seq_lens.resize(options_.dp_size);
  state.dp_is_decode.resize(options_.dp_size, 0);
  if (options_.enable_dp_global_json_object_active) {
    state.dp_global_json_object_active.resize(options_.dp_size);
  }
  prepare_rank_inputs(batches, model_args, state);
  finalize_inputs(state);
  inputs = std::move(state.inputs);
}

void VlmForwardInputFactory::prepare_rank_inputs(BatchGroup& batches,
                                                 const ModelArgs& model_args,
                                                 PreparationState& state) {
  for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
    if (batches[dp_rank].empty()) {
      VlmForwardInput empty_input;
      empty_input.input_params.meta.batch_forward_type = BatchForwardType();
      empty_input.input_params.meta.batch_id = UNINITIALIZED_BATCH_ID;
      state.inputs.emplace_back(std::move(empty_input));
    } else {
      state.inputs.emplace_back(batches[dp_rank].prepare_vlm_forward_input(
          model_args, threadpool_.get()));
    }
    const auto& input = state.inputs[dp_rank];
    const auto& meta = input.input_params.meta;
    state.dp_token_counts[dp_rank] =
        static_cast<int32_t>(input.host_token_ids().numel());
    state.dp_sequence_counts[dp_rank] = meta.num_sequences;
    state.dp_kv_max_seq_lens[dp_rank] = meta.kv_max_seq_len;
    if (options_.enable_dp_global_json_object_active) {
      state.dp_global_json_object_active[dp_rank] =
          !input.json_object_states.empty() ||
          !input.json_object_state_snapshots.empty();
    }
    const auto& current_forward_type = meta.batch_forward_type;
    if (state.batch_forward_type.is_empty() &&
        !current_forward_type.is_empty()) {
      state.batch_forward_type = current_forward_type;
    }
    state.dp_is_decode[dp_rank] =
        current_forward_type.is_decode() && meta.q_max_seq_len == 1;
  }
}

void VlmForwardInputFactory::finalize_inputs(PreparationState& state) {
  // Empty DP ranks participate in graph decode using Worker fake inputs.
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
  for (auto& input : state.inputs) {
    auto& parallel = input.input_params.parallel;
    parallel.dp_global_token_nums = state.dp_token_counts;
    parallel.dp_global_sequence_nums = state.dp_sequence_counts;
    parallel.raw_dp_global_token_nums = state.dp_token_counts;
    parallel.dp_global_kv_max_seq_lens = state.dp_kv_max_seq_lens;
    parallel.dp_global_json_object_active = state.dp_global_json_object_active;
    parallel.dp_is_decode = state.dp_is_decode;
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      input.input_params.meta.batch_forward_type = state.batch_forward_type;
    }
  }
}

}  // namespace xllm
