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

#include "core/framework/batch/rec_forward_input_factory.h"

#include <glog/logging.h>

#include <algorithm>
#include <utility>

#include "core/framework/batch/rec_batch.h"
#include "core/framework/batch/rec_batch_group.h"
#include "core/util/env_var.h"
#include "core/util/threadpool.h"

namespace xllm {

RecForwardInputFactory::RecForwardInputFactory(
    RecForwardInputFactoryOptions options)
    : options_(options) {
  CHECK_GT(options_.dp_size, 0);
}

RecForwardInputFactory::~RecForwardInputFactory() = default;

void RecForwardInputFactory::create_inputs(
    RecBatchGroup& batches,
    const ModelArgs& model_args,
    std::vector<RecForwardInput>& inputs) {
  CHECK_EQ(batches.size(), options_.dp_size);
  std::call_once(threadpool_once_, [this] {
    threadpool_ = std::make_unique<ThreadPool>(
        /*num_threads=*/16,
        /*cpu_binding=*/true,
        /*pool_name=*/"RecForwardInputFactory.forward_input");
  });
  PreparationState state;
  state.inputs.reserve(options_.dp_size);
  state.dp_token_counts.resize(options_.dp_size);
  state.dp_sequence_counts.resize(options_.dp_size);
  state.dp_is_decode.resize(options_.dp_size, 0);
  prepare_rank_inputs(batches, model_args, state);
  finalize_inputs(state);
  inputs = std::move(state.inputs);
}

RecForwardInput RecForwardInputFactory::create_input(
    RecBatch& batch,
    const ModelArgs& model_args) {
  std::call_once(input_builder_threadpool_once_, [this] {
    const int64_t num_threads = std::max<int64_t>(
        1, util::get_int_env("XLLM_REC_INPUT_BUILDER_THREADS", 16));
    input_builder_threadpool_ =
        std::make_unique<MPMCThreadPool>(static_cast<size_t>(num_threads));
  });
  return batch.prepare_rec_forward_input(options_.num_decoding_tokens,
                                         options_.min_decoding_batch_size,
                                         model_args,
                                         input_builder_threadpool_.get());
}

void RecForwardInputFactory::prepare_rank_inputs(RecBatchGroup& batches,
                                                 const ModelArgs& model_args,
                                                 PreparationState& state) {
  for (uint32_t dp_rank = 0; dp_rank < options_.dp_size; ++dp_rank) {
    batches[dp_rank].refresh_forward_type();
    state.inputs.emplace_back(batches[dp_rank].prepare_forward_input(
        model_args, threadpool_.get(), /*cp_size=*/1));
    const auto& input = state.inputs[dp_rank];
    const auto& meta = input.input_params.meta;
    state.dp_token_counts[dp_rank] =
        static_cast<int32_t>(input.host_token_ids().numel());
    state.dp_sequence_counts[dp_rank] = meta.num_sequences;
    if (state.batch_forward_type.is_empty() &&
        !meta.batch_forward_type.is_empty()) {
      state.batch_forward_type = meta.batch_forward_type;
    }
    state.dp_is_decode[dp_rank] =
        state.batch_forward_type.is_decode() && meta.q_max_seq_len == 1;
  }
}

void RecForwardInputFactory::finalize_inputs(PreparationState& state) {
  for (auto& input : state.inputs) {
    auto& parallel = input.input_params.parallel;
    parallel.dp_global_token_nums = state.dp_token_counts;
    parallel.dp_global_sequence_nums = state.dp_sequence_counts;
    parallel.raw_dp_global_token_nums = state.dp_token_counts;
    parallel.dp_is_decode = state.dp_is_decode;
    if (input.input_params.meta.batch_forward_type.is_empty()) {
      input.input_params.meta.batch_forward_type = state.batch_forward_type;
    }
  }
}

}  // namespace xllm
