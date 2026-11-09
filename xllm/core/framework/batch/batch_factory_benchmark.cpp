/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// Measures scheduler-side LLM BatchFactory construction by sequence count,
// data-parallel rank count, and prefill/decode stage. Request and Sequence
// construction stay outside the timed loop. This benchmark does not initialize
// Python or an NPU runtime and does not allocate tensors on an NPU.
//
// Build and run (NPU software environment example):
//   python setup.py test --device npu --test-name batch_factory_benchmark
//   ./batch_factory_benchmark --benchmark_min_time=0.2s
// The device option selects the build backend and its libraries; this benchmark
// measures CPU work and does not require NPU hardware.

#include <benchmark/benchmark.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/batch/batch_factory.h"
#include "core/framework/request/request.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"

namespace xllm {
namespace {

constexpr size_t kBatchFactoryPromptTokens = 512;

// Token ids that do not repeat for ~4M requests (ids are never fed to a model
// or tokenizer here), so no sequence can hit the prefix cache with an earlier
// sequence's prompt.
std::vector<int32_t> next_prompt_tokens(size_t num_tokens) {
  static int64_t next_token = 0;
  std::vector<int32_t> tokens(num_tokens);
  for (int32_t& token : tokens) {
    token = static_cast<int32_t>(next_token++ %
                                 std::numeric_limits<int32_t>::max());
  }
  return tokens;
}

std::shared_ptr<Request> make_llm_batch_factory_request(size_t index,
                                                        int32_t dp_rank) {
  RequestSamplingParam sampling_param;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1 << 20);
  stopping_checker.set_max_context_len(1 << 20);
  stopping_checker.set_ignore_eos(true);
  RequestState state("benchmark",
                     next_prompt_tokens(kBatchFactoryPromptTokens),
                     std::move(sampling_param),
                     SchedulerParam{},
                     std::move(stopping_checker),
                     kBatchFactoryPromptTokens + 64,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  auto request = std::make_shared<Request>(
      std::to_string(index), "", "", std::move(state));
  request->sequences().front()->set_dp_rank(dp_rank);
  return request;
}

enum class BatchFactoryStage { PREFILL, DECODE, CHUNKED_PREFILL, MIXED };

constexpr BatchFactoryStage kMixedStages[] = {
    BatchFactoryStage::PREFILL,
    BatchFactoryStage::DECODE,
    BatchFactoryStage::CHUNKED_PREFILL};

// Owns the requests referenced by the scheduler's sequence view. Keeping the
// fixture outside the timed loop isolates BatchFactory::create_batches from
// request and Sequence construction.
class LlmBatchFactoryFixture final {
 public:
  LlmBatchFactoryFixture(size_t batch_size,
                         int32_t dp_size,
                         BatchFactoryStage stage,
                         bool concentrated_rank = false) {
    requests_.reserve(batch_size);
    sequences_.reserve(batch_size);
    budgets_.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      const int32_t dp_rank =
          concentrated_rank ? 0 : static_cast<int32_t>(i % dp_size);
      auto request = make_llm_batch_factory_request(i, dp_rank);
      Sequence* sequence = request->sequences().front().get();
      // Use three stages rather than alternating two, so DP sizes 2/4/8 do
      // not accidentally isolate each stage on a different rank.
      const BatchFactoryStage sequence_stage =
          stage == BatchFactoryStage::MIXED ? kMixedStages[i % 3] : stage;
      size_t budget = kBatchFactoryPromptTokens;
      switch (sequence_stage) {
        case BatchFactoryStage::PREFILL:
          break;
        case BatchFactoryStage::DECODE:
          sequence->kv_state().set_kv_cache_tokens_num(
              sequence->num_prompt_tokens());
          budget = 1;
          break;
        case BatchFactoryStage::CHUNKED_PREFILL:
          sequence->kv_state().set_kv_cache_tokens_num(
              kBatchFactoryPromptTokens / 4);
          budget = kBatchFactoryPromptTokens / 4;
          break;
        case BatchFactoryStage::MIXED:
          LOG(FATAL) << "Mixed batch must assign a concrete sequence stage";
      }
      requests_.emplace_back(std::move(request));
      sequences_.emplace_back(sequence);
      budgets_.emplace_back(budget);
    }
  }

  LlmBatchFactoryFixture(const LlmBatchFactoryFixture&) = delete;
  LlmBatchFactoryFixture& operator=(const LlmBatchFactoryFixture&) = delete;

  const std::vector<std::shared_ptr<Request>>& requests() const {
    return requests_;
  }
  const std::vector<Sequence*>& sequences() const { return sequences_; }
  const std::vector<size_t>& budgets() const { return budgets_; }

 private:
  std::vector<std::shared_ptr<Request>> requests_;
  std::vector<Sequence*> sequences_;
  std::vector<size_t> budgets_;
};

void benchmark_llm_batch_factory(benchmark::State& state,
                                 BatchFactoryStage stage,
                                 bool concentrated_rank = false) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const int32_t dp_size = static_cast<int32_t>(state.range(1));
  LlmBatchFactoryFixture fixture(batch_size, dp_size, stage, concentrated_rank);
  const BatchFactory factory(dp_size);

  for (auto _ : state) {
    BatchGroup batches =
        factory.create_batches(fixture.sequences(), fixture.budgets());
    benchmark::DoNotOptimize(batches);
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(batch_size));
}

// range(0): scheduled LLM sequences, range(1): DP size, range(2): decode (0/1).
void BM_BatchFactory_CreateLLM(benchmark::State& state) {
  benchmark_llm_batch_factory(state,
                              state.range(2) != 0 ? BatchFactoryStage::DECODE
                                                  : BatchFactoryStage::PREFILL);
}

// range(0): scheduled LLM sequences, range(1): DP size. Each rank sees
// full prefill, chunked prefill and decode sequences in the same batch.
void BM_BatchFactory_CreateLLM_MixedStage(benchmark::State& state) {
  benchmark_llm_batch_factory(state, BatchFactoryStage::MIXED);
}

// range(0): scheduled LLM sequences, range(1): DP size. Each sequence has
// cached 128 prompt tokens and receives a budget for the next 128 tokens.
void BM_BatchFactory_CreateLLM_ChunkedPrefill(benchmark::State& state) {
  benchmark_llm_batch_factory(state, BatchFactoryStage::CHUNKED_PREFILL);
}

// range(0): scheduled LLM sequences, range(1): DP size. All decode sequences
// land on rank 0, leaving the other ranks empty.
void BM_BatchFactory_CreateLLM_ConcentratedRank(benchmark::State& state) {
  benchmark_llm_batch_factory(state,
                              BatchFactoryStage::DECODE,
                              /*concentrated_rank=*/true);
}

BENCHMARK(BM_BatchFactory_CreateLLM)
    ->ArgsProduct({{1, 16, 64, 256, 384, 512, 768, 1024, 1536},
                   {1, 2, 4, 8},
                   {0, 1}})
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_BatchFactory_CreateLLM_MixedStage)
    ->Args({256, 1})
    ->Args({256, 4})
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_BatchFactory_CreateLLM_ChunkedPrefill)
    ->Args({256, 1})
    ->Args({256, 4})
    ->Args({1024, 1})
    ->Args({1024, 4})
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_BatchFactory_CreateLLM_ConcentratedRank)
    ->Args({1, 8})
    ->Args({256, 8})
    ->Args({512, 8})
    ->Args({768, 8})
    ->Args({1024, 8})
    ->Args({1536, 8})
    ->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace xllm
