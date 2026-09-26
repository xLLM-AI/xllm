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

// Hop 3 of the request path: ContinuousScheduler.
//
// The master hands a Request to ContinuousScheduler::add_request; the
// scheduler loop then runs prepare_batch (drain the queue, collect finished
// requests, allocate KV blocks through BlockManagerPool, build Batch objects)
// and, after the engine step, process_batch_output. These benchmarks time each
// of those scheduler-side steps against a FakeEngine that owns a real
// BlockManagerPool but never runs a model, so the numbers are pure scheduler +
// block-manager CPU cost.
//
//   * BM_Scheduler_AddRequest           - enqueue N requests.
//   * BM_Scheduler_PrepareBatch_Prefill - schedule N waiting requests into a
//                                         prefill batch (block allocation
//                                         included).
//   * BM_Scheduler_PrepareBatch_Decode  - schedule N running requests into a
//                                         decode batch.
//   * BM_Scheduler_ProcessBatchOutput   - post-step bookkeeping over N running
//                                         requests, stream on/off.
//
// Every request carries fresh token ids so the prefix cache never short-cuts
// allocation, which is the common case for unrelated online requests.
//
// Build & run (example):
//   python setup.py test --test-name continuous_scheduler_benchmark
//   ./continuous_scheduler_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>
#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_state.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/scheduler/continuous_scheduler.h"

namespace xllm {
namespace {

// benchmark 1.8.x exposes only DoNotOptimize(Tp const&) -- deprecated -- and
// DoNotOptimize(Tp&); there is no rvalue overload, so a temporary or a const
// local resolves to the deprecated one. Sinking the value into a non-const
// parameter first selects the supported overload.
template <typename T>
inline BENCHMARK_ALWAYS_INLINE void do_not_optimize(T value) {
  benchmark::DoNotOptimize(value);
}

constexpr int32_t kBlockSize = 128;
constexpr uint32_t kNumBlocks = 8192;
constexpr int32_t kMaxSeqsPerBatch = 1024;
constexpr int32_t kMaxTokensPerBatch = 1 << 20;
constexpr size_t kPromptTokens = 512;
constexpr int32_t kMaxContextLen = 1 << 20;

class FakeTokenizer final : public Tokenizer {
 public:
  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

// Engine stand-in: real BlockManagerPool, no model.
class FakeEngine final : public Engine {
 public:
  FakeEngine() {
    BlockManagerPool::Options options;
    options.num_blocks(kNumBlocks)
        .block_size(kBlockSize)
        .max_seqs_per_batch(kMaxSeqsPerBatch)
        .enable_prefix_cache(true);
    block_manager_pool_ =
        std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override { return {}; }
  void update_last_step_result(std::vector<Batch>& /*batch*/) override {}
  const Tokenizer* tokenizer() const override { return &fake_tokenizer_; }
  BlockManagerPool* block_manager_pool() const override {
    return block_manager_pool_.get();
  }
  const ModelArgs& model_args() const override { return model_args_; }
  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

 private:
  FakeTokenizer fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_pool_;
  ModelArgs model_args_;
};

// Exposes the response processor so each iteration can drain the async
// completion callbacks before the next one starts.
class BenchContinuousScheduler final : public ContinuousScheduler {
 public:
  BenchContinuousScheduler(Engine* engine, const Options& options)
      : ContinuousScheduler(engine, options) {}

  std::vector<Batch> prepare_batch_test() { return prepare_batch(); }

  void process_batch_output_test() {
    process_batch_output(running_requests_, running_sequences_);
  }

  void wait_for_responses() { response_processor_->wait_completion(); }
};

// One engine + scheduler for the whole process. Google Benchmark invokes each
// BM_* function several times per argument while sizing the run, and tearing
// down a BlockManagerPool with the prefix cache enabled sleeps for seconds in
// PrefixCache's destructor; every iteration leaves the scheduler empty, so
// sharing one instance is safe.
class SchedulerFixture final {
 public:
  static BenchContinuousScheduler& scheduler() {
    static SchedulerFixture fixture;
    return fixture.scheduler_;
  }

 private:
  SchedulerFixture() : scheduler_(&engine_, make_scheduler_options()) {}

  static ContinuousScheduler::Options make_scheduler_options() {
    ContinuousScheduler::Options options;
    options.max_tokens_per_batch(kMaxTokensPerBatch)
        .max_seqs_per_batch(kMaxSeqsPerBatch)
        .max_tokens_per_chunk_for_prefill(kMaxTokensPerBatch)
        .num_speculative_tokens(0)
        .dp_size(1)
        .enable_schedule_overlap(false);
    return options;
  }

  // Declared before the scheduler so it outlives it.
  FakeEngine engine_;
  BenchContinuousScheduler scheduler_;
};

OutputFunc noop_output() {
  return [](const RequestOutput& /*output*/) { return true; };
}

// Token ids that do not repeat for ~4M requests (ids are never fed to a model
// or tokenizer here), so no request can hit the prefix cache with an earlier
// request's prompt.
std::vector<int32_t> next_prompt_tokens() {
  static int64_t next_token = 0;
  std::vector<int32_t> tokens(kPromptTokens);
  for (int32_t& token : tokens) {
    token = static_cast<int32_t>(next_token++ %
                                 std::numeric_limits<int32_t>::max());
  }
  return tokens;
}

std::shared_ptr<Request> make_request(int32_t max_generated_tokens,
                                      bool stream) {
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(max_generated_tokens);
  stopping_checker.set_max_context_len(kMaxContextLen);
  stopping_checker.set_ignore_eos(true);

  RequestState state(/*prompt=*/"bench",
                     next_prompt_tokens(),
                     RequestSamplingParam{},
                     SchedulerParam{},
                     std::move(stopping_checker),
                     /*seq_capacity=*/kPromptTokens + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     stream,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     noop_output(),
                     OutputsFunc{});
  return std::make_shared<Request>(
      "bench-req", "x-rid", "x-rtime", std::move(state), "");
}

std::vector<std::shared_ptr<Request>> make_requests(
    size_t num_requests,
    int32_t max_generated_tokens,
    bool stream = false) {
  std::vector<std::shared_ptr<Request>> requests;
  requests.reserve(num_requests);
  for (size_t i = 0; i < num_requests; ++i) {
    requests.emplace_back(make_request(max_generated_tokens, stream));
  }
  return requests;
}

void add_requests(BenchContinuousScheduler& scheduler,
                  std::vector<std::shared_ptr<Request>>& requests) {
  for (auto& request : requests) {
    const bool added = scheduler.add_request(request);
    CHECK(added) << "scheduler rejected a benchmark request";
  }
}

// Simulates the engine having produced one token for every sequence: the KV
// cache now covers everything that was scheduled, and one token was sampled.
void append_one_token(const std::vector<std::shared_ptr<Request>>& requests) {
  for (const auto& request : requests) {
    for (auto& sequence : request->sequences()) {
      sequence->kv_state().set_kv_cache_tokens_num(sequence->num_tokens());
      sequence->append_token(Token(/*id=*/1));
    }
  }
}

// Retires requests that have hit their max_generated_tokens: prepare_batch's
// collect_finished step frees their blocks and dispatches the completion
// callbacks, which are then awaited so the next iteration starts clean.
void retire_finished(BenchContinuousScheduler& scheduler) {
  std::vector<Batch> batches = scheduler.prepare_batch_test();
  CHECK(batches.empty() || batches[0].empty())
      << "unfinished requests leaked into the next iteration";
  scheduler.wait_for_responses();
}

void BM_Scheduler_AddRequest(benchmark::State& state) {
  const size_t num_requests = static_cast<size_t>(state.range(0));
  BenchContinuousScheduler& scheduler = SchedulerFixture::scheduler();

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::shared_ptr<Request>> requests =
        make_requests(num_requests, /*max_generated_tokens=*/1);
    state.ResumeTiming();

    add_requests(scheduler, requests);

    state.PauseTiming();
    (void)scheduler.prepare_batch_test();
    append_one_token(requests);
    retire_finished(scheduler);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_requests));
}

void BM_Scheduler_PrepareBatch_Prefill(benchmark::State& state) {
  const size_t num_requests = static_cast<size_t>(state.range(0));
  BenchContinuousScheduler& scheduler = SchedulerFixture::scheduler();

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::shared_ptr<Request>> requests =
        make_requests(num_requests, /*max_generated_tokens=*/1);
    add_requests(scheduler, requests);
    state.ResumeTiming();

    std::vector<Batch> batches = scheduler.prepare_batch_test();
    do_not_optimize(batches.data());

    state.PauseTiming();
    CHECK_EQ(batches[0].size(), num_requests) << "prefill batch is incomplete";
    append_one_token(requests);
    retire_finished(scheduler);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_requests));
}

void BM_Scheduler_PrepareBatch_Decode(benchmark::State& state) {
  const size_t num_requests = static_cast<size_t>(state.range(0));
  BenchContinuousScheduler& scheduler = SchedulerFixture::scheduler();

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::shared_ptr<Request>> requests =
        make_requests(num_requests, /*max_generated_tokens=*/2);
    add_requests(scheduler, requests);
    (void)scheduler.prepare_batch_test();  // prefill step
    append_one_token(requests);            // first token sampled
    state.ResumeTiming();

    std::vector<Batch> batches = scheduler.prepare_batch_test();
    do_not_optimize(batches.data());

    state.PauseTiming();
    CHECK_EQ(batches[0].size(), num_requests) << "decode batch is incomplete";
    append_one_token(requests);  // second token -> finished
    retire_finished(scheduler);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_requests));
}

// range(0): running requests, range(1): stream (0/1).
void BM_Scheduler_ProcessBatchOutput(benchmark::State& state) {
  const size_t num_requests = static_cast<size_t>(state.range(0));
  const bool stream = state.range(1) != 0;
  BenchContinuousScheduler& scheduler = SchedulerFixture::scheduler();

  for (auto _ : state) {
    state.PauseTiming();
    std::vector<std::shared_ptr<Request>> requests =
        make_requests(num_requests, /*max_generated_tokens=*/2, stream);
    add_requests(scheduler, requests);
    (void)scheduler.prepare_batch_test();  // prefill step
    append_one_token(requests);            // first token sampled
    state.ResumeTiming();

    // Stream requests are dispatched to the response thread pool here; the
    // dispatch is timed, the asynchronous decode + callback is not.
    scheduler.process_batch_output_test();

    state.PauseTiming();
    scheduler.wait_for_responses();
    append_one_token(requests);  // second token -> finished
    retire_finished(scheduler);
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_requests));
}

BENCHMARK(BM_Scheduler_AddRequest)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scheduler_PrepareBatch_Prefill)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scheduler_PrepareBatch_Decode)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Scheduler_ProcessBatchOutput)
    ->ArgsProduct({{16, 64, 256}, {0, 1}})
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm
