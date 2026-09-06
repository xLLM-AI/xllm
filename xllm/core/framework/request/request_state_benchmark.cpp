/* Copyright 2025-2026 The xLLM Authors.

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

// Micro-benchmark for RequestState construction: COPY path vs MOVE path.
//
// The RequestState constructors take prompt / prompt_tokens / stopping_checker
// (etc.) by value and std::move them into members. When the caller hands in an
// rvalue (std::move), the heap-owning payload is moved (O(1)); when it hands in
// an lvalue, it is deep-copied (O(n)). This benchmark makes that cost visible:
//
//   * BM_RequestState_Copy  - passes lvalues, so prompt + prompt_tokens are
//                             deep-copied on every construction.
//   * BM_RequestState_Move  - passes rvalues (std::move), so they are moved.
//
// Sweeping prompt_tokens length shows the COPY path scaling with the payload
// size while the MOVE path stays flat -- i.e. the concrete win of the
// by-value+std::move constructor change (and of callers using std::move).
//
// Build & run (example):
//   ./request_state_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>

#include <cstdint>
#include <string>
#include <vector>

#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_state.h"

namespace xllm {
namespace {

// A prompt long enough to defeat std::string small-string-optimization so the
// copy path actually allocates + copies a heap buffer.
constexpr size_t kPromptChars = 2048;

OutputFunc NoopOutput() {
  return [](const RequestOutput&) { return true; };
}

// Builds a RequestState carrying a heap prompt + `n` prompt tokens, used as the
// source for the Request-construction benchmarks below.
RequestState MakeRequestState(size_t n) {
  return RequestState(std::string(kPromptChars, 'x'),
                      std::vector<int32_t>(n, 7),
                      RequestSamplingParam{},
                      SchedulerParam{},
                      StoppingChecker{},
                      /*seq_capacity=*/n + 8,
                      /*n=*/1,
                      /*best_of=*/1,
                      /*logprobs=*/false,
                      /*stream=*/false,
                      /*echo=*/false,
                      /*skip_special_tokens=*/true,
                      /*enable_schedule_overlap=*/false,
                      NoopOutput(),
                      OutputsFunc{});
}

void BM_RequestState_Copy(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  const std::string prompt(kPromptChars, 'x');
  const std::vector<int32_t> tokens(n, 7);

  for (auto _ : state) {
    // Lvalue arguments: the by-value parameters copy-construct from these, so
    // prompt + prompt_tokens are deep-copied every iteration.
    RequestState s(prompt,
                   tokens,
                   RequestSamplingParam{},
                   SchedulerParam{},
                   StoppingChecker{},
                   /*seq_capacity=*/n + 8,
                   /*n=*/1,
                   /*best_of=*/1,
                   /*logprobs=*/false,
                   /*stream=*/false,
                   /*echo=*/false,
                   /*skip_special_tokens=*/true,
                   /*enable_schedule_overlap=*/false,
                   NoopOutput(),
                   OutputsFunc{});
    benchmark::DoNotOptimize(s.prompt.data());
    benchmark::DoNotOptimize(s.prompt_tokens.data());
  }
}

void BM_RequestState_Move(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));

  for (auto _ : state) {
    // Building the sources is excluded from timing; only the move-in is timed.
    state.PauseTiming();
    std::string prompt(kPromptChars, 'x');
    std::vector<int32_t> tokens(n, 7);
    state.ResumeTiming();

    RequestState s(std::move(prompt),
                   std::move(tokens),
                   RequestSamplingParam{},
                   SchedulerParam{},
                   StoppingChecker{},
                   /*seq_capacity=*/n + 8,
                   /*n=*/1,
                   /*best_of=*/1,
                   /*logprobs=*/false,
                   /*stream=*/false,
                   /*echo=*/false,
                   /*skip_special_tokens=*/true,
                   /*enable_schedule_overlap=*/false,
                   NoopOutput(),
                   OutputsFunc{});
    benchmark::DoNotOptimize(s.prompt.data());
    benchmark::DoNotOptimize(s.prompt_tokens.data());
  }
  // Intentionally NO SetBytesProcessed here: the timed move only transfers
  // small-object metadata (pointer/size/capacity), not the prompt/token
  // contents. Multiplying by the payload size would report physically
  // meaningless, ever-increasing bandwidth for what is constant work.
}

// ---------------------------------------------------------------------------
// Request construction: COPY state vs MOVE state.
//
// Request's constructor takes RequestState by value and std::move-s it into its
// member (and, transitively, SequencesGroup takes SequenceParams by value +
// std::move). RequestState is move-only, so a deep copy must be requested
// explicitly via clone(); handing over ownership with std::move avoids it.
// Both paths run the same create_sequences_group() work, so the delta between
// them isolates the RequestState copy the by-value+std::move change removed
// (matching what the request factories do via std::move(req_state)).
//
//   * BM_Request_CopyState - clone()s the state first (deep copy), then moves.
//   * BM_Request_MoveState - passes an rvalue (std::move), so it is moved.
// ---------------------------------------------------------------------------

void BM_Request_CopyState(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));
  const RequestState src = MakeRequestState(n);

  for (auto _ : state) {
    // Explicit deep copy of the reusable template state (RequestState is
    // move-only), then moved into the request. This is the cost the by-value +
    // std::move constructor change avoids whenever the caller can hand over
    // ownership instead of cloning.
    Request req("bench-req", "", "", src.clone());
    benchmark::DoNotOptimize(req.state().prompt.data());
    benchmark::DoNotOptimize(req.state().prompt_tokens.data());
  }
}

void BM_Request_MoveState(benchmark::State& state) {
  const size_t n = static_cast<size_t>(state.range(0));

  for (auto _ : state) {
    // Building the source state is excluded from timing; only the request
    // construction (which moves the state in) is timed.
    state.PauseTiming();
    RequestState src = MakeRequestState(n);
    state.ResumeTiming();

    Request req("bench-req", "", "", std::move(src));
    benchmark::DoNotOptimize(req.state().prompt.data());
    benchmark::DoNotOptimize(req.state().prompt_tokens.data());
  }
}

// Sweep prompt_tokens length from 128 to 128K tokens.
BENCHMARK(BM_RequestState_Copy)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_RequestState_Move)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Request_CopyState)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Request_MoveState)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace xllm
