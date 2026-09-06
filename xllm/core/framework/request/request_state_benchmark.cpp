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
  state.SetBytesProcessed(static_cast<int64_t>(state.iterations()) *
                          (static_cast<int64_t>(n) * sizeof(int32_t) +
                           static_cast<int64_t>(kPromptChars)));
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

// Sweep prompt_tokens length from 128 to 128K tokens.
BENCHMARK(BM_RequestState_Copy)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_RequestState_Move)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace xllm
