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

#include <Python.h>
#include <acl/acl.h>
#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/torch.h>
#include <torch_npu/csrc/aten/NPUGeneratorImpl.h>
#include <torch_npu/csrc/core/npu/NPUCachingAllocator.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include <cstdint>
#include <iostream>
#include <limits>
#include <memory>
#include <nlohmann/json.hpp>
#include <string_view>

#include "core/framework/sampling/npu_sampling_policy.h"
#include "core/framework/sampling/sampler.h"
#include "core/framework/speculative/rejection_sampler.h"
#include "core/platform/npu/startup.h"

namespace xllm {
namespace {

void dump_ablation_layout() {
  const nlohmann::ordered_json layout = {
      {"symbol", "xllm_npu_sampling_ablation_mask"}, {"bytes", 1}};
  std::cout << layout.dump() << '\n';
}

// Bits identify independent ablations: A=1, B=2, C=4, D=8, E=16, G=32.
void configure(int64_t mask) {
  CHECK_GE(mask, 0);
  CHECK_LE(mask, 63);
  xllm_npu_sampling_ablation_mask = static_cast<uint8_t>(mask);
}

void synchronize() {
  CHECK_EQ(aclrtSynchronizeStream(c10_npu::getCurrentNPUStream(0).stream()),
           ACL_SUCCESS);
}

void ordinary_sampling(benchmark::State& state) {
  auto generator = at_npu::detail::getDefaultNPUGenerator(0);
  generator.set_current_seed(42);
  const int64_t rows = state.range(0);
  const int64_t vocab = state.range(1);
  const int64_t mask = state.range(2);
  // 0 seeded, 1 mixed seeded, 2 unseeded, 3 selected logprobs, 4 top-5.
  const int64_t mode = state.range(3);
  const torch::ScalarType dtype =
      state.range(4) == 0 ? torch::kFloat32 : torch::kBFloat16;
  const auto options = torch::TensorOptions().device(torch::kPrivateUse1, 0);
  const torch::Tensor original =
      torch::randn({rows, vocab}, options.dtype(dtype));
  torch::Tensor working = torch::empty_like(original);
  SamplingParameters params;
  params.sampling_mode() =
      mode == 1 ? SamplingMode::MIXED : SamplingMode::RANDOM;
  params.num_random_rows() = mode == 1 ? (rows + 3) / 4 : rows;
  params.has_unseeded_sample() = mode == 2;
  params.logprobs() = mode >= 3;
  params.max_top_logprobs() = mode == 4 ? 5 : 0;
  if (mode != 2) {
    params.seeds() = torch::arange(rows, options.dtype(torch::kInt64));
    params.seed_offsets() = torch::zeros_like(params.seeds());
  }
  params.temperatures() =
      torch::full({rows}, 0.7, options.dtype(torch::kFloat32));
  params.top_k() = torch::full({rows}, 20, options.dtype(torch::kInt64));
  params.top_p() = torch::full({rows}, 0.95, options.dtype(torch::kFloat32));
  if (mode == 1) {
    params.do_sample() =
        torch::arange(rows, options.dtype(torch::kInt64)).remainder(4).eq(0);
    params.temperatures().masked_fill_(params.do_sample().logical_not(), 0);
    params.top_k().masked_fill_(params.do_sample().logical_not(), 0);
    params.top_p().masked_fill_(params.do_sample().logical_not(), 1);
  }
  configure(mask);
  for (int64_t warmup = 0; warmup < 5; ++warmup) {
    working.copy_(original);
    sample(working, params);
  }
  synchronize();
  const int64_t allocated_before =
      c10_npu::NPUCachingAllocator::getDeviceStats(0)
          .allocated_bytes[0]
          .current;
  c10_npu::NPUCachingAllocator::resetPeakStats(0);
  for (auto /*unused*/ _ : state) {
    state.PauseTiming();
    working.copy_(original);
    synchronize();
    state.ResumeTiming();
    const SampleOutput output = sample(working, params);
    void* token_data = output.next_tokens.data_ptr();
    benchmark::DoNotOptimize(token_data);
    synchronize();
  }
  state.counters["peak_extra_bytes"] = static_cast<double>(
      c10_npu::NPUCachingAllocator::getDeviceStats(0).allocated_bytes[0].peak -
      allocated_before);
  state.counters["rows"] = static_cast<double>(rows);
  state.counters["vocab"] = static_cast<double>(vocab);
}

void verification(benchmark::State& state, bool point_mass) {
  auto generator = at_npu::detail::getDefaultNPUGenerator(0);
  generator.set_current_seed(42);
  const int64_t rows = state.range(0);
  const int64_t steps = state.range(1);
  const int64_t mask = state.range(2);
  const int64_t vocab = state.range(3);
  constexpr int64_t kSupport = 16;
  const auto options = torch::TensorOptions().device(torch::kPrivateUse1, 0);
  const torch::Tensor ids =
      torch::arange(kSupport, options.dtype(torch::kInt64))
          .view({1, 1, kSupport})
          .expand({rows, steps, kSupport});
  const torch::Tensor small_logits =
      torch::randn({rows, steps, kSupport}, options);
  const torch::Tensor tokens = small_logits.argmax(-1);
  const torch::Tensor original =
      torch::randn({rows * (steps + 1), vocab}, options);
  torch::Tensor working = torch::empty_like(original);
  SamplingParameters params;
  params.sampling_mode() = SamplingMode::RANDOM;
  params.logprobs() = state.range(4) != 0;
  configure(mask);
  const auto execute = [&]() {
    if (point_mass) {
      return rejection_sample(params, DraftProposal(tokens), working);
    }
    // Include both compact-q production and verification. The dense leg uses
    // one scatter, conservatively cheaper than the worker's per-step scatters.
    const DraftProposal proposal =
        (mask & 16) != 0
            ? DraftProposal(tokens,
                            std::make_unique<SparseDraftDistribution>(
                                ids.clone(), small_logits.clone()))
            : DraftProposal(tokens,
                            torch::full({rows, steps, vocab},
                                        -std::numeric_limits<float>::infinity(),
                                        options)
                                .scatter_(2, ids, small_logits));
    return rejection_sample(params, proposal, working);
  };
  for (int64_t warmup = 0; warmup < 5; ++warmup) {
    working.copy_(original);
    execute();
  }
  synchronize();
  const int64_t allocated_before =
      c10_npu::NPUCachingAllocator::getDeviceStats(0)
          .allocated_bytes[0]
          .current;
  c10_npu::NPUCachingAllocator::resetPeakStats(0);
  for (auto /*unused*/ _ : state) {
    state.PauseTiming();
    working.copy_(original);
    synchronize();
    state.ResumeTiming();
    const SampleOutput output = execute();
    void* token_data = output.next_tokens.data_ptr();
    benchmark::DoNotOptimize(token_data);
    synchronize();
  }
  state.counters["peak_extra_bytes"] = static_cast<double>(
      c10_npu::NPUCachingAllocator::getDeviceStats(0).allocated_bytes[0].peak -
      allocated_before);
  if (!point_mass) {
    state.counters["dense_q_bytes"] =
        static_cast<double>(rows * steps * vocab * 4);
    state.counters["sparse_q_bytes"] =
        static_cast<double>(rows * steps * kSupport * 12);
  }
}

void sparse_verification(benchmark::State& state) {
  verification(state, /*point_mass=*/false);
}

void point_mass_verification(benchmark::State& state) {
  verification(state, /*point_mass=*/true);
}

BENCHMARK(ordinary_sampling)
    ->ArgsProduct({{1, 2, 8, 32, 64, 128, 256},
                   {49152, 248320},
                   {0, 1, 2, 3, 5, 7, 8, 9, 13, 15},
                   {0, 1, 2, 3, 4},
                   {0, 1}})
    ->ArgNames({"B", "V", "mask", "mode", "half"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(sparse_verification)
    ->ArgsProduct({{1, 8, 32, 128},
                   {1, 7},
                   {0, 8, 16, 24, 32, 40, 48, 56},
                   {248320},
                   {0, 1}})
    ->ArgNames({"B", "S", "mask", "V", "logprobs"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
// Keep deterministic draft costs visible even when D/G do not dispatch.
// Mask 0 still uses the current interface; compare executable revisions when
// assessing changes shared by every ablation variant.
BENCHMARK(point_mass_verification)
    ->ArgsProduct({{1, 2, 8, 32}, {1, 7}, {0, 56}, {49152, 248320}, {0, 1}})
    ->ArgNames({"B", "S", "mask", "V", "logprobs"})
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm

int main(int argc, char** argv) {
  if (argc == 2 && std::string_view(argv[1]) == "--sampling-ablation-layout") {
    xllm::dump_ablation_layout();
    return 0;
  }
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return 1;
  }
  pybind11::initialize_interpreter(/*init_signal_handlers=*/false);
  if (!xllm::initialize_npu_startup()) {
    return 1;
  }
  torch::set_num_threads(8);
  PyThreadState* saved_thread_state = PyEval_SaveThread();
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  PyEval_RestoreThread(saved_thread_state);
  return PyRun_SimpleString("torch_npu._npu_shutdown()\n") == 0 ? 0 : 1;
}
