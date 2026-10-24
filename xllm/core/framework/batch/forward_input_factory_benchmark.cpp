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

// Host-side LLMEngine input preparation, including per-rank builders and DP
// metadata aggregation. Each case reuses the engine-owned factory/thread pool.
// Sequence construction, KV block allocation, state restoration and output
// destruction are excluded; no model execution or device copies are measured.
//
// Build & run (example on NPU):
//   python setup.py test --device npu \
//       --test-name forward_input_factory_benchmark
// Run the generated executable with --benchmark_min_time=0.2s to set duration.

#include <benchmark/benchmark.h>
#include <glog/logging.h>

#if defined(USE_NPU)
#include <Python.h>
#include <acl/acl.h>
#endif

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/batch/batch_group.h"
#include "core/framework/batch/forward_input_factory.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/kv_cache/block/block_manager_impl.h"

namespace {

#if defined(USE_NPU)
// Pinned CPU allocations enter torch_npu, which requires an initialized Python
// runtime. Match the NPU test environment without including gtest or changing
// production startup configuration. This entire lifetime is outside timing.
class NpuBenchmarkRuntime final {
 public:
  NpuBenchmarkRuntime() {
    const aclError acl_ret = aclInit(nullptr);
    CHECK(acl_ret == ACL_SUCCESS || acl_ret == ACL_ERROR_INTERNAL_ERROR)
        << "aclInit failed with error " << acl_ret;
    CHECK_EQ(setenv("TORCH_DEVICE_BACKEND_AUTOLOAD", "0", /*overwrite=*/1), 0);
    Py_InitializeEx(/*initsigs=*/0);
    const int import_ret = PyRun_SimpleString(
        "import os\n"
        "import torch\n"
        "_orig_get_accelerator = torch._C._get_accelerator\n"
        "try:\n"
        "    torch._C._get_accelerator = lambda: torch.device('cpu')\n"
        "    import torch_npu\n"
        "finally:\n"
        "    torch._C._get_accelerator = _orig_get_accelerator\n"
        "import torch_npu.npu as _npu_mod\n"
        "torch_npu._C._npu_init()\n"
        "_npu_mod._initialized = True\n"
        "_npu_mod._original_pid = os.getpid()\n");
    CHECK_EQ(import_ret, 0) << "torch_npu benchmark runtime init failed";
    saved_thread_state_ = PyEval_SaveThread();
  }

  NpuBenchmarkRuntime(const NpuBenchmarkRuntime&) = delete;
  NpuBenchmarkRuntime& operator=(const NpuBenchmarkRuntime&) = delete;

  ~NpuBenchmarkRuntime() {
    PyEval_RestoreThread(saved_thread_state_);
    const int shutdown_ret = PyRun_SimpleString("torch_npu._npu_shutdown()\n");
    CHECK_EQ(shutdown_ret, 0) << "torch_npu benchmark runtime shutdown failed";
    // Keep Python alive and retain the GIL for torch_npu's global destructors,
    // matching the NPU test environment's process-exit lifecycle.
  }

 private:
  PyThreadState* saved_thread_state_ = nullptr;
};
#endif

}  // namespace

namespace xllm {
namespace {

constexpr uint32_t kBlockSize = 128;

enum class Workload { PREFILL, DECODE, MIXED, EMPTY_RANK };

BlockManager::Options make_block_options(size_t sequence_count,
                                         size_t prompt_tokens) {
  const size_t blocks_per_sequence =
      (prompt_tokens + 1 + kBlockSize - 1) / kBlockSize;
  BlockManager::Options options;
  options.num_blocks(
      static_cast<uint32_t>(sequence_count * blocks_per_sequence + 1));
  options.block_size(kBlockSize).enable_prefix_cache(false);
  return options;
}

// Owns the parameters and managers borrowed by Sequence/Block. Declaration
// order keeps them alive until both sequences and saved KV states are gone.
class BatchGroupFixture final {
 public:
  BatchGroupFixture(size_t dp_size,
                    size_t sequences_per_rank,
                    size_t prompt_tokens,
                    Workload workload)
      : batches_(dp_size),
        token_counts_(dp_size, /*value=*/0),
        sequence_counts_(dp_size, /*value=*/0) {
    stopping_checker_.set_max_generated_tokens(/*max_generated_tokens=*/32);
    stopping_checker_.set_ignore_eos(/*ignore_eos=*/true);
    managers_.reserve(dp_size);
    sequences_.reserve(dp_size * sequences_per_rank);
    initial_kv_states_.reserve(dp_size * sequences_per_rank);
    for (size_t rank = 0; rank < dp_size; ++rank) {
      if (workload == Workload::EMPTY_RANK && rank == dp_size - 1) {
        continue;
      }
      managers_.emplace_back(std::make_unique<BlockManagerImpl>(
          make_block_options(sequences_per_rank, prompt_tokens)));
      Batch& batch = batches_[rank];
      batch.reserve(sequences_per_rank, /*group_count=*/0);
      const size_t prefill_count = workload == Workload::PREFILL
                                       ? sequences_per_rank
                                       : (workload == Workload::MIXED ? 1 : 0);
      add_sequences(batch,
                    *managers_.back(),
                    prefill_count,
                    prompt_tokens,
                    /*decode=*/false);
      add_sequences(batch,
                    *managers_.back(),
                    sequences_per_rank - prefill_count,
                    prompt_tokens,
                    /*decode=*/true);
      token_counts_[rank] = static_cast<int32_t>(
          prefill_count * prompt_tokens + sequences_per_rank - prefill_count);
      sequence_counts_[rank] = static_cast<int32_t>(sequences_per_rank);
    }
  }

  BatchGroup& batches() { return batches_; }

  void restore() {
    for (size_t i = 0; i < sequences_.size(); ++i) {
      // Restore all cursor fields, including the legacy slice-window cursor,
      // so repeated runs always prepare the same scheduled query tokens.
      sequences_[i]->kv_state() = initial_kv_states_[i];
    }
  }

  void validate(const std::vector<LlmForwardInput>& inputs,
                bool is_graph_warmup,
                Workload workload) const {
    CHECK(!is_graph_warmup);
    CHECK_EQ(inputs.size(), batches_.size());
    const BatchForwardType::Value forward_type =
        workload == Workload::PREFILL
            ? BatchForwardType::PREFILL
            : (workload == Workload::MIXED ? BatchForwardType::MIXED
                                           : BatchForwardType::DECODE);
    for (size_t rank = 0; rank < inputs.size(); ++rank) {
      const auto& input = inputs[rank];
      CHECK_EQ(input.host_token_ids().numel(), token_counts_[rank]);
      CHECK_EQ(input.input_params.meta.num_sequences, sequence_counts_[rank]);
      CHECK_EQ(input.input_params.meta.batch_forward_type.value(),
               forward_type);
      CHECK(input.input_params.parallel.dp_global_token_nums == token_counts_);
      CHECK(input.input_params.parallel.dp_global_sequence_nums ==
            sequence_counts_);
    }
    for (const auto& sequence : sequences_) {
      CHECK_EQ(sequence->kv_state().kv_cache_tokens_num(),
               sequence->num_tokens());
    }
  }

  int64_t query_tokens() const {
    return std::accumulate(
        token_counts_.begin(), token_counts_.end(), int64_t{0});
  }

 private:
  std::unique_ptr<Sequence> make_sequence(size_t prompt_tokens, bool decode) {
    const std::vector<int32_t> token_ids(prompt_tokens, /*value=*/7);
    SequenceParams params;
    params.seq_capacity = prompt_tokens + 8;
    params.stopping_checker = &stopping_checker_;
    params.sampling_param = &sampling_param_;
    params.request_id = "benchmark-" + std::to_string(sequences_.size());
    params.skip_special_tokens = true;
    params.enable_schedule_overlap = false;
    IncrementalDecoder decoder(/*prompt=*/"",
                               prompt_tokens,
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    auto sequence = std::make_unique<Sequence>(sequences_.size(),
                                               token_ids,
                                               torch::Tensor(),
                                               MMData(),
                                               std::move(decoder),
                                               params);
    if (decode) {
      sequence->kv_state().set_kv_cache_tokens_num(prompt_tokens);
      sequence->append_token(Token(/*id=*/1));
    }
    return sequence;
  }

  void add_sequences(Batch& batch,
                     BlockManagerImpl& manager,
                     size_t count,
                     size_t prompt_tokens,
                     bool decode) {
    for (size_t i = 0; i < count; ++i) {
      auto sequence = make_sequence(prompt_tokens, decode);
      const size_t num_blocks =
          (sequence->num_tokens() + kBlockSize - 1) / kBlockSize;
      const auto blocks = manager.allocate(num_blocks);
      CHECK_EQ(blocks.size(), num_blocks);
      sequence->add_blocks(BlockType::KV, blocks);
      initial_kv_states_.emplace_back(sequence->kv_state());
      batch.add(sequence.get());
      sequences_.emplace_back(std::move(sequence));
    }
  }

  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
  std::vector<std::unique_ptr<BlockManagerImpl>> managers_;
  std::vector<std::unique_ptr<Sequence>> sequences_;
  std::vector<KVCacheState> initial_kv_states_;
  BatchGroup batches_;
  std::vector<int32_t> token_counts_;
  std::vector<int32_t> sequence_counts_;
};

void benchmark_create_inputs(benchmark::State& state, Workload workload) {
  const uint32_t dp_size = static_cast<uint32_t>(state.range(0));
  const size_t sequences_per_rank = static_cast<size_t>(state.range(1));
  const size_t prompt_tokens = static_cast<size_t>(state.range(2));
  ForwardInputFactoryOptions options;
  options.dp_size = dp_size;
  ForwardInputFactory factory(options);
  BatchGroupFixture fixture(
      dp_size, sequences_per_rank, prompt_tokens, workload);
  const ModelArgs model_args{};
  std::vector<LlmForwardInput> inputs;
  bool is_graph_warmup = false;

  // Warm the thread pool and validate the repeated-call fixture before timing.
  for (size_t i = 0; i < 2; ++i) {
    fixture.restore();
    factory.create_inputs(
        fixture.batches(), model_args, inputs, is_graph_warmup);
    fixture.validate(inputs, is_graph_warmup, workload);
  }
  for (auto _ : state) {
    state.PauseTiming();
    // Production passes a fresh output vector. Release reused storage while
    // timing is paused so allocator cleanup is excluded from the measurement.
    std::vector<LlmForwardInput>().swap(inputs);
    fixture.restore();
    state.ResumeTiming();
    factory.create_inputs(
        fixture.batches(), model_args, inputs, is_graph_warmup);
    benchmark::DoNotOptimize(inputs);
    benchmark::DoNotOptimize(is_graph_warmup);
  }
  fixture.validate(inputs, is_graph_warmup, workload);
  state.SetItemsProcessed(state.iterations() * fixture.query_tokens());
}

void BM_ForwardInputFactory_CreateInputs_Prefill(benchmark::State& state) {
  benchmark_create_inputs(state, Workload::PREFILL);
}

void BM_ForwardInputFactory_CreateInputs_Decode(benchmark::State& state) {
  benchmark_create_inputs(state, Workload::DECODE);
}

void BM_ForwardInputFactory_CreateInputs_Mixed(benchmark::State& state) {
  benchmark_create_inputs(state, Workload::MIXED);
}

void BM_ForwardInputFactory_CreateInputs_EmptyRank(benchmark::State& state) {
  benchmark_create_inputs(state, Workload::EMPTY_RANK);
}

// 16 x 4096 query tokens per rank exercises the builder's thread-pool path.
BENCHMARK(BM_ForwardInputFactory_CreateInputs_Prefill)
    ->Args({1, 1, 128})
    ->Args({1, 1, 4096})
    ->Args({1, 16, 512})
    ->Args({1, 16, 4096})
    ->Args({4, 16, 512})
    ->Args({4, 16, 4096})
    ->ArgNames({"dp_size", "seqs_per_rank", "prompt_tokens"})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ForwardInputFactory_CreateInputs_Decode)
    ->Args({1, 1, 512})
    ->Args({1, 16, 512})
    ->Args({1, 64, 512})
    ->Args({1, 256, 512})
    ->Args({1, 64, 4096})
    ->Args({2, 64, 512})
    ->Args({4, 64, 512})
    ->Args({8, 64, 512})
    ->ArgNames({"dp_size", "seqs_per_rank", "prompt_tokens"})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
// Each active rank has one prefill followed by seqs_per_rank - 1 decodes.
BENCHMARK(BM_ForwardInputFactory_CreateInputs_Mixed)
    ->Args({1, 64, 512})
    ->Args({4, 64, 512})
    ->ArgNames({"dp_size", "seqs_per_rank", "prompt_tokens"})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);
// The last DP rank is empty; other ranks contain decode sequences.
BENCHMARK(BM_ForwardInputFactory_CreateInputs_EmptyRank)
    ->Args({2, 64, 512})
    ->Args({4, 64, 512})
    ->ArgNames({"dp_size", "seqs_per_rank", "prompt_tokens"})
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm

int main(int argc, char** argv) {
  benchmark::Initialize(&argc, argv);
  if (benchmark::ReportUnrecognizedArguments(argc, argv)) {
    return EXIT_FAILURE;
  }
#if defined(USE_NPU)
  NpuBenchmarkRuntime runtime;
#endif
  benchmark::RunSpecifiedBenchmarks();
  benchmark::Shutdown();
  return EXIT_SUCCESS;
}
