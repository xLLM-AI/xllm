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

// Hop 4 of the request path: Batch -> ForwardInput -> worker -> output.
//
// Once the scheduler has built a Batch, LLMEngine::step turns it into a
// ForwardInput (ForwardInputBuilder), serialises that for the remote worker
// (packed proto), the worker unpacks it before the H2D copy, and after the
// model + sampler have run the worker serialises the sampled tokens back
// (proto::ForwardOutput), which the engine turns into a RawForwardOutput and
// feeds to Batch::process_sample_output. Everything here is host-side CPU work
// and is timed with CPU tensors; the model forward, sampler and device copies
// are the only steps of the pipeline that are not covered.
//
//   * BM_BlockManagerPool_AllocateRelease - prefix match + block allocation
//                                           + release for one prompt.
//   * BM_Batch_PrepareForwardInput_Prefill - prefill input for one sequence,
//                                            built by ForwardInputBuilder and
//                                            swept over prompt length.
//   * BM_Batch_PrepareForwardInput_Decode  - ForwardInputBuilder for N decode
//                                            sequences.
//   * BM_ForwardInput_ToPackedProto   - engine side: ForwardInput -> proto.
//   * BM_ForwardInput_FromPackedProto - worker side: proto -> ForwardInput
//                                       (lazy) -> unpacked host tensors.
//   * BM_ForwardOutput_ToProto        - worker side: sampled tensors -> proto.
//   * BM_ForwardOutput_FromProto      - engine side: proto -> RawForwardOutput.
//   * BM_Batch_ProcessSampleOutput    - RawForwardOutput -> Sequence::
//                                       append_token for N sequences.
//
// Build & run (example):
//   python setup.py test --test-name batch_benchmark
//   ./batch_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/batch/batch.h"
#include "core/framework/block/block.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/request/incremental_decoder.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/runtime/forward_params.h"
#include "core/runtime/params_utils.h"
#include "core/util/threadpool.h"
#include "worker.pb.h"

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
constexpr uint32_t kNumBlocks = 16384;
constexpr uint32_t kMaxSeqsPerBatch = 1024;
constexpr size_t kDecodePromptTokens = 512;
// LLMEngine sizes its "LLMEngine.forward_input" pool with 16 threads; the
// builder only fans out to it above 65536 query tokens.
constexpr size_t kForwardInputThreads = 16;

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

// One pool for the whole process. Google Benchmark invokes each BM_* function
// several times per argument while sizing the run, and tearing down a pool
// with the prefix cache enabled sleeps for seconds in PrefixCache's
// destructor; every benchmark releases its blocks, so sharing is safe.
BlockManagerPool::Options make_block_manager_options() {
  BlockManagerPool::Options options;
  options.num_blocks(kNumBlocks)
      .block_size(kBlockSize)
      .max_seqs_per_batch(kMaxSeqsPerBatch)
      .enable_prefix_cache(true);
  return options;
}

BlockManagerPool& shared_block_manager_pool() {
  static BlockManagerPool pool(make_block_manager_options(), /*dp_size=*/1);
  return pool;
}

// Owns the per-request parameter objects every Sequence borrows by pointer.
class SequenceFixture final {
 public:
  explicit SequenceFixture(bool logprobs = false) {
    sampling_param_.logprobs = logprobs;
    stopping_checker_.set_max_generated_tokens(1 << 20);
    stopping_checker_.set_ignore_eos(true);
  }

  // Non-const: SequenceParams holds non-owning mutable pointers to the
  // sampling param and stopping checker.
  std::unique_ptr<Sequence> make_sequence(size_t index,
                                          size_t num_prompt_tokens) {
    const std::vector<int32_t> prompt_token_ids =
        next_prompt_tokens(num_prompt_tokens);
    SequenceParams seq_params;
    seq_params.seq_capacity = num_prompt_tokens + 64;
    seq_params.stopping_checker = &stopping_checker_;
    seq_params.sampling_param = &sampling_param_;
    seq_params.skip_special_tokens = true;
    seq_params.echo = false;
    seq_params.logprobs = sampling_param_.logprobs;
    seq_params.enable_schedule_overlap = false;
    IncrementalDecoder decoder(/*prompt=*/"",
                               num_prompt_tokens,
                               /*echo=*/false,
                               /*skip_special_tokens=*/true);
    return std::make_unique<Sequence>(index,
                                      prompt_token_ids,
                                      /*input_embedding=*/torch::Tensor(),
                                      /*mm_data=*/MMData(),
                                      decoder,
                                      seq_params);
  }

 private:
  RequestSamplingParam sampling_param_;
  StoppingChecker stopping_checker_;
};

// Allocates KV blocks for the whole prompt the way the scheduler does for a
// prefill: prefix match first, then fresh blocks for the remainder.
void allocate_prefill(BlockManagerPool& pool, Sequence& sequence) {
  pool.allocate_shared(&sequence);
  const bool allocated = pool.allocate(&sequence);
  CHECK(allocated) << "block manager pool ran out of blocks";
}

// Moves a prefilled sequence into the decode stage: KV cache covers the prompt
// and one token has been sampled.
void advance_to_decode(BlockManagerPool& pool, Sequence& sequence) {
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  sequence.append_token(Token(/*id=*/1));
  const bool allocated = pool.allocate(&sequence);
  CHECK(allocated) << "block manager pool ran out of blocks";
}

// A decode batch of `batch_size` sequences with blocks allocated from a shared
// pool, ready for prepare_forward_input. Blocks are released on destruction so
// the pool can be reused across iterations.
class DecodeBatchFixture final {
 public:
  DecodeBatchFixture(SequenceFixture& sequence_fixture,
                     BlockManagerPool* pool,
                     size_t batch_size)
      : pool_(pool) {
    sequences_.reserve(batch_size);
    for (size_t i = 0; i < batch_size; ++i) {
      sequences_.emplace_back(
          sequence_fixture.make_sequence(i, kDecodePromptTokens));
      allocate_prefill(*pool_, *sequences_.back());
      advance_to_decode(*pool_, *sequences_.back());
      batch_.add(sequences_.back().get());
    }
  }

  ~DecodeBatchFixture() {
    for (auto& sequence : sequences_) {
      pool_->deallocate(sequence.get());
    }
  }

  DecodeBatchFixture(const DecodeBatchFixture&) = delete;
  DecodeBatchFixture& operator=(const DecodeBatchFixture&) = delete;

  Batch& batch() { return batch_; }

  // prepare_forward_input advances kv_cache_tokens_num by the tokens it
  // schedules; rewind so the next call sees the same decode step again.
  void rewind_kv_cache() {
    for (auto& sequence : sequences_) {
      sequence->kv_state().set_kv_cache_tokens_num(
          sequence->num_prompt_tokens());
    }
  }

 private:
  BlockManagerPool* pool_ = nullptr;
  std::vector<std::unique_ptr<Sequence>> sequences_;
  Batch batch_;
};

ForwardInput build_decode_forward_input(size_t batch_size) {
  SequenceFixture sequence_fixture;
  BlockManagerPool& pool = shared_block_manager_pool();
  DecodeBatchFixture fixture(sequence_fixture, &pool, batch_size);
  ThreadPool thread_pool(kForwardInputThreads);
  return fixture.batch().prepare_forward_input(ModelArgs(), &thread_pool);
}

RawForwardOutput make_raw_output(size_t batch_size, bool logprobs) {
  RawForwardOutput raw_output;
  raw_output.outputs.reserve(batch_size);
  for (size_t i = 0; i < batch_size; ++i) {
    RawToken token;
    token.id = static_cast<int64_t>(100 + i);
    if (logprobs) {
      token.logprob = -0.5f;
    }
    RawSampleOutput sample_output;
    sample_output.tokens.emplace_back(std::move(token));
    raw_output.outputs.emplace_back(std::move(sample_output));
  }
  return raw_output;
}

// --------------------------------------------------------------------------
// Scheduler-side block allocation
// --------------------------------------------------------------------------

void BM_BlockManagerPool_AllocateRelease(benchmark::State& state) {
  const size_t num_prompt_tokens = static_cast<size_t>(state.range(0));
  SequenceFixture sequence_fixture;
  BlockManagerPool& pool = shared_block_manager_pool();

  for (auto _ : state) {
    state.PauseTiming();
    std::unique_ptr<Sequence> sequence =
        sequence_fixture.make_sequence(/*index=*/0, num_prompt_tokens);
    state.ResumeTiming();

    allocate_prefill(pool, *sequence);
    do_not_optimize(sequence->kv_state().num_blocks(BlockType::KV));
    pool.deallocate(sequence.get());

    state.PauseTiming();
    sequence.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_prompt_tokens));
}

// --------------------------------------------------------------------------
// Engine side: Batch -> ForwardInput -> packed proto
// --------------------------------------------------------------------------

void BM_Batch_PrepareForwardInput_Prefill(benchmark::State& state) {
  const size_t num_prompt_tokens = static_cast<size_t>(state.range(0));
  SequenceFixture sequence_fixture;
  BlockManagerPool& pool = shared_block_manager_pool();
  std::unique_ptr<Sequence> sequence =
      sequence_fixture.make_sequence(/*index=*/0, num_prompt_tokens);
  allocate_prefill(pool, *sequence);
  ThreadPool thread_pool(kForwardInputThreads);
  const ModelArgs args;

  for (auto _ : state) {
    // The builder advances kv_cache_tokens_num past the prompt; rewind so every
    // iteration builds the same full prefill.
    sequence->kv_state().set_kv_cache_tokens_num(0);
    Batch batch(sequence.get());
    ForwardInput forward_input =
        batch.prepare_forward_input(args, &thread_pool);
    do_not_optimize(forward_input.token_ids.data_ptr());
  }
  // Return the blocks to the shared pool before the sequence goes away.
  pool.deallocate(sequence.get());
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_prompt_tokens));
}

void BM_Batch_PrepareForwardInput_Decode(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  SequenceFixture sequence_fixture;
  BlockManagerPool& pool = shared_block_manager_pool();
  DecodeBatchFixture fixture(sequence_fixture, &pool, batch_size);
  ThreadPool thread_pool(kForwardInputThreads);
  const ModelArgs args;

  for (auto _ : state) {
    fixture.rewind_kv_cache();
    ForwardInput forward_input =
        fixture.batch().prepare_forward_input(args, &thread_pool);
    do_not_optimize(forward_input.token_ids.data_ptr());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(batch_size));
}

void BM_ForwardInput_ToPackedProto(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const ForwardInput forward_input = build_decode_forward_input(batch_size);
  proto::PackedForwardInput packed_input;

  for (auto _ : state) {
    packed_input.Clear();
    const bool ok = forward_input_to_packed_proto(forward_input, &packed_input);
    do_not_optimize(ok);
    do_not_optimize(packed_input.ByteSizeLong());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(batch_size));
}

// --------------------------------------------------------------------------
// Worker side: packed proto -> ForwardInput (host tensors)
// --------------------------------------------------------------------------

void BM_ForwardInput_FromPackedProto(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const ForwardInput forward_input = build_decode_forward_input(batch_size);
  proto::PackedForwardInput packed_input;
  CHECK(forward_input_to_packed_proto(forward_input, &packed_input));
  const torch::Device cpu(torch::kCPU);

  for (auto _ : state) {
    // WorkerService::ExecuteModel: proto -> lazily unpacked ForwardInput ...
    ForwardInput lazy_input;
    packed_proto_to_forward_input(packed_input, lazy_input, cpu, nullptr);
    // ... which the worker materialises into individual host tensors before
    // the (device-only, not benchmarked) H2D copy.
    ForwardInput unpacked_input;
    const bool ok = detail::unpack_from_input_host_buffer(
        lazy_input,
        cpu,
        torch::kFloat32,
        unpacked_input,
        /*materialize_device_buffer=*/false);
    do_not_optimize(ok);
    do_not_optimize(unpacked_input.token_ids.data_ptr());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(batch_size));
}

// --------------------------------------------------------------------------
// Worker side: sampled tokens -> proto::ForwardOutput
// --------------------------------------------------------------------------

// range(0): batch size, range(1): logprobs (0/1).
void BM_ForwardOutput_ToProto(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  const bool logprobs = state.range(1) != 0;
  const torch::Tensor next_tokens =
      torch::arange(batch_size, torch::dtype(torch::kInt64));
  const torch::Tensor token_logprobs =
      logprobs ? torch::full({batch_size}, -0.5f, torch::dtype(torch::kFloat32))
               : torch::Tensor();
  proto::ForwardOutput pb_output;

  for (auto _ : state) {
    pb_output.Clear();
    forward_output_to_proto(next_tokens,
                            token_logprobs,
                            /*top_tokens=*/torch::Tensor(),
                            /*top_logprobs=*/torch::Tensor(),
                            /*embeddings=*/torch::Tensor(),
                            /*mm_embeddings=*/{},
                            /*speculative_token_stats=*/{},
                            /*expert_load_data=*/torch::Tensor(),
                            /*prepared_token=*/-1,
                            /*src_seq_idxes=*/torch::Tensor(),
                            /*out_tokens=*/torch::Tensor(),
                            /*out_logprobs=*/torch::Tensor(),
                            /*dit_images=*/{},
                            /*dit_text_output=*/{},
                            /*json_object_errors=*/{},
                            &pb_output);
    do_not_optimize(pb_output.outputs_size());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          batch_size);
}

// --------------------------------------------------------------------------
// Engine side: proto::ForwardOutput -> RawForwardOutput -> Sequence
// --------------------------------------------------------------------------

void BM_ForwardOutput_FromProto(benchmark::State& state) {
  const int64_t batch_size = state.range(0);
  const bool logprobs = state.range(1) != 0;
  const torch::Tensor next_tokens =
      torch::arange(batch_size, torch::dtype(torch::kInt64));
  const torch::Tensor token_logprobs =
      logprobs ? torch::full({batch_size}, -0.5f, torch::dtype(torch::kFloat32))
               : torch::Tensor();
  proto::ForwardOutput pb_output;
  forward_output_to_proto(next_tokens,
                          token_logprobs,
                          /*top_tokens=*/torch::Tensor(),
                          /*top_logprobs=*/torch::Tensor(),
                          /*embeddings=*/torch::Tensor(),
                          /*mm_embeddings=*/{},
                          /*speculative_token_stats=*/{},
                          /*expert_load_data=*/torch::Tensor(),
                          /*prepared_token=*/-1,
                          /*src_seq_idxes=*/torch::Tensor(),
                          /*out_tokens=*/torch::Tensor(),
                          /*out_logprobs=*/torch::Tensor(),
                          /*dit_images=*/{},
                          /*dit_text_output=*/{},
                          /*json_object_errors=*/{},
                          &pb_output);

  for (auto _ : state) {
    RawForwardOutput raw_output;
    proto_to_forward_output(pb_output, raw_output);
    do_not_optimize(raw_output.outputs.data());
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          batch_size);
}

// range(0): batch size, range(1): logprobs (0/1).
void BM_Batch_ProcessSampleOutput(benchmark::State& state) {
  const size_t batch_size = static_cast<size_t>(state.range(0));
  const bool logprobs = state.range(1) != 0;
  const RawForwardOutput raw_output = make_raw_output(batch_size, logprobs);
  SequenceFixture sequence_fixture(logprobs);
  BlockManagerPool& pool = shared_block_manager_pool();
  ThreadPool thread_pool(kForwardInputThreads);
  const ModelArgs args;

  for (auto _ : state) {
    // process_sample_output appends to the sequences and consumes the output
    // targets recorded by the preceding prepare_forward_input, so the batch is
    // rebuilt per iteration (outside the timed region).
    state.PauseTiming();
    auto fixture = std::make_unique<DecodeBatchFixture>(
        sequence_fixture, &pool, batch_size);
    (void)fixture->batch().prepare_forward_input(args, &thread_pool);
    state.ResumeTiming();

    fixture->batch().process_sample_output(raw_output,
                                           /*replace_fake_token=*/false);

    state.PauseTiming();
    fixture.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(batch_size));
}

// Prompt length in tokens.
BENCHMARK(BM_BlockManagerPool_AllocateRelease)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batch_PrepareForwardInput_Prefill)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kMicrosecond);
// Decode batch size in sequences.
BENCHMARK(BM_Batch_PrepareForwardInput_Decode)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ForwardInput_ToPackedProto)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ForwardInput_FromPackedProto)
    ->RangeMultiplier(4)
    ->Range(1, 256)
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ForwardOutput_ToProto)
    ->ArgsProduct({{1, 16, 256}, {0, 1}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_ForwardOutput_FromProto)
    ->ArgsProduct({{1, 16, 256}, {0, 1}})
    ->Unit(benchmark::kMicrosecond);
BENCHMARK(BM_Batch_ProcessSampleOutput)
    ->ArgsProduct({{1, 16, 256}, {0, 1}})
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm
