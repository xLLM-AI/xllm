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

// Hop 5 of the request path: sampled tokens -> RequestOutput.
//
// After Batch::process_sample_output has appended the sampled tokens,
// RequestOutputDispatcher turns sequences into RequestOutputs on its response
// thread pool: Sequence::generate_streaming_output for every stream step
// (incremental detokenization of the new suffix) and Request::generate_output
// once a non-stream request finishes (full detokenization + usage). These
// benchmarks time that conversion with a tokenizer whose decode is O(tokens)
// but otherwise trivial, so they measure xLLM's own bookkeeping rather than a
// real tokenizer.
//
//   * BM_Sequence_GenerateStreamingOutput - one stream step that has to emit
//                                           `k` newly sampled tokens.
//   * BM_Request_GenerateOutput           - final output for a request with
//                                           `n` generated tokens, logprobs
//                                           on/off.
//
// Build & run (example):
//   python setup.py test --test-name request_output_benchmark
//   ./request_output_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "core/common/types.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_state.h"
#include "core/framework/request/sequence.h"
#include "core/framework/request/stopping_checker.h"
#include "core/framework/sampling/sampling_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/slice.h"

namespace xllm {
namespace {

constexpr size_t kPromptTokens = 512;
constexpr int32_t kMaxContextLen = 1 << 20;

// Decodes every token to one character: O(tokens) like a real detokenizer,
// without the vocabulary lookups.
class FakeTokenizer final : public Tokenizer {
 public:
  std::string decode(const Slice<int32_t>& ids,
                     bool /*skip_special_tokens*/) const override {
    return std::string(ids.size(), 'a');
  }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

OutputFunc noop_output() {
  return [](const RequestOutput& /*output*/) { return true; };
}

std::unique_ptr<Request> make_request(size_t num_generated_tokens,
                                      bool stream,
                                      bool logprobs) {
  RequestSamplingParam sampling_param;
  sampling_param.logprobs = logprobs;
  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(1 << 20);
  stopping_checker.set_max_context_len(kMaxContextLen);
  stopping_checker.set_ignore_eos(true);

  RequestState state(std::string(kPromptTokens, 'p'),
                     std::vector<int32_t>(kPromptTokens, 7),
                     std::move(sampling_param),
                     SchedulerParam{},
                     std::move(stopping_checker),
                     /*seq_capacity=*/kPromptTokens + num_generated_tokens + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     logprobs,
                     stream,
                     /*echo=*/false,
                     /*skip_special_tokens=*/true,
                     /*enable_schedule_overlap=*/false,
                     noop_output(),
                     OutputsFunc{});
  return std::make_unique<Request>(
      "bench-req", "x-rid", "x-rtime", std::move(state), "");
}

// Appends `count` sampled tokens to the request's only sequence, carrying a
// logprob when the request asked for them (as the sampler would).
void append_generated_tokens(Request& request, size_t count, bool logprobs) {
  Sequence& sequence = *request.sequences()[0];
  // append_token CHECKs that the prompt already has KV cache, as it does after
  // the real prefill step.
  sequence.kv_state().set_kv_cache_tokens_num(sequence.num_prompt_tokens());
  for (size_t i = 0; i < count; ++i) {
    Token token(static_cast<int64_t>(100 + i));
    if (logprobs) {
      token.logprob = -0.5f;
    }
    sequence.append_token(token);
  }
}

// range(0): newly sampled tokens emitted by this stream step.
void BM_Sequence_GenerateStreamingOutput(benchmark::State& state) {
  const size_t new_tokens = static_cast<size_t>(state.range(0));
  const FakeTokenizer tokenizer;

  for (auto _ : state) {
    // A stream step decodes only the suffix appended since the previous step;
    // a fresh sequence with `new_tokens` appended is exactly that state.
    state.PauseTiming();
    std::unique_ptr<Request> request =
        make_request(new_tokens, /*stream=*/true, /*logprobs=*/false);
    append_generated_tokens(*request, new_tokens, /*logprobs=*/false);
    Sequence& sequence = *request->sequences()[0];
    state.ResumeTiming();

    std::optional<SequenceOutput> output =
        sequence.generate_streaming_output(tokenizer);
    bool has_output = output.has_value();
    benchmark::DoNotOptimize(has_output);

    state.PauseTiming();
    request.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(new_tokens));
}

// range(0): generated tokens, range(1): logprobs (0/1).
void BM_Request_GenerateOutput(benchmark::State& state) {
  const size_t num_generated_tokens = static_cast<size_t>(state.range(0));
  const bool logprobs = state.range(1) != 0;
  const FakeTokenizer tokenizer;

  for (auto _ : state) {
    // generate_output advances the incremental decoder, so a request can only
    // be decoded once; rebuild it outside the timed region.
    state.PauseTiming();
    std::unique_ptr<Request> request =
        make_request(num_generated_tokens, /*stream=*/false, logprobs);
    append_generated_tokens(*request, num_generated_tokens, logprobs);
    state.ResumeTiming();

    RequestOutput output = request->generate_output(tokenizer);
    SequenceOutput* outputs = output.outputs.data();
    benchmark::DoNotOptimize(outputs);

    state.PauseTiming();
    request.reset();
    state.ResumeTiming();
  }
  state.SetItemsProcessed(static_cast<int64_t>(state.iterations()) *
                          static_cast<int64_t>(num_generated_tokens));
}

BENCHMARK(BM_Sequence_GenerateStreamingOutput)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_Request_GenerateOutput)
    ->ArgsProduct({{16, 256, 4096}, {0, 1}})
    ->Unit(benchmark::kMicrosecond);

}  // namespace
}  // namespace xllm
