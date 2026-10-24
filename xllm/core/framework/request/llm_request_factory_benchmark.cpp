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

// Hop 2 of the request path: LLMMaster::handle_request -> LLMRequestFactory.
//
// LLMMaster::handle_request schedules a closure onto the "LLMMaster.request"
// thread pool; that closure runs RequestParams::verify_params and then
// LLMRequestFactory::create (tokenize, build sampling / scheduler params and
// the StoppingChecker, construct RequestState + Request + the first Sequence)
// before handing the Request to the scheduler. These benchmarks time exactly
// that work, with a deterministic char->id tokenizer and a constant chat
// template so the numbers isolate xLLM's own overhead from the tokenizer /
// Jinja library cost.
//
//   * BM_Master_ThreadPoolHandoff        - schedule a closure on a 4-thread
//                                          ThreadPool and wait for it to run:
//                                          the pure hand-off latency of the
//                                          LLMMaster.request pool.
//   * BM_RequestFactory_CreateFromPrompt - verify_params + create() from text
//                                          (tokenize + Request build), swept
//                                          over prompt length.
//   * BM_RequestFactory_CreateFromTokens - same, pre-tokenized prompt (vocab
//                                          scan + Request build), swept over
//                                          token count.
//   * BM_RequestFactory_CreateFromMessages - chat overload (Message copy +
//                                          template + create), swept over the
//                                          number of messages.
//
// Build & run (example):
//   python setup.py test --test-name llm_request_factory_benchmark
//   ./llm_request_factory_benchmark --benchmark_min_time=0.2s

#include <benchmark/benchmark.h>

#include <cstddef>
#include <cstdint>
#include <future>
#include <memory>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "core/common/message.h"
#include "core/common/options.h"
#include "core/common/rate_limiter.h"
#include "core/framework/chat_template/chat_template.h"
#include "core/framework/model/model_args.h"
#include "core/framework/request/llm_request_factory.h"
#include "core/framework/request/request.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/threadpool.h"

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

constexpr int32_t kVocabSize = 32000;
constexpr int32_t kMaxPositionEmbeddings = 1 << 20;
// Matches the default size of the LLMMaster.request pool (Options::
// num_request_handling_threads).
constexpr size_t kMasterRequestThreads = 4;

// Deterministic tokenizer: one token per character. Cheap on purpose, so the
// measured time is the factory's own work rather than a real BPE encode.
class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool /*add_special_tokens*/ = true) const override {
    ids->clear();
    ids->reserve(text.size());
    for (const char c : text) {
      ids->emplace_back(static_cast<int32_t>(static_cast<unsigned char>(c)) %
                        kVocabSize);
    }
    return true;
  }

  size_t vocab_size() const override { return static_cast<size_t>(kVocabSize); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>(*this);
  }
};

// Renders every conversation to the same fixed prompt so the chat benchmark
// measures the messages path without a Jinja engine in the loop.
class FakeChatTemplate final : public ChatTemplate {
 public:
  std::optional<std::string> apply(
      const ChatMessages& messages) const override {
    return apply(messages, {}, nlohmann::ordered_json::object());
  }

  std::optional<std::string> apply(
      const ChatMessages& /*messages*/,
      const std::vector<xllm::JsonTool>& /*json_tools*/,
      const nlohmann::ordered_json& /*chat_template_kwargs*/) const override {
    return rendered_prompt_;
  }

 private:
  std::string rendered_prompt_ = std::string(512, 'p');
};

OutputCallback noop_callback() {
  return [](RequestOutput /*output*/) { return true; };
}

// Owns everything LLMRequestFactory borrows by pointer, in the same shape
// LLMMaster wires it up.
class FactoryFixture final {
 public:
  FactoryFixture() {
    model_args_.vocab_size(kVocabSize)
        .max_position_embeddings(kMaxPositionEmbeddings)
        .eos_token_id(2);
    options_.enable_service_routing(false).num_speculative_tokens(0);
    factory_ = std::make_unique<LLMRequestFactory>(
        &tokenizer_,
        &chat_template_,
        &model_args_,
        &options_,
        &rate_limiter_,
        /*task_type=*/"generate",
        [](const std::vector<RequestOutput>&) { return std::vector<bool>{}; });
  }

  LLMRequestFactory& factory() { return *factory_; }
  RateLimiter& rate_limiter() { return rate_limiter_; }

 private:
  FakeTokenizer tokenizer_;
  FakeChatTemplate chat_template_;
  ModelArgs model_args_;
  Options options_;
  RateLimiter rate_limiter_;
  std::unique_ptr<LLMRequestFactory> factory_;
};

RequestParams make_request_params() {
  RequestParams params;
  params.request_id = "cmpl-bench";
  params.max_tokens = 128;
  params.temperature = 0.7f;
  params.top_p = 0.9f;
  params.stop = std::vector<std::string>{"</s>", "<|im_end|>"};
  params.streaming = true;
  return params;
}

std::vector<Message> make_messages(size_t num_messages) {
  std::vector<Message> messages;
  messages.reserve(num_messages);
  messages.emplace_back("system", "You are a helpful assistant.");
  for (size_t i = 1; i < num_messages; ++i) {
    messages.emplace_back((i % 2 == 1) ? "user" : "assistant",
                          std::string(256, 'y'));
  }
  return messages;
}

void BM_Master_ThreadPoolHandoff(benchmark::State& state) {
  ThreadPool threadpool(kMasterRequestThreads);

  for (auto _ : state) {
    std::promise<void> done;
    std::future<void> future = done.get_future();
    threadpool.schedule([&done]() { done.set_value(); });
    future.wait();
  }
}

void BM_RequestFactory_CreateFromPrompt(benchmark::State& state) {
  FactoryFixture fixture;
  const std::string prompt(static_cast<size_t>(state.range(0)), 'x');
  const RequestParams params = make_request_params();
  const OutputCallback callback = noop_callback();

  for (auto _ : state) {
    // The API layer acquires the rate-limit slot before handing off; the
    // Request releases it when destroyed at the end of the iteration.
    const bool limited = fixture.rate_limiter().is_limited();
    do_not_optimize(limited);
    const bool verified = params.verify_params(callback);
    do_not_optimize(verified);
    // create() takes the prompt by value. Passing the lvalue copies it, which
    // matches the production path: CompletionServiceImpl copies the prompt out
    // of the const proto before it is moved through the master's closure.
    std::shared_ptr<Request> request =
        fixture.factory().create(prompt,
                                 /*prompt_tokens=*/std::nullopt,
                                 params,
                                 /*call=*/std::nullopt,
                                 callback);
    do_not_optimize(request.get());
  }
}

void BM_RequestFactory_CreateFromTokens(benchmark::State& state) {
  FactoryFixture fixture;
  const std::vector<int> prompt_tokens(static_cast<size_t>(state.range(0)), 7);
  const RequestParams params = make_request_params();
  const OutputCallback callback = noop_callback();

  for (auto _ : state) {
    const bool limited = fixture.rate_limiter().is_limited();
    do_not_optimize(limited);
    const bool verified = params.verify_params(callback);
    do_not_optimize(verified);
    // The lvalue vector is copied into the std::optional parameter, matching
    // the API layer materialising token_ids out of the proto per request.
    std::shared_ptr<Request> request =
        fixture.factory().create(/*prompt=*/"",
                                 prompt_tokens,
                                 params,
                                 /*call=*/std::nullopt,
                                 callback);
    do_not_optimize(request.get());
  }
}

void BM_RequestFactory_CreateFromMessages(benchmark::State& state) {
  FactoryFixture fixture;
  const std::vector<Message> messages =
      make_messages(static_cast<size_t>(state.range(0)));
  const RequestParams params = make_request_params();
  const OutputCallback callback = noop_callback();

  for (auto _ : state) {
    const bool limited = fixture.rate_limiter().is_limited();
    do_not_optimize(limited);
    const bool verified = params.verify_params(callback);
    do_not_optimize(verified);
    std::shared_ptr<Request> request =
        fixture.factory().create(messages,
                                 /*prompt_tokens=*/std::nullopt,
                                 params,
                                 /*call=*/std::nullopt,
                                 callback);
    do_not_optimize(request.get());
  }
}

BENCHMARK(BM_Master_ThreadPoolHandoff)->Unit(benchmark::kNanosecond);
// Prompt length in characters (== tokens with the char tokenizer).
BENCHMARK(BM_RequestFactory_CreateFromPrompt)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
BENCHMARK(BM_RequestFactory_CreateFromTokens)
    ->RangeMultiplier(8)
    ->Range(128, 128 << 10)
    ->Unit(benchmark::kNanosecond);
// Conversation length in messages.
BENCHMARK(BM_RequestFactory_CreateFromMessages)
    ->RangeMultiplier(4)
    ->Range(1, 64)
    ->Unit(benchmark::kNanosecond);

}  // namespace
}  // namespace xllm
