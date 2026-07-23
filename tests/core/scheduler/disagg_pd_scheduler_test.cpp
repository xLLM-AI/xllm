/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "scheduler/disagg_pd_scheduler.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_pool.h"
#include "framework/model/model_args.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/tokenizer/tokenizer.h"

namespace xllm {
namespace {

class FakeTokenizer final : public Tokenizer {
 public:
  bool encode(const std::string_view& /*text*/,
              std::vector<int32_t>* /*ids*/,
              bool /*add_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string decode(const Slice<int32_t>& /*ids*/,
                     bool /*skip_special_tokens*/) const override {
    NOT_IMPLEMENTED();
  }

  std::optional<int32_t> token_to_id(
      const std::string_view& /*token*/) const override {
    NOT_IMPLEMENTED();
  }

  std::string id_to_token(int32_t /*id*/) const override { NOT_IMPLEMENTED(); }

  size_t vocab_size() const override { NOT_IMPLEMENTED(); }

  std::unique_ptr<Tokenizer> clone() const override {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine final : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, /*dp_size=*/1);
  }

  ForwardOutput step(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(std::vector<Batch>& /*batch*/) override {
    NOT_IMPLEMENTED();
  }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }

  std::vector<int64_t> get_active_activation_memory() const override {
    NOT_IMPLEMENTED();
  }

  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  TestDisaggPDScheduler(Engine* engine, const Options& options)
      : DisaggPDScheduler(engine, options) {}

  void cache_prefill_blocks_for_test(Request* request) {
    cache_prefill_blocks(request);
  }

  bool pop_decode_request_for_test(std::shared_ptr<Request>* request) {
    return request_queue_.read(*request);
  }

  static int64_t amortized_token_latency_for_test(int64_t tbt_ms,
                                                  size_t num_tokens) {
    return amortized_token_latency_ms(tbt_ms, num_tokens);
  }

  void update_metrics(std::vector<Sequence*>& sequences) {
    update_token_latency_metrics(sequences);
  }
};

DisaggPDScheduler::Options make_options() {
  DisaggPDScheduler::Options options;
  options.enable_pd_ooc(true)
      .enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::PREFILL)
      .max_tokens_per_batch(32)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(32)
      .dp_size(1);
  return options;
}

DisaggPDScheduler::Options make_mtp_decode_options() {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE).num_speculative_tokens(1);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids) {
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(64);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*logprobs=*/false,
                     /*skip_special_tokens=*/false,
                     /*include_usage=*/false,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(
      "req", "x-request-id", "x-request-time", state, "service-req");
}

void finish_prefill(Sequence* sequence) {
  CHECK(sequence != nullptr);
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(999));
}

size_t first_cache_size(const BlockManagerPool& block_manager) {
  const std::vector<size_t> cache_sizes =
      block_manager.num_blocks_in_prefix_cache();
  CHECK(!cache_sizes.empty());
  return cache_sizes[0];
}

void release_prefix_cache(BlockManagerPool* block_manager) {
  CHECK(block_manager != nullptr);
  const size_t num_data_blocks = block_manager->num_blocks() - 1;
  std::vector<int32_t> token_ids;
  token_ids.reserve(num_data_blocks * block_manager->block_size());
  for (size_t i = 0; i < num_data_blocks * block_manager->block_size(); ++i) {
    token_ids.push_back(static_cast<int32_t>(1000 + i));
  }

  std::shared_ptr<Request> request = make_request(token_ids);
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  block_manager->deallocate(sequence);
  EXPECT_EQ(first_cache_size(*block_manager), 0u);
}

bool recv_first_generation(DisaggPDScheduler* scheduler,
                           const torch::Tensor& mtp_embedding) {
  return scheduler->decode_recv_first_generation(
      "req",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PUSH",
      /*src_cluster_ids=*/{},
      /*src_addrs=*/{},
      /*src_block_ids=*/{},
      /*src_linear_state_id=*/-1,
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      mtp_embedding);
}

}  // namespace

TEST(DisaggPDSchedulerTest, CachesPrefillBlocksBeforeRelease) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(sequence));
  finish_prefill(sequence);

  scheduler.cache_prefill_blocks_for_test(request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  block_manager->deallocate(request.get());
  EXPECT_EQ(sequence->kv_state().num_blocks(BlockType::KV), 0u);
  EXPECT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> matched_request = make_request({1, 2, 3, 4, 5});
  Sequence* matched_sequence = matched_request->sequences()[0].get();
  block_manager->allocate_shared(matched_sequence);

  EXPECT_EQ(matched_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  block_manager->deallocate(matched_sequence);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, CacheSkipsExistingSharedBlocks) {
  FakeEngine engine(/*num_blocks=*/10, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();

  std::shared_ptr<Request> seed_request = make_request({1, 2, 3, 4});
  Sequence* seed_sequence = seed_request->sequences()[0].get();
  ASSERT_TRUE(block_manager->allocate(seed_sequence));
  finish_prefill(seed_sequence);
  scheduler.cache_prefill_blocks_for_test(seed_request.get());
  block_manager->deallocate(seed_request.get());
  ASSERT_EQ(first_cache_size(*block_manager), 2u);

  std::shared_ptr<Request> extended_request = make_request({1, 2, 3, 4, 5, 6});
  Sequence* extended_sequence = extended_request->sequences()[0].get();
  block_manager->allocate_shared(extended_sequence);
  ASSERT_EQ(extended_sequence->kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_TRUE(block_manager->allocate(extended_sequence,
                                      extended_sequence->num_prompt_tokens()));
  finish_prefill(extended_sequence);

  scheduler.cache_prefill_blocks_for_test(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);

  block_manager->deallocate(extended_request.get());
  EXPECT_EQ(first_cache_size(*block_manager), 3u);
  release_prefix_cache(block_manager);
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationRequiresBootstrapBeforeQueue) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(request->sequences()[0].get()));
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_FALSE(recv_first_generation(&scheduler, torch::Tensor()));
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(scheduler.pop_decode_request_for_test(&queued));
}

TEST(DisaggPDSchedulerTest, MtpFirstGenerationStoresBootstrapThenQueues) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_GE(sequence->get_single_block_id(), 0);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  torch::Tensor embedding = torch::tensor({1.0f, 2.0f});
  EXPECT_TRUE(recv_first_generation(&scheduler, embedding));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->request_id(), "req");
  EXPECT_EQ(queued->sequences()[0]->tokens().back(), 42);
  EXPECT_TRUE(torch::equal(
      queued->sequences()[0]->get_mtp_bootstrap_embedding(), embedding));
}

TEST(DisaggPDSchedulerTest, FirstDecodeTokenLatencyIsNonNegative) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  EXPECT_TRUE(recv_first_generation(&scheduler, torch::Tensor()));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  // Base rebuilt in decode_recv_first_generation must not sit in the future:
  // pre-fix it was created_time + ttft (~now+100ms), yielding a negative ITL.
  int64_t first_itl = queued->sequences()[0]->tbt(absl::Now());
  EXPECT_GE(first_itl, 0);
}

TEST(DisaggPDSchedulerTest, AmortizedTokenLatencyRoundsHalfUp) {
  // Amortized per-token latency is round(tbt_ms / n) via (tbt_ms + n/2) / n.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(100, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(101, 4),
            25);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(102, 4),
            26);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(50, 5), 10);
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(53, 5), 11);
  // With a single committed token amortized latency equals the raw latency.
  EXPECT_EQ(TestDisaggPDScheduler::amortized_token_latency_for_test(37, 1), 37);
}

TEST(DisaggPDSchedulerTest, SpeculativeGaugeReportsBatchMeanTokensPerStep) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());

  std::shared_ptr<Request> first_request = make_request({1, 2, 3, 4});
  Sequence* first_sequence = first_request->sequences()[0].get();
  first_sequence->kv_state().set_kv_cache_tokens_num(
      first_sequence->num_prompt_tokens());
  for (int32_t token_id = 10; token_id < 15; ++token_id) {
    first_sequence->append_token(Token(token_id));
  }

  std::shared_ptr<Request> second_request = make_request({5, 6, 7, 8});
  Sequence* second_sequence = second_request->sequences()[0].get();
  second_sequence->kv_state().set_kv_cache_tokens_num(
      second_sequence->num_prompt_tokens());
  for (int32_t token_id = 20; token_id < 23; ++token_id) {
    second_sequence->append_token(Token(token_id));
  }

  std::vector<Sequence*> sequences = {first_sequence, second_sequence};
  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_tokens_per_decode_step.get_value(),
                   4.0);
  EXPECT_EQ(first_sequence->generated_tokens_since_latency(), 0u);
  EXPECT_EQ(second_sequence->generated_tokens_since_latency(), 0u);
}

TEST(DisaggPDSchedulerTest, SpeculativeMetricsSilentWhenDisabled) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  // make_options() keeps num_speculative_tokens at its default of 0.
  TestDisaggPDScheduler scheduler(&engine, make_options());

  GAUGE_SET(speculative_mean_tokens_per_decode_step, -1.0);

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(10));
  sequence->append_token(Token(11));
  std::vector<Sequence*> sequences = {sequence};

  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_tokens_per_decode_step.get_value(),
                   -1.0);
}

}  // namespace xllm
