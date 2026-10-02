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

#include "scheduler/disagg_pd_scheduler.h"

#include <brpc/closure_guard.h>
#include <brpc/server.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <atomic>
#include <barrier>
#include <chrono>
#include <cstdint>
#include <functional>
#include <future>
#include <memory>
#include <mutex>
#include <optional>
#include <string>
#include <vector>

#include "common/metrics.h"
#include "distributed_runtime/engine.h"
#include "framework/block/block_manager_impl.h"
#include "framework/block/block_manager_pool.h"
#include "framework/kv_cache_transfer/kv_transfer_completion.h"
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
  FakeEngine(int32_t num_blocks,
             int32_t block_size,
             int32_t num_speculative_tokens = 0,
             int32_t dp_size = 1,
             int32_t embedding_blocks = 0) {
    BlockManagerPool::Options options;
    options.num_blocks(num_blocks)
        .block_size(block_size)
        .enable_prefix_cache(true)
        .enable_disagg_pd(true)
        .num_speculative_tokens(num_speculative_tokens)
        .num_embedding_blocks(embedding_blocks == 0 ? num_blocks
                                                    : embedding_blocks);
    tokenizer_ = std::make_unique<FakeTokenizer>();
    block_manager_ = std::make_unique<BlockManagerPool>(options, dp_size);
  }

  std::function<ForwardOutput(BatchGroup&)> forward;

  ForwardOutput step(BatchGroup& batch) {
    if (forward) {
      return forward(batch);
    }
    NOT_IMPLEMENTED();
  }

  void update_last_step_result(BatchGroup& /*batch*/) { NOT_IMPLEMENTED(); }

  const Tokenizer* tokenizer() const override { return tokenizer_.get(); }

  BlockManagerPool* block_manager_pool() const override {
    return block_manager_.get();
  }

  const ModelArgs& model_args() const override { return model_args_; }

  const TokenizerArgs& tokenizer_args() const override { NOT_IMPLEMENTED(); }

  std::vector<int64_t> get_active_activation_memory() const override {
    return {0};
  }

  bool init() override { return true; }

  bool pull_kv_blocks(int32_t /*src_dp_size*/,
                      int32_t /*src_dp_rank*/,
                      const std::vector<uint64_t>& /*src_cluster_ids*/,
                      const std::vector<std::string>& /*src_addrs*/,
                      int32_t /*dst_dp_rank*/,
                      const std::vector<KVTransferMapping>& mappings) override {
    pulled_mappings = mappings;
    return true;
  }

  std::vector<KVTransferMapping> pulled_mappings;

 private:
  std::unique_ptr<Tokenizer> tokenizer_;
  std::unique_ptr<BlockManagerPool> block_manager_;
  ModelArgs model_args_;
};

class TestDisaggPDScheduler final : public DisaggPDScheduler {
 public:
  using DisaggPDScheduler::prepare_batch;

  template <typename TargetEngine>
  TestDisaggPDScheduler(TargetEngine* engine, const Options& options)
      : DisaggPDScheduler(engine, options, SkipRuntimeStart{}) {}

  void admit_prefill(std::shared_ptr<Request> request,
                     proto::DisaggPDService_Stub* stub = nullptr) {
    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      req_to_channel_map_[request->request_id()] = stub;
    }
    request_queue_.write(std::move(request));
  }

  bool has_channel(const std::string& req_id) {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    return req_to_channel_map_.contains(req_id);
  }

  bool reservations_empty() {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    return received_request_map_.empty() &&
           instance_to_received_requests_map_.empty() &&
           request_to_instance_map_.empty();
  }

  void wait_notifications() {
    std::promise<void> done;
    auto future = done.get_future();
    reservation_release_threadpool_.schedule([&done] { done.set_value(); });
    future.get();
  }

  std::future<void> prefill_completion() {
    std::promise<void> done;
    auto future = done.get_future();
    prefill_threadpool_.schedule(
        [done = std::move(done)]() mutable { done.set_value(); });
    return future;
  }

  void cache_prefill_blocks_for_test(Request* request) {
    cache_prefill_blocks(request);
  }

  bool pop_decode_request_for_test(std::shared_ptr<Request>* request) {
    return request_queue_.read(*request);
  }

  void update_metrics(std::vector<Sequence*>& sequences) {
    scheduler_metrics_->update_token_latency_metrics(sequences);
  }
};

class DelayedReservationService final : public proto::DisaggPDService {
 public:
  explicit DelayedReservationService(DisaggPDScheduler* scheduler)
      : scheduler_(scheduler) {
    pending_releases_.reserve(3);
  }

  std::future<void> release_started() { return release_started_.get_future(); }

  void ReleaseReservation(google::protobuf::RpcController* /*controller*/,
                          const proto::ReleaseReservationRequest* request,
                          proto::ReleaseReservationResponse* response,
                          google::protobuf::Closure* done) override {
    std::unique_lock<std::mutex> lock(mutex_);
    if (!delay_releases_) {
      lock.unlock();
      complete_release(request, response, done);
      return;
    }
    // Keep the server-side RPC alive without blocking a brpc worker.
    pending_releases_.emplace_back([this, request, response, done] {
      complete_release(request, response, done);
    });
    if (pending_releases_.size() == 1) {
      release_started_.set_value();
    }
  }

  void FirstGeneration(google::protobuf::RpcController* /*controller*/,
                       const proto::DisaggGenerationsRequests* request,
                       proto::Status* response,
                       google::protobuf::Closure* done) override {
    brpc::ClosureGuard done_guard(done);
    CHECK_EQ(request->multi_gens_size(), 1);
    const auto& gen = request->multi_gens(0);
    CHECK_EQ(gen.tokens_size(), 1);
    CHECK_EQ(gen.kv_cache_transfer_mode(), "PUSH");
    const auto& token = gen.tokens(0);
    response->set_ok(scheduler_->decode_recv_first_generation(
        gen.req_id(),
        gen.reservation_id(),
        token.token_id(),
        token.has_logprob(),
        token.logprob(),
        token.time_to_first_token_latency_seconds(),
        gen.upstream_elapsed_seconds(),
        /*top_tokens=*/{},
        /*top_logprobs=*/{},
        gen.kv_cache_transfer_mode(),
        /*src_cluster_ids=*/{},
        /*src_addrs=*/{},
        /*source_mappings=*/{},
        gen.dp_size(),
        gen.dp_rank()));
  }

  void resume_releases() {
    std::vector<std::function<void()>> pending;
    {
      std::lock_guard<std::mutex> lock(mutex_);
      delay_releases_ = false;
      pending.swap(pending_releases_);
    }
    for (auto& release : pending) {
      release();
    }
  }

 private:
  void complete_release(const proto::ReleaseReservationRequest* request,
                        proto::ReleaseReservationResponse* response,
                        google::protobuf::Closure* done) {
    brpc::ClosureGuard done_guard(done);
    const bool released = scheduler_->release_reservation(
        request->req_id(), request->reservation_id());
    response->set_result(released
                             ? proto::ReleaseReservationResponse::RELEASED
                             : proto::ReleaseReservationResponse::NOT_WAITING);
  }

  DisaggPDScheduler* scheduler_;
  std::mutex mutex_;
  bool delay_releases_ = true;
  std::promise<void> release_started_;
  std::vector<std::function<void()>> pending_releases_;
};

class ReservationService final : public proto::DisaggPDService {
 public:
  explicit ReservationService(DisaggPDScheduler* scheduler)
      : scheduler_(scheduler) {}

  enum class Failure { NONE, LOST_ACK, TRANSPORT, UNSUPPORTED };
  Failure failure = Failure::NONE;
  std::atomic<int32_t> calls{0};
  std::atomic<int32_t> released{0};

  void ReleaseReservation(google::protobuf::RpcController* controller,
                          const proto::ReleaseReservationRequest* request,
                          proto::ReleaseReservationResponse* response,
                          google::protobuf::Closure* done) override {
    const int32_t attempt = ++calls;
    if (failure == Failure::UNSUPPORTED) {
      proto::DisaggPDService::ReleaseReservation(
          controller, request, response, done);
      return;
    }
    if (failure == Failure::TRANSPORT) {
      controller->SetFailed("injected transport error");
    } else {
      const bool freed = scheduler_->release_reservation(
          request->req_id(), request->reservation_id());
      released += freed ? 1 : 0;
      response->set_result(
          freed ? proto::ReleaseReservationResponse::RELEASED
                : proto::ReleaseReservationResponse::NOT_WAITING);
      if (failure == Failure::LOST_ACK && attempt == 1) {
        controller->SetFailed("injected lost release confirmation");
      }
    }
    if (done != nullptr) {
      done->Run();
    }
  }

 private:
  DisaggPDScheduler* scheduler_;
};

DisaggPDScheduler::Options make_options() {
  DisaggPDScheduler::Options options;
  options.enable_disagg_pd(true)
      .enable_schedule_overlap(false)
      .instance_role(InstanceRole::PREFILL)
      .max_tokens_per_batch(32)
      .max_seqs_per_batch(4)
      .max_tokens_per_chunk_for_prefill(32)
      .dp_size(1);
  return options;
}

DisaggPDScheduler::Options make_mtp_decode_options(int32_t dp_size = 1) {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE)
      .num_speculative_tokens(1)
      .dp_size(dp_size);
  return options;
}

DisaggPDScheduler::Options make_decode_options() {
  DisaggPDScheduler::Options options = make_options();
  options.instance_role(InstanceRole::DECODE);
  return options;
}

std::shared_ptr<Request> make_request(
    const std::vector<int32_t>& prompt_token_ids,
    const std::string& req_id = "req",
    bool enable_schedule_overlap = false) {
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
                     enable_schedule_overlap,
                     /*mm_data=*/nullptr,
                     /*service_request_id=*/nullptr);

  return std::make_shared<Request>(req_id,
                                   "x-request-id",
                                   "x-request-time",
                                   std::move(state),
                                   "service-req");
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
                           const torch::Tensor& mtp_embedding,
                           int32_t num_cached_tokens = 0,
                           double time_to_first_token_latency_seconds = 0.1,
                           double upstream_elapsed_seconds = 0.0,
                           const std::string& reservation_id = "") {
  return scheduler->decode_recv_first_generation(
      "req",
      reservation_id,
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      time_to_first_token_latency_seconds,
      upstream_elapsed_seconds,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PUSH",
      /*src_cluster_ids=*/{},
      /*src_addrs=*/{},
      /*source_mappings=*/{},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0,
      mtp_embedding,
      num_cached_tokens);
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
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
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
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_GE(sequence->get_embedding_block_id(), 0);
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

TEST(DisaggPDSchedulerTest, GroupedPullAlignsActiveSwaSuffix) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_decode_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());

  BlockManager::Options swa_options;
  swa_options.num_blocks(8).block_size(2);
  BlockManagerImpl swa_manager(swa_options);
  std::vector<Block> live_swa_blocks = swa_manager.allocate(2);
  std::vector<Block> logical_swa_blocks(2);
  logical_swa_blocks.insert(
      logical_swa_blocks.end(), live_swa_blocks.begin(), live_swa_blocks.end());
  sequence->add_blocks(BlockType::SWA, logical_swa_blocks);
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  KVTransferMapping source_mapping;
  source_mapping.group_id = cache_group_id(BlockType::SWA);
  source_mapping.remote_ids = {101, 102};
  ASSERT_TRUE(scheduler.decode_recv_first_generation(
      "req",
      /*reservation_id=*/"",
      /*token_id=*/42,
      /*has_logprob=*/false,
      /*logprob=*/0.0f,
      /*time_to_first_token_latency_seconds=*/0.1,
      /*upstream_elapsed_seconds=*/0.0,
      /*top_tokens=*/{},
      /*top_logprobs=*/{},
      /*kv_cache_transfer_mode=*/"PULL",
      /*src_cluster_ids=*/{1},
      /*src_addrs=*/{"remote"},
      /*source_mappings=*/{source_mapping},
      /*src_dp_size=*/1,
      /*src_dp_rank=*/0));

  ASSERT_EQ(engine.pulled_mappings.size(), 1U);
  EXPECT_EQ(engine.pulled_mappings[0].group_id, cache_group_id(BlockType::SWA));
  EXPECT_EQ(
      engine.pulled_mappings[0].local_ids,
      (std::vector<uint64_t>{static_cast<uint64_t>(live_swa_blocks[0].id()),
                             static_cast<uint64_t>(live_swa_blocks[1].id())}));
  EXPECT_EQ(engine.pulled_mappings[0].remote_ids,
            (std::vector<uint64_t>{101, 102}));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  engine.block_manager_pool()->deallocate(queued.get());
  queued->sequences()[0]->kv_state().erase_blocks(BlockType::SWA);
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

TEST(DisaggPDSchedulerTest, DecodeLatencyIncludesPrefillTtft) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  constexpr double kPrefillTtftSeconds = 10.0;
  EXPECT_TRUE(recv_first_generation(&scheduler,
                                    torch::Tensor(),
                                    /*num_cached_tokens=*/0,
                                    kPrefillTtftSeconds));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_GE(queued->end_to_end_latency_seconds(), kPrefillTtftSeconds);
  EXPECT_LT(queued->elapsed_seconds(), kPrefillTtftSeconds);
}

TEST(DisaggPDSchedulerTest, DecodeLatencyUsesCumulativeUpstreamElapsed) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  constexpr double kTtftSeconds = 2.0;
  constexpr double kUpstreamElapsedSeconds = 10.0;
  EXPECT_TRUE(recv_first_generation(&scheduler,
                                    torch::Tensor(),
                                    /*num_cached_tokens=*/0,
                                    kTtftSeconds,
                                    kUpstreamElapsedSeconds));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_DOUBLE_EQ(
      queued->sequences()[0]->time_to_first_token_latency_seconds(),
      kTtftSeconds);
  EXPECT_GE(queued->end_to_end_latency_seconds(), kUpstreamElapsedSeconds);
  EXPECT_LT(queued->elapsed_seconds(), kTtftSeconds);

  queued->sequences()[0]->append_token(Token(43));
  queued->sequences()[0]->append_token(Token(44));
  queued->sequences()[0]->append_token(Token(45));
  const size_t generated_tokens =
      queued->sequences()[0]->num_generated_tokens();
  ASSERT_EQ(generated_tokens, 4U);
  const double generation_latency_seconds =
      queued->end_to_end_latency_seconds() - kTtftSeconds;
  const double average_tpot_milliseconds =
      generation_latency_seconds * 1000.0 / (generated_tokens - 1);
  EXPECT_GE(average_tpot_milliseconds,
            (kUpstreamElapsedSeconds - kTtftSeconds) * 1000.0 /
                (generated_tokens - 1));
}

TEST(DisaggPDSchedulerTest, PreservesPrefillCachedTokensOnDecodeRequest) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

  ASSERT_TRUE(recv_first_generation(
      &scheduler, torch::Tensor(), /*num_cached_tokens=*/2));

  std::shared_ptr<Request> queued;
  ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued->num_prefix_cache_tokens(), 2u);
}

TEST(DisaggPDSchedulerTest, PromptAtDecodeBlockCapacityIsNotPermanent) {
  EXPECT_FALSE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/6, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, OnlyOversizedDecodeResponseIsTerminal) {
  EXPECT_FALSE(is_permanent_rejection(/*status_code=*/404));
  EXPECT_TRUE(is_permanent_rejection(kDecodeAddNewPromptTooLongStatusCode));
  EXPECT_FALSE(is_permanent_rejection(/*status_code=*/500));
}

TEST(DisaggPDSchedulerTest, DetectsRankPreservingFlatKvGroups) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::KV));
  response.add_groups()->set_group_id(cache_group_id(BlockType::LINEAR));

  EXPECT_TRUE(has_rank_preserving_kv_groups(response));
}

TEST(DisaggPDSchedulerTest, KeepsExpandedGroupedCacheMappings) {
  proto::DisaggResponse response;
  response.add_groups()->set_group_id(cache_group_id(BlockType::C4));

  EXPECT_FALSE(has_rank_preserving_kv_groups(response));
}

TEST(DisaggPDSchedulerTest, PromptBeyondDecodeBlockCapacityIsPermanent) {
  EXPECT_TRUE(exceeds_decode_capacity(
      /*num_prompt_tokens=*/7, /*block_size=*/2, /*num_blocks=*/4));
}

TEST(DisaggPDSchedulerTest, TemporaryDecodeBlockPressureIsNotPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> holder = make_request({1, 2, 3, 4, 5, 6});
  ASSERT_TRUE(block_manager->try_allocate(holder->sequences()[0].get()));
  std::shared_ptr<Request> request = make_request({7, 8});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_FALSE(scheduler.try_allocate(sequence));
  EXPECT_FALSE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(holder.get());
}

TEST(DisaggPDSchedulerTest, OversizedDecodePromptIsPermanent) {
  FakeEngine engine(/*num_blocks=*/4, /*block_size=*/2);
  TestDisaggPDScheduler scheduler(&engine, make_options());
  BlockManagerPool* block_manager = engine.block_manager_pool();
  std::shared_ptr<Request> request = make_request({1, 2, 3, 4, 5, 6, 7});
  Sequence* sequence = request->sequences()[0].get();

  EXPECT_FALSE(scheduler.try_allocate(sequence));
  EXPECT_TRUE(scheduler.exceeds_decode_capacity(sequence));

  block_manager->deallocate(request.get());
}

TEST(DisaggPDSchedulerTest, InvalidPrefillCachedTokensFallBackToZero) {
  for (int32_t num_cached_tokens : {-1, 5}) {
    FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
    TestDisaggPDScheduler scheduler(&engine, make_options());
    std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
    Sequence* sequence = request->sequences()[0].get();
    ASSERT_TRUE(engine.block_manager_pool()->allocate(sequence));
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    ASSERT_TRUE(scheduler.decode_schedule(request, "prefill"));

    ASSERT_TRUE(
        recv_first_generation(&scheduler, torch::Tensor(), num_cached_tokens));

    std::shared_ptr<Request> queued;
    ASSERT_TRUE(scheduler.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued->num_prefix_cache_tokens(), 0u);
  }
}

TEST(DisaggPDSchedulerTest, EmptyOverlapOutputPreservesLatencyClock) {
  for (int32_t num_speculative_tokens : {0, 3}) {
    SCOPED_TRACE(num_speculative_tokens);
    FakeEngine engine(
        /*num_blocks=*/8, /*block_size=*/2, num_speculative_tokens);
    DisaggPDScheduler::Options options = make_options();
    options.enable_schedule_overlap(true)
        .enable_chunked_prefill(true)
        .num_speculative_tokens(num_speculative_tokens);
    TestDisaggPDScheduler scheduler(&engine, options);
    std::shared_ptr<Request> request =
        make_request({1, 2, 3, 4}, "req", /*enable_schedule_overlap=*/true);
    Sequence* sequence = request->sequences().front().get();
    sequence->kv_state().set_kv_cache_tokens_num(2);
    ASSERT_TRUE(sequence->is_chunked_prefill_stage());
    sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
    sequence->append_token(Token(-1));
    ASSERT_EQ(sequence->stage(), SequenceStage::DECODE);
    ASSERT_EQ(sequence->generated_tokens_since_latency(), 0u);
    const absl::Time start = absl::Now() - absl::Seconds(10);
    sequence->tbt_microseconds(start);
    std::vector<Sequence*> sequences = {sequence};
    std::vector<int64_t> ttft;
    std::vector<int64_t> tbt;

    scheduler.update_metrics(sequences);
    scheduler.update_metrics(sequences);
    scheduler.get_latency_metrics(ttft, tbt);
    EXPECT_TRUE(ttft.empty());
    EXPECT_TRUE(tbt.empty());
    EXPECT_DOUBLE_EQ(sequence->time_to_first_token_latency_seconds(), 0.0);
    EXPECT_EQ(sequence->tbt_microseconds(start + absl::Seconds(7)), 7000000);

    sequence->update_last_step_token(Token(10), /*token_offset=*/0);
    ASSERT_TRUE(sequence->is_first_token());
    scheduler.update_metrics(sequences);
    scheduler.get_latency_metrics(ttft, tbt);
    ASSERT_EQ(ttft.size(), 1u);
    EXPECT_GE(ttft.front(), 3000);
    EXPECT_TRUE(tbt.empty());
    EXPECT_GE(sequence->time_to_first_token_latency_seconds(), 3.0);
    EXPECT_EQ(sequence->generated_tokens_since_latency(), 0u);
    scheduler.update_metrics(sequences);
    scheduler.get_latency_metrics(ttft, tbt);
    EXPECT_TRUE(ttft.empty());
    EXPECT_TRUE(tbt.empty());

    sequence->append_token(Token(-1));
    sequence->update_last_step_token(Token(11), /*token_offset=*/0);
    ASSERT_FALSE(sequence->is_first_token());
    scheduler.update_metrics(sequences);
    scheduler.get_latency_metrics(ttft, tbt);
    EXPECT_TRUE(ttft.empty());
    EXPECT_EQ(tbt.size(), 1u);
    scheduler.update_metrics(sequences);
    scheduler.get_latency_metrics(ttft, tbt);
    EXPECT_TRUE(ttft.empty());
    EXPECT_TRUE(tbt.empty());
  }
}

TEST(DisaggPDSchedulerTest, SchedulerDoesNotOverwriteSpeculativeOutputGauge) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler scheduler(&engine, make_mtp_decode_options());
  GAUGE_SET(speculative_mean_acceptance_length, 4.25);

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

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_acceptance_length.get_value(), 4.25);
  EXPECT_EQ(first_sequence->generated_tokens_since_latency(), 0u);
  EXPECT_EQ(second_sequence->generated_tokens_since_latency(), 0u);
}

TEST(DisaggPDSchedulerTest, SpeculativeMetricsSilentWhenDisabled) {
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  // make_options() keeps num_speculative_tokens at its default of 0.
  TestDisaggPDScheduler scheduler(&engine, make_options());

  GAUGE_SET(speculative_mean_acceptance_length, -1.0);

  std::shared_ptr<Request> request = make_request({1, 2, 3, 4});
  Sequence* sequence = request->sequences()[0].get();
  sequence->kv_state().set_kv_cache_tokens_num(sequence->num_prompt_tokens());
  sequence->append_token(Token(10));
  sequence->append_token(Token(11));
  std::vector<Sequence*> sequences = {sequence};

  scheduler.update_metrics(sequences);

  EXPECT_DOUBLE_EQ(GAUGE_speculative_mean_acceptance_length.get_value(), -1.0);
}

TEST(DisaggPDSchedulerTest, StructuredOutputFieldsPreserveWireTags) {
  proto::DisaggRequest request;
  request.set_include_stop_str_in_output(true);
  request.set_json_object(true);
  request.set_json_reasoning_enabled(true);

  std::string serialized;
  ASSERT_TRUE(request.SerializeToString(&serialized));

  proto::DisaggRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  EXPECT_TRUE(decoded.include_stop_str_in_output());
  EXPECT_TRUE(decoded.json_object());
  EXPECT_TRUE(decoded.json_reasoning_enabled());
  EXPECT_EQ(proto::DisaggRequest::kIncludeStopStrInOutputFieldNumber, 39);
  EXPECT_EQ(proto::DisaggRequest::kJsonObjectFieldNumber, 40);
  EXPECT_EQ(proto::DisaggRequest::kJsonReasoningEnabledFieldNumber, 41);
}

TEST(DisaggPDSchedulerTest, GenerationLatencyFieldsPreserveWireTags) {
  proto::DisaggGenerationsRequest request;
  request.set_upstream_elapsed_seconds(12.5);
  proto::RemoteToken* token = request.add_tokens();
  token->set_time_to_first_token_latency_seconds(2.5);

  std::string serialized;
  ASSERT_TRUE(request.SerializeToString(&serialized));

  proto::DisaggGenerationsRequest decoded;
  ASSERT_TRUE(decoded.ParseFromString(serialized));
  ASSERT_TRUE(decoded.has_upstream_elapsed_seconds());
  EXPECT_DOUBLE_EQ(decoded.upstream_elapsed_seconds(), 12.5);
  ASSERT_EQ(decoded.tokens_size(), 1);
  EXPECT_DOUBLE_EQ(decoded.tokens(0).time_to_first_token_latency_seconds(),
                   2.5);
  EXPECT_EQ(proto::DisaggGenerationsRequest::kUpstreamElapsedSecondsFieldNumber,
            21);
  EXPECT_EQ(proto::RemoteToken::kTimeToFirstTokenLatencySecondsFieldNumber, 6);
}

TEST(DisaggPDSchedulerTest, LocalFailureReturnsDecodeReservation) {
  FakeEngine decode_engine(/*num_blocks=*/16,
                           /*block_size=*/2,
                           /*num_speculative_tokens=*/1);
  TestDisaggPDScheduler decode(&decode_engine, make_mtp_decode_options());
  auto remote = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  auto* pool = decode_engine.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  ASSERT_TRUE(decode.try_allocate(remote->sequences()[0].get()));
  remote->state().pd_reservation_id = "reservation";
  ASSERT_TRUE(decode.decode_schedule(remote, "prefill"));
  ASSERT_GE(remote->sequences()[0]->get_embedding_block_id(), 0);

  ReservationService service(&decode);
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);

  FakeEngine prefill_engine(/*num_blocks=*/2, /*block_size=*/2);
  TestDisaggPDScheduler prefill(&prefill_engine, make_options());
  auto local = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  TransferKVInfo info;
  info.request_id = local->request_id();
  info.remote_instance_info.rpc_address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  local->state().pd_reservation_id = "reservation";
  local->state().decode_rpc_address = info.remote_instance_info.rpc_address;
  local->sequences()[0]->kv_state().set_transfer_kv_info(std::move(info));
  local->state().output_func = [](const RequestOutput&) { return true; };
  prefill.admit_prefill(local);
  auto batch = prefill.prepare_batch();
  EXPECT_TRUE(batch[0].empty());
  prefill.wait_notifications();

  EXPECT_FALSE(prefill.has_channel(local->request_id()));
  EXPECT_TRUE(decode.reservations_empty());
  EXPECT_EQ(service.calls, 1);
  EXPECT_EQ(pool->num_used_blocks()[0], 0u);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(remote->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(first_cache_size(*pool), 0u);
  server.Stop(0);
  server.Join();
}
TEST(DisaggPDSchedulerTest, PendingReleaseDoesNotDelayFirstGeneration) {
  FakeEngine decode_engine(/*num_blocks=*/16, /*block_size=*/2);
  TestDisaggPDScheduler decode(&decode_engine, make_decode_options());
  auto remote_failed = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "failed");
  remote_failed->state().pd_reservation_id = "failed-reservation";
  ASSERT_TRUE(decode.try_allocate(remote_failed->sequences()[0].get()));
  ASSERT_TRUE(decode.decode_schedule(remote_failed, "prefill"));
  auto remote_healthy = make_request({11, 12}, "healthy");
  remote_healthy->state().stream = true;
  remote_healthy->state().pd_reservation_id = "healthy-reservation";
  ASSERT_TRUE(decode.try_allocate(remote_healthy->sequences()[0].get()));
  ASSERT_TRUE(decode.decode_schedule(remote_healthy, "prefill"));

  DelayedReservationService service(&decode);
  brpc::Server server;
  ASSERT_EQ(server.AddService(&service, brpc::SERVER_DOESNT_OWN_SERVICE), 0);
  ASSERT_EQ(server.Start("127.0.0.1:0", nullptr), 0);
  const std::string address =
      "127.0.0.1:" + std::to_string(server.listen_address().port);
  brpc::Channel channel;
  ASSERT_EQ(channel.Init(address.c_str(), nullptr), 0);
  proto::DisaggPDService_Stub stub(&channel);

  FakeEngine prefill_engine(/*num_blocks=*/4, /*block_size=*/2);
  auto options = make_options();
  options.enable_chunked_prefill(false).kv_cache_transfer_mode("PUSH");
  TestDisaggPDScheduler prefill(&prefill_engine, options);
  auto local_failed = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "failed");
  local_failed->state().pd_reservation_id = "failed-reservation";
  local_failed->state().decode_rpc_address = address;
  local_failed->state().output_func = [](const RequestOutput&) { return true; };
  auto release_started = service.release_started();
  prefill.admit_prefill(local_failed);
  EXPECT_TRUE(prefill.prepare_batch()[0].empty());
  // Always resume the deferred server RPCs below, including on test failure.
  EXPECT_EQ(release_started.wait_for(std::chrono::seconds(5)),
            std::future_status::ready);

  auto local_healthy = make_request({11, 12}, "healthy");
  local_healthy->state().stream = true;
  local_healthy->state().pd_reservation_id = "healthy-reservation";
  prefill.admit_prefill(local_healthy, &stub);
  auto batch = prefill.prepare_batch();
  EXPECT_FALSE(batch[0].empty());
  if (!batch[0].empty()) {
    finish_prefill(local_healthy->sequences()[0].get());
    // Supply the engine's PD output without enabling global PD configuration.
    local_healthy->sequences()[0]->first_token() = RemoteToken{.token_id = 999};
    prefill.prefill_send_first_generation();
  }
  auto prefill_done = prefill.prefill_completion();
  // Shorter than the release retry budget: handoff and local reclamation must
  // finish while the release RPC is still awaiting its server-side completion.
  const std::future_status status =
      prefill_done.wait_for(std::chrono::seconds(1));
  EXPECT_EQ(status, std::future_status::ready);
  std::shared_ptr<Request> queued;
  if (status == std::future_status::ready) {
    EXPECT_TRUE(decode.pop_decode_request_for_test(&queued));
    EXPECT_EQ(queued, remote_healthy);
    EXPECT_EQ(remote_healthy->sequences()[0]->num_generated_tokens(), 1u);
    EXPECT_EQ(prefill_engine.block_manager_pool()->num_used_blocks()[0], 0u);
    EXPECT_FALSE(decode.reservations_empty());
  }

  service.resume_releases();
  prefill.wait_notifications();
  prefill_done.get();
  EXPECT_TRUE(decode.reservations_empty());
  if (queued == nullptr) {
    decode.pop_decode_request_for_test(&queued);
  }
  decode_engine.block_manager_pool()->deallocate_without_cache(
      remote_healthy->sequences()[0].get());
  server.Stop(0);
  server.Join();
}

namespace {
class ReservationTest : public ::testing::Test {
 protected:
  FakeEngine engine_{/*num_blocks=*/16,
                     /*block_size=*/2,
                     /*num_speculative_tokens=*/1,
                     /*dp_size=*/2};
  TestDisaggPDScheduler decode_{&engine_,
                                make_mtp_decode_options(/*dp_size=*/2)};
  ReservationService service_{&decode_};
  brpc::Server server_;

  void SetUp() override {
    ASSERT_EQ(server_.AddService(&service_, brpc::SERVER_DOESNT_OWN_SERVICE),
              0);
    ASSERT_EQ(server_.Start("127.0.0.1:0", nullptr), 0);
  }

  void TearDown() override {
    server_.Stop(0);
    server_.Join();
  }

  std::shared_ptr<Request> reserve(
      const std::string& id = "req",
      const std::string& identity = "reservation") {
    auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8}, id);
    request->state().pd_reservation_id = identity;
    CHECK(decode_.try_allocate(request->sequences()[0].get()));
    CHECK(decode_.decode_schedule(request, "prefill"));
    return request;
  }

  std::shared_ptr<Request> accepted(const std::string& id = "req") {
    auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8}, id);
    request->state().pd_reservation_id = "reservation";
    request->state().decode_rpc_address =
        "127.0.0.1:" + std::to_string(server_.listen_address().port);
    request->state().output_func = [](const RequestOutput&) { return true; };
    return request;
  }

  void fail_prefill() {
    FakeEngine engine(/*num_blocks=*/2, /*block_size=*/2);
    auto options = make_options();
    options.enable_chunked_prefill(false);
    TestDisaggPDScheduler prefill(&engine, options);
    auto request = accepted();
    std::promise<Status> error;
    auto status = error.get_future();
    request->state().output_func = [&error](const RequestOutput& output) {
      error.set_value(output.status.value());
      return true;
    };
    prefill.admit_prefill(request);
    EXPECT_TRUE(prefill.prepare_batch()[0].empty());
    prefill.wait_notifications();
    EXPECT_EQ(status.get().code(), StatusCode::RESOURCE_EXHAUSTED);
    EXPECT_FALSE(prefill.has_channel(request->request_id()));
    EXPECT_TRUE(request->state().decode_rpc_address.empty());
    EXPECT_EQ(engine.block_manager_pool()->num_used_blocks()[0], 0u);
  }
};
}  // namespace

TEST_F(ReservationTest, LostConfirmationRetriesWithoutDoubleFree) {
  auto request = reserve();
  BlockManager* embedding = request->sequences()[0]
                                ->kv_state()
                                .blocks(BlockType::EMBEDDING)[0]
                                .manager();
  ASSERT_EQ(embedding->num_used_blocks(), 1u);
  service_.failure = ReservationService::Failure::LOST_ACK;
  fail_prefill();
  EXPECT_EQ(service_.calls, 2);
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
  EXPECT_EQ(embedding->num_used_blocks(), 0u);
  EXPECT_EQ(embedding->num_free_blocks(), 15u);
}

TEST_F(ReservationTest, TransportFailureHasBoundedRetries) {
  auto request = reserve();
  service_.failure = ReservationService::Failure::TRANSPORT;
  fail_prefill();
  EXPECT_EQ(service_.calls, 3);
  EXPECT_EQ(service_.released, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, UnsupportedRpcHasBoundedRetries) {
  auto request = reserve();
  service_.failure = ReservationService::Failure::UNSUPPORTED;
  fail_prefill();
  EXPECT_EQ(service_.calls, 3);
  EXPECT_EQ(service_.released, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, ReleaseNeverConsumesAnotherReservation) {
  auto first = reserve();
  auto other = reserve("other");
  EXPECT_FALSE(decode_.release_reservation("unknown", "reservation"));
  EXPECT_FALSE(decode_.release_reservation("req", "wrong"));
  EXPECT_FALSE(decode_.release_reservation("req", ""));
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  auto replacement = reserve("req", "replacement");
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  EXPECT_GE(replacement->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_GE(other->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("other", "reservation"));
  EXPECT_TRUE(decode_.release_reservation("req", "replacement"));
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
}

TEST_F(ReservationTest, ReleasePreventsLateHandoff) {
  EXPECT_FALSE(decode_.release_reservation("early", "reservation"));
  auto late = make_request({11, 12, 13, 14}, "early");
  late->state().pd_reservation_id = "reservation";
  ASSERT_TRUE(decode_.try_allocate(late->sequences()[0].get()));
  EXPECT_FALSE(decode_.decode_schedule(late, "prefill"));
  EXPECT_EQ(late->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
  auto request = reserve();
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  auto replacement = reserve("req", "replacement");
  EXPECT_FALSE(recv_first_generation(
      &decode_, torch::tensor({1.0f}), 0, 0.1, 0.0, "reservation"));
  std::shared_ptr<Request> queued;
  EXPECT_FALSE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_GE(replacement->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(recv_first_generation(
      &decode_, torch::tensor({1.0f}), 0, 0.1, 0.0, "replacement"));
  ASSERT_TRUE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, replacement);
  EXPECT_TRUE(decode_.reservations_empty());
  engine_.block_manager_pool()->deallocate_without_cache(
      queued->sequences()[0].get());
}

TEST_F(ReservationTest, HandoffPreventsReleaseOfRunningResources) {
  auto request = reserve();
  EXPECT_TRUE(recv_first_generation(
      &decode_, torch::tensor({1.0f}), 0, 0.1, 0.0, "reservation"));
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
  EXPECT_GE(request->sequences()[0]->get_embedding_block_id(), 0);
  std::shared_ptr<Request> queued;
  ASSERT_TRUE(decode_.pop_decode_request_for_test(&queued));
  EXPECT_EQ(queued, request);
  EXPECT_TRUE(decode_.reservations_empty());
  engine_.block_manager_pool()->deallocate_without_cache(
      queued->sequences()[0].get());
}

TEST_F(ReservationTest, ConcurrentHandoffAndReleaseHaveOneOwner) {
  auto request = reserve();
  std::barrier start(2);
  auto release = std::async(std::launch::async, [&] {
    start.arrive_and_wait();
    return decode_.release_reservation("req", "reservation");
  });
  start.arrive_and_wait();
  const bool handed_off = recv_first_generation(
      &decode_, torch::tensor({1.0f}), 0, 0.1, 0.0, "reservation");
  EXPECT_NE(release.get(), handed_off);
  EXPECT_TRUE(decode_.reservations_empty());
  std::shared_ptr<Request> queued;
  EXPECT_EQ(decode_.pop_decode_request_for_test(&queued), handed_off);
  if (handed_off) {
    engine_.block_manager_pool()->deallocate_without_cache(
        queued->sequences()[0].get());
  }
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
}

TEST_F(ReservationTest, ReleasePreservesValidPrefixOnly) {
  auto* pool = engine_.block_manager_pool();
  auto seed = make_request({1, 2, 3, 4}, "seed");
  auto* seq = seed->sequences()[0].get();
  ASSERT_TRUE(pool->allocate(seq));
  const int32_t dp_rank = seq->dp_rank();
  finish_prefill(seq);
  decode_.cache_prefill_blocks_for_test(seed.get());
  pool->deallocate(seed.get());
  const auto cache_before = pool->num_blocks_in_prefix_cache();
  auto request = make_request({1, 2, 3, 4, 5, 6, 7, 8});
  request->state().pd_reservation_id = "reservation";
  request->sequences()[0]->set_dp_rank(dp_rank);
  ASSERT_TRUE(decode_.try_allocate(request->sequences()[0].get()));
  ASSERT_EQ(
      request->sequences()[0]->kv_state().shared_blocks_num(BlockType::KV), 2u);
  ASSERT_TRUE(decode_.decode_schedule(request, "prefill"));
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
  EXPECT_EQ(pool->num_blocks_in_prefix_cache(), cache_before);
  EXPECT_EQ(pool->num_used_blocks(), (std::vector<size_t>{0, 0}));
  auto probe = make_request({1, 2, 3, 4, 5, 6, 7, 8}, "probe");
  probe->sequences()[0]->set_dp_rank(dp_rank);
  pool->allocate_shared(probe->sequences()[0].get());
  EXPECT_EQ(probe->sequences()[0]->kv_state().shared_blocks_num(BlockType::KV),
            2u);
  pool->deallocate_without_cache(probe->sequences()[0].get());
}

TEST_F(ReservationTest, RejectedAllocationRollsBackWithoutReservation) {
  auto* pool = engine_.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  auto rejected = make_request(std::vector<int32_t>(40, 1));
  EXPECT_FALSE(decode_.try_allocate(rejected->sequences()[0].get()));
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_used_blocks(), (std::vector<size_t>{0, 0}));
  EXPECT_EQ(rejected->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_FALSE(decode_.release_reservation("req", "reservation"));
}

TEST_F(ReservationTest, UnacceptedLocalFailureSendsNoRelease) {
  FakeEngine engine(/*num_blocks=*/2, /*block_size=*/2);
  TestDisaggPDScheduler prefill(&engine, make_options());
  auto local = accepted();
  local->state().decode_rpc_address.clear();
  prefill.admit_prefill(local);
  EXPECT_TRUE(prefill.prepare_batch()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.calls, 0);
  EXPECT_FALSE(prefill.has_channel(local->request_id()));
}

class ChunkReservationTest : public ReservationTest,
                             public ::testing::WithParamInterface<bool> {};

TEST_P(ChunkReservationTest, ChunkFailureWaitsForPriorPushCompletion) {
  auto remote = reserve();
  FakeEngine engine(/*num_blocks=*/8, /*block_size=*/2);
  auto options = make_options();
  options.enable_schedule_overlap(GetParam())
      .enable_chunked_prefill(true)
      .max_tokens_per_chunk_for_prefill(2)
      .max_tokens_per_batch(2);
  TestDisaggPDScheduler prefill(&engine, options);
  auto local = accepted();
  prefill.admit_prefill(local);
  folly::Promise<bool> push;
  std::promise<void> started;
  auto entered = started.get_future();
  engine.forward = [&](BatchGroup& batch) {
    KVTransferCompletion completion;
    completion.add(push.getSemiFuture());
    started.set_value();
    CHECK(completion.wait());
    batch[0][0]->kv_state().set_kv_cache_tokens_num(2);
    return ForwardOutput{};
  };
  auto step = std::async(std::launch::async,
                         [&] { prefill.step(absl::ZeroDuration()); });
  entered.get();
  EXPECT_EQ(service_.calls, 0);
  EXPECT_FALSE(decode_.reservations_empty());
  EXPECT_GE(remote->sequences()[0]->get_embedding_block_id(), 0);
  push.setValue(true);
  step.get();
  auto blocker = make_request(std::vector<int32_t>(12, 99), "blocker");
  ASSERT_TRUE(
      engine.block_manager_pool()->allocate(blocker->sequences()[0].get()));
  EXPECT_TRUE(prefill.prepare_batch()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.released, 1);
  EXPECT_TRUE(decode_.reservations_empty());
  EXPECT_EQ(engine_.block_manager_pool()->num_used_blocks(),
            (std::vector<size_t>{0, 0}));
  engine.block_manager_pool()->deallocate_without_cache(
      blocker->sequences()[0].get());
}
INSTANTIATE_TEST_SUITE_P(OverlapModes, ChunkReservationTest, ::testing::Bool());

TEST_F(ReservationTest, BudgetFailureReleasesOnlyTarget) {
  auto remote = reserve();
  auto other = reserve("other");
  FakeEngine engine(/*num_blocks=*/16, /*block_size=*/2);
  auto options = make_options();
  options.enable_chunked_prefill(false).max_tokens_per_batch(2);
  TestDisaggPDScheduler prefill(&engine, options);
  auto local = accepted();
  prefill.admit_prefill(local);
  EXPECT_TRUE(prefill.prepare_batch()[0].empty());
  prefill.wait_notifications();
  EXPECT_EQ(service_.released, 1);
  EXPECT_EQ(remote->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_GE(other->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("other", "reservation"));
  EXPECT_TRUE(decode_.reservations_empty());
}

TEST_F(ReservationTest, DuplicateAdmissionRollsBackUncomputedResources) {
  auto original = reserve();
  auto duplicate = make_request({11, 12, 13, 14});
  duplicate->state().pd_reservation_id = "duplicate";
  ASSERT_TRUE(decode_.try_allocate(duplicate->sequences()[0].get()));
  BlockManager* embedding = duplicate->sequences()[0]
                                ->kv_state()
                                .blocks(BlockType::EMBEDDING)[0]
                                .manager();
  const size_t embedding_used = embedding->num_used_blocks();
  EXPECT_FALSE(decode_.decode_schedule(duplicate, "prefill"));
  EXPECT_EQ(duplicate->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(embedding->num_used_blocks(), embedding_used - 1);
  EXPECT_EQ(engine_.block_manager_pool()->num_blocks_in_prefix_cache(),
            (std::vector<size_t>{0, 0}));
  EXPECT_GE(original->sequences()[0]->get_embedding_block_id(), 0);
  EXPECT_TRUE(decode_.release_reservation("req", "reservation"));
}
TEST(DisaggPDSchedulerTest, MtpExhaustionRollsBackKvReservation) {
  FakeEngine engine(/*num_blocks=*/8,
                    /*block_size=*/2,
                    /*num_speculative_tokens=*/1,
                    /*dp_size=*/1,
                    /*embedding_blocks=*/2);
  TestDisaggPDScheduler decode(&engine, make_mtp_decode_options());
  auto original = make_request({1, 2, 3, 4});
  original->state().pd_reservation_id = "reservation";
  ASSERT_TRUE(decode.try_allocate(original->sequences()[0].get()));
  ASSERT_TRUE(decode.decode_schedule(original, "prefill"));
  auto* pool = engine.block_manager_pool();
  const auto free_before = pool->num_free_blocks();
  auto rejected = make_request({11, 12, 13, 14}, "rejected");
  EXPECT_FALSE(decode.try_allocate(rejected->sequences()[0].get()));
  EXPECT_EQ(rejected->sequences()[0]->get_embedding_block_id(), -1);
  EXPECT_EQ(pool->num_free_blocks(), free_before);
  EXPECT_EQ(pool->num_used_blocks()[0], 2u);
  EXPECT_EQ(first_cache_size(*pool), 0u);
  EXPECT_FALSE(decode.release_reservation("rejected", "reservation"));
  EXPECT_TRUE(decode.release_reservation("req", "reservation"));
  EXPECT_TRUE(decode.reservations_empty());
  EXPECT_EQ(pool->num_used_blocks()[0], 0u);
}
}  // namespace xllm
