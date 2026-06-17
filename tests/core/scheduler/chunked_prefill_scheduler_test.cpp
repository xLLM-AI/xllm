/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "chunked_prefill_scheduler.h"

#include <absl/time/clock.h>
#include <gtest/gtest.h>

#include <cmath>
#include <optional>

#include "core/framework/config/scheduler_config.h"
#include "distributed_runtime/engine.h"
#include "util/utils.h"

namespace xllm {

namespace {
class FakeTokenizer : public Tokenizer {
 public:
  bool encode(const std::string_view& text,
              std::vector<int32_t>* ids,
              bool add_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::string decode(const Slice<int32_t>& ids,
                     bool skip_special_tokens) const {
    NOT_IMPLEMENTED();
  }
  std::optional<int32_t> token_to_id(const std::string_view& token) const {
    NOT_IMPLEMENTED();
  }
  std::string id_to_token(int32_t id) const { NOT_IMPLEMENTED(); }
  size_t vocab_size() const { NOT_IMPLEMENTED(); }
  std::unique_ptr<Tokenizer> clone() const {
    return std::make_unique<FakeTokenizer>();
  }
};

class FakeEngine : public Engine {
 public:
  FakeEngine(int32_t num_blocks, int32_t block_size) {
    BlockManagerPool::Options opt;
    opt.num_blocks_ = num_blocks;
    opt.block_size_ = block_size;
    opt.enable_prefix_cache_ = false;  // we dont consider prefix cache here
    fake_tokenizer_ = std::make_unique<FakeTokenizer>();
    fake_block_manager_ = std::make_unique<BlockManagerPool>(opt, 1);
  }
  ForwardOutput step(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  void update_last_step_result(std::vector<Batch>& batch) { NOT_IMPLEMENTED(); }
  const Tokenizer* tokenizer() const { return fake_tokenizer_.get(); }
  BlockManagerPool* block_manager_pool() const {
    return fake_block_manager_.get();
  }
  const ModelArgs& model_args() const { NOT_IMPLEMENTED(); }
  const TokenizerArgs& tokenizer_args() const { NOT_IMPLEMENTED(); }
  std::vector<int64_t> get_active_activation_memory() const {
    NOT_IMPLEMENTED();
  }
  bool init() override { return true; }

 private:
  std::unique_ptr<Tokenizer> fake_tokenizer_;
  std::unique_ptr<BlockManagerPool> fake_block_manager_;
};

template <typename T>
class ScopedConfigValue final {
 public:
  ScopedConfigValue(T& value, T new_value) : value_(value), old_(value) {
    value_ = new_value;
  }

  ~ScopedConfigValue() { value_ = old_; }

 private:
  T& value_;
  T old_;
};

ContinuousScheduler::Options create_scheduler_options(
    int32_t max_tokens_per_batch,
    int32_t max_seqs_per_batch,
    int32_t num_speculative_tokens,
    int32_t max_tokens_per_chunk_for_prefill,
    int32_t dp_size,
    const std::string& priority_strategy = "fcfs",
    bool enable_profile_kv_blocks = true,
    bool enable_latency_aware_schedule = false,
    int32_t max_global_ttft_ms = std::numeric_limits<int32_t>::max(),
    int32_t max_global_tpot_ms = std::numeric_limits<int32_t>::max()) {
  ContinuousScheduler::Options opt;
  opt.num_speculative_tokens_ = num_speculative_tokens;
  opt.max_tokens_per_chunk_for_prefill_ = max_tokens_per_chunk_for_prefill;
  opt.max_tokens_per_batch_ = max_tokens_per_batch;
  opt.max_seqs_per_batch_ = max_seqs_per_batch;
  opt.dp_size_ = dp_size;
  opt.priority_strategy_ = priority_strategy;
  opt.enable_profile_kv_blocks_ = enable_profile_kv_blocks;
  opt.enable_latency_aware_schedule_ = enable_latency_aware_schedule;
  opt.max_global_ttft_ms_ = max_global_ttft_ms;
  opt.max_global_tpot_ms_ = max_global_tpot_ms;
  return opt;
}

std::vector<std::shared_ptr<Request>> generate_request(
    const std::vector<int32_t>& prompt_lens,
    const std::vector<int32_t>& max_tokens,
    std::optional<std::vector<bool>> offlines,
    std::optional<std::vector<int32_t>> priorities,
    int32_t max_context_len) {
  std::vector<std::shared_ptr<Request>> requests;
  EXPECT_TRUE(prompt_lens.size() == max_tokens.size());

  size_t batch_size = prompt_lens.size();
  std::vector<bool> offline_vec;
  std::vector<int32_t> priority_vec;
  if (offlines.has_value()) {
    offline_vec = *offlines;
  } else {
    offline_vec = std::vector<bool>(batch_size, false);
  }

  if (priorities.has_value()) {
    priority_vec = priorities.value();
  } else {
    priority_vec = std::vector<int32_t>(
        batch_size, static_cast<int32_t>(RequestPriority::NORMAL));
  }

  for (size_t i = 0; i < batch_size; ++i) {
    std::vector<int32_t> prompt_token_ids;
    prompt_token_ids.resize(prompt_lens[i]);
    RequestSamplingParam sampling_param;
    SchedulerParam scheduler_param;
    scheduler_param.offline = offline_vec[i];
    scheduler_param.priority = static_cast<RequestPriority>(priority_vec[i]);

    StoppingChecker stopping_checker;
    stopping_checker.set_max_generated_tokens(max_tokens[i]);
    stopping_checker.set_max_context_len(max_context_len);
    stopping_checker.set_ignore_eos(true);

    RequestState req_state("x",
                           prompt_token_ids,
                           sampling_param,
                           scheduler_param,
                           stopping_checker,
                           prompt_lens[i] + 30000,
                           1,
                           1,
                           false,
                           false,
                           false,
                           false,
                           false,
                           nullptr,
                           nullptr);

    auto request =
        std::make_shared<Request>("1", "1", "1", std::move(req_state), "1");
    requests.emplace_back(request);
  }

  return requests;
}

// dont not consider speculative decoding.
void update_requests(std::vector<std::shared_ptr<Request>> requests) {
  for (auto req : requests) {
    for (auto& seq : req->sequences()) {
      if (seq->kv_state().kv_cache_tokens_num() == 0) {
        seq->kv_state().incr_kv_cache_tokens_num(seq->num_prompt_tokens());
      } else {
        seq->kv_state().incr_kv_cache_tokens_num(1);
      }
      Token token(1);
      seq->append_token(token);
    }
  }
}

}  // namespace

// TEST-1:
// Three independent prefill requests, according to the configs,
// verify how many tokens are processed in one scheduling.
TEST(ChunkedPrefillSchedulerTest, AddNewRequestBase) {
  std::vector<int32_t> prompt_len{10, 1024, 2048};
  std::vector<int32_t> num_blocks{16, 128, 128};
  std::vector<int32_t> block_size{16, 16, 16};
  std::vector<int32_t> validate_allowed_max_tokens{10, 1024, 1024};
  for (size_t idx = 0; idx < prompt_len.size(); ++idx) {
    ContinuousScheduler::Options opt =
        create_scheduler_options(10000, 256, 0, 1024, 1);
    auto engine =
        std::make_unique<FakeEngine>(num_blocks[idx], block_size[idx]);
    auto scheduler =
        std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // create requests
    auto requests = generate_request(
        {prompt_len[idx]}, {10}, std::nullopt, std::nullopt, 10000);
    for (auto req : requests) {
      scheduler->add_request(req);
    }

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    const std::vector<uint32_t>& allowed_max_tokens =
        batch[0].get_allowed_max_tokens();
    for (size_t i = 0; i < batch[0].size(); ++i) {
      auto seq = batch[0][i];
      EXPECT_TRUE(allowed_max_tokens[i] == validate_allowed_max_tokens[idx]);
    }
  }
}

// TEST-2:
// memory or budget not enough
TEST(ChunkedPrefillSchedulerTest, ResourceNotEnough) {
  // case1: max tokens budget not enough
  {
    // max token budget: 0
    ContinuousScheduler::Options opt =
        create_scheduler_options(1, 256, 0, 1024, 1);
    auto engine = std::make_unique<FakeEngine>(16, 16);
    auto scheduler =
        std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // request prompt len: 100
    auto requests =
        generate_request({1, 100}, {1, 10}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);  // can not schedule the second.
  }

  // case2: blocks memory not enough
  {
    ContinuousScheduler::Options opt =
        create_scheduler_options(1000, 256, 0, 1024, 1);
    // free block slot: 1
    auto engine = std::make_unique<FakeEngine>(2, 8);
    auto scheduler =
        std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
    EXPECT_TRUE(scheduler != nullptr);

    // request prompt len: 1000
    auto requests = generate_request(
        {1, 1000}, {1, 100}, std::nullopt, std::nullopt, 10000);
    scheduler->add_request(requests[0]);
    scheduler->add_request(requests[1]);

    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 1);  // can not schedule the second.
  }
}

// TEST-3:
// schdule decoding requests + some prefill requests
TEST(ChunkedPrefillSchedulerTest, NormalSchedule) {
  // set max free blocks: 512, support 512*32=16384 tokens
  int block_num = 512;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // 1. schedule some new prefill requests
  auto requests = generate_request(
      {100, 200, 300}, {10, 20, 30}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto total_blocks =
      util::max(engine->block_manager_pool()->num_free_blocks());
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  auto seq_use_blocks =
      (100 / block_size + 1) + (200 / block_size + 1) + (300 / block_size + 1);
  EXPECT_TRUE(util::max(engine->block_manager_pool()->num_free_blocks()) ==
              (total_blocks - seq_use_blocks));
  update_requests(requests);

  // 2. schedule decoding requets
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  update_requests(requests);

  // 3. add new prefill requests
  auto requests1 =
      generate_request({400, 500}, {40, 50}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests1) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 5);
  update_requests(requests);
  update_requests(requests1);

  // 4. add long new prefill requests,
  // memory is enough, can handled tokens from
  // max_tokens_per_chunk_for_prefill(1024) -> remain_max_tokens_budget
  auto requests2 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests2) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 6);
  const std::vector<uint32_t>& allowed_max_tokens =
      batch[0].get_allowed_max_tokens();
  // memory is enough, can handled tokens from 1024 -> remain_max_tokens_budget
  EXPECT_TRUE(allowed_max_tokens[5] == 10000 - 5);
  update_requests(requests);
  update_requests(requests1);
  update_requests(requests2);

  // 5. add long new prefill requests,
  // memory is not enough, only handled `max_tokens_per_chunk_for_prefill`(1024)
  // tokens
  auto requests3 =
      generate_request({10000}, {10}, std::nullopt, std::nullopt, 30000);
  for (auto req : requests3) {
    scheduler->add_request(req);
  }
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 7);
  const std::vector<uint32_t>& allowed_max_tokens1 =
      batch[0].get_allowed_max_tokens();
  // only can handle max_tokens_per_chunk_for_prefill tokens.
  EXPECT_TRUE(allowed_max_tokens1[6] == 1024);
}

// TEST-4:
// test preempt
TEST(ChunkedPrefillSchedulerTest, PreemptSchedule) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule some new prefill requests
  // request-1 has higher priority than request-2
  auto requests =
      generate_request({127, 127}, {10, 10}, std::nullopt, std::nullopt, 30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
  // append a new block
  block_manager_pool->allocate(batch[0][0]);
  // remove preempted request from running_requests
  running_requests.pop_back();
  update_requests(running_requests);

  // 2.The preempted request was saved in waiting_priority_queue_,
  // it just like a new prefill request.
  // Continue to run, the new prefill request will not preempt
  // request in running_queue.
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
}

// TEST-5:
// test on/offline preempt
TEST(ChunkedPrefillSchedulerTest, OnDecodePreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
  BlockManagerPool* block_manager_pool = engine->block_manager_pool();
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. schedule one online and one prefill prefill requests
  auto requests = generate_request({127, 127},
                                   {10, 10},
                                   std::vector<bool>{true, false},
                                   std::vector<int32_t>{2, 2},
                                   30000);
  running_requests = requests;
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  batch = scheduler->prepare_batch_test();

  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  update_requests(running_requests);

  int free_blocks_before_preempt =
      util::max(block_manager_pool->num_free_blocks());
  // 2. after 2 step, preemption should happen
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 1);
  int free_blocks_after_preempt =
      util::max(block_manager_pool->num_free_blocks());
  EXPECT_TRUE(free_blocks_after_preempt > free_blocks_before_preempt);
  // check the running request is online request
  EXPECT_TRUE(scheduler->get_running_requests().size() == 1);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
  EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);
}

// TEST-6:
// test on/offline preempt
TEST(ChunkedPrefillSchedulerTest, OnPrefillPreemptOffDecode) {
  // set max free blocks: 9, support 9*32=288 tokens
  // actually only 8 free blocks , because default 1 block is for padding
  int block_num = 9;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1);
  ScopedConfigValue<double> memory_threshold(
      SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold(),
      2.0);

  {
    // 1. two offline decode requests then one online prefill request preempt
    // them
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler =
        std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;

    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, true},
                                     std::vector<int32_t>{2, 2},
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request({80},
                                         {10},
                                         std::vector<bool>{false},
                                         std::vector<int32_t>{2},
                                         30000);  // use 3 blocks
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);

    // online prefill request preempt offline decode request
    EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
    EXPECT_TRUE(scheduler->get_running_requests()[0]->offline() == false);
    EXPECT_TRUE(scheduler->get_waiting_requests_num() == 1);

    // offline is evicted
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 1);
  }

  // 2. another case: longer online prefill request arrives, but can not evict
  // offline because evicting offline is not enough
  {
    auto engine = std::make_unique<FakeEngine>(block_num, block_size);
    auto scheduler =
        std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
    BlockManagerPool* block_manager_pool = engine->block_manager_pool();
    EXPECT_TRUE(scheduler != nullptr);

    std::vector<std::shared_ptr<Request>> running_requests;
    // 1. schedule one online and one offline
    auto requests = generate_request({100, 100},
                                     {10, 10},
                                     std::vector<bool>{true, false},
                                     std::vector<int32_t>{2, 2},
                                     30000);
    running_requests = requests;
    for (auto req : requests) {
      scheduler->add_request(req);
    }
    auto batch = scheduler->prepare_batch_test();
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(util::max(block_manager_pool->num_free_blocks()) == 0);
    update_requests(running_requests);

    auto new_requests = generate_request(
        {200}, {10}, std::vector<bool>{false}, std::vector<int32_t>{2}, 30000);
    scheduler->add_request(new_requests[0]);
    batch = scheduler->prepare_batch_test();

    // 2. online is still waiting
    EXPECT_TRUE(batch.size() == 1);
    EXPECT_TRUE(batch[0].size() == 2);
    EXPECT_TRUE(scheduler->get_waiting_requests().size() == 1);
    EXPECT_TRUE(scheduler->get_waiting_requests()[0].get() ==
                new_requests[0].get());
  }
}

// TEST-7:
// test priority schedule
TEST(ChunkedPrefillSchedulerTest, PrioritySchedule) {
  // set max free blocks: 12
  // actually only 11 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 1024;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt = create_scheduler_options(
      10000, 256, 0, max_tokens_per_chunk_for_prefill, 1, "priority");
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1: HIGH, 2: NORMAL, 3: LOW
  auto requests = generate_request({127, 127, 127},
                                   {10, 10, 10},
                                   std::vector<bool>{false, false, false},
                                   std::vector<int32_t>{3, 3, 2},
                                   30000);
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::LOW /*LOW*/);

  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // new HIGH priority request arrives, its prefill starts
  auto new_requests = generate_request({32},
                                       {10},
                                       std::vector<bool>{false},
                                       std::vector<int32_t>{1},
                                       30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  // check there are 3 running requests owing to decode-maximal
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 3);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 3);
  running_requests.push_back(new_requests[0]);
  update_requests(running_requests);
  // preemption happens, only HIGH and NORMAL decode requests
  batch = scheduler->prepare_batch_test();
  EXPECT_TRUE(batch.size() == 1);
  EXPECT_TRUE(batch[0].size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests().size() == 2);
  EXPECT_TRUE(scheduler->get_running_requests()[0]->priority() ==
              RequestPriority::HIGH /*HIGH*/);
  EXPECT_TRUE(scheduler->get_running_requests()[1]->priority() ==
              RequestPriority::NORMAL /*NORMAL*/);
}

// TEST-8:
// test latency budget
TEST(ChunkedPrefillSchedulerTest, LatencySchedule) {
  // set max free blocks: 3
  // actually only 2 free blocks , because default 1 block is for padding
  int block_num = 12;
  int block_size = 32;
  int max_tokens_per_chunk_for_prefill = 4;
  // set chunked max_tokens budgets 10000 per step
  ContinuousScheduler::Options opt =
      create_scheduler_options(10000,
                               256,
                               0,
                               max_tokens_per_chunk_for_prefill,
                               1,
                               "fcfs",
                               false,
                               true,
                               350,
                               150);
  auto engine = std::make_unique<FakeEngine>(block_num, block_size);
  auto scheduler = std::make_unique<ChunkedPrefillScheduler>(engine.get(), opt);
  EXPECT_TRUE(scheduler != nullptr);

  // mannuly created profile data for y=0.5x^2+10x
  std::vector<std::pair<int32_t, double>> created_profile_data = {
      {2, 22}, {4, 48}, {6, 78}, {8, 112}};
  auto profile_manager = scheduler->get_profile_manager();
  // fit y=0.5x^2+10x
  profile_manager->train_prefill_time_predictor(created_profile_data);

  auto requests = generate_request(
      {10, 10, 10}, {10, 10, 10}, std::nullopt, std::nullopt, 30000);
  // check if time equation fits well
  EXPECT_TRUE(
      static_cast<int32_t>(std::round(profile_manager->predict_step_time(
          requests[0]->sequences()[0].get(), true, true))) == 150);
  EXPECT_TRUE(static_cast<int32_t>(std::round(
                  profile_manager->predict_step_time(2, 0, true, true))) == 22);

  std::vector<std::shared_ptr<Request>> running_requests;

  // 1. three requests enter prefill
  for (auto req : requests) {
    scheduler->add_request(req);
  }
  auto batch = scheduler->prepare_batch_test();
  EXPECT_EQ(batch.size(), 1);
  EXPECT_EQ(batch[0].size(), 3);
  EXPECT_EQ(scheduler->get_running_requests().size(), 3);
  running_requests = scheduler->get_running_requests();
  update_requests(running_requests);

  // 2. three requests enter decode, one new request chunked prefill
  auto new_requests = generate_request(
      {10}, {10}, std::nullopt, std::nullopt, 30000);  // use 1 blocks
  scheduler->add_request(new_requests[0]);
  batch = scheduler->prepare_batch_test();
  auto running_sequences_budgets = scheduler->get_running_sequences_budgets();

  EXPECT_EQ(batch.size(), 1);
  // tpot = 150 > 3 * 10 , all requests enter decode. But not enough for extra
  // whole prefill, then do chunked
  EXPECT_EQ(batch[0].size(), 4);
  EXPECT_EQ(scheduler->get_running_requests().size(), 4);
  EXPECT_EQ(running_sequences_budgets.size(), 4);
  EXPECT_EQ(running_sequences_budgets[3], max_tokens_per_chunk_for_prefill);
}

}  // namespace xllm
