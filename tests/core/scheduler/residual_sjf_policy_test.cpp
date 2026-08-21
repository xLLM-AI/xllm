/* Copyright 2026 The xLLM Authors.

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

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <gtest/gtest.h>

#include <cstdint>
#include <map>
#include <memory>
#include <string>
#include <vector>

#include "continuous_scheduler.h"
#include "core/framework/block/block_manager_pool.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "scheduler/request_priority_queue.h"
#include "scheduler/scheduler_policy.h"

namespace xllm {
namespace {

std::shared_ptr<Request> make_request(const std::string& request_id,
                                      size_t prompt_tokens,
                                      int32_t token_id = 1) {
  std::vector<int32_t> prompt_token_ids(prompt_tokens, token_id);
  RequestSamplingParam sampling_param;
  SchedulerParam scheduler_param;

  StoppingChecker stopping_checker;
  stopping_checker.set_max_generated_tokens(4);
  stopping_checker.set_max_context_len(4096);
  stopping_checker.set_ignore_eos(true);

  RequestState state("prompt",
                     prompt_token_ids,
                     sampling_param,
                     scheduler_param,
                     stopping_checker,
                     prompt_token_ids.size() + 8,
                     /*n=*/1,
                     /*best_of=*/1,
                     /*logprobs=*/false,
                     /*stream=*/false,
                     /*echo=*/false,
                     /*skip_special_tokens=*/false,
                     /*enable_schedule_overlap=*/false,
                     /*output_func=*/nullptr,
                     /*outputs_func=*/nullptr);
  return std::make_shared<Request>(
      request_id, "x-request-id", "x-request-time", state);
}

std::vector<std::string> collect_ids(const RequestPriorityQueue& queue) {
  std::vector<std::string> request_ids;
  for (auto it = queue.begin(); it != queue.end(); ++it) {
    request_ids.emplace_back((*it)->request_id());
  }
  return request_ids;
}

ResidualSJFComparator::ResidualCostFn cost_from_map(
    const std::map<std::string, size_t>& cost_by_id) {
  return [&cost_by_id](const std::shared_ptr<Request>& request) -> size_t {
    const auto found = cost_by_id.find(request->request_id());
    return found == cost_by_id.end() ? 0 : found->second;
  };
}

void sort_residual_sjf(DequeQueue& queue,
                       const std::map<std::string, size_t>& cost_by_id,
                       int32_t max_wait_ms,
                       absl::Time now) {
  queue.sort(
      ResidualSJFComparator(cost_from_map(cost_by_id), max_wait_ms, now));
}

TEST(ResidualSJFPolicyTest, QuantizesResidualCostToBlocks) {
  EXPECT_EQ(residual_sjf_cost_blocks(1024, 4, 128), 4);
  EXPECT_EQ(residual_sjf_cost_blocks(1024, 0, 128), 8);
  EXPECT_EQ(residual_sjf_cost_blocks(1024, 8, 128), 0);
  EXPECT_EQ(residual_sjf_cost_blocks(1024, 16, 128), 0);
  EXPECT_EQ(residual_sjf_cost_blocks(1024, 4, 0), 0);
}

TEST(ResidualSJFPolicyTest, SortsFreshRequestsByResidualCostThenArrival) {
  DequeQueue queue;
  queue.push(make_request("cold-short", 256), /*if_back=*/false);
  queue.push(make_request("warm-long", 2048), /*if_back=*/false);
  queue.push(make_request("cold-long", 2048), /*if_back=*/false);
  queue.push(make_request("warm-short", 256), /*if_back=*/false);

  sort_residual_sjf(queue,
                    {{"warm-long", 1},
                     {"cold-short", 2},
                     {"warm-short", 2},
                     {"cold-long", 16}},
                    /*max_wait_ms=*/60000,
                    absl::Now());

  // warm-long has the smallest residual cost; the two cost-2 requests
  // tie-break by arrival (cold-short was created before warm-short).
  const std::vector<std::string> expected = {
      "warm-long", "cold-short", "warm-short", "cold-long"};
  EXPECT_EQ(collect_ids(queue), expected);
}

TEST(ResidualSJFPolicyTest, AgedRequestsPrecedeFreshInFcfsOrder) {
  std::shared_ptr<Request> old_long = make_request("old-long", 2048);
  // Give the first request a real age so it crosses the max-wait threshold
  // while the second one stays fresh.
  absl::SleepFor(absl::Milliseconds(20));
  std::shared_ptr<Request> new_short = make_request("new-short", 256);
  DequeQueue queue;
  queue.push(old_long, /*if_back=*/false);
  queue.push(new_short, /*if_back=*/false);

  // The aged request is promoted ahead of the fresh one in FCFS order,
  // ignoring residual cost (old-long cost 100 > new-short cost 1).
  sort_residual_sjf(queue,
                    {{"old-long", 100}, {"new-short", 1}},
                    /*max_wait_ms=*/10,
                    absl::Now());

  const std::vector<std::string> expected = {"old-long", "new-short"};
  EXPECT_EQ(collect_ids(queue), expected);
}

TEST(ResidualSJFPolicyTest, ZeroMaxWaitBehavesLikeFcfs) {
  DequeQueue queue;
  queue.push(make_request("first-expensive", 2048), /*if_back=*/false);
  queue.push(make_request("second-cheap", 256), /*if_back=*/false);

  // max_wait_ms=0 ages every request immediately, so cost is ignored and the
  // earlier arrival wins.
  sort_residual_sjf(queue,
                    {{"first-expensive", 100}, {"second-cheap", 1}},
                    /*max_wait_ms=*/0,
                    absl::Now());

  const std::vector<std::string> expected = {"first-expensive", "second-cheap"};
  EXPECT_EQ(collect_ids(queue), expected);
}

TEST(ResidualSJFPolicyTest, RecoveryPrecedesAgedAndFresh) {
  std::shared_ptr<Request> aged = make_request("aged", 256);
  absl::SleepFor(absl::Milliseconds(20));
  std::shared_ptr<Request> fresh = make_request("fresh", 256);
  std::shared_ptr<Request> recovery = make_request("recovery", 256);
  recovery->set_preempted();
  DequeQueue queue;
  queue.push(fresh, /*if_back=*/false);
  queue.push(recovery, /*if_back=*/false);
  queue.push(aged, /*if_back=*/false);

  sort_residual_sjf(queue,
                    {{"fresh", 1}, {"recovery", 1}, {"aged", 1}},
                    /*max_wait_ms=*/10,
                    absl::Now());

  const std::vector<std::string> expected = {"recovery", "aged", "fresh"};
  EXPECT_EQ(collect_ids(queue), expected);
}

TEST(ResidualSJFPolicyTest, ReadOnlyProbeReportsLocalPrefixHit) {
  BlockManagerPool::Options opt;
  opt.num_blocks_ = 4096;
  opt.block_size_ = 128;
  opt.max_seqs_per_batch_ = 1024;
  opt.enable_prefix_cache_ = true;
  BlockManagerPool pool(opt, /*dp_size=*/1);

  std::shared_ptr<Request> warm = make_request("warm", 1024);
  Sequence* warm_seq = warm->sequences()[0].get();
  EXPECT_EQ(pool.get_num_local_computed_blocks(warm_seq), 0);
  ASSERT_TRUE(pool.allocate(warm_seq));
  // Simulate a completed prefill so the final cache flush covers the prompt.
  warm_seq->kv_state().set_kv_cache_tokens_num(1024);
  pool.cache(warm_seq);

  std::shared_ptr<Request> hit = make_request("hit", 1024);
  Sequence* hit_seq = hit->sequences()[0].get();
  // All 8 full blocks are locally cached: the read-only probe reports the
  // hit block count without allocating or mutating state.
  EXPECT_EQ(pool.get_num_local_computed_blocks(hit_seq), 8);

  std::shared_ptr<Request> miss = make_request("miss", 1024, /*token_id=*/2);
  Sequence* miss_seq = miss->sequences()[0].get();
  EXPECT_EQ(pool.get_num_local_computed_blocks(miss_seq), 0);

  // A second probe on the same sequence stays stable (no state mutation).
  EXPECT_EQ(pool.get_num_local_computed_blocks(hit_seq), 8);
}

TEST(ResidualSJFPolicyTest, ProbeIsZeroWhenPrefixCacheDisabled) {
  BlockManagerPool::Options opt;
  opt.num_blocks_ = 4096;
  opt.block_size_ = 128;
  opt.max_seqs_per_batch_ = 1024;
  opt.enable_prefix_cache_ = false;
  BlockManagerPool pool(opt, /*dp_size=*/1);

  std::shared_ptr<Request> request = make_request("cold", 1024);
  EXPECT_EQ(pool.get_num_local_computed_blocks(request->sequences()[0].get()),
            0);
}

TEST(ResidualSJFPolicyTest, FactorySelectsResidualSJFPolicy) {
  ContinuousScheduler::Options options;
  options.enable_disagg_pd(true).enable_chunked_prefill(true);
  options.instance_role(InstanceRole::PREFILL);
  BatchMode mode;
  mode.priority_strategy = "residual_sjf";
  std::unique_ptr<SchedulerPolicy> policy =
      create_scheduler_policy(mode, options);
  EXPECT_NE(dynamic_cast<ResidualSJFPolicy*>(policy.get()), nullptr);
}

TEST(ResidualSJFPolicyTest, FactoryDefaultsToPrefillFirstForFcfs) {
  ContinuousScheduler::Options options;
  options.enable_disagg_pd(true).enable_chunked_prefill(true);
  options.instance_role(InstanceRole::PREFILL);
  BatchMode mode;
  mode.priority_strategy = "fcfs";
  std::unique_ptr<SchedulerPolicy> policy =
      create_scheduler_policy(mode, options);
  EXPECT_NE(dynamic_cast<PrefillFirstPolicy*>(policy.get()), nullptr);
}

TEST(ResidualSJFPolicyTest, FactoryRejectsResidualSJfWithoutDisaggPd) {
  ContinuousScheduler::Options options;
  BatchMode mode;
  mode.priority_strategy = "residual_sjf";
  EXPECT_DEATH(create_scheduler_policy(mode, options), "enable_disagg_pd");
}

TEST(ResidualSJFPolicyTest, FactoryRejectsResidualSJfWithMixBatch) {
  ContinuousScheduler::Options options;
  options.enable_disagg_pd(true).enable_chunked_prefill(true);
  options.instance_role(InstanceRole::PREFILL);
  BatchMode mode;
  mode.priority_strategy = "residual_sjf";
  mode.enable_mix_batch = true;
  EXPECT_DEATH(create_scheduler_policy(mode, options), "enable_mix_batch");
}

}  // namespace
}  // namespace xllm
