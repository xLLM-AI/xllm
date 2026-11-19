#include "rate_limiter.h"

#include <gtest/gtest.h>

#include <atomic>
#include <barrier>
#include <thread>

#include "core/framework/config/service_config.h"
#include "core/util/scope_guard.h"

namespace xllm {

TEST(RequestLimiterTest, Basic) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;

  EXPECT_FALSE(rate_limiter.is_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);

  EXPECT_TRUE(rate_limiter.is_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);

  rate_limiter.decrease_one_request();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  EXPECT_FALSE(rate_limiter.is_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);

  rate_limiter.decrease_one_request();
}

TEST(RequestLimiterTest, NoLimitWhenMaxIsZero) {
  ServiceConfig::get_instance().max_concurrent_requests(0);
  RateLimiter rate_limiter;

  for (int i = 0; i < 100; ++i) {
    EXPECT_FALSE(rate_limiter.is_limited());
  }
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 100);

  for (int i = 0; i < 100; ++i) {
    rate_limiter.decrease_one_request();
  }
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
}

TEST(RequestLimiterTest, SleepBlocksAcquisition) {
  ServiceConfig::get_instance().max_concurrent_requests(10);
  RateLimiter rate_limiter;

  EXPECT_TRUE(rate_limiter.try_set_sleeping());
  EXPECT_TRUE(rate_limiter.is_sleeping());
  // is_limited returns true (reject) while sleeping and does NOT change the
  // sleep sentinel or increment anything.
  EXPECT_TRUE(rate_limiter.is_limited());
  EXPECT_TRUE(rate_limiter.is_sleeping());

  EXPECT_TRUE(rate_limiter.try_wakeup());
  EXPECT_FALSE(rate_limiter.is_sleeping());
  EXPECT_FALSE(rate_limiter.is_limited());
  rate_limiter.decrease_one_request();
}

TEST(RequestLimiterTest, AdmissionPreservesRejectionReasonAndCounter) {
  auto& config = ServiceConfig::get_instance();
  const int32_t previous_limit = config.max_concurrent_requests();
  config.max_concurrent_requests(1);
  RateLimiter rate_limiter;

  EXPECT_TRUE(rate_limiter.acquire().ok());
  for (int32_t attempt = 0; attempt < 3; ++attempt) {
    const Status status = rate_limiter.acquire();
    EXPECT_EQ(status.code(), StatusCode::RATE_LIMITED);
    EXPECT_EQ(status.message(),
              "The number of concurrent requests has reached the limit.");
    EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);
  }
  EXPECT_FALSE(rate_limiter.try_set_sleeping());
  rate_limiter.decrease_one_request();
  EXPECT_TRUE(rate_limiter.try_set_sleeping());
  const Status sleeping = rate_limiter.acquire();
  EXPECT_EQ(sleeping.code(), StatusCode::UNAVAILABLE);
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), RateLimiter::kSleeping);
  EXPECT_TRUE(rate_limiter.try_wakeup());
  // The returned rejection reason is not a subsequent sleep-state snapshot.
  EXPECT_EQ(sleeping.code(), StatusCode::UNAVAILABLE);
  EXPECT_TRUE(rate_limiter.acquire().ok());
  rate_limiter.decrease_one_request();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  config.max_concurrent_requests(previous_limit);
}

TEST(RequestLimiterTest, BatchSlotsReleaseIndividuallyBeforeSleep) {
  auto& config = ServiceConfig::get_instance();
  const int32_t previous_limit = config.max_concurrent_requests();
  ScopeGuard restore([&] { config.max_concurrent_requests(previous_limit); });
  config.max_concurrent_requests(3);
  RateLimiter limiter;

  ASSERT_TRUE(limiter.acquire(2).ok());
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 2);
  limiter.decrease_one_request();
  EXPECT_FALSE(limiter.try_set_sleeping());
  limiter.decrease_one_request();
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 0);
  ASSERT_TRUE(limiter.try_set_sleeping());
  EXPECT_EQ(limiter.acquire(2).code(), StatusCode::UNAVAILABLE);
  EXPECT_TRUE(limiter.try_wakeup());
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 0);
}

TEST(RequestLimiterTest, BatchRejectionDoesNotReservePartialCapacity) {
  auto& config = ServiceConfig::get_instance();
  const int32_t previous_limit = config.max_concurrent_requests();
  ScopeGuard restore([&] { config.max_concurrent_requests(previous_limit); });
  config.max_concurrent_requests(3);
  RateLimiter limiter;

  ASSERT_TRUE(limiter.acquire(2).ok());
  EXPECT_EQ(limiter.acquire(2).code(), StatusCode::RATE_LIMITED);
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 2);
  ASSERT_TRUE(limiter.acquire().ok());
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 3);
  for (int32_t index = 0; index < 3; ++index) {
    limiter.decrease_one_request();
  }
  EXPECT_TRUE(limiter.try_set_sleeping());
}

TEST(RequestLimiterTest, ConcurrentBatchesReserveWholeRequests) {
  auto& config = ServiceConfig::get_instance();
  const int32_t previous_limit = config.max_concurrent_requests();
  ScopeGuard restore([&] { config.max_concurrent_requests(previous_limit); });
  config.max_concurrent_requests(3);
  RateLimiter limiter;
  std::barrier start(3);
  std::atomic<int32_t> accepted{0};
  const auto reserve = [&] {
    start.arrive_and_wait();
    if (limiter.acquire(2).ok()) {
      accepted.fetch_add(1, std::memory_order_relaxed);
    }
  };
  std::thread first(reserve);
  std::thread second(reserve);
  start.arrive_and_wait();
  first.join();
  second.join();

  EXPECT_EQ(accepted.load(), 1);
  EXPECT_EQ(limiter.get_num_concurrent_requests(), 2);
  limiter.decrease_one_request();
  limiter.decrease_one_request();
  EXPECT_TRUE(limiter.try_set_sleeping());
}

}  // namespace xllm
