#include "rate_limiter.h"

#include <gtest/gtest.h>

#include "core/framework/config/service_config.h"

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

}  // namespace xllm
