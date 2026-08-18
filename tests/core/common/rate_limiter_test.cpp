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

}  // namespace xllm
