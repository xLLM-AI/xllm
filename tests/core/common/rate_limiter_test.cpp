#include "rate_limiter.h"

#include <gtest/gtest.h>

#include "core/framework/config/service_config.h"

namespace xllm {

TEST(RequestLimiterTest, CheckAndIncrement) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;

  // count=0, not limited; check_limited() must not have side effects.
  EXPECT_FALSE(rate_limiter.check_limited());
  EXPECT_FALSE(rate_limiter.check_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);

  // increment() must be called separately.
  rate_limiter.increment();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);

  // count=1 reached the max, now limited.
  EXPECT_TRUE(rate_limiter.check_limited());

  rate_limiter.decrease_one_request();
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);
  EXPECT_FALSE(rate_limiter.check_limited());
}

TEST(RequestLimiterTest, IsLimitedIsReadOnlyAlias) {
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;

  // is_limited() must be a read-only alias for check_limited().
  EXPECT_FALSE(rate_limiter.is_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 0);

  rate_limiter.increment();
  EXPECT_TRUE(rate_limiter.is_limited());
  EXPECT_EQ(rate_limiter.get_num_concurrent_requests(), 1);

  rate_limiter.decrease_one_request();
}

TEST(RequestLimiterTest, SleepBlocksCheck) {
  ServiceConfig::get_instance().max_concurrent_requests(10);
  RateLimiter rate_limiter;

  EXPECT_TRUE(rate_limiter.try_set_sleeping());
  EXPECT_TRUE(rate_limiter.check_limited());
  EXPECT_TRUE(rate_limiter.is_sleeping());

  EXPECT_TRUE(rate_limiter.try_wakeup());
  EXPECT_FALSE(rate_limiter.check_limited());
}

}  // namespace xllm
