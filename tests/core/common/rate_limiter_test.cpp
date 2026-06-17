#include "rate_limiter.h"

#include <gtest/gtest.h>

#include "core/framework/config/service_config.h"

namespace xllm {

TEST(RequestLimiterTest, Basic) {
  // Set the maximum number of concurrent requests to 1.
  ServiceConfig::get_instance().max_concurrent_requests(1);
  RateLimiter rate_limiter;
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
  // The current number of concurrent requests is 1, rate limiting is applied.
  EXPECT_EQ(rate_limiter.is_limited(), true);
  // Decrease the number of concurrent requests by one, changing the concurrency
  // from 1 to 0.
  rate_limiter.decrease_one_request();
  // The current number of concurrent requests is 0, no rate limiting is
  // applied.
  EXPECT_EQ(rate_limiter.is_limited(), false);
}

}  // namespace xllm
