/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/config/disagg_pd_config.h"

#include <gtest/gtest.h>

#include <string>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"

namespace xllm {
namespace {

struct PrefixRoleCase {
  std::string role;
  bool keep_prefix_cache;
};

void set_unsupported_values(DisaggPDConfig& disagg_pd_config,
                            KVCacheConfig& kv_cache_config,
                            SchedulerConfig& scheduler_config) {
  disagg_pd_config.kv_cache_transfer_type("LlmDataDist")
      .kv_cache_transfer_mode("PULL")
      .enable_pd_ooc(true);
  kv_cache_config.kv_cache_dtype("fp8").enable_prefix_cache(true);
  scheduler_config.enable_schedule_overlap(true);
}

void expect_forced_defaults(const DisaggPDConfig& disagg_pd_config,
                            const KVCacheConfig& kv_cache_config,
                            const SchedulerConfig& scheduler_config) {
  EXPECT_EQ(disagg_pd_config.kv_cache_transfer_type(), "Mooncake");
  EXPECT_EQ(disagg_pd_config.kv_cache_transfer_mode(), "PUSH");
  EXPECT_FALSE(disagg_pd_config.enable_pd_ooc());
  EXPECT_EQ(kv_cache_config.kv_cache_dtype(), "auto");
  EXPECT_FALSE(scheduler_config.enable_schedule_overlap());
}

TEST(DisaggPDConfigTest, KeepsMluPrefixCacheForPrefillSideRoles) {
  const PrefixRoleCase cases[] = {
      {"PREFILL", true},
      {"MIX", true},
      {"DECODE", false},
      {"DEFAULT", false},
  };

  for (const PrefixRoleCase& test_case : cases) {
    DisaggPDConfig disagg_pd_config;
    KVCacheConfig kv_cache_config;
    SchedulerConfig scheduler_config;
    disagg_pd_config.instance_role(test_case.role);
    set_unsupported_values(disagg_pd_config, kv_cache_config, scheduler_config);

    disagg_pd_config.normalize_mlu(kv_cache_config, scheduler_config);

    SCOPED_TRACE(test_case.role);
    expect_forced_defaults(disagg_pd_config, kv_cache_config, scheduler_config);
    EXPECT_EQ(kv_cache_config.enable_prefix_cache(),
              test_case.keep_prefix_cache);
  }
}

}  // namespace
}  // namespace xllm
