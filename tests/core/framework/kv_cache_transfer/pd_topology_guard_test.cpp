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

#include "framework/kv_cache_transfer/pd_topology_guard.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>
#include <vector>

namespace xllm {

namespace {

void set_death_style() { GTEST_FLAG_SET(death_test_style, "threadsafe"); }

InstanceInfo make_info(int32_t dp_size,
                       const std::vector<uint64_t>& cluster_ids) {
  InstanceInfo info;
  info.dp_size = dp_size;
  info.cluster_ids = cluster_ids;
  return info;
}

TEST(PdTopologyGuardTest, HomoTopoBypass) {
  const InstanceInfo local_info = make_info(2, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(2, {0, 1, 2, 3});

  const PdTopo topo = get_pd_topo(local_info);
  EXPECT_EQ(topo.dp_size, 2);
  EXPECT_EQ(topo.tp_size, 2);

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PULL", false);
  EXPECT_EQ(result.status, PdTopoStatus::ALLOW_HOMO);
  EXPECT_TRUE(result.reason.empty());
}

TEST(PdTopologyGuardTest, RegistrationOmitsCpAndKeepsTopologyCompatible) {
  InstanceInfo prefill = make_info(1, {0, 1, 2, 3});
  prefill.name = "prefill";
  prefill.type = "PREFILL";
  prefill.kv_split_size = 1;
  InstanceInfo decode = make_info(1, {4, 5, 6, 7});
  decode.name = "decode";
  decode.type = "DECODE";
  decode.kv_split_size = 1;

  const nlohmann::json prefill_registration = prefill.serialize_to_json();
  const nlohmann::json decode_registration = decode.serialize_to_json();
  EXPECT_FALSE(prefill_registration.contains("cp_size"));
  EXPECT_FALSE(prefill_registration.contains("requested_cp_size"));
  EXPECT_FALSE(decode_registration.contains("cp_size"));
  EXPECT_FALSE(decode_registration.contains("requested_cp_size"));

  const PdTopoResult result = check_pd_topo(prefill, decode, "PUSH", true);
  EXPECT_EQ(result.status, PdTopoStatus::ALLOW_HOMO);
  EXPECT_TRUE(result.reason.empty());
}

TEST(PdTopologyGuardTest, TryGetPdTopoReturnTopo) {
  const InstanceInfo info = make_info(2, {0, 1, 2, 3});

  PdTopo topo;
  std::string reason;
  EXPECT_TRUE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(topo.dp_size, 2);
  EXPECT_EQ(topo.tp_size, 2);
  EXPECT_TRUE(reason.empty());
}

TEST(PdTopologyGuardTest, HeteroTopoNeedMla) {
  const InstanceInfo local_info = make_info(2, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(1, {0, 1, 2, 3});

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PUSH", false);
  EXPECT_EQ(result.status, PdTopoStatus::DENY_HETERO);
  EXPECT_EQ(result.reason, "hetero pd requires enable_mla=true");
}

TEST(PdTopologyGuardTest, HeteroTopoNeedPushKv) {
  const InstanceInfo local_info = make_info(2, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(1, {0, 1, 2, 3});

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PULL", true);
  EXPECT_EQ(result.status, PdTopoStatus::DENY_HETERO);
  EXPECT_EQ(result.reason, "hetero pd requires kv_mode=PUSH");
}

TEST(PdTopologyGuardTest, HeteroTopoAllowOnPushMla) {
  const InstanceInfo local_info = make_info(2, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(1, {0, 1, 2, 3});

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PUSH", true);
  EXPECT_EQ(result.status, PdTopoStatus::ALLOW_HETERO);
  EXPECT_TRUE(result.reason.empty());
}

TEST(PdTopologyGuardTest, CheckPdTopoRejectInvalidLocalTopo) {
  const InstanceInfo local_info = make_info(0, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(1, {0, 1, 2, 3});

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PUSH", true);
  EXPECT_EQ(result.status, PdTopoStatus::INVALID_LOCAL);
  EXPECT_EQ(result.reason,
            "invalid local pd topo: dp_size must be greater than 0");
}

TEST(PdTopologyGuardTest, CheckPdTopoRejectInvalidRemoteTopo) {
  const InstanceInfo local_info = make_info(1, {0, 1, 2, 3});
  const InstanceInfo remote_info = make_info(2, {0, 1, 2});

  const PdTopoResult result =
      check_pd_topo(local_info, remote_info, "PUSH", true);
  EXPECT_EQ(result.status, PdTopoStatus::INVALID_REMOTE);
  EXPECT_EQ(result.reason,
            "invalid remote pd topo: cluster_ids.size() must be divisible by "
            "dp_size");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectBadClusterSplit) {
  const InstanceInfo info = make_info(2, {0, 1, 2});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "cluster_ids.size() must be divisible by dp_size");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectEmptyClusterIds) {
  const InstanceInfo info = make_info(2, {});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "cluster_ids must not be empty");
}

TEST(PdTopologyGuardTest, TryGetPdTopoRejectZeroDpSize) {
  const InstanceInfo info = make_info(0, {0, 1, 2, 3});

  PdTopo topo;
  std::string reason;
  EXPECT_FALSE(try_get_pd_topo(info, &topo, &reason));
  EXPECT_EQ(reason, "dp_size must be greater than 0");
}

TEST(PdTopologyGuardTest, GetPdTopoRejectBadClusterSplit) {
  set_death_style();
  const InstanceInfo info = make_info(2, {0, 1, 2});

  EXPECT_DEATH(get_pd_topo(info),
               "cluster_ids.size\\(\\) must be divisible by dp_size");
}

TEST(PdTopologyGuardTest, GetPdTopoRejectEmptyClusterIds) {
  set_death_style();
  const InstanceInfo info = make_info(2, {});

  EXPECT_DEATH(get_pd_topo(info), "cluster_ids must not be empty");
}

}  // namespace

}  // namespace xllm
