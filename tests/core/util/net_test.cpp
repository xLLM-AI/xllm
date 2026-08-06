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

#include "core/util/net.h"

#include <gtest/gtest.h>

namespace xllm {
namespace net {
namespace {

TEST(NetTest, SelectsLoopbackForLoopbackRemote) {
  EXPECT_EQ(get_route_ip("127.0.0.1:19888"), "127.0.0.1");
}

TEST(NetTest, ResolvesRemoteHostname) {
  EXPECT_EQ(get_route_ip("localhost:19888"), "127.0.0.1");
}

TEST(NetTest, NormalizesLocalBindHosts) {
  const std::string local_ip = get_local_ip_addr();
  ASSERT_FALSE(local_ip.empty());

  EXPECT_EQ(extract_ip("0.0.0.0"), local_ip);
  EXPECT_EQ(extract_ip("127.0.0.1"), local_ip);
  EXPECT_EQ(extract_ip("localhost"), local_ip);
}

TEST(NetTest, ReturnsEmptyWhenRemoteHasNoUsableRoute) {
  EXPECT_TRUE(get_route_ip("255.255.255.255:19888").empty());
}

TEST(NetTest, ReturnsEmptyWhenRemoteCannotBeResolved) {
  EXPECT_TRUE(get_route_ip("worker.invalid:19888").empty());
}

}  // namespace
}  // namespace net
}  // namespace xllm
