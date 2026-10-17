/* Copyright 2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "api_service/control_service_impl.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <string>
#include <system_error>
#include <vector>

#include "core/common/options.h"

namespace xllm {

class ControlServiceImplTest : public testing::Test {
 protected:
  static Status parse_fork_master_request(const proto::MasterInfos& request,
                                          Options& options) {
    return ControlServiceImpl::parse_fork_master_request(request, options);
  }

  static WakeupOptions parse_wakeup_options(const proto::MasterInfos& request) {
    return ControlServiceImpl::parse_wakeup_options(request);
  }

  void SetUp() override {
    std::string directory_template =
        (std::filesystem::temp_directory_path() / "xllm-control-request-XXXXXX")
            .string();
    const char* directory = mkdtemp(directory_template.data());
    ASSERT_NE(directory, nullptr);
    temp_directory_ = directory;
    model_path_ = temp_directory_ / "model";
    ASSERT_TRUE(std::filesystem::create_directory(model_path_));
    ASSERT_TRUE(std::filesystem::create_directory(temp_directory_ / "sibling"));
  }

  void TearDown() override {
    if (temp_directory_.empty()) {
      return;
    }
    for (const auto& path : {temp_directory_ / "model",
                             temp_directory_ / "sibling",
                             temp_directory_}) {
      std::error_code error;
      std::filesystem::remove(path, error);
      EXPECT_FALSE(error) << path << ": " << error.message();
    }
  }

  proto::MasterInfos fork_request() const {
    proto::MasterInfos request;
    request.set_model_path(model_path_.string());
    return request;
  }

  std::filesystem::path temp_directory_;
  std::filesystem::path model_path_;
};

TEST_F(ControlServiceImplTest,
       NormalizesModelIdWithoutChangingRequestedModelPath) {
  const std::string model_path = model_path_.string();
  const std::vector<std::string> paths = {
      model_path,
      model_path + "/",
      model_path + "//",
      model_path + "/.",
      model_path + "/./",
      (temp_directory_ / "sibling" / ".." / "model").string(),
      model_path + "/../model",
      model_path + "/../model/."};

  for (const std::string& path : paths) {
    SCOPED_TRACE(path);
    proto::MasterInfos request;
    request.set_model_path(path);
    Options options;
    const Status status = parse_fork_master_request(request, options);
    ASSERT_TRUE(status.ok()) << status.message();
    EXPECT_EQ(options.model_id(), "model");
    EXPECT_EQ(options.model_path(), path);
  }
}

TEST_F(ControlServiceImplTest, CopiesForkStatusAndMasterNodeAddress) {
  proto::MasterInfos request = fork_request();
  request.set_model_id("unused-request-alias");
  request.set_master_node_addr("127.0.0.1:9000");
  request.set_master_status(proto::MasterStatus::DEEP_SLEEP);
  Options options;

  ASSERT_TRUE(parse_fork_master_request(request, options).ok());
  EXPECT_EQ(options.model_id(), "model");
  ASSERT_TRUE(options.master_node_addr().has_value());
  EXPECT_EQ(*options.master_node_addr(), "127.0.0.1:9000");
  EXPECT_EQ(options.master_status(), MasterStatus::DEEP_SLEEP);
}

TEST_F(ControlServiceImplTest, UsesDefaultParallelSizesForOmittedFields) {
  const proto::MasterInfos request = fork_request();
  Options options;

  ASSERT_TRUE(parse_fork_master_request(request, options).ok());
  EXPECT_EQ(options.nnodes(), 1);
  EXPECT_EQ(options.dp_size(), 1);
}

TEST_F(ControlServiceImplTest, AppliesPositiveParallelOverrides) {
  proto::MasterInfos request = fork_request();
  request.set_nnodes(8);
  request.set_dp_size(2);
  Options options;

  ASSERT_TRUE(parse_fork_master_request(request, options).ok());
  EXPECT_EQ(options.nnodes(), 8);
  EXPECT_EQ(options.dp_size(), 2);
}

TEST_F(ControlServiceImplTest, AppliesParallelOverridesIndependently) {
  proto::MasterInfos request = fork_request();
  request.set_nnodes(8);
  Options node_options;
  ASSERT_TRUE(parse_fork_master_request(request, node_options).ok());
  EXPECT_EQ(node_options.nnodes(), 8);
  EXPECT_EQ(node_options.dp_size(), 1);

  request.set_nnodes(0);
  request.set_dp_size(2);
  Options data_parallel_options;
  ASSERT_TRUE(parse_fork_master_request(request, data_parallel_options).ok());
  EXPECT_EQ(data_parallel_options.nnodes(), 1);
  EXPECT_EQ(data_parallel_options.dp_size(), 2);
}

TEST_F(ControlServiceImplTest, PreservesParallelOptionsForNonpositiveFields) {
  for (const int32_t requested_size : {0, -1}) {
    SCOPED_TRACE(requested_size);
    proto::MasterInfos request = fork_request();
    request.set_nnodes(requested_size);
    request.set_dp_size(requested_size);
    Options options;
    options.nnodes() = 8;
    options.dp_size() = 2;

    ASSERT_TRUE(parse_fork_master_request(request, options).ok());
    EXPECT_EQ(options.nnodes(), 8);
    EXPECT_EQ(options.dp_size(), 2);
  }
}

TEST_F(ControlServiceImplTest, RejectsEmptyModelPath) {
  proto::MasterInfos request;
  Options options;

  const Status status = parse_fork_master_request(request, options);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "Failed to parse fork master request");
}

TEST_F(ControlServiceImplTest,
       RejectsMissingModelPathWithoutChangingExistingOptions) {
  proto::MasterInfos request = fork_request();
  request.set_model_path((temp_directory_ / "missing-model").string());
  request.set_master_node_addr("new-address");
  request.set_master_status(proto::MasterStatus::DEEP_SLEEP);
  request.set_nnodes(8);
  request.set_dp_size(2);
  Options options;
  options.model_id() = "existing-model";
  options.model_path() = "existing-path";
  options.master_node_addr() = "existing-address";

  const Status status = parse_fork_master_request(request, options);
  EXPECT_FALSE(status.ok());
  EXPECT_EQ(status.code(), StatusCode::INVALID_ARGUMENT);
  EXPECT_EQ(status.message(), "Failed to parse fork master request");
  EXPECT_EQ(options.model_id(), "existing-model");
  EXPECT_EQ(options.model_path(), "existing-path");
  ASSERT_TRUE(options.master_node_addr().has_value());
  EXPECT_EQ(*options.master_node_addr(), "existing-address");
  EXPECT_EQ(options.master_status(), MasterStatus::WAKEUP);
  EXPECT_EQ(options.nnodes(), 1);
  EXPECT_EQ(options.dp_size(), 1);
}

TEST_F(ControlServiceImplTest, EmptyRequestUsesLocalWakeupDefaults) {
  const proto::MasterInfos request;

  const WakeupOptions options = parse_wakeup_options(request);
  EXPECT_EQ(options.master_status, MasterStatus::WAKEUP);
  EXPECT_TRUE(options.remote_addrs.empty());
  EXPECT_TRUE(options.src_weight_segments.empty());
}

TEST_F(ControlServiceImplTest, PreservesRemoteAddressesWithoutSegments) {
  proto::MasterInfos request;
  request.add_remote_addrs("second:9000");
  request.add_remote_addrs("first:9000");

  const WakeupOptions options = parse_wakeup_options(request);
  EXPECT_EQ(options.remote_addrs,
            (std::vector<std::string>{"second:9000", "first:9000"}));
  EXPECT_TRUE(options.src_weight_segments.empty());
}

TEST_F(ControlServiceImplTest, PreservesRemoteAndWeightSegmentOrder) {
  proto::MasterInfos request;
  request.add_remote_addrs("second:9000");
  request.add_remote_addrs("first:9000");
  request.add_remote_addrs("empty:9000");
  auto* first_list = request.add_src_weight_segments();
  auto* first_segment = first_list->add_segments();
  first_segment->set_offset(256);
  first_segment->set_size(128);
  auto* second_segment = first_list->add_segments();
  second_segment->set_offset(0);
  second_segment->set_size(64);
  auto* second_list = request.add_src_weight_segments();
  auto* large_segment = second_list->add_segments();
  constexpr uint64_t kLargeOffset = (uint64_t{1} << 40) + 512;
  constexpr uint64_t kLargeSize = (uint64_t{1} << 32) + 3;
  large_segment->set_offset(kLargeOffset);
  large_segment->set_size(kLargeSize);
  request.add_src_weight_segments();

  const WakeupOptions options = parse_wakeup_options(request);
  EXPECT_EQ(
      options.remote_addrs,
      (std::vector<std::string>{"second:9000", "first:9000", "empty:9000"}));
  ASSERT_EQ(options.src_weight_segments.size(), 3);
  ASSERT_EQ(options.src_weight_segments[0].size(), 2);
  EXPECT_EQ(options.src_weight_segments[0][0].offset, 256);
  EXPECT_EQ(options.src_weight_segments[0][0].size, 128);
  EXPECT_EQ(options.src_weight_segments[0][1].offset, 0);
  EXPECT_EQ(options.src_weight_segments[0][1].size, 64);
  ASSERT_EQ(options.src_weight_segments[1].size(), 1);
  EXPECT_EQ(options.src_weight_segments[1][0].offset, kLargeOffset);
  EXPECT_EQ(options.src_weight_segments[1][0].size, kLargeSize);
  EXPECT_TRUE(options.src_weight_segments[2].empty());
}

TEST_F(ControlServiceImplTest, IgnoresSourceSegmentsWithoutRemoteAddresses) {
  proto::MasterInfos request;
  auto* segment = request.add_src_weight_segments()->add_segments();
  segment->set_offset(256);
  segment->set_size(128);

  const WakeupOptions options = parse_wakeup_options(request);
  EXPECT_TRUE(options.remote_addrs.empty());
  EXPECT_TRUE(options.src_weight_segments.empty());
}

}  // namespace xllm
