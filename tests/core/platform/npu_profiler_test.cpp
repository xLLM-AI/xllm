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

#include "core/platform/npu/npu_profiler.h"

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <future>
#include <set>
#include <string>
#include <vector>

// Replace only the CANN profiling boundary. No NPU work is needed to exercise
// process/device ownership, concurrent worker calls, or injected API failures.
struct aclprofConfig {
  uint32_t device_id;
};

namespace {

constexpr aclError kFailure = 1;

struct FakeProfiler {
  int32_t init_calls = 0;
  int32_t finalize_calls = 0;
  int32_t start_calls = 0;
  int32_t stop_calls = 0;
  int32_t destroy_calls = 0;
  bool fail_init = false;
  bool fail_create = false;
  bool fail_start = false;
  bool fail_stop = false;
  bool fail_destroy = false;
  bool fail_finalize = false;
  std::set<uint32_t> active_devices;
  std::vector<std::string> paths;
};

FakeProfiler fake_profiler;

class NpuProfilerTest : public ::testing::Test {
 protected:
  void SetUp() override {
    fake_profiler = {};
    output_dir_ = std::filesystem::temp_directory_path() /
                  ("xllm_profiler_test_" + std::to_string(getpid()));
    std::filesystem::create_directories(output_dir_);
  }

  void TearDown() override {
    fake_profiler.fail_stop = false;
    fake_profiler.fail_destroy = false;
    fake_profiler.fail_finalize = false;
    EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
    EXPECT_TRUE(profiler_.stop(/*device_id=*/1));
    EXPECT_TRUE(fake_profiler.active_devices.empty());
    std::filesystem::remove_all(output_dir_);
  }

  xllm::NpuProfiler& profiler_ = xllm::NpuProfiler::get_instance();
  std::filesystem::path output_dir_;
};

}  // namespace

extern "C" {

aclError aclprofInit(const char* path, size_t length) {
  ++fake_profiler.init_calls;
  fake_profiler.paths.emplace_back(path, length);
  return fake_profiler.fail_init ? kFailure : ACL_SUCCESS;
}

aclprofConfig* aclprofCreateConfig(uint32_t* devices,
                                   uint32_t device_count,
                                   aclprofAicoreMetrics metrics,
                                   const aclprofAicoreEvents* events,
                                   uint64_t activities) {
  EXPECT_EQ(device_count, 1U);
  EXPECT_EQ(metrics, ACL_AICORE_NONE);
  EXPECT_EQ(events, nullptr);
  EXPECT_EQ(activities,
            ACL_PROF_ACL_API | ACL_PROF_TASK_TIME | ACL_PROF_HCCL_TRACE);
  if (fake_profiler.fail_create) {
    return nullptr;
  }
  return new aclprofConfig{devices[0]};
}

aclError aclprofStart(const aclprofConfig* config) {
  ++fake_profiler.start_calls;
  if (fake_profiler.fail_start) {
    return kFailure;
  }
  EXPECT_TRUE(fake_profiler.active_devices.insert(config->device_id).second);
  return ACL_SUCCESS;
}

aclError aclprofStop(const aclprofConfig* config) {
  ++fake_profiler.stop_calls;
  if (fake_profiler.fail_stop) {
    return kFailure;
  }
  EXPECT_EQ(fake_profiler.active_devices.erase(config->device_id), 1U);
  return ACL_SUCCESS;
}

aclError aclprofDestroyConfig(const aclprofConfig* config) {
  ++fake_profiler.destroy_calls;
  EXPECT_FALSE(fake_profiler.active_devices.contains(config->device_id));
  if (fake_profiler.fail_destroy) {
    return kFailure;
  }
  delete config;
  return ACL_SUCCESS;
}

aclError aclprofFinalize() {
  ++fake_profiler.finalize_calls;
  EXPECT_TRUE(fake_profiler.active_devices.empty());
  return fake_profiler.fail_finalize ? kFailure : ACL_SUCCESS;
}

}  // extern "C"

TEST_F(NpuProfilerTest, DuplicateCallsAndRepeatedWindows) {
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  for (int32_t window = 0; window < 2; ++window) {
    ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
    ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
    ASSERT_TRUE(profiler_.stop(/*device_id=*/0));
    ASSERT_TRUE(profiler_.stop(/*device_id=*/0));
  }
  EXPECT_EQ(fake_profiler.init_calls, 2);
  EXPECT_EQ(fake_profiler.start_calls, 2);
  EXPECT_EQ(fake_profiler.stop_calls, 2);
  EXPECT_EQ(fake_profiler.destroy_calls, 2);
  EXPECT_EQ(fake_profiler.finalize_calls, 2);
  ASSERT_EQ(fake_profiler.paths.size(), 2U);
  EXPECT_NE(fake_profiler.paths[0], fake_profiler.paths[1]);
}

TEST_F(NpuProfilerTest, FinalizeOnlyAfterLastDeviceStops) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/1));
  ASSERT_TRUE(profiler_.stop(/*device_id=*/0));
  ASSERT_TRUE(profiler_.stop(/*device_id=*/0));
  EXPECT_EQ(fake_profiler.finalize_calls, 0);
  EXPECT_TRUE(fake_profiler.active_devices.contains(1));
  ASSERT_TRUE(profiler_.stop(/*device_id=*/1));
  EXPECT_EQ(fake_profiler.init_calls, 1);
  EXPECT_EQ(fake_profiler.finalize_calls, 1);
}

TEST_F(NpuProfilerTest, ConcurrentWorkersShareInitialization) {
  auto first = std::async(std::launch::async, [this]() {
    return profiler_.start(output_dir_, /*device_id=*/0);
  });
  auto second = std::async(std::launch::async, [this]() {
    return profiler_.start(output_dir_, /*device_id=*/1);
  });
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  EXPECT_EQ(fake_profiler.init_calls, 1);
  first = std::async(std::launch::async,
                     [this]() { return profiler_.stop(/*device_id=*/0); });
  second = std::async(std::launch::async,
                      [this]() { return profiler_.stop(/*device_id=*/1); });
  EXPECT_TRUE(first.get());
  EXPECT_TRUE(second.get());
  EXPECT_EQ(fake_profiler.finalize_calls, 1);
}

TEST_F(NpuProfilerTest, InvalidDestinationDoesNotInitialize) {
  const auto file = output_dir_ / "file";
  std::ofstream(file).put('x');
  EXPECT_FALSE(profiler_.start(file / "trace", /*device_id=*/0));
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/-1));
  EXPECT_EQ(fake_profiler.init_calls, 0);
}

TEST_F(NpuProfilerTest, InitFailureDoesNotFinalizeUnownedProfiler) {
  fake_profiler.fail_init = true;
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0));
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  EXPECT_EQ(fake_profiler.finalize_calls, 0);
  fake_profiler.fail_init = false;
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
}

TEST_F(NpuProfilerTest, CreateFailureFinalizesInitializedSession) {
  fake_profiler.fail_create = true;
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0));
  EXPECT_EQ(fake_profiler.finalize_calls, 1);
  fake_profiler.fail_create = false;
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
}

TEST_F(NpuProfilerTest, StartFailureDestroysConfigAndAllowsRetry) {
  fake_profiler.fail_start = true;
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0));
  EXPECT_EQ(fake_profiler.destroy_calls, 1);
  EXPECT_EQ(fake_profiler.finalize_calls, 1);
  fake_profiler.fail_start = false;
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
}

TEST_F(NpuProfilerTest, OtherDeviceFailureKeepsActiveCaptureOwned) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
  fake_profiler.fail_start = true;
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/1));
  EXPECT_EQ(fake_profiler.finalize_calls, 0);
  EXPECT_TRUE(fake_profiler.active_devices.contains(0));
}

TEST_F(NpuProfilerTest, StopFailureRetainsConfigForRetry) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
  fake_profiler.fail_stop = true;
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  EXPECT_EQ(fake_profiler.destroy_calls, 0);
  EXPECT_EQ(fake_profiler.finalize_calls, 0);
  fake_profiler.fail_stop = false;
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
}

TEST_F(NpuProfilerTest, FinalizeFailureBlocksNewWindowUntilCleanedUp) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
  fake_profiler.fail_finalize = true;
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/1));
  fake_profiler.fail_finalize = false;
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  EXPECT_TRUE(profiler_.start(output_dir_, /*device_id=*/1));
}

TEST_F(NpuProfilerTest, DestroyFailureDoesNotStopDeviceTwice) {
  ASSERT_TRUE(profiler_.start(output_dir_, /*device_id=*/0));
  fake_profiler.fail_destroy = true;
  EXPECT_FALSE(profiler_.stop(/*device_id=*/0));
  EXPECT_FALSE(profiler_.start(output_dir_, /*device_id=*/0));
  EXPECT_EQ(fake_profiler.finalize_calls, 0);
  fake_profiler.fail_destroy = false;
  EXPECT_TRUE(profiler_.stop(/*device_id=*/0));
  EXPECT_EQ(fake_profiler.stop_calls, 1);
  EXPECT_EQ(fake_profiler.finalize_calls, 1);
}
