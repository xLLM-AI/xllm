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

#include <gtest/gtest.h>
#include <unistd.h>

#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

#include "core/platform/npu/vendor_paths.h"

namespace xllm {
namespace {

class NpuVendorPathsTest : public ::testing::Test {
 protected:
  void SetUp() override {
    root_ = std::filesystem::temp_directory_path() /
            ("xllm_vendor_paths_" + std::to_string(getpid()));
    bundle_ = root_ / "wheel/opp/vendors/custom_xllm_math";
    system_vendors_ = root_ / "toolkit/opp/vendors";
    add_vendor(bundle_);
    add_vendor(system_vendors_ / "custom_xllm_math");
    add_vendor(system_vendors_ / "glm_next_transformer");
    add_vendor(system_vendors_ / "custom_transformer");
  }

  void TearDown() override { std::filesystem::remove_all(root_); }

  static void add_vendor(const std::filesystem::path& path) {
    std::filesystem::create_directories(path / "op_api/lib");
    std::ofstream(path / "op_api/lib/libcust_opapi.so").put('x');
  }

  std::filesystem::path root_;
  std::filesystem::path bundle_;
  std::filesystem::path system_vendors_;
};

TEST_F(NpuVendorPathsTest, PythonPreservesTransformerPriority) {
  const std::vector<std::string> expected{
      (system_vendors_ / "glm_next_transformer").string(),
      (system_vendors_ / "custom_transformer").string(),
      bundle_.string()};
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_,
                                     system_vendors_,
                                     {},
                                     /*is_python=*/true),
            expected);
}

TEST_F(NpuVendorPathsTest, NativePrefersBundledMathAndRetainsExternalVendors) {
  const std::string external = (root_ / "extra").string();
  const std::vector<std::string> existing{
      external,
      bundle_.string() + "/",
      (system_vendors_ / "custom_xllm_math").string() + "/",
      external};
  const std::vector<std::string> expected{
      bundle_.string(),
      (system_vendors_ / "custom_transformer").string(),
      (system_vendors_ / "glm_next_transformer").string(),
      external};
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_,
                                     system_vendors_,
                                     existing,
                                     /*is_python=*/false),
            expected);
}

TEST_F(NpuVendorPathsTest, ExplicitTransformerPathPrecedesSystemCopy) {
  const std::filesystem::path external = root_ / "extra/glm_next_transformer";
  add_vendor(external);
  const std::vector<std::string> expected{
      external.string(),
      (system_vendors_ / "custom_transformer").string(),
      bundle_.string()};
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_,
                                     system_vendors_,
                                     {external.string()},
                                     /*is_python=*/true),
            expected);
}

TEST_F(NpuVendorPathsTest, MissingSystemVendorsDoNotHideBundledOps) {
  std::filesystem::remove_all(system_vendors_);
  const std::vector<std::string> expected{bundle_.string()};
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_,
                                     system_vendors_,
                                     {},
                                     /*is_python=*/true),
            expected);
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_, {}, {}, /*is_python=*/false),
            expected);
}

TEST_F(NpuVendorPathsTest, RepeatedRegistrationIsIdempotent) {
  const std::vector<std::string> first = bundled_npu_vendor_paths(
      bundle_, system_vendors_, {"", "/extra"}, /*is_python=*/true);
  EXPECT_EQ(bundled_npu_vendor_paths(bundle_,
                                     system_vendors_,
                                     first,
                                     /*is_python=*/true),
            first);
}

}  // namespace
}  // namespace xllm
