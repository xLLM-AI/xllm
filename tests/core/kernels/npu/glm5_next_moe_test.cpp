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

#include <acl/acl.h>
#include <c10/core/impl/DeviceGuardImplInterface.h>
#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>

#include "core/kernels/xllm_torch_ops.h"

namespace py = pybind11;

namespace xllm {
namespace {

std::filesystem::path source_root() {
  std::filesystem::path root(__FILE__);
  for (int32_t depth = 0; depth < 5; ++depth) {
    root = root.parent_path();
  }
  return root;
}

py::dict load_cases() {
  const std::filesystem::path cases =
      std::filesystem::path(__FILE__).parent_path() / "glm5_next_moe_cases.py";
  return py::module_::import("runpy")
      .attr("run_path")(cases.string())
      .cast<py::dict>();
}

void run_python_case(const char* name) {
  py::gil_scoped_acquire gil;
  try {
    py::dict cases = load_cases();
    cases[name]();
  } catch (const py::error_already_set& error) {
    // Report Python assertions while their exception objects still have the
    // GIL available for cleanup. This does not turn a failure into a pass.
    ADD_FAILURE() << error.what();
  }
}

class Glm5NextMoeTest : public ::testing::Test {
 protected:
  void SetUp() override {
    ensure_xllm_torch_ops_registered();
    if (c10::impl::getDeviceGuardImpl(c10::DeviceType::PrivateUse1)
            ->deviceCount() == 0) {
      GTEST_SKIP() << "NPU is unavailable.";
    }
    // cc_test supplies tests/npu_test_environment.cpp, as for the existing
    // npu_xllm_ops_test. Do not substitute torch_npu ops or CPU pytest stubs.
    ASSERT_TRUE(Py_IsInitialized());
    py::gil_scoped_acquire gil;
    try {
      py::module_::import("sys").attr("path").attr("insert")(
          0, source_root().string());
      // npu_test_environment.cpp already initialized torch and torch_npu.
      // The full-model bootstrap also loads optional attention operator
      // libraries; do not mutate the initialized ACL vendor set in a local MoE
      // test.
      py::module_::import("xllm.python").attr("initialize_runtime")();
    } catch (const py::error_already_set& error) {
      FAIL() << error.what();
    }
  }
};

TEST_F(Glm5NextMoeTest, V2ClampedActivationMatchesIndependentReference) {
  run_python_case("check_v2_clamped_activation");
}

TEST_F(Glm5NextMoeTest, GroupedLocalExpertsMatchIndependentReference) {
  run_python_case("check_grouped_local_experts");
}

TEST_F(Glm5NextMoeTest, LocalShardGraphBuffersMatchEager) {
  run_python_case("check_local_shard_graph_buffers");
}

}  // namespace
}  // namespace xllm

int main(int argc, char** argv) {
  ::testing::InitGoogleTest(&argc, argv);
  const int32_t result = RUN_ALL_TESTS();
  if (!Py_IsInitialized()) {
    return result;
  }
  // npu_test_environment.cpp initialized ACL and restored the main-thread GIL.
  // Release GE resources while torch_npu's allocators are still alive, rather
  // than relying on the order of shared-library destructors at process exit.
  PyThreadState* state = PyEval_SaveThread();
  const aclError status = aclFinalize();
  PyEval_RestoreThread(state);
  if (status != ACL_SUCCESS) {
    std::fprintf(stderr,
                 "ACL test finalization failed: %d\n",
                 static_cast<int32_t>(status));
    return 1;
  }
  return result;
}
