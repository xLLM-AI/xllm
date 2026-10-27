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

#pragma once

#include <gtest/gtest.h>

#include <cstdint>

#include "core/framework/config/parallel_config.h"

namespace xllm::test {

class DcpIndexerConfigTest : public ::testing::TestWithParam<int32_t> {
 protected:
  void SetUp() override {
    ParallelConfig::get_instance().kv_split_size(GetParam()).cp_size(4);
  }

  void TearDown() override { ParallelConfig::get_instance() = saved_config_; }

  int64_t expected_factor() const {
#if defined(USE_NPU)
    return GetParam() == 0 ? 4 : GetParam();
#else
    return 1;
#endif
  }

 private:
  ParallelConfig saved_config_ = ParallelConfig::get_instance();
};

}  // namespace xllm::test
