/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "mapping_npu.h"

#include <glog/logging.h>
#include <gtest/gtest.h>

#include <cstdlib>
#include <optional>
#include <string>

namespace xllm {
namespace {

class ScopedEnvVar final {
 public:
  explicit ScopedEnvVar(const std::string& name) : name_(name) {
    const char* old_value = std::getenv(name_.c_str());
    if (old_value != nullptr) {
      old_value_ = old_value;
    }
  }

  ~ScopedEnvVar() {
    if (old_value_.has_value()) {
      setenv(name_.c_str(), old_value_.value().c_str(), 1);
    } else {
      unsetenv(name_.c_str());
    }
  }

  void unset() { unsetenv(name_.c_str()); }

  void set(const std::string& value) {
    setenv(name_.c_str(), value.c_str(), 1);
  }

 private:
  std::string name_;
  std::optional<std::string> old_value_;
};

MappingNPU::Options get_mapping_options() {
  MappingNPU::Options options;
  options.dp_size(2)
      .tp_size(8)
      .moe_tp_size(2)
      .moe_ep_size(8)
      .pp_size(1)
      .sp_size(1);
  return options;
}

}  // namespace

TEST(TestMappingNPU, ToJson) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.unset();

  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 6, options);
  nlohmann::json data = mapping.to_json();
  LOG(INFO) << "Mapping INFO:\n" << data.dump(2);
  nlohmann::json attn_dp = data["attnDp"];
  int32_t attn_dp_group_id = attn_dp["groupId"];
  EXPECT_EQ(attn_dp_group_id, 6);
  nlohmann::json attn_tp = data["attnTp"];
  int32_t attn_tp_group_id = attn_tp["groupId"];
  EXPECT_EQ(attn_tp_group_id, 0);
  nlohmann::json mlp_tp = data["mlpTp"];
  int32_t mlp_tp_buffer_size = mlp_tp["bufferSize"];
  EXPECT_EQ(mlp_tp_buffer_size, 512);
}

TEST(TestMappingNPU, ToJsonUsesHcclBuffsizeEnv) {
  ScopedEnvVar hccl_buffsize("HCCL_BUFFSIZE");
  hccl_buffsize.set("128");

  std::string rank_table_file;
  MappingNPU::Options options = get_mapping_options();
  MappingNPU mapping(rank_table_file, 16, 6, options);
  nlohmann::json data = mapping.to_json();
  nlohmann::json mlp_tp = data["mlpTp"];
  int32_t mlp_tp_buffer_size = mlp_tp["bufferSize"];
  EXPECT_EQ(mlp_tp_buffer_size, 128);
}

}  // namespace xllm
