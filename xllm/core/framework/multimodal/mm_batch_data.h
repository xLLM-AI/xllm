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

#pragma once

#include <torch/torch.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

#include "mm_data.h"
#include "mm_type.h"
#include "worker.pb.h"

namespace xllm {

class MMBatchData {
 public:
  MMBatchData() = default;
  MMBatchData(const std::vector<MMData>& datas);
  MMBatchData(uint32_t type, const MMDict& items);

  bool has(uint32_t type) const { return type & type_ != 0; }
  bool valid() const { return type_ != MMType::NONE; }

  uint32_t type() const { return type_; }
  const MMDict& data() const { return data_; }

  bool has(const MMKey& key) const;
  void get(const MMKey& key, std::vector<torch::Tensor>& vec) const;

  void to(const torch::Device& device);
  static MMBatchData to(const MMBatchData& mm_data,
                        const torch::Device& device);

  void batch(const std::vector<MMData>& mm_datas);

  template <typename T>
  std::optional<T> get(const MMKey& key) const {
    if (!valid()) return std::nullopt;

    const auto& itor = data_.find(key);
    if (itor != data_.end()) {
      return std::get<T>(itor->second);
    } else {
      return std::nullopt;
    }
  }

  void replace(const MMDict& data) { data_ = std::move(data); }
  const std::vector<MMData>& mm_data_vec() const { return mm_datas_; }

  bool foreach (MMData::IVisitor& v) {
    for (auto& item : mm_datas_) {
      if (!v.visit(item)) return false;
    }
    return true;
  }

  template <typename T>
  bool foreach (T& v) {
    for (auto& data : mm_datas_) {
      if (!data.foreach (v)) return false;
    }
    return true;
  }

  void debug_print() const;

 private:
  uint32_t type_ = MMType::NONE;
  MMDict data_;
  std::vector<MMData> mm_datas_;
};

bool mmdata_to_proto(const xllm::MMBatchData& cpp_mmdata,
                     proto::MMData* pb_mmdata);

bool proto_to_mmdata(const proto::MMData& pb_mmdata,
                     xllm::MMBatchData* cpp_mmdata);

}  // namespace xllm
