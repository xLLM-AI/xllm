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

#include "core/util/tensor_helper.h"
#include "mm_data.h"

namespace xllm {

MMDataItem::MMDataItem(MMType type) : type_(type) {}

MMDataItem::MMDataItem(MMType type, const MMDict& data)
    : type_(type), data_(std::move(data)) {}

MMDataItem::MMDataItem(MMType type,
                       const MMDict& data,
                       const MMMetadata& metadata)
    : type_(type), data_(std::move(data)), metadata_(std::move(metadata)) {}

bool MMDataItem::has(const MMKey& key) const {
  if (!valid()) return false;

  const auto& itor = data_.find(key);
  return itor != data_.end();
}

void MMDataItem::get(const MMKey& key, std::vector<torch::Tensor>& vec) const {
  if (!valid()) return;

  const auto& itor = data_.find(key);
  if (itor == data_.end()) return;

  if (std::holds_alternative<torch::Tensor>(itor->second)) {
    vec.push_back(std::get<torch::Tensor>(itor->second));
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(itor->second)) {
    const auto& data = std::get<std::vector<torch::Tensor>>(itor->second);
    vec.insert(vec.end(), data.begin(), data.end());
  }
}

void MMDataItem::debug_print() const {
  const std::string type_str = type_.to_string().value_or("none");
  LOG(INFO) << "MMDataItem"
            << " type=" << type_str << " seq_index=" << state_.seq_index()
            << " data_count=" << data_.size();
  LOG(INFO) << "  token_pos"
            << " offset=" << state_.token_pos().offset
            << " length=" << state_.token_pos().length;
  LOG(INFO) << "  schedule"
            << " start_pos=" << state_.schedule_data().start_pos
            << " end_pos=" << state_.schedule_data().end_pos;

  if (data_.empty()) {
    LOG(INFO) << "  data: empty";
    return;
  }

  for (const auto& pair : data_) {
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      const torch::Tensor& item = std::get<torch::Tensor>(pair.second);
      LOG(INFO) << "  tensor"
                << " key=" << pair.first << " device=" << item.device()
                << " dtype=" << item.dtype() << " shape=" << item.sizes();
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);
      LOG(INFO) << "  tensor_list"
                << " key=" << pair.first << " size=" << lst.size();
      for (size_t i = 0; i < lst.size(); ++i) {
        const torch::Tensor& item = lst[i];
        LOG(INFO) << "    item[" << i << "]"
                  << " device=" << item.device() << " dtype=" << item.dtype()
                  << " shape=" << item.sizes();
      }
    }
  }
}

}  // namespace xllm
