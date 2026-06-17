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

#include "mm_data.h"

#include "core/util/tensor_helper.h"

namespace xllm {

MMData::MMData(uint32_t type, const MMItemVec& items)
    : type_(type), items_(std::move(items)) {}

MMData::MMData(uint32_t type, const MMDict& items)
    : type_(type), items_(std::move(items)) {}

bool MMData::has(const MMKey& key) const {
  if (!valid()) return false;

  if (hold<MMDict>()) {
    const auto& dict = items<MMDict>();
    const auto& itor = dict.find(key);

    return itor != dict.end();
  } else if (hold<MMItemVec>()) {
    const auto& vec = items<MMItemVec>();
    for (const auto& item : vec) {
      if (item.has(key)) {
        return true;
      }
    }
    return false;
  }
  return false;
}

size_t MMData::size() const {
  if (!valid()) return 0;

  if (std::holds_alternative<MMDict>(items_)) {
    return std::get<MMDict>(items_).size();
  } else if (std::holds_alternative<MMItemVec>(items_)) {
    return std::get<MMItemVec>(items_).size();
  }
  return 0;
}

bool MMData::add(MMType type, const MMDataItem& item) {
  if (!hold<MMItemVec>()) items_ = MMItemVec();

  auto& vec = items<MMItemVec>();

  type_ |= type;
  vec.emplace_back(item);

  return true;
}

MMDataItem& MMData::add(MMType type) {
  if (!hold<MMItemVec>()) items_ = MMItemVec();

  auto& vec = items<MMItemVec>();

  type_ |= type;
  vec.emplace_back(type);

  return vec.back();
}

void MMData::get(uint32_t type, MMItemVec& vec) const {
  if (!valid()) return;

  if (!hold<MMItemVec>()) return;

  const auto& data = items<MMItemVec>();

  for (const auto& item : data) {
    if (item.type() & type) {
      vec.emplace_back(item);
    }
  }
}

void MMData::get(const MMKey& key, std::vector<torch::Tensor>& vec) const {
  if (!valid()) return;

  if (hold<MMDict>()) {
    const auto& dict = items<MMDict>();
    const auto& itor = dict.find(key);

    if (itor == dict.end()) return;

    if (std::holds_alternative<torch::Tensor>(itor->second)) {
      vec.push_back(std::get<torch::Tensor>(itor->second));
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   itor->second)) {
      const auto& data = std::get<std::vector<torch::Tensor>>(itor->second);
      vec.insert(vec.end(), data.begin(), data.end());
    }
  } else if (hold<MMItemVec>()) {
    const auto& lst = items<MMItemVec>();
    for (const auto& item : lst) {
      item.get(key, vec);
    }
  }
}

bool MMData::foreach (MMDataItem::IVisitor& v) {
  if (!valid()) return false;

  if (!hold<MMItemVec>()) return false;

  auto& vec = items<MMItemVec>();
  for (auto& item : vec) {
    if (!v.visit(item)) {
      return false;
    }
  }
  return true;
}

bool MMData::foreach (MMDictItem::IVisitor& v) {
  if (!valid()) return false;

  if (!hold<MMDict>()) return false;

  auto& dict = items<MMDict>();
  for (auto& pair : dict) {
    if (!v.visit(pair.first, pair.second)) {
      return false;
    }
  }
  return true;
}

bool MMData::foreach (IItemVisitor& v) {
  if (!valid()) return false;

  if (hold<MMItemVec>()) {
    auto& vec = items<MMItemVec>();
    for (auto& item : vec) {
      if (!v.visit(item)) {
        return false;
      }
    }
  } else if (hold<MMDict>()) {
    auto& dict = items<MMDict>();
    for (auto& pair : dict) {
      if (!v.visit(pair.first, pair.second)) {
        return false;
      }
    }
  }
  return true;
}

void MMData::debug_print() const {
  LOG(INFO) << "mm data debug print, type:" << type_;

  if (hold<MMItemVec>()) {
    const auto& vec = items<MMItemVec>();
    for (const auto& item : vec) {
      item.debug_print();
    }
  } else if (hold<MMDict>()) {
    const auto& dict = items<MMDict>();
    for (const auto& pair : dict) {
      if (std::holds_alternative<torch::Tensor>(pair.second)) {
        torch::Tensor item = std::get<torch::Tensor>(pair.second);
        LOG(INFO) << " single tensor, key:" << pair.first
                  << " device:" << item.device() << " dtype:" << item.dtype()
                  << " shape:" << item.sizes();
      } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                     pair.second)) {
        const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);

        for (const auto& item : lst) {
          LOG(INFO) << " vector tensor, key:" << pair.first
                    << " device:" << item.device() << " dtype:" << item.dtype()
                    << " shape:" << item.sizes();
        }
      }
    }
  }
}

}  // namespace xllm
