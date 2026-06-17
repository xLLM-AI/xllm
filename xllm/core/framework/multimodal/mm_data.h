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

#include "core/util/tensor_helper.h"
#include "mm_data_item.h"
#include "mm_type.h"

namespace xllm {

namespace MMDictItem {
class IVisitor {
 public:
  virtual ~IVisitor() = default;
  virtual bool visit(const MMKey& key, MMValue& value) = 0;
};
}  // namespace MMDictItem

using MMItemVec = std::vector<MMDataItem>;

class MMData {
 public:
  class IItemVisitor : public MMDataItem::IVisitor,
                       public MMDictItem::IVisitor {
   public:
    virtual ~IItemVisitor() = default;

    using MMDataItem::IVisitor::visit;
    using MMDictItem::IVisitor::visit;
  };

  class IVisitor {
   public:
    virtual ~IVisitor() = default;
    virtual bool visit(MMData& data) = 0;
  };

  using MMItems = std::variant<MMItemVec, MMDict>;

 public:
  MMData() = default;
  MMData(uint32_t type, const MMItemVec& items);
  MMData(uint32_t type, const MMDict& items);

  bool has(uint32_t type) const { return type & type_ != 0; }
  bool has(MMType type) const { return type & type_ != 0; }

  bool has(const MMKey& key) const;
  bool valid() const { return type_ != MMType::NONE; }
  bool empty() const { return size() == 0; }

  uint32_t type() const { return type_; }
  size_t size() const;

  bool add(MMType type, const MMDataItem& item);
  MMDataItem& add(MMType type);

  void get(uint32_t type, std::vector<MMDataItem>& items) const;
  void get(const MMKey& key, std::vector<torch::Tensor>& items) const;

  template <typename T>
  bool add(MMType type, const MMKey& key, const T& value) {
    if (!hold<MMDict>()) items_ = MMDict();

    auto& dict = items<MMDict>();

    const auto& itor = dict.find(key);
    if (itor != dict.end()) return false;

    type_ |= type;
    dict.insert({key, value});
    return true;
  }

  template <typename T>
  std::optional<T> get(const MMKey& key) const {
    if (!valid()) return std::nullopt;

    if (hold<MMDict>()) {
      auto& dict = items<MMDict>();

      const auto& itor = dict.find(key);
      if (itor != dict.end()) {
        return std::get<T>(itor->second);
      } else {
        return std::nullopt;
      }
    } else if (hold<MMItemVec>()) {
      auto& vec = items<MMItemVec>();

      std::vector<torch::Tensor> lst;
      for (const auto& item : vec) {
        item.get(key, lst);
      }

      if (!lst.size()) {
        return std::nullopt;
      }

      torch::Tensor ts;
      bool res = safe_concat(lst, ts);

      MMValue value;
      if (res && std::is_same_v<T, std::vector<torch::Tensor>>) {
        value = {ts};
      } else if (res) {
        value = ts;
      } else {
        value = lst;
      }

      return std::get<T>(value);
    }
    return std::nullopt;
  }

  template <typename T>
  void set(uint32_t type, const T& item) {
    type_ = type;
    items_ = item;
  }

  template <typename T>
  const T& items() const {
    return std::get<T>(items_);
  }

  template <typename T>
  T& items() {
    return std::get<T>(items_);
  }

  template <typename T>
  bool hold() const {
    if (std::holds_alternative<T>(items_)) {
      return true;
    } else {
      return false;
    }
  }

  template <typename T>
  void get_metadata(MMType type, std::vector<T>& metadatas) const {
    if (!valid()) return;

    if (!hold<MMItemVec>()) return;

    const auto& item_vec = items<MMItemVec>();
    for (const auto& item : item_vec) {
      if (item.type() != type) continue;

      if (auto res = item.template get_metadata<T>()) {
        metadatas.push_back(res.value());
      }
    }
  }

  bool foreach (MMDataItem::IVisitor& v);
  bool foreach (MMDictItem::IVisitor& v);

  bool foreach (IItemVisitor& v);
  void debug_print() const;

 private:
  uint32_t type_ = MMType::NONE;
  MMItems items_;
};

}  // namespace xllm
