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

#include "mm_batch_data.h"

#include <cstring>

#include "core/util/tensor_helper.h"
#include "core/util/utils.h"
#include "mm_visitor.h"

namespace xllm {

MMBatchData::MMBatchData(const std::vector<MMData>& datas) {
  this->batch(datas);
}

MMBatchData::MMBatchData(uint32_t type, const MMDict& items)
    : type_(type), data_(std::move(items)) {}

bool MMBatchData::has(const MMKey& key) const {
  if (!valid()) return false;

  const auto& itor = data_.find(key);
  return itor != data_.end();
}

void MMBatchData::get(const MMKey& key, std::vector<torch::Tensor>& vec) const {
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

void MMBatchData::to(const torch::Device& device) {
  MMDict dict;

  for (const auto& pair : data_) {
    if (std::holds_alternative<torch::Tensor>(pair.second)) {
      dict[pair.first] =
          safe_to(std::get<torch::Tensor>(pair.second), device, true);
    } else if (std::holds_alternative<std::vector<torch::Tensor>>(
                   pair.second)) {
      const auto& lst = std::get<std::vector<torch::Tensor>>(pair.second);

      std::vector<torch::Tensor> vec;
      vec.reserve(lst.size());

      for (const auto& item : lst) {
        vec.emplace_back(safe_to(item, device, true));
      }
      dict[pair.first] = std::move(vec);
    }
  }

  data_ = std::move(dict);
}

MMBatchData MMBatchData::to(const MMBatchData& mm_data,
                            const torch::Device& device) {
  MMBatchData new_mm_data = mm_data;
  new_mm_data.to(device);
  return new_mm_data;
}

void MMBatchData::batch(const std::vector<MMData>& mm_datas) {
  mm_datas_ = std::move(mm_datas);
  CollectMMDataTensorVisitor visitor;
  this->foreach (static_cast<MMData::IVisitor&>(visitor));

  MMDict dict;
  for (const auto& pair : visitor.datas_) {
    torch::Tensor tar;
    if (safe_concat(pair.second, tar)) {
      dict[pair.first] = tar;
    } else {
      dict[pair.first] = std::move(pair.second);
    }
  }

  type_ = visitor.type_;
  data_ = std::move(dict);
}

void MMBatchData::debug_print() const {
  LOG(INFO) << "mm batch data debug print, type:" << type_;
  LOG(INFO) << "=============== mm batch vec data ================";
  LOG(INFO) << "mm batch data vec count:" << mm_datas_.size();
  for (const auto& mm_data : mm_datas_) {
    mm_data.debug_print();
  }
  LOG(INFO) << "=============== mm batch data dict data ================";
  for (const auto& pair : data_) {
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

namespace {
bool mmvalue_to_proto(const xllm::MMValue& cpp_value,
                      proto::MMValue* pb_value) {
  if (!pb_value) {
    LOG(ERROR) << "PB MMValue pointer is null";
    return false;
  }

  if (std::holds_alternative<torch::Tensor>(cpp_value)) {
    auto& torch_tensor = std::get<torch::Tensor>(cpp_value);
    proto::Tensor* pb_tensor = pb_value->mutable_single_tensor();
    if (!util::torch_to_proto(torch_tensor, pb_tensor)) {
      LOG(ERROR) << "Failed to convert torch Tensor to PB Tensor";
      return false;
    }
  } else if (std::holds_alternative<std::vector<torch::Tensor>>(cpp_value)) {
    auto& torch_tensor_vec = std::get<std::vector<torch::Tensor>>(cpp_value);
    proto::TensorList* pb_tensor_list = pb_value->mutable_tensor_list();
    pb_tensor_list->mutable_tensors()->Reserve(torch_tensor_vec.size());
    for (const auto& torch_tensor : torch_tensor_vec) {
      proto::Tensor* pb_tensor = pb_tensor_list->add_tensors();
      if (!util::torch_to_proto(torch_tensor, pb_tensor)) {
        LOG(ERROR) << "Failed to convert torch Tensor to PB Tensor (list item)";
        return false;
      }
    }
  } else {
    LOG(ERROR) << "Unsupported struct MMValue type";
    return false;
  }

  return true;
}

std::optional<xllm::MMValue> proto_to_mmvalue(const proto::MMValue& pb_value) {
  if (pb_value.has_single_tensor()) {
    const auto& pb_tensor = pb_value.single_tensor();
    torch::Tensor torch_tensor = util::proto_to_torch(pb_tensor);
    if (!torch_tensor.defined()) {
      LOG(ERROR) << "Failed to convert PB Tensor to torch Tensor";
      return std::nullopt;
    }
    return xllm::MMValue(torch_tensor);
  } else if (pb_value.has_tensor_list()) {
    const auto& pb_tensor_list = pb_value.tensor_list();
    std::vector<torch::Tensor> torch_tensor_vec;
    torch_tensor_vec.reserve(pb_tensor_list.tensors_size());
    for (const auto& pb_tensor : pb_tensor_list.tensors()) {
      torch::Tensor torch_tensor = util::proto_to_torch(pb_tensor);
      if (!torch_tensor.defined()) {
        LOG(ERROR) << "Failed to convert PB Tensor to torch Tensor (list item)";
        return std::nullopt;
      }
      torch_tensor_vec.emplace_back(std::move(torch_tensor));
    }
    return xllm::MMValue(torch_tensor_vec);
  } else {
    LOG(ERROR) << "PB MMValue has no valid value";
    return std::nullopt;
  }
}

template <typename ProtoMap>
bool mmdict_to_proto(const xllm::MMDict& cpp_dict, ProtoMap* pb_dict) {
  if (!pb_dict) {
    LOG(ERROR) << "PB MMDict pointer is null";
    return false;
  }

  for (const auto& [key, cpp_value] : cpp_dict) {
    proto::MMValue& pb_value = (*pb_dict)[key];
    if (!mmvalue_to_proto(cpp_value, &pb_value)) {
      LOG(ERROR) << "Failed to convert struct MMValue for key: " << key;
      return false;
    }
  }

  return true;
}

template <typename ProtoMap>
std::optional<xllm::MMDict> proto_to_mmdict(const ProtoMap& pb_dict) {
  xllm::MMDict cpp_dict;

  for (const auto& [key, pb_value] : pb_dict) {
    auto cpp_value_opt = proto_to_mmvalue(pb_value);
    if (!cpp_value_opt) {
      LOG(ERROR) << "Failed to convert PB MMValue for key: " << key;
      return std::nullopt;
    }

    cpp_dict.emplace(key, std::move(*cpp_value_opt));
  }

  return cpp_dict;
}

bool mmitem_state_to_proto(const xllm::MMItemState& cpp_state,
                           proto::MMItemState* pb_state) {
  pb_state->set_token_pos_offset(cpp_state.token_pos().offset);
  pb_state->set_token_pos_length(cpp_state.token_pos().length);
  if (cpp_state.mm_token_mask().defined() &&
      !util::torch_to_proto(cpp_state.mm_token_mask(),
                            pb_state->mutable_mm_token_mask())) {
    LOG(ERROR) << "Failed to convert MMItemState mm_token_mask";
    return false;
  }
  pb_state->set_schedule_data_key(std::string(
      reinterpret_cast<const char*>(cpp_state.schedule_data().key.data),
      XXH3_128BITS_HASH_VALUE_LEN));
  pb_state->set_schedule_data_start_pos(cpp_state.schedule_data().start_pos);
  pb_state->set_schedule_data_end_pos(cpp_state.schedule_data().end_pos);
  return true;
}

bool proto_to_mmitem_state(const proto::MMItemState& pb_state,
                           xllm::MMItemState* cpp_state) {
  if (!cpp_state) {
    LOG(ERROR) << "Struct MMItemState pointer is null";
    return false;
  }

  cpp_state->mutable_token_pos().offset = pb_state.token_pos_offset();
  cpp_state->mutable_token_pos().length = pb_state.token_pos_length();

  if (pb_state.has_mm_token_mask()) {
    torch::Tensor mm_token_mask =
        util::proto_to_torch(pb_state.mm_token_mask());
    if (!mm_token_mask.defined()) {
      LOG(ERROR) << "Failed to convert PB MMItemState mm_token_mask";
      return false;
    }
    cpp_state->mutable_mm_token_mask() = std::move(mm_token_mask);
  }

  std::memset(cpp_state->mutable_schedule_data().key.data,
              0,
              XXH3_128BITS_HASH_VALUE_LEN);
  const std::string& schedule_data_key = pb_state.schedule_data_key();
  if (!schedule_data_key.empty()) {
    if (schedule_data_key.size() != XXH3_128BITS_HASH_VALUE_LEN) {
      LOG(ERROR) << "Invalid MMItemState schedule_data key size: "
                 << schedule_data_key.size();
      return false;
    }
    std::memcpy(cpp_state->mutable_schedule_data().key.data,
                schedule_data_key.data(),
                XXH3_128BITS_HASH_VALUE_LEN);
  }
  cpp_state->mutable_schedule_data().start_pos =
      pb_state.schedule_data_start_pos();
  cpp_state->mutable_schedule_data().end_pos = pb_state.schedule_data_end_pos();

  return true;
}

bool mmdata_item_to_proto(const xllm::MMDataItem& cpp_item,
                          proto::MMDataItem* pb_item) {
  if (!pb_item) {
    LOG(ERROR) << "PB MMDataItem pointer is null";
    return false;
  }

  pb_item->set_type(cpp_item.type());
  pb_item->set_seq_index(cpp_item.state().seq_index());
  if (!mmdict_to_proto(cpp_item.data(), pb_item->mutable_dict())) {
    LOG(ERROR) << "Failed to convert MMDataItem dict";
    return false;
  }
  if (!mmitem_state_to_proto(cpp_item.state(), pb_item->mutable_state())) {
    LOG(ERROR) << "Failed to convert MMDataItem state";
    return false;
  }

  return true;
}

std::optional<xllm::MMDataItem> proto_to_mmdata_item(
    const proto::MMDataItem& pb_item) {
  auto dict_opt = proto_to_mmdict(pb_item.dict());
  if (!dict_opt) {
    LOG(ERROR) << "Failed to convert PB MMDataItem dict";
    return std::nullopt;
  }

  xllm::MMType type{static_cast<xllm::MMType::Value>(pb_item.type())};
  xllm::MMDataItem cpp_item(type, std::move(*dict_opt));
  cpp_item.mutable_state().mutable_seq_index() = pb_item.seq_index();
  if (!proto_to_mmitem_state(pb_item.state(), &cpp_item.mutable_state())) {
    return std::nullopt;
  }

  return cpp_item;
}

bool mmdata_entry_to_proto(const xllm::MMData& cpp_entry,
                           proto::MMDataEntry* pb_entry) {
  if (!pb_entry) {
    LOG(ERROR) << "PB MMDataEntry pointer is null";
    return false;
  }

  pb_entry->set_type(cpp_entry.type());
  if (cpp_entry.hold<xllm::MMItemVec>()) {
    pb_entry->set_is_item_vec(true);
    const auto& cpp_items = cpp_entry.items<xllm::MMItemVec>();
    pb_entry->mutable_items()->Reserve(cpp_items.size());
    for (const auto& cpp_item : cpp_items) {
      if (!mmdata_item_to_proto(cpp_item, pb_entry->add_items())) {
        LOG(ERROR) << "Failed to convert MMDataEntry item";
        return false;
      }
    }
  } else if (cpp_entry.hold<xllm::MMDict>()) {
    pb_entry->set_is_item_vec(false);
    if (!mmdict_to_proto(cpp_entry.items<xllm::MMDict>(),
                         pb_entry->mutable_dict())) {
      LOG(ERROR) << "Failed to convert MMDataEntry dict";
      return false;
    }
  } else {
    LOG(ERROR) << "Unsupported MMDataEntry items type";
    return false;
  }

  return true;
}

std::optional<xllm::MMData> proto_to_mmdata_entry(
    const proto::MMDataEntry& pb_entry) {
  if (pb_entry.is_item_vec()) {
    xllm::MMItemVec cpp_items;
    cpp_items.reserve(pb_entry.items_size());
    for (const auto& pb_item : pb_entry.items()) {
      auto cpp_item_opt = proto_to_mmdata_item(pb_item);
      if (!cpp_item_opt) {
        LOG(ERROR) << "Failed to convert PB MMDataEntry item";
        return std::nullopt;
      }
      cpp_items.emplace_back(std::move(*cpp_item_opt));
    }
    return xllm::MMData(pb_entry.type(), std::move(cpp_items));
  }

  auto dict_opt = proto_to_mmdict(pb_entry.dict());
  if (!dict_opt) {
    LOG(ERROR) << "Failed to convert PB MMDataEntry dict";
    return std::nullopt;
  }
  return xllm::MMData(pb_entry.type(), std::move(*dict_opt));
}
}  // namespace

bool mmdata_to_proto(const xllm::MMBatchData& cpp_mmdata,
                     proto::MMData* pb_mmdata) {
  if (!pb_mmdata) {
    LOG(ERROR) << "PB MMData pointer is null";
    return false;
  }
  if (!cpp_mmdata.valid()) {
    LOG(ERROR) << "Struct MMData is invalid (type=NONE)";
    return false;
  }

  pb_mmdata->Clear();
  pb_mmdata->set_type(cpp_mmdata.type());
  if (!mmdict_to_proto(cpp_mmdata.data(), pb_mmdata->mutable_dict())) {
    LOG(ERROR) << "Failed to convert MMBatchData dict";
    return false;
  }

  const auto& cpp_entries = cpp_mmdata.mm_data_vec();
  pb_mmdata->mutable_entries()->Reserve(cpp_entries.size());
  for (const auto& cpp_entry : cpp_entries) {
    if (!mmdata_entry_to_proto(cpp_entry, pb_mmdata->add_entries())) {
      LOG(ERROR) << "Failed to convert MMBatchData entry";
      return false;
    }
  }

  return true;
}

bool proto_to_mmdata(const proto::MMData& pb_mmdata,
                     xllm::MMBatchData* cpp_mmdata) {
  if (!cpp_mmdata) {
    LOG(ERROR) << "Struct MMData pointer is null";
    return false;
  }

  uint32_t type = pb_mmdata.type();
  if (pb_mmdata.entries_size() > 0) {
    std::vector<xllm::MMData> cpp_entries;
    cpp_entries.reserve(pb_mmdata.entries_size());
    for (const auto& pb_entry : pb_mmdata.entries()) {
      auto cpp_entry_opt = proto_to_mmdata_entry(pb_entry);
      if (!cpp_entry_opt) {
        LOG(ERROR) << "Failed to convert PB MMBatchData entry";
        return false;
      }
      cpp_entries.emplace_back(std::move(*cpp_entry_opt));
    }

    cpp_mmdata->batch(cpp_entries);
    if (cpp_mmdata->type() != type) {
      LOG(WARNING) << "MMBatchData proto type mismatch, top-level type: "
                   << type << ", entries type: " << cpp_mmdata->type();
    }
  } else {
    auto cpp_dict_opt = proto_to_mmdict(pb_mmdata.dict());
    if (!cpp_dict_opt) {
      LOG(ERROR) << "Failed to convert PB MMBatchData dict";
      return false;
    }
    *cpp_mmdata = xllm::MMBatchData(type, std::move(*cpp_dict_opt));
  }

  if (!cpp_mmdata->valid() && type != xllm::MMType::NONE) {
    LOG(ERROR) << "Converted MMBatchData is invalid, proto type: " << type;
    return false;
  }

  return true;
}

}  // namespace xllm
