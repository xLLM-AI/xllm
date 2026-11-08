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

#include "core/kv_cache/transfer/kv_transport_codec.h"

#include "core/kv_cache/transfer/prefetch_result.h"
#include "worker.pb.h"

namespace xllm {
namespace {

bool valid_transfer_type(int32_t value) {
  return proto::TransferType_IsValid(value);
}

bool valid_block_type(int32_t value) { return proto::BlockType_IsValid(value); }

bool valid_block_id(int32_t value) { return value >= 0; }

bool valid_prefetch_source_id(int32_t value) {
  // Store GET addresses its source by hash; -1 denotes no local source block.
  return value == -1 || valid_block_id(value);
}

}  // namespace

bool proto_to_block_transfer_info(
    const proto::BlockTransferInfos& pb_block_transfer_info,
    std::vector<BlockTransferInfo>& block_transfer_info,
    uint64_t* batch_id) {
  if (batch_id == nullptr || pb_block_transfer_info.transfer_infos().empty() ||
      !valid_transfer_type(pb_block_transfer_info.transfer_type())) {
    return false;
  }

  std::vector<BlockTransferInfo> decoded_infos;
  decoded_infos.reserve(pb_block_transfer_info.transfer_infos_size());

  for (const proto::BlockTransferInfo& pb_info :
       pb_block_transfer_info.transfer_infos()) {
    if (!valid_block_id(pb_info.src_block_id()) ||
        !valid_block_id(pb_info.dst_block_id()) ||
        pb_info.hash_key().size() != XXH3_128BITS_HASH_VALUE_LEN ||
        !valid_block_type(pb_info.block_type())) {
      return false;
    }
    decoded_infos.emplace_back(
        pb_info.src_block_id(),
        pb_info.dst_block_id(),
        reinterpret_cast<const uint8_t*>(pb_info.hash_key().data()),
        static_cast<TransferType>(pb_block_transfer_info.transfer_type()),
        static_cast<BlockType>(pb_info.block_type()));
  }

  *batch_id = pb_block_transfer_info.batch_id();
  block_transfer_info = std::move(decoded_infos);
  return true;
}

bool block_transfer_info_to_proto(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info) {
  if (pb_block_transfer_info == nullptr || block_transfer_info.empty()) {
    return false;
  }
  const TransferType transfer_type = block_transfer_info[0].transfer_type;
  proto::BlockTransferInfos encoded_info;
  encoded_info.mutable_transfer_infos()->Reserve(block_transfer_info.size());
  for (const BlockTransferInfo& info : block_transfer_info) {
    if (transfer_type != info.transfer_type) {
      LOG(ERROR) << "Convert to BlockTransferInfos fail, TransferType must be "
                    "same, but got "
                 << static_cast<uint32_t>(transfer_type) << " and "
                 << static_cast<uint32_t>(info.transfer_type);
      return false;
    }

    if (!valid_transfer_type(static_cast<int32_t>(info.transfer_type)) ||
        !valid_block_id(info.src_block_id) ||
        !valid_block_id(info.dst_block_id) ||
        !valid_block_type(static_cast<int32_t>(info.block_type))) {
      return false;
    }
    proto::BlockTransferInfo* pb_cache = encoded_info.add_transfer_infos();
    pb_cache->set_src_block_id(info.src_block_id);
    pb_cache->set_dst_block_id(info.dst_block_id);
    pb_cache->set_hash_key(info.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
    pb_cache->set_block_type(
        static_cast<proto::BlockType>(static_cast<int8_t>(info.block_type)));
  }
  encoded_info.set_transfer_type(
      static_cast<proto::TransferType>(static_cast<int8_t>(transfer_type)));
  pb_block_transfer_info->Swap(&encoded_info);

  return true;
}

bool block_transfer_info_to_proto(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info,
    proto::BlockTransferInfos* pb_block_transfer_info) {
  if (!block_transfer_info_to_proto(block_transfer_info,
                                    pb_block_transfer_info)) {
    return false;
  }
  pb_block_transfer_info->set_batch_id(batch_id);
  return true;
}

bool StoragePrefetchRequest::to_proto(
    proto::PrefetchRequest* proto_request) const {
  if (proto_request == nullptr || !valid()) {
    return false;
  }

  proto::PrefetchRequest encoded_request;
  encoded_request.mutable_units()->Reserve(units.size());
  size_t begin = 0;
  for (const PrefetchUnit& unit : units) {
    proto::PrefetchUnit* proto_unit = encoded_request.add_units();
    proto_unit->set_has_non_gated(unit.has_non_gated);
    for (size_t offset = begin; offset < unit.end; ++offset) {
      const BlockTransferInfo& info = transfer_infos[offset];
      if (!valid_transfer_type(static_cast<int32_t>(info.transfer_type)) ||
          !valid_prefetch_source_id(info.src_block_id) ||
          !valid_block_id(info.dst_block_id) ||
          !valid_block_type(static_cast<int32_t>(info.block_type))) {
        return false;
      }
      auto* proto_info = offset < unit.gated_end
                             ? proto_unit->add_gated_blocks()
                             : proto_unit->add_non_gated_blocks();
      proto_info->set_src_block_id(info.src_block_id);
      proto_info->set_dst_block_id(info.dst_block_id);
      proto_info->set_hash_key(info.hash_key, XXH3_128BITS_HASH_VALUE_LEN);
      proto_info->set_block_type(
          static_cast<proto::BlockType>(static_cast<int8_t>(info.block_type)));
    }
    begin = unit.end;
  }
  proto_request->Swap(&encoded_request);
  return true;
}

bool StoragePrefetchRequest::from_proto(
    const proto::PrefetchRequest& proto_request) {
  StoragePrefetchRequest decoded_request;
  auto append_transfer =
      [&decoded_request](const proto::BlockTransferInfo& proto_info) {
        if (!valid_prefetch_source_id(proto_info.src_block_id()) ||
            !valid_block_id(proto_info.dst_block_id()) ||
            proto_info.hash_key().size() != XXH3_128BITS_HASH_VALUE_LEN ||
            !valid_block_type(proto_info.block_type())) {
          return false;
        }
        decoded_request.transfer_infos.emplace_back(
            proto_info.src_block_id(),
            proto_info.dst_block_id(),
            reinterpret_cast<const uint8_t*>(proto_info.hash_key().data()),
            TransferType::G2H,
            static_cast<BlockType>(proto_info.block_type()));
        return true;
      };
  size_t transfer_count = 0;
  for (const proto::PrefetchUnit& proto_unit : proto_request.units()) {
    transfer_count += static_cast<size_t>(proto_unit.gated_blocks_size()) +
                      proto_unit.non_gated_blocks_size();
  }
  decoded_request.transfer_infos.reserve(transfer_count);
  decoded_request.units.reserve(proto_request.units_size());
  for (const proto::PrefetchUnit& proto_unit : proto_request.units()) {
    PrefetchUnit unit;
    unit.has_non_gated = proto_unit.has_non_gated();
    for (const proto::BlockTransferInfo& proto_info :
         proto_unit.gated_blocks()) {
      if (!append_transfer(proto_info)) {
        return false;
      }
    }
    unit.gated_end = decoded_request.transfer_infos.size();
    for (const proto::BlockTransferInfo& proto_info :
         proto_unit.non_gated_blocks()) {
      if (!append_transfer(proto_info)) {
        return false;
      }
    }
    unit.end = decoded_request.transfer_infos.size();
    decoded_request.units.emplace_back(unit);
  }
  if (!decoded_request.valid()) {
    return false;
  }
  *this = std::move(decoded_request);
  return true;
}

}  // namespace xllm
