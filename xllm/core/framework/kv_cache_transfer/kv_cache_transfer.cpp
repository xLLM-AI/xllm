/* Copyright 2025-2026 The xLLM Authors.

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

#include "framework/kv_cache_transfer/kv_cache_transfer.h"

#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <unordered_set>

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"

#if defined(USE_NPU)
#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"
#endif

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif

namespace xllm {

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<KVTransferMapping>& mappings,
    const std::string& request_id,
    int32_t kv_split_size) {
  if (kv_split_size < 1) {
    LOG(ERROR) << "KV cache transfer requires kv_split_size >= 1, request_id="
               << request_id << ", kv_split_size=" << kv_split_size;
    return false;
  }

  std::unordered_set<int32_t> group_ids;
  group_ids.reserve(mappings.size());
  for (const KVTransferMapping& mapping : mappings) {
    if (!group_ids.emplace(mapping.group_id).second) {
      LOG(ERROR) << "Duplicate KV cache transfer mapping, request_id="
                 << request_id << ", group_id=" << mapping.group_id;
      return false;
    }

    const bool validate_full_kv_split_coverage =
        kv_split_size > 1 && mapping.group_id == cache_group_id(BlockType::KV);
    if (!validate_full_kv_split_coverage) {
      if (mapping.local_ids.size() != mapping.remote_ids.size()) {
        LOG(ERROR) << "KV cache transfer mapping size mismatch, request_id="
                   << request_id << ", group_id=" << mapping.group_id
                   << ", local=" << mapping.local_ids.size()
                   << ", remote=" << mapping.remote_ids.size();
        return false;
      }
      continue;
    }

    const size_t local_count = mapping.local_ids.size();
    const size_t remote_count = mapping.remote_ids.size();
    if (local_count == 0) {
      if (remote_count != 0) {
        LOG(ERROR) << "KV-split mapping has remote ids without local ids, "
                   << "request_id=" << request_id
                   << ", group_id=" << mapping.group_id
                   << ", remote=" << remote_count;
        return false;
      }
      continue;
    }

    const size_t split_size = static_cast<size_t>(kv_split_size);
    if (local_count > std::numeric_limits<size_t>::max() / split_size) {
      LOG(ERROR) << "KV-split mapping coverage size overflow, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count
                 << ", kv_split_size=" << kv_split_size;
      return false;
    }
    const size_t max_remote_count = local_count * split_size;
    const size_t min_remote_count = max_remote_count - split_size + 1;
    if (remote_count < min_remote_count || remote_count > max_remote_count) {
      LOG(ERROR) << "KV-split mapping remote coverage mismatch, request_id="
                 << request_id << ", group_id=" << mapping.group_id
                 << ", local=" << local_count << ", remote=" << remote_count
                 << ", kv_split_size=" << kv_split_size
                 << ", expected_remote_range=[" << min_remote_count << ", "
                 << max_remote_count << "]";
      return false;
    }
  }
  return true;
}

bool KVCacheTransfer::validate_transfer_mappings(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    int32_t kv_split_size) {
  for (const TransferKVInfo& info : transfer_kv_infos) {
    if (!validate_transfer_mappings(
            info.mappings, info.request_id, kv_split_size)) {
      return false;
    }
  }
  return true;
}

folly::SemiFuture<bool> KVCacheTransfer::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<KVTransferMapping>& mappings) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  if (!validate_transfer_mappings(
          mappings, /*request_id=*/"PULL", /*kv_split_size=*/1)) {
    promise.setValue(false);
    return future;
  }
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        mappings,
                        promise = std::move(promise)]() mutable {
    const bool success = pull_kv_blocks(src_cluster_id, src_addr, mappings);
    promise.setValue(success);
  });
  return future;
}

// In KV-split mode, the KV mapping's local_ids already contains only this
// rank's physical blocks. remote_ids holds the full D-side block entries; this
// rank maps local_ids[k] to remote_ids[kv_split_rank + k * kv_split_size]. The
// function rebuilds remote_ids accordingly and drops infos with no mappings.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos) {
  std::vector<TransferKVInfo> filtered_kv_infos;
  for (const TransferKVInfo& kv_info : kv_infos) {
    TransferKVInfo filtered = kv_info;
    for (KVTransferMapping& mapping : filtered.mappings) {
      if (mapping.group_id != cache_group_id(BlockType::KV)) {
        continue;
      }
      const std::vector<uint64_t> remote_ids = mapping.remote_ids;
      mapping.remote_ids.clear();
      size_t mapped_local = 0;
      mapping.remote_ids.reserve(mapping.local_ids.size());
      for (size_t k = 0; k < mapping.local_ids.size(); ++k) {
        const size_t remote_idx = static_cast<size_t>(kv_split_rank) +
                                  k * static_cast<size_t>(kv_split_size);
        if (remote_idx >= remote_ids.size()) {
          break;
        }
        mapping.remote_ids.emplace_back(remote_ids[remote_idx]);
        ++mapped_local;
      }
      mapping.local_ids.resize(mapped_local);
    }
    // local_ids[k] maps to remote_ids[kv_split_rank + k * kv_split_size]. When
    // the strided remote index runs past the D-side block list (the prompt
    // spans multiple logical blocks and the last one is not full, which only
    // happens for kv_split_rank > 0), the loop above stops early. local_ids
    // must then be truncated to the blocks that actually got a remote target;
    // otherwise the two sides differ in size and PushKvBlocks rejects the whole
    // transfer. The dropped tail blocks correspond to tokens beyond the prompt
    // length, so the truncation is loss-free.
    const bool has_mapping = std::any_of(filtered.mappings.begin(),
                                         filtered.mappings.end(),
                                         [](const KVTransferMapping& mapping) {
                                           return !mapping.local_ids.empty() &&
                                                  !mapping.remote_ids.empty();
                                         });
    if (has_mapping) {
      filtered_kv_infos.push_back(std::move(filtered));
    }
  }
  return filtered_kv_infos;
}

std::vector<std::string> KVCacheTransfer::rotate_dst_rank(
    const std::vector<std::string>& keys,
    int32_t kv_split_rank) {
  int32_t offset = kv_split_rank;
  std::vector<std::string> rotated_keys;
  auto sorted_keys = keys;
  std::sort(sorted_keys.begin(), sorted_keys.end());
  for (int32_t i = 0; i < keys.size(); i++) {
    rotated_keys.emplace_back(sorted_keys[(i + offset) % sorted_keys.size()]);
  }
  return rotated_keys;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
folly::SemiFuture<bool> KVCacheTransfer::push_kv_blocks_async(
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args,
    std::shared_ptr<KVPushSynchronizerImpl> layer_synchronizer,
    bool is_spec_draft) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        transfer_kv_infos,
                        &parallel_args,
                        layer_synchronizer,
                        is_spec_draft,
                        promise = std::move(promise)]() mutable {
    std::unordered_map<std::string, KVCacheInfo> merged_kv_infos;
    std::vector<TransferKVInfo> filtered_kv_infos;
    const std::vector<TransferKVInfo>* kv_infos = &transfer_kv_infos;
    // Filter when KV is actually sharded across ranks. When kv_split_size==1
    // (each CP rank holds a full KV replica) the filter degenerates to a copy,
    // so we skip it and let each rank consume remote_ids 1:1.
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    if (!validate_transfer_mappings(*kv_infos, kv_split_size)) {
      promise.setValue(false);
      return;
    }
    if (kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
    }
    if (!validate_transfer_mappings(*kv_infos, /*kv_split_size=*/1)) {
      promise.setValue(false);
      return;
    }
    merge_kv_blocks(merged_kv_infos, *kv_infos, parallel_args);
    bool success = true;
    if (!merged_kv_infos.empty()) {
      success = this->push_kv_blocks(merged_kv_infos,
                                     layer_synchronizer,
                                     is_spec_draft,
                                     parallel_args.kv_split_rank(),
                                     parallel_args.kv_split_size_effective());
    }
    promise.setValue(success);
  });
  return future;
}
#endif

void KVCacheTransfer::merge_kv_blocks(
    std::unordered_map<std::string, KVCacheInfo>& merged_kv_infos,
    const std::vector<TransferKVInfo>& transfer_kv_infos,
    const ParallelArgs& parallel_args) {
  // Obtain the parallel parameters of the source instance.
  // When CP is enabled on the P side, the per-DP worker count is
  // cp_size * tp_size. We need the *actual* TP size (excluding CP) so that
  // src_dp_local_tp_rank correctly reflects only the TP dimension.
  // Using cp_size * tp_size here would make CP rank > 0 workers appear to
  // have a tp_rank >= dst_world_size, causing the linked_dp_ranks filter to
  // skip all requests for those workers.
  int32_t src_rank = parallel_args.rank();
  int32_t src_dp_size = parallel_args.dp_size();
  int32_t src_kv_split_size = parallel_args.kv_split_size_effective();
  int32_t src_world_size = parallel_args.world_size();
  int32_t src_tp_size = src_world_size / src_dp_size / src_kv_split_size;
  int32_t src_dp_local_tp_rank = src_rank % src_tp_size;
  auto append_mappings = [](std::vector<KVTransferMapping>& dst,
                            const std::vector<KVTransferMapping>& src) {
    for (const KVTransferMapping& src_mapping : src) {
      auto it = std::find_if(dst.begin(),
                             dst.end(),
                             [&src_mapping](const KVTransferMapping& mapping) {
                               return mapping.group_id == src_mapping.group_id;
                             });
      if (it == dst.end()) {
        dst.emplace_back(src_mapping);
        continue;
      }
      it->local_ids.insert(it->local_ids.end(),
                           src_mapping.local_ids.begin(),
                           src_mapping.local_ids.end());
      it->remote_ids.insert(it->remote_ids.end(),
                            src_mapping.remote_ids.begin(),
                            src_mapping.remote_ids.end());
    }
  };
  for (auto& info : transfer_kv_infos) {
    // Obtain the parallel parameters of the destination instance.
    int32_t dst_dp_rank = info.dp_rank;
    int32_t dst_dp_size = info.remote_instance_info.dp_size;
    int32_t dst_world_size = info.remote_instance_info.cluster_ids.size();
    int32_t dst_tp_size = dst_world_size / dst_dp_size;
    // Get the DP groups of the destination instance connected to the current
    // worker.
    std::unordered_set<int32_t> linked_dp_ranks;
    for (int32_t i = src_dp_local_tp_rank; i < dst_world_size;
         i += src_tp_size) {
      int32_t linked_dp_rank = i / dst_tp_size;
      linked_dp_ranks.emplace(linked_dp_rank);
    }
    // If the target DP rank of the request is not linked to the current worker,
    // skip the request.
    if (linked_dp_ranks.find(dst_dp_rank) == linked_dp_ranks.end()) {
      continue;
    }
    // The current worker needs to push the KV Cache to all workers in the
    // destination DP group it is connected to.
    for (int32_t i =
             src_dp_local_tp_rank % dst_tp_size + dst_tp_size * dst_dp_rank;
         i < dst_tp_size * (dst_dp_rank + 1);
         i += src_tp_size) {
      uint64_t dst_cluster_id = info.remote_instance_info.cluster_ids[i];
      auto& dst_addr = info.remote_instance_info.addrs[i];
      std::string key = std::to_string(dst_cluster_id) + "_" + dst_addr;
      // Merge all kv blocks with the same destination worker into a single
      // vector.
      if (merged_kv_infos.find(key) == merged_kv_infos.end()) {
        KVCacheInfo kv_info;
        kv_info.dst_cluster_id = dst_cluster_id;
        kv_info.dst_addr = dst_addr;
        append_mappings(kv_info.mappings, info.mappings);

        // XTensor mode: copy destination offsets
        if (!info.dst_xtensor_layer_offsets.empty()) {
          kv_info.dst_xtensor_layer_offsets = info.dst_xtensor_layer_offsets;
        }
        merged_kv_infos[key] = std::move(kv_info);
      } else {
        append_mappings(merged_kv_infos[key].mappings, info.mappings);

        // XTensor mode: merge destination offsets (append to each layer)
        if (!info.dst_xtensor_layer_offsets.empty()) {
          auto& existing = merged_kv_infos[key].dst_xtensor_layer_offsets;
          // Initialize if not already done
          if (existing.empty()) {
            existing = info.dst_xtensor_layer_offsets;
          } else {
            // Append offsets for each layer
            for (size_t layer = 0;
                 layer < info.dst_xtensor_layer_offsets.size() &&
                 layer < existing.size();
                 ++layer) {
              existing[layer].k_offsets.insert(
                  existing[layer].k_offsets.end(),
                  info.dst_xtensor_layer_offsets[layer].k_offsets.begin(),
                  info.dst_xtensor_layer_offsets[layer].k_offsets.end());
              existing[layer].v_offsets.insert(
                  existing[layer].v_offsets.end(),
                  info.dst_xtensor_layer_offsets[layer].v_offsets.begin(),
                  info.dst_xtensor_layer_offsets[layer].v_offsets.end());
            }
          }
        }
      }
    }
  }
}

std::shared_ptr<KVCacheTransfer> KVCacheTransferFactory::create(
    const std::string& transfer_type,
    uint16_t transfer_listen_port,
    InstanceRole instance_role,
    const Device& device,
    bool enable_lighting_indexer,
    const std::string& model_type,
    const std::string& model_id) {
  std::shared_ptr<KVCacheTransfer> transfer;

  int32_t device_id = device.index();

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  LOG(INFO) << "Create KVCacheTransfer for " << transfer_type << "flag"
            << ::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type();
  if (transfer_type == "LlmDataDist") {
#if defined(USE_NPU)
    transfer = std::make_shared<LlmDataDistTransfer>(
        transfer_listen_port, instance_role, enable_lighting_indexer);
#else
    LOG(FATAL) << "LlmDataDist is not supported on this backend.";
#endif
  } else if (transfer_type == "Mooncake") {
    std::shared_ptr<MooncakeKVCacheTransferBase> mooncake_transfer;
#if defined(USE_NPU)
    if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
      auto xtensor_transfer = std::make_shared<MooncakeKVCacheTransferXTensor>(
          device_id, transfer_listen_port, device);
      if (!model_id.empty()) {
        xtensor_transfer->set_model_id(model_id);
        LOG(INFO)
            << "XTensor mode enabled for MooncakeKVCacheTransfer, model_id="
            << model_id;
      }
      mooncake_transfer = xtensor_transfer;
    } else {
      mooncake_transfer = std::make_shared<MooncakeKVCacheTransferDefault>(
          device_id, transfer_listen_port, device, model_type);
    }
#else
    mooncake_transfer = std::make_shared<MooncakeKVCacheTransferDefault>(
        device_id, transfer_listen_port, device, model_type);
#endif

    transfer = mooncake_transfer;
  } else {
    LOG(FATAL) << "Unsupported KVCacheTransfer type : " << transfer_type;
  }
#endif

  return transfer;
}

}  // namespace xllm
