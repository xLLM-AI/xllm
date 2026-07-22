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

#include "common/global_flags.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUFormat.h>
#endif

#if defined(USE_NPU)
#include "framework/kv_cache_transfer/llm_data_dist_transfer.h"
#endif

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
#include "framework/kv_cache_transfer/mooncake_kv_cache_transfer.h"
#endif

namespace xllm {

folly::SemiFuture<bool> KVCacheTransfer::pull_kv_blocks_async(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        src_cluster_id,
                        src_addr,
                        src_blocks,
                        dst_blocks,
                        src_linear_state_ids,
                        dst_linear_state_ids,
                        promise = std::move(promise)]() mutable {
    const bool success = pull_kv_blocks(src_cluster_id,
                                        src_addr,
                                        src_blocks,
                                        dst_blocks,
                                        src_linear_state_ids,
                                        dst_linear_state_ids);
    promise.setValue(success);
  });
  return future;
}

// In KV-split mode, local_blocks_ids already contains only this KV-split
// rank's physical blocks. remote_blocks_ids holds the full D-side
// total_blocks entries; this rank maps local_block[k] to
// remote_blocks_ids[kv_split_rank + k * kv_split_size]. The function rebuilds
// remote_blocks_ids accordingly and drops infos with no local blocks.
std::vector<TransferKVInfo> filter_kv_split_infos(
    int32_t kv_split_rank,
    int32_t kv_split_size,
    const std::vector<TransferKVInfo>& kv_infos) {
  std::vector<TransferKVInfo> filtered_kv_infos;
  for (const auto& kv_info : kv_infos) {
    if (kv_info.local_blocks_ids.empty() &&
        kv_info.local_linear_state_ids.empty()) {
      continue;
    }
    const size_t n_local = kv_info.local_blocks_ids.size();
    TransferKVInfo filtered = kv_info;
    filtered.remote_blocks_ids.clear();
    size_t mapped_local = 0;
    if (n_local > 0) {
      filtered.remote_blocks_ids.reserve(n_local);
      for (size_t k = 0; k < n_local; ++k) {
        const size_t remote_idx = static_cast<size_t>(kv_split_rank) +
                                  k * static_cast<size_t>(kv_split_size);
        if (remote_idx >= kv_info.remote_blocks_ids.size()) {
          break;
        }
        filtered.remote_blocks_ids.emplace_back(
            kv_info.remote_blocks_ids[remote_idx]);
        ++mapped_local;
      }
    }
    // local_block[k] maps to remote_blocks_ids[kv_split_rank + k *
    // kv_split_size]. When the strided remote index runs past the D-side block
    // list (the prompt spans multiple logical blocks and the last one is not
    // full, which only happens for kv_split_rank > 0), the loop above stops
    // early. local_blocks_ids must then be truncated to the blocks that
    // actually got a remote target; otherwise src/dst counts differ and
    // PushKvBlocks rejects the whole transfer, leaving decode with
    // un-transferred KV (-> repetition). The dropped tail blocks correspond to
    // tokens beyond the prompt length, so the truncation is loss-free.
    filtered.local_blocks_ids.resize(mapped_local);
    if (!filtered.remote_blocks_ids.empty() ||
        !filtered.remote_linear_state_ids.empty()) {
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
    // so we skip it and let each rank consume remote_blocks_ids 1:1.
    const int32_t kv_split_size = parallel_args.kv_split_size_effective();
    if (kv_split_size > 1) {
      filtered_kv_infos = filter_kv_split_infos(
          parallel_args.kv_split_rank(), kv_split_size, *kv_infos);
      kv_infos = &filtered_kv_infos;
      if (kv_infos->empty()) {
        promise.setValue(true);
        return;
      }
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
  auto append_transfer_groups =
      [](std::vector<KVBlockTransferGroup>& dst,
         const std::vector<KVBlockTransferGroup>& src) {
        for (const auto& src_group : src) {
          auto it =
              std::find_if(dst.begin(),
                           dst.end(),
                           [&src_group](const KVBlockTransferGroup& group) {
                             return group.group_id == src_group.group_id;
                           });
          if (it == dst.end()) {
            dst.emplace_back(src_group);
            continue;
          }
          it->local_blocks_ids.insert(it->local_blocks_ids.end(),
                                      src_group.local_blocks_ids.begin(),
                                      src_group.local_blocks_ids.end());
          it->remote_blocks_ids.insert(it->remote_blocks_ids.end(),
                                       src_group.remote_blocks_ids.begin(),
                                       src_group.remote_blocks_ids.end());
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
        kv_info.src_blocks.insert(kv_info.src_blocks.end(),
                                  info.local_blocks_ids.begin(),
                                  info.local_blocks_ids.end());
        kv_info.dst_blocks.insert(kv_info.dst_blocks.end(),
                                  info.remote_blocks_ids.begin(),
                                  info.remote_blocks_ids.end());
        kv_info.src_linear_state_ids.insert(kv_info.src_linear_state_ids.end(),
                                            info.local_linear_state_ids.begin(),
                                            info.local_linear_state_ids.end());
        kv_info.dst_linear_state_ids.insert(
            kv_info.dst_linear_state_ids.end(),
            info.remote_linear_state_ids.begin(),
            info.remote_linear_state_ids.end());

        // XTensor mode: copy destination offsets
        if (!info.dst_xtensor_layer_offsets.empty()) {
          kv_info.dst_xtensor_layer_offsets = info.dst_xtensor_layer_offsets;
        }
        append_transfer_groups(kv_info.block_transfer_groups,
                               info.block_transfer_groups);

        merged_kv_infos[key] = std::move(kv_info);
      } else {
        merged_kv_infos[key].src_blocks.insert(
            merged_kv_infos[key].src_blocks.end(),
            info.local_blocks_ids.begin(),
            info.local_blocks_ids.end());
        merged_kv_infos[key].dst_blocks.insert(
            merged_kv_infos[key].dst_blocks.end(),
            info.remote_blocks_ids.begin(),
            info.remote_blocks_ids.end());
        merged_kv_infos[key].src_linear_state_ids.insert(
            merged_kv_infos[key].src_linear_state_ids.end(),
            info.local_linear_state_ids.begin(),
            info.local_linear_state_ids.end());
        merged_kv_infos[key].dst_linear_state_ids.insert(
            merged_kv_infos[key].dst_linear_state_ids.end(),
            info.remote_linear_state_ids.begin(),
            info.remote_linear_state_ids.end());

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
        append_transfer_groups(merged_kv_infos[key].block_transfer_groups,
                               info.block_transfer_groups);
      }
    }
  }
}

#if defined(USE_NPU)
std::vector<torch::Tensor> KVCacheTransfer::convert_to_torch_tensor(
    const std::vector<int64_t>& dims,
    const torch::ScalarType dtype,
    const std::vector<uintptr_t>& addresses,
    const aclFormat format) {
  std::vector<torch::Tensor> torch_tensors;
  c10::DeviceType device_type = c10::DeviceType::PrivateUse1;
  torch::TensorOptions option =
      torch::TensorOptions().dtype(dtype).device(device_type);

  torch_tensors.reserve(addresses.size());
  for (auto dev_addr : addresses) {
    auto tensor = torch::empty({0}, option);
    auto address = reinterpret_cast<void*>(dev_addr);
    torch::DataPtr c10_data_ptr(
        address, address, [](void*) {}, tensor.device());

    size_t tensor_nbytes = at::detail::computeStorageNbytesContiguous(
        dims, tensor.dtype().itemsize());
    torch::Storage storage;
    // get npu storage constructor from register and construct storage
    auto fptr = c10::GetStorageImplCreate(device_type);
    auto allocator = c10::GetAllocator(device_type);

    // PyTorch 2.7+: StorageImpl now takes DataPtr instead of raw allocator
    storage = fptr(c10::StorageImpl::use_byte_size_t(),
                   c10::SymInt(tensor_nbytes),
                   std::move(c10_data_ptr),
                   allocator,
                   true);

    tensor.set_(storage, 0, dims);
    auto* tensor_storage = static_cast<torch_npu::NPUStorageImpl*>(
        tensor.storage().unsafeGetStorageImpl());
    tensor_storage->npu_desc_.npu_format_ = format;
    torch_tensors.emplace_back(std::move(tensor));
  }
  return torch_tensors;
}
#endif

std::shared_ptr<KVCacheTransfer> KVCacheTransferFactory::create(
    const std::string& transfer_type,
    uint16_t transfer_listen_port,
    InstanceRole instance_role,
    const Device& device,
    const KVCacheShape& kv_cache_shape,
    torch::ScalarType dtype,
    std::vector<xllm::KVCache>& kv_caches,
    int64_t num_layers,
    AllocateKVCacheFunc allocate_kv_cache_func,
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

    transfer->initialize(device_id);
    CHECK(allocate_kv_cache_func(kv_cache_shape,
                                 /*use_huge_page_allocator=*/true,
                                 /*tensor_allocator=*/nullptr))
        << "Allocate KV cache failed.";
    transfer->register_kv_cache(kv_caches, kv_cache_shape, dtype);
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

    mooncake_transfer->initialize(device_id);
#if defined(USE_MLU)
    CHECK(allocate_kv_cache_func(kv_cache_shape,
                                 /*use_huge_page_allocator=*/false,
                                 mlu_mooncake_tensor_allocator()))
        << "Allocate KV cache failed.";
#else
    // TODO(xllm-kv-allocator): NPU/DCU/XTensor remains on its existing
    // physical allocation path in the MLU Mooncake migration. A follow-up
    // must route this backend through
    // KVCacheCreateOptions::tensor_allocator without moving cache structure
    // decisions into Transfer. Do not add indexer layer-mask handling here.
    mooncake_transfer->allocate_kv_cache(
        kv_caches, num_layers, kv_cache_shape, dtype);
#endif
    mooncake_transfer->register_kv_cache(kv_caches, kv_cache_shape, dtype);

    transfer = mooncake_transfer;
  } else {
    LOG(FATAL) << "Unsupported KVCacheTransfer type : " << transfer_type;
  }
#endif

  return transfer;
}

}  // namespace xllm
