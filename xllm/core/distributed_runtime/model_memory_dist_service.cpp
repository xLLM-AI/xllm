/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/distributed_runtime/model_memory_dist_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <optional>
#include <vector>

#include "core/common/device_monitor.h"
#include "core/framework/allocator/global_memory_region.h"
#include "core/framework/allocator/model_memory_manager.h"
#include "core/framework/allocator/virtual_memory/physical_page_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/platform/device.h"

namespace xllm {

ModelMemoryDistService::ModelMemoryDistService(int32_t global_rank,
                                               int32_t world_size,
                                               const torch::Device& device)
    : global_rank_(global_rank),
      world_size_(world_size),
      device_(device),
      initialized_(false) {}

void ModelMemoryDistService::Hello(
    ::google::protobuf::RpcController* controller,
    const proto::Status* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  brpc::ClosureGuard done_guard(done);
  auto ctrl = reinterpret_cast<brpc::Controller*>(controller);
  if (!initialized_) {
    ctrl->SetFailed("Server is not initialized");
  } else {
    response->set_ok(true);
  }
}

void ModelMemoryDistService::GetMemoryInfo(
    ::google::protobuf::RpcController* controller,
    const proto::Status* request,
    proto::MemoryInfoResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    Device device(device_);
    device.set_device();

    // Empty torch cache to get accurate memory info
    int32_t device_id = device_.index();

    const auto available_memory = device.free_memory();
    const auto total_memory = device.total_memory();

    // Update device monitor
    DeviceMonitor::get_instance().set_total_memory(device_id, total_memory);

    LOG(INFO) << "GetMemoryInfo: global_rank=" << global_rank_
              << ", available_memory=" << available_memory
              << ", total_memory=" << total_memory;

    // Returns 0 for both fields on failure (handled by caller)
    response->set_available_memory(available_memory);
    response->set_total_memory(total_memory);
  });
}

void ModelMemoryDistService::InitPhysicalPagePool(
    ::google::protobuf::RpcController* controller,
    const proto::InitPhysicalPagePoolRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    int64_t num_pages = request->num_pages();
    LOG(INFO) << "InitPhysicalPagePool: global_rank=" << global_rank_
              << ", num_pages=" << num_pages;

    try {
      // Initialize PhysicalPagePool with specified number of pages
      const size_t page_size = static_cast<size_t>(
          KVCacheConfig::get_instance().phy_page_granularity_size());
      PhysicalPagePool::get_instance().init(device_, num_pages, page_size);

      // Initialize GlobalMemoryRegion after PhysicalPagePool
      GlobalMemoryRegion::get_instance().init(device_);
      LOG(INFO) << "GlobalMemoryRegion initialized on worker " << global_rank_;

      response->set_ok(true);
    } catch (const std::exception& e) {
      LOG(ERROR) << "Failed to init PhysicalPagePool/GlobalMemoryRegion: "
                 << e.what();
      response->set_ok(false);
    }
  });
}

void ModelMemoryDistService::MapToKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call ModelMemoryManager to map
    auto& allocator = ModelMemoryManager::get_instance();
    bool success = allocator.map_to_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void ModelMemoryDistService::UnmapFromKvTensors(
    ::google::protobuf::RpcController* controller,
    const proto::KvTensorRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto offsets to vector
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (int i = 0; i < request->offsets_size(); ++i) {
      offsets.push_back(request->offsets(i));
    }

    // Call ModelMemoryManager to unmap
    auto& allocator = ModelMemoryManager::get_instance();
    bool success = allocator.unmap_from_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void ModelMemoryDistService::AllocWeightPages(
    ::google::protobuf::RpcController* controller,
    const proto::AllocWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    size_t num_pages = request->num_pages();

    LOG(INFO) << "AllocWeightPages: model_id=" << model_id
              << ", num_pages=" << num_pages;

    auto& pool = PhysicalPagePool::get_instance();
    auto& allocator = ModelMemoryManager::get_instance();

    // Try contiguous allocation first (from GlobalMemoryRegion)
    page_id_t start_page = pool.allocate_contiguous_from_right(num_pages);
    if (start_page >= 0) {
      if (!allocator.record_weight_allocation(
              model_id, start_page, num_pages)) {
        response->set_ok(false);
        return;
      }
      response->set_ok(true);
      LOG(INFO) << "AllocWeightPages success: model_id=" << model_id
                << ", start_page=" << start_page << ", num_pages=" << num_pages;
      return;
    }

    // Fallback: try non-contiguous allocation using MappedMemoryRegion
    LOG(WARNING)
        << "Contiguous allocation failed for " << num_pages
        << " pages, trying non-contiguous fallback (MappedMemoryRegion)";

    std::vector<page_id_t> page_ids = pool.allocate_pages_from_right(num_pages);
    if (page_ids.empty()) {
      LOG(ERROR) << "Failed to allocate " << num_pages
                 << " weight pages (both contiguous and non-contiguous)";
      response->set_ok(false);
      return;
    }

    if (!allocator.record_weight_fallback_allocation(model_id, page_ids)) {
      response->set_ok(false);
      return;
    }
    response->set_ok(true);
    LOG(INFO) << "AllocWeightPages success (fallback): model_id=" << model_id
              << ", num_pages=" << num_pages;
  });
}

void ModelMemoryDistService::FreeWeightPages(
    ::google::protobuf::RpcController* controller,
    const proto::FreeWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    LOG(INFO) << "FreeWeightPages: model_id=" << model_id;

    // Free weight pages via ModelMemoryManager (frees pages in
    // PhysicalPagePool)
    auto& allocator = ModelMemoryManager::get_instance();
    size_t num_freed = allocator.free_weight(model_id);

    response->set_ok(num_freed > 0);

    LOG(INFO) << "FreeWeightPages: freed " << num_freed << " pages for model "
              << model_id;
  });
}

void ModelMemoryDistService::GetKVCacheOffsets(
    ::google::protobuf::RpcController* controller,
    const proto::GetKVCacheOffsetsRequest* request,
    proto::GetKVCacheOffsetsResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();
    uint64_t block_size_bytes = request->block_size_bytes();

    // Convert proto block_ids to vector
    std::vector<int32_t> block_ids;
    block_ids.reserve(request->block_ids_size());
    for (int i = 0; i < request->block_ids_size(); ++i) {
      block_ids.push_back(request->block_ids(i));
    }

    auto& allocator = ModelMemoryManager::get_instance();
    if (!allocator.is_initialized()) {
      LOG(ERROR) << "ModelMemoryManager not initialized on worker";
      return;
    }

    const std::optional<int64_t> model_num_layers =
        allocator.get_kv_cache_num_layers(model_id);
    if (!model_num_layers.has_value()) {
      LOG(ERROR) << "Model " << model_id << " not found in ModelMemoryManager";
      return;
    }

    const int64_t num_layers = model_num_layers.value();

    // Calculate offsets for each layer and each block
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      auto* layer_offsets_proto = response->add_layer_offsets();

      for (const auto& block_id : block_ids) {
        auto [k_offset, v_offset] = allocator.get_global_offsets_for_block(
            model_id, layer_id, block_id, block_size_bytes);

        if (k_offset == UINT64_MAX || v_offset == UINT64_MAX) {
          LOG(ERROR) << "Failed to get offsets for block " << block_id
                     << " at layer " << layer_id << " for model " << model_id;
          response->clear_layer_offsets();
          return;
        }

        layer_offsets_proto->add_k_offsets(k_offset);
        layer_offsets_proto->add_v_offsets(v_offset);
      }
    }

    VLOG(1) << "GetKVCacheOffsets: model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
  });
}

}  // namespace xllm
