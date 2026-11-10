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

#include "core/distributed_runtime/worker_memory_service.h"

#include <brpc/closure_guard.h>
#include <brpc/controller.h>
#include <glog/logging.h>

#include <limits>
#include <optional>
#include <vector>

#include "core/platform/device.h"
#include "core/runtime/worker_memory_resources.h"

namespace xllm {

WorkerMemoryService::WorkerMemoryService(int32_t global_rank,
                                         int32_t world_size,
                                         const torch::Device& device)
    : initialized_(false),
      global_rank_(global_rank),
      world_size_(world_size),
      device_(device),
      threadpool_(/*num_threads=*/4,
                  [device]() { Device(device).set_device(); },
                  /*cpu_binding=*/false,
                  /*pool_name=*/"WorkerMemoryService.rpc") {}

void WorkerMemoryService::Hello(::google::protobuf::RpcController* controller,
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

void WorkerMemoryService::GetMemoryInfo(
    ::google::protobuf::RpcController* controller,
    const proto::Status* request,
    proto::MemoryInfoResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    Device device(device_);
    device.set_device();

    const WorkerMemoryCapacity capacity =
        WorkerMemoryResources::get_instance().memory_capacity_snapshot();
    const int64_t available_memory = capacity.available_memory;
    const int64_t total_memory = capacity.total_memory;

    // Live admission queries this on the serving path; keep snapshots out of
    // the default per-request logs.
    VLOG(1) << "GetMemoryInfo: global_rank=" << global_rank_
            << ", available_memory=" << available_memory
            << ", total_memory=" << total_memory;

    // Returns 0 for both fields on failure (handled by caller)
    response->set_available_memory(available_memory);
    response->set_total_memory(total_memory);
    response->set_pool_initialized(capacity.pool_initialized);
    response->set_pool_capacity_pages(capacity.pool_capacity_pages);
    response->set_pool_committed_pages(capacity.pool_committed_pages);
    response->set_pool_allocatable_pages(capacity.pool_allocatable_pages);
    response->set_execution_memory_reserve_bytes(
        capacity.execution_memory_reserve_bytes);
    response->set_pool_active_pages(capacity.pool_active_pages);
    response->set_growth_memory_reserve_bytes(
        capacity.growth_memory_reserve_bytes);
  });
}

void WorkerMemoryService::InitPhysicalPagePool(
    ::google::protobuf::RpcController* controller,
    const proto::InitPhysicalPagePoolRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    const int64_t num_pages = request->num_pages();
    const int64_t initial_pages =
        request->has_initial_pages() ? request->initial_pages() : num_pages;
    if (num_pages <= 0 || initial_pages < 0 || initial_pages > num_pages ||
        (!request->lazy() && initial_pages != num_pages)) {
      response->set_ok(false);
      return;
    }
    LOG(INFO) << "InitPhysicalPagePool: global_rank=" << global_rank_
              << ", num_pages=" << num_pages
              << ", initial_pages=" << initial_pages;

    WorkerMemoryResources::get_instance().init_physical_page_pool(
        num_pages,
        initial_pages,
        request->lazy(),
        static_cast<size_t>(request->growth_memory_reserve_bytes()));
    LOG(INFO) << "GlobalVirtualMemoryRegion initialized on worker "
              << global_rank_;

    response->set_ok(true);
  });
}

void WorkerMemoryService::PrepareKvMapping(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::PrepareKvMappingRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    std::vector<offset_t> offsets;
    offsets.reserve(request->offsets_size());
    for (uint64_t offset : request->offsets()) {
      if (offset >
          static_cast<uint64_t>(std::numeric_limits<offset_t>::max())) {
        response->set_ok(false);
        return;
      }
      offsets.emplace_back(static_cast<offset_t>(offset));
    }
    response->set_ok(WorkerMemoryResources::get_instance().prepare_kv_mapping(
        request->transaction_id(), request->model_id(), offsets));
  });
}

void WorkerMemoryService::CommitKvMapping(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::KvMappingTransactionRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(WorkerMemoryResources::get_instance().commit_kv_mapping(
        request->transaction_id()));
  });
}

void WorkerMemoryService::AbortKvMapping(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::KvMappingTransactionRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(WorkerMemoryResources::get_instance().abort_kv_mapping(
        request->transaction_id()));
  });
}

void WorkerMemoryService::PrepareWeightPages(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::PrepareWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    if (request->num_pages() > std::numeric_limits<size_t>::max()) {
      response->set_ok(false);
      return;
    }
    response->set_ok(WorkerMemoryResources::get_instance().prepare_weight_pages(
        request->transaction_id(),
        request->model_id(),
        static_cast<size_t>(request->num_pages())));
  });
}

void WorkerMemoryService::CommitWeightPages(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::KvMappingTransactionRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(WorkerMemoryResources::get_instance().commit_weight_pages(
        request->transaction_id()));
  });
}

void WorkerMemoryService::AbortWeightPages(
    ::google::protobuf::RpcController* /*controller*/,
    const proto::KvMappingTransactionRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([request, response, done]() {
    brpc::ClosureGuard done_guard(done);
    response->set_ok(WorkerMemoryResources::get_instance().abort_weight_pages(
        request->transaction_id()));
  });
}

void WorkerMemoryService::MapToKvTensors(
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
    for (int32_t i = 0; i < request->offsets_size(); ++i) {
      offsets.emplace_back(request->offsets(i));
    }

    auto& resources = WorkerMemoryResources::get_instance();
    bool success = resources.map_to_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void WorkerMemoryService::UnmapFromKvTensors(
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
    for (int32_t i = 0; i < request->offsets_size(); ++i) {
      offsets.emplace_back(request->offsets(i));
    }

    auto& resources = WorkerMemoryResources::get_instance();
    bool success = resources.unmap_from_kv_tensors(model_id, offsets);
    response->set_ok(success);
  });
}

void WorkerMemoryService::AllocWeightPages(
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

    auto& resources = WorkerMemoryResources::get_instance();
    response->set_ok(resources.alloc_weight_pages(model_id, num_pages));
  });
}

void WorkerMemoryService::FreeWeightPages(
    ::google::protobuf::RpcController* controller,
    const proto::FreeWeightPagesRequest* request,
    proto::Status* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    LOG(INFO) << "FreeWeightPages: model_id=" << model_id;

    // Release the model's weight pages back to the worker's PhysicalPagePool.
    auto& resources = WorkerMemoryResources::get_instance();
    size_t num_freed = resources.free_weight(model_id);

    // Free is idempotent: a successful earlier RPC may have lost its reply.
    response->set_ok(true);

    LOG(INFO) << "FreeWeightPages: freed " << num_freed << " pages for model "
              << model_id;
  });
}

void WorkerMemoryService::GetVirtualKVCacheOffsets(
    ::google::protobuf::RpcController* controller,
    const proto::GetVirtualKVCacheOffsetsRequest* request,
    proto::GetVirtualKVCacheOffsetsResponse* response,
    ::google::protobuf::Closure* done) {
  threadpool_.schedule([this, request, response, done]() mutable {
    brpc::ClosureGuard done_guard(done);

    std::string model_id = request->model_id();

    // Convert proto block_ids to vector
    std::vector<int32_t> block_ids;
    block_ids.reserve(request->block_ids_size());
    for (int32_t i = 0; i < request->block_ids_size(); ++i) {
      block_ids.emplace_back(request->block_ids(i));
    }

    auto& resources = WorkerMemoryResources::get_instance();
    if (!resources.is_initialized()) {
      LOG(ERROR) << "WorkerMemoryResources not initialized on worker";
      return;
    }

    const std::optional<int64_t> model_num_layers =
        resources.get_kv_cache_num_layers(model_id);
    if (!model_num_layers.has_value()) {
      LOG(ERROR) << "Model " << model_id
                 << " not found in WorkerMemoryResources";
      return;
    }

    const int64_t num_layers = model_num_layers.value();

    // Calculate offsets for each layer and each block
    for (int64_t layer_id = 0; layer_id < num_layers; ++layer_id) {
      auto* layer_offsets_proto = response->add_layer_offsets();

      for (const auto& block_id : block_ids) {
        const auto offsets = resources.get_global_offsets_for_block(
            model_id, layer_id, block_id);

        if (!offsets.has_value()) {
          LOG(ERROR) << "Failed to get offsets for block " << block_id
                     << " at layer " << layer_id << " for model " << model_id;
          response->clear_layer_offsets();
          return;
        }

        layer_offsets_proto->add_k_offsets(offsets->k_offset);
        layer_offsets_proto->add_v_offsets(offsets->v_offset);
        if (offsets->index_offset.has_value()) {
          layer_offsets_proto->add_index_offsets(offsets->index_offset.value());
        }
      }
    }

    VLOG(1) << "GetVirtualKVCacheOffsets: model_id=" << model_id
            << ", num_blocks=" << block_ids.size()
            << ", num_layers=" << num_layers;
  });
}

}  // namespace xllm
