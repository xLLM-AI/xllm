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

#include "core/distributed_runtime/worker_memory_rpc_client.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>
#include <stdexcept>
#include <thread>
#include <utility>

#include "core/framework/config/service_config.h"

namespace xllm {
namespace {

constexpr int32_t kDefaultTransactionTimeoutMs = 30000;

int32_t get_transaction_timeout_ms() {
  const int32_t timeout_ms =
      ServiceConfig::get_instance().rpc_channel_timeout_ms();
  // Ownership transactions need a deadline even when other RPCs are unbounded.
  return timeout_ms > 0 ? timeout_ms : kDefaultTransactionTimeoutMs;
}

}  // namespace

WorkerMemoryRpcClient::WorkerMemoryRpcClient(int32_t global_rank,
                                             const std::string& server_address,
                                             const torch::Device& device)
    : global_rank_(global_rank),
      device_(device),
      transaction_timeout_ms_(get_transaction_timeout_ms()) {
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;
  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc channel to " << server_address;
    return;
  }
  stub_.reset(new proto::ModelMemoryDist_Stub(&channel_));
  wait_for_server_ready(server_address);
}

bool WorkerMemoryRpcClient::wait_for_server_ready(
    const std::string& server_address) {
  proto::Status req;
  proto::Status resp;

  int32_t try_count = 0;
  brpc::Controller cntl;
  constexpr int32_t kReconnectDelaySeconds = 3;
  while (try_count <
         ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    cntl.Reset();
    stub_->Hello(&cntl, &req, &resp, nullptr);
    if (cntl.Failed() || !resp.ok()) {
      std::this_thread::sleep_for(std::chrono::seconds(kReconnectDelaySeconds));
    } else {
      LOG(INFO) << "WorkerMemoryRpcClient connected to server: "
                << server_address << ", global_rank: " << global_rank_;
      break;
    }
    try_count++;
  }
  if (try_count >=
      ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    LOG(ERROR) << "WorkerMemoryRpcClient Hello failed, global_rank: "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }
  return true;
}

folly::SemiFuture<MemoryInfo> WorkerMemoryRpcClient::get_memory_info_async(
    int32_t timeout_ms) {
  folly::Promise<MemoryInfo> promise;
  auto future = promise.getSemiFuture();
  capacity_query_threadpool_.schedule(
      [this, timeout_ms, promise = std::move(promise)]() mutable {
        proto::Status req;
        proto::MemoryInfoResponse resp;
        brpc::Controller cntl;
        cntl.set_timeout_ms(timeout_ms);
        stub_->GetMemoryInfo(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "GetMemoryInfo failed: " << cntl.ErrorText();
          promise.setException(std::runtime_error(cntl.ErrorText()));
          return;
        }
        promise.setValue(MemoryInfo{
            resp.available_memory(),
            resp.total_memory(),
            resp.pool_initialized(),
            static_cast<size_t>(resp.pool_capacity_pages()),
            static_cast<size_t>(resp.pool_committed_pages()),
            static_cast<size_t>(resp.pool_allocatable_pages()),
            static_cast<size_t>(resp.execution_memory_reserve_bytes()),
            static_cast<size_t>(resp.pool_active_pages()),
            static_cast<size_t>(resp.growth_memory_reserve_bytes())});
      });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::init_physical_page_pool_async(
    int64_t num_pages,
    bool lazy) {
  return init_physical_page_pool_async(
      num_pages, num_pages, lazy, /*growth_memory_reserve_bytes=*/0);
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::init_physical_page_pool_async(
    int64_t max_pages,
    int64_t initial_pages,
    bool lazy,
    size_t growth_memory_reserve_bytes) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        max_pages,
                        initial_pages,
                        lazy,
                        growth_memory_reserve_bytes,
                        promise = std::move(promise)]() mutable {
    proto::InitPhysicalPagePoolRequest req;
    req.set_num_pages(max_pages);
    req.set_lazy(lazy);
    req.set_initial_pages(initial_pages);
    req.set_growth_memory_reserve_bytes(growth_memory_reserve_bytes);
    proto::Status resp;
    brpc::Controller cntl;
    stub_->InitPhysicalPagePool(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "InitPhysicalPagePool failed: " << cntl.ErrorText();
      promise.setValue(false);
      return;
    }
    promise.setValue(resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::prepare_kv_mapping_async(
    uint64_t transaction_id,
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        transaction_id,
                        model_id,
                        offsets,
                        promise = std::move(promise)]() mutable {
    proto::PrepareKvMappingRequest req;
    req.set_transaction_id(transaction_id);
    req.set_model_id(model_id);
    for (offset_t offset : offsets) {
      if (offset < 0) {
        promise.setValue(false);
        return;
      }
      req.add_offsets(static_cast<uint64_t>(offset));
    }
    proto::Status resp;
    brpc::Controller cntl;
    cntl.set_timeout_ms(transaction_timeout_ms_);
    stub_->PrepareKvMapping(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "PrepareKvMapping failed: " << cntl.ErrorText();
    }
    promise.setValue(!cntl.Failed() && resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::commit_kv_mapping_async(
    uint64_t transaction_id) {
  return finish_transaction_async(
      transaction_id, &proto::ModelMemoryDist_Stub::CommitKvMapping);
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::abort_kv_mapping_async(
    uint64_t transaction_id) {
  return finish_transaction_async(transaction_id,
                                  &proto::ModelMemoryDist_Stub::AbortKvMapping);
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::prepare_weight_pages_async(
    uint64_t transaction_id,
    const std::string& model_id,
    size_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        transaction_id,
                        model_id,
                        num_pages,
                        promise = std::move(promise)]() mutable {
    proto::PrepareWeightPagesRequest req;
    req.set_transaction_id(transaction_id);
    req.set_model_id(model_id);
    req.set_num_pages(num_pages);
    proto::Status resp;
    brpc::Controller cntl;
    cntl.set_timeout_ms(transaction_timeout_ms_);
    stub_->PrepareWeightPages(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "PrepareWeightPages failed: " << cntl.ErrorText();
    }
    promise.setValue(!cntl.Failed() && resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::commit_weight_pages_async(
    uint64_t transaction_id) {
  return finish_transaction_async(
      transaction_id, &proto::ModelMemoryDist_Stub::CommitWeightPages);
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::abort_weight_pages_async(
    uint64_t transaction_id) {
  return finish_transaction_async(
      transaction_id, &proto::ModelMemoryDist_Stub::AbortWeightPages);
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::finish_transaction_async(
    uint64_t transaction_id,
    TransactionRpc rpc) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, transaction_id, rpc, promise = std::move(promise)]() mutable {
        proto::KvMappingTransactionRequest req;
        req.set_transaction_id(transaction_id);
        proto::Status resp;
        brpc::Controller cntl;
        cntl.set_timeout_ms(transaction_timeout_ms_);
        (stub_.get()->*rpc)(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "Worker memory transaction completion failed: "
                     << cntl.ErrorText();
        }
        promise.setValue(!cntl.Failed() && resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::map_to_kv_tensors_async(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_id,
                        offsets = offsets,
                        promise = std::move(promise)]() mutable {
    proto::KvTensorRequest req;
    req.set_model_id(model_id);
    for (offset_t offset : offsets) {
      req.add_offsets(offset);
    }
    proto::Status resp;
    brpc::Controller cntl;
    stub_->MapToKvTensors(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "MapToKvTensors failed: " << cntl.ErrorText();
      promise.setValue(false);
      return;
    }
    promise.setValue(resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::unmap_from_kv_tensors_async(
    const std::string& model_id,
    const std::vector<offset_t>& offsets) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_id,
                        offsets = offsets,
                        promise = std::move(promise)]() mutable {
    proto::KvTensorRequest req;
    req.set_model_id(model_id);
    for (offset_t offset : offsets) {
      req.add_offsets(offset);
    }
    proto::Status resp;
    brpc::Controller cntl;
    stub_->UnmapFromKvTensors(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "UnmapFromKvTensors failed: " << cntl.ErrorText();
      promise.setValue(false);
      return;
    }
    promise.setValue(resp.ok());
  });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::alloc_weight_pages_async(
    const std::string& model_id,
    size_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_id, num_pages, promise = std::move(promise)]() mutable {
        proto::AllocWeightPagesRequest req;
        req.set_model_id(model_id);
        req.set_num_pages(num_pages);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->AllocWeightPages(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "AllocWeightPages failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> WorkerMemoryRpcClient::free_weight_pages_async(
    const std::string& model_id) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, model_id, promise = std::move(promise)]() mutable {
        proto::FreeWeightPagesRequest req;
        req.set_model_id(model_id);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->FreeWeightPages(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "FreeWeightPages failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

std::vector<VirtualKVCacheLayerGlobalOffsets>
WorkerMemoryRpcClient::get_virtual_kv_cache_offsets(
    const std::string& model_id,
    const std::vector<int32_t>& block_ids) {
  proto::GetVirtualKVCacheOffsetsRequest req;
  req.set_model_id(model_id);
  req.mutable_block_ids()->Reserve(static_cast<int32_t>(block_ids.size()));
  for (int32_t block_id : block_ids) {
    req.add_block_ids(block_id);
  }

  proto::GetVirtualKVCacheOffsetsResponse resp;
  brpc::Controller cntl;
  stub_->GetVirtualKVCacheOffsets(&cntl, &req, &resp, brpc::DoNothing());
  brpc::Join(cntl.call_id());
  if (cntl.Failed()) {
    LOG(ERROR) << "GetVirtualKVCacheOffsets failed: " << cntl.ErrorText();
    return {};
  }

  std::vector<VirtualKVCacheLayerGlobalOffsets> layer_offsets;
  layer_offsets.reserve(resp.layer_offsets_size());
  for (int32_t i = 0; i < resp.layer_offsets_size(); ++i) {
    const auto& layer_proto = resp.layer_offsets(i);
    const size_t num_blocks = block_ids.size();
    if (static_cast<size_t>(layer_proto.k_offsets_size()) != num_blocks ||
        static_cast<size_t>(layer_proto.v_offsets_size()) != num_blocks ||
        (layer_proto.index_offsets_size() != 0 &&
         static_cast<size_t>(layer_proto.index_offsets_size()) != num_blocks)) {
      LOG(ERROR) << "Invalid virtual-memory offset count at layer " << i;
      return {};
    }
    VirtualKVCacheLayerGlobalOffsets layer;
    layer.k_offsets.assign(layer_proto.k_offsets().begin(),
                           layer_proto.k_offsets().end());
    layer.v_offsets.assign(layer_proto.v_offsets().begin(),
                           layer_proto.v_offsets().end());
    layer.index_offsets.assign(layer_proto.index_offsets().begin(),
                               layer_proto.index_offsets().end());
    layer_offsets.emplace_back(std::move(layer));
  }
  return layer_offsets;
}

}  // namespace xllm
