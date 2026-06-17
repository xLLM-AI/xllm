/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "xtensor_dist_client.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <chrono>
#include <thread>

#include "common/global_flags.h"
#include "core/framework/config/service_config.h"

namespace xllm {

XTensorDistClient::XTensorDistClient(int32_t global_rank,
                                     const std::string& server_address,
                                     const torch::Device& device)
    : global_rank_(global_rank), device_(device) {
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;
  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc channel to " << server_address;
    return;
  }
  stub_.reset(new proto::XTensorDist_Stub(&channel_));
  wait_for_server_ready(server_address);
}

bool XTensorDistClient::wait_for_server_ready(
    const std::string& server_address) {
  proto::Status req;
  proto::Status resp;

  int try_count = 0;
  brpc::Controller cntl;
  const int sleep_time_second = 3;
  while (try_count <
         ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    cntl.Reset();
    stub_->Hello(&cntl, &req, &resp, nullptr);
    if (cntl.Failed() || !resp.ok()) {
      std::this_thread::sleep_for(std::chrono::seconds(sleep_time_second));
    } else {
      LOG(INFO) << "XTensorDistClient connected to server: " << server_address
                << ", global_rank: " << global_rank_;
      break;
    }
    try_count++;
  }
  if (try_count >=
      ::xllm::ServiceConfig::get_instance().max_reconnect_count()) {
    LOG(ERROR) << "XTensorDistClient Hello failed, global_rank: "
               << global_rank_ << ", error: " << cntl.ErrorText();
    return false;
  }
  return true;
}

folly::SemiFuture<MemoryInfo> XTensorDistClient::get_memory_info_async() {
  folly::Promise<MemoryInfo> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    proto::Status req;
    proto::MemoryInfoResponse resp;
    brpc::Controller cntl;
    stub_->GetMemoryInfo(&cntl, &req, &resp, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "GetMemoryInfo failed: " << cntl.ErrorText();
      promise.setValue(MemoryInfo{0, 0});
      return;
    }
    // Returns 0 for both fields on failure
    promise.setValue(MemoryInfo{resp.available_memory(), resp.total_memory()});
  });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::init_phy_page_pool_async(
    int64_t num_pages) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, num_pages, promise = std::move(promise)]() mutable {
        proto::InitPhyPagePoolRequest req;
        req.set_num_pages(num_pages);
        proto::Status resp;
        brpc::Controller cntl;
        stub_->InitPhyPagePool(&cntl, &req, &resp, nullptr);
        if (cntl.Failed()) {
          LOG(ERROR) << "InitPhyPagePool failed: " << cntl.ErrorText();
          promise.setValue(false);
          return;
        }
        promise.setValue(resp.ok());
      });
  return future;
}

folly::SemiFuture<bool> XTensorDistClient::map_to_kv_tensors_async(
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

folly::SemiFuture<bool> XTensorDistClient::unmap_from_kv_tensors_async(
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

folly::SemiFuture<bool> XTensorDistClient::alloc_weight_pages_async(
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

folly::SemiFuture<bool> XTensorDistClient::free_weight_pages_async(
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

folly::SemiFuture<
    std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>>
XTensorDistClient::get_xtensor_offsets_async(
    const std::string& model_id,
    const std::vector<int32_t>& block_ids,
    uint64_t block_size_bytes) {
  using ResultType =
      std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>;
  folly::Promise<ResultType> promise;
  auto future = promise.getSemiFuture();

  threadpool_.schedule([this,
                        model_id,
                        block_ids = block_ids,
                        block_size_bytes,
                        promise = std::move(promise)]() mutable {
    proto::GetXTensorOffsetsRequest req;
    req.set_model_id(model_id);
    for (int32_t block_id : block_ids) {
      req.add_block_ids(block_id);
    }
    req.set_block_size_bytes(block_size_bytes);

    proto::GetXTensorOffsetsResponse resp;
    brpc::Controller cntl;
    stub_->GetXTensorOffsets(&cntl, &req, &resp, nullptr);

    if (cntl.Failed()) {
      LOG(ERROR) << "GetXTensorOffsets failed: " << cntl.ErrorText();
      promise.setValue(ResultType{});
      return;
    }

    // Convert proto response to vector of pairs
    ResultType layer_offsets;
    layer_offsets.reserve(resp.layer_offsets_size());

    for (int i = 0; i < resp.layer_offsets_size(); ++i) {
      const auto& layer_proto = resp.layer_offsets(i);
      std::vector<uint64_t> k_offsets(layer_proto.k_offsets().begin(),
                                      layer_proto.k_offsets().end());
      std::vector<uint64_t> v_offsets(layer_proto.v_offsets().begin(),
                                      layer_proto.v_offsets().end());
      layer_offsets.emplace_back(std::move(k_offsets), std::move(v_offsets));
    }

    promise.setValue(std::move(layer_offsets));
  });

  return future;
}

}  // namespace xllm
