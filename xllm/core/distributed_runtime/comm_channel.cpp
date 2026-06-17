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

#include "comm_channel.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <future>

#include "common/global_flags.h"

namespace xllm {

bool CommChannel::init_brpc(const std::string& server_address) {
  options_.connection_type = "pooled";
  options_.timeout_ms = -1;
  options_.connect_timeout_ms = -1;
  options_.max_retry = 3;

  if (channel_.Init(server_address.c_str(), "", &options_) != 0) {
    LOG(ERROR) << "Failed to initialize brpc Channel";
    return false;
  }

  stub_.reset(new proto::DistributeWorker_Stub(&channel_));
  return true;
}

bool CommChannel::hello() {
  proto::Status req;
  proto::Status resp;
  brpc::Controller cntl;

  cntl.Reset();
  stub_->Hello(&cntl, &req, &resp, nullptr);
  if (cntl.Failed() || !resp.ok()) {
    LOG(ERROR) << "Hello request failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::check_health() {
  proto::Status req;
  proto::Status resp;
  brpc::Controller cntl;

  // Set a timeout for health check
  // check hang status: 10min(magic num)
  cntl.set_timeout_ms(600000);
  stub_->Hello(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(WARNING) << "Health check failed: " << cntl.ErrorText();
    return false;
  }

  return resp.ok();
}

bool CommChannel::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  if (!kv_cache_shape.has_key_cache_shape() ||
      kv_cache_shape.key_cache_shape().empty()) {
    LOG(ERROR) << "allocate_kv_cache failed: key cache shape is empty";
    return false;
  }
  proto::AllocateKVCacheRequest request;
  kv_cache_shape.to_proto(request.mutable_kv_cache_shape());

  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateKVCache(&cntl, &request, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "allocate_kv_cache failed: " << cntl.ErrorText();
    return false;
  }

  return true;
}

bool CommChannel::get_cache_info(uint64_t& cluster_id,
                                 std::string& addr,
                                 uint16_t& port) {
  proto::Empty req;
  proto::CacheInfo resp;
  brpc::Controller cntl;

  stub_->GetCacheInfo(&cntl, &req, &resp, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetCacheInfo failed: " << cntl.ErrorText();
    return false;
  }

  cluster_id = resp.cluster_id();
  addr = resp.addr();
  port = static_cast<uint16_t>(resp.listen_port());
  return true;
}

bool CommChannel::link_cluster(const std::vector<uint64_t>& cluster_ids,
                               const std::vector<std::string>& addrs,
                               const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_ports()->Reserve(ports.size());

  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->LinkCluster(&cntl, &cluster_info, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "LinkCluster failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                 const std::vector<std::string>& addrs,
                                 const std::vector<uint16_t>& ports) {
  proto::ClusterInfo cluster_info;
  cluster_info.mutable_cluster_ids()->Reserve(cluster_ids.size());
  cluster_info.mutable_addrs()->Reserve(addrs.size());
  cluster_info.mutable_ports()->Reserve(ports.size());

  for (size_t i = 0; i < cluster_ids.size(); ++i) {
    cluster_info.add_cluster_ids(cluster_ids[i]);
    cluster_info.add_addrs(addrs[i]);
    cluster_info.add_ports(ports[i]);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->UnlinkCluster(&cntl, &cluster_info, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "UnlinkCluster failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::link_p2p(const std::string& remote_addr) {
  proto::P2PLinkWorkerRequest req;
  req.set_remote_addr(remote_addr);

  proto::Status s;
  brpc::Controller cntl;
  stub_->LinkP2P(&cntl, &req, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "LinkP2P failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::unlink_p2p(const std::string& remote_addr) {
  proto::P2PLinkWorkerRequest req;
  req.set_remote_addr(remote_addr);

  proto::Status s;
  brpc::Controller cntl;
  stub_->UnlinkP2P(&cntl, &req, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "UnlinkP2P failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::init_model(const std::string& model_weights_path,
                             int32_t random_seed,
                             MasterStatus master_status) {
  proto::InitModelRequest request;

  request.set_model_weights_path(model_weights_path);
  request.set_random_seed(random_seed);
  request.set_master_status(master_status.to_proto());
  proto::Status response;
  brpc::Controller cntl;
  stub_->InitModel(&cntl, &request, &response, nullptr);
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "init_model failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::init_model_async(const std::string& model_weights_path,
                                   int32_t random_seed,
                                   folly::Promise<bool>& promise,
                                   MasterStatus master_status) {
  proto::InitModelRequest request;

  request.set_model_weights_path(model_weights_path);
  request.set_random_seed(random_seed);
  request.set_master_status(master_status.to_proto());
  auto done = new InitModelClosure();
  done->promise = std::move(promise);
  stub_->InitModel(&done->cntl, &request, &done->response, done);

  return true;
}

bool CommChannel::estimate_kv_cache_capacity(int64_t& available_memory,
                                             int64_t& total_memory) {
  proto::Empty req;
  proto::DeviceMemory mem;
  brpc::Controller cntl;

  stub_->ProfileDeviceMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "estimate_kv_cache_capacity failed: " << cntl.ErrorText();
    return false;
  }

  available_memory = mem.available_memory();
  total_memory = mem.total_memory();
  return true;
}

bool CommChannel::pull_kv_blocks(
    const uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
  proto::PullKVCacheRequest request;
  request.set_cluster_id(src_cluster_id);
  request.set_addr(src_addr);

  ADD_VECTOR_TO_PROTO(request.mutable_src_blocks(), src_blocks);
  ADD_VECTOR_TO_PROTO(request.mutable_dst_blocks(), dst_blocks);
  ADD_VECTOR_TO_PROTO(request.mutable_src_linear_state_ids(),
                      src_linear_state_ids);
  ADD_VECTOR_TO_PROTO(request.mutable_dst_linear_state_ids(),
                      dst_linear_state_ids);

  proto::Status s;
  brpc::Controller cntl;
  stub_->PullKVCache(&cntl, &request, &s, nullptr);

  return !cntl.Failed() && s.ok();
}

void CommChannel::execute_model_async(
    const ForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_model_with_brpc(input, promise);
}

bool CommChannel::process_group_test() {
  proto::Empty req;
  proto::Status s;
  brpc::Controller cntl;

  stub_->ProcessGroupTest(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "process_group_test failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  if (!kv_cache_shape.has_key_cache_shape() ||
      kv_cache_shape.key_cache_shape().empty()) {
    LOG(ERROR)
        << "AllocateKVCacheWithTransfer failed: key cache shape is empty";
    return false;
  }
  proto::AllocateKVCacheRequest request;
  kv_cache_shape.to_proto(request.mutable_kv_cache_shape());

  proto::Status s;
  brpc::Controller cntl;
  stub_->AllocateKVCacheWithTransfer(&cntl, &request, &s, nullptr);

  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "AllocateKVCacheWithTransfer failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

void CommChannel::transfer_kv_blocks(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    folly::Promise<uint32_t>& promise) {
  proto::BlockTransferInfos pb_block_transfer_info;
  if (!block_transfer_info_to_proto(block_transfer_info,
                                    &pb_block_transfer_info)) {
    LOG(ERROR) << "transfer_kv_blocks fail: create proto fail!";
    promise.setValue(0);
    return;
  }

  auto done = new TransferBlocksClosure();
  done->promise = std::move(promise);
  stub_->TransferBlocks(
      &done->cntl, &pb_block_transfer_info, &done->response, done);
}

void CommChannel::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  proto::BlockTransferInfos pb_block_transfer_info;
  if (!block_transfer_info_to_proto(
          batch_id, block_transfer_info, &pb_block_transfer_info)) {
    LOG(ERROR) << "transfer_kv_blocks with batch id " << batch_id
               << " fail: create proto fail!";
    return;
  }
  brpc::Controller cntl;
  proto::TransferStatus response;
  stub_->TransferBlocks(&cntl, &pb_block_transfer_info, &response, nullptr);
}

bool CommChannel::sleep(MasterStatus master_status) {
  proto::SleepRequest req;
  proto::Status s;
  brpc::Controller cntl;

  req.set_master_status(master_status.to_proto());
  stub_->Sleep(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "Sleep failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::wakeup(const WakeupOptions& options) {
  proto::Status s;
  brpc::Controller cntl;
  proto::WakeupRequest req;

  req.set_master_status(options.master_status.to_proto());
  if (!options.remote_addrs.empty()) {
    req.mutable_remote_addrs()->Assign(options.remote_addrs.begin(),
                                       options.remote_addrs.end());
  }
  // Marshal weight segments (new multi-segment format)
  for (const auto& segments : options.src_weight_segments) {
    auto* seg_list = req.add_src_weight_segments();
    for (const auto& seg : segments) {
      auto* proto_seg = seg_list->add_segments();
      proto_seg->set_offset(seg.offset);
      proto_seg->set_size(seg.size);
    }
  }
  stub_->Wakeup(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "Wakeup failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::start_profile() {
  proto::Empty req;
  proto::Status s;
  brpc::Controller cntl;

  stub_->StartProfile(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "StartProfile failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::stop_profile() {
  proto::Empty req;
  proto::Status s;
  brpc::Controller cntl;

  stub_->StopProfile(&cntl, &req, &s, nullptr);
  if (cntl.Failed() || !s.ok()) {
    LOG(ERROR) << "StopProfile failed: " << cntl.ErrorText();
    return false;
  }
  return true;
}

class ClientStreamReceiver : public brpc::StreamInputHandler {
 private:
  std::shared_ptr<std::atomic<int32_t>> termination_flag_;
  std::shared_ptr<std::atomic<uint32_t>> success_cnt_;
  std::promise<void> close_promise_;
  std::atomic<bool> promise_set_{false};

 public:
  ClientStreamReceiver(std::shared_ptr<std::atomic<int32_t>> termination_flag,
                       std::shared_ptr<std::atomic<uint32_t>> success_cnt)
      : termination_flag_(termination_flag), success_cnt_(success_cnt) {}

  ~ClientStreamReceiver() {
    if (!promise_set_.exchange(true)) {
      close_promise_.set_value();
    }
  }

  std::future<void> get_close_future() { return close_promise_.get_future(); }

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    for (size_t i = 0; i < size; ++i) {
      std::string msg_str = messages[i]->to_string();
      int32_t success_cnt = std::stoi(msg_str);

      if (success_cnt > 0 &&
          termination_flag_->load(std::memory_order_acquire) > 0) {
        success_cnt_->fetch_add(success_cnt, std::memory_order_relaxed);
      } else {
        termination_flag_->fetch_sub(1, std::memory_order_release);
        if (!promise_set_.exchange(true)) {
          close_promise_.set_value();
        }
        break;
      }
    }
    return 0;
  }

  virtual void on_idle_timeout(brpc::StreamId id) override {
    if (!promise_set_.exchange(true)) {
      close_promise_.set_value();
    }
  }

  virtual void on_closed(brpc::StreamId id) override {
    if (!promise_set_.exchange(true)) {
      close_promise_.set_value();
    }
  }
};

void CommChannel::prefetch_from_storage(
    const std::vector<BlockTransferInfo>& block_transfer_info,
    std::shared_ptr<std::atomic<int32_t>> flag,
    std::shared_ptr<std::atomic<uint32_t>> success_cnt) {
  proto::BlockTransferInfos pb_block_transfer_info;
  if (!block_transfer_info_to_proto(block_transfer_info,
                                    &pb_block_transfer_info)) {
    LOG(ERROR) << "prefetch_from_storage fail: create proto fail!";
    return;
  }
  ClientStreamReceiver receiver(flag, success_cnt);
  brpc::Controller cntl;
  brpc::StreamOptions stream_options;
  brpc::StreamId stream_id;
  proto::Status response;
  stream_options.handler = &receiver;
  stream_options.idle_timeout_ms = 30;
  if (brpc::StreamCreate(&stream_id, cntl, &stream_options) != 0) {
    LOG(ERROR) << "Failed to create stream";
    return;
  }

  stub_->PrefetchFromStorage(
      &cntl, &pb_block_transfer_info, &response, nullptr);

  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "Fail to connect stream, " << cntl.ErrorText();
    return;
  }

  receiver.get_close_future().wait();
  brpc::StreamClose(stream_id);
}

bool CommChannel::get_last_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::Empty req;
  proto::ForwardOutput pb_output;
  brpc::Controller cntl;
  stub_->GetLastStepResult(&cntl, &req, &pb_output, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "Get last step model output result failed, "
               << cntl.ErrorText();
    return false;
  }

  // parse tokens
  RawForwardOutput raw_forward_output;
  proto_to_forward_output(pb_output, raw_forward_output);
  promise.setValue(std::move(raw_forward_output));

  return true;
}

bool CommChannel::get_active_activation_memory(int64_t& memory) {
  proto::Empty req;
  proto::ActivationMemory mem;
  brpc::Controller cntl;

  stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "GetActiveActivationMemory failed: " << cntl.ErrorText();
    return false;
  }

  memory = mem.active_activation_memory();
  return true;
}

bool CommChannel::get_active_activation_memory_async(
    folly::Promise<int64_t>& promise) {
  proto::Empty req;
  proto::ActivationMemory mem;
  brpc::Controller cntl;

  stub_->GetActiveActivationMemory(&cntl, &req, &mem, nullptr);
  if (cntl.Failed()) {
    LOG(ERROR) << "get_active_activation_memory_async failed: "
               << cntl.ErrorText();
    promise.setValue(0);
    return false;
  }
  promise.setValue(mem.active_activation_memory());
  return true;
}

bool CommChannel::execute_model_with_brpc(
    const ForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::ForwardInput pb_forward_input;
  auto* packed_input = pb_forward_input.mutable_packed_input();
  if (!forward_input_to_packed_proto(input, packed_input)) {
    LOG(ERROR) << "failed to pack ForwardInput for remote execution";
    promise.setValue(std::optional<RawForwardOutput>(std::nullopt));
    return false;
  }

  auto done = new ExecuteModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteModel(&done->cntl, &pb_forward_input, &done->pb_output, done);
  return true;
}

void ExecuteModelClosure::Run() {
  std::unique_ptr<ExecuteModelClosure> self_guard(this);

  if (cntl.Failed()) {
    LOG(ERROR) << "Execute_model_async failed. Error code : "
               << cntl.ErrorCode() << ", error message : " << cntl.ErrorText();
  }

  RawForwardOutput raw_forward_output;
  proto_to_forward_output(pb_output, raw_forward_output);
  promise.setValue(raw_forward_output);

  return;
}

void InitModelClosure::Run() {
  std::unique_ptr<InitModelClosure> self_guard(this);

  bool success = !cntl.Failed() && response.ok();
  if (!success) {
    LOG(ERROR) << "Init_model_async failed, " << cntl.ErrorText();
  } else {
    LOG(INFO) << "Init_model_async succeed.";
  }
  promise.setValue(success);

  return;
}

void TransferBlocksClosure::Run() {
  std::unique_ptr<TransferBlocksClosure> self_guard(this);

  bool success = !cntl.Failed();
  if (!success) {
    promise.setValue(0);
  } else {
    promise.setValue(response.success_cnt());
  }
  return;
}

}  // namespace xllm
