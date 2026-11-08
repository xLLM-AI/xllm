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

#include "comm_channel.h"

#include <brpc/controller.h>
#include <glog/logging.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>

#include "common/global_flags.h"
#include "core/distributed_runtime/worker_model_codec.h"
#include "core/framework/speculative/adaptive/spec_startup_calibration.h"
#include "core/kv_cache/transfer/kv_transport_codec.h"
#include "runtime/rec_forward_params.h"
#include "runtime/vlm_forward_params.h"

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

std::optional<std::string> CommChannel::spec_calibration(
    const SpecCalibrationCommand& command) {
  proto::DSparkCalibrationRequest request;
  request.set_action(
      static_cast<proto::DSparkCalibrationRequest::Action>(command.action));
  request.set_payload(command.payload);
  proto::DSparkCalibrationResponse response;
  brpc::Controller controller;
  stub_->DSparkCalibration(&controller, &request, &response, nullptr);
  if (controller.Failed() || !response.ok()) {
    LOG(ERROR) << "DSpark startup calibration control failed: "
               << controller.ErrorText();
    return std::nullopt;
  }
  return response.payload();
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
    const std::vector<KVTransferMapping>& mappings) {
  proto::PullKVCacheRequest request;
  request.set_cluster_id(src_cluster_id);
  request.set_addr(src_addr);

  for (const KVTransferMapping& mapping : mappings) {
    proto::KVTransferMapping* proto_mapping = request.add_mappings();
    proto_mapping->set_group_id(mapping.group_id);
    ADD_VECTOR_TO_PROTO(proto_mapping->mutable_local_ids(), mapping.local_ids);
    ADD_VECTOR_TO_PROTO(proto_mapping->mutable_remote_ids(),
                        mapping.remote_ids);
  }

  proto::Status s;
  brpc::Controller cntl;
  stub_->PullKVCache(&cntl, &request, &s, nullptr);

  return !cntl.Failed() && s.ok();
}

void CommChannel::execute_llm_model_async(
    const LlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_llm_model_with_brpc(input, promise);
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
  CHECK(block_transfer_info_to_proto(
      batch_id, block_transfer_info, &pb_block_transfer_info))
      << "Failed to serialize H2D transfer for batch_id=" << batch_id;
  brpc::Controller cntl;
  proto::TransferStatus response;
  stub_->TransferBlocks(&cntl, &pb_block_transfer_info, &response, nullptr);
  CHECK(!cntl.Failed()) << "H2D transfer registration RPC failed for batch_id="
                        << batch_id << ": " << cntl.ErrorText();
  CHECK_EQ(response.success_cnt(), block_transfer_info.size())
      << "H2D transfer registration was not acknowledged for batch_id="
      << batch_id;
}

bool CommChannel::wait_for_kv_blocks_transfer(uint64_t batch_id) {
  proto::WaitForKVBlocksTransferRequest request;
  request.set_batch_id(batch_id);
  proto::Status response;
  brpc::Controller controller;
  stub_->WaitForKVBlocksTransfer(&controller, &request, &response, nullptr);
  if (controller.Failed() || !response.ok()) {
    LOG(ERROR) << "Host KV restore completion RPC failed for batch_id="
               << batch_id << ": " << controller.ErrorText();
    return false;
  }
  return true;
}

bool CommChannel::sleep(MasterStatus master_status, uint64_t transition_id) {
  proto::SleepRequest req;
  proto::Status s;
  brpc::Controller cntl;

  req.set_master_status(master_status.to_proto());
  req.set_transition_id(transition_id);
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
  req.set_transition_id(options.transition_id);
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

class ClientStreamReceiver final : public brpc::StreamInputHandler {
 private:
  std::shared_ptr<PrefetchResult> result_;
  std::vector<uint8_t> gated_hits_;
  std::vector<uint8_t> non_gated_hits_;
  size_t worker_index_ = 0;
  bool stop_sent_ = false;
  bool failed_ = false;

  void fail_and_close(brpc::StreamId id) {
    failed_ = true;
    result_->mark_worker_failed();
    brpc::StreamClose(id);
  }

  bool send_control(brpc::StreamId id, PrefetchControl control) {
    const uint8_t control_byte = static_cast<uint8_t>(control);
    butil::IOBuf response;
    response.append(&control_byte, sizeof(control_byte));
    if (brpc::StreamWrite(id, response) == 0) {
      stop_sent_ = control == PrefetchControl::STOP;
      return true;
    }
    LOG(ERROR) << "Failed to write Mooncake prefetch decision: worker="
               << worker_index_;
    fail_and_close(id);
    return false;
  }

 public:
  ClientStreamReceiver(std::shared_ptr<PrefetchResult> result,
                       size_t worker_index)
      : result_(std::move(result)), worker_index_(worker_index) {
    CHECK(result_ != nullptr);
    CHECK_LT(worker_index_, result_->worker_count());
    gated_hits_.resize(result_->batch_wire_size());
    non_gated_hits_.resize(result_->batch_wire_size());
  }

  int on_received_messages(brpc::StreamId id,
                           butil::IOBuf* const messages[],
                           size_t size) override {
    if (stop_sent_) {
      fail_and_close(id);
      return -1;
    }
    const size_t batch_size = result_->batch_wire_size();
    // The worker pads a short final batch to the configured batch size. The
    // result object validates the configured size and ignores padding units.
    const size_t expected_bytes = batch_size * 2;
    if (size != 1 || messages[0]->length() != expected_bytes) {
      LOG(ERROR) << "Invalid Mooncake prefetch result frame: worker="
                 << worker_index_ << ", messages=" << size
                 << ", bytes=" << (size == 1 ? messages[0]->length() : 0)
                 << ", expected_bytes=" << expected_bytes;
      fail_and_close(id);
      return -1;
    }

    messages[0]->copy_to(gated_hits_.data(), gated_hits_.size());
    messages[0]->copy_to(
        non_gated_hits_.data(), non_gated_hits_.size(), gated_hits_.size());
    const std::optional<PrefetchControl> control = result_->record_batch_result(
        worker_index_, gated_hits_, non_gated_hits_);
    if (!control.has_value()) {
      LOG(ERROR) << "Unexpected Mooncake prefetch result: worker="
                 << worker_index_;
      fail_and_close(id);
      return -1;
    }
    if (!send_control(id, *control)) {
      return -1;
    }
    return 0;
  }

  void on_idle_timeout(brpc::StreamId id) override {
    if (stop_sent_) {
      brpc::StreamClose(id);
      return;
    }
    fail_and_close(id);
  }

  void on_failed(brpc::StreamId /*id*/,
                 int /*error_code*/,
                 const std::string& /*error_text*/) override {
    failed_ = true;
    result_->mark_worker_failed();
  }

  void on_closed(brpc::StreamId /*id*/) override {
    // STOP is sent only after a completed batch result. A transport close
    // without STOP does not certify that the remote write has finished.
    result_->mark_worker_ended(
        worker_index_, stop_sent_ && !failed_, /*worker_quiescent=*/stop_sent_);
    delete this;
  }
};

void CommChannel::prefetch_from_storage(const StoragePrefetchRequest& request,
                                        std::shared_ptr<PrefetchResult> result,
                                        size_t worker_index) {
  CHECK(result != nullptr);
  proto::PrefetchRequest proto_request;
  if (!request.to_proto(&proto_request)) {
    LOG(ERROR) << "Failed to serialize Mooncake prefetch request.";
    result->mark_worker_ended(
        worker_index, /*worker_ok=*/false, /*worker_quiescent=*/true);
    return;
  }

  proto_request.set_batch_size(result->batch_wire_size());

  auto* receiver = new ClientStreamReceiver(result, worker_index);
  brpc::Controller cntl;
  brpc::StreamOptions stream_options;
  brpc::StreamId stream_id;
  proto::Status response;
  stream_options.handler = receiver;
  stream_options.idle_timeout_ms = result->stream_idle_timeout_ms();
  result->begin_worker_rpc(worker_index);
  if (brpc::StreamCreate(&stream_id, cntl, &stream_options) != 0) {
    LOG(ERROR) << "Failed to create Mooncake prefetch stream.";
    delete receiver;
    result->complete_worker_rpc(worker_index, /*rejected_before_start=*/true);
    return;
  }

  stub_->PrefetchFromStorage(&cntl, &proto_request, &response, nullptr);
  result->complete_worker_rpc(
      worker_index, /*rejected_before_start=*/!cntl.Failed() && !response.ok());
  if (cntl.Failed() || !response.ok()) {
    LOG(ERROR) << "Failed to connect Mooncake prefetch stream: "
               << cntl.ErrorText();
    result->mark_worker_failed();
    brpc::StreamClose(stream_id);
  }
}

bool CommChannel::get_last_llm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::Empty request;
  proto::LLMForwardResponse response;
  brpc::Controller controller;
  stub_->GetLastLLMStepResult(&controller, &request, &response, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "Get last LLM step result failed: " << controller.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  RawForwardOutput output;
  if (!decode_llm_output(response, output)) {
    LOG(ERROR) << "Failed to decode last LLM step result";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  promise.setValue(std::move(output));
  return true;
}

bool CommChannel::get_last_vlm_step_result_async(
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::Empty request;
  proto::VLMForwardResponse response;
  brpc::Controller controller;
  stub_->GetLastVLMStepResult(&controller, &request, &response, nullptr);
  if (controller.Failed()) {
    LOG(ERROR) << "Get last VLM step result failed: " << controller.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  RawForwardOutput output;
  if (!decode_vlm_output(response, output)) {
    LOG(ERROR) << "Failed to decode last VLM step result";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  promise.setValue(std::move(output));
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

bool CommChannel::execute_llm_model_with_brpc(
    const LlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::LLMForwardRequest request;
  if (!input.to_packed_proto(request.mutable_packed_input())) {
    LOG(ERROR) << "Failed to pack LlmForwardInput for remote execution";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  auto* done = new ExecuteLLMModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteLLMModel(&done->cntl, &request, &done->pb_output, done);
  return true;
}

void ExecuteLLMModelClosure::Run() {
  std::unique_ptr<ExecuteLLMModelClosure> self_guard(this);
  if (cntl.Failed()) {
    LOG(ERROR) << "ExecuteLLMModel failed: " << cntl.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  RawForwardOutput output;
  if (!decode_llm_output(pb_output, output)) {
    LOG(ERROR) << "Failed to decode LLM model output";
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  promise.setValue(std::move(output));
}

void CommChannel::execute_dit_model_async(
    const DiTForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_dit_model_with_brpc(input, promise);
}

bool CommChannel::execute_dit_model_with_brpc(
    const DiTForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::DITForwardRequest request;
  if (!input.to_packed_proto(request.mutable_packed_input())) {
    LOG(ERROR) << "Failed to pack DiTForwardInput for remote execution";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  auto* done = new ExecuteDITModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteDITModel(&done->cntl, &request, &done->pb_output, done);
  return true;
}

void ExecuteDITModelClosure::Run() {
  std::unique_ptr<ExecuteDITModelClosure> self_guard(this);
  if (cntl.Failed()) {
    LOG(ERROR) << "ExecuteDITModel failed: " << cntl.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  RawForwardOutput output;
  if (!decode_dit_output(pb_output, output)) {
    LOG(ERROR) << "Failed to decode DIT model output";
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  promise.setValue(std::move(output));
}

void CommChannel::execute_rec_model_async(
    const RecForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_rec_model_with_brpc(input, promise);
}

bool CommChannel::execute_rec_model_with_brpc(
    const RecForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::RECForwardRequest request;
  if (!input.to_packed_proto(request.mutable_packed_input())) {
    LOG(ERROR) << "Failed to pack RecForwardInput for remote execution";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  auto* done = new ExecuteRECModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteRECModel(&done->cntl, &request, &done->pb_output, done);
  return true;
}

void ExecuteRECModelClosure::Run() {
  std::unique_ptr<ExecuteRECModelClosure> self_guard(this);
  if (cntl.Failed()) {
    LOG(ERROR) << "ExecuteRECModel failed: " << cntl.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  RawForwardOutput output;
  if (!decode_rec_output(pb_output, output)) {
    LOG(ERROR) << "Failed to decode REC model output";
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  promise.setValue(std::move(output));
}

void CommChannel::execute_vlm_model_async(
    const VlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  execute_vlm_model_with_brpc(input, promise);
}

bool CommChannel::execute_vlm_model_with_brpc(
    const VlmForwardInput& input,
    folly::Promise<std::optional<RawForwardOutput>>& promise) {
  proto::VLMForwardRequest request;
  if (!input.to_packed_proto(request.mutable_packed_input())) {
    LOG(ERROR) << "Failed to pack VlmForwardInput for remote execution";
    promise.setValue(std::optional<RawForwardOutput>{});
    return false;
  }
  auto* done = new ExecuteVLMModelClosure();
  done->promise = std::move(promise);
  stub_->ExecuteVLMModel(&done->cntl, &request, &done->pb_output, done);
  return true;
}

void ExecuteVLMModelClosure::Run() {
  std::unique_ptr<ExecuteVLMModelClosure> self_guard(this);
  if (cntl.Failed()) {
    LOG(ERROR) << "ExecuteVLMModel failed: " << cntl.ErrorText();
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  RawForwardOutput output;
  if (!decode_vlm_output(pb_output, output)) {
    LOG(ERROR) << "Failed to decode VLM model output";
    promise.setValue(std::optional<RawForwardOutput>{});
    return;
  }
  promise.setValue(std::move(output));
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
