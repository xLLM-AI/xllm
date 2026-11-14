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

#include "core/disaggregation/pd_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/server.h>
#include <butil/endpoint.h>
#include <glog/logging.h>

#include <algorithm>
#include <limits>
#include <queue>
#include <vector>

#include "common/macros.h"
#include "core/disaggregation/pd_request_converter.h"
#include "core/distributed_runtime/distributed_vmm_coordinator.h"
#include "core/distributed_runtime/kv_transfer_coordinator.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/kv_cache/transfer/pd_topology_guard.h"
#include "core/scheduler/profile/profile_manager.h"
#include "core/util/scope_guard.h"
#include "core/util/tensor_codec.h"
#include "disagg_pd.pb.h"
#include "distributed_runtime/xservice_client.h"
#include "framework/block/block_manager_pool.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "scheduler/continuous_scheduler.h"
#include "server/xllm_server_registry.h"
#include "util/timer.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {

namespace {

using RequestPtr = std::shared_ptr<Request>;

constexpr std::chrono::minutes kReservationReleaseRetention{5};

// Returns true when |a| has lower scheduling priority than |b| (i.e. |a|
// should be dispatched after |b|).  std::priority_queue is a max-heap, so the
// comparator that returns true for "lower priority" places the highest-priority
// request on top.  Ordering mirrors the original
// wait_dequeue_timed(online) / try_dequeue(offline) loop:
//   1. online requests always come before offline ones, and
//   2. within each group the earliest created_time wins.
struct RequestPriorityComparator {
  bool operator()(const RequestPtr& a, const RequestPtr& b) const {
    const bool a_online = !a->offline();
    const bool b_online = !b->offline();
    if (a_online != b_online) {
      return !a_online;
    }
    return a->created_time() > b->created_time();
  }
};

using PendingQueue = std::priority_queue<RequestPtr,
                                         std::vector<RequestPtr>,
                                         RequestPriorityComparator>;

// ConcurrentQueue is not FIFO across the HTTP handling threads. Drain what is
// currently queued so dispatch can restore created_time order.
void drain_dispatch_queue(
    moodycamel::BlockingConcurrentQueue<RequestPtr>& queue,
    PendingQueue* pending,
    bool* exit_requested) {
  RequestPtr incoming;
  while (queue.try_dequeue(incoming)) {
    if (incoming == nullptr) {
      *exit_requested = true;
      return;
    }
    pending->push(std::move(incoming));
  }
}

void send_reservation_release(const std::string& req_id,
                              const std::string& reservation_id,
                              const std::string& address) {
  constexpr int32_t kMaxAttempts = 3;
  constexpr int32_t kTimeoutMs = 1000;
  constexpr int32_t kBackoffMs = 50;
  brpc::Channel channel;
  brpc::ChannelOptions options;
  options.timeout_ms = kTimeoutMs;
  options.connect_timeout_ms = kTimeoutMs;
  options.max_retry = 0;
  if (channel.Init(address.c_str(), nullptr, &options) != 0) {
    LOG(ERROR) << "Decode reservation release failed, request_id=" << req_id
               << ", peer=" << address << ", channel initialization failed; "
               << "reservation may remain allocated";
    return;
  }
  proto::DisaggPDService_Stub stub(&channel);
  proto::ReleaseReservationRequest req;
  req.set_req_id(req_id);
  req.set_reservation_id(reservation_id);
  for (int32_t attempt = 1; attempt <= kMaxAttempts; ++attempt) {
    brpc::Controller cntl;
    proto::ReleaseReservationResponse resp;
    stub.ReleaseReservation(&cntl, &req, &resp, nullptr);
    if (!cntl.Failed() &&
        (resp.result() == proto::ReleaseReservationResponse::RELEASED ||
         resp.result() == proto::ReleaseReservationResponse::NOT_WAITING)) {
      LOG(INFO) << "Decode reservation release, request_id=" << req_id
                << ", peer=" << address << ", result=" << resp.result()
                << ", attempt=" << attempt;
      return;
    }
    LOG(WARNING) << "Decode reservation release retry, request_id=" << req_id
                 << ", peer=" << address << ", attempt=" << attempt
                 << ", result=" << resp.result()
                 << ", rpc_error=" << cntl.ErrorText();
    if (attempt < kMaxAttempts) {
      absl::SleepFor(absl::Milliseconds(kBackoffMs));
    }
  }
  LOG(ERROR) << "Decode reservation release exhausted retries, request_id="
             << req_id << ", peer=" << address
             << "; reservation may remain allocated";
}

}  // namespace

bool is_permanent_rejection(int32_t status_code) {
  return status_code == kDecodeAddNewInvalidRequestStatusCode ||
         status_code == kDecodeAddNewPromptTooLongStatusCode;
}

bool exceeds_decode_capacity(size_t num_prompt_tokens,
                             size_t block_size,
                             size_t num_blocks) {
  if (block_size == 0) {
    return true;
  }
  const size_t needed_blocks = util::ceil_div(num_prompt_tokens, block_size);
  const size_t usable_blocks = num_blocks == 0 ? 0 : num_blocks - 1;
  return needed_blocks > usable_blocks;
}

bool has_rank_preserving_kv_groups(const proto::DisaggResponse& response) {
  return std::all_of(
      response.groups().begin(),
      response.groups().end(),
      [](const proto::KVTransferGroup& group) {
        const std::optional<BlockType> block_type =
            block_type_from_cache_group_id(group.group_id());
        return block_type.has_value() &&
               (block_type.value() == BlockType::KV ||
                !is_kv_split_cache_block_type(block_type.value()));
      });
}

PDScheduler::~PDScheduler() {
  if (xservice_client_ != nullptr) {
    xservice_client_->clear_heartbeat_callback(
        heartbeat_callback_registration_);
  }

  // The master loop has joined. RPC handlers and the handoff worker may still
  // be waiting for block retirement, so keep consuming their commands while
  // another thread stops the producers. Server::stop itself waits for RPCs.
  auto rpc_server = ServerRegistry::get_instance().try_get_server(server_name_);
  std::thread stop_producers([this, rpc_server] {
    if (rpc_server != nullptr) {
      rpc_server->stop();
    }

    if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
      rpc_server_thread_->join();
    }

    if (dispatch_thread_ && dispatch_thread_->joinable()) {
      CHECK(prefill_request_queue_.enqueue(nullptr));
      dispatch_thread_->join();
    }

    // The FIFO barrier follows every handoff task. All release producers have
    // stopped when it enqueues the terminal command.
    prefill_threadpool_.schedule(
        [this] { CHECK(request_block_releases_.enqueue(nullptr)); });
  });
  std::unique_ptr<RequestBlockRelease> release;
  while (true) {
    request_block_releases_.wait_dequeue(release);
    if (release == nullptr) {
      break;
    }
    complete_request_block_release(*release);
  }
  stop_producers.join();
  // ConcurrentQueue preserves producer order, not a global order. Other
  // producers' final commands can be observed after the terminal command.
  drain_request_releases();

  if (rpc_server != nullptr) {
    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

void PDScheduler::initialize_rpc_server(const std::string& server_name) {
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  while (!rpc_server || !rpc_server->has_initialized()) {
    absl::SleepFor(absl::Milliseconds(100));
    rpc_server = ServerRegistry::get_instance().get_server(server_name);
  }
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
  const XServiceClient::HeartbeatCallbackRegistration previous_registration =
      heartbeat_callback_registration_;
  heartbeat_callback_registration_ = xservice_client_->set_heartbeat_callback(
      [this](xllm_service::proto::HeartbeatRequest& request) {
        populate_heartbeat_request(request, true);
      });
  xservice_client_->clear_heartbeat_callback(previous_registration);
}

void PDScheduler::register_instance_info(const std::string& server_name) {
  instance_info_.name = xservice_client_->get_instance_name();
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  instance_info_.rpc_address = rpc_server->listen_address();
  instance_info_.type = options_.instance_role().value().to_string();
  LOG(INFO) << "Instance info: instance name = " << instance_info_.name
            << ", instance rpc_address = " << instance_info_.rpc_address
            << ", instance type = " << instance_info_.type;

  distributed_worker_manager_->get_cache_info(
      instance_info_.cluster_ids, instance_info_.addrs, instance_info_.ports);
  instance_info_.dp_size = options_.dp_size();

#if defined(USE_NPU)
  if (::xllm::KVCacheConfig::get_instance().enable_vmm_kvcache()) {
    auto& memory_coordinator = DistributedVmmCoordinator::get_instance();
    if (memory_coordinator.is_initialized()) {
      instance_info_.total_phy_pages =
          memory_coordinator.get_num_total_phy_pages();
    }
  }
#endif
}

void PDScheduler::profile_ttft() {
  LOG(INFO) << "Start profiling TTFT.";
  auto& model_args = *model_args_;
  int32_t max_context_len = model_args.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }

  profile_manager_->run_request(max_context_len, 0);

  for (int32_t token_length = max_context_len; token_length > 1;
       token_length *= 0.9) {
    double latency = profile_manager_->run_request(token_length, 0);
    instance_info_.ttft_profiling_data.emplace_back(
        std::make_pair(token_length, latency));
  }
}

void PDScheduler::profile_tpot() {
  LOG(INFO) << "Start profiling TPOT.";
  auto& model_args = *model_args_;
  int32_t max_context_len = model_args.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }

  const size_t resource_capacity = block_manager_pool_->admission_capacity(0);
  const int32_t block_size = block_manager_pool_->block_size();
  int32_t max_seqs_per_batch = options_.max_seqs_per_batch();
  int32_t max_batch_size =
      resource_capacity / block_manager_pool_->sequence_resource_units(
                              (max_context_len / block_size + 1) * block_size);

  profile_manager_->run_request(
      max_context_len, max_context_len - 1, max_batch_size);

  for (int32_t token_length = max_context_len; token_length > 64;
       token_length >>= 1) {
    max_batch_size =
        resource_capacity / block_manager_pool_->sequence_resource_units(
                                (token_length / block_size + 1) * block_size);
    int32_t current_max_batch_size = max_batch_size > max_seqs_per_batch
                                         ? max_seqs_per_batch
                                         : max_batch_size;
    for (int32_t batch_size = current_max_batch_size; batch_size > 0;
         batch_size *= 0.9) {
      double latency = profile_manager_->profile_decode_step_time(
          token_length, batch_size, /*min_context_len=*/64, max_context_len);
      instance_info_.tpot_profiling_data.emplace_back(
          token_length, batch_size, latency);
    }
  }
}

void PDScheduler::start_rpc_server(PDServiceOptions response_options) {
  std::unique_ptr<PDService> service =
      std::make_unique<PDService>(this, std::move(response_options));
  auto rpc_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!rpc_server->start(std::move(service))) {
    const int32_t port =
        ::xllm::DisaggPDConfig::get_instance().disagg_pd_port();
    LOG(ERROR) << "Failed to start brpc disagg pd server on port " << port;
    return;
  }
}

void PDScheduler::step(const absl::Duration& timeout) {
  ContinuousScheduler::step(timeout);
  // Always check (not gated on last_step_prefill_) because chunked prefill
  // mode does not set that flag and a chunked prefill may complete at any step.
  // prefill_send_first_generation() internally checks
  // num_generated_tokens() == 1 so spurious calls are harmless.
  if (options_.instance_role() != InstanceRole::DECODE) {
    prefill_send_first_generation();
  }
}

size_t PDScheduler::num_queued_requests() const {
  return ContinuousScheduler::num_queued_requests() +
         pending_dispatch_requests_;
}

bool PDScheduler::enqueue_ready_request(std::shared_ptr<Request> request) {
  const bool offline = request->offline();
  const bool queued =
      offline ? prefill_request_queue_offline_.enqueue(std::move(request))
              : prefill_request_queue_.enqueue(std::move(request));
  if (queued) {
    ++pending_dispatch_requests_;
  }
  return queued;
}

void PDScheduler::release_dispatch_admission() {
  std::lock_guard<std::mutex> lock(request_admission_mutex_);
  CHECK_GT(pending_dispatch_requests_, 0u);
  --pending_dispatch_requests_;
}

bool PDScheduler::enqueue_dispatched_request(std::shared_ptr<Request> request) {
  bool queued = false;
  {
    std::lock_guard<std::mutex> lock(request_admission_mutex_);
    CHECK_GT(pending_dispatch_requests_, 0u);
    queued = ContinuousScheduler::enqueue_ready_request(request);
    --pending_dispatch_requests_;
  }
  if (!queued) {
    release_failed_request(request);
    output_dispatcher_->process_failed_request(
        std::move(request),
        {StatusCode::RESOURCE_EXHAUSTED,
         "Prefill request queue is full after Decode allocation"});
  }
  return queued;
}

void PDScheduler::dispatch_requests() {
  PendingQueue pending;
  while (true) {
    bool exit_requested = false;
    drain_dispatch_queue(prefill_request_queue_, &pending, &exit_requested);
    if (exit_requested) {
      break;
    }

    if (pending.empty()) {
      const auto timeout = std::chrono::milliseconds(100);
      // Wait for online request until timeout.
      // If timeout, try to get offline request once. If no offline request,
      // continue to wait for online request. This can avoid offline request
      // blocking online request for too long time.
      std::shared_ptr<Request> incoming;
      if (!prefill_request_queue_.wait_dequeue_timed(incoming, timeout)) {
        if (!prefill_request_queue_offline_.try_dequeue(incoming)) {
          continue;
        }
      }
      if (incoming == nullptr) {
        break;
      }
      pending.push(std::move(incoming));
      drain_dispatch_queue(prefill_request_queue_, &pending, &exit_requested);
      if (exit_requested) {
        break;
      }
    }

    std::shared_ptr<Request> request = pending.top();
    pending.pop();
    ScopeGuard release_admission([this] { release_dispatch_admission(); });

    std::string selected_instance = request->state().decode_address;
    if (selected_instance.empty()) {
      output_dispatcher_->process_failed_request(
          request,
          {StatusCode::INVALID_ARGUMENT,
           "No decode address provided to the prefill instance"});
      continue;
    }
    if (request->source_xservice_addr().empty() ||
        request->service_request_id().empty()) {
      output_dispatcher_->process_failed_request(
          request,
          {StatusCode::INVALID_ARGUMENT,
           "PD request requires a service address and request ID"});
      continue;
    }

    const InstanceInfo remote_info =
        xservice_client_->get_instance_info(selected_instance);
    if (remote_info.name.empty()) {
      output_dispatcher_->process_failed_request(
          request,
          {StatusCode::UNKNOWN, "failed to fetch remote decode instance info"});
      continue;
    }

    const PdTopoResult topo_result = check_pd_topo(
        instance_info_, remote_info, options_.kv_cache_transfer_mode());
    const bool allow_pd_topo = topo_result.status == PdTopoStatus::ALLOW_HOMO ||
                               topo_result.status == PdTopoStatus::ALLOW_HETERO;
    if (!allow_pd_topo) {
      output_dispatcher_->process_failed_request(
          request,
          {StatusCode::INVALID_ARGUMENT,
           "decode instance " + selected_instance +
               " is incompatible: " + topo_result.reason});
      continue;
    }
    if (topo_result.status == PdTopoStatus::ALLOW_HETERO && VLOG_IS_ON(1)) {
      const PdTopo local_topo = get_pd_topo(instance_info_);
      const PdTopo remote_topo = get_pd_topo(remote_info);
      VLOG(1) << "Allow hetero pd topo guard: local dp/tp="
              << local_topo.dp_size << "/" << local_topo.tp_size
              << ", remote dp/tp=" << remote_topo.dp_size << "/"
              << remote_topo.tp_size;
    }
    proto::DisaggPDService_Stub* stub = nullptr;
    {
      std::lock_guard<std::mutex> lock(instance_channel_map_mutex_);
      auto it = instance_channel_map_.find(remote_info.name);
      if (it != instance_channel_map_.end()) {
        stub = it->second.get();
      } else {
        LOG(INFO) << "Create rpc channel to instance: " << remote_info.name;
        auto channel = std::make_unique<brpc::Channel>();
        brpc::ChannelOptions options;
        options.timeout_ms =
            ::xllm::ServiceConfig::get_instance().rpc_channel_timeout_ms();
        options.max_retry = 3;
        if (channel->Init(remote_info.rpc_address.c_str(), "", &options) != 0) {
          LOG(ERROR) << "Fail to initialize channel for "
                     << remote_info.rpc_address;
        } else {
          auto owned_stub = std::make_unique<proto::DisaggPDService_Stub>(
              channel.get(), google::protobuf::Service::STUB_OWNS_CHANNEL);
          channel.release();
          stub = owned_stub.get();
          instance_channel_map_[remote_info.name] = std::move(owned_stub);
        }
      }
    }
    if (stub == nullptr) {
      output_dispatcher_->process_failed_request(
          request, {StatusCode::UNKNOWN, "Fail to create rpc channel"});
      continue;
    }

    if (request->state().pd_reservation_id.empty()) {
      request->state().pd_reservation_id = ShortUUID().random();
    }

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      req_to_channel_map_[request->request_id()] = stub;
    }

    Timer decode_allocation_timer;
    xllm::proto::DisaggRequests reqs;
    xllm::proto::DisaggResponses resps;
    reqs.set_prefill_name(xservice_client_->get_instance_name());
    fill_pd_request(*request, reqs.add_reqs());
    reqs.mutable_cluster_infos()->mutable_cluster_ids()->Add(
        instance_info_.cluster_ids.begin(), instance_info_.cluster_ids.end());
    reqs.mutable_cluster_infos()->mutable_addrs()->Add(
        instance_info_.addrs.begin(), instance_info_.addrs.end());
    reqs.mutable_cluster_infos()->mutable_ports()->Add(
        instance_info_.ports.begin(), instance_info_.ports.end());
    reqs.mutable_cluster_infos()->set_dp_size(options_.dp_size());
    const double allocation_prepare_seconds =
        decode_allocation_timer.elapsed_seconds();

    brpc::Controller cntl;
    Timer allocation_rpc_timer;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);
    const double allocation_rpc_seconds =
        allocation_rpc_timer.elapsed_seconds();
    if (cntl.Failed()) {
      LOG(ERROR) << "Failed to add new requests to decode instance : "
                 << selected_instance << ", error text : " << cntl.ErrorText();
      request->state().decode_rpc_address =
          cntl.remote_side().port > 0
              ? butil::endpoint2str(cntl.remote_side()).c_str()
              : remote_info.rpc_address;
      release_failed_request(request);
      output_dispatcher_->process_failed_request(
          request,
          {StatusCode::UNKNOWN,
           "Failed to add new requests to decode instance"});
      continue;
    }

    CHECK_EQ(resps.resps_size(), 1)
        << "selected_instance : " << selected_instance;
    {
      std::lock_guard<std::mutex> lock(linked_instances_mutex_);
      linked_instance_.emplace(selected_instance);
    }
    const proto::DisaggResponse& resp = resps.resps(0);
    if (resp.status_code() != 200) {
      if (is_permanent_rejection(resp.status_code())) {
        LOG(ERROR) << "Decode permanently rejected request, status_code="
                   << resp.status_code()
                   << ", request_id=" << request->request_id()
                   << ", prompt_tokens="
                   << request->state().prompt_tokens.size()
                   << ", selected_instance=" << selected_instance;
        do_permanent_rejection(request, resp.status_code());
        continue;
      }
      // Keep created_time order: retry from the local pending queue instead
      // of the MPMC queue, which is not FIFO across HTTP threads.
      pending.push(request);
      release_admission.dismiss();
    } else {
      for (auto& sequence : request->sequences()) {
        TransferKVInfo info;
        info.request_id = request->request_id();
        info.rank_local_mapping = instance_info_.kv_split_size > 1 &&
                                  has_rank_preserving_kv_groups(resp);
        info.mappings.reserve(resp.groups_size());
        for (const proto::KVTransferGroup& proto_group : resp.groups()) {
          KVTransferMapping mapping;
          mapping.group_id = proto_group.group_id();
          mapping.remote_ids.assign(proto_group.ids().begin(),
                                    proto_group.ids().end());
          mapping.remote_shared_num = proto_group.remote_shared_num();
          info.mappings.emplace_back(std::move(mapping));
        }
        info.dp_rank = resp.dp_rank();
        info.remote_instance_info = remote_info;

        // VirtualMemory mode: save destination offsets from D-node
        if (resp.virtual_kv_cache_layer_global_offsets_size() > 0) {
          info.dst_virtual_kv_cache_layer_global_offsets.reserve(
              resp.virtual_kv_cache_layer_global_offsets_size());
          for (const auto& layer_offsets :
               resp.virtual_kv_cache_layer_global_offsets()) {
            VirtualKVCacheLayerGlobalOffsets layer;
            layer.k_offsets.assign(layer_offsets.k_offsets().begin(),
                                   layer_offsets.k_offsets().end());
            layer.v_offsets.assign(layer_offsets.v_offsets().begin(),
                                   layer_offsets.v_offsets().end());
            layer.index_offsets.assign(layer_offsets.index_offsets().begin(),
                                       layer_offsets.index_offsets().end());
            info.dst_virtual_kv_cache_layer_global_offsets.emplace_back(
                std::move(layer));
          }
          VLOG(5) << "Received VirtualMemory offsets from D-node for request "
                  << request->request_id() << ", num_layers="
                  << info.dst_virtual_kv_cache_layer_global_offsets.size();
        }

        sequence->kv_state().set_transfer_kv_info(std::move(info));

        // Compress per-group transfer cursors to the D-side shared count.
        // advance_group_transfer_block_idx is monotonic max, so this is a
        // no-op if the group's remote_shared_num is 0 (SWA / LINEAR under
        // DECODE, or D that got no prefix cache hit); when it is >0, P
        // starts pushing at the first block D actually needs, skipping the
        // shared prefix without touching P's own local shared_num.
        const auto& mappings =
            sequence->kv_state().transfer_kv_info()->mappings;
        for (const KVTransferMapping& mapping : mappings) {
          if (mapping.remote_shared_num == 0) {
            continue;
          }
          const std::optional<BlockType> block_type =
              block_type_from_cache_group_id(mapping.group_id);
          if (!block_type.has_value()) {
            LOG(ERROR) << "Unknown KV cache transfer group_id: "
                       << mapping.group_id;
            continue;
          }
          if (block_type.value() == BlockType::KV) {
            sequence->kv_state().advance_transfer_block_idx(
                static_cast<size_t>(mapping.remote_shared_num));
          } else {
            sequence->kv_state().advance_group_transfer_block_idx(
                block_type.value(),
                static_cast<size_t>(mapping.remote_shared_num));
          }
        }
      }

      // Retain the peer that actually accepted this RPC, even if discovery
      // changes before the asynchronous release task runs.
      request->state().decode_rpc_address =
          butil::endpoint2str(cntl.remote_side()).c_str();
      // Push to request_queue_; it will be executed by the engine.
      release_admission.dismiss();
      enqueue_dispatched_request(request);
    }
    VLOG(1) << "Prefill Decode allocation request_id=" << request->request_id()
            << ", prepare_ms=" << allocation_prepare_seconds * 1000.0
            << ", rpc_ms=" << allocation_rpc_seconds * 1000.0 << ", total_ms="
            << decode_allocation_timer.elapsed_seconds() * 1000.0;
  }
}

void PDScheduler::prefill_send_first_generation() {
  if (running_sequences_.empty()) {
    return;
  }

  std::vector<std::shared_ptr<Request>> requests;
  for (auto& request : running_requests_) {
    // Check if the request is a recently completed prefill request
    if (request->sequences()[0]->num_generated_tokens() == 1) {
      if (!options_.disable_log_stats()) {
        request->log_statistic(request->elapsed_seconds());
      }
      // Retain the accepting peer until FirstGeneration succeeds so a failed
      // handoff can release the matching decode reservation.
      requests.emplace_back(std::move(request));
    }
  }
  // No prefill request needs to be transferred to decode.
  if (requests.empty()) {
    return;
  }

  prefill_threadpool_.schedule([this,
                                requests = std::move(requests)]() mutable {
    // TODO: here we only support one sequence for now.
    auto fail_request = [this](const std::shared_ptr<Request>& request,
                               Status status) {
      release_failed_request(request);
      output_dispatcher_->process_failed_request(request, status);
      output_dispatcher_->wait_completion();
      deallocate_sequences(*request, /*cache_prefix=*/true);
    };
    for (auto& request : requests) {
      if (request->cancelled()) {
        fail_request(
            request,
            {StatusCode::CANCELLED, "Request cancelled before decode handoff"});
        continue;
      }
      Timer first_generation_timer;
      // TODO: support batch request later
      proto::DisaggGenerationsRequests gens;
      auto gen = gens.mutable_multi_gens()->Add();
      gen->set_req_id(request->request_id());
      gen->set_reservation_id(request->state().pd_reservation_id);
      gen->set_x_request_id(request->x_request_id());
      gen->set_x_request_time(request->x_request_time());
      const size_t num_cached_tokens = request->num_prefix_cache_tokens();
      if (num_cached_tokens >
          static_cast<size_t>(std::numeric_limits<int32_t>::max())) {
        LOG(WARNING) << "Invalid num_cached_tokens on prefill, request_id: "
                     << request->request_id()
                     << ", num_cached_tokens: " << num_cached_tokens
                     << ". Falling back to 0.";
        gen->set_num_cached_tokens(0);
      } else {
        gen->set_num_cached_tokens(static_cast<int32_t>(num_cached_tokens));
      }
      if (request->sequences()[0]->first_token().has_value()) {
        auto token = gen->mutable_tokens()->Add();
        token->set_token_id(
            request->sequences()[0]->first_token().value().token_id);
        token->set_time_to_first_token_latency_seconds(
            request->sequences()[0]->time_to_first_token_latency_seconds());
        if (request->sequences()[0]
                ->first_token()
                .value()
                .token_logprob.has_value()) {
          token->set_logprob(request->sequences()[0]
                                 ->first_token()
                                 .value()
                                 .token_logprob.value());
          token->set_has_logprob(true);
        } else {
          token->set_has_logprob(false);
        }
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_tokens(),
            request->sequences()[0]->first_token().value().token_top_tokens);
        ADD_VECTOR_TO_PROTO(
            token->mutable_top_logprobs(),
            request->sequences()[0]->first_token().value().token_top_logprobs);
      }
      gen->set_kv_cache_transfer_mode(options_.kv_cache_transfer_mode());
      // PUSH is completed before FirstGeneration. Only native PULL needs
      // source cache metadata in this request.
      if (options_.kv_cache_transfer_mode() == "PULL") {
        ADD_VECTOR_TO_PROTO(gen->mutable_cluster_ids(),
                            instance_info_.cluster_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_addrs(), instance_info_.addrs);
        Sequence* sequence = request->sequences()[0].get();
        if (sequence->kv_state().has_multi_block_export()) {
          const auto export_view =
              sequence->kv_state().multi_block_export_view();
          for (const auto& [block_type, blocks_ptr] : export_view) {
            proto::KVTransferGroup* group = gen->add_source_groups();
            group->set_group_id(cache_group_id(block_type));
            group->mutable_ids()->Reserve(blocks_ptr->size());
            for (const Block& block : *blocks_ptr) {
              if (block.is_valid()) {
                group->add_ids(static_cast<uint64_t>(block.id()));
              }
            }
          }
        } else {
          const Slice<Block> blocks =
              sequence->kv_state().blocks(BlockType::KV);
          proto::KVTransferGroup* group = gen->add_source_groups();
          group->set_group_id(cache_group_id(BlockType::KV));
          group->mutable_ids()->Reserve(blocks.size());
          for (const Block& block : blocks) {
            CHECK(block.is_valid());
            group->add_ids(static_cast<uint64_t>(block.id()));
          }
        }
        if (has_linear_attention_layers(*model_args_)) {
          const int32_t linear_state_id = sequence->get_linear_state_slot_id();
          CHECK_GE(linear_state_id, 0)
              << "Prefill did not allocate a linear-state slot.";
          proto::KVTransferGroup* group = gen->add_source_groups();
          group->set_group_id(cache_group_id(BlockType::LINEAR));
          group->add_ids(static_cast<uint64_t>(linear_state_id));
        }
        gen->set_dp_size(instance_info_.dp_size);
        gen->set_dp_rank(sequence->dp_rank());
      }
      if (requires_bootstrap_hidden_) {
        torch::Tensor embedding =
            request->sequences()[0]->get_mtp_bootstrap_embedding();
        if (!embedding.defined()) {
          LOG(ERROR) << "Missing MTP bootstrap embedding, request_id: "
                     << request->request_id();
          fail_request(
              request,
              {StatusCode::UNKNOWN, "Missing MTP bootstrap embedding"});
          continue;
        }
        torch::Tensor embedding_cpu = safe_to(embedding, torch::kCPU);
        if (!util::torch_to_proto(embedding_cpu,
                                  gen->mutable_mtp_bootstrap_embedding())) {
          LOG(ERROR) << "Failed to serialize MTP bootstrap embedding, "
                     << "request_id: " << request->request_id();
          fail_request(request,
                       {StatusCode::UNKNOWN,
                        "Failed to serialize MTP bootstrap embedding"});
          continue;
        }
        request->sequences()[0]->clear_mtp_bootstrap_embedding();
      }
      const double prepare_seconds = first_generation_timer.elapsed_seconds();

      proto::DisaggPDService_Stub* stub = nullptr;
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        // now we only support one request once.
        stub = req_to_channel_map_[request->request_id()];
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      Timer rpc_timer;
      gen->set_upstream_elapsed_seconds(request->end_to_end_latency_seconds());
      stub->FirstGeneration(&cntl, &gens, &resp, nullptr);
      const double rpc_seconds = rpc_timer.elapsed_seconds();
      VLOG(1) << "Prefill first-generation request_id=" << request->request_id()
              << ", prepare_ms=" << prepare_seconds * 1000.0
              << ", rpc_ms=" << rpc_seconds * 1000.0 << ", total_ms="
              << first_generation_timer.elapsed_seconds() * 1000.0;

      const bool sent_first_generation = !cntl.Failed() && resp.ok();
      if (!sent_first_generation) {
        LOG(ERROR) << "Failed to send first generation to decode instance : "
                   << request->state().decode_address
                   << ", error text : " << cntl.ErrorText()
                   << ", response status: " << resp.ok();
        fail_request(request,
                     {StatusCode::UNAVAILABLE,
                      "Failed to send first generation to decode instance"});
        continue;
      }
      // Measure the handoff on the original ingress clock. Decode acknowledges
      // only after restoring KV and accepting the first output; this includes
      // the acknowledgement's return trip without comparing clocks on hosts.
      request->record_server_first_token();
      request->state().decode_rpc_address.clear();

      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      output_dispatcher_->wait_completion();
      enqueue_request_block_release(std::move(request), /*cache_prefix=*/true);
    }
  });
}

void PDScheduler::release_failed_request(
    const std::shared_ptr<Request>& request) {
  {
    std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
    req_to_channel_map_.erase(request->request_id());
  }
  std::string address = std::move(request->state().decode_rpc_address);
  request->state().decode_rpc_address.clear();
  if (address.empty()) {
    return;
  }
  const std::string req_id = request->request_id();
  const std::string reservation_id = request->state().pd_reservation_id;
  if (reservation_id.empty()) {
    LOG(ERROR)
        << "Cannot release Decode reservation without identity, request_id="
        << req_id << ", peer=" << address;
    return;
  }

  // This request has been removed from the scheduling queue. LLMEngine::step
  // waits for every worker, whose step_internal waits for all PUSH futures
  // (also in overlap mode). Thus prior chunks cannot still write remote KV,
  // and this failed request cannot enter a later batch or FirstGeneration.
  // Capture values only: neither Request nor the scheduler's cached raw stub
  // is needed by retries. The pool drains these tasks during destruction.
  reservation_release_threadpool_.schedule([req_id, reservation_id, address] {
    send_reservation_release(req_id, reservation_id, address);
  });
}

std::shared_ptr<Request> PDScheduler::take_waiting_request(
    const std::string& req_id) {
  auto it = received_request_map_.find(req_id);
  if (it == received_request_map_.end()) {
    return nullptr;
  }
  auto request = std::move(it->second);
  received_request_map_.erase(it);
  auto inst_it = request_to_instance_map_.find(req_id);
  if (inst_it != request_to_instance_map_.end()) {
    auto reverse_it = instance_to_received_requests_map_.find(inst_it->second);
    if (reverse_it != instance_to_received_requests_map_.end()) {
      reverse_it->second.erase(req_id);
      if (reverse_it->second.empty()) {
        instance_to_received_requests_map_.erase(reverse_it);
      }
    }
    request_to_instance_map_.erase(inst_it);
  }
  return request;
}

std::future<void> PDScheduler::enqueue_request_block_release(
    std::shared_ptr<Request> request,
    bool cache_prefix) {
  auto release = std::make_unique<RequestBlockRelease>(
      RequestBlockRelease{std::move(request), cache_prefix, {}});
  auto completion = release->completion.get_future();
  CHECK(request_block_releases_.enqueue(std::move(release)));
  notifier_->notify();
  return completion;
}

void PDScheduler::release_request_blocks(std::shared_ptr<Request> request,
                                         bool cache_prefix) {
  enqueue_request_block_release(std::move(request), cache_prefix).get();
}

void PDScheduler::deallocate_sequences(Request& request, bool cache_prefix) {
  if (cache_prefix) {
    block_manager_pool_->deallocate(&request);
    return;
  }
  for (auto& sequence : request.sequences()) {
    if (sequence != nullptr) {
      block_manager_pool_->deallocate_without_cache(sequence.get());
    }
  }
}

void PDScheduler::complete_request_block_release(RequestBlockRelease& release) {
  deallocate_sequences(*release.request, release.cache_prefix);
  release.completion.set_value();
}

void PDScheduler::drain_request_releases() {
  std::unique_ptr<RequestBlockRelease> release;
  while (request_block_releases_.try_dequeue(release)) {
    complete_request_block_release(*release);
  }
}

bool PDScheduler::release_reservation(const std::string& req_id,
                                      const std::string& reservation_id) {
  if (reservation_id.empty()) {
    return false;
  }
  std::shared_ptr<Request> request;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    const std::chrono::steady_clock::time_point now =
        std::chrono::steady_clock::now();
    std::erase_if(released_reservations_,
                  [now](const auto& entry) { return entry.second <= now; });
    released_reservations_.insert_or_assign(
        ReservationIdentity{req_id, reservation_id},
        now + kReservationReleaseRetention);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end() ||
        it->second->state().pd_reservation_id != reservation_id) {
      return false;
    }
    request = take_waiting_request(req_id);
  }
  deallocate_sequences(*request, /*cache_prefix=*/false);
  return true;
}

// request is received from prefill
bool PDScheduler::decode_schedule(std::shared_ptr<Request>& request,
                                  const std::string& prefill_instance_name) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  {
    std::unique_lock<std::mutex> lock(received_request_map_mutex_);
    bool released = false;
    if (!released_reservations_.empty()) {
      auto marker = released_reservations_.find(ReservationIdentity{
          request->request_id(), request->state().pd_reservation_id});
      if (marker != released_reservations_.end()) {
        released = marker->second > std::chrono::steady_clock::now();
        if (!released) {
          released_reservations_.erase(marker);
        }
      }
    }
    if (released || received_request_map_.find(request->request_id()) !=
                        received_request_map_.end()) {
      LOG(ERROR) << "Decode rejected duplicate or released reservation, "
                 << "request_id=" << request->request_id()
                 << ", reservation_id=" << request->state().pd_reservation_id;
      lock.unlock();
      deallocate_sequences(*request, /*cache_prefix=*/false);
      return false;
    }
    received_request_map_[request->request_id()] = request;
    instance_to_received_requests_map_[prefill_instance_name].insert(
        request->request_id());
    request_to_instance_map_[request->request_id()] = prefill_instance_name;
  }

  return true;
}

bool PDScheduler::decode_recv_first_generation(
    const std::string& req_id,
    const std::string& reservation_id,
    int64_t token_id,
    bool has_logprob,
    float logprob,
    double time_to_first_token_latency_seconds,
    double upstream_elapsed_seconds,
    std::vector<int64_t> top_tokens,
    std::vector<float> top_logprobs,
    const std::string& kv_cache_transfer_mode,
    std::vector<uint64_t> src_cluster_ids,
    std::vector<std::string> src_addrs,
    std::vector<KVTransferMapping> source_mappings,
    int32_t src_dp_size,
    int32_t src_dp_rank,
    torch::Tensor mtp_bootstrap_embedding,
    int32_t num_cached_tokens) {
  Timer receive_timer;
  // push to request_queue_, and will be executed by engine.
  std::shared_ptr<Request> request = nullptr;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << req_id;
      return false;
    }
    if (it->second->state().pd_reservation_id != reservation_id) {
      LOG(WARNING) << "Ignoring stale first generation, request_id: " << req_id;
      return false;
    }
    request = take_waiting_request(req_id);
  }
  auto& sequences = request->sequences();
  bool kv_restored = false;
  ScopeGuard release_unqueued_blocks([this, &request, &kv_restored] {
    deallocate_sequences(*request, /*cache_prefix=*/kv_restored);
  });
  if (sequences.empty() || sequences[0] == nullptr) {
    LOG(ERROR) << "Request has no valid sequences, request_id: " << req_id;
    return false;
  }
  Sequence* sequence = request->sequences()[0].get();
  if (num_cached_tokens < 0 ||
      static_cast<size_t>(num_cached_tokens) > sequence->num_prompt_tokens()) {
    LOG(WARNING) << "Invalid num_cached_tokens from prefill, request_id: "
                 << req_id << ", num_cached_tokens: " << num_cached_tokens
                 << ", num_prompt_tokens: " << sequence->num_prompt_tokens()
                 << ". Falling back to 0.";
    num_cached_tokens = 0;
  }
  request->record_num_prefix_cache_tokens(
      static_cast<size_t>(num_cached_tokens));
  if (requires_bootstrap_hidden_) {
    const int32_t slot_id = sequence->get_embedding_block_id();
    if (slot_id < 0) {
      LOG(ERROR) << "Invalid MTP bootstrap slot, request_id: " << req_id;
      return false;
    }
    if (token_id < 0 ||
        token_id > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      LOG(ERROR) << "Invalid MTP bootstrap token, request_id: " << req_id
                 << ", token_id: " << token_id;
      return false;
    }
    if (!mtp_bootstrap_embedding.defined()) {
      LOG(ERROR) << "Missing MTP bootstrap embedding, request_id: " << req_id;
      return false;
    }

    sequence->update_mtp_bootstrap_embedding(mtp_bootstrap_embedding);
  }

  Token first_token(token_id);
  if (has_logprob) {
    first_token.logprob = logprob;
    if (!top_tokens.empty() && !top_logprobs.empty()) {
      // NOTE: slice vector here, to avoid copy
      // so we need keep the vector `top_tokens` and `top_logprobs` lifetime
      first_token.top_tokens = top_tokens;
      first_token.top_logprobs = top_logprobs;
    }
  }
  // Decode owns the complete stream for both HTTP and brpc, including the
  // received first token and its logprobs.

  // update latency metrics
  restore_disaggregated_latency(request.get(),
                                time_to_first_token_latency_seconds,
                                upstream_elapsed_seconds);
  // Rebase the ITL clock to the moment Decode receives the first token. The
  // cumulative handoff latency is tracked by Request, while ITL only measures
  // the local interval after the first token is received.
  sequence->tbt(absl::Now());

  // TODO: we only support one sequence for currently.
  // Heterogeneous TP now restores the Prefill KV/linear state before Decode,
  // so its first token follows the same overlap protocol as every other PD
  // transfer path.  Appending the real token here bypasses last_step_token and
  // makes the first Decode step consume misaligned sequence state.
  if (enable_schedule_overlap()) {
    Token fake_token(-1);
    sequence->append_token(fake_token);
    sequence->update_last_step_token(first_token);
  } else {
    sequence->append_token(first_token);
  }
  const double prepare_seconds = receive_timer.elapsed_seconds();

  // Pull KV cache only in native PULL mode. Heterogeneous PUSH writes directly
  // into the final destination tensors before FirstGeneration is sent.
  if (kv_cache_transfer_mode == "PULL") {
    Timer pull_timer;
    for (KVTransferMapping& mapping : source_mappings) {
      const std::optional<BlockType> block_type =
          block_type_from_cache_group_id(mapping.group_id);
      if (!block_type.has_value()) {
        LOG(ERROR) << "Unknown source KV transfer group, request_id=" << req_id
                   << ", group_id=" << mapping.group_id;
        return false;
      }

      if (block_type.value() == BlockType::LINEAR ||
          block_type.value() == BlockType::EMBEDDING) {
        const int32_t local_id = block_type.value() == BlockType::LINEAR
                                     ? sequence->get_linear_state_slot_id()
                                     : sequence->get_embedding_block_id();
        if (local_id >= 0) {
          mapping.local_ids.emplace_back(static_cast<uint64_t>(local_id));
        }
      } else {
        const Slice<Block> blocks =
            sequence->kv_state().blocks(block_type.value());
        size_t local_begin = 0;
        if (block_type.value() == BlockType::SWA &&
            mapping.remote_ids.size() < blocks.size()) {
          local_begin = blocks.size() - mapping.remote_ids.size();
        }
        mapping.local_ids.reserve(blocks.size() - local_begin);
        for (size_t index = local_begin; index < blocks.size(); ++index) {
          const Block& block = blocks[index];
          if (block.is_valid()) {
            mapping.local_ids.emplace_back(static_cast<uint64_t>(block.id()));
          }
        }
      }
      if (mapping.local_ids.size() != mapping.remote_ids.size()) {
        LOG(ERROR) << "PULL KV mapping size mismatch, request_id=" << req_id
                   << ", group_id=" << mapping.group_id
                   << ", local=" << mapping.local_ids.size()
                   << ", remote=" << mapping.remote_ids.size();
        return false;
      }
    }

    const int32_t dst_dp_rank = sequence->dp_rank();
    CHECK(kv_transfer_coordinator_ != nullptr);
    const bool pulled =
        kv_transfer_coordinator_->pull_kv_blocks(src_dp_size,
                                                 src_dp_rank,
                                                 src_cluster_ids,
                                                 src_addrs,
                                                 dst_dp_rank,
                                                 source_mappings);
    if (!pulled) {
      LOG(ERROR) << "Failed to pull KV blocks, request_id: " << req_id;
      return false;
    }
    VLOG(1) << "Decode KV restore request_id=" << req_id
            << ", pull_ms=" << pull_timer.elapsed_seconds() * 1000.0;
  }

  kv_restored = true;
  // Publish first-token output before making the request visible to Decode.
  // The dispatcher prepares output on the response thread and queues service
  // delivery in order. The client chooses the response format on that sender.
  if (request->finished()) {
    output_dispatcher_->process_completed_request(request);
    return true;
  }
  if (request->state().stream) {
    output_dispatcher_->process_stream_request(request);
  }
  Timer enqueue_timer;
  if (!request_queue_.write(request)) {
    LOG(ERROR) << "Failed to enqueue decode request, request_id: " << req_id;
    output_dispatcher_->process_failed_request(
        request,
        {StatusCode::RESOURCE_EXHAUSTED, "Decode request queue is full"});
    return false;
  }
  notifier_->notify();
  release_unqueued_blocks.dismiss();
  VLOG(1) << "Decode first-generation request_id=" << req_id
          << ", prepare_ms=" << prepare_seconds * 1000.0
          << ", enqueue_ms=" << enqueue_timer.elapsed_seconds() * 1000.0
          << ", total_ms=" << receive_timer.elapsed_seconds() * 1000.0;
  return true;
}

void PDScheduler::restore_disaggregated_latency(
    Request* request,
    double time_to_first_token_latency_seconds,
    double upstream_elapsed_seconds) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  Sequence* sequence = request->sequences()[0].get();
  if (time_to_first_token_latency_seconds > 0 &&
      sequence->time_to_first_token_latency_seconds() <= 0) {
    sequence->set_time_to_first_token_latency_seconds(
        time_to_first_token_latency_seconds);
  }

  const double cumulative_upstream_latency_seconds =
      std::max(upstream_elapsed_seconds,
               sequence->time_to_first_token_latency_seconds());
  if (cumulative_upstream_latency_seconds > 0) {
    request->set_upstream_latency_seconds(cumulative_upstream_latency_seconds);
  }
}

void PDScheduler::start_runtime(PDServiceOptions response_options) {
  CHECK(distributed_worker_manager_ != nullptr)
      << "Disaggregated PD requires a distributed worker manager.";
  CHECK(kv_transfer_coordinator_ != nullptr)
      << "Disaggregated PD requires a KV cache transfer coordinator.";
  dispatch_thread_ =
      std::make_unique<std::thread>(&PDScheduler::dispatch_requests, this);
  server_name_.append(std::to_string(options_.server_idx()));
  rpc_server_thread_ = std::make_unique<std::thread>(
      [this, response_options = std::move(response_options)]() mutable {
        start_rpc_server(std::move(response_options));
      });
  initialize_rpc_server(server_name_);
  register_instance_info(server_name_);
  if (!options_.disable_ttft_profiling() &&
      options_.instance_role().value() == InstanceRole::MIX) {
    profile_ttft();
    profile_tpot();
  }
}

bool PDScheduler::try_allocate(Sequence* sequence) {
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  return block_manager_pool_->kv_cache_utilization() <
             ::xllm::SchedulerConfig::get_instance()
                 .prefill_scheduling_memory_usage_threshold() &&
         block_manager_pool_->try_allocate(sequence);
}

bool PDScheduler::exceeds_decode_capacity(Sequence* sequence) const {
  CHECK(sequence != nullptr);
  const BlockManagerPool::Options& block_options =
      block_manager_pool_->options();
  if (block_options.enable_vmm_kvcache() ||
      !block_options.manager_types().empty() ||
      (block_options.enable_host_offload() &&
       options_.instance_role() != InstanceRole::DECODE)) {
    return false;
  }
  return ::xllm::exceeds_decode_capacity(
      sequence->num_prompt_tokens(),
      static_cast<size_t>(block_manager_pool_->block_size()),
      static_cast<size_t>(block_manager_pool_->num_blocks()));
}

void PDScheduler::do_permanent_rejection(
    const std::shared_ptr<Request>& request,
    int32_t status_code) {
  CHECK(request != nullptr);
  const bool invalid_request =
      status_code == kDecodeAddNewInvalidRequestStatusCode;
  output_dispatcher_->process_failed_request(
      request,
      {invalid_request ? StatusCode::INVALID_ARGUMENT
                       : StatusCode::RESOURCE_EXHAUSTED,
       invalid_request ? "Decode rejected invalid PD service metadata"
                       : "Request prompt exceeds decode KV cache capacity"});
  deallocate_sequences(*request, /*cache_prefix=*/true);
  std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
  req_to_channel_map_.erase(request->request_id());
}

void PDScheduler::get_latency_metrics(std::vector<int64_t>& ttft,
                                      std::vector<int64_t>& tbt) {
  scheduler_metrics_->get_latency_metrics(ttft, tbt);
}

bool PDScheduler::link_instance(const std::string& instance_name,
                                const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<uint16_t>& ports,
                                const int32_t dp_size,
                                const int32_t src_kv_split_size) {
  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (distributed_worker_manager_ == nullptr) {
    LOG(ERROR)
        << "Cannot link an instance without a distributed worker manager.";
    return false;
  }
  if (!distributed_worker_manager_->link_cluster(
          cluster_ids, addrs, ports, dp_size, src_kv_split_size)) {
    LOG(ERROR) << "Link instance failed, instance_name: " << instance_name;
    return false;
  }
  LOG(INFO) << "Successfully linked instance, instance_name: " << instance_name
            << ", prefill_kv_split_size: " << src_kv_split_size;
  linked_instance_.emplace(instance_name);
  return true;
}

bool PDScheduler::unlink_instance(const std::string& instance_name,
                                  const std::vector<uint64_t>& cluster_ids,
                                  const std::vector<std::string>& addrs,
                                  const std::vector<uint16_t>& ports,
                                  const int32_t dp_size,
                                  const int32_t src_kv_split_size) {
  // Clear received requests from this instance
  std::vector<std::shared_ptr<Request>> requests;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = instance_to_received_requests_map_.find(instance_name);
    if (it != instance_to_received_requests_map_.end()) {
      requests.reserve(it->second.size());
      for (const auto& req_id : it->second) {
        requests.emplace_back(std::move(received_request_map_.at(req_id)));
        received_request_map_.erase(req_id);
        request_to_instance_map_.erase(req_id);
      }
      instance_to_received_requests_map_.erase(it);
    }
  }
  for (auto& request : requests) {
    release_request_blocks(std::move(request), /*cache_prefix=*/false);
  }

  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (distributed_worker_manager_ == nullptr) {
    LOG(ERROR)
        << "Cannot unlink an instance without a distributed worker manager.";
    return false;
  }
  if (!distributed_worker_manager_->unlink_cluster(
          cluster_ids, addrs, ports, dp_size, src_kv_split_size)) {
    LOG(ERROR) << "Unlink instance failed, instance_name: " << instance_name;
    return false;
  }
  LOG(INFO) << "Successfully unlinked instance, instance_name: "
            << instance_name;
  linked_instance_.erase(instance_name);
  return true;
}

}  // namespace xllm
