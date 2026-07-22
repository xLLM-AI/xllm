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

#include "scheduler/disagg_pd_scheduler.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <brpc/server.h>
#include <glog/logging.h>

#include <limits>
#include <random>

#include "common/global_flags.h"
#include "common/macros.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "disagg_pd.pb.h"
#include "disagg_pd_scheduler.h"
#include "distributed_runtime/engine.h"
#include "framework/batch/batch_factory.h"
#include "framework/kv_cache_transfer/pd_topology_guard.h"
#include "framework/request/request.h"
#include "framework/request/request_state.h"
#include "framework/request/sequence.h"
#include "framework/xtensor/page_allocator.h"
#include "runtime/xservice_client.h"
#include "scheduler/continuous_scheduler.h"
#include "util/env_var.h"
#include "util/utils.h"

namespace xllm {

DisaggPDScheduler::DisaggPDScheduler(Engine* engine, const Options& options)
    : ContinuousScheduler(engine, options), server_name_("DisaggPDServer") {
  if (!options_.instance_role().has_value()) {
    LOG(FATAL) << "Instance type is not set in disagg pd mode.";
  }

  // Only initialize for non-OOC mode
  // OOC mode (PDOOCScheduler) will handle initialization in its own constructor
  if (!options_.enable_pd_ooc()) {
    // Start dispatch thread for prefill instance
    dispatch_thread_ = std::make_unique<std::thread>(
        &DisaggPDScheduler::dispatch_requests, this);

    // Start RPC server thread
    server_name_.append(std::to_string(options.server_idx()));
    rpc_server_thread_ = std::make_unique<std::thread>(
        &DisaggPDScheduler::start_rpc_server, this);
    initialize_rpc_server(server_name_);
    register_instance_info(server_name_, engine);

    // Profile ttft & topt and update instance info (for mix instances)
    if (!options_.disable_ttft_profiling() &&
        options_.instance_role().value() == InstanceRole::MIX) {
      profile_ttft();
      profile_tpot();
    }
  }
}

DisaggPDScheduler::~DisaggPDScheduler() {
  // Clean up common threads (shared by both OOC and non-OOC modes)
  if (rpc_server_thread_ && rpc_server_thread_->joinable()) {
    rpc_server_thread_->join();
  }

  // Clean up dispatch thread (created in base class for non-OOC mode,
  // or in subclass for OOC mode)
  if (dispatch_thread_ && dispatch_thread_->joinable()) {
    dispatch_thread_->join();
  }

  auto rpc_server = ServerRegistry::get_instance().get_server(server_name_);
  if (rpc_server != nullptr) {
    rpc_server->stop();

    ServerRegistry::get_instance().unregister_server(server_name_);
  }
}

void DisaggPDScheduler::initialize_rpc_server(const std::string& server_name) {
  // wait rpc server initialized
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  while (!rpc_server || !rpc_server->has_initialized()) {
    absl::SleepFor(absl::Milliseconds(100));
    rpc_server = ServerRegistry::get_instance().get_server(server_name);
  }
  // connect to master service
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
  xservice_client_->set_scheduler(this);
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    xservice_client_->set_engine(engine_);
  }
}

void DisaggPDScheduler::register_instance_info(const std::string& server_name,
                                               Engine* engine) {
  // register instance info
  instance_info_.name = xservice_client_->get_instance_name();
  auto rpc_server = ServerRegistry::get_instance().get_server(server_name);
  instance_info_.rpc_address = rpc_server->listen_address();
  instance_info_.type = options_.instance_role().value().to_string();
  LOG(INFO) << "Instance info: instance name = " << instance_info_.name
            << ", instance rpc_address = " << instance_info_.rpc_address
            << ", instance type = " << instance_info_.type;

  engine->get_cache_info(
      instance_info_.cluster_ids, instance_info_.addrs, instance_info_.ports);
  instance_info_.dp_size = options_.dp_size();
  instance_info_.kv_split_size =
      ::xllm::ParallelConfig::get_instance().kv_split_size_effective();

  // Get total physical pages per worker (for etcd registration)
#if defined(USE_NPU)
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    auto& page_allocator = PageAllocator::get_instance();
    if (page_allocator.is_initialized()) {
      instance_info_.total_phy_pages = page_allocator.get_num_total_phy_pages();
    }
  }
#endif
}

void DisaggPDScheduler::profile_ttft() {
  LOG(INFO) << "Start profiling TTFT.";
  // get the maximum prefill token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }

  // warm up
  profile_manager_->run_request(max_context_len, 0);

  // get TTFT starting from max_context_len
  for (int32_t token_length = max_context_len; token_length > 1;
       token_length *= 0.9) {
    double latency = profile_manager_->run_request(token_length, 0);
    instance_info_.ttft_profiling_data.emplace_back(
        std::make_pair(token_length, latency));
  }
}

void DisaggPDScheduler::profile_tpot() {
  LOG(INFO) << "Start profiling TPOT.";
  // get the maximum token length
  auto& model_args = engine_->model_args();
  int32_t max_context_len = model_args.max_position_embeddings();
  if (!options_.enable_chunked_prefill()) {
    max_context_len =
        std::min(max_context_len, options_.max_tokens_per_batch());
  }

  int32_t num_blocks = kv_cache_manager_->num_blocks();
  int32_t block_size = kv_cache_manager_->block_size();
  int32_t max_seqs_per_batch = options_.max_seqs_per_batch();
  int32_t request_blocks = max_context_len / block_size + 1;
  int32_t max_batch_size = num_blocks / request_blocks;

  // warm up
  profile_manager_->run_request(
      max_context_len, max_context_len - 1, max_batch_size);

  // get TPOT starting from max_context_len, dividing the token length by 2 in
  // each loop iteration. Skip small token lengths to speed up profiling.
  for (int32_t token_length = max_context_len; token_length > 64;
       token_length >>= 1) {
    max_batch_size = num_blocks / (token_length / block_size + 1);
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

void DisaggPDScheduler::cache_prefill_blocks(Request* request) {
  CHECK(request != nullptr);
  for (auto& sequence : request->sequences()) {
    if (sequence->if_cache_block_for_prefill()) {
      kv_cache_manager_->cache(sequence.get());
    }
  }
}

// TODO: maybe we should consider update info case even if info already exists
// in local.
bool DisaggPDScheduler::check_remote_instance_info(
    const std::string& instance_name) {
  if (remote_instances_info_.find(instance_name) !=
      remote_instances_info_.end()) {
    return true;
  }

  InstanceInfo instance_info =
      xservice_client_->get_instance_info(instance_name);
  if (instance_info.name.empty()) {
    LOG(ERROR)
        << "Failed to get instance info from master server, instance name: "
        << instance_name;
    return false;
  }

  remote_instances_info_[instance_name] = instance_info;
  return true;
}

proto::DisaggPDService_Stub* DisaggPDScheduler::create_rpc_channel(
    const std::string& instance_name) {
  std::lock_guard<std::mutex> lock(instance_channel_map_mutex_);
  auto it = instance_channel_map_.find(instance_name);
  if (it == instance_channel_map_.end()) {
    LOG(INFO) << "Create rpc channel to instance: " << instance_name;
    // check prefill instance info
    if (!check_remote_instance_info(instance_name)) {
      LOG(ERROR) << "Check remote instance info failed, instance name: "
                 << instance_name;
      return nullptr;
    }
    // create channel to prefill instance
    brpc::Channel* channel = new brpc::Channel();
    brpc::ChannelOptions options;
    options.timeout_ms =
        ::xllm::ServiceConfig::get_instance().rpc_channel_timeout_ms();
    options.max_retry = 3;
    std::string load_balancer = "";
    if (channel->Init(remote_instances_info_[instance_name].rpc_address.c_str(),
                      load_balancer.c_str(),
                      &options) != 0) {
      LOG(ERROR) << "Fail to initialize channel for "
                 << remote_instances_info_[instance_name].rpc_address;
      remote_instances_info_.erase(instance_name);
      delete channel;
      return nullptr;
    }

    proto::DisaggPDService_Stub* stub =
        new proto::DisaggPDService_Stub(channel);
    instance_channel_map_[instance_name] = stub;
    return stub;
  }

  return it->second;
}

void DisaggPDScheduler::start_rpc_server() {
  std::unique_ptr<DisaggPDService> service =
      std::make_unique<DisaggPDService>(this, engine_);
  auto rpc_server =
      ServerRegistry::get_instance().register_server(server_name_);
  if (!rpc_server->start(std::move(service))) {
    LOG(ERROR) << "Failed to start brpc disagg pd server on port "
               << ::xllm::DisaggPDConfig::get_instance().disagg_pd_port();
    return;
  }
}

void DisaggPDScheduler::step(const absl::Duration& timeout) {
  ContinuousScheduler::step(timeout);
  // Send first generation token to decode instance.
  // Always check (not gated on last_step_prefill_) because chunked prefill
  // mode does not set that flag and a chunked prefill may complete at any step.
  // prefill_send_first_generation() internally checks
  // num_generated_tokens() == 1 so spurious calls are harmless.
  if (options_.instance_role() != InstanceRole::DECODE) {
    prefill_send_first_generation();
  }
}

std::vector<Batch> DisaggPDScheduler::prepare_batch() {
  // For PREFILL / MIX in disagg PD, drain newly arrived requests here and
  // skip the eager expand_sequences(false) that the base prepare_batch would
  // otherwise call when enable_prefix_cache is off. For best_of_n requests,
  // expansion to best_of sequences is deferred to the DECODE instance (where
  // prefix cache lets seq[1..best_of-1] reuse seq[0]'s prompt KV). Without
  // this guard, the PREFILL instance would waste N x prefill compute on
  // candidates that are never used.
  const bool is_decode =
      options_.instance_role().has_value() &&
      options_.instance_role().value() == InstanceRole::DECODE;
  if (!is_decode) {
    std::shared_ptr<Request> request;
    while (request_queue_.read(request)) {
      CHECK(request);
      if (request->sequences()[0]->kv_state().kv_cache_tokens_num() == 0) {
        prefill_queue_->push(request);
      } else {
        // request from prefill instance in disagge pd mode.
        running_requests_.emplace_back(request);
      }
    }
  }

  // The BatchMode config (resolved from options) routes scheduling correctly
  // for both chunked-prefill and exclusive modes.
  return ContinuousScheduler::prepare_batch();
}

bool DisaggPDScheduler::add_request(std::shared_ptr<Request>& request) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  kv_cache_manager_->prefetch_from_storage(request);

  if (request->offline()) {
    // offline request, push to offline queue
    prefill_request_queue_offline_.enqueue(request);
    return true;
  }
  // push and wait
  prefill_request_queue_.enqueue(request);

  return true;
}

// prefill send new request to remote instance
void DisaggPDScheduler::dispatch_requests() {
  while (true) {
    const auto timeout = std::chrono::milliseconds(100);
    // Wait for online request until timeout.
    // If timeout, try to get offline request once. If no offline request,
    // continue to wait for online request. This can avoid offline request
    // blocking online request for too long time.
    std::shared_ptr<Request> request;
    if (!prefill_request_queue_.wait_dequeue_timed(request, timeout)) {
      if (!prefill_request_queue_offline_.try_dequeue(request)) {
        continue;
      }
    }

    if (request == nullptr) {
      // nullptr is a signal to exit
      break;
    }

    if (request->state().decode_address.empty()) {
      // No decode address provided to the prefill instance, just finish the
      // request.
      response_processor_->process_failed_request(
          request,
          {StatusCode::INVALID_ARGUMENT,
           "No decode address provided to the prefill instance"});
      continue;
    }

    std::vector<std::shared_ptr<Request>> requests;
    requests.emplace_back(request);
    std::string selected_instance = request->state().decode_address;

    const InstanceInfo remote_info =
        xservice_client_->get_instance_info(selected_instance);
    if (remote_info.name.empty()) {
      response_processor_->process_failed_request(
          request,
          {StatusCode::UNKNOWN, "failed to fetch remote decode instance info"});
      continue;
    }
    remote_instances_info_[selected_instance] = remote_info;

    const bool enable_mla = engine_->model_args().enable_mla();
    const PdTopoResult topo_result =
        check_pd_topo(instance_info_,
                      remote_info,
                      options_.kv_cache_transfer_mode(),
                      enable_mla);
    const bool allow_pd_topo = topo_result.status == PdTopoStatus::ALLOW_HOMO ||
                               topo_result.status == PdTopoStatus::ALLOW_HETERO;
    if (!allow_pd_topo) {
      if (topo_result.status == PdTopoStatus::INVALID_REMOTE) {
        remote_instances_info_.erase(selected_instance);
      }
      response_processor_->process_failed_request(
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

    proto::DisaggPDService_Stub* stub = create_rpc_channel(selected_instance);
    if (stub == nullptr) {
      response_processor_->process_failed_request(
          request, {StatusCode::UNKNOWN, "Fail to create rpc channel"});
      continue;
    }

    // NOTE: TODO: maybe we need to support batch disatch
    // later, this meybe decrease the communication cost.
    // currently we only support one request per dispatch.

    // TODO: try to get a batch request.

    {
      std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
      for (auto& req : requests) {
        req_to_channel_map_[req->request_id()] = stub;
      }
    }

    // TODO: send the request to the selected D instance
    // Send 'DisaggRequests' and recv 'DisaggResponses'
    xllm::proto::DisaggRequests reqs;
    xllm::proto::DisaggResponses resps;
    // prefill name (ID)
    reqs.set_prefill_name(xservice_client_->get_instance_name());
    reqs.mutable_reqs()->Reserve(requests.size());
    // currently we only support one request once.
    for (size_t i = 0; i < requests.size(); ++i) {
      // proto::DisaggRequest req;
      auto req = reqs.mutable_reqs()->Add();
      req->set_req_id(requests[i]->request_id());
      req->set_service_req_id(requests[i]->service_request_id());
      req->set_source_xservice_addr(requests[i]->source_xservice_addr());
      req->set_tokens_num(requests[i]->state().prompt_tokens.size());
      req->set_prompt(requests[i]->state().prompt);
      ADD_VECTOR_TO_PROTO(req->mutable_prompt_tokens(),
                          requests[i]->state().prompt_tokens);
      req->set_stream(requests[i]->state().stream);
      req->set_x_request_id(requests[i]->x_request_id());
      req->set_x_request_time(requests[i]->x_request_time());
      req->set_seq_capacity(requests[i]->state().seq_capacity);
      req->set_max_tokens(
          requests[i]->state().stopping_checker.get_max_generated_tokens());
      req->set_max_context_len(
          requests[i]->state().stopping_checker.get_max_context_len());
      req->set_ignore_eos(
          requests[i]->state().stopping_checker.get_ignore_eos());
      req->set_eos_token_id(
          requests[i]->state().stopping_checker.get_eos_token());
      if (requests[i]->state().stopping_checker.get_stop_tokens().size() > 0) {
        ADD_VECTOR_TO_PROTO(
            req->mutable_stop_token_ids(),
            requests[i]->state().stopping_checker.get_stop_tokens());
      }
      if (requests[i]->state().stopping_checker.get_stop_sequences().size() >
          0) {
        for (auto& stop_sequence :
             requests[i]->state().stopping_checker.get_stop_sequences()) {
          // proto::StopSequence proto_seq;
          auto proto_seq = req->mutable_stop_sequences()->Add();
          ADD_VECTOR_TO_PROTO(proto_seq->mutable_seq_tokens(), stop_sequence);
          //*req->mutable_stop_sequences()->Add() = proto_seq;
        }
      }
      req->set_n(requests[i]->state().n);
      req->set_best_of(requests[i]->state().best_of);
      req->set_frequency_penalty(
          requests[i]->state().sampling_param.frequency_penalty);
      req->set_presence_penalty(
          requests[i]->state().sampling_param.presence_penalty);
      req->set_repetition_penalty(
          requests[i]->state().sampling_param.repetition_penalty);
      req->set_temperature(requests[i]->state().sampling_param.temperature);
      req->set_top_p(requests[i]->state().sampling_param.top_p);
      req->set_top_k(requests[i]->state().sampling_param.top_k);
      req->set_logprobs(requests[i]->state().sampling_param.logprobs);
      req->set_top_logprobs(requests[i]->state().sampling_param.top_logprobs);
      req->set_is_embeddings(requests[i]->state().sampling_param.is_embeddings);
      req->set_echo(requests[i]->state().echo);
      req->set_skip_special_tokens(requests[i]->state().skip_special_tokens);
      //*reqs.mutable_reqs()->Add() = req;
    }
    reqs.mutable_cluster_infos()->mutable_cluster_ids()->Add(
        instance_info_.cluster_ids.begin(), instance_info_.cluster_ids.end());
    reqs.mutable_cluster_infos()->mutable_addrs()->Add(
        instance_info_.addrs.begin(), instance_info_.addrs.end());
    reqs.mutable_cluster_infos()->mutable_ports()->Add(
        instance_info_.ports.begin(), instance_info_.ports.end());
    reqs.mutable_cluster_infos()->set_dp_size(options_.dp_size());

    // TODO: sync rpc here currently
    brpc::Controller cntl;
    stub->AddNewRequests(&cntl, &reqs, &resps, nullptr);
    if (cntl.Failed()) {
      LOG(ERROR) << "Failed to add new requests to decode instance : "
                 << selected_instance << ", error text : " << cntl.ErrorText();
      for (auto& request : requests) {
        response_processor_->process_failed_request(
            request,
            {StatusCode::UNKNOWN,
             "Failed to add new requests to decode instance"});

        {
          std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
          req_to_channel_map_.erase(request->request_id());
        }
      }
      continue;
    }

    // check reqs which can not dispatch to D instance,
    // and push back to prefill_request_queue_
    CHECK_EQ(requests.size(), resps.resps().size())
        << "selected_instance : " << selected_instance;
    // insert instance name to linked_instance_
    {
      std::lock_guard<std::mutex> lock(linked_instances_mutex_);
      linked_instance_.emplace(selected_instance);
    }
    for (size_t i = 0; i < requests.size(); ++i) {
      if (resps.resps()[i].status_code() != 200) {
        // push back to prefill_request_queue_
        if (requests[i]->offline()) {
          prefill_request_queue_offline_.enqueue(requests[i]);
        } else {
          prefill_request_queue_.enqueue(requests[i]);
        }

      } else {
        for (auto& sequence : requests[i]->sequences()) {
          TransferKVInfo info;
          info.request_id = requests[i]->request_id();
          const auto& resp = resps.resps()[i];
          const bool has_grouped_cache = resp.kv_block_groups_size() > 0;
          if (has_grouped_cache) {
            for (const auto& resp_group : resp.kv_block_groups()) {
              KVBlockTransferGroup group;
              group.group_id = resp_group.group_id();
              group.remote_blocks_ids.reserve(resp_group.block_ids_size());
              for (const int32_t block_id : resp_group.block_ids()) {
                group.remote_blocks_ids.emplace_back(
                    static_cast<uint64_t>(block_id));
              }
              info.block_transfer_groups.emplace_back(std::move(group));
            }
          } else {
            for (const int32_t block_id : resp.blocks_ids()) {
              info.remote_blocks_ids.emplace_back(
                  static_cast<uint64_t>(block_id));
            }
          }
          if (resp.linear_state_id() >= 0) {
            info.remote_linear_state_ids.emplace_back(resp.linear_state_id());
          }
          if (!has_grouped_cache) {
            const size_t prompt_blocks =
                (requests[i]->state().prompt_tokens.size() +
                 kv_cache_manager_->block_size() - 1) /
                kv_cache_manager_->block_size();
            info.local_blocks_ids.resize(prompt_blocks);
          }
          info.dp_rank = resps.resps()[i].dp_rank();
          // TODO: remote_instances_info_ is not multi-thread safe.
          info.remote_instance_info = remote_instances_info_[selected_instance];

          // XTensor mode: save destination offsets from D-node
          if (resp.xtensor_layer_offsets_size() > 0) {
            info.dst_xtensor_layer_offsets.reserve(
                resp.xtensor_layer_offsets_size());
            for (const auto& layer_offsets : resp.xtensor_layer_offsets()) {
              XTensorLayerOffsets layer;
              layer.k_offsets.assign(layer_offsets.k_offsets().begin(),
                                     layer_offsets.k_offsets().end());
              layer.v_offsets.assign(layer_offsets.v_offsets().begin(),
                                     layer_offsets.v_offsets().end());
              info.dst_xtensor_layer_offsets.emplace_back(std::move(layer));
            }
            VLOG(5) << "Received XTensor offsets from D-node for request "
                    << requests[i]->request_id()
                    << ", num_layers=" << info.dst_xtensor_layer_offsets.size();
          }

          sequence->kv_state().set_transfer_kv_info(std::move(info));
        }

        // Push to request_queue_; it will be executed by the engine.
        request_queue_.write(requests[i]);
      }
    }
  }
}

void DisaggPDScheduler::prefill_send_first_generation() {
  if (running_sequences_.size() == 0) {
    return;
  }

  std::vector<std::shared_ptr<Request>> requests;
  std::vector<std::shared_ptr<Request>> non_stream_requests;
  requests.reserve(running_requests_.size());
  non_stream_requests.reserve(running_requests_.size());
  for (size_t i = 0; i < running_requests_.size(); ++i) {
    auto request = running_requests_[i];
    // Check if the request is a recently completed prefill request
    if (request->sequences()[0]->num_generated_tokens() == 1) {
      if (!options_.disable_log_stats()) {
        request->log_statistic(request->elapsed_seconds());
      }
      requests.emplace_back(request);
      if (!request->state().stream) {
        non_stream_requests.emplace_back(request);
      }
      running_requests_[i] = nullptr;
    }
  }
  // call non_stream_request's callback in P instance when its prefill ends
  if (!non_stream_requests.empty()) {
    response_processor_->process_completed_requests(non_stream_requests);
  }

  // No prefill request needs to be transferred to decode.
  if (requests.size() == 0) {
    return;
  }

  prefill_threadpool_.schedule([this,
                                requests = std::move(requests)]() mutable {
    // send request first token to remote instance
    // TODO: here we only support one sequence for now.
    auto fail_request = [this](const std::shared_ptr<Request>& request,
                               Status status) {
      response_processor_->process_failed_request(request, status);
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      response_processor_->wait_completion();
      kv_cache_manager_->deallocate(request.get());
    };
    for (auto& request : requests) {
      // TODO: support batch request later
      proto::DisaggGenerationsRequests gens;
      auto gen = gens.mutable_multi_gens()->Add();
      gen->set_req_id(request->request_id());
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
      if (options_.kv_cache_transfer_mode() == "PULL") {
        ADD_VECTOR_TO_PROTO(gen->mutable_cluster_ids(),
                            instance_info_.cluster_ids);
        ADD_VECTOR_TO_PROTO(gen->mutable_addrs(), instance_info_.addrs);

        const auto blocks =
            request->sequences()[0]->kv_state().blocks(BlockType::KV);
        std::vector<uint64_t> block_ids;
        block_ids.reserve(blocks.size());
        for (const auto& block : blocks) {
          block_ids.push_back(block.id());
        }
        ADD_VECTOR_TO_PROTO(gen->mutable_block_ids(), block_ids);
        gen->set_linear_state_id(
            request->sequences()[0]->get_single_block_id());
        gen->set_dp_size(instance_info_.dp_size);
        gen->set_dp_rank(request->sequences()[0]->dp_rank());
      }
      if (options_.num_speculative_tokens() > 0) {
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

      // send first gens to remote instance
      proto::DisaggPDService_Stub* stub = nullptr;
      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        // now we only support one request once.
        stub = req_to_channel_map_[request->request_id()];
      }

      // TODO: Async call later
      proto::Status resp;
      brpc::Controller cntl;
      stub->FirstGeneration(&cntl, &gens, &resp, nullptr);

      const bool sent_first_generation = !cntl.Failed() && resp.ok();
      if (!sent_first_generation) {
        LOG(ERROR) << "Failed to send first generation to decode instance : "
                   << request->state().decode_address
                   << ", error text : " << cntl.ErrorText()
                   << ", response status: " << resp.ok();
      }

      {
        std::lock_guard<std::mutex> lock(req_to_channel_map_mutex_);
        req_to_channel_map_.erase(request->request_id());
      }
      response_processor_->wait_completion();
      if (sent_first_generation) {
        cache_prefill_blocks(request.get());
      }
      kv_cache_manager_->deallocate(request.get());
    }
  });
}

// request is received from prefill
bool DisaggPDScheduler::decode_schedule(
    std::shared_ptr<Request>& request,
    const std::string& prefill_instance_name) {
  CHECK(request != nullptr);
  CHECK(!request->sequences().empty());

  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    if (received_request_map_.find(request->request_id()) !=
        received_request_map_.end()) {
      LOG(ERROR) << "Decode receive duplicate request_id from prefill: "
                 << request->request_id();
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
    received_request_map_[request->request_id()] = request;
    instance_to_received_requests_map_[prefill_instance_name].insert(
        request->request_id());
    request_to_instance_map_[request->request_id()] = prefill_instance_name;
  }

  return true;
}

bool DisaggPDScheduler::decode_recv_first_generation(
    const std::string& req_id,
    int64_t token_id,
    bool has_logprob,
    float logprob,
    double time_to_first_token_latency_seconds,
    std::vector<int64_t> top_tokens,
    std::vector<float> top_logprobs,
    const std::string& kv_cache_transfer_mode,
    std::vector<uint64_t> src_cluster_ids,
    std::vector<std::string> src_addrs,
    std::vector<uint64_t> src_block_ids,
    int32_t src_linear_state_id,
    int32_t src_dp_size,
    int32_t src_dp_rank,
    torch::Tensor mtp_bootstrap_embedding,
    int32_t num_cached_tokens) {
  // push to request_queue_, and will be executed by engine.
  std::shared_ptr<Request> request = nullptr;
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = received_request_map_.find(req_id);
    if (it == received_request_map_.end()) {
      LOG(ERROR) << "Failed to find request, request id: " << req_id;
      return false;
    }
    request = it->second;
    received_request_map_.erase(it);

    auto inst_it = request_to_instance_map_.find(req_id);
    if (inst_it != request_to_instance_map_.end()) {
      instance_to_received_requests_map_[inst_it->second].erase(req_id);
      request_to_instance_map_.erase(inst_it);
    }
  }
  auto& sequences = request->sequences();
  if (sequences.empty() || sequences[0] == nullptr) {
    LOG(ERROR) << "Request has no valid sequences, request_id: " << req_id;
    for (auto& sequence : sequences) {
      if (sequence != nullptr) {
        kv_cache_manager_->deallocate(sequence.get());
      }
    }
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
  const bool need_mtp_bootstrap = options_.num_speculative_tokens() > 0;
  if (need_mtp_bootstrap) {
    const int32_t slot_id = sequence->get_single_block_id();
    if (slot_id < 0) {
      LOG(ERROR) << "Invalid MTP bootstrap slot, request_id: " << req_id;
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
    if (token_id < 0 ||
        token_id > static_cast<int64_t>(std::numeric_limits<int32_t>::max())) {
      LOG(ERROR) << "Invalid MTP bootstrap token, request_id: " << req_id
                 << ", token_id: " << token_id;
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
    if (!mtp_bootstrap_embedding.defined()) {
      LOG(ERROR) << "Missing MTP bootstrap embedding, request_id: " << req_id;
      kv_cache_manager_->deallocate(request.get());
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
  // Enable checking whether to skip the prefill token
  if (request->state().stream) {
    sequence->enable_checking_prefill_token();
  }

  // update latency metrics
  sequence->set_time_to_first_token_latency_seconds(
      time_to_first_token_latency_seconds);
  // Rebase the ITL clock to the moment Decode receives the first token. The
  // prefill->decode transfer cost is attributed to TTFT, not ITL;
  // reconstructing prefill's first-token timestamp would require cross-machine
  // clock sync.
  sequence->tbt(absl::Now());

  // TODO: we only support one sequence for currently.
  if (enable_schedule_overlap()) {
    Token fake_token(-1);
    sequence->append_token(fake_token);
    sequence->update_last_step_token(first_token);
  } else {
    sequence->append_token(first_token);
  }

  // pull kv cache
  if (kv_cache_transfer_mode == "PULL") {
    const auto blocks = sequence->kv_state().blocks(BlockType::KV);
    std::vector<uint64_t> dst_block_ids;
    dst_block_ids.reserve(blocks.size());
    for (const auto& block : blocks) {
      dst_block_ids.push_back(block.id());
    }
    std::vector<uint64_t> src_linear_state_ids;
    std::vector<uint64_t> dst_linear_state_ids;
    if (src_linear_state_id >= 0 && sequence->get_single_block_id() >= 0) {
      src_linear_state_ids.emplace_back(src_linear_state_id);
      dst_linear_state_ids.emplace_back(sequence->get_single_block_id());
    }

    int32_t dst_dp_rank = sequence->dp_rank();
    const bool pulled = engine_->pull_kv_blocks(src_dp_size,
                                                src_dp_rank,
                                                src_cluster_ids,
                                                src_addrs,
                                                src_block_ids,
                                                dst_dp_rank,
                                                dst_block_ids,
                                                src_linear_state_ids,
                                                dst_linear_state_ids);
    if (!pulled) {
      LOG(ERROR) << "Failed to pull KV blocks, request_id: " << req_id;
      kv_cache_manager_->deallocate(request.get());
      return false;
    }
  }

  if (!request_queue_.write(request)) {
    LOG(ERROR) << "Failed to enqueue decode request, request_id: " << req_id;
    kv_cache_manager_->deallocate(request.get());
    return false;
  }
  return true;
}

bool DisaggPDScheduler::try_allocate(Sequence* sequence) {
  // When the KV Cache usage reaches the threshold, prefill requests will no
  // longer be scheduled to avoid frequent preemption.
  if (kv_cache_manager_->kv_cache_utilization() <
      ::xllm::SchedulerConfig::get_instance()
          .prefill_scheduling_memory_usage_threshold()) {
    return kv_cache_manager_->try_allocate(sequence);
  } else {
    return false;
  }
}

void DisaggPDScheduler::update_token_latency_metrics(
    std::vector<Sequence*>& sequences) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);

  const auto now = absl::Now();
  const bool speculative_metrics_enabled =
      options_.num_speculative_tokens() > 0;
  int64_t step_committed_tokens = 0;
  int64_t step_decode_seqs = 0;
  for (Sequence* sequence : sequences) {
    if (sequence->is_chunked_prefill_stage() ||
        sequence->last_token_handled()) {
      continue;
    }
    // Read the committed-token count before tbt(), which resets it.
    const size_t committed_tokens = sequence->generated_tokens_since_latency();
    int64_t tbt_milliseconds = sequence->tbt(now);
    if (sequence->is_first_token()) {
      HISTOGRAM_OBSERVE(time_to_first_token_latency_milliseconds,
                        tbt_milliseconds);
      sequence->set_time_to_first_token_latency_seconds(
          static_cast<double>(tbt_milliseconds) / 1000);
      recent_ttft_.emplace_back(tbt_milliseconds);
    } else {
      HISTOGRAM_OBSERVE(inter_token_latency_milliseconds, tbt_milliseconds);
      recent_tbt_.emplace_back(tbt_milliseconds);
      if (speculative_metrics_enabled && committed_tokens > 0) {
        HISTOGRAM_OBSERVE(
            speculative_per_token_latency_milliseconds,
            amortized_token_latency_ms(tbt_milliseconds, committed_tokens));
        step_committed_tokens += static_cast<int64_t>(committed_tokens);
        ++step_decode_seqs;
      }
    }
  }
  if (step_decode_seqs > 0) {
    GAUGE_SET(speculative_mean_tokens_per_decode_step,
              static_cast<double>(step_committed_tokens) / step_decode_seqs);
  }
}

void DisaggPDScheduler::get_latency_metrics(std::vector<int64_t>& ttft,
                                            std::vector<int64_t>& tbt) {
  std::lock_guard<std::mutex> lock(latency_metrics_mutex_);
  ttft = std::move(recent_ttft_);
  tbt = std::move(recent_tbt_);
}

bool DisaggPDScheduler::link_instance(const std::string& instance_name,
                                      const std::vector<uint64_t>& cluster_ids,
                                      const std::vector<std::string>& addrs,
                                      const std::vector<uint16_t>& ports,
                                      const int32_t dp_size,
                                      const int32_t src_kv_split_size) {
  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (!engine_->link_cluster(
          cluster_ids, addrs, ports, dp_size, src_kv_split_size)) {
    LOG(ERROR) << "Link instance failed, instance_name: " << instance_name;
    return false;
  }
  LOG(INFO) << "Successfully linked instance, instance_name: " << instance_name
            << ", prefill_kv_split_size: " << src_kv_split_size;
  linked_instance_.emplace(instance_name);
  return true;
}

bool DisaggPDScheduler::unlink_instance(
    const std::string& instance_name,
    const std::vector<uint64_t>& cluster_ids,
    const std::vector<std::string>& addrs,
    const std::vector<uint16_t>& ports,
    const int32_t dp_size,
    const int32_t src_kv_split_size) {
  // Clear received requests from this instance
  {
    std::lock_guard<std::mutex> lock(received_request_map_mutex_);
    auto it = instance_to_received_requests_map_.find(instance_name);
    if (it != instance_to_received_requests_map_.end()) {
      for (const auto& req_id : it->second) {
        received_request_map_.erase(req_id);
        request_to_instance_map_.erase(req_id);
      }
      instance_to_received_requests_map_.erase(it);
    }
  }

  std::lock_guard<std::mutex> lock(linked_instances_mutex_);
  if (!engine_->unlink_cluster(
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
