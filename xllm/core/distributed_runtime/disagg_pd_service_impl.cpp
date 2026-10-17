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

#include "core/distributed_runtime/disagg_pd_service_impl.h"

#include <glog/logging.h>

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include "common/types.h"
#include "core/distributed_runtime/disagg_pd_kv_response.h"
#include "core/distributed_runtime/disagg_pd_request_converter.h"
#include "core/distributed_runtime/distributed_virtual_memory_coordinator.h"
#include "core/distributed_runtime/engine.h"
#include "core/framework/block/block_manager_pool.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/sampling/json_object_grammar.h"
#include "framework/request/request_output.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "util/utils.h"

namespace xllm {

DisaggPDServiceImpl::DisaggPDServiceImpl(DisaggPDScheduler* scheduler,
                                         Engine* engine,
                                         int32_t /*kv_cache_block_tokens*/)
    : scheduler_(scheduler),
      engine_(engine),
      block_manager_(engine->block_manager_pool()) {
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
}

std::shared_ptr<const JsonObjectGrammar>
DisaggPDServiceImpl::get_json_object_grammar(bool reasoning_enabled,
                                             std::string* error) {
  std::lock_guard<std::mutex> lock(json_object_grammar_mutex_);
  std::shared_ptr<const JsonObjectGrammar>& grammar =
      reasoning_enabled ? json_reasoning_grammar_ : json_object_grammar_;
  if (grammar == nullptr) {
    grammar = JsonObjectGrammar::create_from_tokenizer(
        *engine_->tokenizer(),
        engine_->model_args().eos_token_id(),
        engine_->model_args().stop_token_ids(),
        engine_->model_args().vocab_size(),
        reasoning_enabled,
        error);
  }
  return grammar;
}

std::shared_ptr<Request> DisaggPDServiceImpl::generate_request(
    const proto::DisaggRequest& req) {
  // best_of > 1 in disaggregated PD requires prefix cache on the decode
  // instance: only the first sequence's KV is pulled from prefill, and
  // the remaining best_of-1 sequences are expanded locally and reuse the
  // first sequence's prompt KV via prefix cache.
  if (req.best_of() > 1 &&
      !::xllm::KVCacheConfig::get_instance().enable_prefix_cache()) {
    LOG(ERROR) << "best_of > 1 in disaggregated PD mode requires "
               << "enable_prefix_cache=true on the decode instance, "
               << "request_id=" << req.req_id()
               << ", best_of=" << req.best_of();
    return nullptr;
  }

  std::string prompt = req.prompt();
  std::vector<int32_t> prompt_tokens(req.prompt_tokens().begin(),
                                     req.prompt_tokens().end());

  RequestSamplingParam sampling_param = make_disagg_sampling_param(req);
  const bool json_object = sampling_param.json_object;
  SchedulerParam scheduler_param = make_disagg_scheduler_param(req);
  StoppingChecker stopping_checker = make_disagg_stopping_checker(req);

  auto output_callback = [this](const RequestOutput& output) -> bool {
    // response to xllm service to avoid the redirect cost.
    if (xservice_client_ == nullptr) {
      return false;
    }
    return xservice_client_->generation(output);
  };

  auto batch_output_callback =
      [this](const std::vector<RequestOutput>& outputs) {
        // response to xllm service to avoid the redirect cost.
        if (xservice_client_ == nullptr) {
          return std::vector<bool>(outputs.size(), false);
        }
        return xservice_client_->generations(outputs);
      };

  RequestState req_state(std::move(prompt),
                         std::move(prompt_tokens),
                         std::move(sampling_param),
                         std::move(scheduler_param),
                         std::move(stopping_checker),
                         req.seq_capacity(),
                         req.n(),
                         req.best_of(),
                         req.logprobs(),
                         req.stream(),
                         req.echo(),
                         req.skip_special_tokens(),
                         scheduler_->enable_schedule_overlap(),
                         output_callback,
                         batch_output_callback);
  req_state.include_stop_str_in_output = req.include_stop_str_in_output();

  if (json_object) {
    std::string grammar_error;
    req_state.json_object_grammar =
        get_json_object_grammar(req.json_reasoning_enabled(), &grammar_error);
    if (req_state.json_object_grammar == nullptr) {
      LOG(ERROR) << "Failed to initialize json_object constraint: "
                 << grammar_error << ", request_id=" << req.req_id();
      return nullptr;
    }
    req_state.json_reasoning_enabled = req.json_reasoning_enabled();
  }

  req_state.pd_reservation_id = req.reservation_id();
  auto new_request = std::make_shared<Request>(req.req_id(),
                                               req.x_request_id(),
                                               req.x_request_time(),
                                               std::move(req_state),
                                               req.service_req_id(),
                                               req.source_xservice_addr());

  // add one sequence, rest will be added by scheduler
  return new_request;
}

void DisaggPDServiceImpl::release_reservation(
    const proto::ReleaseReservationRequest* request,
    proto::ReleaseReservationResponse* response) {
  const bool released = scheduler_->release_reservation(
      request->req_id(), request->reservation_id());
  response->set_result(released
                           ? proto::ReleaseReservationResponse::RELEASED
                           : proto::ReleaseReservationResponse::NOT_WAITING);
}

bool DisaggPDServiceImpl::append_virtual_kv_cache_offsets(
    int32_t dp_rank,
    const std::string& request_id,
    const std::vector<int32_t>& block_ids,
    proto::DisaggResponse* response) {
  if (block_manager_ == nullptr ||
      !block_manager_->options().enable_virtual_memory() || block_ids.empty()) {
    return true;
  }

  std::vector<VirtualKVCacheLayerGlobalOffsets> layer_offsets;
  const auto& block_options = block_manager_->options();
  const bool offsets_available =
      DistributedVirtualMemoryCoordinator::get_instance()
          .get_virtual_kv_cache_offsets(
              dp_rank, block_options.model_id(), block_ids, layer_offsets);
  if (!offsets_available) {
    LOG(ERROR) << "Failed to get VirtualMemory offsets for request "
               << request_id;
    return false;
  }

  auto* layers = response->mutable_virtual_kv_cache_layer_global_offsets();
  layers->Reserve(layers->size() + static_cast<int32_t>(layer_offsets.size()));
  for (const auto& offsets : layer_offsets) {
    auto* layer = layers->Add();
    layer->mutable_k_offsets()->Reserve(
        static_cast<int32_t>(offsets.k_offsets.size()));
    layer->mutable_v_offsets()->Reserve(
        static_cast<int32_t>(offsets.v_offsets.size()));
    layer->mutable_index_offsets()->Reserve(
        static_cast<int32_t>(offsets.index_offsets.size()));
    for (uint64_t offset : offsets.k_offsets) {
      layer->add_k_offsets(offset);
    }
    for (uint64_t offset : offsets.v_offsets) {
      layer->add_v_offsets(offset);
    }
    for (uint64_t offset : offsets.index_offsets) {
      layer->add_index_offsets(offset);
    }
  }
  VLOG(5) << "VirtualMemory offsets added for request " << request_id
          << ", num_blocks=" << block_ids.size()
          << ", num_layers=" << layer_offsets.size();
  return true;
}

void DisaggPDServiceImpl::decode_recv_new_requests(
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response) {
  response->mutable_resps()->Reserve(response->resps_size() +
                                     request->reqs_size());
  for (const auto& req : request->reqs()) {
    auto* resp = response->add_resps();
    resp->set_req_id(req.req_id());

    auto new_request = generate_request(req);
    if (new_request == nullptr) {
      resp->set_status_code(500);
      continue;
    }

    Sequence* sequence = new_request->sequences()[0].get();
    if (!scheduler_->try_allocate(sequence)) {
      if (scheduler_->exceeds_decode_capacity(sequence)) {
        LOG(ERROR) << "Decode cannot fit request prompt at any load, "
                   << "request_id=" << req.req_id()
                   << ", prompt_tokens=" << sequence->num_prompt_tokens();
        resp->set_status_code(kDecodeAddNewPromptTooLongStatusCode);
      } else {
        // Current cache pressure can be relieved, so Prefill may retry.
        resp->set_status_code(404);
      }
      continue;
    }

    resp->set_dp_rank(sequence->dp_rank());
    std::vector<int32_t> virtual_memory_block_ids;
    append_disagg_kv_transfer_groups(
        sequence->kv_state(),
        has_linear_attention_layers(engine_->model_args()),
        resp,
        block_manager_ != nullptr &&
                block_manager_->options().enable_virtual_memory()
            ? &virtual_memory_block_ids
            : nullptr);
    if (!append_virtual_kv_cache_offsets(sequence->dp_rank(),
                                         req.req_id(),
                                         virtual_memory_block_ids,
                                         resp)) {
      engine_->block_manager_pool()->deallocate_without_cache(sequence);
      resp->clear_groups();
      resp->clear_virtual_kv_cache_layer_global_offsets();
      resp->set_status_code(500);
      continue;
    }

    // Publish the reservation only after its transfer metadata is complete.
    if (!scheduler_->decode_schedule(new_request, request->prefill_name())) {
      LOG(ERROR) << "Failed to schedule new decode instance request: "
                 << req.req_id();
      // Request and blocks are released in scheduler.
      resp->clear_groups();
      resp->clear_virtual_kv_cache_layer_global_offsets();
      resp->set_status_code(500);
      continue;
    }
    resp->set_status_code(200);
  }
}

// TODO: support embedding later, now we only support tokens
void DisaggPDServiceImpl::decode_recv_first_generation(
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response) {
  // TODO: we only support one request generation currently
  for (auto& gen : request->multi_gens()) {
    // Process the first token from the tokens array
    if (gen.tokens().empty()) {
      response->set_ok(false);
      return;
    }

    const auto& first_token = gen.tokens(0);
    std::vector<int64_t> top_tokens(first_token.top_tokens().begin(),
                                    first_token.top_tokens().end());
    std::vector<float> top_logprobs(first_token.top_logprobs().begin(),
                                    first_token.top_logprobs().end());
    std::vector<uint64_t> cluster_ids(gen.cluster_ids().begin(),
                                      gen.cluster_ids().end());
    std::vector<std::string> addrs(gen.addrs().begin(), gen.addrs().end());
    std::vector<KVTransferMapping> source_mappings;
    source_mappings.reserve(gen.source_groups_size());
    for (const proto::KVTransferGroup& proto_group : gen.source_groups()) {
      KVTransferMapping mapping;
      mapping.group_id = proto_group.group_id();
      mapping.remote_ids.assign(proto_group.ids().begin(),
                                proto_group.ids().end());
      source_mappings.emplace_back(std::move(mapping));
    }
    torch::Tensor mtp_bootstrap_embedding;
    if (gen.has_mtp_bootstrap_embedding()) {
      mtp_bootstrap_embedding =
          util::proto_to_torch(gen.mtp_bootstrap_embedding());
    }

    bool success = scheduler_->decode_recv_first_generation(
        gen.req_id(),
        gen.reservation_id(),
        first_token.token_id(),
        first_token.has_logprob(),
        first_token.logprob(),
        first_token.time_to_first_token_latency_seconds(),
        gen.upstream_elapsed_seconds(),
        std::move(top_tokens),
        std::move(top_logprobs),
        gen.kv_cache_transfer_mode(),
        std::move(cluster_ids),
        std::move(addrs),
        std::move(source_mappings),
        gen.dp_size(),
        gen.dp_rank(),
        mtp_bootstrap_embedding,
        gen.num_cached_tokens());
    if (!success) {
      response->set_ok(false);
      return;
    }
  }

  response->set_ok(true);
}

void DisaggPDServiceImpl::link_instance(
    const proto::InstanceClusterInfo* request,
    proto::Status* response) {
  // Extract parameters from proto request
  std::vector<uint64_t> cluster_ids(request->cluster_ids().begin(),
                                    request->cluster_ids().end());
  std::vector<std::string> addrs(request->addrs().begin(),
                                 request->addrs().end());
  std::vector<uint16_t> ports(request->ports().begin(), request->ports().end());
  int32_t dp_size = request->dp_size();
  int32_t kv_split_size = request->kv_split_size();

  // Call scheduler's link_instance method
  bool success = scheduler_->link_instance(request->instance_name(),
                                           cluster_ids,
                                           addrs,
                                           ports,
                                           dp_size,
                                           kv_split_size);

  if (!success) {
    LOG(ERROR) << "Failed to link instance: " << request->instance_name();
    response->set_ok(false);
    return;
  }

  response->set_ok(true);
}

void DisaggPDServiceImpl::unlink_instance(
    const proto::InstanceClusterInfo* request,
    proto::Status* response) {
  // Extract parameters from proto request
  std::vector<uint64_t> cluster_ids(request->cluster_ids().begin(),
                                    request->cluster_ids().end());
  std::vector<std::string> addrs(request->addrs().begin(),
                                 request->addrs().end());
  std::vector<uint16_t> ports(request->ports().begin(), request->ports().end());
  int32_t dp_size = request->dp_size();
  int32_t kv_split_size = request->kv_split_size();

  // Call scheduler's unlink_instance method
  bool success = scheduler_->unlink_instance(request->instance_name(),
                                             cluster_ids,
                                             addrs,
                                             ports,
                                             dp_size,
                                             kv_split_size);

  if (!success) {
    LOG(ERROR) << "Failed to unlink instance: " << request->instance_name();
    response->set_ok(false);
    return;
  }

  response->set_ok(true);
}

}  // namespace xllm
