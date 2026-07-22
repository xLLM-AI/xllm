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

#include "disagg_pd_service_impl.h"

#include <glog/logging.h>

#include "common/global_flags.h"
#include "common/types.h"
#include "core/framework/config/kv_cache_config.h"
#include "distributed_runtime/llm_engine.h"
#include "framework/request/request_output.h"
#include "scheduler/disagg_pd_scheduler.h"
#include "util/utils.h"

namespace xllm {

DisaggPDServiceImpl::DisaggPDServiceImpl(DisaggPDScheduler* scheduler,
                                         Engine* engine)
    : scheduler_(scheduler), engine_(engine) {
  xservice_client_ = XServiceClient::get_instance();
  if (!xservice_client_->initialize_done()) {
    LOG(FATAL) << "XServiceClient not init.";
    return;
  }
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
  std::vector<int> prompt_tokens(req.prompt_tokens().begin(),
                                 req.prompt_tokens().end());

  RequestSamplingParam sampling_param;
  sampling_param.frequency_penalty = req.frequency_penalty();
  sampling_param.presence_penalty = req.presence_penalty();
  sampling_param.repetition_penalty = req.repetition_penalty();
  sampling_param.temperature = req.temperature();
  sampling_param.top_p = req.top_p();
  sampling_param.top_k = req.top_k();
  sampling_param.logprobs = req.logprobs();
  sampling_param.top_logprobs = req.top_logprobs();
  sampling_param.is_embeddings = req.is_embeddings();

  SchedulerParam scheduler_param;
  scheduler_param.offline = req.offline();
  scheduler_param.priority = static_cast<xllm::RequestPriority>(req.priority());
  if (!req.offline()) {
    scheduler_param.ttft_slo_ms = req.ttft_slo_ms();
    scheduler_param.tpot_slo_ms = req.tpot_slo_ms();
    scheduler_param.ttlt_slo_ms = req.ttlt_slo_ms();
    scheduler_param.tpot_priority_weight = req.tpot_priority_weight();
    scheduler_param.ttft_priority_weight = req.ttft_priority_weight();
    scheduler_param.ttlt_priority_weight = req.ttlt_priority_weight();
    scheduler_param.priority_weight = req.priority_weight();
  }

  std::unordered_set<int32_t> stop_tokens;
  for (auto& stop_token_id : req.stop_token_ids()) {
    stop_tokens.insert(stop_token_id);
  }
  std::vector<std::vector<int32_t>> stop_sequences;
  for (auto& stop_sequence : req.stop_sequences()) {
    auto stop_seq_tokens = std::vector<int32_t>(
        stop_sequence.seq_tokens().begin(), stop_sequence.seq_tokens().end());
    stop_sequences.push_back(std::move(stop_seq_tokens));
  }
  StoppingChecker stopping_checker(req.max_tokens(),
                                   req.max_context_len(),
                                   req.eos_token_id(),
                                   req.ignore_eos(),
                                   std::move(stop_tokens),
                                   std::move(stop_sequences));

  auto output_callback = [this](const RequestOutput& output) -> bool {
    // response to xllm service to avoid the redirect cost.
    if (xservice_client_ == nullptr) return false;
    auto return_status = xservice_client_->generations({output});
    CHECK_EQ(return_status.size(), 1)
        << "return size of generations is not equal to 1";
    return return_status[0];
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

  auto new_request = std::make_shared<Request>(req.req_id(),
                                               req.x_request_id(),
                                               req.x_request_time(),
                                               std::move(req_state),
                                               req.service_req_id(),
                                               req.source_xservice_addr());

  // add one sequence, rest will be added by scheduler
  return new_request;
}

void DisaggPDServiceImpl::decode_recv_new_requests(
    const proto::DisaggRequests* request,
    proto::DisaggResponses* response) {
  for (auto& req : request->reqs()) {
    auto resp = response->add_resps();
    resp->set_req_id(req.req_id());

    auto new_request = generate_request(req);
    if (new_request == nullptr) {
      resp->set_status_code(500);
      continue;
    }

    auto& sequences = new_request->sequences();
    Sequence* sequence = sequences[0].get();

    if (!scheduler_->try_allocate(sequence)) {
      // FIXME: set status code
      resp->set_status_code(404);
    } else {
      // push the request to scheduler request buffer
      bool success =
          scheduler_->decode_schedule(new_request, request->prefill_name());
      if (!success) {
        LOG(ERROR) << "Failed to schedule new decode instance request: "
                   << req.req_id();
        // request and blocks are released in scheduler
        resp->set_status_code(500);
        continue;
      }

      auto dp_rank = sequence->dp_rank();
      resp->set_dp_rank(dp_rank);
      resp->set_linear_state_id(sequence->get_single_block_id());

      std::vector<int32_t> block_ids;
      if (sequence->kv_state().has_multi_block_export()) {
        const auto export_view = sequence->kv_state().multi_block_export_view();
        for (const auto& [block_type, blocks_ptr] : export_view) {
          auto* group = resp->mutable_kv_block_groups()->Add();
          group->set_group_id(cache_group_id(block_type));
          group->mutable_block_ids()->Reserve(blocks_ptr->size());
          for (const auto& block : *blocks_ptr) {
            CHECK(block.is_valid())
                << "Decode allocated an invalid grouped KV block, group_id="
                << cache_group_id(block_type);
            group->mutable_block_ids()->Add(block.id());
          }
        }
      } else {
        const size_t shared_num =
            sequence->kv_state().shared_blocks_num(BlockType::KV);
        const auto blocks = sequence->kv_state().blocks(BlockType::KV);

        // Collect block IDs
        block_ids.reserve(blocks.size() - shared_num);
        for (size_t i = shared_num; i < blocks.size(); i++) {
          int32_t block_id = blocks[i].id();
          *(resp->mutable_blocks_ids()->Add()) = block_id;
          block_ids.push_back(block_id);
        }
      }
      // XTensor mode: calculate and return GlobalXTensor offsets
      if (::xllm::KVCacheConfig::get_instance().enable_xtensor() &&
          !block_ids.empty()) {
        std::vector<std::pair<std::vector<uint64_t>, std::vector<uint64_t>>>
            layer_offsets;
        if (engine_->get_xtensor_offsets_for_blocks(
                dp_rank, block_ids, layer_offsets)) {
          // Fill proto with per-layer offsets
          for (const auto& [k_offsets, v_offsets] : layer_offsets) {
            auto* layer_proto = resp->add_xtensor_layer_offsets();
            for (const auto& k_off : k_offsets) {
              layer_proto->add_k_offsets(k_off);
            }
            for (const auto& v_off : v_offsets) {
              layer_proto->add_v_offsets(v_off);
            }
          }
          VLOG(5) << "XTensor offsets added for request " << req.req_id()
                  << ", num_blocks=" << block_ids.size()
                  << ", num_layers=" << layer_offsets.size();
        } else {
          LOG(WARNING) << "Failed to get XTensor offsets for request "
                       << req.req_id();
        }
      }

      resp->set_status_code(200);
    }
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
    std::vector<uint64_t> block_ids(gen.block_ids().begin(),
                                    gen.block_ids().end());
    int32_t linear_state_id = gen.linear_state_id();
    torch::Tensor mtp_bootstrap_embedding;
    if (gen.has_mtp_bootstrap_embedding()) {
      mtp_bootstrap_embedding =
          util::proto_to_torch(gen.mtp_bootstrap_embedding());
    }

    bool success = scheduler_->decode_recv_first_generation(
        gen.req_id(),
        first_token.token_id(),
        first_token.has_logprob(),
        first_token.logprob(),
        first_token.time_to_first_token_latency_seconds(),
        std::move(top_tokens),
        std::move(top_logprobs),
        gen.kv_cache_transfer_mode(),
        std::move(cluster_ids),
        std::move(addrs),
        std::move(block_ids),
        linear_state_id,
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
