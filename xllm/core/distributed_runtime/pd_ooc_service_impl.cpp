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

#include "pd_ooc_service_impl.h"

#include <glog/logging.h>

#include "common/types.h"
#include "distributed_runtime/llm_engine.h"
#include "framework/request/request_output.h"
#include "scheduler/pd_ooc_scheduler.h"

namespace xllm {

PDOOCServiceImpl::PDOOCServiceImpl(PDOOCScheduler* scheduler, Engine* engine)
    : DisaggPDServiceImpl(scheduler, engine), pd_ooc_scheduler_(scheduler) {}

void PDOOCServiceImpl::decode_recv_multi_generations(
    const proto::DisaggGenerationsRequests* request,
    proto::Status* response) {
  bool overall_success = true;

  for (auto& multi_gen : request->multi_gens()) {
    // Convert proto repeated field to vector
    std::vector<proto::RemoteToken> migration_tokens(multi_gen.tokens().begin(),
                                                     multi_gen.tokens().end());

    std::vector<uint64_t> cluster_ids(multi_gen.cluster_ids().begin(),
                                      multi_gen.cluster_ids().end());
    std::vector<std::string> addrs(multi_gen.addrs().begin(),
                                   multi_gen.addrs().end());
    std::vector<uint64_t> block_ids(multi_gen.block_ids().begin(),
                                    multi_gen.block_ids().end());
    int32_t linear_state_id = multi_gen.linear_state_id();

    bool success = pd_ooc_scheduler_->decode_recv_multi_generations(
        multi_gen.req_id(),
        migration_tokens,
        multi_gen.kv_cache_transfer_mode(),
        std::move(cluster_ids),
        std::move(addrs),
        std::move(block_ids),
        linear_state_id,
        multi_gen.dp_size(),
        multi_gen.dp_rank());

    if (!success) {
      overall_success = false;
      break;
    }
  }

  response->set_ok(overall_success);
}

void PDOOCServiceImpl::prefill_recv_pull_signal(
    const proto::PullSignal* request,
    proto::Status* response) {
  // Put the pull signal into a queue and response
  bool result =
      pd_ooc_scheduler_->write_pull_signal(proto::PullSignal(*request));

  if (response) {
    response->set_ok(result);
  }
}

}  // namespace xllm
