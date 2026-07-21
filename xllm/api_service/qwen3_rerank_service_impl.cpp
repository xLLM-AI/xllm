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

#include "api_service/qwen3_rerank_service_impl.h"

#include "distributed_runtime/llm_master.h"
#include "framework/request/request_params.h"

namespace xllm {

Qwen3RerankServiceImpl::Qwen3RerankServiceImpl(
    LLMMaster* master,
    const std::vector<std::string>& models)
    : RerankServiceImpl(master, models) {}

void Qwen3RerankServiceImpl::process_async_impl(
    std::shared_ptr<RerankCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  auto query = rpc_request.query();
  std::vector<std::string> documents;
  if (rpc_request.documents_size() > 0) {
    documents = std::vector<std::string>(rpc_request.documents().begin(),
                                         rpc_request.documents().end());
  }

  std::vector<std::string> reqs;
  reqs.reserve(documents.size());
  for (size_t i = 0; i < documents.size(); ++i) {
    reqs.emplace_back(query + documents[i]);
  }

  RequestParams request_params(
      rpc_request, call->x_request_id(), call->x_request_time());
  std::vector<RequestParams> sps(documents.size(), request_params);
  auto request_id = request_params.request_id;

  int32_t top_n = static_cast<int32_t>(documents.size());
  if (rpc_request.has_top_n()) {
    top_n = std::min(top_n, rpc_request.top_n());
  }

  // Logprobs-based score computer for Qwen3 rerank
  auto compute_scores = [](const std::vector<std::string>& documents,
                           const std::vector<RequestOutput>& req_outputs)
      -> std::vector<RerankRequestOutput> {
    std::vector<RerankRequestOutput> rerank_outputs;
    rerank_outputs.reserve(documents.size());

    for (size_t i = 0; i < documents.size(); ++i) {
      if (req_outputs[i].outputs[0].logprobs.has_value()) {
        auto score = req_outputs[i].outputs[0].logprobs.value()[0].logprob;
        rerank_outputs.emplace_back(i, documents[i], score);
      }
    }
    return rerank_outputs;
  };

  auto ctx = std::make_shared<RerankContext>(call,
                                             std::move(documents),
                                             model,
                                             request_id,
                                             top_n,
                                             sps.size(),
                                             compute_scores);

  auto batch_callback = [ctx](size_t index, RequestOutput output) -> bool {
    ctx->on_complete(index, std::move(output));
    return true;
  };

  master_->handle_batch_request(reqs, sps, batch_callback);
}

}  // namespace xllm
