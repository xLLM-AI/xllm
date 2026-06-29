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

#include "rerank_service_impl.h"

#include <torch/torch.h>

#include <string>

#include "common/instance_name.h"
#include "distributed_runtime/llm_master.h"
#include "framework/request/request_params.h"
#include "util/utils.h"
#include "util/uuid.h"

namespace xllm {

void RerankContext::finalize() {
  auto rerank_outputs = compute_scores(documents, req_outputs);

  if (rerank_outputs.empty()) {
    call->finish_with_error(StatusCode::UNKNOWN, "Failed to compute scores");
    return;
  }

  std::sort(rerank_outputs.begin(),
            rerank_outputs.end(),
            [](const RerankRequestOutput& a, const RerankRequestOutput& b) {
              return a.score > b.score;
            });

  auto& response = call->response();
  response.set_id(request_id);
  response.set_model(model);

  response.mutable_results()->Reserve(top_n);
  for (int32_t i = 0;
       i < top_n && i < static_cast<int32_t>(rerank_outputs.size());
       ++i) {
    auto* result = response.add_results();
    result->set_index(rerank_outputs[i].index);
    result->mutable_document()->set_text(rerank_outputs[i].document);
    result->set_relevance_score(rerank_outputs[i].score);
  }

  int32_t num_prompt_tokens = 0;
  int32_t num_generated_tokens = 0;
  int32_t num_total_tokens = 0;
  for (const auto& req_output : req_outputs) {
    if (req_output.usage.has_value()) {
      const auto& usage = req_output.usage.value();
      num_prompt_tokens += usage.num_prompt_tokens;
      num_generated_tokens += usage.num_generated_tokens;
      num_total_tokens += usage.num_total_tokens;
    }
  }
  if (num_total_tokens > 0) {
    auto* proto_usage = response.mutable_usage();
    proto_usage->set_prompt_tokens(num_prompt_tokens);
    proto_usage->set_completion_tokens(num_generated_tokens);
    proto_usage->set_total_tokens(num_total_tokens);
  }

  call->write_and_finish(response);
}

RerankServiceImpl::RerankServiceImpl(LLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

void RerankServiceImpl::process_async_impl(std::shared_ptr<RerankCall> call) {
  const auto& rpc_request = call->request();
  const auto& model = rpc_request.model();
  if (!models_.contains(model)) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  std::vector<std::string> documents;
  if (rpc_request.documents_size() > 0) {
    documents = std::vector<std::string>(rpc_request.documents().begin(),
                                         rpc_request.documents().end());
  }
  documents.emplace_back(rpc_request.query());

  RequestParams request_params(
      rpc_request, call->x_request_id(), call->get_x_request_time());
  std::vector<RequestParams> sps(documents.size(), request_params);
  auto request_id = request_params.request_id;

  int32_t top_n = static_cast<int32_t>(documents.size() - 1);
  if (rpc_request.has_top_n()) {
    top_n = std::min(top_n, rpc_request.top_n());
  }

  // Cosine similarity score computer for embedding-based rerank
  auto compute_scores = [](const std::vector<std::string>& documents,
                           const std::vector<RequestOutput>& req_outputs)
      -> std::vector<RerankRequestOutput> {
    size_t doc_size = documents.size() - 1;
    auto& query_output = req_outputs[doc_size];
    if (!query_output.outputs[0].embeddings.has_value()) {
      return {};
    }

    auto query_embed = query_output.outputs[0].embeddings.value();
    auto query_tensor =
        torch::from_blob(query_embed.data(),
                         {static_cast<int64_t>(query_embed.size())},
                         torch::kFloat32);

    std::vector<RerankRequestOutput> rerank_outputs;
    rerank_outputs.reserve(doc_size);
    for (size_t i = 0; i < doc_size; ++i) {
      if (req_outputs[i].outputs[0].embeddings.has_value()) {
        auto doc_embed = req_outputs[i].outputs[0].embeddings.value();
        auto doc_tensor =
            torch::from_blob(doc_embed.data(),
                             {static_cast<int64_t>(doc_embed.size())},
                             torch::kFloat32);
        auto score =
            torch::cosine_similarity(query_tensor, doc_tensor, 0).item<float>();
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

  master_->handle_batch_request(ctx->documents, sps, batch_callback);
}

}  // namespace xllm
