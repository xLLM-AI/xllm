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

#include "sample_service_impl.h"

#include <absl/time/clock.h>
#include <absl/time/time.h>
#include <glog/logging.h>

#include <algorithm>
#include <condition_variable>
#include <mutex>

#include "common/instance_name.h"
#include "core/distributed_runtime/llm_master.h"
#include "core/framework/request/request_output.h"
#include "core/framework/request/request_params.h"
#include "core/framework/request/sample_slot.h"
#include "core/util/uuid.h"

namespace xllm {
namespace {
thread_local ShortUUID short_uuid;
const std::string kSelectorMatchFinishReason = "selector_match";
const std::string kEmptyLogprobsFinishReason = "empty_logprobs";

std::string generate_sample_request_id() {
  return "sample-" + InstanceName::name()->get_name_hash() + "-" +
         short_uuid.random();
}

void initialize_response(const std::string& request_id,
                         const std::string& model,
                         uint32_t created_time,
                         proto::SampleResponse* response) {
  CHECK(response != nullptr);
  response->Clear();
  response->set_id(request_id);
  response->set_object("sample_completion");
  response->set_created(created_time);
  response->set_model(model);
}

void set_choice_logprobs(proto::Choice* choice,
                         const std::optional<std::vector<LogProb>>& logprobs) {
  if (choice == nullptr || !logprobs.has_value() || logprobs->empty()) {
    return;
  }

  const auto& sampled_logprob = logprobs->front();
  auto* proto_logprobs = choice->mutable_logprobs();
  if (sampled_logprob.top_logprobs.has_value() &&
      !sampled_logprob.top_logprobs->empty()) {
    for (const auto& top_logprob : sampled_logprob.top_logprobs.value()) {
      proto_logprobs->add_tokens(top_logprob.token);
      proto_logprobs->add_token_ids(top_logprob.token_id);
      proto_logprobs->add_token_logprobs(top_logprob.logprob);
    }
    return;
  }

  proto_logprobs->add_tokens(sampled_logprob.token);
  proto_logprobs->add_token_ids(sampled_logprob.token_id);
  proto_logprobs->add_token_logprobs(sampled_logprob.logprob);
}

bool has_choice_logprobs(const SequenceOutput& output) {
  return output.logprobs.has_value() && !output.logprobs->empty();
}

std::string get_choice_text(const SequenceOutput& output) {
  if (!has_choice_logprobs(output)) {
    return output.text;
  }

  const auto& sampled_logprob = output.logprobs->front();
  if (sampled_logprob.top_logprobs.has_value() &&
      !sampled_logprob.top_logprobs->empty()) {
    return sampled_logprob.top_logprobs->front().token;
  }
  return sampled_logprob.token;
}

std::string get_finish_reason(const SequenceOutput& output) {
  if (output.finish_reason.has_value()) {
    return output.finish_reason.value();
  }
  return has_choice_logprobs(output) ? kSelectorMatchFinishReason
                                     : kEmptyLogprobsFinishReason;
}

uint32_t get_requested_logprobs(const proto::SampleRequest& request) {
  return request.has_logprobs()
             ? request.logprobs()
             : sample_service_internal::kDefaultSampleLogprobs;
}

Status get_rate_limit_status(LLMMaster* master) {
  CHECK(master != nullptr);
  if (!master->get_rate_limiter()->is_limited()) {
    return Status();
  }

  if (master->get_rate_limiter()->is_sleeping()) {
    return Status(StatusCode::UNAVAILABLE,
                  "Model is currently in sleep state.");
  }
  return Status(StatusCode::RESOURCE_EXHAUSTED,
                "The number of concurrent requests has reached the limit.");
}

}  // namespace

namespace sample_service_internal {

Status validate_request(const proto::SampleRequest& request) {
  if (request.model().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "model is required");
  }
  if (request.prompt().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "prompt is required");
  }
  if (!request.has_selector()) {
    return Status(StatusCode::INVALID_ARGUMENT, "selector is required");
  }
  if (request.selector().type() != "literal") {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "selector.type must be literal");
  }
  if (request.selector().value().empty()) {
    return Status(StatusCode::INVALID_ARGUMENT, "selector.value is required");
  }
  if (request.has_logprobs() && (request.logprobs() < kMinSampleLogprobs ||
                                 request.logprobs() > kMaxSampleLogprobs)) {
    return Status(StatusCode::INVALID_ARGUMENT,
                  "logprobs must be between 1 and 5");
  }
  return Status();
}

Status validate_runtime_config(bool enable_schedule_overlap) {
  if (enable_schedule_overlap) {
    return Status(StatusCode::UNAVAILABLE,
                  "/v1/sample does not support async scheduling "
                  "(enable_schedule_overlap=true)");
  }
  return Status();
}

bool build_request_params(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          RequestParams* request_params) {
  if (request_params == nullptr) {
    return false;
  }

  RequestParams params;
  params.request_id = request.has_request_id() ? request.request_id()
                                               : generate_sample_request_id();
  params.logprobs = true;
  params.top_logprobs =
      request.has_logprobs() ? request.logprobs() : kDefaultSampleLogprobs;
  params.max_tokens = 1;
  params.n = 1;
  params.best_of = 1;
  params.add_special_tokens = true;
  params.is_sample_request = true;

  if (!build_sample_slots(params.request_id,
                          request.prompt(),
                          request.selector().value(),
                          tokenizer,
                          &params.sample_slots)) {
    return false;
  }

  *request_params = std::move(params);
  return true;
}

bool build_empty_response(const proto::SampleRequest& request,
                          const Tokenizer& tokenizer,
                          const std::string& request_id,
                          proto::SampleResponse* response) {
  if (response == nullptr) {
    return false;
  }

  std::vector<int32_t> prompt_tokens;
  if (!tokenizer.encode(request.prompt(), &prompt_tokens, true)) {
    return false;
  }

  initialize_response(request_id,
                      request.model(),
                      static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now())),
                      response);
  response->mutable_choices();
  auto* usage = response->mutable_usage();
  const int32_t prompt_tokens_count =
      static_cast<int32_t>(prompt_tokens.size());
  usage->set_prompt_tokens(prompt_tokens_count);
  usage->set_completion_tokens(0);
  usage->set_total_tokens(prompt_tokens_count);
  return true;
}

bool build_response(const std::string& request_id,
                    const std::string& model,
                    uint32_t created_time,
                    const RequestOutput& req_output,
                    proto::SampleResponse* response) {
  if (response == nullptr) {
    return false;
  }

  initialize_response(request_id, model, created_time, response);

  std::vector<const SequenceOutput*> sorted_outputs;
  sorted_outputs.reserve(req_output.outputs.size());
  for (const auto& output : req_output.outputs) {
    sorted_outputs.push_back(&output);
  }
  std::stable_sort(
      sorted_outputs.begin(),
      sorted_outputs.end(),
      [](const auto* lhs, const auto* rhs) { return lhs->index < rhs->index; });

  response->mutable_choices()->Reserve(sorted_outputs.size());
  for (const auto* output : sorted_outputs) {
    auto* choice = response->add_choices();
    choice->set_index(output->index);
    choice->set_text(get_choice_text(*output));
    set_choice_logprobs(choice, output->logprobs);
    if (!has_choice_logprobs(*output)) {
      choice->mutable_logprobs();
    }
    choice->set_finish_reason(get_finish_reason(*output));
  }

  if (req_output.usage.has_value()) {
    const auto& usage = req_output.usage.value();
    auto* proto_usage = response->mutable_usage();
    proto_usage->set_prompt_tokens(usage.num_prompt_tokens);
    proto_usage->set_completion_tokens(usage.num_generated_tokens);
    proto_usage->set_total_tokens(usage.num_total_tokens);
  }

  return true;
}

}  // namespace sample_service_internal

SampleServiceImpl::SampleServiceImpl(LLMMaster* master,
                                     const std::vector<std::string>& models)
    : APIServiceImpl(models), master_(master) {
  CHECK(master_ != nullptr);
}

bool SampleServiceImpl::process_request(const proto::SampleRequest& request,
                                        proto::SampleResponse* response,
                                        Status* status) const {
  CHECK(response != nullptr);
  CHECK(status != nullptr);
  response->Clear();

  *status = sample_service_internal::validate_request(request);
  if (!status->ok()) {
    return false;
  }

  if (!models_.contains(request.model())) {
    *status = Status(StatusCode::UNKNOWN, "Model not supported");
    return false;
  }

  *status = sample_service_internal::validate_runtime_config(
      master_->options().enable_schedule_overlap());
  if (!status->ok()) {
    return false;
  }

  RequestParams request_params;
  if (!sample_service_internal::build_request_params(
          request, master_->tokenizer(), &request_params)) {
    *status = Status(StatusCode::UNKNOWN,
                     "Failed to build sample selector runtime mapping");
    return false;
  }

  if (request_params.sample_slots.empty()) {
    if (!sample_service_internal::build_empty_response(
            request,
            master_->tokenizer(),
            request_params.request_id,
            response)) {
      *status = Status(StatusCode::UNKNOWN,
                       "Failed to build sample no-match response");
      return false;
    }
    *status = Status();
    return true;
  }

  *status = get_rate_limit_status(master_);
  if (!status->ok()) {
    return false;
  }

  RequestOutput final_output;
  bool has_final_output = false;
  std::mutex mu;
  std::condition_variable cv;
  const auto request_id = request_params.request_id;
  const size_t match_count = request_params.sample_slots.size();
  const auto created_time =
      static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now()));

  master_->handle_request(
      request.prompt(),
      std::nullopt,
      std::move(request_params),
      std::nullopt,
      [this, &mu, &cv, &final_output, &has_final_output](
          const RequestOutput& req_output) -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value() && !req_output.status->ok()) {
          master_->get_rate_limiter()->decrease_one_request();
        } else if (req_output.finished || req_output.cancelled ||
                   req_output.finished_on_prefill_instance) {
          master_->get_rate_limiter()->decrease_one_request();
        } else {
          return true;
        }

        {
          std::lock_guard<std::mutex> lock(mu);
          if (!has_final_output) {
            final_output = req_output;
            has_final_output = true;
          }
        }
        cv.notify_one();
        return true;
      });

  {
    std::unique_lock<std::mutex> lock(mu);
    cv.wait(lock, [&has_final_output]() { return has_final_output; });
  }

  if (final_output.status.has_value() && !final_output.status->ok()) {
    *status = final_output.status.value();
    return false;
  }

  if (!sample_service_internal::build_response(
          request_id, request.model(), created_time, final_output, response)) {
    *status = Status(StatusCode::UNKNOWN, "Failed to build sample response");
    return false;
  }

  *status = Status();
  return true;
}

void SampleServiceImpl::process_async_impl(std::shared_ptr<SampleCall> call) {
  const auto& request = call->request();
  Status status = sample_service_internal::validate_request(request);
  if (!status.ok()) {
    call->finish_with_error(status.code(), status.message());
    return;
  }

  if (!models_.contains(request.model())) {
    call->finish_with_error(StatusCode::UNKNOWN, "Model not supported");
    return;
  }

  status = sample_service_internal::validate_runtime_config(
      master_->options().enable_schedule_overlap());
  if (!status.ok()) {
    call->finish_with_error(status.code(), status.message());
    return;
  }

  RequestParams request_params;
  if (!sample_service_internal::build_request_params(
          request, master_->tokenizer(), &request_params)) {
    call->finish_with_error(StatusCode::UNKNOWN,
                            "Failed to build sample selector runtime mapping");
    return;
  }
  request_params.set_x_request_id_if_absent(call->x_request_id());

  if (request_params.sample_slots.empty()) {
    if (!sample_service_internal::build_empty_response(
            request,
            master_->tokenizer(),
            request_params.request_id,
            &call->response())) {
      call->finish_with_error(StatusCode::UNKNOWN,
                              "Failed to build sample no-match response");
      return;
    }
    call->write_and_finish(call->response());
    return;
  }

  status = get_rate_limit_status(master_);
  if (!status.ok()) {
    call->finish_with_error(status.code(), status.message());
    return;
  }

  const auto request_id = request_params.request_id;
  const size_t match_count = request_params.sample_slots.size();
  const auto created_time =
      static_cast<uint32_t>(absl::ToUnixSeconds(absl::Now()));

  master_->handle_request(
      request.prompt(),
      std::nullopt,
      std::move(request_params),
      call.get(),
      [call,
       master = master_,
       model = request.model(),
       request_id,
       match_count,
       created_time](const RequestOutput& req_output) -> bool {
        req_output.log_request_status();
        if (req_output.status.has_value()) {
          const auto& output_status = req_output.status.value();
          if (!output_status.ok()) {
            master->get_rate_limiter()->decrease_one_request();
            return call->finish_with_error(output_status.code(),
                                           output_status.message());
          }
        }

        if (req_output.finished || req_output.cancelled ||
            req_output.finished_on_prefill_instance) {
          master->get_rate_limiter()->decrease_one_request();
        }

        if (!sample_service_internal::build_response(request_id,
                                                     model,
                                                     created_time,
                                                     req_output,
                                                     &call->response())) {
          return call->finish_with_error(StatusCode::UNKNOWN,
                                         "Failed to build sample response");
        }

        return call->write_and_finish(call->response());
      });
}

}  // namespace xllm
