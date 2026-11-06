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

#include "core/scheduler/request_output_dispatcher.h"

#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/metrics.h"
#include "core/framework/config/service_config.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/blocking_counter.h"

namespace xllm {

namespace {

struct StreamOutputWork {
  bool has_open_sequences = false;
  std::vector<size_t> indexes;
  std::vector<size_t> num_tokens;
};

StreamOutputWork prepare_stream_output(Request& request) {
  StreamOutputWork work;
  for (size_t i = 0; i < request.sequences().size(); ++i) {
    auto& sequence = request.sequences()[i];
    if (sequence->is_closed()) {
      continue;
    }
    work.has_open_sequences = true;

    if (sequence->has_new_tokens_generated() || sequence->finished()) {
      work.indexes.push_back(i);
      work.num_tokens.push_back(sequence->num_tokens());
    }

    if (sequence->finished()) {
      sequence->close();
    }
  }
  return work;
}

}  // namespace

RequestOutputDispatcher::RequestOutputDispatcher(
    const Tokenizer* tokenizer,
    bool enable_service_routing,
    bool disable_log_stats,
    std::function<void(std::shared_ptr<Request>)> cancel_request)
    : tokenizer_(tokenizer->clone()),
      response_threadpool_(
          /*num_threads=*/::xllm::ServiceConfig::get_instance()
              .num_response_handling_threads(),
          [response_tokenizer = tokenizer_.get()]() {
            response_tokenizer->warmup();
          },
          /*cpu_binding=*/false,
          /*pool_name=*/"RequestOutputDispatcher.response"),
      rpc_threadpool_(/*num_threads=*/1,
                      /*cpu_binding=*/false,
                      /*pool_name=*/"RequestOutputDispatcher.rpc"),
      generate_output_threadpool_(
          /*num_threads=*/16,
          [output_tokenizer = tokenizer_.get()]() {
            output_tokenizer->warmup();
          },
          /*cpu_binding=*/false,
          /*pool_name=*/"RequestOutputDispatcher.generate_output"),
      enable_batch_response_(enable_service_routing),
      disable_log_stats_(disable_log_stats),
      cancel_request_(std::move(cancel_request)) {}

RequestOutputDispatcher::~RequestOutputDispatcher() { wait_completion(); }

void RequestOutputDispatcher::schedule_response(
    const std::shared_ptr<Request>& request,
    ThreadPool::Runnable runnable) {
  const int32_t response_thread_id = request->state().response_thread_id;
  if (response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(std::move(runnable));
  } else {
    response_threadpool_.schedule_with_tid(
        std::move(runnable), static_cast<size_t>(response_thread_id));
  }
}

void RequestOutputDispatcher::process_failed_request(
    std::shared_ptr<Request> request,
    Status status) {
  // schedule the response handling
  const bool disable_log_stats = disable_log_stats_;
  auto runnable = [disable_log_stats, request = request, status = status]() {
    if (!disable_log_stats) {
      request->log_error_statistic(status);
    }
    RequestOutput output;
    output.request_id = request->request_id();
    output.service_request_id = request->service_request_id();
    output.target_xservice_addr = request->source_xservice_addr();
    output.stream = request->state().stream;
    output.response_formatter = request->state().response_formatter;
    output.status = status;
    request->state().output_func(output);
  };
  if (enable_batch_response_) {
    rpc_threadpool_.schedule(std::move(runnable));
  } else {
    schedule_response(request, std::move(runnable));
  }
}

void RequestOutputDispatcher::process_completed_request(
    std::shared_ptr<Request> request) {
  if (enable_batch_response_) {
    std::vector<std::shared_ptr<Request>> requests{std::move(request)};
    batch_process_completed_requests(requests);
    return;
  }
  // Keep terminal output behind earlier stream work on the response thread.
  // Snapshot before overlapped results can change the live request counters.
  const SpeculativeTokenStats speculative_stats =
      request->speculative_token_stats_snapshot();
  auto runnable = [this, request = request, speculative_stats]() mutable {
    AUTO_COUNTER(responsing_latency_seconds_non_stream);

    double end_2_end_latency_seconds = request->end_to_end_latency_seconds();
    // update the metrics for the request
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
    RequestOutput req_output =
        request->generate_output(*tokenizer_, &generate_output_threadpool_);
    if (!disable_log_stats_) {
      if (req_output.status.has_value() && !req_output.status->ok()) {
        request->log_error_statistic(req_output.status.value());
      } else {
        request->log_statistic(end_2_end_latency_seconds, speculative_stats);
      }
    }
    request->state().output_func(req_output);
  };
  schedule_response(request, std::move(runnable));
}

void RequestOutputDispatcher::batch_process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (size_t i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    const SpeculativeTokenStats speculative_stats =
        request->speculative_token_stats_snapshot();
    auto runnable = [counter,
                     this,
                     request = request,
                     speculative_stats,
                     request_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_non_stream);
      double end_2_end_latency_seconds = request->end_to_end_latency_seconds();
      // update the metrics for the request
      HISTOGRAM_OBSERVE(
          end_2_end_latency_milliseconds,
          static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
      *request_output = std::move(request->generate_output(*tokenizer_));
      if (!disable_log_stats_) {
        if (request_output->status.has_value() &&
            !request_output->status->ok()) {
          request->log_error_statistic(request_output->status.value());
        } else if (request->finished() || request->cancelled()) {
          request->log_statistic(end_2_end_latency_seconds, speculative_stats);
        }
      }
      counter->decrement_count();
    };
    schedule_response(request, std::move(runnable));
  }

  rpc_threadpool_.schedule([counter = std::unique_ptr<BlockingCounter>(counter),
                            requests = std::move(requests),
                            request_outputs = std::move(request_outputs),
                            cancel_request = cancel_request_]() mutable {
    counter->wait();
    auto& resp_callback = requests[0]->state().outputs_func;
    const auto accepted = resp_callback(request_outputs);
    CHECK_EQ(accepted.size(), requests.size());
    for (size_t index = 0; index < accepted.size(); ++index) {
      if (!accepted[index]) {
        cancel_request(requests[index]);
      }
    }
  });
}

// process non-stream requests
void RequestOutputDispatcher::process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  std::vector<std::shared_ptr<Request>> output_requests;
  output_requests.reserve(requests.size());
  for (const auto& request : requests) {
    // P owns computation and handoff; D owns every successful response.
    if (!request->state().decode_address.empty() &&
        !request->error_status().has_value()) {
      continue;
    }
    output_requests.emplace_back(request);
  }
  if (enable_batch_response_) {
    batch_process_completed_requests(output_requests);
  } else {
    for (auto& request : output_requests) {
      process_completed_request(std::move(request));
    }
  }
}

void RequestOutputDispatcher::process_stream_request(
    std::shared_ptr<Request> request) {
  CHECK(request->state().stream) << "request is not a streaming request";
  if (enable_batch_response_) {
    std::vector<std::shared_ptr<Request>> requests{std::move(request)};
    batch_process_stream_requests(requests);
    return;
  }

  StreamOutputWork work = prepare_stream_output(*request);
  if (work.indexes.empty() && !work.has_open_sequences) {
    return;
  }

  // output the delta text til the end of the sequence to the client

  auto runnable = [cancel_request = cancel_request_,
                   request,
                   this,
                   indexes = std::move(work.indexes),
                   num_tokens = std::move(work.num_tokens)]() {
    AUTO_COUNTER(responsing_latency_seconds_stream);

    RequestOutput req_output;
    req_output.request_id = request->request_id();
    req_output.service_request_id = request->service_request_id();
    req_output.target_xservice_addr = request->source_xservice_addr();
    req_output.stream = request->state().stream;
    req_output.response_formatter = request->state().response_formatter;
    for (size_t i = 0; i < indexes.size(); ++i) {
      const size_t index = indexes[i];
      const size_t size = num_tokens[i];
      auto& seq = request->sequences()[index];
      auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
      if (seq_output.has_value()) {
        req_output.outputs.push_back(std::move(seq_output.value()));
      }
    }
    if (!request->state().output_func(req_output)) {
      cancel_request(request);
    }
  };
  schedule_response(request, std::move(runnable));
}

void RequestOutputDispatcher::batch_process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (requests.empty()) {
    return;
  }
  std::vector<StreamOutputWork> output_work;
  output_work.reserve(requests.size());
  for (const auto& request : requests) {
    CHECK(request->state().stream) << "request is not a streaming request";
    output_work.push_back(prepare_stream_output(*request));
  }

  const size_t requests_size = requests.size();
  auto counter = new BlockingCounter(static_cast<int>(requests_size));
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (size_t i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    StreamOutputWork& work = output_work[i];
    if (work.indexes.empty()) {
      request_outputs[i].request_id = request->request_id();
      request_outputs[i].service_request_id = request->service_request_id();
      request_outputs[i].target_xservice_addr = request->source_xservice_addr();
      request_outputs[i].stream = request->state().stream;
      request_outputs[i].response_formatter =
          request->state().response_formatter;
      counter->decrement_count();
      continue;
    }

    // output the delta text til the end of the sequence to the client
    auto runnable = [this,
                     counter,
                     request,
                     indexes = std::move(work.indexes),
                     num_tokens = std::move(work.num_tokens),
                     req_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_stream);
      const absl::Time response_start_time = absl::Now();

      // RequestOutput req_output;
      req_output->request_id = request->request_id();
      req_output->service_request_id = request->service_request_id();
      req_output->target_xservice_addr = request->source_xservice_addr();
      req_output->stream = request->state().stream;
      req_output->response_formatter = request->state().response_formatter;
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];

        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output->outputs.push_back(std::move(seq_output.value()));
        }
      }
      if (req_output->finished_on_prefill_instance) {
        VLOG(1) << "Prefill response generation request_id="
                << request->request_id() << ", response_thread_id="
                << request->state().response_thread_id << ", total_ms="
                << absl::ToDoubleMilliseconds(absl::Now() -
                                              response_start_time);
      }
      counter->decrement_count();
    };
    schedule_response(request, std::move(runnable));
  }

  rpc_threadpool_.schedule([cancel_request = cancel_request_,
                            counter = std::unique_ptr<BlockingCounter>(counter),
                            requests = std::move(requests),
                            request_outputs =
                                std::move(request_outputs)]() mutable {
    auto& resp_callback = requests[0]->state().outputs_func;
    const absl::Time wait_start_time = absl::Now();
    counter->wait();
    const double wait_ms =
        absl::ToDoubleMilliseconds(absl::Now() - wait_start_time);
    const absl::Time rpc_start_time = absl::Now();
    std::vector<bool> status_set = resp_callback(request_outputs);
    CHECK_EQ(status_set.size(), requests.size());
    if (!request_outputs.empty() &&
        request_outputs[0].finished_on_prefill_instance) {
      VLOG(1) << "Prefill response RPC request_id=" << requests[0]->request_id()
              << ", response_wait_ms=" << wait_ms << ", rpc_ms="
              << absl::ToDoubleMilliseconds(absl::Now() - rpc_start_time);
    }
    for (size_t i = 0; i < requests.size(); ++i) {
      if (!status_set[i]) {
        cancel_request(requests[i]);
      }
    }
  });
}

// process stream requests
void RequestOutputDispatcher::process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  std::vector<std::shared_ptr<Request>> output_requests;
  output_requests.reserve(requests.size());
  for (const auto& request : requests) {
    // P owns computation and handoff; D owns every successful response.
    if (!request->state().decode_address.empty()) {
      continue;
    }
    output_requests.emplace_back(request);
  }
  if (enable_batch_response_) {
    batch_process_stream_requests(output_requests);
  } else {
    for (auto& request : output_requests) {
      process_stream_request(std::move(request));
    }
  }
}

// for batch generate, wait all response done.
void RequestOutputDispatcher::wait_completion() {
  auto wait_threadpool = [](ThreadPool& threadpool) {
    size_t thread_num = threadpool.size();
    // Add a task to each thread, and when all tasks are completed, it indicates
    // that all previously scheduled tasks in the thread pool have finished.
    BlockingCounter counter(thread_num);
    for (size_t i = 0; i < thread_num; ++i) {
      auto runnable = [&counter]() mutable { counter.decrement_count(); };
      threadpool.schedule_with_tid(std::move(runnable), i);
    }
    counter.wait();
  };
  wait_threadpool(response_threadpool_);
  wait_threadpool(rpc_threadpool_);
}

}  // namespace xllm
