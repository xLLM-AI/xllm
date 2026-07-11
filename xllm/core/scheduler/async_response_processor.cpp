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

#include "async_response_processor.h"

#include <absl/synchronization/notification.h>
#include <absl/time/clock.h>
#include <glog/logging.h>

#include <memory>

#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/service_config.h"
#include "framework/request/finish_reason.h"
#include "framework/request/request.h"
#include "framework/request/sequence.h"
#include "util/blocking_counter.h"
#include "util/env_var.h"

namespace xllm {

AsyncResponseProcessor::AsyncResponseProcessor(
    const Tokenizer* tokenizer,
    const std::optional<InstanceRole>& role,
    bool enable_service_routing,
    bool disable_log_stats,
    std::function<void(std::shared_ptr<Request>)> cancel_request)
    : response_threadpool_(
          /*num_threads=*/::xllm::ServiceConfig::get_instance()
              .num_response_handling_threads(),
          /*cpu_binding=*/false,
          /*pool_name=*/"AsyncResponseProcessor.response"),
      rpc_threadpool_(/*num_threads=*/1,
                      /*cpu_binding=*/false,
                      /*pool_name=*/"AsyncResponseProcessor.rpc"),
      generate_output_threadpool_(
          /*num_threads=*/16,
          /*cpu_binding=*/false,
          /*pool_name=*/"AsyncResponseProcessor.generate_output"),
      tokenizer_(tokenizer->clone()),
      role_(role.value_or(InstanceRole::DEFAULT)),
      enable_batch_response_(enable_service_routing),
      disable_log_stats_(disable_log_stats),
      cancel_request_(std::move(cancel_request)) {}

AsyncResponseProcessor::~AsyncResponseProcessor() { wait_completion(); }

void AsyncResponseProcessor::process_failed_request(
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
    output.status = status;
    request->state().output_func(output);
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::process_completed_request(
    std::shared_ptr<Request> request) {
  // schedule the response handling

  // NOTE: Maybe refactor later.
  // For prefill instance in 'ENABLE_SERVICE_DISPATCH_REQUEST' scenario,
  // Currently, the xLLM service uses the BRPC HTTP interface to forward
  // requests. When the prefill process returns the first token to the xLLM
  // service, the result is only returned after the OutputFunc callback function
  // is destructed. Therefore, to ensure the TTFT, the Output callback function
  // needs to be recycled once its execution is complete, allowing the first
  // token to be returned immediately. Additionally, the processing here runs in
  // parallel with the subsequent step where the prefill sends the first token
  // to the decode instance.
  //  When this completes, it destructs the request, which in turn causes the
  //  Output object
  // to be destructed. To prevent the scenario where the request is recycled
  // before the response is fully returned to the xLLM service, the Output
  // object must be detached to avoid premature destruction.
  auto runnable = [this, request = request]() mutable {
    AUTO_COUNTER(responsing_latency_seconds_non_stream);

    double end_2_end_latency_seconds = request->elapsed_seconds();
    // update the metrics for the request
    HISTOGRAM_OBSERVE(end_2_end_latency_milliseconds,
                      static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
    RequestOutput req_output =
        request->generate_output(*tokenizer_, &generate_output_threadpool_);
    if (!disable_log_stats_) {
      request->log_statistic(end_2_end_latency_seconds);
    }
    request->state().output_func(req_output);
  };
  if (request->state().response_thread_id < 0) {
    request->state().response_thread_id =
        response_threadpool_.schedule(runnable);
  } else {
    response_threadpool_.schedule_with_tid(runnable,
                                           request->state().response_thread_id);
  }
}

void AsyncResponseProcessor::batch_process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (int i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    auto runnable = [counter,
                     this,
                     request = request,
                     request_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_non_stream);
      double end_2_end_latency_seconds = request->elapsed_seconds();
      // update the metrics for the request
      HISTOGRAM_OBSERVE(
          end_2_end_latency_milliseconds,
          static_cast<int64_t>(end_2_end_latency_seconds * 1000.0));
      if (request->finished() || request->cancelled()) {
        if (!disable_log_stats_) {
          request->log_statistic(end_2_end_latency_seconds);
        }
      }

      *request_output = std::move(request->generate_output(*tokenizer_));
      if (request->sequences()[0]->num_generated_tokens() == 1) {
        // currently only support one sequence when enable_service_routing
        request_output->finished_on_prefill_instance = true;
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule(
      [counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs)]() mutable {
        counter->wait();
        auto& resp_callback = requests[0]->state().outputs_func;
        resp_callback(request_outputs);
      });
}

// process non-stream requests
void AsyncResponseProcessor::process_completed_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (!enable_batch_response_) {
    for (size_t i = 0; i < requests.size(); ++i) {
      process_completed_request(std::move(requests[i]));
    }
  } else {
    // send request response to xllm service in batch through rpc
    batch_process_completed_requests(requests);
  }
}

void AsyncResponseProcessor::process_stream_request(
    std::shared_ptr<Request> request) {
  CHECK(request->state().stream) << "request is not a streaming request";

  std::vector<size_t> indexes;
  std::vector<size_t> num_tokens;
  bool is_all_seqs_closed = true;
  for (size_t i = 0; i < request->sequences().size(); ++i) {
    auto& seq = request->sequences()[i];
    is_all_seqs_closed &= seq->is_closed();
    if (seq->is_closed()) {
      // skip already closed sequences
      continue;
    }

    // check if the sequence has enough tokens to output
    if (seq->has_new_tokens_generated() || seq->finished()) {
      indexes.push_back(i);
      num_tokens.push_back(seq->num_tokens());
    }

    // close the sequence after sending finish reason
    if (seq->finished()) {
      seq->close();
    }
  }

  if (!is_all_seqs_closed) {
    // output the delta text til the end of the sequence to the client

    auto runnable = [cancel_request = cancel_request_,
                     request,
                     this,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens)]() {
      AUTO_COUNTER(responsing_latency_seconds_stream);

      RequestOutput req_output;
      req_output.request_id = request->request_id();
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
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }
}

void AsyncResponseProcessor::batch_process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  size_t requests_size = requests.size();
  auto counter = new BlockingCounter(requests_size);
  std::vector<RequestOutput> request_outputs;
  request_outputs.resize(requests_size);
  for (int i = 0; i < requests_size; ++i) {
    auto& request = requests[i];
    CHECK(request->state().stream) << "request is not a streaming request";

    std::vector<size_t> indexes;
    std::vector<size_t> num_tokens;
    for (size_t i = 0; i < request->sequences().size(); ++i) {
      auto& seq = request->sequences()[i];
      if (seq->is_closed()) {
        // skip already closed sequences
        continue;
      }

      // check if the sequence has enough tokens to output
      if (seq->has_new_tokens_generated() || seq->finished()) {
        indexes.push_back(i);
        num_tokens.push_back(seq->num_tokens());
      }

      // close the sequence after sending finish reason
      if (seq->finished()) {
        seq->close();
      }
    }

    // output the delta text til the end of the sequence to the client
    auto runnable = [this,
                     counter,
                     request,
                     indexes = std::move(indexes),
                     num_tokens = std::move(num_tokens),
                     req_output = &request_outputs[i]]() mutable {
      AUTO_COUNTER(responsing_latency_seconds_stream);

      // RequestOutput req_output;
      req_output->request_id = request->request_id();
      req_output->service_request_id = request->service_request_id();
      req_output->target_xservice_addr = request->source_xservice_addr();
      for (size_t i = 0; i < indexes.size(); ++i) {
        const size_t index = indexes[i];
        const size_t size = num_tokens[i];
        auto& seq = request->sequences()[index];

        auto seq_output = seq->generate_streaming_output(size, *tokenizer_);
        if (seq_output.has_value()) {
          req_output->outputs.push_back(std::move(seq_output.value()));
          if (seq->num_generated_tokens() == 1) {
            // currently only support one sequence when enable_service_routing
            // IMPROVE LATER: support enable_schedule_overlap in Default mode
            // for stream request
            req_output->finished_on_prefill_instance = true;
          }
        }
      }
      counter->decrement_count();
    };
    if (request->state().response_thread_id < 0) {
      request->state().response_thread_id =
          response_threadpool_.schedule(runnable);
    } else {
      response_threadpool_.schedule_with_tid(
          runnable, request->state().response_thread_id);
    }
  }

  rpc_threadpool_.schedule(
      [cancel_request = cancel_request_,
       counter = std::unique_ptr<BlockingCounter>(counter),
       requests = std::move(requests),
       request_outputs = std::move(request_outputs)]() mutable {
        auto& resp_callback = requests[0]->state().outputs_func;
        counter->wait();
        std::vector<bool> status_set = resp_callback(request_outputs);
        for (size_t i = 0; i < requests.size(); ++i) {
          if (!status_set[i]) {
            cancel_request(requests[i]);
          }
        }
      });
}

// process stream requests
void AsyncResponseProcessor::process_stream_requests(
    std::vector<std::shared_ptr<Request>>& requests) {
  if (!enable_batch_response_) {
    for (auto& req : requests) {
      process_stream_request(req);
    }
  } else {
    // send request response to xllm service in batch through rpc
    batch_process_stream_requests(requests);
  }
}

// for batch generate, wait all response done.
void AsyncResponseProcessor::wait_completion() {
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
