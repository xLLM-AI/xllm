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

#include "common/metrics.h"

// llm server impl metrics
DEFINE_COUNTER(request_status_total_ok, "Total number of request status OK");
DEFINE_COUNTER(request_status_total_cancelled,
               "Total number of request status CANCELLED");
DEFINE_COUNTER(request_status_total_unknown,
               "Total number of request status UNKNOWN");
DEFINE_COUNTER(request_status_total_invalid_argument,
               "Total number of request status INVALID_ARGUMENT");
DEFINE_COUNTER(request_status_total_deadline_exceeded,
               "Total number of request status DEADLINE_EXCEEDED");
DEFINE_COUNTER(request_status_total_resource_exhausted,
               "Total number of request status RESOURCE_EXHAUSTED");
DEFINE_COUNTER(request_status_total_unauthenticated,
               "Total number of request status UNAUTHENTICATED");
DEFINE_COUNTER(request_status_total_unavailable,
               "Total number of request status UNAVAILABLE");
DEFINE_COUNTER(request_status_total_unimplemented,
               "Total number of request status UNIMPLEMENTED");

DEFINE_COUNTER(request_handling_latency_seconds_chat,
               "Latency of chat request handling in seconds");
DEFINE_COUNTER(request_handling_latency_seconds_completion,
               "Latency of completion request handling in seconds");

DEFINE_COUNTER(tokenization_latency_seconds,
               "Prompt tokenization latency in seconds");
DEFINE_COUNTER(chat_template_latency_seconds,
               "Chat template latency in seconds");

// block manager metrics
DEFINE_COUNTER(prefix_cache_latency_seconds_insert,
               "Latency of prefix cache insert in seconds");
DEFINE_COUNTER(prefix_cache_latency_seconds_match,
               "Latency of prefix cache match in seconds");
DEFINE_COUNTER(prefix_cache_latency_seconds_evict,
               "Latency of prefix cache evict in seconds");

DEFINE_COUNTER(prefix_cache_match_length_total,
               "Length of matched prefix in tokens");

DEFINE_COUNTER(allocate_blocks_latency_seconds,
               "Latency of blocks allocation in seconds");

DEFINE_HISTOGRAM(prefix_cache_block_matched_rate,
                 "Histogram of prefix cache block match rate");

DEFINE_HISTOGRAM(prefix_cache_block_matched_num,
                 "Histogram of prefix cache block matched number");

// sequence metrics
DEFINE_COUNTER(detokenization_latency_seconds_stream,
               "Latency of stream detokenization in seconds");
DEFINE_COUNTER(detokenization_latency_seconds_non_stream,
               "Latency of non-stream detokenization in seconds");

// executor metrics
DEFINE_COUNTER(num_model_execution_total_eager,
               "Total number of model execution");

// worker metrics
DEFINE_COUNTER(execution_latency_seconds_model,
               "Latency of model execution in seconds");
DEFINE_COUNTER(execution_latency_seconds_logits_processing,
               "Latency of logits processing in seconds");
DEFINE_COUNTER(execution_latency_seconds_sampling,
               "Latency of sampling in seconds");

// scheduler metrics
DEFINE_GAUGE(num_pending_requests, "Number of pending requests in scheduler");
DEFINE_GAUGE(num_running_requests, "Number of running requests in scheduler");
DEFINE_GAUGE(num_waiting_requests, "Number of waiting requests in scheduler");
DEFINE_GAUGE(num_preempted_requests,
             "Number of preempted requests in scheduler");
DEFINE_GAUGE(num_offline_decode_preempt_offline_requests,
             "Number of offline decode preempt offline requests in scheduler");
DEFINE_GAUGE(num_online_decode_preempt_online_requests,
             "Number of online decode preempt online requests in scheduler");
DEFINE_GAUGE(num_online_prefill_preempt_offline_requests,
             "Number of online prefill preempt offline requests in scheduler");
DEFINE_GAUGE(num_online_decode_preempt_offline_requests,
             "Number of online decode preempt offline requests in scheduler");

DEFINE_GAUGE(num_running_sequences, "Number of running sequences");

DEFINE_GAUGE(kv_cache_utilization_perc,
             "Utilization of the kv cache in percentage");
DEFINE_GAUGE(num_blocks_in_prefix_cache,
             "Number of blocks in the prefix cache");
DEFINE_GAUGE(num_free_blocks, "Number of free blocks in the block allocator");
DEFINE_GAUGE(num_used_blocks, "Effective number of blocks in use");

DEFINE_COUNTER(scheduling_latency_seconds, "Latency of scheduling in seconds");

DEFINE_COUNTER(num_processing_tokens_total_prompt,
               "Total number of processing prompt tokens");
DEFINE_COUNTER(num_processing_tokens_total_generated,
               "Total number of processing generated tokens");

DEFINE_HISTOGRAM(num_prompt_tokens_per_request,
                 "Histogram of the prompt token number per request");

DEFINE_HISTOGRAM(num_generated_tokens_per_request,
                 "Histogram of the generated token number per request");

// ttft latency histogram
DEFINE_HISTOGRAM(time_to_first_token_latency_milliseconds,
                 "Histogram of time to first token latency in milliseconds");
// inter token latency histogram
DEFINE_HISTOGRAM(inter_token_latency_milliseconds,
                 "Histogram of inter token latency in milliseconds");
// speculative/MTP amortized inter token latency histogram. Each observation is
// the per-step wall-clock latency divided by the tokens committed that step
// (per-step amortized, not token-weighted).
DEFINE_HISTOGRAM(
    speculative_per_token_latency_milliseconds,
    "Histogram of speculative per-token amortized inter token latency in "
    "milliseconds (per-step amortized, not token-weighted)");

// response metrics
DEFINE_COUNTER(responsing_latency_seconds_stream,
               "Latency of stream responding in seconds");
DEFINE_COUNTER(responsing_latency_seconds_non_stream,
               "Latency of non-stream responding in seconds");

DEFINE_HISTOGRAM(end_2_end_latency_milliseconds,
                 "Histogram of end to end latency in milliseconds");

// xllm service metrics
DEFINE_COUNTER(server_request_in_total,
               "Total number of request that server received");

DEFINE_COUNTER(server_request_total_ok,
               "Total number of ok request that server processed");
DEFINE_COUNTER(server_request_total_limit,
               "Total number of limit request that server processed");
DEFINE_COUNTER(server_request_total_fail,
               "Total number of fail request that server processed");

DEFINE_GAUGE(num_concurrent_requests,
             "Number of concurrent requests in server");

DEFINE_GAUGE(xllm_cpu_num, "The cpu number pre instance for xllm");
DEFINE_GAUGE(xllm_cpu_utilization, "The cpu utilization pre instance for xllm");
DEFINE_GAUGE(xllm_gpu_num, "The gpu number pre instance for xllm");
DEFINE_GAUGE(xllm_gpu_utilization, "The gpu utilization pre instance for xllm");

// speculative metrics
DEFINE_COUNTER(speculative_execution_latency_seconds_draft,
               "Latency of draft execution in seconds");
DEFINE_COUNTER(speculative_execution_latency_seconds_target,
               "Latency of target execution in seconds");
DEFINE_COUNTER(speculative_execution_latency_seconds_validation,
               "Latency of validation in seconds");

DEFINE_COUNTER(speculative_num_accepted_tokens_total,
               "Total number of accepted tokens in validation");
DEFINE_COUNTER(speculative_num_draft_tokens_total,
               "Total number of draft tokens");
DEFINE_GAUGE(speculative_mean_tokens_per_decode_step,
             "Batch-mean tokens committed per decode step, i.e. the TPOT "
             "speedup factor (1.0 without speculative decoding)");

// proto metrics
DEFINE_COUNTER(proto_latency_seconds_proto2i,
               "Latency of proto2i convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_i2proto,
               "Latency of i2proto convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_proto2o,
               "Latency of proto2o convert in seconds");
DEFINE_COUNTER(proto_latency_seconds_o2proto,
               "Latency of o2proto convert in seconds");

// engine metrics
DEFINE_COUNTER(prepare_input_latency_seconds,
               "Latency of preparing input in seconds");

// rec engine metrics
DEFINE_HISTOGRAM(rec_total_latency_microseconds,
                 "Latency of rec engine in microseconds");
DEFINE_HISTOGRAM(prepare_input_latency_microseconds,
                 "Latency of preparing input in microseconds");
DEFINE_HISTOGRAM(rec_first_token_latency_microseconds,
                 "Latency of rec first token generation in microseconds");
DEFINE_HISTOGRAM(rec_second_token_latency_microseconds,
                 "Latency of rec second token generation in microseconds");
DEFINE_HISTOGRAM(rec_third_token_latency_microseconds,
                 "Latency of rec third token generation in microseconds");
DEFINE_HISTOGRAM(rec_sampling_latency_microseconds,
                 "Latency of rec sampling in microseconds");

DEFINE_GAUGE(rec_beam_search_width, "rec beam search width");

DEFINE_HISTOGRAM(rec_input_token_num_per_request,
                 "Histogram of input token number per request in rec engine");

// multi node metrics
DEFINE_COUNTER(worker_service_latency_seconds,
               "Worker service execution latency in seconds");
DEFINE_COUNTER(engine_latency_seconds, "Engine execution latency in seconds");

// memory metrics
DEFINE_GAUGE(total_memory_size_in_kilobytes, "Total memory size in kilobytes");
DEFINE_GAUGE(weight_size_in_kilobytes, "Weight size in kilobytes");
DEFINE_GAUGE(total_kv_cache_size_in_kilobytes, "KV cache size in kilobytes");
DEFINE_GAUGE(total_activation_size_in_kilobytes,
             "Total activation size in kilobytes");

DEFINE_MULTI_HISTOGRAM(active_kv_cache_size_in_kilobytes,
                       "dp_rank",
                       "Active kv cache size in kilobytes per dp rank");
DEFINE_MULTI_HISTOGRAM(
    prefill_active_activation_size_in_kilobytes,
    "dp_rank",
    "Active activation size in kilobytes per dp rank during prefill phase");
DEFINE_MULTI_HISTOGRAM(
    decode_active_activation_size_in_kilobytes,
    "dp_rank",
    "Active activation size in kilobytes per dp rank during decode phase");
