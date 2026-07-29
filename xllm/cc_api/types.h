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

#pragma once

#include <cstdint>
#include <functional>
#include <optional>
#include <string>
#include <vector>

#include "macros.h"

namespace xllm {

struct XLLM_CAPI_EXPORT XLLM_ChatMessage {
  // The role of the messages author. One of "system", "user", "assistant".
  std::string role;

  // The content of the message. null for assistant messages with function
  // calls.
  std::string content;
};

struct XLLM_CAPI_EXPORT XLLM_InitLLMOptions {
  bool enable_chunked_prefill = false;

  // Context parallel size. Backend and model support is required.
  int32_t cp_size = 1;

  bool enable_prefix_cache = false;

  // Whether to enable disaggregated prefill and decode execution
  bool enable_disagg_pd = false;

  // Whether to enable online-offline co-location in disaggregated PD mode
  bool enable_pd_ooc = false;

  // Whether to enable schedule overlap
  bool enable_schedule_overlap = false;

  // Whether to disable TTFT profiling
  bool disable_ttft_profiling = false;

  // Whether to enable forward interruption
  bool enable_forward_interruption = false;

  // Whether to enable shared memory for executing model
  bool enable_shm = false;

  // Input shared memory size (MB)
  uint64_t input_shm_size = 1024;

  // Output shared memory size (MB)
  uint64_t output_shm_size = 128;

  bool is_local = true;

  // The KVCacheTranfer listen port
  int transfer_listen_port = 26000;

  // The number of multi-nodes
  int nnodes = 1;

  // The node rank
  int node_rank = 0;

  // Data parallel size for MLA attention
  int dp_size = 1;

  // Expert parallel size for MoE model
  int ep_size = 1;

  // Number of slots per kv cache block. Default is 128
  int block_size = 128;

  // Max gpu memory size for kv cache. Default is 0, which means cache size is
  // caculated by available memory
  int max_cache_size = 0;

  // Max number of tokens per batch
  int max_tokens_per_batch = 20480;

  // Max number of sequences per batch
  int max_seqs_per_batch = 256;

  // Max number of token per chunk in prefill stage
  int max_tokens_per_chunk_for_prefill = -1;

  // Number of speculative tokens
  int num_speculative_tokens = 0;

  // Number of threads for handling input requests
  int num_request_handling_threads = 4;

  // Expert parallel degree
  int expert_parallel_degree = 0;

  // The fraction of GPU memory to be used for model inference, including model
  // weights and kv cache
  float max_memory_utilization = 0.9;

  // The task to use the model for(e.g. generate, embed)
  std::string task = "generate";

  // NPU communication backend.(e.g. lccl, hccl). When enable dp, use hccl
  std::string communication_backend = "lccl";

  // ATB HCCL rank table file
  std::string rank_tablefile = "";

  // The role of instance(e.g. DEFAULT, PREFILL, DECODE, MIX)
  std::string instance_role = "DEFAULT";

  // The master address for multi-node distributed serving(e.g. 10.18.1.1:9999)
  std::string master_node_addr = "127.0.0.1:18899";

  std::string instance_name = "";

  // The mode of kv cache transfer(e.g. PUSH, PULL)
  std::string kv_cache_transfer_mode = "PUSH";

  std::string log_dir;

  // draft hf model path to the model file
  std::optional<std::string> draft_model = std::nullopt;

  // Index ID for internal server ID, which must be set different values
  // if the model supports multiple version or there are multiple models.
  int64_t server_idx = 0;
};

struct XLLM_CAPI_EXPORT XLLM_RequestParams {
  // Whether to include the original prompt in the response. default = false
  bool echo = false;

  // whether is a offline request. default = false.
  bool offline = false;

  // Whether to return the log probabilities of the tokens. default = false.
  bool logprobs = false;

  // Whether to ignore the end of sequence token. default = false.
  bool ignore_eos = false;

  // Whether to skip special tokens in the output text. default = true.
  bool skip_special_tokens = true;

  // Number of completions to return for each prompt. default = 1
  uint32_t n = 1;

  // Number of tokens to generate
  // the prompt token count + max_tokens can't exceed the model's max context
  // length.
  uint32_t max_tokens = 5120;

  // Number of sequences to generate for each prompt and select n best among.
  std::optional<uint32_t> best_of;

  int32_t ttlt_slo_ms;

  int32_t ttft_slo_ms;

  int32_t tpot_slo_ms;

  int32_t beam_width = 0;

  int32_t num_return_sequences = 0;

  // Number of top log probabilities to return. default = 0.
  int64_t top_logprobs = 0;

  // top_k sampling cutoff. default = -1 to disable.
  int64_t top_k = -1;

  // top_p sampling cutoff, between [0.0, 1.0]. default = 1.0
  float top_p = 1.0;

  // frequency penalty to reduce the likelihood of generating the same word
  // multiple times. values between [0.0, 2.0]. 0.0 means no penalty. default =
  // 0.0 Positive values penalize new tokens based on their existing frequency
  // in the text.
  float frequency_penalty = 0.0;

  // presence penalty to reduce the likelihood of generating words already in
  // the prompt. values between [-2.0, 2.0]. Positive values penalize new tokens
  // based on their existing in the prompt. default = 0.0
  float presence_penalty = 0.0;

  // Repetition penalty to penalize new tokens based on their occurence in the
  // text. values > 1.0 encourage the model to use new tokens, while values
  // < 1.0 encourage the model to repeat tokens. default = 1.0
  float repetition_penalty = 1.0;

  // Temperature of the sampling, between [0, 2]. default = 0.0
  // higher value will make the ouput more random.
  float temperature = 0.0;

  // The list of strings to stop generating further tokens.
  // the output will contain the stop string.
  std::optional<std::vector<std::string>> stop;

  // The list of token ids to stop generating further tokens.
  std::optional<std::vector<int32_t>> stop_token_ids;
};

enum XLLM_CAPI_EXPORT XLLM_StatusCode {
  kSuccess = 0,         // Request succeeded
  kNotInitialized = 1,  // LLM instance not initialized
  kModelNotFound = 2,   // Specified model ID not loaded
  kTimeout = 3,         // Request timed out
  kInvalidRequest = 4,  // Invalid input parameters
  kInternalError = 5,   // Internal system error
};

struct XLLM_CAPI_EXPORT XLLM_Usage {
  // The number of tokens in the prompt.
  int32_t prompt_tokens;

  // The number of tokens in the generated completion.
  int32_t completion_tokens;

  // The total number of tokens used in the request (prompt + completion).
  int32_t total_tokens;

  // The number of prompt tokens served from prefix cache.
  int32_t cached_tokens;
};

struct XLLM_CAPI_EXPORT XLLM_LogProbData {
  // Token
  std::string token;

  // Token id.
  int32_t token_id;

  // Log probability of the token.
  float logprob;
};

struct XLLM_CAPI_EXPORT XLLM_LogProb {
  // Token
  std::string token;

  // Token id.
  int32_t token_id;

  // Log probability of the token.
  float logprob;

  // Log probability of top tokens.
  std::vector<XLLM_LogProbData> top_logprobs;
};

struct XLLM_CAPI_EXPORT XLLM_Choice {
  // The index of the generated completion
  uint32_t index;

  // The generated text for completions inference
  std::optional<std::string> text;

  // The generated item for rec inference
  std::optional<uint64_t> item_id;

  // The generated message for chatcompletions inference
  std::optional<XLLM_ChatMessage> message;

  // The log probabilities of output tokens.
  std::optional<std::vector<XLLM_LogProb>> logprobs;

  // The reason of the model stoped generating tokens.
  // "stop" - the model hit a natural stop point or a provided stop sequence.
  // "length" - the maximum number of tokens specified in the request was
  // reached. "function_call" - the model called a function.
  std::string finish_reason;
};

struct XLLM_CAPI_EXPORT XLLM_Response {
  // Return code indicating request status (0 = success, non-zero = error)
  XLLM_StatusCode status_code = XLLM_StatusCode::kSuccess;

  // Optional error details (populated if status_code != kSuccess)
  std::string error_info;

  // Unique id for the completion request
  std::string id;

  // The object type, which is always "text_completion".
  std::string object;

  // The unix timestamp (in seconds) of when the completion was created.
  int64_t created;

  // The model used for the completion
  std::string model;

  // List of generated completion choices for the input prompt
  std::vector<XLLM_Choice> choices;

  // Usage statistics for the completion request.
  XLLM_Usage usage;
};

}  // namespace xllm
