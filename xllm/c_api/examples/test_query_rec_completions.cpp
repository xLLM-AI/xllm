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

#include <unistd.h>

#include <cstring>
#include <iostream>
#include <random>

#include "rec.h"

std::string model_name = "Qwen3-0.6B";
std::string model_path = "/export/home/models/Qwen3-0.6B";

XLLM_REC_Handler* service_startup_hook() {
  XLLM_REC_Handler* rec_handler = xllm_rec_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_REC_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_rec_init_options_default(&init_options);
  init_options.block_size = 1;
  init_options.max_tokens_per_batch = 8192;
  init_options.max_seqs_per_batch = 4;
  init_options.max_memory_utilization = 0.8;
  init_options.max_cache_size = 500000;
  init_options.beam_width = 64;
  init_options.max_decode_rounds = 3;
  init_options.enable_chunked_prefill = false;
  init_options.enable_prefix_cache = false;
#if defined(USE_NPU)
  init_options.enable_graph = false;
  init_options.enable_graph_mode_decode_no_padding = false;
  init_options.enable_prefill_piecewise_graph = false;
  init_options.rec_worker_max_concurrency = 1;
#endif

  bool ret =
      xllm_rec_initialize(rec_handler, model_path.c_str(), &init_options);
  if (!ret) {
    std::cout << "REC init failed" << std::endl;
    xllm_rec_destroy(rec_handler);
    return nullptr;
  }

  std::cout << "REC init successfully" << std::endl;

  return rec_handler;
}

void service_stop_hook(XLLM_REC_Handler* rec_handler) {
  xllm_rec_destroy(rec_handler);
  std::cout << "REC stop" << std::endl;
}

int main(int argc, char** argv) {
  if (argc > 1) {
    model_path = argv[1];
  }

  std::cout << "Using model path: " << model_path << std::endl;

  XLLM_REC_Handler* rec_handler = service_startup_hook();
  if (nullptr == rec_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_rec_request_params_default(&request_params);
  request_params.max_tokens = 3;
  request_params.beam_width = 64;
  request_params.logprobs = true;
  // request_params.temperature = 1.0;
  request_params.top_k = 64;
  request_params.top_logprobs = 64;
  // request_params.top_p = 1.0;
  // request_params.repetition_penalty = 1.0;

  // Qwen3-0.6B tokenizer ids for: "where is bejing?".
  std::vector<int32_t> token_ids = {2870, 374, 387, 98168, 30};

  size_t token_size = token_ids.size();
  const int32_t* token_ids_ptr = token_ids.data();

  XLLM_Response* resp = xllm_rec_token_completions(rec_handler,
                                                   model_name.c_str(),
                                                   token_ids_ptr,
                                                   token_size,
                                                   100000,
                                                   &request_params);
  if (nullptr == resp) {
    std::cout << "REC completions failed, response is nullptr" << std::endl;
    service_stop_hook(rec_handler);
    return -1;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::cout << "REC completions failed, status code:" << resp->status_code
              << ", error info:" << resp->error_info << std::endl;
  } else {
    std::cout << "REC completions successfully, size:"
              << resp->choices.entries_size << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        std::cout << "token size: " << choice.token_size
                  << ",logprobs size:" << choice.logprobs.entries_size
                  << std::endl;

        for (int j = 0; j < choice.token_size; j++) {
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << choice.token_ids[j] << std::endl;
        }

        for (int j = 0; j < choice.logprobs.entries_size; j++) {
          XLLM_LogProb& logprob = choice.logprobs.entries[j];
          std::cout << "xllm answer[" << choice.index
                    << "]: token id=" << logprob.token_id
                    << ", token logprob=" << logprob.logprob << std::endl;
        }
      }
    }
  }

  xllm_rec_free_response(resp);

  service_stop_hook(rec_handler);

  return 0;
}