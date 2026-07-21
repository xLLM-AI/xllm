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

#include "llm.h"

std::string model_name = "Qwen3-8B";
std::string model_path = "/export/home/models/Qwen3-8B";

XLLM_LLM_Handler* service_startup_hook() {
  XLLM_LLM_Handler* llm_handler = xllm_llm_create();

  // If there is no separate setting, init_options can be passed as nullptr, and
  // the default value(XLLM_INIT_LLM_OPTIONS_DEFAULT) will be used
  XLLM_InitOptions init_options;
  xllm_llm_init_options_default(&init_options);
  init_options.cp_size = 1;
  snprintf(
      init_options.log_dir, sizeof(init_options.log_dir), "/export/xllm/log");

  bool ret =
      xllm_llm_initialize(llm_handler, model_path.c_str(), &init_options);
  if (!ret) {
    std::cout << "LLM init failed" << std::endl;
    xllm_llm_destroy(llm_handler);
    return nullptr;
  }

  std::cout << "LLM init successfully" << std::endl;

  return llm_handler;
}

void service_stop_hook(XLLM_LLM_Handler* llm_handler) {
  xllm_llm_destroy(llm_handler);
  std::cout << "LLM stop" << std::endl;
}

int main(int argc, char** argv) {
  XLLM_LLM_Handler* llm_handler = service_startup_hook();
  if (nullptr == llm_handler) {
    return -1;
  }

  // If there is no separate setting, request_params can be passed as nullptr,
  // and the default value(XLLM_REQUEST_PARAMS_DEFAULT) will be used
  XLLM_RequestParams request_params;
  xllm_llm_request_params_default(&request_params);
  request_params.max_tokens = 300;

  std::string prompt = "please briefly introduce XLLM for me";

  XLLM_Response* resp = xllm_llm_completions(
      llm_handler, model_name.c_str(), prompt.c_str(), 10000, &request_params);
  if (nullptr == resp) {
    std::cout << "LLM completions failed, response is nullptr" << std::endl;
    service_stop_hook(llm_handler);
    return -1;
  }

  if (resp->status_code != XLLM_StatusCode::kSuccess) {
    std::cout << "LLM completions failed, status code:" << resp->status_code
              << ", error info:" << resp->error_info << std::endl;
  } else {
    std::cout << "LLM completions successfully" << std::endl;

    if (nullptr != resp->choices.entries) {
      for (int i = 0; i < resp->choices.entries_size; ++i) {
        XLLM_Choice& choice = resp->choices.entries[i];
        std::cout << "xllm answer[" << choice.index << "]:" << choice.text
                  << std::endl;
      }
    }
  }

  xllm_llm_free_response(resp);

  service_stop_hook(llm_handler);

  return 0;
}
