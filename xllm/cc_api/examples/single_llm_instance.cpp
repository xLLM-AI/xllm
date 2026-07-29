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

#include <iostream>

#include "llm.h"
#include "service_request.h"

/**
 * In most scenes, you can follow this example to integrate xllm as an internal
 * inference engine.
 */

std::string model_path = "/export/home/models/Qwen3-4B";

int main(int argc, char** argv) {
  xllm::LLM llm_instance;
  xllm::XLLM_InitLLMOptions options;
  options.cp_size = 1;
  bool ret = llm_instance.Initialize(model_path, options);
  if (!ret) {
    std::cout << "LLM init failed." << std::endl;
    return -1;
  }

  std::cout << "LLM init succefully." << std::endl;

  const std::string model_name = "Qwen3-4B";

  xllm::cc_api_test::run_completion_request(model_name, &llm_instance);

  xllm::cc_api_test::run_chat_completion_request(model_name, &llm_instance);

  sleep(10);

  return 0;
}
