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
#include <memory>

#include "llm.h"
#include "service_request.h"

/**
 * In some scenes, such as, one model needs to support multiple versions, or
 * multiple models needs to be supported, then you can follow this example.
 */

std::string model_path = "/export/home/models/Qwen3-4B";

int main(int argc, char** argv) {
  std::cout << "Start to bootup the first LLM instance." << std::endl;

  std::shared_ptr<xllm::LLM> llm_instance_01 = std::make_shared<xllm::LLM>();
  xllm::XLLM_InitLLMOptions options_01;
  options_01.max_memory_utilization = 0.45;
  bool ret = llm_instance_01->Initialize(model_path, options_01);
  if (!ret) {
    std::cout << "LLM instance init failed." << std::endl;
    return -1;
  }
  std::cout << "LLM init succefully." << std::endl;

  const std::string model_name = "Qwen3-4B";

  xllm::cc_api_test::run_completion_request(model_name, llm_instance_01.get());

  xllm::cc_api_test::run_chat_completion_request(model_name,
                                                 llm_instance_01.get());

  std::cout << "Start to bootup the second LLM instance." << std::endl;

  std::shared_ptr<xllm::LLM> llm_instance_02 = std::make_shared<xllm::LLM>();
  xllm::XLLM_InitLLMOptions options_02;

  // The following options must be set to create different internal servers.
  options_02.master_node_addr = "127.0.0.1:28899";
  options_02.transfer_listen_port = 27000;
  options_02.server_idx = 1;
  options_02.max_memory_utilization = 0.9;

  ret = llm_instance_02->Initialize(model_path, options_02);
  if (!ret) {
    std::cout << "LLM instance init failed." << std::endl;
    return -1;
  }
  std::cout << "LLM init succefully." << std::endl;

  xllm::cc_api_test::run_completion_request(model_name, llm_instance_02.get());

  xllm::cc_api_test::run_chat_completion_request(model_name,
                                                 llm_instance_02.get());

  sleep(10);

  return 0;
}