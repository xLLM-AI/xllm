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

#include "service_request.h"

#include <iostream>

namespace xllm {
namespace cc_api_test {

void run_completion_request(const std::string& model_name,
                            xllm::LLM* llm_instance) {
  std::cout << "*** llm completions start ***" << std::endl;

  std::string prompt =
      "recommend 3 cheap and easy-to-use electric shavers, briefly "
      "describe the product name, price, and features";
  xllm::XLLM_RequestParams params;
  params.max_tokens = 500;
  xllm::XLLM_Response response =
      llm_instance->Completions(model_name, prompt, 20000, params);

  if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
    std::cout << "LLM completions failed, error info: " << response.error_info
              << std::endl;
    return;
  } else {
    for (auto choice : response.choices) {
      if (choice.text.has_value()) {
        std::cout << "LLM completions output: " << choice.text.value().c_str()
                  << std::endl;
      }
    }
  }

  std::cout << "*** llm completions end ***" << std::endl;
}

void run_chat_completion_request(const std::string& model_name,
                                 xllm::LLM* llm_instance) {
  std::cout << "*** llm chat completions start ***" << std::endl;

  xllm::XLLM_ChatMessage message;
  message.role = "user";
  message.content =
      "You are an expert in e-commerce scenarios. The current scenario is an "
      "e-commerce search engine with a comprehensive range of business "
      "categories. Your task is to determine whether 'user query' and "
      "'product title' are related in the e-commerce search engine. "
      "Discrimination criteria: If the search for 'user query' returns' "
      "product title 'that meets the user's needs, then the task is "
      "relevant. Output requirement: Please provide the answer in the "
      "'related' or 'unrelated' section, without mentioning any other "
      "content. User query: 'Hotpot sauce'. Product title: 'Grassland Red "
      "Sun Hotpot Base Dip Multi flavored Barbecue Sauce Tomato Sauce Leek "
      "Flower Sauce Nightsnack Paired with [New] Spicy Barbecue Sauce 100g'";

  std::vector<xllm::XLLM_ChatMessage> messages;
  messages.emplace_back(message);
  xllm::XLLM_RequestParams params;
  params.max_tokens = 100;

  xllm::XLLM_Response response =
      llm_instance->ChatCompletions(model_name, messages, 20000, params);

  if (response.status_code != xllm::XLLM_StatusCode::kSuccess) {
    std::cout << "LLM completions failed, error info: " << response.error_info
              << std::endl;
    return;
  } else {
    for (auto choice : response.choices) {
      if (choice.message.has_value()) {
        std::cout << "LLM completions output: role: "
                  << choice.message.value().role.c_str()
                  << ",content: " << choice.message.value().content.c_str()
                  << std::endl;
      }
    }
  }

  std::cout << "*** llm chat completions end ***" << std::endl;
}

}  // namespace cc_api_test
}  // namespace xllm
