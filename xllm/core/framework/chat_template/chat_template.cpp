/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "framework/chat_template/chat_template.h"

#include <glog/logging.h>

#include "core/framework/config/model_config.h"
#include "framework/chat_template/deepseek_v32_cpp_template.h"
#include "framework/chat_template/deepseek_v4_cpp_template.h"
#include "framework/chat_template/jinja_chat_template.h"

namespace xllm {

std::unique_ptr<ChatTemplate> ChatTemplate::create(
    const TokenizerArgs& tokenizer_args,
    const std::string& model_type) {
  if (::xllm::ModelConfig::get_instance().use_cpp_chat_template()) {
    if (model_type == "deepseek_v32") {
      LOG(INFO) << "Using native C++ chat template for "
                << "model_type: " << model_type;
      return std::make_unique<DeepseekV32CppTemplate>(tokenizer_args);
    } else if (model_type == "deepseek_v4") {
      LOG(INFO) << "Using native C++ chat template for "
                << "model_type: " << model_type;
      return std::make_unique<DeepseekV4CppTemplate>(tokenizer_args);
    } else {
      LOG(FATAL) << "cpp_chat_template only support deepseekv32 and deepseekv4 "
                    "models currently.";
    }
  }
  LOG(INFO) << "Using Jinja chat template for "
            << "model_type: " << model_type;
  return std::make_unique<JinjaChatTemplate>(tokenizer_args);
}

}  // namespace xllm
