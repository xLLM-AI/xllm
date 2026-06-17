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

#include <gtest/gtest.h>

#include "core/framework/config/model_config.h"
#include "framework/chat_template/deepseek_v32_cpp_template.h"
#include "framework/chat_template/deepseek_v4_cpp_template.h"
#include "framework/chat_template/jinja_chat_template.h"

namespace xllm {
namespace {

class ScopedUseCppChatTemplate final {
 public:
  explicit ScopedUseCppChatTemplate(bool enabled)
      : old_value_(ModelConfig::get_instance().use_cpp_chat_template()) {
    ModelConfig::get_instance().use_cpp_chat_template(enabled);
  }

  ~ScopedUseCppChatTemplate() {
    ModelConfig::get_instance().use_cpp_chat_template(old_value_);
  }

 private:
  bool old_value_;
};

TEST(ChatTemplateFactory, DeepseekV32FallsBackToJinjaWhenFlagDisabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/false);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplate::create(args, /*model_type=*/"deepseek_v32");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4FallsBackToJinjaWhenFlagDisabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/false);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplate::create(args, /*model_type=*/"deepseek_v4");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, DeepseekV4UsesCppTemplateWhenFlagEnabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/true);
  TokenizerArgs args;

  std::unique_ptr<ChatTemplate> impl =
      ChatTemplate::create(args, /*model_type=*/"deepseek_v4");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_EQ(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_NE(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

TEST(ChatTemplateFactory, NonDeepseekModelUsesJinjaWhenFlagEnabled) {
  ScopedUseCppChatTemplate scoped_flag(/*enabled=*/true);
  TokenizerArgs args;

  ModelConfig::get_instance().use_cpp_chat_template(false);
  std::unique_ptr<ChatTemplate> impl =
      ChatTemplate::create(args, /*model_type=*/"qwen3");

  ASSERT_TRUE(impl != nullptr);
  EXPECT_NE(dynamic_cast<JinjaChatTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV32CppTemplate*>(impl.get()), nullptr);
  EXPECT_EQ(dynamic_cast<DeepseekV4CppTemplate*>(impl.get()), nullptr);
}

}  // namespace
}  // namespace xllm
