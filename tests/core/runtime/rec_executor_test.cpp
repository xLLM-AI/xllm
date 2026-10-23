/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/rec_executor.h"

#include <gtest/gtest.h>

#include "core/framework/config/model_config.h"

namespace xllm {
namespace {

TEST(RecExecutorTest, RejectsIluGraphBeforeExecutorLookup) {
  runtime::Options options;
  options.backend("rec").enable_graph(true);
  const ModelArgs model_args;

  EXPECT_DEATH(
      {
        ModelConfig::get_instance().model_impl("auto");
        RecExecutor executor(nullptr,
                             TransformerModelArgs(model_args),
                             torch::Device(torch::kCPU),
                             options);
      },
      "ILU does not support native Rec graph execution");
}

}  // namespace
}  // namespace xllm
