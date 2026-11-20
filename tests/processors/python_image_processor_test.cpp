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

#include <gtest/gtest.h>
#include <pybind11/embed.h>
#include <pybind11/stl.h>
#include <torch/torch.h>

#include <array>
#include <memory>
#include <string>
#include <utility>
#include <vector>

#include "core/framework/config/model_config.h"
#include "core/framework/tokenizer/tokenizer.h"
#include "core/util/scope_guard.h"
#include "processors/multimodal_processor_pipeline.h"
#include "processors/qwen2_vl_image_processor.h"
#include "processors/text_prompt_processor.h"

namespace py = pybind11;

namespace xllm {
namespace {

TEST(PythonImageProcessorTest, ModelDirectoriesRemainLocalToTheirProcessors) {
  py::scoped_interpreter interpreter(/*init_signal_handlers=*/false);
  py::dict scope;
  py::exec(R"PY(
import sys
import types
import torch

module = types.ModuleType("xllm.pybind.multimodal")
module.initialized = []
module.processed = []

def _initialize(model: str) -> None:
    module.initialized.append(model)

def _preprocess(images: list[torch.Tensor], model: str) -> dict[str, torch.Tensor]:
    module.processed.append(model)
    value = {"model-a": 3.0, "model-b": 7.0}[model]
    return {
        "pixel_values": torch.full((4, 2), value),
        "image_grid_thw": torch.tensor([[1, 2, 2]]),
    }

module.initialize_image_processor = _initialize
module.preprocess_tensors = _preprocess
sys.modules[module.__name__] = module
)PY",
           scope);

  auto& config = ModelConfig::get_instance();
  const std::string original_model = config.model();
  const std::string original_impl = config.model_impl();
  ScopeGuard restore_config([&]() {
    config.model() = original_model;
    config.model_impl() = original_impl;
  });
  config.model().clear();
  config.model_impl() = "python";

  ModelArgs args;
  args.mm_image_normalize_mean({0.0F, 0.0F, 0.0F})
      .mm_image_normalize_std({1.0F, 1.0F, 1.0F});
  using Pipeline =
      MultimodalProcessor<TextPromptProcessor, Qwen2VLImageProcessor>;
  Pipeline first(args,
                 std::make_shared<Tokenizer>(),
                 TokenizerArgs{},
                 /*max_sequence_length=*/0,
                 /*model_path=*/"model-a");
  Pipeline second(args,
                  std::make_shared<Tokenizer>(),
                  TokenizerArgs{},
                  /*max_sequence_length=*/0,
                  /*model_path=*/"model-b");

  MMInputItem image;
  image.type = MMType::IMAGE;
  image.decode_image = torch::zeros({3, 8, 8}, torch::dtype(torch::kUInt8));
  const std::array cases{std::pair{&first, 3.0F},
                         std::pair{&second, 7.0F},
                         std::pair{&first, 3.0F}};
  for (const auto& [processor, expected] : cases) {
    MMData data;
    ASSERT_TRUE(processor->process_multimodal(MMInput({image}), data));
    const auto& items = data.items<MMItemVec>();
    ASSERT_EQ(items.size(), 1u);
    EXPECT_TRUE(torch::equal(
        std::get<torch::Tensor>(items.front().data().at("pixel_values")),
        torch::full({4, 2}, expected)));
    EXPECT_TRUE(torch::equal(
        std::get<torch::Tensor>(items.front().data().at("image_grid_thw")),
        torch::tensor({{1, 2, 2}}, torch::dtype(torch::kInt64))));
  }
  py::object module = scope["module"];
  EXPECT_EQ(module.attr("initialized").cast<std::vector<std::string>>(),
            (std::vector<std::string>{"model-a", "model-b"}));
  EXPECT_EQ(module.attr("processed").cast<std::vector<std::string>>(),
            (std::vector<std::string>{"model-a", "model-b", "model-a"}));
  EXPECT_TRUE(config.model().empty());
}

}  // namespace
}  // namespace xllm
