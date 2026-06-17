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

#pragma once

#include <torch/torch.h>

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <tuple>
#include <unordered_map>

#include "common/macros.h"
#include "core/framework/model/model_args.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/util/slice.h"

namespace xllm {

class MPositionGenerator {
 public:
  virtual ~MPositionGenerator() = default;

  virtual std::tuple<torch::Tensor, int32_t> generate(
      Slice<int32_t> tokens,
      const MMData& mm_data,
      const ModelArgs& model_args) const = 0;
};

class MPositionGeneratorFactory {
 public:
  using Creator = std::function<std::unique_ptr<MPositionGenerator>()>;

  static MPositionGeneratorFactory& get_instance();

  bool register_creator(const std::string& name, Creator creator);

  std::unique_ptr<MPositionGenerator> create_mposition_generator(
      const std::string& name);

  DISALLOW_COPY_AND_ASSIGN(MPositionGeneratorFactory);

 private:
  MPositionGeneratorFactory() = default;

  ~MPositionGeneratorFactory() = default;

  std::unordered_map<std::string, Creator> creators_;
};

class QwenVLMPositionGenerator final : public MPositionGenerator {
 public:
  std::tuple<torch::Tensor, int32_t> generate(
      Slice<int32_t> tokens,
      const MMData& mm_data,
      const ModelArgs& model_args) const override;
};

class Qwen3VLMPositionGenerator final : public MPositionGenerator {
 public:
  std::tuple<torch::Tensor, int32_t> generate(
      Slice<int32_t> tokens,
      const MMData& mm_data,
      const ModelArgs& model_args) const override;
};

class Glm4VMPositionGenerator final : public MPositionGenerator {
 public:
  std::tuple<torch::Tensor, int32_t> generate(
      Slice<int32_t> tokens,
      const MMData& mm_data,
      const ModelArgs& model_args) const override;
};

#define REGISTER_MPOSITION_GENERATOR(ModelType, MPositionGeneratorClass)       \
  namespace {                                                                  \
  bool ModelType##_mposition_generator_registered = []() -> bool {             \
    return ::xllm::MPositionGeneratorFactory::get_instance().register_creator( \
        #ModelType,                                                            \
        []() { return std::make_unique<MPositionGeneratorClass>(); });         \
  }();                                                                         \
  }

}  // namespace xllm
