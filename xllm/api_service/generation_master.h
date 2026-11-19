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

#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <optional>
#include <utility>
#include <variant>
#include <vector>

#include "core/distributed_runtime/llm_master.h"
#include "core/distributed_runtime/vlm_master.h"

namespace xllm {

// Keeps the selected model alive through asynchronous API callbacks. Each
// concrete master retains its own admission and execution path.
class GenerationMaster final {
 private:
  using Masters =
      std::variant<std::shared_ptr<LLMMaster>, std::shared_ptr<VLMMaster>>;
  explicit GenerationMaster(Masters master) : master_(std::move(master)) {}

  template <typename Function>
  decltype(auto) visit(Function&& function) const {
    return std::visit(std::forward<Function>(function), master_);
  }

 public:
  static std::optional<GenerationMaster> from(std::shared_ptr<Master> master) {
    if (auto llm = std::dynamic_pointer_cast<LLMMaster>(master)) {
      return GenerationMaster(std::move(llm));
    }
    if (auto vlm = std::dynamic_pointer_cast<VLMMaster>(master)) {
      return GenerationMaster(std::move(vlm));
    }
    return std::nullopt;
  }

  const Options& options() const {
    return visit(
        [](const auto& master) -> const Options& { return master->options(); });
  }

  const ModelArgs* model_args() const {
    return visit([](const auto& master) { return master->model_args(); });
  }

  bool uses_multimodal_processor() const {
    return std::holds_alternative<std::shared_ptr<VLMMaster>>(master_);
  }

  bool supports_image_input() const {
    return uses_multimodal_processor() && model_args()->image_token_id() != 0;
  }

  const Tokenizer& tokenizer() const {
    return visit([](const auto& master) -> const Tokenizer& {
      return master->tokenizer();
    });
  }

  const ChatTemplate& chat_template() const {
    return visit([](const auto& master) -> const ChatTemplate& {
      return master->chat_template();
    });
  }

  RateLimiter* get_rate_limiter() const {
    return visit([](const auto& master) { return master->get_rate_limiter(); });
  }

  template <typename Input>
  void handle_request(Input input,
                      std::optional<std::vector<int32_t>> tokens,
                      RequestParams params,
                      std::optional<Call*> call,
                      OutputCallback callback) const {
    visit([&](const auto& master) {
      master->handle_request(std::move(input),
                             std::move(tokens),
                             std::move(params),
                             call,
                             std::move(callback));
    });
  }

  void count_chat_tokens(std::vector<Message> messages,
                         RequestParams params,
                         std::function<void(Status, int32_t)> callback) const {
    visit([&](const auto& master) {
      master->count_chat_tokens(
          std::move(messages), std::move(params), std::move(callback));
    });
  }

  bool handle_rpc_response(const RequestOutput& output) const {
    return visit([&](const auto& master) {
      return master->handle_rpc_response(output);
    });
  }

 private:
  Masters master_;
};

}  // namespace xllm
