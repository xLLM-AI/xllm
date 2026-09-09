/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include <type_traits>
#include <utility>

namespace xllm {

// RAII object that invoked a callback on destruction.
template <typename Fun>
class ScopeGuard final {
 public:
  template <typename FuncArg>
  ScopeGuard(FuncArg&& f) : callback_(std::forward<FuncArg>(f)) {}

  // disallow copy and move
  ScopeGuard(const ScopeGuard&) = delete;
  ScopeGuard(ScopeGuard&&) = delete;
  ScopeGuard& operator=(const ScopeGuard&) = delete;
  ScopeGuard& operator=(ScopeGuard&&) = delete;

  ~ScopeGuard() noexcept {
    if (!dismissed_) {
      callback_();
    }
  }

  void dismiss() noexcept { dismissed_ = true; }

 private:
  Fun callback_;
  bool dismissed_ = false;
};

// allow function-to-pointer implicit conversions
template <typename Fun>
ScopeGuard(Fun&&) -> ScopeGuard<std::decay_t<Fun>>;

}  // namespace xllm

// Declares a ScopeGuard object with the given callback.
// Example: SCOPE_GUARD([&]{...});
#define SCOPE_GUARD xllm::ScopeGuard SAFE_CONCAT(scope_guard, __LINE__)
