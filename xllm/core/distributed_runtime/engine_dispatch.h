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

#include <glog/logging.h>

#include <utility>

namespace xllm {

namespace detail {

// Unreachable: dispatch_engine CHECKs that exactly one engine is non-null.
template <typename Function, typename Engine, typename... Rest>
decltype(auto) invoke_first_engine(Function&& function,
                                   Engine* engine,
                                   Rest*... rest) {
  if (engine != nullptr) {
    return std::forward<Function>(function)(engine);
  }
  if constexpr (sizeof...(rest) > 0) {
    return invoke_first_engine(std::forward<Function>(function), rest...);
  } else {
    LOG(FATAL) << "dispatch_engine requires a non-null engine";
  }
}

}  // namespace detail

// Invokes function on the active engine; exactly one pointer is non-null.
template <typename Function, typename... Engines>
decltype(auto) dispatch_engine(Function&& function, Engines*... engines) {
  CHECK_EQ(((engines != nullptr) + ...), 1)
      << "exactly one engine must be non-null";
  return detail::invoke_first_engine(std::forward<Function>(function),
                                     engines...);
}

}  // namespace xllm
