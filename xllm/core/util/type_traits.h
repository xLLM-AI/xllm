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
#include <optional>

namespace xllm {

template <typename value_type>
struct remove_optional {
  using type = value_type;
};

// specialization for optional
template <typename value_type>
struct remove_optional<std::optional<value_type>> {
  using type = value_type;
};

/// alias template for remove_optional
template <typename value_type>
using remove_optional_t = typename remove_optional<value_type>::type;

}  // namespace xllm
