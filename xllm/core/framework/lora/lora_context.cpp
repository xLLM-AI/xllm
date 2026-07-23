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

#include "framework/lora/lora_context.h"

#include <vector>

namespace xllm {
namespace {

thread_local std::vector<LoRAContextFrame> tls_stack;

}  // namespace

void push_lora_context(LoRAContextFrame frame) {
  tls_stack.push_back(std::move(frame));
}

void pop_lora_context() {
  if (!tls_stack.empty()) {
    tls_stack.pop_back();
  }
}

const LoRAContextFrame* current_lora_context() {
  if (tls_stack.empty()) return nullptr;
  return &tls_stack.back();
}

void set_lora_context_layer(int32_t layer_index) {
  if (!tls_stack.empty()) {
    tls_stack.back().layer_index = layer_index;
  }
}

}  // namespace xllm
