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

#include <atb/atb_infer.h>
#include <c10/util/Logging.h>

#include <memory>
#include <unordered_map>

namespace xllm_kernel {

// Contexts retain ATB tiling resources used by captured graphs. Keep them for
// the worker lifetime, independently of each capture's temporary operation.
class AtbContext final {
 public:
  explicit AtbContext(aclrtStream stream) {
    CHECK((atb::CreateContext(&context_)) == 0);
    CHECK((context_->SetExecuteStream(stream)) == 0);
  }

  ~AtbContext() { CHECK(atb::DestroyContext(context_) == 0); }

  AtbContext(const AtbContext&) = delete;
  AtbContext& operator=(const AtbContext&) = delete;

  atb::Context* get() const { return context_; }

 private:
  atb::Context* context_ = nullptr;
};

inline std::shared_ptr<AtbContext> get_context(aclrtStream stream) {
  static thread_local std::unordered_map<aclrtStream,
                                         std::shared_ptr<AtbContext>>
      contexts;
  auto& context = contexts[stream];
  if (!context) {
    context = std::make_shared<AtbContext>(stream);
  }
  return context;
}

}  // namespace xllm_kernel
