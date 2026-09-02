/* Copyright 2025-2026 The xLLM Authors.

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

#include <google/protobuf/service.h>

#include "butil/macros.h"

namespace xllm {

// RAII: Call Run() of the closure on destruction unless release() was used.
class ClosureGuard {
 public:
  ClosureGuard() : done_(nullptr) {}

  explicit ClosureGuard(google::protobuf::Closure* done) : done_(done) {}

  ~ClosureGuard() {
    if (done_) {
      done_->Run();
    }
  }

  void reset(google::protobuf::Closure* done) {
    if (done_) {
      done_->Run();
    }
    done_ = done;
  }

  google::protobuf::Closure* release() {
    google::protobuf::Closure* const prev_done = done_;
    done_ = nullptr;
    return prev_done;
  }

  bool empty() const { return done_ == nullptr; }

 private:
  DISALLOW_COPY_AND_ASSIGN(ClosureGuard);

  google::protobuf::Closure* done_;
};

}  // namespace xllm
