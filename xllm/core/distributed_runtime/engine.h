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

#include <glog/logging.h>

#include <memory>

#include "common/types.h"
#include "core/kv_cache/manager/kv_cache_manager.h"
#include "framework/model/model_args.h"
#include "framework/tokenizer/tokenizer.h"
#include "framework/tokenizer/tokenizer_args.h"
#include "runtime/options.h"

namespace xllm {

class Engine {
 public:
  virtual ~Engine() = default;

  virtual bool init() { return true; };

  virtual bool init(MasterStatus master_status) { return true; };

  // return the tokenizer
  virtual const Tokenizer* tokenizer() const { return tokenizer_.get(); }

  // The initialized engine owns the manager throughout scheduler use.
  virtual KVCacheManager* kv_cache_manager() const {
    return kv_cache_manager_.get();
  }

  // return the model args
  virtual const ModelArgs& model_args() const { return args_; }

  // return the tokenizer args
  virtual const TokenizerArgs& tokenizer_args() const {
    return tokenizer_args_;
  }

  // Start/stop online timeline profiling on all workers. CUDA only for now.
  virtual bool start_profile() {
    LOG(ERROR) << "start_profile is not implemented for this engine!";
    return false;
  };

  virtual bool stop_profile() {
    LOG(ERROR) << "stop_profile is not implemented for this engine!";
    return false;
  };

 protected:
  // model args
  ModelArgs args_;

  // Tokenizer args
  TokenizerArgs tokenizer_args_;

  // Request-cache coordination selected by KVCacheManagerFactory.
  std::unique_ptr<KVCacheManager> kv_cache_manager_;

  // tokenizer
  std::unique_ptr<Tokenizer> tokenizer_;
};

}  // namespace xllm
