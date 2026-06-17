/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include <folly/Function.h>

#include <functional>
#include <future>
#include <vector>

#include "common/macros.h"
#include "common/options.h"
#include "common/rate_limiter.h"
#include "common/types.h"
#include "engine.h"
#include "framework/request/request_params.h"
namespace xllm {

class Master {
 public:
  explicit Master(const Options& options, EngineType type);
  virtual ~Master() = default;
  virtual void run() = 0;
  virtual const Options& options() const { return options_; }
  EngineType engine_type() const { return engine_type_; }

  virtual bool sleep() { return false; }

  virtual bool wakeup() { return false; }

  virtual bool wakeup(const WakeupOptions& options) { return false; }

  virtual bool link_p2p(const std::vector<std::string>& remote_addrs) {
    return false;
  }

  // Start/stop online timeline profiling on all workers. Forwards to the
  // engine, which broadcasts to every worker. CUDA only for now.
  virtual bool start_profile() {
    return engine_ ? engine_->start_profile() : false;
  }

  virtual bool stop_profile() {
    return engine_ ? engine_->stop_profile() : false;
  }

  virtual bool unlink_p2p(const std::vector<std::string>& remote_addrs) {
    return false;
  }

  MasterStatus get_master_status() const { return master_status_; }

  bool is_sleeping() const { return master_status_ != MasterStatus::WAKEUP; }

  void set_master_status(MasterStatus master_status) {
    master_status_ = master_status;
  }

  RateLimiter* get_rate_limiter() { return &rate_limiter_; }

 protected:
  Options options_;
  EngineType engine_type_ = EngineType::INVALID;
  std::unique_ptr<Engine> engine_;
  RateLimiter rate_limiter_;
  MasterStatus master_status_{MasterStatus::WAKEUP};
};

std::unique_ptr<Master> create_master(const std::string& backend,
                                      const Options& options);

std::unique_ptr<Master> fork_master(Master* master, const Options& options);

}  // namespace xllm
