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

#include <mutex>
#include <unordered_map>

#include "xllm_server.h"

namespace xllm {

class ServerRegistry {
 public:
  static ServerRegistry& get_instance() {
    static ServerRegistry instance;
    return instance;
  }

  XllmServer* register_server(const std::string& name);
  void unregister_server(const std::string& name);
  XllmServer* get_server(const std::string& name);
  XllmServer* try_get_server(const std::string& name);

 private:
  ServerRegistry() = default;
  ~ServerRegistry() = default;
  DISALLOW_COPY_AND_ASSIGN(ServerRegistry);

  std::unordered_map<std::string, std::unique_ptr<XllmServer>> servers_;
  std::mutex mutex_;
};

}  // namespace xllm
