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

#include "xllm_server_registry.h"

namespace xllm {

XllmServer* ServerRegistry::register_server(const std::string& name) {
  {
    LOG(INFO) << "Register server " << name << ".";

    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) != servers_.end()) {
      LOG(ERROR) << "Register server failed, " << name
                 << " has been registered already.";
      return servers_[name].get();
    }

    servers_[name] = XllmServerFactory::create_xllm_server();
    return servers_[name].get();
  }
}

void ServerRegistry::unregister_server(const std::string& name) {
  {
    LOG(INFO) << "Unregister server " << name << ".";

    std::lock_guard<std::mutex> lock(mutex_);
    auto iter = servers_.find(name);
    if (iter == servers_.end()) {
      LOG(ERROR) << "Unregister server failed, " << name
                 << " isn't registered.";
    } else {
      servers_.erase(iter);
    }
  }
}

XllmServer* ServerRegistry::get_server(const std::string& name) {
  {
    std::lock_guard<std::mutex> lock(mutex_);
    if (servers_.find(name) == servers_.end()) {
      LOG(ERROR) << "Server " << name << " doesn't exist.";
      return nullptr;
    }

    return servers_[name].get();
  }
}

XllmServer* ServerRegistry::try_get_server(const std::string& name) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto it = servers_.find(name);
  if (it == servers_.end()) {
    return nullptr;
  }
  return it->second.get();
}

}  // namespace xllm
