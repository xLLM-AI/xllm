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

#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
// a singleton mode by version
template <typename T>
class VersionSingleton {
 public:
  template <typename... Args>
  static T* GetInstance(const std::string& version,
                        bool delete_old_versions = true,
                        int reserved_version_size =
                            2,  // default retention of the last two versions
                        Args&&... args) {
    T* instance = nullptr;

    {
      std::shared_lock<std::shared_mutex> lock(instance_map_mutex_);
      auto it = instance_map_.find(version);
      if (it != instance_map_.end()) {
        instance = it->second.get();
      }
    }

    if (instance == nullptr) {
      std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);

      auto it = instance_map_.find(version);
      if (it == instance_map_.end()) {
        instance = new T(std::forward<Args>(args)...);
        instance_map_[version] = std::unique_ptr<T>(instance);
        instance_version_list_.push_front(version);
        if (delete_old_versions) {
          if (instance_version_list_.size() > reserved_version_size) {
            auto it = instance_version_list_.begin();
            std::advance(it, reserved_version_size);
            for (; it != instance_version_list_.end(); it++) {
              instance_map_.erase(*it);
            }
            instance_version_list_.resize(reserved_version_size);
          }
        }
      } else {
        instance = it->second.get();
      }
    }

    return instance;
  }

  static std::vector<std::string> GetVersions() {
    std::shared_lock<std::shared_mutex> lock(instance_map_mutex_);
    std::vector<std::string> versions;
    for (const auto& pair : instance_map_) {
      versions.push_back(pair.first);
    }
    return versions;
  }

  static void DestroyAllInstances() {
    std::unique_lock<std::shared_mutex> lock(instance_map_mutex_);
    instance_map_.clear();
    instance_version_list_.clear();
  }

  VersionSingleton(const VersionSingleton&) = delete;
  VersionSingleton& operator=(const VersionSingleton&) = delete;

 private:
  VersionSingleton() = default;
  ~VersionSingleton() = default;

  static std::unordered_map<std::string, std::unique_ptr<T>> instance_map_;
  static std::list<std::string> instance_version_list_;
  static std::shared_mutex instance_map_mutex_;
};

template <typename T>
std::unordered_map<std::string, std::unique_ptr<T>>
    VersionSingleton<T>::instance_map_;
template <typename T>
std::list<std::string> VersionSingleton<T>::instance_version_list_;
template <typename T>
std::shared_mutex VersionSingleton<T>::instance_map_mutex_;

}  // namespace xllm
