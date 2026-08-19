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

#include <atomic>
#include <chrono>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include "macros.h"

namespace xllm {

// Singleton class to manage cluster health status.
// When master detects any worker disconnection,
// it will set the cluster health status to unhealthy.
class HealthCheckManager {
 public:
  using HealthCheckFunc = std::function<bool()>;

  static HealthCheckManager& instance() {
    static HealthCheckManager manager;
    return manager;
  }

  // Check if the cluster is healthy
  bool is_healthy() const {
    return is_healthy_.load(std::memory_order_acquire);
  }

  // Set cluster health status to unhealthy
  void set_unhealthy(const std::string& reason = "") {
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      unhealthy_reason_ = reason;
    }
    is_healthy_.store(false, std::memory_order_release);
  }

  // Set cluster health status to healthy
  void set_healthy() {
    is_healthy_.store(true, std::memory_order_release);
    {
      std::lock_guard<std::mutex> lock(reason_mutex_);
      unhealthy_reason_.clear();
    }
  }

  // Get the reason for unhealthy status
  std::string unhealthy_reason() const {
    std::lock_guard<std::mutex> lock(reason_mutex_);
    return unhealthy_reason_;
  }

  // Register a health check function for a worker
  void register_health_check(int worker_rank, HealthCheckFunc func) {
    health_checks_[worker_rank] = std::move(func);
  }

  // Start background health check thread
  void start_health_check_thread(int interval_ms = 3000) {
    if (health_check_running_.load()) {
      return;
    }

    health_check_running_.store(true);
    health_check_thread_ = std::thread([this, interval_ms]() {
      while (health_check_running_.load()) {
        check_all_workers();
        std::this_thread::sleep_for(std::chrono::milliseconds(interval_ms));
      }
    });
  }

  // Stop background health check thread
  void stop_health_check_thread() {
    health_check_running_.store(false);
    if (health_check_thread_.joinable()) {
      health_check_thread_.join();
    }
  }

 private:
  HealthCheckManager() : is_healthy_(true), health_check_running_(false) {}
  ~HealthCheckManager() { stop_health_check_thread(); }

  DISALLOW_COPY_AND_ASSIGN(HealthCheckManager);

  void check_all_workers() {
    for (const auto& [rank, check_func] : health_checks_) {
      if (!check_func()) {
        {
          std::lock_guard<std::mutex> lock(reason_mutex_);
          unhealthy_reason_ =
              "Worker " + std::to_string(rank) + " disconnected";
        }
        is_healthy_.store(false, std::memory_order_release);
        return;
      }
    }
  }

  std::atomic<bool> is_healthy_;
  std::atomic<bool> health_check_running_;
  mutable std::mutex reason_mutex_;
  std::string unhealthy_reason_;
  std::unordered_map<int, HealthCheckFunc> health_checks_;
  std::thread health_check_thread_;
};

}  // namespace xllm
