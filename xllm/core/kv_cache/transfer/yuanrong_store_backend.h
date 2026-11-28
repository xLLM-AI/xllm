/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <cstdint>
#include <string>
#include <vector>

#include "core/kv_cache/transfer/kv_cache_store_backend.h"
#include "core/kv_cache/transfer/yuanrong_store_client.h"

namespace xllm {

class YuanRongStoreBackend final : public KVCacheStoreBackend {
 public:
  YuanRongStoreBackend() = default;
  YuanRongStoreBackend(const YuanRongStoreBackend&) = delete;
  YuanRongStoreBackend& operator=(const YuanRongStoreBackend&) = delete;
  ~YuanRongStoreBackend() override;

  bool init(const std::string& host, int32_t port, int32_t timeout_ms);

  std::vector<uint8_t> batch_put(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) override;

  std::vector<uint8_t> batch_get(
      const std::vector<std::string>& keys,
      const std::vector<StoreBuffer>& buffers) override;

 private:
  void* library_ = nullptr;
  void* client_ = nullptr;
  decltype(&xllm_yr_store_destroy) destroy_ = nullptr;
  decltype(&xllm_yr_store_mset) mset_ = nullptr;
  decltype(&xllm_yr_store_mget) mget_ = nullptr;
  decltype(&xllm_yr_store_thread_error) last_error_ = nullptr;
};

}  // namespace xllm
