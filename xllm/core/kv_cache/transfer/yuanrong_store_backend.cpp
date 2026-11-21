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

#include "core/kv_cache/transfer/yuanrong_store_backend.h"

#include <absl/strings/escaping.h>
#include <dlfcn.h>
#include <glog/logging.h>

#include <algorithm>
#include <string>

namespace xllm {
namespace {

// Own the encoded keys until the SDK call returns. Range data stays in the
// caller's StoreBuffers.
struct StoreBatch {
  std::vector<std::string> keys;
  std::vector<const char*> key_ptrs;
  std::vector<XllmYrStoreBuffer> ranges;
};

bool prepare_batch(const std::vector<std::string>& raw_keys,
                   const std::vector<StoreBuffer>& buffers,
                   StoreBatch& batch) {
  if (raw_keys.size() != buffers.size()) {
    LOG(ERROR) << "YuanRong store requires one buffer per key";
    return false;
  }
  batch.keys.reserve(raw_keys.size());
  batch.key_ptrs.reserve(raw_keys.size());
  batch.ranges.reserve(buffers.size());
  for (size_t i = 0; i < buffers.size(); ++i) {
    const auto& buffer = buffers[i];
    if (buffer.addresses.size() != buffer.sizes.size()) {
      LOG(ERROR) << "Mismatched YuanRong store buffer range arrays";
      return false;
    }
    batch.keys.emplace_back(absl::BytesToHexString(raw_keys[i]));
    batch.key_ptrs.push_back(batch.keys.back().c_str());
    batch.ranges.push_back(
        {buffer.addresses.data(), buffer.sizes.data(), buffer.sizes.size()});
  }
  return true;
}

template <typename Function>
Function load_function(void* library, const char* name) {
  auto fn = reinterpret_cast<Function>(dlsym(library, name));
  if (fn == nullptr) {
    LOG(ERROR) << "Cannot resolve " << name << ": " << dlerror();
  }
  return fn;
}

}  // namespace

YuanRongStoreBackend::~YuanRongStoreBackend() {
  if (client_) {
    destroy_(client_);
  }
  if (library_) {
    dlclose(library_);
  }
}

bool YuanRongStoreBackend::init(const std::string& host,
                                int32_t port,
                                int32_t timeout_ms) {
  if (client_) {
    return true;
  }
  // DataSystem has process-wide state that outlives individual KV clients.
  if (!library_) {
    library_ = dlopen("libxllm_yuanrong_store.so",
                      RTLD_NOW | RTLD_LOCAL | RTLD_NODELETE);
  }
  if (!library_) {
    LOG(ERROR) << "Cannot load YuanRong store client: " << dlerror();
    return false;
  }
  auto create = load_function<decltype(&xllm_yr_store_create)>(
      library_, "xllm_yr_store_create");
  destroy_ =
      load_function<decltype(destroy_)>(library_, "xllm_yr_store_destroy");
  mset_ = load_function<decltype(mset_)>(library_, "xllm_yr_store_mset");
  mget_ = load_function<decltype(mget_)>(library_, "xllm_yr_store_mget");
  last_error_ = load_function<decltype(last_error_)>(
      library_, "xllm_yr_store_thread_error");
  if (!create || !destroy_ || !mset_ || !mget_ || !last_error_) {
    return false;
  }
  client_ = create(host.c_str(), port, timeout_ms);
  if (!client_) {
    LOG(ERROR) << "YuanRong store client create failed for " << host << ":"
               << port << ": " << last_error_();
    return false;
  }
  LOG(INFO) << "YuanRong store backend ready, worker=" << host << ":" << port;
  return true;
}

std::vector<uint8_t> YuanRongStoreBackend::batch_put(
    const std::vector<std::string>& keys,
    const std::vector<StoreBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), 0);
  StoreBatch batch;
  if (!client_ || keys.empty() || !prepare_batch(keys, buffers, batch)) {
    return statuses;
  }
  if (mset_(client_, batch.key_ptrs.data(), batch.ranges.data(), keys.size())) {
    std::fill(statuses.begin(), statuses.end(), 1);
  } else {
    LOG(ERROR) << "YuanRong store MSet failed: " << last_error_();
  }
  return statuses;
}

std::vector<uint8_t> YuanRongStoreBackend::batch_get(
    const std::vector<std::string>& keys,
    const std::vector<StoreBuffer>& buffers) {
  std::vector<uint8_t> statuses(keys.size(), 0);
  StoreBatch batch;
  if (!client_ || keys.empty() || !prepare_batch(keys, buffers, batch)) {
    return statuses;
  }
  if (!mget_(client_,
             batch.key_ptrs.data(),
             batch.ranges.data(),
             statuses.data(),
             keys.size())) {
    LOG(ERROR) << "YuanRong store MGet failed: " << last_error_();
    std::fill(statuses.begin(), statuses.end(), 0);
  }
  return statuses;
}

}  // namespace xllm
