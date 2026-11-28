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

#include "core/kv_cache/transfer/yuanrong_store_client.h"

#include <datasystem/kv_client.h>

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <exception>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

namespace {

// Establishing a connection is a one-off start-up cost, while a store read is
// on the inference critical path. The two must not share one timeout: a cache
// miss has to fail fast instead of holding a prefetch thread. SDK defaults are
// 9 s for connect and 0 (unbounded) for requests.
constexpr int32_t kConnectTimeoutMs = 10000;
constexpr int32_t kDefaultRequestTimeoutMs = 500;
constexpr int32_t kMaxRequestTimeoutMs = 5000;

thread_local std::string g_last_error;

bool set_error(std::string error) {
  g_last_error = std::move(error);
  return false;
}

// Validate the scatter/gather layout before accessing SDK-owned storage.
uint64_t payload_size(const XllmYrStoreBuffer& buffer) {
  if (buffer.count == 0 || buffer.addresses == nullptr ||
      buffer.sizes == nullptr) {
    throw std::invalid_argument("YuanRong store buffer has no ranges");
  }
  uint64_t total = 0;
  for (size_t index = 0; index < buffer.count; ++index) {
    const uint64_t size = buffer.sizes[index];
    if (size != 0 && buffer.addresses[index] == nullptr) {
      throw std::invalid_argument("YuanRong store buffer has a null range");
    }
    if (size > INT64_MAX - total) {
      throw std::length_error("YuanRong store buffer size overflow");
    }
    total += size;
  }
  return total;
}

std::vector<std::string> collect_keys(const char* const* keys, size_t count) {
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t index = 0; index < count; ++index) {
    if (keys[index] == nullptr) {
      throw std::invalid_argument("YuanRong store batch has a null key");
    }
    result.emplace_back(keys[index]);
  }
  return result;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void* xllm_yr_store_create(
    const char* host,
    int32_t port,
    int32_t request_timeout_ms) {
  try {
    datasystem::ConnectOptions options;
    options.host = host;
    options.port = port;
    options.connectTimeoutMs = kConnectTimeoutMs;
    options.requestTimeoutMs =
        request_timeout_ms > 0
            ? std::min(request_timeout_ms, kMaxRequestTimeoutMs)
            : kDefaultRequestTimeoutMs;
    // Bound-worker MSet(buffers) reports an error for any failed publication.
    options.enableLocalCache = true;
    auto client = std::make_unique<datasystem::KVClient>(options);
    const auto status = client->Init();
    if (status.IsError()) {
      g_last_error = status.ToString();
      return nullptr;
    }
    return client.release();
  } catch (const std::exception& error) {
    g_last_error = error.what();
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) void xllm_yr_store_destroy(
    void* opaque) {
  std::unique_ptr<datasystem::KVClient> client(
      static_cast<datasystem::KVClient*>(opaque));
  if (client != nullptr) {
    client->ShutDown();
  }
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mset(
    void* opaque,
    const char* const* keys,
    const XllmYrStoreBuffer* buffers,
    size_t count) {
  auto* client = static_cast<datasystem::KVClient*>(opaque);
  if (client == nullptr || keys == nullptr || buffers == nullptr ||
      count == 0) {
    return set_error("Invalid YuanRong store Put batch");
  }

  try {
    const auto key_vec = collect_keys(keys, count);
    std::vector<uint64_t> sizes;
    sizes.reserve(count);
    for (size_t index = 0; index < count; ++index) {
      const uint64_t size = payload_size(buffers[index]);
      if (size == 0) {
        return set_error("YuanRong store cannot publish an empty KV block");
      }
      sizes.emplace_back(size);
    }

    datasystem::SetParam param;
    param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    param.existence = datasystem::ExistenceOpt::NX;
    std::vector<std::shared_ptr<datasystem::Buffer>> objects;
    auto status = client->MCreate(key_vec, sizes, param, objects);
    if (status.IsError()) {
      return set_error(status.ToString());
    }
    if (objects.size() != count) {
      return set_error("YuanRong MCreate returned an unexpected buffer count");
    }
    for (size_t index = 0; index < count; ++index) {
      const auto& object = objects[index];
      if (object == nullptr) {
        return set_error("YuanRong MCreate returned a null buffer");
      }
      const int64_t size = object->GetSize();
      if (size == 0) {
        continue;  // NX placeholder for an existing key.
      }
      if (size != static_cast<int64_t>(sizes[index])) {
        return set_error("YuanRong MCreate returned an unexpected buffer size");
      }
      auto* dst = static_cast<uint8_t*>(object->MutableData());
      if (dst == nullptr) {
        return set_error("YuanRong MCreate returned no writable data");
      }
      const auto& source = buffers[index];
      for (size_t range = 0; range < source.count; ++range) {
        const uint64_t bytes = source.sizes[range];
        if (bytes != 0) {
          std::memcpy(dst, source.addresses[range], bytes);
          dst += bytes;
        }
      }
    }
    status = client->MSet(objects);
    if (status.IsError()) {
      return set_error(status.ToString());
    }
    return true;
  } catch (const std::exception& error) {
    return set_error(error.what());
  }
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_store_mget(
    void* opaque,
    const char* const* keys,
    const XllmYrStoreBuffer* buffers,
    uint8_t* statuses,
    size_t count) {
  auto* client = static_cast<datasystem::KVClient*>(opaque);
  if (client == nullptr || keys == nullptr || buffers == nullptr ||
      statuses == nullptr || count == 0) {
    return set_error("Invalid YuanRong store Get batch");
  }
  std::memset(statuses, 0, count);

  try {
    const auto key_vec = collect_keys(keys, count);
    std::vector<datasystem::Optional<datasystem::Buffer>> objects;
    const auto status = client->Get(key_vec, objects);
    if (status.GetCode() == datasystem::K_NOT_FOUND) {
      return true;  // All keys missed; statuses are already zero.
    }
    if (status.IsError()) {
      return set_error(status.ToString());
    }
    if (objects.size() > count) {
      return set_error("YuanRong Get returned extra buffers");
    }
    for (size_t index = 0; index < objects.size(); ++index) {
      auto& object = objects[index];
      if (!object || object->GetSize() <= 0) {
        continue;
      }
      const auto& target = buffers[index];
      if (payload_size(target) != static_cast<uint64_t>(object->GetSize())) {
        continue;
      }
      const auto copied = object->CopyDataWithRLatch([&] {
        const auto* src = static_cast<const uint8_t*>(object->ImmutableData());
        if (src == nullptr) {
          return datasystem::Status(datasystem::K_RUNTIME_ERROR,
                                    "YuanRong Get returned no readable data");
        }
        for (size_t range = 0; range < target.count; ++range) {
          const uint64_t bytes = target.sizes[range];
          if (bytes != 0) {
            std::memcpy(target.addresses[range], src, bytes);
            src += bytes;
          }
        }
        return datasystem::Status::OK();
      });
      if (copied.IsError()) {
        return set_error(copied.ToString());
      }
      statuses[index] = 1;
    }
    return true;
  } catch (const std::exception& error) {
    return set_error(error.what());
  }
}

extern "C" __attribute__((visibility("default"))) const char*
xllm_yr_store_thread_error() {
  return g_last_error.c_str();
}
