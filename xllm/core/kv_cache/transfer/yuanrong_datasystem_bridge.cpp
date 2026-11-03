/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <datasystem/hetero_client.h>
#include <dlfcn.h>

#include <cstddef>
#include <cstdint>
#include <exception>
#include <memory>
#include <string>
#include <vector>

#include "acl/acl_rt.h"

extern "C" {

struct XllmYrBridgeBlob {
  void* data;
  uint64_t size;
};

struct XllmYrBridgeBlobList {
  int32_t device_idx;
  const XllmYrBridgeBlob* blobs;
  size_t blob_count;
};

struct XllmYrAclForwardVTable {
  void* aclInit;
  void* aclFinalize;
  void* aclrtSetDevice;
  void* aclrtGetDevice;
  void* aclrtGetDeviceCount;
  void* aclrtQueryDeviceStatus;
  void* aclrtResetDevice;
  void* aclrtMalloc;
  void* aclrtFree;
  void* aclrtMallocHost;
  void* aclrtFreeHost;
  void* aclrtMemcpy;
  void* aclrtMemcpyAsync;
  void* aclrtMemcpyBatch;
  void* aclrtCreateStream;
  void* aclrtDestroyStream;
  void* aclrtDestroyStreamForce;
  void* aclrtSynchronizeStream;
  void* aclrtSynchronizeStreamWithTimeout;
  void* aclrtCreateEvent;
  void* aclrtRecordEvent;
  void* aclrtSynchronizeEvent;
  void* aclrtSynchronizeEventWithTimeout;
  void* aclrtDestroyEvent;
  void* aclrtQueryEventStatus;
  void* aclrtLaunchCallback;
  void* aclrtProcessReport;
  void* aclrtSubscribeReport;
  void* aclrtUnSubscribeReport;
};

}  // extern "C"

namespace {

struct BridgeClient {
  std::unique_ptr<datasystem::HeteroClient> client;
  std::string last_error;
};

thread_local std::string g_last_error;
XllmYrAclForwardVTable g_acl_vtable{};
void* g_acl_plugin_handle = nullptr;

bool require_acl_symbol(void* symbol, const char* name) {
  if (symbol != nullptr) {
    return true;
  }
  g_last_error = std::string("ACL forward symbol is not initialized: ") + name;
  return false;
}

std::vector<std::string> copy_keys(const char* const* keys, size_t count) {
  std::vector<std::string> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    result.emplace_back(keys[i] == nullptr ? "" : keys[i]);
  }
  return result;
}

std::vector<datasystem::DeviceBlobList> copy_blob_lists(
    const XllmYrBridgeBlobList* lists,
    size_t count) {
  std::vector<datasystem::DeviceBlobList> result;
  result.reserve(count);
  for (size_t i = 0; i < count; ++i) {
    datasystem::DeviceBlobList list;
    list.deviceIdx = lists[i].device_idx;
    list.blobs.reserve(lists[i].blob_count);
    for (size_t j = 0; j < lists[i].blob_count; ++j) {
      const auto& blob = lists[i].blobs[j];
      list.blobs.emplace_back(datasystem::Blob{blob.data, blob.size});
    }
    result.emplace_back(std::move(list));
  }
  return result;
}

bool set_error(BridgeClient* client, const std::string& error) {
  if (client != nullptr) {
    client->last_error = error;
  }
  g_last_error = error;
  return false;
}

}  // namespace

extern "C" __attribute__((visibility("default"))) void* xllm_yr_ds_create(
    const char* host,
    int32_t port,
    int32_t timeout_ms,
    int32_t device_id,
    int32_t enable_remote_h2d) {
  try {
    auto bridge = std::make_unique<BridgeClient>();
    datasystem::ConnectOptions options;
    options.host = host == nullptr || host[0] == '\0' ? "127.0.0.1" : host;
    options.port = port;
    options.connectTimeoutMs = timeout_ms;
    options.requestTimeoutMs = timeout_ms;
    options.deviceId = std::to_string(device_id);
    options.enableRemoteH2D = enable_remote_h2d != 0;
    bridge->client = std::make_unique<datasystem::HeteroClient>(options);
    const auto status = bridge->client->Init();
    if (status.IsError()) {
      g_last_error = status.ToString();
      return nullptr;
    }
    return bridge.release();
  } catch (const std::exception& e) {
    g_last_error = e.what();
    return nullptr;
  }
}

extern "C" __attribute__((visibility("default"))) void xllm_yr_ds_destroy(
    void* opaque) {
  auto* bridge = reinterpret_cast<BridgeClient*>(opaque);
  if (bridge == nullptr) {
    return;
  }
  if (bridge->client) {
    bridge->client->ShutDown();
  }
  delete bridge;
}

extern "C" __attribute__((visibility("default"))) bool
xllm_yr_ds_set_acl_vtable(const XllmYrAclForwardVTable* vtable,
                          const char* acl_plugin_path) {
  if (vtable == nullptr) {
    g_last_error = "ACL forward vtable is null";
    return false;
  }
  g_acl_vtable = *vtable;
  if (acl_plugin_path != nullptr && acl_plugin_path[0] != '\0') {
    dlerror();
    g_acl_plugin_handle = dlopen(acl_plugin_path, RTLD_NOW | RTLD_LOCAL);
    if (g_acl_plugin_handle == nullptr) {
      g_last_error = std::string("Failed to load ACL forward plugin ") +
                     acl_plugin_path + ": " + dlerror();
      return false;
    }
    using SetVTableFn = bool (*)(const XllmYrAclForwardVTable*);
    dlerror();
    auto* symbol = dlsym(g_acl_plugin_handle, "XllmYrAclPluginSetVTable");
    const char* error = dlerror();
    if (error != nullptr || symbol == nullptr) {
      g_last_error = std::string("Failed to load XllmYrAclPluginSetVTable: ") +
                     (error == nullptr ? "" : error);
      return false;
    }
    if (!reinterpret_cast<SetVTableFn>(symbol)(vtable)) {
      g_last_error = "ACL forward plugin rejected vtable";
      return false;
    }
  }
  g_last_error.clear();
  return true;
}

extern "C" __attribute__((visibility("default"))) void* xllm_yr_ds_acl_malloc(
    int32_t device_id,
    uint64_t bytes) {
  if (bytes == 0) {
    g_last_error = "aclrtMalloc requires non-zero bytes";
    return nullptr;
  }
  if (!require_acl_symbol(g_acl_vtable.aclrtSetDevice, "aclrtSetDevice") ||
      !require_acl_symbol(g_acl_vtable.aclrtMalloc, "aclrtMalloc")) {
    return nullptr;
  }
  using SetDeviceFn = int (*)(int32_t);
  using MallocFn = int (*)(void**, size_t, aclrtMemMallocPolicy);
  const auto set_ret =
      reinterpret_cast<SetDeviceFn>(g_acl_vtable.aclrtSetDevice)(device_id);
  if (set_ret != ACL_SUCCESS) {
    g_last_error =
        "aclrtSetDevice failed, device=" + std::to_string(device_id) +
        ", ret=" + std::to_string(set_ret);
    return nullptr;
  }
  void* buffer = nullptr;
  const auto malloc_ret = reinterpret_cast<MallocFn>(g_acl_vtable.aclrtMalloc)(
      &buffer, bytes, ACL_MEM_MALLOC_HUGE_ONLY);
  if (malloc_ret != ACL_SUCCESS) {
    g_last_error = "aclrtMalloc failed, bytes=" + std::to_string(bytes) +
                   ", ret=" + std::to_string(malloc_ret);
    return nullptr;
  }
  g_last_error.clear();
  return buffer;
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_ds_acl_free(
    int32_t device_id,
    void* buffer) {
  if (buffer == nullptr) {
    return true;
  }
  if (!require_acl_symbol(g_acl_vtable.aclrtSetDevice, "aclrtSetDevice") ||
      !require_acl_symbol(g_acl_vtable.aclrtFree, "aclrtFree")) {
    return false;
  }
  using SetDeviceFn = int (*)(int32_t);
  using FreeFn = int (*)(void*);
  const auto set_ret =
      reinterpret_cast<SetDeviceFn>(g_acl_vtable.aclrtSetDevice)(device_id);
  if (set_ret != ACL_SUCCESS) {
    g_last_error = "aclrtSetDevice failed before free, device=" +
                   std::to_string(device_id) +
                   ", ret=" + std::to_string(set_ret);
    return false;
  }
  const auto free_ret =
      reinterpret_cast<FreeFn>(g_acl_vtable.aclrtFree)(buffer);
  if (free_ret != ACL_SUCCESS) {
    g_last_error = "aclrtFree failed, ret=" + std::to_string(free_ret);
    return false;
  }
  g_last_error.clear();
  return true;
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_ds_mset(
    void* opaque,
    const char* const* keys,
    const XllmYrBridgeBlobList* lists,
    size_t count) {
  auto* bridge = reinterpret_cast<BridgeClient*>(opaque);
  if (bridge == nullptr || bridge->client == nullptr) {
    return set_error(bridge, "DataSystem bridge client is null");
  }
  try {
    auto key_vec = copy_keys(keys, count);
    auto list_vec = copy_blob_lists(lists, count);
    datasystem::SetParam set_param;
    set_param.writeMode = datasystem::WriteMode::NONE_L2_CACHE_EVICT;
    const auto status = bridge->client->MSetD2H(key_vec, list_vec, set_param);
    if (status.IsError()) {
      return set_error(bridge, status.ToString());
    }
    bridge->last_error.clear();
    return true;
  } catch (const std::exception& e) {
    return set_error(bridge, e.what());
  }
}

extern "C" __attribute__((visibility("default"))) bool xllm_yr_ds_mget(
    void* opaque,
    const char* const* keys,
    const XllmYrBridgeBlobList* lists,
    size_t count,
    int32_t timeout_ms) {
  auto* bridge = reinterpret_cast<BridgeClient*>(opaque);
  if (bridge == nullptr || bridge->client == nullptr) {
    return set_error(bridge, "DataSystem bridge client is null");
  }
  try {
    auto key_vec = copy_keys(keys, count);
    auto list_vec = copy_blob_lists(lists, count);
    std::vector<std::string> failed_keys;
    const auto status =
        bridge->client->MGetH2D(key_vec, list_vec, failed_keys, timeout_ms);
    if (status.IsError() || !failed_keys.empty()) {
      return set_error(
          bridge,
          status.ToString() + ", failed=" + std::to_string(failed_keys.size()));
    }
    bridge->last_error.clear();
    return true;
  } catch (const std::exception& e) {
    return set_error(bridge, e.what());
  }
}

extern "C" __attribute__((visibility("default"))) const char*
xllm_yr_ds_last_error(void* opaque) {
  auto* bridge = reinterpret_cast<BridgeClient*>(opaque);
  if (bridge != nullptr && !bridge->last_error.empty()) {
    return bridge->last_error.c_str();
  }
  return g_last_error.c_str();
}
