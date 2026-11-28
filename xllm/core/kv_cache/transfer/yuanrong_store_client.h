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

#include <cstddef>
#include <cstdint>

// Only host memory ranges and opaque handles cross the SDK's libstdc++ ABI.
struct XllmYrStoreBuffer {
  void* const* addresses;
  const size_t* sizes;
  size_t count;
};

extern "C" {
void* xllm_yr_store_create(const char* host, int32_t port, int32_t timeout_ms);
void xllm_yr_store_destroy(void* client);
bool xllm_yr_store_mset(void* client,
                        const char* const* keys,
                        const XllmYrStoreBuffer* buffers,
                        size_t count);
bool xllm_yr_store_mget(void* client,
                        const char* const* keys,
                        const XllmYrStoreBuffer* buffers,
                        uint8_t* statuses,
                        size_t count);
const char* xllm_yr_store_thread_error();
}
