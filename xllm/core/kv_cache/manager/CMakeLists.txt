# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_library)

# Scheduler and engine consumers depend on request operations, not pool leaves.
cc_library(
  NAME
    kv_cache_manager_interface
  HDRS
    kv_cache_manager.h
  DEPS
    :batch
    :request
)

# Actual DP selection, sequence transitions and cache-tier orchestration.
cc_library(
  NAME
    kv_cache_manager
  HDRS
    kv_cache_manager_options.h
    block_manager_pool.h
    composite_block_manager.h
    hierarchy_block_manager_pool.h
  SRCS
    block_manager_pool.cpp
    composite_block_manager.cpp
    hierarchy_block_manager_pool.cpp
  DEPS
    :kv_cache_manager_interface
    :block
    :paged_cache_partition
    :recurrent_cache_partition
    :kv_cache_block
    :kv_transfer_completion
    :kv_cache_memory_interfaces
    :config
    :common
    :request
    glog::glog
    Folly::folly
    torch
)

# The construction boundary binds model layouts to the selected cache backend.
cc_library(
  NAME
    kv_cache_manager_factory
  HDRS
    kv_cache_manager_factory.h
  SRCS
    kv_cache_manager_factory.cpp
  DEPS
    :kv_cache_manager
    :kv_cache
    :static_kv_tensor_allocator
    :distributed_vmm_coordinator
)
