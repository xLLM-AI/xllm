# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

add_subdirectory(virtual_memory)
add_subdirectory(kv_cache)
add_subdirectory(weight)

cc_library(
  NAME
    worker_page_budget
  HDRS
    worker_page_budget.h
  SRCS
    worker_page_budget.cpp
  DEPS
    glog::glog
)

# Shared runtime mapping and transfer registration; the generic mapping lives
# in :virtual_memory and has no transport or cache responsibilities.
cc_library(
  NAME
    global_memory_region
  HDRS
    global_memory_region.h
  SRCS
    global_memory_region.cpp
  DEPS
    :virtual_memory
    glog::glog
)

# Model allocation policy depends on cache domain ports and generic VMM.
# RPC adapters remain distributed_runtime sources, with a single runtime
# facade target to preserve the existing initialization/destruction lifetime.
cc_library(
  NAME
    model_memory_manager
  HDRS
    model_memory_options.h
    model_page_allocator.h
    model_memory_manager.h
    kv_cache/paged_kv_cache_tensor_allocator.h
    kv_cache/paged_kv_cache_transfer_memory_provider.h
    ../../distributed_runtime/model_memory_cluster.h
    ../../distributed_runtime/model_memory_dist_client.h
    ../../distributed_runtime/model_memory_dist_service.h
    ../../distributed_runtime/model_memory_dist_server.h
  SRCS
    model_page_allocator.cpp
    model_memory_manager.cpp
    kv_cache/paged_kv_cache_tensor_allocator.cpp
    kv_cache/paged_kv_cache_transfer_memory_provider.cpp
    ../../distributed_runtime/model_memory_cluster.cpp
    ../../distributed_runtime/model_memory_dist_client.cpp
    ../../distributed_runtime/model_memory_dist_service.cpp
    ../../distributed_runtime/model_memory_dist_server.cpp
  DEPS
    torch
    Folly::folly
    :common
    :config
    glog::glog
    proto::xllm_proto
    :collective_service
    :platform
    :xllm_server
    :global_memory_region
    :kv_cache_page_state
    :kv_cache_memory_interfaces
    :worker_page_budget
    :kv_cache_memory_regions
    :model_weight_store
)
