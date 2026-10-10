# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

cc_library(
  NAME
    model_weight_store
  HDRS
    model_weight_store.h
    weight_allocation.h
    weight_page_coordinator.h
    weight_transfer_segments.h
  SRCS
    model_weight_store.cpp
    weight_allocation.cpp
    weight_page_coordinator.cpp
  DEPS
    :common
    :virtual_memory
    glog::glog
    torch
)

# Weight transfer is a runtime adapter using the shared Mooncake engine.
set(XLLM_WEIGHT_TRANSFER_SRCS)
if(USE_NPU)
  list(APPEND XLLM_WEIGHT_TRANSFER_SRCS mooncake_weight_transfer.cpp)
endif()

cc_library(
  NAME
    mooncake_weight_transfer
  HDRS
    mooncake_weight_transfer.h
  SRCS
    ${XLLM_WEIGHT_TRANSFER_SRCS}
  DEPS
    :model_memory_manager
    :mooncake_transfer_engine
    torch
)
