# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

set(MOONCAKE_TRANSFER_ENGINE_SRCS)
if(USE_NPU OR USE_MLU OR USE_DCU)
  list(APPEND MOONCAKE_TRANSFER_ENGINE_SRCS mooncake_transfer_engine.cpp)
endif()

cc_library(
  NAME
    mooncake_transfer_engine
  HDRS
    mooncake_transfer_engine.h
  SRCS
    ${MOONCAKE_TRANSFER_ENGINE_SRCS}
  DEPS
    :cache_layout
    :reshard_planner
    :common
    :platform
    :util
    glog::glog
    torch
    mooncake_store
    proto::xllm_proto
    brpc
)
