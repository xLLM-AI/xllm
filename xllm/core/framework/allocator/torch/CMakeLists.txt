# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

cc_library(
  NAME
    virtual_memory_torch
  HDRS
    mapped_memory_tensor_view.h
  SRCS
    mapped_memory_tensor_view.cpp
  DEPS
    :virtual_memory
    :platform
    torch
    glog::glog
)
