# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

cc_library(
  NAME
    virtual_memory
  HDRS
    physical_page.h
    physical_page_pool.h
    mapped_virtual_memory_region.h
    shared_physical_page_mapping.h
  SRCS
    physical_page.cpp
    physical_page_pool.cpp
    mapped_virtual_memory_region.cpp
    shared_physical_page_mapping.cpp
  DEPS
    :platform
    torch
    glog::glog
)
