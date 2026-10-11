# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

add_subdirectory(virtual_memory)
add_subdirectory(torch)

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

# Shared worker address space; the generic mapping lives in :virtual_memory
# and has no transport or cache responsibilities.
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
