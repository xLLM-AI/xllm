# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_library)

# Concrete paged storage backend, not part of the core cache domain.
cc_library(
  NAME
    kv_cache_memory_regions
  HDRS
    kv_cache_memory_regions.h
  SRCS
    kv_cache_memory_regions.cpp
  DEPS
    :virtual_memory_torch
    glog::glog
    torch
)
