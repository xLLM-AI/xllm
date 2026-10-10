include(cc_library)

# linear_state_restore depends on both :kv_cache and :model, so it lives in a
# separate target to avoid pulling :model into the base :kv_cache library
# (which would create a cycle, since :model already depends on :kv_cache).
cc_library(
  NAME
    linear_state_restore
  HDRS
    linear_state_restore.h
  SRCS
    linear_state_restore.cpp
  DEPS
    :common
    :kv_cache
    :model
    :platform
    :util
    glog::glog
    torch
    $<$<BOOL:${USE_NPU}>:ascendcl>
    $<$<BOOL:${USE_NPU}>:torch_npu>
)
