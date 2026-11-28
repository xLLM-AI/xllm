include(cc_library)

# Framework adapters for sequence-scoped and virtual cache resource leaves.
cc_library(
  NAME
    block
  HDRS
    embedding_block_manager.h
    vmm_kv_block_manager.h
  SRCS
    embedding_block_manager.cpp
    vmm_kv_block_manager.cpp
  DEPS
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
