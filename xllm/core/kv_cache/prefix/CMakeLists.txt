include(cc_binary)
include(cc_library)

cc_library(
  NAME
    block_hasher
  HDRS
    block_hasher.h
  SRCS
    block_hasher.cpp
  DEPS
    :config
    :multimodal
    glog::glog
    torch
    xxHash
)

cc_library(
  NAME
    prefix_cache
  HDRS
    prefix_cache.h
    linear_state_prefix_cache.h
    prefix_cache_factory.h
  SRCS
    prefix_cache.cpp
    linear_state_prefix_cache.cpp
    prefix_cache_factory.cpp
  DEPS
    :block_hasher
    :config
    :kv_cache_block_types
    :multimodal
    :common
    glog::glog
    absl::strings
    absl::time
    Boost::serialization
    SMHasherSupport
    torch
    xxHash
)




cc_binary(
  NAME
    prefix_cache_benchmark
  SRCS
    prefix_cache_benchmark.cpp
  DEPS
    torch_python
    :kv_cache
    :prefix_cache
    :kv_cache_block
    benchmark::benchmark
    benchmark::benchmark_main
    :xllm_server
    SMHasherSupport
    xxHash
)

target_link_libraries(prefix_cache_benchmark PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(prefix_cache_benchmark brpc-static)
