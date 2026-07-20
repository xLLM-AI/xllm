include(cc_binary)
include(cc_library)

cc_library(
  NAME
    prefix_cache
  HDRS
    block_hasher.h
    prefix_cache.h
    linear_state_prefix_cache.h
    prefix_cache_factory.h
  SRCS
    block_hasher.cpp
    prefix_cache.cpp
    linear_state_prefix_cache.cpp
    prefix_cache_factory.cpp
  DEPS
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_NPU}>:graph>
    :request
    :common
    glog::glog
    Boost::serialization
    SMHasherSupport
    torch
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
    :block
    benchmark::benchmark
    benchmark::benchmark_main
    :xllm_server
    SMHasherSupport
    xxHash
)

target_link_libraries(prefix_cache_benchmark PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(prefix_cache_benchmark brpc-static)
