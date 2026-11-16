include(cc_test)

# Topology checks must run before automatic NPU runtime initialization.
cc_test(
  NAME
    cpu_affinity_test
  CPU_ONLY
  SRCS
    cpu_affinity_test.cpp
  DEPS
    :util
    :config
    $<$<BOOL:${USE_NPU}>:npu_startup>
    GTest::gtest_main
)

cc_test(
  NAME
    util_test
  CPU_ONLY
  SRCS
    binary_payload_test.cpp
    blocking_counter_test.cpp
    http_downloader_test.cpp
    hash_util_test.cpp
    net_test.cpp
    shared_memory_manager_test.cpp
    threadpool_test.cpp
    verbose_trace_logger_test.cpp
  DEPS
    util
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
)
target_link_libraries(util_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(util_test brpc-static)

cc_test(
  NAME
    tensor_proto_test
  CPU_ONLY
  SRCS
    tensor_proto_test.cpp
  DEPS
    :util
    GTest::gtest_main
)

add_subdirectory(partial_json_parser)
