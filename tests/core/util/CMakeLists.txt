include(cc_test)

cc_test(
  NAME
    util_test
  SRCS
    blocking_counter_test.cpp
    device_name_utils_test.cpp
    http_downloader_test.cpp
    suffix_decoding_cache_test.cpp
    threadpool_test.cpp
    vocab_validation_test.cpp
  DEPS
    util
    :platform
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
)
target_link_libraries(util_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto)
add_dependencies(util_test brpc-static)
