include(cc_test)

cc_test(
  NAME
    prefix_test
  SRCS
    prefix_cache_test.cpp
  DEPS
    :config
    :kv_cache
    :prefix_cache
    :block
    absl::random_random
    Boost::serialization
    GTest::gtest_main
)
target_link_libraries(prefix_test PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto Folly::folly :xllm_server)
add_dependencies(prefix_test brpc-static)
