include(cc_test)

cc_test(
  NAME
    concurrent_block_manager_test
  SRCS
    concurrent_block_manager_test.cpp
  DEPS
    :block
    :config
    :prefix_cache
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME
      block_test
    SRCS
      block_manager_test.cpp
      single_block_manager_test.cpp
      composite_block_manager_test.cpp
    DEPS
      :xllm_server
      :block
      :config
      :kv_cache
      :prefix_cache
      :linear_state_prefix_cache
      absl::random_random
      Boost::serialization
      GTest::gtest_main
  )

  target_link_libraries(block_test PRIVATE Folly::folly OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf c_sec)
  target_link_libraries(block_test PRIVATE
    "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
  add_dependencies(block_test brpc-static)
endif()
