include(cc_test)

# Run the existing cache contracts alongside MTP regressions on every backend.
cc_test(
  NAME
    mtp_prefix_cache_test
  SRCS
    mtp_prefix_cache_test.cpp
    composite_block_manager_test.cpp
    hierarchy_block_manager_pool_test.cpp
  DEPS
    :block
    :config
    :request
    :prefix_cache
    GTest::gtest_main
)
target_link_libraries(mtp_prefix_cache_test
                      PRIVATE
                      Python::Python
                      $<$<BOOL:${USE_MLU}>:torch_mlu>)
target_link_options(mtp_prefix_cache_test
                    PRIVATE
                    $<$<BOOL:${USE_MLU}>:-Wl,--no-as-needed>)
if(USE_NPU)
  # Let CMake resolve static library cycles without a partial RESCAN group.
  target_link_libraries(mtp_prefix_cache_test PRIVATE
    Folly::folly OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf c_sec
    :xtensor :xllm_server)
  add_dependencies(mtp_prefix_cache_test brpc-static)
endif()

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
      embedding_block_manager_test.cpp
    DEPS
      :xllm_server
      :block
      :config
      :kv_cache
      :prefix_cache
      absl::random_random
      Boost::serialization
      GTest::gtest_main
  )

  target_link_libraries(block_test PRIVATE Folly::folly OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf c_sec)
  target_link_libraries(block_test PRIVATE
    "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
  add_dependencies(block_test brpc-static)
endif()
