include(cc_test)

if(USE_NPU OR USE_MLU)
cc_test(
  NAME
    mooncake_transfer_engine_test
  SRCS
    mooncake_transfer_engine_test.cpp
  DEPS
    :kv_cache_transfer
    :xllm_server
    GTest::gtest_main
)

# Resolve static link order between xtensor and xllm_server for this test target.
target_link_libraries(mooncake_transfer_engine_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
endif()
