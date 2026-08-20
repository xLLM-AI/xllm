include(cc_test)

cc_test(
  NAME
    host_page_aligned_region_test
  SRCS
    host_page_aligned_region_test.cpp
  DEPS
    :kv_cache
    :platform
    GTest::gtest_main
    cndrv
    torch
)

cc_test(
  NAME
    mlu_batch_memcpy_test
  SRCS
    mlu_batch_memcpy_test.cpp
  DEPS
    :kv_cache
    :platform
    GTest::gtest_main
    cndrv
    torch
)
target_link_options(mlu_batch_memcpy_test PRIVATE
  -Wl,--wrap=cnMemcpyBatchAsync
  -Wl,--wrap=cnQueueSync)

cc_test(
  NAME
    mlu_layer_synchronizer_test
  SRCS
    mlu_layer_synchronizer_test.cpp
  DEPS
    :platform
    GTest::gtest_main
    torch
)

# Resolve the existing static dependency cycle between platform, config, and
# util for this focused platform test.
target_link_libraries(mlu_layer_synchronizer_test PRIVATE
  "$<LINK_GROUP:RESCAN,platform,config,util>")
