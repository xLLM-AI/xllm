include(cc_test)

cc_test(
  NAME
    kv_cache_test
  SRCS
    kv_cache_test.cpp
  DEPS
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(kv_cache_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
