include(cc_test)

cc_test(
  NAME similarity_residual_cache_test
  CPU_ONLY
  SRCS similarity_residual_cache_test.cpp h3_qualified_cache_rows_test.cpp
  DEPS :dit_cache torch GTest::gtest_main
)
