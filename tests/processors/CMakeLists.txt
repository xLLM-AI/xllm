include(cc_test)

cc_test(
  NAME
    minimax_m3_vl_processor_test
  SRCS
    minimax_m3_vl_processor_test.cpp
  DEPS
    :processors
    GTest::gtest_main
)
