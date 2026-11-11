include(cc_test)

cc_test(
  NAME
    multimodal_processor_test
  SRCS
    multimodal_processor_test.cpp
  DEPS
    :processors
    GTest::gtest_main
)
