include(cc_test)

cc_test(
  NAME
    caching_multimodal_processor_test
  SRCS
    caching_multimodal_processor_test.cpp
  DEPS
    :processors
    :util
    :config
    GTest::gtest_main
    torch
)

cc_test(
  NAME
    caching_multimodal_processor_request_test
  SRCS
    caching_multimodal_processor_request_test.cpp
  DEPS
    :processors
    :util
    :config
    GTest::gtest_main
    torch
    ${OpenCV_LIBS}
)
