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

cc_test(
  NAME
    vision_token_scanner_test
  CPU_ONLY
  SRCS
    vision_token_scanner_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    prompt_processor_test
  CPU_ONLY
  SRCS
    prompt_processor_test.cpp
  DEPS
    :processors
    GTest::gtest_main
)

# Python preprocessing and the model-directory boundary use CPU tensors only.
cc_test(
  NAME python_image_processor_test
  CPU_ONLY
  SRCS python_image_processor_test.cpp
  DEPS :processors Python::Python GTest::gtest_main
)
