include(cc_test)

add_subdirectory(partial_json_parser)

function(add_detector_test TEST_NAME)
  cc_test(
    CPU_ONLY
    NAME
      ${TEST_NAME}
    SRCS
      ${TEST_NAME}.cpp
    DEPS
      :function_call
      GTest::gtest
      GTest::gtest_main
      nlohmann_json::nlohmann_json
  )
endfunction()

add_detector_test(qwen25_detector_test)
add_detector_test(qwen3_coder_detector_test)
add_detector_test(kimik2_detector_test)
add_detector_test(deepseekv3_detector_test)
add_detector_test(glm45_detector_test)
add_detector_test(glm47_detector_test)
add_detector_test(deepseekv32_detector_test)
