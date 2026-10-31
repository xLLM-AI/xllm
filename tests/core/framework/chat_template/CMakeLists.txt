include(cc_test)

cc_test (
  CPU_ONLY
  NAME
    chat_template_test
  SRCS
    chat_template_test.cpp
  DEPS
    :config
    :chat_template_factory
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    GTest::gtest_main
)

cc_test(
  CPU_ONLY
  NAME
    jinja_chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :jinja_chat_template
    GTest::gtest_main
)

cc_test (
  CPU_ONLY
  NAME
    deepseek_v32_cpp_template_test
  SRCS
    deepseek_v32_cpp_template_test.cpp
  DEPS
    :deepseek_v32_cpp_template
    GTest::gtest_main
)

cc_test (
  CPU_ONLY
  NAME
    deepseek_v4_cpp_template_test
  SRCS
    deepseek_v4_cpp_template_test.cpp
  DEPS
    :deepseek_v4_cpp_template
    GTest::gtest_main
)
