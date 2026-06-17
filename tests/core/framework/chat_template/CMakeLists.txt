include(cc_test)

cc_test (
  NAME
    chat_template_test
  SRCS
    chat_template_test.cpp
  DEPS
    :config
    :chat_template
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    GTest::gtest_main
)

cc_test(
  NAME
    jinja_chat_template_test
  SRCS
    jinja_chat_template_test.cpp
  DEPS
    :chat_template
    GTest::gtest_main
)

cc_test (
  NAME
    deepseek_v32_cpp_template_test
  SRCS
    deepseek_v32_cpp_template_test.cpp
  DEPS
    :deepseek_v32_cpp_template
    GTest::gtest_main
)

cc_test (
  NAME
    deepseek_v4_cpp_template_test
  SRCS
    deepseek_v4_cpp_template_test.cpp
  DEPS
    :deepseek_v4_cpp_template
    GTest::gtest_main
)
