include(cc_test)

if(USE_MLU)
  set(_api_json_test_env
    "LD_LIBRARY_PATH=$ENV{PYTORCH_MLU_INSTALL_PATH}/csrc/lib:$ENV{LD_LIBRARY_PATH}")
endif()

cc_test(
  NAME
    anthropic_protocol_test
  SRCS
    anthropic_protocol_test.cpp
  DEPS
    proto::xllm_proto
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    glog::glog
)

# Integration test for Anthropic service - requires running server.
# Tests are prefixed with DISABLED_ in code to skip by default.
# Run manually: ./anthropic_service_test --gtest_also_run_disabled_tests
cc_test(
  NAME
    anthropic_service_test
  SRCS
    anthropic_service_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    glog::glog
)
target_link_libraries(anthropic_service_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(anthropic_service_test brpc-static)

cc_test(
  NAME
    api_service_test
  SRCS
    chat_json_parser_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
  ENVIRONMENT
    "${_api_json_test_env}"
)

cc_test(
  NAME
    completion_json_parser_test
  SRCS
    completion_json_parser_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    anthropic_json_test
  SRCS
    anthropic_json_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
  ENVIRONMENT
    "${_api_json_test_env}"
)

cc_test(
  NAME
    anthropic_stream_utils_test
  SRCS
    anthropic_stream_utils_test.cpp
    ../../xllm/api_service/anthropic_stream_utils.cpp
  DEPS
    proto::xllm_proto
    GTest::gtest_main
)

cc_test(
  NAME
    usage_json_test
  SRCS
    usage_json_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
  ENVIRONMENT
    "${_api_json_test_env}"
)

cc_test(
  NAME
    sample_service_impl_test
  SRCS
    sample_service_impl_test.cpp
  DEPS
    api_service
    GTest::gtest_main
)

# Integration smoke tests for OpenAI-compatible endpoints. These require a
# running server and are disabled by default in code.
cc_test(
  NAME
    openai_service_test
  SRCS
    openai_service_test.cpp
  DEPS
    GTest::gtest
    GTest::gtest_main
    nlohmann_json::nlohmann_json
    glog::glog
)
target_link_libraries(openai_service_test PRIVATE brpc leveldb::leveldb OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(openai_service_test brpc-static)
