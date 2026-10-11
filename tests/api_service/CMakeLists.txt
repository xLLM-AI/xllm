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
    chat_request_decoder_test
  SRCS
    chat_request_decoder_test.cpp
  DEPS
    api_service
    proto::xllm_proto
    GTest::gtest_main
  ENVIRONMENT
    "${_api_json_test_env}"
)

cc_test(
  NAME
    json_fast_path_test
  SRCS
    json_fast_path_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
)

cc_test(
  NAME
    request_id_test
  SRCS
    request_id_test.cpp
  DEPS
    api_service
    GTest::gtest_main
)

cc_test(
  NAME
    control_service_impl_test
  NO_NPU_RUNTIME
  SRCS
    control_service_impl_test.cpp
  DEPS
    api_service
    GTest::gtest_main
)

cc_test(
  NAME
    mm_service_utils_test
  SRCS
    mm_service_utils_test.cpp
  DEPS
    proto::xllm_proto
    brpc
    GTest::gtest_main
)

cc_test(
  NAME
    mm_embedding_roundtrip_test
  SRCS
    mm_embedding_roundtrip_test.cpp
  DEPS
    api_service
    GTest::gtest_main
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

cc_test(
  NAME
    models_service_impl_test
  SRCS
    models_service_impl_test.cpp
  DEPS
    api_service
    GTest::gtest_main
    nlohmann_json::nlohmann_json
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

option(XLLM_ENABLE_SDK_TESTS
       "Enable official SDK compatibility tests in CTest"
       OFF)

foreach(_sdk_client openai_chat_completion_sdk_client.py anthropic_messages_sdk_client.py)
  configure_file("${_sdk_client}" "${CMAKE_CURRENT_BINARY_DIR}/${_sdk_client}"
    @ONLY
    FILE_PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                     GROUP_READ GROUP_EXECUTE WORLD_READ WORLD_EXECUTE)
endforeach()

cc_test(
  NAME chat_completion_protocol_test
  NO_NPU_RUNTIME
  SRCS chat_completion_protocol_test.cpp http_protocol_test_fixture.cpp
  DEPS api_service GTest::gtest_main nlohmann_json::nlohmann_json
)
target_compile_definitions(chat_completion_protocol_test PRIVATE
  XLLM_CHAT_COMPLETION_SDK_CLIENT="${CMAKE_CURRENT_BINARY_DIR}/openai_chat_completion_sdk_client.py")

cc_test(
  NAME text_completion_protocol_test
  NO_NPU_RUNTIME
  SRCS text_completion_protocol_test.cpp
  DEPS api_service GTest::gtest_main nlohmann_json::nlohmann_json
)

cc_test(
  NAME embeddings_protocol_test
  NO_NPU_RUNTIME
  SRCS embeddings_protocol_test.cpp
  DEPS api_service GTest::gtest_main nlohmann_json::nlohmann_json
)

cc_test(
  NAME anthropic_messages_protocol_test
  NO_NPU_RUNTIME
  SRCS anthropic_messages_protocol_test.cpp http_protocol_test_fixture.cpp
  DEPS api_service GTest::gtest_main nlohmann_json::nlohmann_json
  ENVIRONMENT
    "${_api_json_test_env}"
)
target_compile_definitions(anthropic_messages_protocol_test PRIVATE
  XLLM_MESSAGES_SDK_CLIENT="${CMAKE_CURRENT_BINARY_DIR}/anthropic_messages_sdk_client.py")

set(_sdk_tests
  ChatCompletionProtocolTest.OfficialSdkCompatibility
  AnthropicMessagesProtocolTest.OfficialSdkCompatibility)
set_tests_properties(${_sdk_tests} PROPERTIES LABELS "sdk")
if(XLLM_ENABLE_SDK_TESTS)
  execute_process(
    COMMAND "${Python3_EXECUTABLE}" -c
      "from openai import OpenAI; from anthropic import Anthropic, DefaultHttpxClient"
    RESULT_VARIABLE _sdk_import_result
    ERROR_VARIABLE _sdk_import_error)
  if(NOT _sdk_import_result STREQUAL "0")
    message(FATAL_ERROR
      "XLLM_ENABLE_SDK_TESTS requires the official OpenAI and Anthropic SDKs "
      "in ${Python3_EXECUTABLE}: ${_sdk_import_error}")
  endif()
  set_tests_properties(${_sdk_tests} PROPERTIES DISABLED FALSE)
endif()

cc_test(
  NAME
    anthropic_request_utils_test
  SRCS
    anthropic_request_utils_test.cpp
  DEPS
    api_service
    GTest::gtest_main
  ENVIRONMENT
    "${_api_json_test_env}"
)

cc_test(
  NAME sampling_protocol_test
  NO_NPU_RUNTIME
  SRCS sampling_protocol_test.cpp
  DEPS api_service GTest::gtest_main nlohmann_json::nlohmann_json
)
