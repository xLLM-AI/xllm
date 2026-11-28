# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

include(cc_test)

cc_test(
  NAME
    kv_cache_event_journal_test
  SRCS
    kv_cache_event_journal_test.cpp
  DEPS
    :kv_cache_events
    GTest::gtest_main
)
target_link_options(kv_cache_event_journal_test PRIVATE
  "-Wl,--wrap=zmq_msg_recv")
if(ZMQ_INCLUDE_DIR AND ZMQ_LIBRARY)
  target_include_directories(kv_cache_event_journal_test SYSTEM PRIVATE
    ${ZMQ_INCLUDE_DIR})
  target_link_libraries(kv_cache_event_journal_test PRIVATE ${ZMQ_LIBRARY})
else()
  find_package(ZeroMQ CONFIG REQUIRED)
  if(TARGET libzmq-static)
    target_link_libraries(kv_cache_event_journal_test PRIVATE libzmq-static)
  else()
    target_link_libraries(kv_cache_event_journal_test PRIVATE libzmq)
  endif()
endif()

include(cc_binary)
cc_binary(
  NAME
    kv_cache_event_fixture
  SRCS
    kv_cache_event_fixture.cpp
  DEPS
    :block
    :kv_cache_manager
    :kv_cache_events
    :xllm_server
)
target_include_directories(kv_cache_event_fixture PRIVATE
  ${PROJECT_SOURCE_DIR}/xllm)
target_link_libraries(kv_cache_event_fixture PRIVATE
  xllm_server
  brpc OpenSSL::SSL OpenSSL::Crypto Folly::folly c_sec)
add_dependencies(kv_cache_event_fixture brpc-static)
