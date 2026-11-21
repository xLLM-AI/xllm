# Copyright 2026 The xLLM Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

cmake_minimum_required(VERSION 3.26)

execute_process(COMMAND "${READELF}" --wide --dynamic "${STORE_FILE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE dynamic ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect YuanRong store dependencies: ${error}")
endif()
if(dynamic MATCHES "\\(TEXTREL\\)|\\(FLAGS\\)[^\n]*TEXTREL")
  message(FATAL_ERROR "YuanRong store contains text relocations; SDK archives must use PIC.")
endif()

# A private Host KV SDK may import system runtimes, but no separate DS,
# third-party RPC/allocator libraries, or device runtimes.
string(REGEX MATCHALL "\\(NEEDED\\)[^\n]*\\[[^]]+\\]" needed "${dynamic}")
foreach(entry IN LISTS needed)
  string(REGEX REPLACE ".*\\[([^]]+)\\].*" "\\1" library "${entry}")
  if(NOT library MATCHES "^lib(c|m|pthread|dl|rt|resolv|util|anl)\\.so\\.[0-9]+$"
      AND NOT library MATCHES "^lib(stdc\\+\\+|gcc_s|atomic)\\.so\\.[0-9]+$"
      AND NOT library MATCHES "^ld-linux[^/]*\\.so\\.[0-9]+$")
    message(FATAL_ERROR
      "YuanRong store has an unexpected dynamic dependency: ${library}. "
      "The DS static PIC SDK must supply its private dependency closure.")
  endif()
endforeach()

execute_process(COMMAND "${NM}" --dynamic --defined-only --format=posix "${STORE_FILE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE defined ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect YuanRong store exports: ${error}")
endif()
set(required "${STORE_EXPORTS}")
string(REPLACE "\n" ";" lines "${defined}")
foreach(line IN LISTS lines)
  if(line STREQUAL "")
    continue()
  endif()
  string(REGEX MATCH "^[^ \t]+" symbol "${line}")
  if(NOT symbol IN_LIST required)
    message(FATAL_ERROR "YuanRong store exports a private symbol: ${symbol}")
  endif()
  list(REMOVE_ITEM required "${symbol}")
endforeach()
if(required)
  message(FATAL_ERROR "YuanRong store is missing C ABI exports: ${required}")
endif()

execute_process(COMMAND "${NM}" --dynamic --undefined-only --demangle "${STORE_FILE}"
  RESULT_VARIABLE result OUTPUT_VARIABLE undefined ERROR_VARIABLE error)
if(NOT result EQUAL 0)
  message(FATAL_ERROR "Cannot inspect YuanRong store imports: ${error}")
endif()
if(undefined MATCHES "(datasystem|google|gflags|brpc|butil|bthread|bvar|absl|grpc|grpc_core|leveldb|re2|ds_spdlog|tbb)::|FLAGS_|[ \t](bthread_|grpc_|gpr_|upb_|_upb_|utf8_range_|__TBB_)")
  message(FATAL_ERROR
    "YuanRong store still imports private SDK symbols:\n${undefined}")
endif()
message(STATUS "YuanRong store passed dependency and symbol checks: ${STORE_FILE}")
