# Copyright 2026 The xLLM Authors.
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

cmake_minimum_required(VERSION 3.16)
project(t5_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
file(GLOB_RECURSE API_HEADERS "${T5_OPP_ROOT}/aclnn_t5_split_qkv_append_cache.h")
file(GLOB_RECURSE API_LIBS "${T5_OPP_ROOT}/libcust_opapi.so")
list(LENGTH API_HEADERS HEADER_COUNT)
list(LENGTH API_LIBS LIB_COUNT)
if(NOT HEADER_COUNT EQUAL 1 OR NOT LIB_COUNT EQUAL 1)
  message(FATAL_ERROR "Expected one private API header and library")
endif()
list(GET API_HEADERS 0 API_HEADER)
list(GET API_LIBS 0 API_LIB)
get_filename_component(API_INCLUDE "${API_HEADER}" DIRECTORY)
add_executable(t5_example aclnn_example.cpp)
target_include_directories(t5_example PRIVATE "${API_INCLUDE}" "${CANN_ROOT}/include")
target_link_directories(t5_example PRIVATE "${CANN_ROOT}/lib64")
target_link_libraries(t5_example PRIVATE "${API_LIB}" ascendcl nnopbase)
