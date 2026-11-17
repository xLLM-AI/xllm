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
project(t5_direct LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "CANN merge script requires a nonempty build type" FORCE)
endif()
if(NOT DEFINED ASCEND_CANN_PACKAGE_PATH)
  set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}")
endif()
set(ASCEND_COMPUTE_UNIT "ascend910_93" CACHE STRING "Ascend A3 compile family")
set(SOC_VERSION "Ascend910_9382" CACHE STRING "Concrete target SoC required by CANN kernel launch")
set(RUN_MODE "npu" CACHE STRING "NPU kernel launch mode")
foreach(required ASCEND_CANN_PACKAGE_PATH SOC_VERSION CMAKE_BUILD_TYPE RUN_MODE)
  if(NOT DEFINED ${required} OR "${${required}}" STREQUAL "")
    message(FATAL_ERROR "Required CANN direct-launch setting is empty: ${required}")
  endif()
endforeach()
set(T5_CONFIG "${ASCEND_CANN_PACKAGE_PATH}|${SOC_VERSION}|${CMAKE_BUILD_TYPE}|${RUN_MODE}")
set(T5_CONFIG_FILE "${CMAKE_BINARY_DIR}/t5_config.txt")
if(EXISTS "${T5_CONFIG_FILE}")
  file(READ "${T5_CONFIG_FILE}" T5_PREVIOUS_CONFIG)
  if(NOT "${T5_PREVIOUS_CONFIG}" STREQUAL "${T5_CONFIG}")
    message(FATAL_ERROR "Build configuration changed. Preserve this directory and use a fresh build directory.")
  endif()
endif()
file(WRITE "${T5_CONFIG_FILE}" "${T5_CONFIG}")
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
ascendc_library(t5_direct SHARED t5_direct.cpp)

# CANN forwards this interface property to its separate device build projects.
get_filename_component(T5_PROJECT_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
ascendc_include_directories(t5_direct PRIVATE "${T5_PROJECT_ROOT}")
get_target_property(T5_DEVICE_INCLUDES t5_direct_interface INCLUDES)
if(NOT "${T5_PROJECT_ROOT}" IN_LIST T5_DEVICE_INCLUDES)
  message(FATAL_ERROR "CANN direct precompile/device include path was not registered")
endif()
