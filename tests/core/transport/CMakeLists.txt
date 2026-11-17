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
# ==============================================================================

include(cc_test)

cc_test(
  NO_NPU_RUNTIME
  NAME
    shm_channel_test
  SRCS
    shm_channel_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(shm_channel_test PRIVATE nnopbase)
endif()
target_link_options(shm_channel_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NO_NPU_RUNTIME
  NAME
    comm_channel_test
  SRCS
    comm_channel_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME shared_memory_output_test
  SRCS shared_memory_output_test.cpp
  DEPS :runtime GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(shared_memory_output_test PRIVATE nnopbase)
endif()
target_link_options(shared_memory_output_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NO_NPU_RUNTIME
  NAME shared_memory_input_test
  SRCS shared_memory_input_test.cpp
  DEPS :runtime GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(shared_memory_input_test PRIVATE nnopbase)
endif()
target_link_options(shared_memory_input_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NAME transport_codec_test
  SRCS transport_codec_test.cpp
  DEPS :runtime :worker_model_codec GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(transport_codec_test PRIVATE nnopbase)
endif()
target_link_options(transport_codec_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")
