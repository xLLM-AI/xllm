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

add_custom_target(python_tests)
add_dependencies(all_tests python_tests)

if(USE_NPU)
  file(GLOB test_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py")
  # These modules require a CUDA device and contain no NPU-testable cases.
  list(FILTER test_sources EXCLUDE REGEX "/test_(flashinfer_attention|kda_triton_ops|triton_ops|phi3_cuda)\\.py$")
elseif(USE_CUDA)
  # Register the CUDA model path and its shared loading/backend regressions.
  # The remaining Python suite includes NPU-specific imports and entrypoints.
  set(test_sources)
  foreach(test_name IN ITEMS phi3 phi3_checkpoint phi3_cuda olmo2 olmo2_numerics validate_olmo2_cuda flashinfer_attention registry decode_cuda_graph model_executor)
    list(APPEND test_sources "${CMAKE_CURRENT_SOURCE_DIR}/test_${test_name}.py")
  endforeach()
endif()
# These distributed modules are executed directly by torchrun below.
list(FILTER test_sources EXCLUDE REGEX "/test_(all_gather_nz_bytes|bitwise_nz_bytes|matmul_nz_bytes)\\.py$")
foreach(test_source IN LISTS test_sources)
  get_filename_component(test_name "${test_source}" NAME_WE)
  string(REGEX REPLACE "^test_" "python_tests." test_target "${test_name}")
  add_custom_target(${test_target})
  add_dependencies(python_tests ${test_target})
  add_dependencies(${test_target} xllm_export)

  add_test(NAME ${test_target}
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "XLLM_EXPORT_PATH=$<TARGET_FILE:xllm_export>"
      "${Python3_EXECUTABLE}" -m pytest -q "${test_source}")
  set_tests_properties(${test_target} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS python_tests
    # Preserve serial device use within the former single pytest process.
    RESOURCE_LOCK python_tests)
endforeach()

# Distributed NZ tests are standalone torchrun entrypoints; pytest skips them.
if(USE_NPU)
  foreach(nz_test IN ITEMS all_gather_nz_bytes bitwise_nz_bytes matmul_nz_bytes)
    set(test_target python_distributed_tests.${nz_test})
    add_custom_target(${test_target})
    add_dependencies(python_tests ${test_target})
    add_dependencies(${test_target} xllm_export)
    add_test(NAME ${test_target}
      COMMAND "${CMAKE_COMMAND}" -E env
        "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
        "XLLM_EXPORT_PATH=$<TARGET_FILE:xllm_export>"
        "${Python3_EXECUTABLE}" -m torch.distributed.run
        --standalone --nproc_per_node=2
        "${CMAKE_CURRENT_SOURCE_DIR}/test_${nz_test}.py")
    set_tests_properties(${test_target} PROPERTIES
      WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
      LABELS "python_tests;distributed;npu"
      RUN_SERIAL TRUE
      TIMEOUT 300)
  endforeach()
endif()
