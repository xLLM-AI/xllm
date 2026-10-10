# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

add_custom_target(python_kernel_tests)
add_dependencies(all_tests python_kernel_tests)

file(GLOB test_sources CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/test_*.py"
  "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_*.py")
foreach(test_source IN LISTS test_sources)
  get_filename_component(test_name "${test_source}" NAME_WE)
  string(REGEX REPLACE "^test_" "python_kernel_tests." test_target "${test_name}")
  add_custom_target(${test_target})
  add_dependencies(python_kernel_tests ${test_target})
  add_dependencies(${test_target} xllm_export)

  set(kernel_implementation "npu.xllm_native.rms_norm")
  if(test_name STREQUAL "test_kernel_xlite_rms_norm_npu")
    set(kernel_implementation "npu.xlite.rms_norm")
  endif()
  add_test(NAME ${test_target}
    COMMAND "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}/xllm-kernel/python:${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "XLLM_EXPORT_PATH=$<TARGET_FILE:xllm_export>"
      "XLLM_KERNEL_RMS_NORM_IMPL=${kernel_implementation}"
      "${Python3_EXECUTABLE}" -m pytest -q "${test_source}")
  set_tests_properties(${test_target} PROPERTIES
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    LABELS python_kernel_tests
    RESOURCE_LOCK python_tests)
endforeach()
