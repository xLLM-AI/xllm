if(NOT BUILD_TESTING)
  return()
endif()

if(USE_NPU OR USE_CUDA)
  add_subdirectory(python)
endif()

if(USE_NPU)
  get_filename_component(_xllm_build_import_root
    "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

  # The multi-process HCCL probe requires exclusive device use.
  add_test(
    NAME python_hccl_decode_graph_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${_xllm_build_import_root}:${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q --import-mode=importlib
      "${CMAKE_CURRENT_SOURCE_DIR}/npu/test_hccl_decode_graph.py"
  )
  set_tests_properties(
    python_hccl_decode_graph_test
    PROPERTIES
      WORKING_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}"
      RUN_SERIAL TRUE
  )
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)

add_subdirectory(processors)
