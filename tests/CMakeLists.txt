if(NOT BUILD_TESTING)
  return()
endif()

if(USE_NPU)
  add_test(
    NAME python_executor_test
    COMMAND
      "${CMAKE_COMMAND}" -E env
      "PYTHONPATH=${PROJECT_SOURCE_DIR}:$ENV{PYTHONPATH}"
      "${Python3_EXECUTABLE}" -m pytest -q "${CMAKE_CURRENT_SOURCE_DIR}/python"
  )
  set_tests_properties(
    python_executor_test
    PROPERTIES WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
  )
endif()

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
