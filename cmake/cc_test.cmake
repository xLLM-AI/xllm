include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_test()
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
# NO_NPU_RUNTIME: skip automatic NPU initialization
# CPU_ONLY: skip NPU initialization and label all cases as CPU tests.
# SRCS: List of source files for the binary
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# LINKOPTS: List of link options
# ARGS: Command line arguments to test case
#
# Usage:
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
#
# cc_test(
#   NAME
#     awesome_test
#   SRCS
#     "awesome_test.cc"
#   DEPS
#     :awesome
#     GTest::gmock
# )
#
function(cc_test)
  if(NOT BUILD_TESTING)
    return()
  endif()

  cmake_parse_arguments(
    CC_TEST # prefix
    "NO_NPU_RUNTIME;CPU_ONLY" # options
    "NAME;ENVIRONMENT" # one value args
    "SRCS;COPTS;LINKOPTS;DEPS;INCLUDES;ARGS;DATA" # multi value args
    ${ARGN}
  )

  # place test data in build directory
  if(CC_TEST_DATA)
    foreach(data ${CC_TEST_DATA})
      configure_file(${data} ${CMAKE_CURRENT_BINARY_DIR}/${data} COPYONLY)
    endforeach()
  endif()

  set(_CC_TEST_SRCS "")
  set(_CC_TEST_INCLUDE_DIRS ${CC_TEST_INCLUDES})
  list(APPEND _CC_TEST_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})

  # xllm test sources live under tests and often include private headers
  # from the mirrored production source directory.
  if(DEFINED XLLM_TESTS_DIR)
    list(APPEND _CC_TEST_INCLUDE_DIRS
      ${PROJECT_SOURCE_DIR}/xllm
      ${XLLM_TESTS_DIR}
      ${XLLM_TESTS_DIR}/core
    )
  endif()

  foreach(src IN LISTS CC_TEST_SRCS)
    if(IS_ABSOLUTE "${src}")
      list(APPEND _CC_TEST_SRCS "${src}")
      get_filename_component(src_dir "${src}" DIRECTORY)
      list(APPEND _CC_TEST_INCLUDE_DIRS "${src_dir}")
      continue()
    endif()

    set(src_path "${CMAKE_CURRENT_SOURCE_DIR}/${src}")
    if(EXISTS "${src_path}")
      list(APPEND _CC_TEST_SRCS "${src}")
      get_filename_component(src_dir "${src_path}" DIRECTORY)
      list(APPEND _CC_TEST_INCLUDE_DIRS "${src_dir}")
      if(DEFINED XLLM_TESTS_DIR)
        file(RELATIVE_PATH xllm_test_src_dir "${XLLM_TESTS_DIR}" "${src_dir}")
        if(NOT xllm_test_src_dir MATCHES "^\\.\\.")
          list(APPEND _CC_TEST_INCLUDE_DIRS
            "${PROJECT_SOURCE_DIR}/xllm/${xllm_test_src_dir}"
          )
        endif()
      endif()
      continue()
    endif()

    list(APPEND _CC_TEST_SRCS "${src}")
  endforeach()

  list(REMOVE_DUPLICATES _CC_TEST_INCLUDE_DIRS)

  add_executable(${CC_TEST_NAME})
  target_sources(${CC_TEST_NAME} PRIVATE ${_CC_TEST_SRCS})
  target_include_directories(${CC_TEST_NAME}
    PUBLIC 
      "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>" 
      ${_CC_TEST_INCLUDE_DIRS}
  )

  target_compile_options(${CC_TEST_NAME}
    PRIVATE ${CC_TEST_COPTS}
  )

  target_link_libraries(${CC_TEST_NAME}
    PUBLIC ${CC_TEST_DEPS}
    PRIVATE ${CC_TEST_LINKOPTS}
  )

  if(USE_NPU AND NOT CC_TEST_NO_NPU_RUNTIME AND NOT CC_TEST_CPU_ONLY)
    if(NOT TARGET xllm_npu_test_runtime)
      # Reuse one object while retaining the global gtest Environment in each
      # executable; a static archive could discard its registration initializer.
      add_library(xllm_npu_test_runtime OBJECT
        "${PROJECT_SOURCE_DIR}/tests/npu_test_environment.cpp"
      )
      target_include_directories(xllm_npu_test_runtime PRIVATE
        "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
      )
      target_link_libraries(xllm_npu_test_runtime PUBLIC
        GTest::gtest ascendcl Python::Python torch_npu torch_python
      )
    endif()
    target_link_libraries(${CC_TEST_NAME} PRIVATE xllm_npu_test_runtime)
  endif()

  add_dependencies(all_tests ${CC_TEST_NAME})

  # The GoogleTest source scanner does not match multiline test declarations.
  # Normalize whitespace only in discovery copies; compile the original sources
  # and watch them for changes so adding a test still regenerates CTest entries.
  set(_cc_test_discovery_sources "")
  foreach(src IN LISTS _CC_TEST_SRCS)
    get_filename_component(_original_source "${src}" ABSOLUTE
      BASE_DIR "${CMAKE_CURRENT_SOURCE_DIR}")
    file(READ "${_original_source}" _discovery_contents)
    string(REGEX REPLACE "[\r\n\t]" " "
      _discovery_contents "${_discovery_contents}")
    string(SHA256 _source_hash "${_original_source}")
    set(_discovery_source
      "${CMAKE_CURRENT_BINARY_DIR}/gtest-discovery/${_source_hash}.cpp")
    file(CONFIGURE OUTPUT "${_discovery_source}"
      CONTENT "@_discovery_contents@" @ONLY)
    list(APPEND _cc_test_discovery_sources "${_discovery_source}")
    set_property(DIRECTORY APPEND PROPERTY
      CMAKE_CONFIGURE_DEPENDS "${_original_source}")
  endforeach()

  gtest_add_tests(
    TARGET ${CC_TEST_NAME}
    SOURCES ${_cc_test_discovery_sources}
    SKIP_DEPENDENCY
    EXTRA_ARGS ${CC_TEST_ARGS}
    TEST_LIST _cc_test_${CC_TEST_NAME}_tests
  )

  if(CC_TEST_CPU_ONLY)
    set_property(TEST ${_cc_test_${CC_TEST_NAME}_tests}
      APPEND PROPERTY LABELS cpu)
  endif()

  if(CC_TEST_ENVIRONMENT)
    set_tests_properties(${_cc_test_${CC_TEST_NAME}_tests}
      PROPERTIES ENVIRONMENT "${CC_TEST_ENVIRONMENT}")
  endif()
  #add_test(NAME ${CC_TEST_NAME} COMMAND ${CC_TEST_NAME} ${CC_TEST_ARGS})
endfunction()
