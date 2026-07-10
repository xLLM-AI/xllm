include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_test()
# CMake function to imitate Bazel's cc_test rule.
#
# Parameters:
# NAME: name of target (see Usage below)
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
    "" # options
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

  if(USE_NPU)
    set(COMMON_LIBS Python::Python torch_npu torch_python)
    target_link_libraries(${CC_TEST_NAME} PRIVATE ${COMMON_LIBS})
  endif()

  add_dependencies(all_tests ${CC_TEST_NAME})

  gtest_add_tests(
    TARGET ${CC_TEST_NAME}
    EXTRA_ARGS ${CC_TEST_ARGS}
    TEST_LIST _cc_test_${CC_TEST_NAME}_tests
  )

  if(CC_TEST_ENVIRONMENT)
    set_tests_properties(${_cc_test_${CC_TEST_NAME}_tests}
      PROPERTIES ENVIRONMENT "${CC_TEST_ENVIRONMENT}")
  endif()
  #add_test(NAME ${CC_TEST_NAME} COMMAND ${CC_TEST_NAME} ${CC_TEST_ARGS})
endfunction()
