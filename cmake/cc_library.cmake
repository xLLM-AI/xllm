include(CMakeParseArguments)

# inspired by https://github.com/abseil/abseil-cpp
# cc_library()
# CMake function to imitate Bazel's cc_library rule.
#
# Parameters:
# NAME: name of target
# HDRS: List of public header files for the library
# SRCS: List of source files for the library
# DEPS: List of other libraries to be linked in to the binary targets
# COPTS: List of private compile options
# DEFINES: List of public defines
# LINKOPTS: List of link options
#
# cc_library(
#   NAME
#     awesome
#   HDRS
#     "a.h"
#   SRCS
#     "a.cc"
# )
# cc_library(
#   NAME
#     fantastic_lib
#   SRCS
#     "b.cc"
#   DEPS
#     :awesome
# )
#
function(cc_library)
  cmake_parse_arguments(
    CC_LIB # prefix
    "TESTONLY" # options
    "NAME" # one value args
    "HDRS;SRCS;COPTS;DEFINES;LINKOPTS;DEPS;INCLUDES" # multi value args
    ${ARGN}
  )

  if(CC_LIB_TESTONLY AND (NOT BUILD_TESTING))
    return()
  endif()

  # Check if this is a header only library
  set(_CC_SRCS "${CC_LIB_SRCS}")
  foreach(src_file IN LISTS _CC_SRCS)
    if(${src_file} MATCHES ".*\\.(h|hpp|hxx|inc|inl)$")
      list(REMOVE_ITEM _CC_SRCS "${src_file}")
    endif()
  endforeach()

  if(_CC_SRCS STREQUAL "")
    set(CC_LIB_IS_INTERFACE 1)
  else()
    set(CC_LIB_IS_INTERFACE 0)
  endif()

  if(NOT CC_LIB_IS_INTERFACE)
    add_library(${CC_LIB_NAME} STATIC)
    target_sources(${CC_LIB_NAME} 
      PRIVATE ${CC_LIB_SRCS} ${CC_LIB_HDRS})
    target_link_libraries(${CC_LIB_NAME}
      PUBLIC ${CC_LIB_DEPS}
      PRIVATE ${CC_LIB_LINKOPTS}
    )
    target_include_directories(${CC_LIB_NAME}
      PUBLIC 
        "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
        ${CC_LIB_INCLUDES}
    )
    target_compile_options(${CC_LIB_NAME} PRIVATE ${CC_LIB_COPTS})
    target_compile_definitions(${CC_LIB_NAME} PUBLIC ${CC_LIB_DEFINES})
  else()
    # Generating header only library
    add_library(${CC_LIB_NAME} INTERFACE)
    target_include_directories(${CC_LIB_NAME}
      INTERFACE 
        "$<BUILD_INTERFACE:${COMMON_INCLUDE_DIRS}>"
        ${CC_LIB_INCLUDES}
    )

    target_link_libraries(${CC_LIB_NAME}
      INTERFACE ${CC_LIB_DEPS} ${CC_LIB_LINKOPTS}
    )
    target_compile_definitions(${CC_LIB_NAME} INTERFACE ${CC_LIB_DEFINES})
  endif()

  # add alias for the library target
  add_library(:${CC_LIB_NAME} ALIAS ${CC_LIB_NAME})
endfunction()
