include(cc_binary)
include(cc_shared_library)

include_directories(.)

# Generate build-time version header from version.txt.
file(READ "${CMAKE_SOURCE_DIR}/version.txt" XLLM_BUILD_VERSION_RAW)
string(STRIP "${XLLM_BUILD_VERSION_RAW}" XLLM_BUILD_VERSION)
if(XLLM_BUILD_VERSION MATCHES "^v(.+)$")
  set(XLLM_BUILD_VERSION "${CMAKE_MATCH_1}")
endif()
if(XLLM_BUILD_VERSION STREQUAL "")
  set(XLLM_BUILD_VERSION "unknown")
endif()
file(MAKE_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/core/common")
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/core/common/xllm_build_info.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/core/common/xllm_build_info.h"
  @ONLY)
include_directories("${CMAKE_CURRENT_BINARY_DIR}")

# Set warning-as-error for xllm code (third-party headers are marked as SYSTEM)
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
  if(USE_MUSA)
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-defaulted-function-deleted>
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-sign-conversion>
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-inconsistent-missing-override>
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-conversion>
    )
  else()
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Werror>
      $<$<COMPILE_LANGUAGE:C>:-Werror>
    )
  endif()

  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(
      $<$<COMPILE_LANGUAGE:CXX>:-Wno-macro-redefined>
      $<$<COMPILE_LANGUAGE:C>:-Wno-macro-redefined>
    )
  endif()
endif()

option(GENERATE_SO "Enable compile to generate .so" OFF)

add_subdirectory(api_service)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(parser)
add_subdirectory(processors)
add_subdirectory(proto)
add_subdirectory(pybind)
add_subdirectory(server)

if(GENERATE_SO)
  message(STATUS "build libxllm.so and install")
  cc_shared_library(
    NAME
      xllm
    HDRS
      c_api/llm.h
      c_api/rec.h
      c_api/default.h
      c_api/types.h
      c_api/internal/helper.h
    SRCS
      c_api/internal/llm.cpp
      c_api/internal/rec.cpp
      c_api/internal/helper.cpp
    DEPS
      :config
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

else()
  message(STATUS "build xllm binary and install")
  cc_binary(
    NAME
      xllm
    SRCS
      xllm.cpp
    DEPS
      :config
      :xllm_server
      :master
      absl::strings
      Boost::serialization
      gflags::gflags
      glog::glog
      Folly::folly
      nlohmann_json::nlohmann_json
  )

endif()

# link brpc
target_link_libraries(xllm PRIVATE glog::glog brpc leveldb::leveldb protobuf::libprotobuf ${OpenCV_LIBS} ${FFMPEG_LIBRARIES})
add_dependencies(xllm brpc-static)

if(USE_NPU)
  set(COMMON_LIBS torch_npu torch_python Python::Python ascendcl atb_customize unified_dlog hccl c_sec nnopbase ms_tools_ext)
elseif(USE_MLU)
  set(COMMON_LIBS Python::Python)
endif()

if(USE_MSPTI)
  list(APPEND COMMON_LIBS mspti)
endif()
target_link_libraries(xllm PUBLIC ${COMMON_LIBS})
if(USE_MUSA)
  target_link_libraries(xllm PUBLIC atomic musa_python torch_cpu c10)
endif()

if(USE_MUSA)
  target_link_options(xllm PRIVATE
    "LINKER:--whole-archive,$<TARGET_FILE:runtime>,--no-whole-archive"
  )
endif()

# install xllm
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install all dependencies for xllm
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
        RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
        UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
        EXECUTABLES $<TARGET_FILE:xllm>)

  file(INSTALL
        DESTINATION "${CMAKE_INSTALL_PREFIX}/lib"
        FILES ${DEPENDENCIES}
        FOLLOW_SYMLINK_CHAIN)

  # This should not be possible, but error out when a dependency cannot
  # be resolved.
  list(LENGTH UNRESOLVED_DEPENDENCIES UNRESOLVED_LENGTH)
  if(${UNRESOLVED_LENGTH} GREATER 0)
      message(FATAL_ERROR "Unresolved dependencies: ${UNRESOLVED_DEPENDENCIES}")
  endif()
]])
