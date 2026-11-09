include(cc_binary)

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

if(NOT DEFINED NATIVE_LIBRARY_BUILD_REVISION OR
   NATIVE_LIBRARY_BUILD_REVISION STREQUAL "")
  execute_process(
    COMMAND git rev-parse HEAD
    WORKING_DIRECTORY "${PROJECT_SOURCE_DIR}"
    OUTPUT_VARIABLE NATIVE_LIBRARY_BUILD_REVISION
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET)
endif()
if(NOT NATIVE_LIBRARY_BUILD_REVISION MATCHES "^[0-9a-f]+$")
  set(NATIVE_LIBRARY_BUILD_REVISION "unknown")
endif()
configure_file(
  "${CMAKE_CURRENT_SOURCE_DIR}/core/common/native_library_build_info.h.in"
  "${CMAKE_CURRENT_BINARY_DIR}/core/common/native_library_build_info.h"
  @ONLY)

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

add_subdirectory(api_service)
add_subdirectory(compiler)
add_subdirectory(core)
add_subdirectory(function_call)
add_subdirectory(models)
add_subdirectory(parser)
add_subdirectory(processors)
add_subdirectory(proto)
add_subdirectory(pybind)
add_subdirectory(server)

message(STATUS "build xllm binary and install")
cc_binary(
  NAME
    xllm
  SRCS
    xllm.cpp
  DEPS
    :distributed_vmm_coordinator
    :worker_memory_resources
    $<$<BOOL:${USE_NPU}>:npu_startup>
    :config
    :api_service
    :xllm_server
    :master
    :model_loader
    :kv_cache_store_config_validation
    :util
    absl::strings
    Boost::serialization
    gflags::gflags
    glog::glog
    Folly::folly
    nlohmann_json::nlohmann_json
)

xllm_export_native_host(xllm)
xllm_export_native_host(xllm_export)

if(USE_NPU)
  include(npu_runtime)
  foreach(_native_host xllm xllm_export)
    xllm_configure_npu_runtime(${_native_host})
  endforeach()
endif()

# link brpc
target_link_libraries(xllm PRIVATE glog::glog brpc leveldb::leveldb protobuf::libprotobuf ${OpenCV_LIBS} ${FFMPEG_LIBRARIES})
add_dependencies(xllm brpc-static)

if(USE_NPU)
  set(COMMON_LIBS
      torch_npu
      torch_python
      npu_profiler
      Python::Python
      ascendcl
      atb_customize
      unified_dlog
      hccl
      c_sec
      nnopbase
      ms_tools_ext)
  # Some Torch wheels embed TensorPipe in libtorch_cpu instead of shipping
  # a standalone library. Link it explicitly only when this Torch provides it.
  find_library(TENSORPIPE_LIBRARY NAMES tensorpipe
    HINTS "${TORCH_INSTALL_PREFIX}/lib"
          "$ENV{PYTORCH_INSTALL_PATH}/lib"
          "$ENV{PYTORCH_NPU_INSTALL_PATH}/lib"
    NO_DEFAULT_PATH)
  if(TENSORPIPE_LIBRARY)
    list(APPEND COMMON_LIBS "${TENSORPIPE_LIBRARY}")
  endif()
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

if (USE_XLITE)
  xllm_link_xlite(xllm)
endif()
# install xllm
install(TARGETS xllm RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Install all dependencies for xllm
install(CODE [[
  file(GET_RUNTIME_DEPENDENCIES
        RESOLVED_DEPENDENCIES_VAR DEPENDENCIES
        UNRESOLVED_DEPENDENCIES_VAR UNRESOLVED_DEPENDENCIES
        EXECUTABLES
        $<TARGET_FILE:xllm>)

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
