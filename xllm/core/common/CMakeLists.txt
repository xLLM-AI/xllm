include(cc_library)

cc_library(
  NAME extension_registry
  HDRS extension_registry.h
  SRCS extension_registry.cpp
)

cc_library(
  NAME native_library
  HDRS native_library.h
  SRCS native_library.cpp
  DEPS glog::glog ${CMAKE_DL_LIBS}
)
target_include_directories(native_library PUBLIC
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/xllm>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/xllm>)
target_include_directories(native_library SYSTEM PUBLIC ${TORCH_INCLUDE_DIRS})

cc_library(
  NAME 
    common
  HDRS
    constants.h
    etcd_client.h
    global_flags.h
    instance_name.h
    macros.h
    message.h
    metrics.h
    $<$<BOOL:${USE_NPU}>:mspti_helper.h>
    options.h
    rate_limiter.h
    types.h
    device_monitor.h
    version_singleton.h
    flash_comm1_context.h
  SRCS
    etcd_client.cpp
    metrics.cpp
    $<$<BOOL:${USE_NPU}>:mspti_helper.cpp>
    options.cpp
    rate_limiter.cpp
    device_monitor.cpp
    flash_comm1_context.cpp
  DEPS
    :native_library
    :extension_registry
    :config
    util
    parallel_state
    absl::random_random
    absl::strings
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    $<$<BOOL:${USE_CUDA}>:CUDA::cudart>
    $<$<BOOL:${USE_MSPTI}>:mspti>
    $<$<BOOL:${USE_NPU}>:ms_tools_ext>
    Boost::serialization
    cpprest
    etcd-cpp-api
    $<$<BOOL:${USE_MLU}>:torch_mlu>
    $<$<BOOL:${USE_MUSA}>:torch_musa>
)

target_link_libraries(common PRIVATE OpenSSL::SSL OpenSSL::Crypto protobuf::libprotobuf)
add_dependencies(common brpc-static)
