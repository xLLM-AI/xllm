include(cc_library)

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
