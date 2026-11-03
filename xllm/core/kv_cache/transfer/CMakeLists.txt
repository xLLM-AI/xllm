include(cc_library)

cc_library(
  NAME
    cache_layout
  HDRS
    cache_layout.h
  SRCS
    cache_layout.cpp
  DEPS
    :common
    glog::glog
    proto::xllm_proto
)

cc_library(
  NAME
    reshard_planner
  HDRS
    reshard_planner.h
  SRCS
    reshard_planner.cpp
  DEPS
    :cache_layout
    :common
    :transfer_types
    glog::glog
)

cc_library(
  NAME
    push_route
  HDRS
    push_route.h
  SRCS
    push_route.cpp
)

cc_library(
  NAME
    kv_content_hash
  HDRS
    kv_content_hash.h
  SRCS
    kv_content_hash.cpp
  DEPS
    glog::glog
    torch
    xxHash
)

cc_library(
  NAME
    pd_topology_guard
  HDRS
    pd_topology_guard.h
  SRCS
    pd_topology_guard.cpp
  DEPS
    :cache_layout
    :common
    glog::glog
)

cc_library(
  NAME
    kv_transfer_completion
  HDRS
    kv_transfer_completion.h
  SRCS
    kv_transfer_completion.cpp
  DEPS
    glog::glog
    Folly::folly
)

cc_library(
  NAME
    kv_cache_transfer
  HDRS
    kv_cache_transfer.h
    kv_cache_transfer_coordinator_base.h
    virtual_kv_cache_transfer_memory_provider.h
    kv_cache_store.h
    mooncake_store_backend.h
    host_transfer/layout.h
    host_transfer/transfer.h
    host_transfer/basic_transfer.h
    host_transfer/compact_transfer.h
    prefetch_result.h
    kv_transport_codec.h
    hierarchy_kv_cache_transfer.h
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer.h>
    mooncake_kv_cache_transfer_engine.h
    $<$<BOOL:${USE_YUANRONG_DATASYSTEM}>:yuanrong_kv_cache_transfer.h>
  SRCS
    kv_cache_transfer.cpp
    kv_transport_codec.cpp
    kv_cache_store.cpp
    mooncake_store_backend.cpp
    host_transfer/layout.cpp
    host_transfer/transfer.cpp
    host_transfer/transfer_utils.cpp
    host_transfer/basic_transfer.cpp
    host_transfer/compact_transfer.cpp
    host_transfer/compact_load_executor.cpp
    host_transfer/compact_offload_executor.cpp
    hierarchy_kv_cache_transfer.cpp
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer.cpp>
    $<$<OR:$<BOOL:${USE_NPU}>,$<BOOL:${USE_MLU}>,$<BOOL:${USE_DCU}>>:mooncake_kv_cache_transfer_engine.cpp>
    $<$<BOOL:${USE_YUANRONG_DATASYSTEM}>:yuanrong_kv_cache_transfer.cpp>
  DEPS
    :cache_layout
    :common
    :kv_cache
    :mooncake_transfer_engine
    :push_route
    :reshard_planner
    :util
    :platform
    $<$<BOOL:${USE_NPU}>:graph>
    glog::glog
    $<$<BOOL:${USE_NPU}>:c_sec>
    torch
    $<$<BOOL:${USE_NPU}>:torch_npu>
    mooncake_store
    proto::xllm_proto
    $<$<BOOL:${USE_NPU}>:platform_npu>
    $<$<BOOL:${USE_DCU}>:hip::host>
    $<$<BOOL:${USE_MLU}>:mlu_kernels>
    $<$<BOOL:${USE_YUANRONG_DATASYSTEM}>:kv_content_hash>
    ${CMAKE_DL_LIBS}
)

# The DataSystem SDK is never linked into the main binary: the adapter talks to
# it through a pure C ABI bridge, and only this shim links libdatasystem.
if(USE_YUANRONG_DATASYSTEM)
  if(NOT YUANRONG_DATASYSTEM_INCLUDE_DIR OR NOT YUANRONG_DATASYSTEM_LIBRARY)
    message(FATAL_ERROR
      "USE_YUANRONG_DATASYSTEM requires the DataSystem SDK. Set "
      "-DYUANRONG_DATASYSTEM_ROOT=/path/to/datasystem.")
  endif()

  file(WRITE "${CMAKE_BINARY_DIR}/xllm_yuanrong_bridge_symbols.map"
       "{\n  global:\n    xllm_yr_ds_*;\n  local:\n    *;\n};\n")
  add_library(xllm_yuanrong_datasystem_bridge SHARED
    yuanrong_datasystem_bridge.cpp
  )
  target_include_directories(xllm_yuanrong_datasystem_bridge
    PRIVATE
      ${COMMON_INCLUDE_DIRS}
      "${YUANRONG_DATASYSTEM_INCLUDE_DIR}"
  )
  target_link_libraries(xllm_yuanrong_datasystem_bridge
    PRIVATE
      "${YUANRONG_DATASYSTEM_LIBRARY}"
      ${CMAKE_DL_LIBS}
  )
  set_target_properties(xllm_yuanrong_datasystem_bridge PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/xllm/core/server"
    BUILD_RPATH "$ORIGIN;${YUANRONG_DATASYSTEM_LIB_DIR}"
    INSTALL_RPATH "$ORIGIN;${YUANRONG_DATASYSTEM_LIB_DIR}"
    CXX_VISIBILITY_PRESET hidden
  )
endif()
