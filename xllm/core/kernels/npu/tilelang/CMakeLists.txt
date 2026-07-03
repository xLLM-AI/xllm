include(cc_library)
include(CMakeParseArguments)

if(NOT USE_NPU)
  return()
endif()

if(NOT DEFINED ENV{NPU_HOME_PATH})
  message(FATAL_ERROR "NPU_HOME_PATH is not set")
endif()

function(tilelang_import_kernel_manifest)
  set(options "")
  set(oneValueArgs PREFIX MANIFEST_PATH)
  cmake_parse_arguments(TL "${options}" "${oneValueArgs}" "" ${ARGN})

  if(NOT TL_PREFIX OR NOT TL_MANIFEST_PATH)
    message(FATAL_ERROR "tilelang_import_kernel_manifest requires PREFIX and MANIFEST_PATH")
  endif()

  if(NOT EXISTS "${TL_MANIFEST_PATH}")
    message(FATAL_ERROR
      "Missing TileLang manifest: ${TL_MANIFEST_PATH}\n"
      "Run the Python TileLang compile entry before invoking CMake build.")
  endif()

  file(READ "${TL_MANIFEST_PATH}" TL_MANIFEST_JSON)
  string(JSON _schema_version GET "${TL_MANIFEST_JSON}" schema_version)
  if(NOT _schema_version EQUAL 2)
    message(FATAL_ERROR
      "Unsupported TileLang manifest schema_version=${_schema_version}: ${TL_MANIFEST_PATH}")
  endif()

  string(JSON _variants_inc GET "${TL_MANIFEST_JSON}" variants_inc)
  if(NOT EXISTS "${_variants_inc}")
    message(FATAL_ERROR "TileLang variants.inc does not exist: ${_variants_inc}")
  endif()

  string(JSON _registry_inc ERROR_VARIABLE _registry_inc_error GET "${TL_MANIFEST_JSON}" registry_inc)
  if(_registry_inc_error)
    set(_registry_inc "${_variants_inc}")
  endif()
  if(NOT EXISTS "${_registry_inc}")
    message(FATAL_ERROR "TileLang registry.inc does not exist: ${_registry_inc}")
  endif()

  set(_kernel_objects "")
  string(JSON _variants_len LENGTH "${TL_MANIFEST_JSON}" variants)
  if(_variants_len LESS 1)
    message(FATAL_ERROR "TileLang manifest contains no variants: ${TL_MANIFEST_PATH}")
  endif()

  math(EXPR _variants_last "${_variants_len} - 1")
  foreach(_idx RANGE 0 ${_variants_last})
    string(JSON _generated_source GET "${TL_MANIFEST_JSON}" variants ${_idx} generated_source)
    string(JSON _compiled_binary GET "${TL_MANIFEST_JSON}" variants ${_idx} compiled_binary)

    if(NOT EXISTS "${_generated_source}")
      message(FATAL_ERROR "TileLang generated source does not exist: ${_generated_source}")
    endif()
    if(NOT EXISTS "${_compiled_binary}")
      message(FATAL_ERROR "TileLang compiled binary does not exist: ${_compiled_binary}")
    endif()

    set_source_files_properties("${_compiled_binary}" PROPERTIES
      GENERATED TRUE
      EXTERNAL_OBJECT TRUE
    )
    list(APPEND _kernel_objects "${_compiled_binary}")
  endforeach()

  set(${TL_PREFIX}_KERNEL_OBJECTS "${_kernel_objects}" PARENT_SCOPE)
  set(${TL_PREFIX}_VARIANTS_INC "${_variants_inc}" PARENT_SCOPE)
  set(${TL_PREFIX}_REGISTRY_INC "${_registry_inc}" PARENT_SCOPE)
endfunction()

set(TILELANG_GENERATED_ROOT "${CMAKE_BINARY_DIR}/xllm/compiler/tilelang")

function(tilelang_register_runtime_kernel)
  set(options "")
  set(oneValueArgs NAME)
  set(multiValueArgs WRAPPER_SRCS)
  cmake_parse_arguments(TL_RUNTIME "${options}" "${oneValueArgs}" "${multiValueArgs}" ${ARGN})

  if(NOT TL_RUNTIME_NAME OR NOT TL_RUNTIME_WRAPPER_SRCS)
    message(FATAL_ERROR
      "tilelang_register_runtime_kernel requires NAME and WRAPPER_SRCS")
  endif()
  if(NOT DEFINED TILELANG_GENERATED_ROOT)
    message(FATAL_ERROR "TILELANG_GENERATED_ROOT must be set before registering TileLang kernels")
  endif()

  set(_manifest_path
    "${TILELANG_GENERATED_ROOT}/targets/ascend/${TL_RUNTIME_NAME}/manifest.json")
  set(_kernel_prefix "tilelang_${TL_RUNTIME_NAME}")
  tilelang_import_kernel_manifest(
    PREFIX "${_kernel_prefix}"
    MANIFEST_PATH "${_manifest_path}"
  )

  set(_kernel_srcs ${TILELANG_KERNEL_SRCS})
  list(APPEND _kernel_srcs
    ${TL_RUNTIME_WRAPPER_SRCS}
    ${${_kernel_prefix}_KERNEL_OBJECTS}
  )
  set(TILELANG_KERNEL_SRCS "${_kernel_srcs}" PARENT_SCOPE)

  string(TOUPPER "${TL_RUNTIME_NAME}" _kernel_name_upper)
  set(_kernel_private_definitions ${TILELANG_KERNEL_PRIVATE_DEFINITIONS})
  list(APPEND _kernel_private_definitions
    XLLM_TL_${_kernel_name_upper}_REGISTRY_INC=\"${${_kernel_prefix}_REGISTRY_INC}\"
  )
  set(TILELANG_KERNEL_PRIVATE_DEFINITIONS
    "${_kernel_private_definitions}" PARENT_SCOPE)
endfunction()

# Add more TileLang kernels here:
# tilelang_register_runtime_kernel(
#   NAME foo
#   WRAPPER_SRCS foo_wrapper.cpp
# )
set(TILELANG_KERNEL_SRCS)
set(TILELANG_KERNEL_PRIVATE_DEFINITIONS)

tilelang_register_runtime_kernel(
  NAME rope
  WRAPPER_SRCS rope_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME fused_gdn_gating
  WRAPPER_SRCS fused_gdn_gating_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME split_qkv_rmsnorm_mrope
  WRAPPER_SRCS split_qkv_rmsnorm_mrope_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME chunk_gated_delta_rule_fwd_h
  WRAPPER_SRCS chunk_gated_delta_rule_fwd_h_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME causal_conv1d_decode
  WRAPPER_SRCS causal_conv1d_decode_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME causal_conv1d
  WRAPPER_SRCS causal_conv1d_wrapper.cpp
)

tilelang_register_runtime_kernel(
  NAME fused_sigmoid_gating_delta_rule
  WRAPPER_SRCS fused_sigmoid_gating_delta_rule_wrapper.cpp
)

cc_library(
  NAME
    tilelang_kernels
  HDRS
    dispatch_registry.h
    tilelang_ops_api.h
  SRCS
    ${TILELANG_KERNEL_SRCS}
  DEPS
    torch
    torch_npu
)

target_compile_definitions(tilelang_kernels PRIVATE
  ${TILELANG_KERNEL_PRIVATE_DEFINITIONS}
)

target_link_libraries(tilelang_kernels
  PUBLIC
    "$ENV{NPU_HOME_PATH}/lib64/libascendc_runtime.a"
    "$ENV{NPU_HOME_PATH}/lib64/libtiling_api.a"
    "$ENV{NPU_HOME_PATH}/lib64/libascendcl.so"
    "$ENV{NPU_HOME_PATH}/lib64/libruntime.so"
    "$ENV{NPU_HOME_PATH}/lib64/libplatform.so"
    "$ENV{NPU_HOME_PATH}/lib64/libc_sec.so"
    "$ENV{NPU_HOME_PATH}/lib64/libascend_dump.so"
    "$ENV{NPU_HOME_PATH}/lib64/libprofapi.so"
    "$ENV{NPU_HOME_PATH}/lib64/libmmpa.so"
    m
    dl
)
