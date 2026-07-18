include(cc_library)
include(utils)

set(DCU_HEADER_FILES
  dcu_ops_api.h
)

set(CUDA_KERNEL_DIR ${CMAKE_SOURCE_DIR}/xllm/core/kernels/cuda)

file(GLOB DCU_CUDA_KERNEL_SOURCES CONFIGURE_DEPENDS
  ${CUDA_KERNEL_DIR}/*.cu
  ${CUDA_KERNEL_DIR}/moe/*.cu
)

file(GLOB DCU_CUDA_KERNEL_HEADERS CONFIGURE_DEPENDS
  ${CUDA_KERNEL_DIR}/*.cuh
  ${CUDA_KERNEL_DIR}/moe/*.cuh
)

list(FILTER DCU_CUDA_KERNEL_SOURCES EXCLUDE REGEX ".*/.*_test\\.cu$")

set(DCU_HIPIFY_EXCLUDED_CUDA_SOURCES
  ${CUDA_KERNEL_DIR}/rec_beam_search.cu
#  ${CUDA_KERNEL_DIR}/fp8_quant.cu
)

set(DCU_HIPIFY_EXCLUDED_CUDA_HEADERS
  # ${CUDA_KERNEL_DIR}/fp8_quant_utils.cuh
)

foreach(excluded IN LISTS DCU_HIPIFY_EXCLUDED_CUDA_SOURCES)
  list(REMOVE_ITEM DCU_CUDA_KERNEL_SOURCES ${excluded})
endforeach()

foreach(excluded IN LISTS DCU_HIPIFY_EXCLUDED_CUDA_HEADERS)
  list(REMOVE_ITEM DCU_CUDA_KERNEL_HEADERS ${excluded})
endforeach()

list(REMOVE_DUPLICATES DCU_CUDA_KERNEL_SOURCES)
list(REMOVE_DUPLICATES DCU_CUDA_KERNEL_HEADERS)

set(DCU_LOCAL_HIP_SOURCES
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/matmul.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/scaled_quantize.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/scaled_matmul.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/build_block_table_from_paged_kv.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/gather_mla_latent_cache.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/random_sample.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/rejection_sample.hip

  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/attention_runner.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/piecewise_graphs.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/global_capture_instance.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/group_gemm.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/flash_mla_adapter.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/aiter_moe_adapter.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/aiter_quant_adapter.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/hipblaslt_fp8_adapter.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/topk_gate.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/gated_layer_norm.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/gemma_rms_norm.cpp

  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/causal_conv1d_adapter.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/causal_conv1d/causal_conv1d_fwd.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/causal_conv1d/causal_conv1d_update.hip
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/causal_conv1d/causal_conv1d_new_interface.hip
)

hipify_sources_target(
  DCU_GENERATED_SRCS
  DCU_HIPIFY_TARGET
  dcu_kernels
  "${DCU_CUDA_KERNEL_SOURCES}"
  EXTRA_HIPIFY_FILES
    ${DCU_CUDA_KERNEL_HEADERS}
)

set(DCU_SOURCE_FILES
  ${DCU_GENERATED_SRCS}
  ${DCU_LOCAL_HIP_SOURCES}
)

if(NOT DEFINED FLASH_MLA_LIB OR NOT EXISTS "${FLASH_MLA_LIB}")
  message(FATAL_ERROR
    "FLASH_MLA_LIB is required for DCU FlashMLA. Install flash_mla and set "
    "FLASH_MLA_LIB before invoking setup.py.")
endif()
add_library(flash_mla SHARED IMPORTED)
set_target_properties(flash_mla PROPERTIES IMPORTED_LOCATION "${FLASH_MLA_LIB}")

if(NOT DEFINED AITER_CPP_API_LIB OR NOT EXISTS "${AITER_CPP_API_LIB}")
  message(FATAL_ERROR
    "AITER_CPP_API_LIB is required for DCU grouped topk and FP8 activation "
    "quantization. Install aiter and set AITER_CPP_API_LIB before invoking "
    "setup.py.")
endif()
add_library(aiter_cpp_api SHARED IMPORTED)
set_target_properties(aiter_cpp_api PROPERTIES
  IMPORTED_LOCATION "${AITER_CPP_API_LIB}")

if(NOT DEFINED AITER_MOE_C_KERNEL_LIB OR NOT EXISTS "${AITER_MOE_C_KERNEL_LIB}")
  message(FATAL_ERROR
    "AITER_MOE_C_KERNEL_LIB is required for DCU AITER MoE FP8 kernels. "
    "Install aiter and set AITER_MOE_C_KERNEL_LIB before invoking setup.py.")
endif()
add_library(aiter_moe_c_kernel SHARED IMPORTED)
set_target_properties(aiter_moe_c_kernel PROPERTIES
  IMPORTED_LOCATION "${AITER_MOE_C_KERNEL_LIB}")

cc_library(
  NAME
    dcu_kernels
  HDRS
    ${DCU_HEADER_FILES}
  SRCS
    ${DCU_SOURCE_FILES}
  DEPS
    hip::host
    hipblaslt
    torch
    glog::glog
    :platform
    aiter_cpp_api
    aiter_moe_c_kernel
    flash_mla
)

add_dependencies(dcu_kernels ${DCU_HIPIFY_TARGET})

get_filename_component(FLASH_MLA_LIB_DIR "${FLASH_MLA_LIB}" DIRECTORY)
get_filename_component(AITER_CPP_API_LIB_DIR "${AITER_CPP_API_LIB}" DIRECTORY)
get_filename_component(AITER_MOE_C_KERNEL_LIB_DIR
                       "${AITER_MOE_C_KERNEL_LIB}" DIRECTORY)
set_property(TARGET dcu_kernels APPEND PROPERTY BUILD_RPATH "${FLASH_MLA_LIB_DIR}")
set_property(TARGET dcu_kernels APPEND PROPERTY BUILD_RPATH "${AITER_CPP_API_LIB_DIR}")
set_property(TARGET dcu_kernels APPEND PROPERTY BUILD_RPATH
             "${AITER_MOE_C_KERNEL_LIB_DIR}")

target_include_directories(dcu_kernels
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
    ${AITER_CPP_API_LIB_DIR}/../../csrc/include
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/..
)

target_compile_definitions(dcu_kernels PRIVATE
  _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI}
)

# ---- CK Tile grouped_gemm: pre-compiled shared library ----
add_library(module_cpp_api SHARED IMPORTED)
set_target_properties(module_cpp_api PROPERTIES
    IMPORTED_LOCATION "${AITER_CPP_API_LIB_DIR}/module_cpp_api.so"
)
target_link_libraries(dcu_kernels PRIVATE module_cpp_api)
