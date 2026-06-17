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
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/random_sample.hip

  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/attention_runner.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/piecewise_graphs.cpp
  ${CMAKE_SOURCE_DIR}/xllm/core/kernels/dcu/global_capture_instance.cpp
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
)

add_dependencies(dcu_kernels ${DCU_HIPIFY_TARGET})

target_include_directories(dcu_kernels
  PUBLIC
    ${CMAKE_CURRENT_LIST_DIR}
    ${CMAKE_CURRENT_BINARY_DIR}
  PRIVATE
    ${CMAKE_CURRENT_BINARY_DIR}/..
)
