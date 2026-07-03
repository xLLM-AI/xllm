include(cc_library)

if(USE_MUSA)
  file(GLOB_RECURSE MUSA_HEADER_FILES
    "${CMAKE_CURRENT_LIST_DIR}/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/cuda/*.h"
    "${CMAKE_CURRENT_LIST_DIR}/cuda/*.cuh"
  )

  file(GLOB MUSA_LOCAL_CPP "${CMAKE_CURRENT_LIST_DIR}/*.cpp")
  file(GLOB MUSA_CUDA_CPP "${CMAKE_CURRENT_LIST_DIR}/../cuda/*.cpp")
  set(MUSA_SOURCE_FILES
    ${MUSA_LOCAL_CPP}
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/activation.cu
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/norm.cu
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/reshape_paged_cache.cu
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/rope.cu
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_fused_topk.cu
    ${MUSA_CUDA_CPP}
  )

  cc_library(
    NAME
      musa_kernels
    HDRS
      ${MUSA_HEADER_FILES}
    SRCS
      ${MUSA_SOURCE_FILES}
    DEPS
      tvm_ffi
      torch
      :util
      :platform
  )
endif()

if(XLLM_TORCH_MUSA)
  include_directories(${CMAKE_CURRENT_LIST_DIR})
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda)
  include_directories(${CMAKE_CURRENT_LIST_DIR}/../cuda/moe)

  # Qwen3.5-27B text serving path (bf16 dense, FlashInfer + GDN hybrid).

  set(_TORCH_MUSA_KERNEL_HDRS
    attention_runner.h
    musa_ops_api.h
    musa_tvmffi_stream.h
    gdn_ops.h
    global_capture_instance.h
    llm_decode_metadata_update.h
    piecewise_graphs.h
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk.cuh
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_sigmoid_kernels.cuh
    ${CMAKE_CURRENT_LIST_DIR}/../cuda/moe/moe_topk_softmax_kernels.cuh
  )

  set(_TORCH_MUSA_KERNEL_SRCS
    # FlashInfer attention (12 full-attn layers)
    attention_runner.cpp
    attention_decode.cpp
    attention_prefill.cpp
    # GDN hybrid layers (36 linear-attn layers)
    gdn_decode.cu
    gdn_prefill.cpp
    # Norm / MLP / projections
    gemma_norm.cu
    activation.cu
    rope.cu
    paged_cache.cu
    random_sample.cpp
    # Mate FFI + graph helpers
    musa_tvmffi_stream.cpp
    global_capture_instance.cpp
    piecewise_graphs.cpp
    llm_decode_metadata_update.cu
  )

  cc_library(
    NAME
      cuda_kernels
    HDRS
      ${_TORCH_MUSA_KERNEL_HDRS}
    SRCS
      ${_TORCH_MUSA_KERNEL_SRCS}
    COPTS
      ${_XLLM_MUSIFY_COPTS}
    DEPS
      tvm_ffi
      torch
      :util
      :platform
      musa
      musart
  )
endif()
