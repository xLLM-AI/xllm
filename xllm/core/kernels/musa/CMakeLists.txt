include(cc_library)

# MUSA kernel build - FlashInfer/Mate path with mcc_wrapper (C1 baseline).
if(USE_MUSA)
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
    # Link stub: CUTLASS fused MoE is not available on MUSA.
    cutlass_fused_moe_stub.cpp
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
