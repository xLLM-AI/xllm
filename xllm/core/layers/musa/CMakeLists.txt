include(cc_library)

set(_MUSA_LAYERS_HDRS)
set(_MUSA_LAYERS_SRCS)
set(_MUSA_LAYERS_DEPS)

if(USE_MUSA)
  list(APPEND _MUSA_LAYERS_HDRS
    ../cuda/flashinfer_workspace.h
    layer_base.h
    qwen3_decoder_layer_impl.h
    lm_head_impl.h
    attention.h
    mlp.h
  )
  list(APPEND _MUSA_LAYERS_SRCS
    ../cuda/flashinfer_workspace.cpp
    qwen3_decoder_layer_impl.cpp
    lm_head_impl.cpp
    attention.cpp
    mlp.cpp
  )
  list(APPEND _MUSA_LAYERS_DEPS
    kv_cache
    state_dict
    model
    common
    torch
    torch_musa
    MTTOplib
  )
endif()

if(XLLM_TORCH_MUSA)
  list(APPEND _MUSA_LAYERS_HDRS
    attention.h
    flashinfer_attention.h
    flashinfer_planinfo.h
    ../cuda/base_attention_impl.h
    ../cuda/flashinfer_workspace.h
    qwen3_gated_delta_net_base.h
    qwen3_next_attention.h
    qwen3_next_gated_delta_net.h
    qwen3_next_hybrid_decoder_layer_base.h
    qwen3_next_decoder_layer_impl.h
    qwen3_5_decoder_layer_impl.h
  )
  list(APPEND _MUSA_LAYERS_SRCS
    attention.cpp
    flashinfer_attention.cpp
    flashinfer_planinfo.cpp
    ../cuda/flashinfer_workspace.cpp
    qwen3_gated_delta_net_base.cpp
    qwen3_next_attention.cpp
    qwen3_next_gated_delta_net.cpp
    qwen3_next_hybrid_decoder_layer_base.cpp
    qwen3_next_decoder_layer_impl.cpp
    qwen3_5_decoder_layer_impl.cpp
  )
  list(APPEND _MUSA_LAYERS_DEPS
    :cuda_kernels
    :common_layers
    :parallel_state
    :state_dict
    :kv_cache
    :prefix_cache
    :block
    glog::glog
    gflags::gflags
    torch
  )
endif()

if(_MUSA_LAYERS_HDRS)
  cc_library(
    NAME
      musa_layers
    HDRS
      ${_MUSA_LAYERS_HDRS}
    SRCS
      ${_MUSA_LAYERS_SRCS}
    COPTS
      $<$<BOOL:${XLLM_TORCH_MUSA}>:${_XLLM_MUSIFY_COPTS}>
    DEPS
      "-Wl,--whole-archive"
      "-Wl,--no-whole-archive"
      ${_MUSA_LAYERS_DEPS}
  )
endif()
