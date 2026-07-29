include(cc_library)

if(NOT TARGET flash_attention)
  if(NOT DEFINED FLASH_ATTENTION_LIB OR NOT EXISTS "${FLASH_ATTENTION_LIB}")
    message(FATAL_ERROR
      "FLASH_ATTENTION_LIB is required for DCU FlashAttention. Install "
      "flash_attn in the current Python environment or set FLASH_ATTENTION_LIB "
      "to libflash_attention.so before invoking setup.py.")
  endif()
  add_library(flash_attention SHARED IMPORTED)
  set_target_properties(
    flash_attention PROPERTIES
    IMPORTED_LOCATION "${FLASH_ATTENTION_LIB}"
    INTERFACE_COMPILE_DEFINITIONS "BUILD_FA_KVCACHE")
  message(STATUS "Found libflash_attention: ${FLASH_ATTENTION_LIB}")
endif()

cc_library(
  NAME
    dcu_layers
  HDRS
    attention.h
    base_attention_impl.h
    torch_attention.h
    flash_attention.h
    fused_moe.h
    deepseek_v2_attention.h
    deepseek_v2_decoder_layer_impl.h
    qwen3_5_decoder_layer.h
    qwen3_gated_delta_net_base.h
    qwen3_gated_delta_net_helpers.h
    qwen3_gated_delta_net_impls.h
    qwen3_5_gated_delta_net.h
  SRCS
    attention.cpp
    base_attention_impl.cpp
    torch_attention.cpp
    flash_attention.cpp
    fused_moe.cpp
    deepseek_v2_attention.cpp
    deepseek_v2_decoder_layer_impl.cpp
    qwen3_5_decoder_layer.cpp
    qwen3_gated_delta_net_base.cpp
    qwen3_gated_delta_net_torch_impl.cpp
    qwen3_gated_delta_net_kernel_impl.cpp
    qwen3_5_gated_delta_net.cpp
  DEPS
    :common_layers
    flash_attention
    dcu_kernels
)
