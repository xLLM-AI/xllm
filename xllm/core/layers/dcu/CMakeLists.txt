include(cc_library)

if(NOT TARGET flash_attention)
  if(NOT FLASH_ATTENTION_LIB)
    set(FLASH_ATTENTION_LIB "/opt/dtk/lib")
  endif()

  set(FLASH_ATTENTION_LIB
      "${FLASH_ATTENTION_LIB}"
      CACHE PATH "Directory containing libflash_attention.so")

  find_library(
    _flash_attn_lib_path flash_attention
    PATHS "${FLASH_ATTENTION_LIB}"
    NO_DEFAULT_PATH)

  if(_flash_attn_lib_path)
    add_library(flash_attention SHARED IMPORTED)
    set_target_properties(
      flash_attention PROPERTIES
      IMPORTED_LOCATION "${_flash_attn_lib_path}"
      INTERFACE_COMPILE_DEFINITIONS "BUILD_FA_KVCACHE")
    message(STATUS "Found libflash_attention: ${_flash_attn_lib_path}")
  else()
    message(FATAL_ERROR
      "libflash_attention.so not found in ${FLASH_ATTENTION_LIB}.")
  endif()
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
  SRCS
    attention.cpp
    base_attention_impl.cpp
    torch_attention.cpp
    flash_attention.cpp
    fused_moe.cpp
    deepseek_v2_attention.cpp
    deepseek_v2_decoder_layer_impl.cpp
  DEPS
    :common_layers
    flash_attention
    dcu_kernels
)
