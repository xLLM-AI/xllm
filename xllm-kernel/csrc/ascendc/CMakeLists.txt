# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
cmake_minimum_required(VERSION 3.19)
project(xllm_kernel_ascendc LANGUAGES CXX)
find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

set(vendor xllm_kernel)
set(CMAKE_SKIP_RPATH ON)
set(ENABLE_SOURCE_PACKAGE OFF)
npu_op_package(${vendor} TYPE RUN CONFIG INSTALL_PATH "${CMAKE_BINARY_DIR}")

set(gamma_host "${CMAKE_CURRENT_SOURCE_DIR}/gamma_add_rms_norm/op_host")
set(layer_norm_host "${CMAKE_CURRENT_SOURCE_DIR}/layer_norm_fwd/op_host")
set(quant_host "${CMAKE_CURRENT_SOURCE_DIR}/rms_norm_dynamic_quant/op_host")
set(rope_host "${CMAKE_CURRENT_SOURCE_DIR}/inplace_partial_rotary_mul/op_host")
set(op_definitions
    "${gamma_host}/gamma_add_rms_norm_def.cpp"
    "${layer_norm_host}/layer_norm_fwd_def.cpp"
    "${quant_host}/rms_norm_dynamic_quant_def.cpp"
    "${rope_host}/inplace_partial_rotary_mul_def.cpp"
)
set(hc_components hc_pre_inv_rms hc_pre_sinkhorn hc_post)
foreach(component IN LISTS hc_components)
    set(hc_host "${CMAKE_CURRENT_SOURCE_DIR}/${component}/op_host")
    list(APPEND op_definitions "${hc_host}/${component}_def.cpp")
    list(APPEND hc_tiling_sources
        "${hc_host}/${component}_proto.cpp"
        "${hc_host}/${component}_tiling.cpp"
    )
endforeach()
if("ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    set(hc_va_host "${CMAKE_CURRENT_SOURCE_DIR}/hc_pre_va/op_host")
    list(APPEND op_definitions "${hc_va_host}/hc_pre_va_def.cpp")
    list(APPEND hc_tiling_sources "${hc_va_host}/hc_pre_va_tiling.cpp")
    list(APPEND hc_components hc_pre_va)
endif()
foreach(component IN LISTS hc_components)
    list(APPEND hc_aclnn_sources "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_${component}.cpp")
endforeach()
set(moe_hash_host "${CMAKE_CURRENT_SOURCE_DIR}/moe_gating_top_k_hash/op_host")
set(moe_routing_host "${CMAKE_CURRENT_SOURCE_DIR}/moe_init_routing_v3/op_host")
set(moe_gmm_host "${CMAKE_CURRENT_SOURCE_DIR}/moe_grouped_matmul_swiglu_quant/op_host")
set(moe_dequant_host "${CMAKE_CURRENT_SOURCE_DIR}/dequant_swiglu_quant/op_host")
set(conv_host "${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d/op_host")
set(conv_qkv_host "${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d_qkv/op_host")
set(chunk_gdn_host "${CMAKE_CURRENT_SOURCE_DIR}/mega_chunk_gdn/op_host")
set(recurrent_gdn_host "${CMAKE_CURRENT_SOURCE_DIR}/recurrent_gated_delta_rule/op_host")
set(mega_prefill_host "${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_prefill_op/op_host")
set(mega_decode_host "${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_decode/op_host")
set(mega_mtp_host "${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_mtp_decode/op_host")
set(mega_varlen_host "${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_mtp_decode_varlen/op_host")
set(qli "${CMAKE_CURRENT_SOURCE_DIR}/quant_lightning_indexer")
set(qli_metadata "${CMAKE_CURRENT_SOURCE_DIR}/quant_lightning_indexer_metadata")
set(scatter "${CMAKE_CURRENT_SOURCE_DIR}/scatter_nd_update_v2")
set(sas_metadata "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attn_sharedkv_metadata")
set(sparse_attention_components compressor sparse_attn_sharedkv sparse_flash_attention_lse)
foreach(component IN LISTS sparse_attention_components)
    set(host "${CMAKE_CURRENT_SOURCE_DIR}/${component}/op_host")
    list(APPEND sparse_attention_definitions "${host}/${component}_def.cpp")
    list(APPEND sparse_attention_tiling "${host}/${component}_tiling.cpp")
endforeach()
set(pto_include "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pto-isa/include")
if(NOT EXISTS "${pto_include}/pto/pto-inst.hpp")
    message(FATAL_ERROR "MegaChunkGdn needs PTO headers. "
        "In a Git checkout run git submodule update --init xllm-kernel/third_party/pto-isa.")
endif()
list(APPEND op_definitions
    "${moe_hash_host}/moe_gating_top_k_hash_def.cpp"
    "${moe_gmm_host}/moe_grouped_matmul_swiglu_quant_def.cpp"
    "${moe_dequant_host}/dequant_swiglu_quant_def.cpp"
    "${conv_host}/causal_conv1d_def.cpp"
    "${conv_qkv_host}/causal_conv1d_qkv_def.cpp"
    "${chunk_gdn_host}/mega_chunk_gdn_def.cpp"
    "${mega_prefill_host}/mega_gdn_prefill_op_def.cpp"
    "${mega_decode_host}/mega_gdn_decode_def.cpp"
    "${mega_mtp_host}/mega_gdn_mtp_decode_def.cpp"
    "${qli}/op_host/quant_lightning_indexer_def.cpp"
    "${scatter}/op_host/scatter_nd_update_v2_def.cpp"
    ${sparse_attention_definitions}
)
if("ascend910b" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    list(APPEND op_definitions
        "${moe_routing_host}/moe_init_routing_v3_def.cpp"
        "${recurrent_gdn_host}/recurrent_gated_delta_rule_def.cpp"
        "${mega_varlen_host}/mega_gdn_mtp_decode_varlen_def.cpp"
    )
    set(mega_varlen_api "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_mega_gdn_mtp_decode_varlen.cpp")
    set(mega_varlen_tiling
        "${mega_varlen_host}/mega_gdn_mtp_decode_varlen_tiling.cpp"
        "${mega_varlen_host}/mega_gdn_mtp_decode_varlen_infershape.cpp"
    )
    set(recurrent_gdn_api
        "${recurrent_gdn_host}/op_api/aclnn_recurrent_gated_delta_rule.cpp"
        "${recurrent_gdn_host}/op_api/recurrent_gated_delta_rule_l0.cpp"
    )
    set(recurrent_gdn_tiling
        "${recurrent_gdn_host}/recurrent_gated_delta_rule_tiling.cpp"
        "${recurrent_gdn_host}/recurrent_gated_delta_rule_proto.cpp"
    )
    set(moe_routing_api
        "${moe_routing_host}/op_api/aclnn_moe_init_routing_v3.cpp"
        "${moe_routing_host}/op_api/moe_init_routing_v3_l0.cpp"
    )
    set(moe_routing_tiling
        "${moe_routing_host}/moe_init_routing_v3_proto.cpp"
        "${moe_routing_host}/moe_init_routing_v3_tiling.cpp"
        "${moe_routing_host}/moe_init_routing_v3_tiling_base.cpp"
    )
endif()
# The two quantized projections have device implementations on the A2/A3 targets.
if("ascend910b" IN_LIST ASCEND_COMPUTE_UNIT OR "ascend910_93" IN_LIST ASCEND_COMPUTE_UNIT)
    set(quant_gemm_host "${CMAKE_CURRENT_SOURCE_DIR}/quant_matmul_nz_decode/op_host")
    set(mla_host "${CMAKE_CURRENT_SOURCE_DIR}/mla_preprocess_v2/op_host")
    set(catlass_include "${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/catlass/include")
    if(NOT EXISTS "${catlass_include}/catlass/catlass.hpp")
        message(FATAL_ERROR "QuantMatmulNzDecode needs the CATLASS headers. "
            "In a Git checkout run git submodule update --init xllm-kernel/third_party/catlass.")
    endif()
    list(APPEND op_definitions
        "${quant_gemm_host}/quant_matmul_nz_decode_def.cpp"
        "${mla_host}/mla_preprocess_v2_def.cpp"
    )
    set(a2_aclnn_sources
        "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_quant_matmul_nz_decode.cpp"
        "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_mla_preprocess_v2.cpp"
    )
    set(a2_tiling_sources
        "${quant_gemm_host}/quant_matmul_nz_decode_proto.cpp"
        "${quant_gemm_host}/quant_matmul_nz_decode_tiling.cpp"
        "${mla_host}/mla_preprocess_tiling.cpp"
        "${mla_host}/mla_preprocess_v2_proto.cpp"
        "${mla_host}/mla_preprocess_v2_tiling.cpp"
    )
endif()
if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    set(reshape_a5_host "${CMAKE_CURRENT_SOURCE_DIR}/reshape_and_cache_a5/op_host")
    list(APPEND op_definitions "${reshape_a5_host}/reshape_and_cache_a5_def.cpp")
    set(reshape_a5_api "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_reshape_and_cache_a5.cpp")
    set(reshape_a5_tiling
        "${reshape_a5_host}/reshape_and_cache_a5_proto.cpp"
        "${reshape_a5_host}/reshape_and_cache_a5_tiling.cpp"
    )
endif()
npu_op_code_gen(
    SRC ${op_definitions}
    PACKAGE ${vendor}
    OUT_DIR "${ASCEND_AUTOGEN_PATH}"
)
foreach(component IN LISTS sparse_attention_components)
    list(APPEND sparse_attention_api "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_${component}.cpp")
endforeach()
npu_op_library(cust_opapi ACLNN
    "${gamma_host}/op_api/aclnn_gamma_add_rms_norm.cpp"
    "${gamma_host}/op_api/gamma_add_rms_norm.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_layer_norm_fwd.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_rms_norm_dynamic_quant.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_inplace_partial_rotary_mul.cpp"
    ${hc_aclnn_sources}
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_moe_gating_top_k_hash.cpp"
    ${moe_routing_api}
    "${moe_gmm_host}/op_api/aclnn_moe_grouped_matmul_swiglu_quant.cpp"
    "${moe_gmm_host}/op_api/moe_grouped_matmul_swiglu_quant_l0.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_dequant_swiglu_quant.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_dequant_swiglu_quant_v2.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_causal_conv1d.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_causal_conv1d_qkv.cpp"
    "${conv_qkv_host}/causal_conv1d_qkv_abi.cpp"
    "${chunk_gdn_host}/op_api/aclnn_mega_chunk_gdn.cpp"
    "${chunk_gdn_host}/op_api/mega_chunk_gdn_l0.cpp"
    ${recurrent_gdn_api}
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_mega_gdn_prefill_op.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_mega_gdn_decode.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_mega_gdn_mtp_decode.cpp"
    ${mega_varlen_api}
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_quant_lightning_indexer.cpp"
    "${qli_metadata}/op_api/aclnn_quant_lightning_indexer_metadata.cpp"
    "${qli_metadata}/op_api/l0_quant_lightning_indexer_metadata.cpp"
    "${scatter}/op_host/op_api/aclnn_scatter_nd_update_v2.cpp"
    "${scatter}/op_host/op_api/scatter_nd_update_v2.cpp"
    ${sparse_attention_api}
    ${reshape_a5_api}
    "${sas_metadata}/op_api/aclnn_sparse_attn_sharedkv_metadata.cpp"
    "${sas_metadata}/op_api/l0_sparse_attn_sharedkv_metadata.cpp"
    ${a2_aclnn_sources}
)
target_include_directories(cust_opapi PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn"
    "${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/aicpu"
    "${ASCEND_AUTOGEN_PATH}"
)
target_link_libraries(cust_opapi PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libopapi.so"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so"
)

set(generated_proto "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
file(GLOB generated_fallback CONFIGURE_DEPENDS "${ASCEND_AUTOGEN_PATH}/fallback_*.cpp")
npu_op_library(cust_op_proto GRAPH
    ${generated_proto}
    "${qli_metadata}/op_graph/quant_lightning_indexer_metadata_proto.cpp"
    "${sas_metadata}/op_graph/sparse_attn_sharedkv_metadata_proto.cpp"
)
npu_op_library(cust_optiling TILING
    ${op_definitions}
    "${gamma_host}/gamma_add_rms_norm_infershape.cpp"
    "${gamma_host}/gamma_add_rms_norm_tiling.cpp"
    "${gamma_host}/gamma_add_rms_norm_tiling_arch35.cpp"
    "${gamma_host}/tiling_util.cpp"
    "${layer_norm_host}/layer_norm_fwd_proto.cpp"
    "${layer_norm_host}/layer_norm_fwd_tiling.cpp"
    "${quant_host}/rms_norm_dynamic_quant_tiling.cpp"
    "${rope_host}/inplace_partial_rotary_mul_tiling.cpp"
    "${rope_host}/inplace_partial_rotary_mul_a3_tiling.cpp"
    "${rope_host}/rope_regbase_tiling_base.cpp"
    "${rope_host}/rope_regbase_tiling_a_and_b.cpp"
    "${rope_host}/rope_regbase_tiling_ab.cpp"
    "${rope_host}/rope_regbase_tiling_aba_and_ba.cpp"
    "${rope_host}/rope_regbase_tiling_bab.cpp"
    ${hc_tiling_sources}
    "${moe_hash_host}/moe_gating_top_k_hash_tiling.cpp"
    ${moe_routing_tiling}
    "${moe_gmm_host}/moe_grouped_matmul_swiglu_quant_proto.cpp"
    "${moe_gmm_host}/moe_grouped_matmul_swiglu_quant_tiling.cpp"
    "${moe_dequant_host}/dequant_swiglu_quant_infershape.cpp"
    "${moe_dequant_host}/dequant_swiglu_quant_tiling.cpp"
    "${moe_dequant_host}/dequant_swiglu_quant_tiling_base.cpp"
    "${moe_dequant_host}/dequant_swiglu_quant_tiling_arch35.cpp"
    "${conv_host}/causal_conv1d_proto.cpp"
    "${conv_host}/causal_conv1d_tiling.cpp"
    "${conv_qkv_host}/causal_conv1d_qkv_proto.cpp"
    "${conv_qkv_host}/causal_conv1d_qkv_tiling.cpp"
    "${chunk_gdn_host}/mega_chunk_gdn_infershape.cpp"
    "${chunk_gdn_host}/mega_chunk_gdn_tiling.cpp"
    ${recurrent_gdn_tiling}
    "${mega_prefill_host}/mega_gdn_prefill_op_tiling.cpp"
    "${mega_prefill_host}/mega_gdn_prefill_op_infershape.cpp"
    "${mega_decode_host}/mega_gdn_decode_tiling.cpp"
    "${mega_decode_host}/mega_gdn_decode_proto.cpp"
    "${mega_mtp_host}/mega_gdn_mtp_decode_tiling.cpp"
    "${mega_mtp_host}/mega_gdn_mtp_decode_proto.cpp"
    ${mega_varlen_tiling}
    "${qli}/op_host/quant_lightning_indexer_infershape.cpp"
    "${qli}/op_host/quant_lightning_indexer_tiling.cpp"
    "${qli_metadata}/op_host/quant_lightning_indexer_metadata_infershape.cpp"
    "${scatter}/op_host/scatter_nd_update_infershape.cpp"
    "${scatter}/op_host/scatter_nd_update_v2_tiling.cpp"
    ${sparse_attention_tiling}
    ${reshape_a5_tiling}
    "${CMAKE_CURRENT_SOURCE_DIR}/compressor/op_host/compressor_proto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse_attn_sharedkv/op_host/sparse_attn_sharedkv_proto.cpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/sparse_flash_attention_lse/op_host/sparse_flash_attention_lse_infershape.cpp"
    "${sas_metadata}/op_host/sparse_attn_sharedkv_metadata_infershape.cpp"
    ${a2_tiling_sources}
    ${generated_fallback}
)
foreach(target cust_optiling cust_op_proto)
    target_include_directories(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include/op_common")
    target_link_libraries(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libops_base.so")
endforeach()
target_include_directories(cust_optiling PRIVATE
    "${CMAKE_CURRENT_SOURCE_DIR}"
    "${qli}/op_host"
    "${ASCEND_CANN_PACKAGE_PATH}/pkg_inc"
    "${ASCEND_CANN_PACKAGE_PATH}/include/op_common/op_host"
)
target_link_libraries(cust_optiling PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendalog.so")

npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelGammaAddRmsNorm
    KERNEL_DIR gamma_add_rms_norm/op_kernel KERNEL_FILE xllm_kernel_gamma_add_rms_norm.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelLayerNormFwd
    KERNEL_DIR layer_norm_fwd/op_kernel KERNEL_FILE xllm_kernel_layer_norm_fwd.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelRmsNormDynamicQuant
    KERNEL_DIR rms_norm_dynamic_quant/op_kernel KERNEL_FILE xllm_kernel_rms_norm_dynamic_quant.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelInplacePartialRotaryMul
    KERNEL_DIR inplace_partial_rotary_mul/op_kernel KERNEL_FILE xllm_kernel_inplace_partial_rotary_mul.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelGammaAddRmsNorm
    OPTIONS --cce-auto-sync=off "-I${CMAKE_CURRENT_SOURCE_DIR}")
npu_op_kernel_options(ascendc_kernels XllmKernelLayerNormFwd
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error)
npu_op_kernel_options(ascendc_kernels XllmKernelRmsNormDynamicQuant
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
    -mllvm -cce-aicore-hoist-movemask=false --op_relocatable_kernel_binary=true)
npu_op_kernel_options(ascendc_kernels XllmKernelInplacePartialRotaryMul
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror
    -mllvm -cce-aicore-hoist-movemask=false --op_relocatable_kernel_binary=true)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelHcPreInvRms
    KERNEL_DIR hc_pre_inv_rms/op_kernel KERNEL_FILE xllm_kernel_hc_pre_inv_rms.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelHcPreSinkhorn
    KERNEL_DIR hc_pre_sinkhorn/op_kernel KERNEL_FILE xllm_kernel_hc_pre_sinkhorn.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelHcPost
    KERNEL_DIR hc_post/op_kernel KERNEL_FILE xllm_kernel_hc_post.cpp)
set(hc_kernel_types XllmKernelHcPreInvRms XllmKernelHcPreSinkhorn XllmKernelHcPost)
if(hc_va_host)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelHcPreVa
        KERNEL_DIR hc_pre_va/op_kernel KERNEL_FILE xllm_kernel_hc_pre_va.cpp)
    list(APPEND hc_kernel_types XllmKernelHcPreVa)
endif()
foreach(op_type IN LISTS hc_kernel_types)
    npu_op_kernel_options(ascendc_kernels ${op_type}
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
        -mllvm -cce-aicore-hoist-movemask=false --op_relocatable_kernel_binary=true)
endforeach()
npu_op_kernel_options(ascendc_kernels XllmKernelHcPreSinkhorn OPTIONS -Wno-constant-conversion)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMoeGatingTopKHash
    KERNEL_DIR moe_gating_top_k_hash/op_kernel KERNEL_FILE xllm_kernel_moe_gating_top_k_hash.cpp)
if(moe_routing_api)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMoeInitRoutingV3
        KERNEL_DIR moe_init_routing_v3/op_kernel KERNEL_FILE xllm_kernel_moe_init_routing_v3.cpp)
    npu_op_kernel_options(ascendc_kernels XllmKernelMoeInitRoutingV3
        OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error)
endif()
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelDequantSwigluQuant
    KERNEL_DIR dequant_swiglu_quant/op_kernel KERNEL_FILE xllm_kernel_dequant_swiglu_quant.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelDequantSwigluQuant
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMoeGroupedMatmulSwigluQuant
    KERNEL_DIR moe_grouped_matmul_swiglu_quant/op_kernel KERNEL_FILE xllm_kernel_moe_grouped_matmul_swiglu_quant.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelMoeGatingTopKHash
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror
    -mllvm -cce-aicore-hoist-movemask=false --op_relocatable_kernel_binary=true)
npu_op_kernel_options(ascendc_kernels XllmKernelMoeGroupedMatmulSwigluQuant
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelCausalConv1d
    KERNEL_DIR causal_conv1d/op_kernel KERNEL_FILE xllm_kernel_causal_conv1d.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelCausalConv1dQkv
    KERNEL_DIR causal_conv1d_qkv/op_kernel KERNEL_FILE xllm_kernel_causal_conv1d_qkv.cpp)
foreach(op_type XllmKernelCausalConv1d XllmKernelCausalConv1dQkv)
    npu_op_kernel_options(ascendc_kernels ${op_type}
        OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Werror "-I${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMegaChunkGdn
    KERNEL_DIR mega_chunk_gdn/op_kernel KERNEL_FILE xllm_kernel_mega_chunk_gdn.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelMegaChunkGdn
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error
    -DGDN_KERNEL_NAME=xllm_kernel_mega_chunk_gdn -DGDN_D=128 -DGDN_C=128
    "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${pto_include}" "-I${ASCEND_CANN_PACKAGE_PATH}/include")
if(recurrent_gdn_api)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelRecurrentGatedDeltaRule
        KERNEL_DIR recurrent_gated_delta_rule/op_kernel KERNEL_FILE xllm_kernel_recurrent_gated_delta_rule.cpp)
    npu_op_kernel_options(ascendc_kernels XllmKernelRecurrentGatedDeltaRule
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Wno-error)
endif()
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMegaGdnPrefillOp
    KERNEL_DIR mega_gdn_prefill_op/op_kernel KERNEL_FILE xllm_kernel_mega_gdn_prefill_op.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMegaGdnDecode
    KERNEL_DIR mega_gdn_decode/op_kernel KERNEL_FILE xllm_kernel_mega_gdn_decode.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMegaGdnMtpDecode
    KERNEL_DIR mega_gdn_mtp_decode/op_kernel KERNEL_FILE xllm_kernel_mega_gdn_mtp_decode.cpp)
set(mega_kernel_types XllmKernelMegaGdnDecode XllmKernelMegaGdnMtpDecode)
if(mega_varlen_api)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMegaGdnMtpDecodeVarlen
        KERNEL_DIR mega_gdn_mtp_decode_varlen/op_kernel KERNEL_FILE xllm_kernel_mega_gdn_mtp_decode_varlen.cpp)
    list(APPEND mega_kernel_types XllmKernelMegaGdnMtpDecodeVarlen)
endif()
foreach(op_type IN LISTS mega_kernel_types)
    npu_op_kernel_options(ascendc_kernels ${op_type}
        OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error
        "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${pto_include}"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_decode/op_kernel"
        "-I${CMAKE_CURRENT_SOURCE_DIR}/mega_gdn_mtp_decode/op_kernel")
endforeach()
npu_op_kernel_options(ascendc_kernels XllmKernelMegaGdnPrefillOp
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error
    -DGDN_KERNEL_NAME=xllm_kernel_mega_gdn_prefill_op -DGDN_D=128 -DGDN_C=128
    "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${pto_include}"
    "-I${ASCEND_CANN_PACKAGE_PATH}/include" "-I${ASCEND_CANN_PACKAGE_PATH}/pkg_inc"
    "-I${ASCEND_CANN_PACKAGE_PATH}/pkg_inc/profiling"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/causal_conv1d/op_kernel"
    "-I${CMAKE_CURRENT_SOURCE_DIR}/mega_chunk_gdn/op_kernel")
if("ascend950" IN_LIST ASCEND_COMPUTE_UNIT)
    file(STRINGS "${ASCEND_CANN_PACKAGE_PATH}/opp/version.info" mega_cann_version REGEX "^Version=" LIMIT_COUNT 1)
    string(REGEX REPLACE "^Version=" "" mega_cann_version "${mega_cann_version}")
    if(mega_cann_version VERSION_LESS "9.1.0")
        message(FATAL_ERROR "MegaGdnPrefillOp on Ascend950 requires CANN 9.1.0 or newer.")
    endif()

endif()
if(a2_aclnn_sources)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelQuantMatmulNzDecode
        KERNEL_DIR quant_matmul_nz_decode/op_kernel KERNEL_FILE xllm_kernel_quant_matmul_nz_decode.cpp)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelMlaPreprocessV2
        KERNEL_DIR mla_preprocess_v2/op_kernel KERNEL_FILE xllm_kernel_mla_preprocess_v2.cpp)
    npu_op_kernel_options(ascendc_kernels XllmKernelQuantMatmulNzDecode
        OPTIONS --cce-auto-sync=off -DCATLASS_ARCH=2201 -Wno-deprecated-declarations -Wno-error
        "-I${catlass_include}")
    npu_op_kernel_options(ascendc_kernels XllmKernelMlaPreprocessV2
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror)
endif()
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelQuantLightningIndexer
    KERNEL_DIR quant_lightning_indexer/op_kernel KERNEL_FILE xllm_kernel_quant_lightning_indexer.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelQuantLightningIndexer
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Wno-error
    -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false
    "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${qli}/op_kernel")
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelScatterNdUpdateV2
    KERNEL_DIR scatter_nd_update_v2/op_kernel KERNEL_FILE xllm_kernel_scatter_nd_update_v2.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelScatterNdUpdateV2
    OPTIONS --cce-auto-sync=on -Wno-deprecated-declarations -Wno-error)

npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelCompressor
    KERNEL_DIR compressor/op_kernel KERNEL_FILE xllm_kernel_compressor.cpp)
npu_op_kernel_options(ascendc_kernels XllmKernelCompressor
    OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Wno-error
    -mllvm -cce-aicore-hoist-movemask=false --op_relocatable_kernel_binary=true
    "-I${CMAKE_CURRENT_SOURCE_DIR}" "-I${CMAKE_CURRENT_SOURCE_DIR}/compressor/op_kernel")
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelSparseAttnSharedkv
    KERNEL_DIR sparse_attn_sharedkv/op_kernel KERNEL_FILE xllm_kernel_sparse_attn_sharedkv.cpp)
npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelSparseFlashAttentionLse
    KERNEL_DIR sparse_flash_attention_lse/op_kernel KERNEL_FILE xllm_kernel_sparse_flash_attention_lse.cpp)
foreach(op_type XllmKernelSparseAttnSharedkv XllmKernelSparseFlashAttentionLse)
    npu_op_kernel_options(ascendc_kernels ${op_type}
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Wno-error
        -mllvm -cce-vf-remove-membar=false -mllvm -cce-aicore-hoist-movemask=false
        "-I${CMAKE_CURRENT_SOURCE_DIR}")
endforeach()

if(reshape_a5_api)
    npu_op_kernel_sources(ascendc_kernels OP_TYPE XllmKernelReshapeAndCacheA5
        KERNEL_DIR reshape_and_cache_a5/op_kernel KERNEL_FILE reshape_and_cache_a5.cpp)
    npu_op_kernel_options(ascendc_kernels XllmKernelReshapeAndCacheA5
        OPTIONS --cce-auto-sync=off -Wno-deprecated-declarations -Werror)
endif()

npu_op_kernel_library(ascendc_kernels SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}" TILING_LIBRARY cust_optiling)

include("${CMAKE_CURRENT_SOURCE_DIR}/aicpu.cmake")

npu_op_package_add(${vendor} LIBRARY cust_opapi cust_op_proto cust_optiling ascendc_kernels)
