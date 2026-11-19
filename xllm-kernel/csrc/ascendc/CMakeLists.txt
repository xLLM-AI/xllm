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
npu_op_code_gen(
    SRC ${op_definitions}
    PACKAGE ${vendor}
    OUT_DIR "${ASCEND_AUTOGEN_PATH}"
)
npu_op_library(cust_opapi ACLNN
    "${gamma_host}/op_api/aclnn_gamma_add_rms_norm.cpp"
    "${gamma_host}/op_api/gamma_add_rms_norm.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_layer_norm_fwd.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_rms_norm_dynamic_quant.cpp"
    "${ASCEND_AUTOGEN_PATH}/aclnn_xllm_kernel_inplace_partial_rotary_mul.cpp"
    ${a2_aclnn_sources}
)
target_include_directories(cust_opapi PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn")
target_link_libraries(cust_opapi PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libopapi.so"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so"
)

set(generated_proto "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
file(GLOB generated_fallback CONFIGURE_DEPENDS "${ASCEND_AUTOGEN_PATH}/fallback_*.cpp")
npu_op_library(cust_op_proto GRAPH ${generated_proto})
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
    ${a2_tiling_sources}
    ${generated_fallback}
)
foreach(target cust_optiling cust_op_proto)
    target_include_directories(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include/op_common")
    target_link_libraries(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libops_base.so")
endforeach()

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
npu_op_kernel_library(ascendc_kernels SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}" TILING_LIBRARY cust_optiling)
npu_op_package_add(${vendor} LIBRARY cust_opapi cust_op_proto cust_optiling ascendc_kernels)
