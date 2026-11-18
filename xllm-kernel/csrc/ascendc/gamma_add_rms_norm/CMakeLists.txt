# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
cmake_minimum_required(VERSION 3.19)
project(xllm_kernel_gamma_add_rms_norm LANGUAGES CXX)
find_package(ASC REQUIRED HINTS "${ASCEND_CANN_PACKAGE_PATH}/compiler/tikcpp/ascendc_kernel_cmake")

set(vendor xllm_kernel)
set(CMAKE_SKIP_RPATH ON)
set(ENABLE_SOURCE_PACKAGE OFF)
npu_op_package(${vendor} TYPE RUN CONFIG INSTALL_PATH "${CMAKE_BINARY_DIR}")
set(host_dir "${CMAKE_CURRENT_SOURCE_DIR}/op_host")
set(host_sources
    "${host_dir}/gamma_add_rms_norm_def.cpp"
    "${host_dir}/gamma_add_rms_norm_infershape.cpp"
    "${host_dir}/gamma_add_rms_norm_tiling.cpp"
    "${host_dir}/gamma_add_rms_norm_tiling_arch35.cpp"
    "${host_dir}/tiling_util.cpp"
)
npu_op_code_gen(
    SRC "${host_dir}/gamma_add_rms_norm_def.cpp"
    PACKAGE ${vendor}
    OUT_DIR "${ASCEND_AUTOGEN_PATH}"
)
npu_op_library(cust_opapi ACLNN
    "${host_dir}/op_api/aclnn_gamma_add_rms_norm.cpp"
    "${host_dir}/op_api/gamma_add_rms_norm.cpp"
)
target_include_directories(cust_opapi PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include/aclnn")
target_link_libraries(cust_opapi PRIVATE
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libopapi.so"
    "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so"
)
set(generated_proto "${ASCEND_AUTOGEN_PATH}/op_proto.cc")
file(GLOB generated_fallback CONFIGURE_DEPENDS "${ASCEND_AUTOGEN_PATH}/fallback_*.cpp")
npu_op_library(cust_op_proto GRAPH ${generated_proto})
npu_op_library(cust_optiling TILING ${host_sources} ${generated_fallback})
foreach(target cust_optiling cust_op_proto)
    target_include_directories(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/include/op_common")
    target_link_libraries(${target} PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libops_base.so")
endforeach()
npu_op_kernel_sources(ascendc_kernels KERNEL_DIR ./)
get_filename_component(ascendc_include_dir "${CMAKE_CURRENT_SOURCE_DIR}" DIRECTORY)
npu_op_kernel_options(ascendc_kernels ALL OPTIONS --cce-auto-sync=off "-I${ascendc_include_dir}")
npu_op_kernel_library(ascendc_kernels SRC_BASE "${CMAKE_CURRENT_SOURCE_DIR}/op_kernel" TILING_LIBRARY cust_optiling)
npu_op_package_add(${vendor} LIBRARY cust_opapi cust_op_proto cust_optiling ascendc_kernels)
