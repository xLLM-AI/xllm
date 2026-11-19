# Copyright 2026 The xLLM Authors. All Rights Reserved.

cmake_minimum_required(VERSION 3.18)
project(dual_affine_layer_norm_validation LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CANN_ROOT "/usr/local/Ascend/cann-9.1.0" CACHE PATH "CANN installation")
set(DUAL_AFFINE_VENDOR "" CACHE PATH "Private custom OPP vendor directory")
set(DUAL_AFFINE_DIRECT_BUILD "" CACHE PATH "Built direct kernel directory")
add_executable(dual_affine_ln_example aclnn_example.cpp)
target_include_directories(dual_affine_ln_example PRIVATE
  ${CANN_ROOT}/include ${DUAL_AFFINE_VENDOR}/op_api/include
  ${DUAL_AFFINE_VENDOR}/op_api/include/aclnnop)
target_link_directories(dual_affine_ln_example PRIVATE
  ${CANN_ROOT}/lib64 ${DUAL_AFFINE_VENDOR}/op_api/lib)
target_link_libraries(dual_affine_ln_example PRIVATE cust_opapi opapi nnopbase ascendcl)
set_target_properties(dual_affine_ln_example PROPERTIES BUILD_RPATH
  "${DUAL_AFFINE_VENDOR}/op_api/lib;${CANN_ROOT}/lib64")

option(BUILD_TORCH_BINDING "Build ACLNN and direct PyTorch test binding" OFF)
if(BUILD_TORCH_BINDING)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  find_package(Torch REQUIRED)
  execute_process(COMMAND ${Python3_EXECUTABLE} -c
    "import pathlib,torch_npu; print(pathlib.Path(torch_npu.__file__).parent)"
    OUTPUT_VARIABLE TORCH_NPU_ROOT OUTPUT_STRIP_TRAILING_WHITESPACE COMMAND_ERROR_IS_FATAL ANY)
  get_filename_component(XLLM_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/../../.." ABSOLUTE)
  add_library(dual_affine_ln_test SHARED direct_binding.cpp
    ${XLLM_ROOT}/xllm/core/kernels/npu/xllm_ops/dual_affine_layer_norm.cpp)
  target_compile_options(dual_affine_ln_test PRIVATE -O2 ${TORCH_CXX_FLAGS})
  target_compile_definitions(dual_affine_ln_test PRIVATE TORCH_HIGHER_THAN_PTA6)
  target_include_directories(dual_affine_ln_test PRIVATE
    ${XLLM_ROOT}/xllm ${TORCH_INCLUDE_DIRS} ${TORCH_NPU_ROOT}/include
    ${CANN_ROOT}/include ${Python3_INCLUDE_DIRS}
    ${DUAL_AFFINE_DIRECT_BUILD}/include/dual_affine_ln_direct_kernel)
  target_link_directories(dual_affine_ln_test PRIVATE
    ${TORCH_NPU_ROOT}/lib ${CANN_ROOT}/lib64 ${DUAL_AFFINE_DIRECT_BUILD}/lib)
  target_link_libraries(dual_affine_ln_test PRIVATE ${TORCH_LIBRARIES}
    torch_npu ascendcl glog dual_affine_ln_direct_kernel Python3::Python)
  set_target_properties(dual_affine_ln_test PROPERTIES PREFIX "" BUILD_RPATH
    "${TORCH_NPU_ROOT}/lib;${DUAL_AFFINE_DIRECT_BUILD}/lib;${CANN_ROOT}/lib64")
endif()
