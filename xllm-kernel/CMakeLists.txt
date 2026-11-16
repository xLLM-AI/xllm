# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.
cmake_minimum_required(VERSION 3.26)
project(xllm_kernel LANGUAGES CXX)

option(XLLM_KERNEL_USE_NPU "Build the Ascend runtime and ACLNN launcher" OFF)
add_library(xllm_kernel_native SHARED csrc/runtime.cpp)
add_library(xllm_kernel::native ALIAS xllm_kernel_native)
target_compile_features(xllm_kernel_native PUBLIC cxx_std_17)
target_include_directories(xllm_kernel_native PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>)
target_link_libraries(xllm_kernel_native PRIVATE ${CMAKE_DL_LIBS})
set_target_properties(xllm_kernel_native PROPERTIES
  EXPORT_NAME native PREFIX "" OUTPUT_NAME "libxllm_kernel" BUILD_RPATH "$ORIGIN" INSTALL_RPATH "$ORIGIN")

if(XLLM_KERNEL_USE_NPU)
  find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
  execute_process(COMMAND "${Python3_EXECUTABLE}" -c
    "import pathlib,torch,torch_npu; print(pathlib.Path(torch.__file__).parent); print(pathlib.Path(torch_npu.__file__).parent); print(int(torch.compiled_with_cxx11_abi()))"
    OUTPUT_VARIABLE runtime_paths OUTPUT_STRIP_TRAILING_WHITESPACE
    COMMAND_ERROR_IS_FATAL ANY)
  string(REPLACE "\n" ";" runtime_paths "${runtime_paths}")
  list(GET runtime_paths 0 torch_root)
  list(GET runtime_paths 1 torch_npu_root)
  list(GET runtime_paths 2 torch_abi)
  find_path(cann_include acl/acl.h HINTS "$ENV{ASCEND_HOME_PATH}/include" REQUIRED)
  find_library(ascendcl ascendcl HINTS "$ENV{ASCEND_HOME_PATH}/lib64" REQUIRED)
  find_package(glog CONFIG REQUIRED)
  target_sources(xllm_kernel_native PRIVATE csrc/npu/runtime.cpp)
  target_compile_definitions(xllm_kernel_native PUBLIC _GLIBCXX_USE_CXX11_ABI=${torch_abi})
  target_include_directories(xllm_kernel_native SYSTEM PUBLIC
    "${torch_root}/include" "${torch_root}/include/torch/csrc/api/include"
    "${torch_npu_root}/include" "${torch_npu_root}" "${torch_npu_root}/.."
    "${cann_include}" ${Python3_INCLUDE_DIRS})
  target_link_libraries(xllm_kernel_native PUBLIC
    "${torch_root}/lib/libtorch.so" "${torch_root}/lib/libtorch_cpu.so"
    "${torch_root}/lib/libc10.so" "${torch_npu_root}/lib/libtorch_npu.so"
    "${ascendcl}" glog::glog ${Python3_LIBRARIES})
endif()

include(GNUInstallDirs)
install(TARGETS xllm_kernel_native EXPORT xllm_kernel_native_targets LIBRARY DESTINATION .)
install(DIRECTORY include/ DESTINATION include)
install(EXPORT xllm_kernel_native_targets NAMESPACE xllm_kernel:: DESTINATION cmake)

include(CMakePackageConfigHelpers)
configure_package_config_file(cmake/xllm_kernelConfig.cmake.in
  "${CMAKE_CURRENT_BINARY_DIR}/xllm_kernelConfig.cmake"
  INSTALL_DESTINATION cmake)
install(FILES "${CMAKE_CURRENT_BINARY_DIR}/xllm_kernelConfig.cmake" DESTINATION cmake)
