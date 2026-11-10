#
# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# ==============================================================================

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# Extensions import host registry functions at load time. This interface must
# never link :models, :runtime, or another archive containing host singletons.
add_library(native_extension INTERFACE)
add_library(XLLM::native_extension ALIAS native_extension)
target_compile_features(native_extension INTERFACE cxx_std_20)
target_include_directories(native_extension INTERFACE
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/xllm>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/xllm/core>
  $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/xllm>
  $<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/xllm/proto>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/core>
  $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/proto>
)
target_link_libraries(native_extension INTERFACE
  torch
  glog::glog
  gflags::gflags
  Folly::folly
  Boost::boost
  protobuf::libprotobuf
  absl::flat_hash_map
  absl::strings
  nlohmann_json::nlohmann_json
)
separate_arguments(_xllm_native_torch_options NATIVE_COMMAND "${TORCH_CXX_FLAGS}")
target_compile_options(native_extension INTERFACE ${_xllm_native_torch_options})
target_compile_definitions(native_extension INTERFACE
  _GLIBCXX_USE_CXX11_ABI=${USE_CXX11_ABI} C10_USE_GLOG
)
foreach(_device NPU CUDA MLU MUSA DCU ILU MACA)
  if(USE_${_device})
    target_compile_definitions(native_extension INTERFACE USE_${_device})
  endif()
endforeach()
if(USE_NPU)
  get_filename_component(_xllm_native_atb_parent
    "${XLLM_ATB_LAYERS_SOURCE_DIR}" DIRECTORY)
  target_compile_definitions(native_extension INTERFACE
    BUILD_LIBTORCH TORCH_SETCUSTOMHANDLER=ON TORCH_HIGHER_THAN_PTA6
  )
  target_include_directories(native_extension INTERFACE
    $<BUILD_INTERFACE:${_xllm_native_atb_parent}>
    $<BUILD_INTERFACE:${XLLM_ATB_LAYERS_SOURCE_DIR}>
    $<BUILD_INTERFACE:${XLLM_ATB_LAYERS_SOURCE_DIR}/core/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/spdlog/include>
    $<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/third_party/torch_npu_ops>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/xllm_atb_layers>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/xllm_atb_layers/core/include>
    $<INSTALL_INTERFACE:${CMAKE_INSTALL_INCLUDEDIR}/xllm/torch_npu_ops>
  )
endif()
if(USE_MLU)
  target_include_directories(native_extension INTERFACE
    $<BUILD_INTERFACE:$ENV{PYTORCH_MLU_INSTALL_PATH}>
    $<BUILD_INTERFACE:$ENV{PYTORCH_MLU_INSTALL_PATH}/../>
    $<BUILD_INTERFACE:$ENV{PYTORCH_MLU_INSTALL_PATH}/csrc>
    $<BUILD_INTERFACE:$ENV{PYTORCH_MLU_INSTALL_PATH}/csrc/include>
    $<BUILD_INTERFACE:$ENV{NEUWARE_HOME}/include>
  )
endif()
if(USE_MUSA)
  target_include_directories(native_extension INTERFACE
    "$<BUILD_INTERFACE:${MUSA_INCLUDE_DIRS}>"
    "$<BUILD_INTERFACE:${TorchMusa_INCLUDE_DIRS}>"
    $<BUILD_INTERFACE:${MUSA_PATH}/include>
    $<BUILD_INTERFACE:${_TORCH_MUSA_ROOT}/share/torch_musa_codegen>
    $<BUILD_INTERFACE:${_TORCH_MUSA_ROOT}/share/generated_cuda_compatible/include>
    $<BUILD_INTERFACE:${_TORCH_MUSA_ROOT}/share/generated_cuda_compatible/include/torch/csrc/api/include>
    $<BUILD_INTERFACE:${_TORCH_MUSA_ROOT}/share/generated_cuda_compatible/include/torch/csrc/musa>
  )
endif()
if(USE_CUDA OR USE_MUSA)
  target_compile_definitions(native_extension INTERFACE TORCH_CUDA=1)
endif()

install(TARGETS native_extension EXPORT XLLMNativeTargets)
install(EXPORT XLLMNativeTargets NAMESPACE XLLM::
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XLLMNative)
install(DIRECTORY ${PROJECT_SOURCE_DIR}/xllm/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm
  FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
install(FILES
  ${PROJECT_BINARY_DIR}/xllm/core/common/native_library_build_info.h
  ${PROJECT_BINARY_DIR}/xllm/core/common/xllm_build_info.h
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm/core/common)
install(DIRECTORY ${PROJECT_BINARY_DIR}/xllm/proto/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm/proto
  FILES_MATCHING PATTERN "*.pb.h")
install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/xxHash/
  DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm/xxHash
  FILES_MATCHING PATTERN "*.h")
if(USE_NPU)
  # Public ATB headers use header-only spdlog, including its bundled fmt headers.
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/spdlog/include/spdlog
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm
    FILES_MATCHING PATTERN "*.h")
  install(DIRECTORY ${XLLM_ATB_LAYERS_SOURCE_DIR}/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm/xllm_atb_layers
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
  install(DIRECTORY ${PROJECT_SOURCE_DIR}/third_party/torch_npu_ops/
    DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}/xllm/torch_npu_ops
    FILES_MATCHING PATTERN "*.h" PATTERN "*.inc")
endif()
configure_package_config_file(
  ${PROJECT_SOURCE_DIR}/cmake/XLLMNativeConfig.cmake.in
  ${PROJECT_BINARY_DIR}/XLLMNativeConfig.cmake
  INSTALL_DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XLLMNative)
install(FILES ${PROJECT_BINARY_DIR}/XLLMNativeConfig.cmake
  DESTINATION ${CMAKE_INSTALL_LIBDIR}/cmake/XLLMNative)

function(xllm_export_native_host target)
  get_target_property(_type ${target} TYPE)
  if(_type STREQUAL "EXECUTABLE")
    set_target_properties(${target} PROPERTIES ENABLE_EXPORTS ON)
  elseif(_type STREQUAL "SHARED_LIBRARY")
    # Python imports xllm_export with RTLD_LOCAL.  The loader promotes the
    # module with RTLD_GLOBAL before opening an extension, so the host registry
    # symbols must be present in the module's dynamic symbol table.
    target_link_options(${target} PRIVATE -Wl,--export-dynamic)
  endif()
endfunction()
