include(cc_library)

if(USE_NPU)
  add_subdirectory(npu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()

if(USE_MUSA)
  set(CMAKE_CXX_COMPILER ${CMAKE_MUSA_COMPILER})
  string(REPLACE "-Wno-format-truncation" "" CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-error=macro-redefined \
  -Wno-error=defaulted-function-deleted -x musa --offload-arch=mp_31 \
  -fplugin=$ENV{MUSAMAPPING_PATH}/libMusaMapping.so")
  set(CMAKE_MUSA_FLAGS "${CMAKE_MUSA_FLAGS} -fplugin=$ENV{MUSAMAPPING_PATH}/libMusaMapping.so")
  add_subdirectory(musa)
endif()

if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_ILU)
  add_subdirectory(ilu)
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()

cc_library(
  NAME
    kernels
  HDRS
    param.h
    ops_api.h
    xllm_torch_ops.h
  SRCS
    ops_api.cpp
  DEPS
    torch
    $<$<BOOL:${USE_NPU}>:triton_adapter>
    $<$<BOOL:${USE_NPU}>:npu_kernels>
    $<$<BOOL:${USE_NPU}>:xllm_ops>
    $<$<BOOL:${USE_MLU}>:mlu_kernels>
    $<$<BOOL:${USE_MUSA}>:musa_kernels>
    $<$<BOOL:${USE_CUDA}>:cuda_kernels>
    $<$<BOOL:${USE_ILU}>:ilu_kernels>
    $<$<BOOL:${USE_DCU}>:dcu_kernels>
)
