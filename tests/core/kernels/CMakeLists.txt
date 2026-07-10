if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_NPU)
  add_subdirectory(npu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()
