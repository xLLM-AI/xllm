include(cc_test)

cc_test(
  NAME
    npu_vendor_paths_test
  NO_NPU_RUNTIME
  SRCS
    npu_vendor_paths_test.cpp
  DEPS
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME
      npu_runtime_test
    SRCS
      npu_runtime_test.cpp
    DEPS
      :platform_npu
      :xllm_ops
      torch
      torch_npu
      GTest::gtest_main
      glog::glog
      pybind11::embed
  )

  # A separate process verifies startup before the shared test runtime exists.
  cc_test(
    NAME
      npu_startup_test
    NO_NPU_RUNTIME
    SRCS
      npu_startup_test.cpp
    DEPS
      :npu_startup
      :platform_npu
      torch
      torch_npu
      GTest::gtest_main
      glog::glog
      pybind11::embed
  )

  cc_test(
    NAME
      npu_torch_profiler_test
    SRCS
      npu_torch_profiler_test.cpp
      ${PROJECT_SOURCE_DIR}/xllm/core/platform/npu/npu_torch_profiler.cpp
    DEPS
      GTest::gtest_main
      glog::glog
      pybind11::embed
  )

  cc_test(
    NAME
      npu_profiler_test
    SRCS
      npu_profiler_test.cpp
      ${PROJECT_SOURCE_DIR}/xllm/core/platform/npu/npu_profiler.cpp
    DEPS
      GTest::gtest_main
      glog::glog
  )
endif()

if(USE_NPU OR USE_CUDA OR USE_ILU OR USE_DCU)
  cc_test(
    NAME
      multi_platform_vmm_test
    SRCS
      shared_vmm_allocator_test.cpp
    DEPS
      :platform
      :config
      GTest::gtest_main
      glog::glog
  )
  if(USE_DCU)
    set_target_properties(multi_platform_vmm_test PROPERTIES LINKER_LANGUAGE HIP)
  endif()
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()
