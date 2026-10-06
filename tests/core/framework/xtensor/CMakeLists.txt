# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    weight_transfer_segments_test
  NO_NPU_RUNTIME
  SRCS
    weight_transfer_segments_test.cpp
  DEPS
    GTest::gtest_main
    glog::glog
)

cc_test(
  NAME
    virt_page_test
  NO_NPU_RUNTIME
  SRCS
    virt_page_test.cpp
    ${PROJECT_SOURCE_DIR}/xllm/core/framework/xtensor/virt_page.cpp
  DEPS
    GTest::gtest_main
    glog::glog
)

if(USE_NPU)
  # Keep validation/death tests independent of device runtime initialization.
  foreach(test_name IN ITEMS page_allocator_test xtensor_bootstrap_test)
    cc_test(
      NAME ${test_name}
      NO_NPU_RUNTIME
      SRCS ${test_name}.cpp
      DEPS :xtensor :config GTest::gtest_main glog::glog
    )
  endforeach()

  foreach(test_name IN ITEMS xtensor_test xtensor_weight_allocator_test)
    cc_test(
      NAME ${test_name}
      SRCS ${test_name}.cpp
      DEPS :xtensor :config GTest::gtest_main glog::glog
    )
  endforeach()
endif()

# Bound concurrency regressions and device initialization failures in CTest.
get_property(xtensor_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${xtensor_tests} PROPERTIES TIMEOUT 60)

if(USE_NPU)
  # Threadsafe death tests restart the NPU runtime for each assertion.
  set_tests_properties(
    XTensorTest.TensorViewsRespectByteBoundsAndAlignment
    PROPERTIES TIMEOUT 120
  )
endif()
