# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    worker_page_budget_test
  SRCS
    worker_page_budget_test.cpp
  DEPS
    :worker_page_budget
    GTest::gtest_main
)

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

if(USE_NPU)
  # Keep validation/death tests independent of device runtime initialization.
  foreach(test_name IN ITEMS model_page_allocator_test model_memory_bootstrap_test)
    cc_test(
      NAME ${test_name}
      NO_NPU_RUNTIME
      SRCS ${test_name}.cpp
      DEPS :model_memory_manager :config GTest::gtest_main glog::glog
    )
  endforeach()

  foreach(test_name IN ITEMS virtual_memory_test model_weight_allocator_test)
    cc_test(
      NAME ${test_name}
      SRCS ${test_name}.cpp
      DEPS :model_memory_manager :config GTest::gtest_main glog::glog
    )
  endforeach()
endif()

# Bound concurrency regressions and device initialization failures in CTest.
get_property(allocator_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${allocator_tests} PROPERTIES TIMEOUT 60)

if(USE_NPU)
  # Threadsafe death tests restart the NPU runtime for each assertion.
  set_tests_properties(
    VirtualMemoryTest.TensorViewsRespectByteBoundsAndAlignment
    PROPERTIES TIMEOUT 120
  )
endif()
