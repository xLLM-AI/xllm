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

if(USE_NPU)
  cc_test(
    NAME
      virtual_memory_test
    SRCS
      virtual_memory_test.cpp
    DEPS
      :virtual_memory
      :virtual_memory_torch
      :global_memory_region
      GTest::gtest_main
      glog::glog
  )

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
