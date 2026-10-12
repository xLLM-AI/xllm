# Copyright 2026 The xLLM Authors.
# Licensed under the Apache License, Version 2.0.

include(cc_test)

cc_test(
  NAME
    worker_physical_page_budget_test
  NO_NPU_RUNTIME
  SRCS
    worker_physical_page_budget_test.cpp
  DEPS
    :worker_physical_page_budget
    GTest::gtest_main
)

# Byte-range rejection needs no device storage. Keep death-test children free
# of Python and NPU runtime initialization.
cc_test(
  NAME
    mapped_virtual_memory_tensor_view_test
  NO_NPU_RUNTIME
  SRCS
    mapped_virtual_memory_tensor_view_test.cpp
  DEPS
    GTest::gtest_main
    glog::glog
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
      :global_virtual_memory_region
      GTest::gtest_main
      glog::glog
  )

endif()

# Bound concurrency regressions and device initialization failures in CTest.
get_property(allocator_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${allocator_tests} PROPERTIES TIMEOUT 60)

if(USE_NPU)
  # Each case owns the device-0 page pool; isolate it from other CTest processes.
  set(virtual_memory_tests ${allocator_tests})
  list(FILTER virtual_memory_tests INCLUDE REGEX "^VirtualMemoryTest\\.")
  set_tests_properties(${virtual_memory_tests} PROPERTIES RUN_SERIAL TRUE)
endif()
