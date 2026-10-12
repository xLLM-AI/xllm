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

if(USE_NPU)
  cc_test(
    NAME
      model_weight_allocator_test
    SRCS
      model_weight_allocator_test.cpp
    DEPS
      :model_weight_store
      :worker_memory_resources
      :config
      GTest::gtest_main
      glog::glog
  )
endif()

get_property(weight_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${weight_tests} PROPERTIES TIMEOUT 60)

# Fatal checks run in child processes; isolate them from runtime startup load.
set_tests_properties(
  WeightTransferSegmentsDeathTest.RejectsInvalidPageGeometry
  PROPERTIES RUN_SERIAL TRUE
)

if(USE_NPU)
  # These cases initialize the NPU runtime and own the device-0 page pool.
  set(model_weight_allocator_tests ${weight_tests})
  list(FILTER model_weight_allocator_tests INCLUDE
    REGEX "^ModelWeightAllocatorTest\\.")
  set_tests_properties(${model_weight_allocator_tests} PROPERTIES RUN_SERIAL TRUE)
endif()
