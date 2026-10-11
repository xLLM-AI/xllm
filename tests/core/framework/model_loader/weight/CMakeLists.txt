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
      :model_memory_manager
      :config
      GTest::gtest_main
      glog::glog
  )
endif()

get_property(weight_tests DIRECTORY PROPERTY TESTS)
set_tests_properties(${weight_tests} PROPERTIES TIMEOUT 60)
