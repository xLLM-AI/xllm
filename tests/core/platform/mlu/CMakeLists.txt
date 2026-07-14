include(cc_test)

cc_test(
  NAME
    mlu_rdma_memory_plan_test
  SRCS
    mlu_rdma_memory_plan_test.cpp
  DEPS
    :mlu_rdma_memory_plan
    GTest::gtest_main
)

cc_test(
  NAME
    mlu_tensor_alloc_test
  SRCS
    mlu_tensor_alloc_test.cpp
  DEPS
    :kv_cache_estimation
    :platform
    GTest::gtest_main
    cnrt
    glog::glog
    torch
)
