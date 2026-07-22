include(cc_test)

# Backend-independent DSA sharing policy. Keep this target outside platform
# subdirectories so CUDA, MLU, NPU, and DCU test builds all register it.
cc_test(
  NAME
    dsa_topk_share_plan_test
  SRCS
    dsa_topk_share_plan_test.cpp
  DEPS
    GTest::gtest_main
    glog::glog
)

if(USE_CUDA)
  add_subdirectory(cuda)
endif()

if(USE_MLU)
  add_subdirectory(mlu)
endif()

if(USE_NPU)
  add_subdirectory(npu_torch)
endif()

if(USE_DCU)
  add_subdirectory(dcu)
endif()
