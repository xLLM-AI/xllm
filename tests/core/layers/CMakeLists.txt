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

cc_test(
  NAME
    rotary_embedding_util_test
  SRCS
    rotary_embedding_util_test.cpp
  DEPS
    :common_layers
    glog::glog
    torch
    GTest::gtest_main
)

# DeepSeek-V4 prefill CP split arithmetic. Kept outside the platform
# subdirectories: the split math has no accelerator or ProcessGroup dependency,
# so every platform build can verify it.
cc_test(
  NAME
    deepseek_v4_cp_split_test
  SRCS
    deepseek_v4_cp_split_test.cpp
  DEPS
    :parallel_state
    glog::glog
    torch
    GTest::gtest_main
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
