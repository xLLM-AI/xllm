add_subdirectory(utils)

if(NOT USE_NPU)
  return()
endif()

include(cc_test)

cc_test(
  NAME
    minimax_h3_transformer_test
  CPU_ONLY
  SRCS
    minimax_h3_transformer_test.cpp
  DEPS
    models
    GTest::gtest_main
)

# The test main spawns fresh ranks before the NPU environment initializes.
cc_test(
  NAME
    minimax_h3_attention_npu_test
  SRCS
    minimax_h3_attention_npu_test.cpp
  DEPS
    models
    GTest::gtest
)
set_tests_properties(
  MiniMaxH3AttentionNpuTest.PaddingAndBatchSegmentsPreserveLiveAttention
  MiniMaxH3AttentionNpuTest.ProjectionConstraintsFailAtModelBoundary
  MiniMaxH3AttentionNpuTest.OutputHeadsProjectAllLocalRowsInFp32
  PROPERTIES RUN_SERIAL TRUE TIMEOUT 180
)

cc_test(
  NAME
    minimax_h3_video_vae_test
  SRCS
    minimax_h3_video_vae_test.cpp
  DEPS
    models
    GTest::gtest_main
)
