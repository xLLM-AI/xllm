include(cc_test)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

cc_test(
  NAME
    dcp_kernels_test
  SRCS
    dcp_candidate_score_test.cpp
    dcp_correct_attn_out_test.cpp
    dcp_merge_topk_test.cpp
    dcp_localize_topk_test.cpp
  DEPS
    :triton_jit
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

# Opt-in wide-address cases allocate multi-GiB backing storage.
set_tests_properties(
  DcpCandidateScoreTest.WideRowAddresses
  DcpCandidateScoreTest.WideInnerAddresses
  PROPERTIES RUN_SERIAL TRUE
)
