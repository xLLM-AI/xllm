include(cc_test)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

cc_test(
  NAME
    triton_jit_ops_test
  SRCS
    fused_gdn_gating_test.cpp
    causal_conv1d_update_decode_test.cpp
    fused_recurrent_gated_delta_rule_packed_decode_test.cpp
    causal_conv1d_fn_test.cpp
    fused_recurrent_gated_delta_rule_test.cpp
    chunk_gated_delta_rule_regression_test.cpp
    fused_recurrent_gated_delta_rule_beta_regression_test.cpp
    fused_sigmoid_update_regression_test.cpp
  DEPS
    :triton_jit
    :mlu_kernels
    torch
    glog::glog
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)
