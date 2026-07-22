include(cc_test)

get_filename_component(_xllm_build_import_root
  "${CMAKE_LIBRARY_OUTPUT_DIRECTORY}" DIRECTORY)

# Add test for Qwen3.5 MLU layers (gated delta net operators, GDN layer, and
# hybrid attention layer).
cc_test(
  NAME
    qwen3_5_layer_test
  SRCS
    qwen3_5_gated_delta_net_ops_test.cpp
    qwen3_5_gated_delta_net_test.cpp
    qwen3_5_attention_test.cpp
    qwen3_next_rms_norm_test.cpp
    tests_utils.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
  ENVIRONMENT
    "PYTHONPATH=${_xllm_build_import_root}:$ENV{PYTHONPATH}"
)

# Add test for MLU layers
cc_test(
  NAME
    layer_test
  SRCS
    dense_mlp_test.cpp
    tests_utils.cpp
    deepseek_v32_sp_utils_test.cpp
    dp_utils_test.cpp
    indexer_test.cpp
    mla_test.cpp
    deepseek_v2_decoder_layer_test.cpp
    qwen2_attention_test.cpp
    qwen2_vision_attention_test.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_attention_layer_test
  SRCS
    deepseek_v4_attention_test.cpp
    deepseek_v4_compressor_test.cpp
    deepseek_v4_indexer_test.cpp
    deepseek_v4_ref_utils.cpp
    tests_utils.cpp
  DEPS
    :mlu_layers
    :common_layers
    :block
    :util
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_hyper_connection_test
  SRCS
    deepseek_v4_hyper_connection_test.cpp
    tests_utils.cpp
  DEPS
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_decoder_layer_test
  SRCS
    deepseek_v4_decoder_layer_test.cpp
    deepseek_v4_ref_utils.cpp
    tests_utils.cpp
  DEPS
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

cc_test(
  NAME
    deepseek_v4_dsa_metadata_builder_test
  SRCS
    dsa_metadata_builder_mlu_test.cpp
  DEPS
    :mlu_layers
    :common_layers
    glog::glog
    torch
    GTest::gtest_main
)

# MoE Gate unit test for MLU (uses layers/mlu/moe_gate)
cc_test(
  NAME
    moe_layer_test
  SRCS
    moe_gate_test.cpp
    fused_moe_test.cpp
    moe_weight_loader_helper_test.cpp
    deepseek_v2_sparse_moe_block_test.cpp
    deepseek_v4_sparse_moe_block_test.cpp
    tests_utils.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    glog::glog
    torch
    GTest::gtest_main
)

# Add test for DeepEP
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    deep_ep_test
  SRCS
    deep_ep_test.cpp
    tests_utils.cpp
  DEPS
    :mlu_layers
    :common_layers
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

# Add test for FusedMoE All2All path
# This test must exist individually, because it contains forked processes
#  which does not allow any device init on main process
cc_test(
  NAME
    fused_moe_all2all_test
  SRCS
    fused_moe_all2all_test.cpp
    tests_utils.cpp
  DEPS
    :config
    :mlu_layers
    :common_layers
    :block
    :parallel_state
    :model
    :model_context
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

cc_test(
  NAME
    deepseek_v2_attention_multi_device_test
  SRCS
    deepseek_v2_attention_multi_device_test.cpp
    tests_utils.cpp
  DEPS
    :config
    :block
    :mlu_layers
    :parallel_state
    :state_dict
    GTest::gtest_main
    torch
    glog::glog
)

# Cross-layer DSA top-k relay state is pure CPU logic, but the protocol belongs
# to the MLU decoder/attention boundary.
cc_test(
  NAME
    dsa_topk_relay_test
  SRCS
    dsa_topk_relay_test.cpp
  DEPS
    :mlu_layers
    GTest::gtest_main
    torch
    glog::glog
)
