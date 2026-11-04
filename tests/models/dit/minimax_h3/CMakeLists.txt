# Copyright 2026 The xLLM Authors. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# https://github.com/xLLM-AI/xllm/blob/main/LICENSE

include(cc_test)

cc_test(
  NAME minimax_h3_contract_test
  CPU_ONLY
  SRCS minimax_h3_qualified_contract_test.cpp
  DEPS :model_loader state_dict torch GTest::gtest_main
)

cc_test(
  NAME minimax_h3_math_test
  CPU_ONLY
  SRCS minimax_h3_qualified_math_test.cpp
  DEPS :model_loader state_dict torch GTest::gtest_main
)

cc_test(
  NAME minimax_h3_phase_timing_test
  CPU_ONLY
  SRCS minimax_h3_phase_timing_test.cpp
  DEPS glog::glog nlohmann_json::nlohmann_json GTest::gtest_main
)

cc_test(
  NAME minimax_h3_tp_test
  CPU_ONLY
  SRCS minimax_h3_qualified_tp_test.cpp
  DEPS :parallel_state torch GTest::gtest_main
)

cc_test(
  NAME minimax_h3_backend_selection_test
  CPU_ONLY
  SRCS minimax_h3_backend_selection_test.cpp
  DEPS :config proto::xllm_proto torch GTest::gtest_main
)
