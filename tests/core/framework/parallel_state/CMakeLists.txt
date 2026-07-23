include(cc_test)

if(USE_NPU)
  cc_test(
    NAME
      mapping_npu_test
    SRCS
      mapping_npu_test.cpp
    DEPS
      parallel_state
      absl::synchronization
      absl::time
      GTest::gtest_main
      xllm_atb_layers
      nnopbase
      ascendcl
      atb
      c_sec
      spdlog::spdlog
  )

  cc_test(
    NAME
      npu_dp_ep_padding_test
    SRCS
      npu_dp_ep_padding_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )
  target_link_libraries(npu_dp_ep_padding_test
                        PUBLIC Python::Python
                               ascendcl
                               hccl
                               c_sec
                               nnopbase)

  cc_test(
    NAME
      npu_cp_prefill_plan_test
    SRCS
      npu_cp_prefill_plan_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )

  cc_test(
    NAME
      npu_cp_prefill_plan_cmp_test
    SRCS
      npu_cp_prefill_plan_cmp_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )

  cc_test(
    NAME
      npu_cp_ep_padding_cmp_test
    SRCS
      npu_cp_ep_padding_cmp_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )

  cc_test(
    NAME
      npu_cp_slots_cmp_test
    SRCS
      npu_cp_slots_cmp_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )

  cc_test(
    NAME
      npu_cp_closure_test
    SRCS
      npu_cp_closure_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )

  cc_test(
    NAME
      cp_group_ranks_test
    SRCS
      cp_group_ranks_test.cpp
    DEPS
      parallel_state
      GTest::gtest_main
  )

  cc_test(
    NAME
      npu_cp_ep_padding_plan_test
    SRCS
      npu_cp_ep_padding_plan_test.cpp
    DEPS
      parallel_state
      torch
      absl::synchronization
      absl::time
      GTest::gtest_main
  )
  target_link_libraries(npu_cp_ep_padding_plan_test
                        PUBLIC Python::Python
                               ascendcl
                               hccl
                               c_sec
                               nnopbase)

  cc_test(
    NAME
      npu_cp_capability_test
    SRCS
      npu_cp_capability_test.cpp
    DEPS
      models
      parallel_state
      GTest::gtest_main
  )
  target_link_libraries(npu_cp_capability_test
                        PUBLIC Python::Python
                               ascendcl
                               hccl
                               c_sec
                               nnopbase)
endif()

if(USE_MLU)
  # Add test for reduce_scatter
  # This test must exist individually, because it contains forked processes
  # which does not allow any device init on main process
  cc_test(
    NAME
      parallel_state_test
    SRCS
      parallel_state_test.cpp
    DEPS
      :parallel_state
      GTest::gtest_main
      torch
      glog::glog
  )
endif()
