include(cc_test)

# dual_parallel_args_test verifies the atomic Mode discriminator on
# DualParallelArgs. Kept as a source file for reference / hand-run under a
# CPU environment; not wired into CMake because parallel_args.h under
# USE_NPU transitively includes xllm_atb_layers/atb which needs the full
# ATB toolchain, and stubbing all of that just for a header-only test
# yields more surface area than value. The lopsided_defer_gate_test
# (tests/core/scheduler/) and atomic_pool_handle_test (tests/core/runtime/)
# cover the runtime CP<->DP switch's higher-value invariants.

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
