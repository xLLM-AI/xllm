include(cc_test)

# Host-buildable: exercises the FlashComm1 thread_local context RAII and the
# device-free shard_dim0_padded sharding math. No device / process group.
cc_test(
  NAME
    flash_comm1_context_test
  SRCS
    flash_comm1_context_test.cpp
  DEPS
    parallel_state
    GTest::gtest_main
    torch
    glog::glog
)

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
