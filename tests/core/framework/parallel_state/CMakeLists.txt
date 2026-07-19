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
      mega_moe_comm_resource_test
    SRCS
      mega_moe_comm_resource_test.cpp
    DEPS
      mega_moe_comm_resource
      torch
      GTest::gtest_main
      hccl
      hcomm
  )
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
