include(cc_test)

add_subdirectory(profile)

# Standalone test for the DP lopsided-batch defer state machine
# (algorithm mirrored from ContinuousScheduler::step()). Depends only on
# absl::time; runs on any platform and without a real Engine so we can
# hit the state transitions in isolation.
cc_test(
  NAME
    lopsided_defer_gate_test
  SRCS
    lopsided_defer_gate_test.cpp
  DEPS
    absl::time
    GTest::gtest_main
)

cc_test(
  NAME
    chunked_prefill_scheduler_test
  SRCS
    chunked_prefill_scheduler_test.cpp
    prefill_only_scheduler_test.cpp
    disagg_pd_chunked_prefill_scheduler_test.cpp
    disagg_pd_scheduler_test.cpp
    continuous_scheduler_test.cpp
    fixed_steps_scheduler_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
target_link_libraries(chunked_prefill_scheduler_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>)
target_link_libraries(chunked_prefill_scheduler_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
