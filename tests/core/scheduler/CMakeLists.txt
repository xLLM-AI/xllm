include(cc_test)

add_subdirectory(profile)

cc_test(
  NAME
    scheduler_test
  SRCS
    disagg_pd_chunked_prefill_scheduler_test.cpp
    disagg_pd_scheduler_test.cpp
    continuous_scheduler_test.cpp
    fixed_steps_scheduler_test.cpp
    scheduler_policy_test.cpp
  DEPS
    :config
    :scheduler
    :xllm_server
    :block
    GTest::gtest_main
    $<$<BOOL:${USE_NPU}>:nnopbase>
)
target_link_libraries(scheduler_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>)
target_link_libraries(scheduler_test PRIVATE
  "$<LINK_GROUP:RESCAN,xtensor,xllm_server>")
