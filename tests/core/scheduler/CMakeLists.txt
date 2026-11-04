include(cc_test)

add_subdirectory(profile)

# Keep notifier synchronization tests independent of device initialization.
cc_test(
  NAME
    scheduler_notifier_test
  NO_NPU_RUNTIME
  SRCS
    scheduler_notifier_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    scheduler_test
  SRCS
    pd_scheduler_test.cpp
    scheduler_metrics_test.cpp
    continuous_scheduler_test.cpp
    scheduler_policy_test.cpp
  DEPS
    :config
    :scheduler
    :scheduler_factory
    :disaggregation
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

cc_test(
  NO_NPU_RUNTIME
  NAME
    dit_failure_delivery_test
  SRCS
    dit_failure_delivery_test.cpp
  DEPS
    :scheduler
    GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(dit_failure_delivery_test PRIVATE nnopbase)
endif()
target_link_options(dit_failure_delivery_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")
