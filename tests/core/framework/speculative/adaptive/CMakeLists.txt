include(cc_test)

cc_test(
  NAME dspark_dp_coordinator_test
  NO_NPU_RUNTIME
  SRCS
    dspark_dp_coordinator_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/dspark_dp_coordinator.cpp"
  DEPS GTest::gtest_main
)

cc_test(
  NAME
    draft_step_selector_test
  SRCS
    draft_step_selector_test.cpp
  DEPS
    :speculative
    GTest::gtest_main
)

cc_test(
  NAME spec_startup_calibration_test
  NO_NPU_RUNTIME
  SRCS
    spec_startup_calibration_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_cost_profile_store.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/verify_budget_planner.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_startup_calibration.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_cost_model.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_window_collector.cpp"
  DEPS
    nlohmann_json::nlohmann_json
    GTest::gtest_main
)

cc_test(
  NAME spec_window_collector_test
  NO_NPU_RUNTIME
  SRCS
    spec_window_collector_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/verify_budget_planner.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_window_collector.cpp"
  DEPS GTest::gtest_main
)

cc_test(
  NAME spec_cost_model_test
  NO_NPU_RUNTIME
  SRCS
    spec_cost_model_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_cost_model.cpp"
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/verify_budget_planner.cpp"
  DEPS
    nlohmann_json::nlohmann_json
    GTest::gtest_main
)

cc_test(
  NAME spec_cost_profile_store_test
  NO_NPU_RUNTIME
  SRCS
    spec_cost_profile_store_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/spec_cost_profile_store.cpp"
  DEPS GTest::gtest_main
)

cc_test(
  NAME verify_budget_planner_test
  NO_NPU_RUNTIME
  SRCS
    verify_budget_planner_test.cpp
    "${PROJECT_SOURCE_DIR}/xllm/core/framework/speculative/adaptive/verify_budget_planner.cpp"
  DEPS GTest::gtest_main
)
