# vlm_master_test is a standalone manual test binary and has no registered
# CTest target yet.

include(cc_test)

cc_test(
  NO_NPU_RUNTIME
  NAME
    kv_transfer_topology_test
  SRCS
    kv_transfer_topology_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    spawn_worker_protocol_test
  SRCS
    spawn_worker_protocol_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NAME
    heartbeat_callback_registry_test
  SRCS
    heartbeat_callback_registry_test.cpp
  DEPS
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME
    collective_service_test
  SRCS
    collective_service_test.cpp
  DEPS
    :collective_service
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME
    distributed_worker_manager_test
  SRCS
    distributed_worker_manager_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME
    kv_cache_transfer_coordinator_test
  SRCS
    kv_cache_transfer_coordinator_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME
    model_memory_controller_test
  SRCS
    model_memory_controller_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)

if(USE_NPU)
  cc_test(
    NAME
      worker_service_test
    SRCS
      worker_service_test.cpp
    DEPS
      :worker_service
      :runtime
      GTest::gtest_main
  )
  target_link_libraries(worker_service_test
                        PRIVATE
                        torch_npu
                        ascendcl
                        hccl
                        c_sec
                        nnopbase)
  target_link_options(worker_service_test PRIVATE
    "-Wl,--whole-archive"
    "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
    "-Wl,--no-whole-archive")
endif()
