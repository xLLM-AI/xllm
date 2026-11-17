include(cc_test)

cc_test(
  NO_NPU_RUNTIME
  NAME
    shm_channel_test
  SRCS
    shm_channel_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(shm_channel_test PRIVATE nnopbase)
endif()
target_link_options(shm_channel_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NO_NPU_RUNTIME
  NAME
    comm_channel_test
  SRCS
    comm_channel_test.cpp
  DEPS
    :distributed_runtime
    GTest::gtest_main
)

cc_test(
  NO_NPU_RUNTIME
  NAME forward_shared_memory_output_test
  SRCS forward_shared_memory_output_test.cpp
  DEPS :runtime GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(forward_shared_memory_output_test PRIVATE nnopbase)
endif()
target_link_options(forward_shared_memory_output_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NO_NPU_RUNTIME
  NAME forward_shared_memory_input_test
  SRCS forward_shared_memory_input_test.cpp
  DEPS :runtime GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(forward_shared_memory_input_test PRIVATE nnopbase)
endif()
target_link_options(forward_shared_memory_input_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")

cc_test(
  NAME transport_codec_test
  SRCS transport_codec_test.cpp
  DEPS :runtime :worker_model_codec GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(transport_codec_test PRIVATE nnopbase)
endif()
target_link_options(transport_codec_test PRIVATE
  "-Wl,--whole-archive"
  "${CMAKE_BINARY_DIR}/third_party/spdlog/libspdlog.a"
  "-Wl,--no-whole-archive")
