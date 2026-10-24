include(cc_test)

set(SAMPLER_DEVICE_TEST_SRCS)
if(USE_NPU)
  list(APPEND SAMPLER_DEVICE_TEST_SRCS seeded_sampler_test.cpp)
endif()

cc_test(
  NAME
    sampler_test
  SRCS
    gumbel_sampling_test.cpp
    rejection_sampler_test.cpp
    sampling_params_test.cpp
    ${SAMPLER_DEVICE_TEST_SRCS}
  DEPS
    absl::strings
    GTest::gtest_main
    :config
    :sampler
    :speculative
    glog::glog
    :state_dict
)
target_link_libraries(sampler_test PRIVATE brpc OpenSSL::SSL OpenSSL::Crypto leveldb::leveldb protobuf::libprotobuf)
target_link_libraries(sampler_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:unified_dlog>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
add_dependencies(sampler_test brpc-static)

set(SAMPLER_CONTROLS_DEVICE_TEST_SRCS)
if(USE_CUDA AND NOT USE_MACA)
  list(APPEND SAMPLER_CONTROLS_DEVICE_TEST_SRCS cuda_logits_transform_test.cpp)
endif()

cc_test(
  NAME sampler_controls_test
  NO_NPU_RUNTIME
  SRCS logits_processor_test.cpp sampler_controls_test.cpp ${SAMPLER_CONTROLS_DEVICE_TEST_SRCS}
  DEPS absl::strings GTest::gtest_main :config :sampler :speculative glog::glog :state_dict
)
target_link_libraries(sampler_controls_test PRIVATE
  brpc OpenSSL::SSL OpenSSL::Crypto leveldb::leveldb protobuf::libprotobuf
  Python::Python
  $<$<BOOL:${USE_NPU}>:torch_npu>
  $<$<BOOL:${USE_NPU}>:ascendcl>
  $<$<BOOL:${USE_NPU}>:unified_dlog>
  $<$<BOOL:${USE_NPU}>:hccl>
  $<$<BOOL:${USE_NPU}>:c_sec>
  $<$<BOOL:${USE_NPU}>:nnopbase>)
add_dependencies(sampler_controls_test brpc-static)

cc_test(
  NAME
    sampler_filter_mask_test
  SRCS
    sampler_filter_mask_test.cpp
  DEPS
    :platform
    :sampler
    GTest::gtest_main
)
target_link_libraries(sampler_filter_mask_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:unified_dlog>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)
