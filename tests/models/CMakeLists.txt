include(cc_test)

add_subdirectory(dit)

# OLMo2 configuration loading is host-only and available on every backend.
cc_test(
  NAME
    olmo2_args_test
  NO_NPU_RUNTIME
  SRCS
    olmo2_args_test.cpp
  DEPS
    :model_loader
    GTest::gtest_main
  DATA
    fixtures/olmo2/config.json
)
target_compile_definitions(olmo2_args_test PRIVATE
  OLMO2_CONFIG_PATH="${CMAKE_CURRENT_BINARY_DIR}/fixtures/olmo2/config.json"
)

if(USE_NPU)
  cc_test(
    NAME
      model_registry_test
    SRCS
      model_registry_test.cpp
    DEPS
      models
      GTest::gtest_main
  )
  set(npu_model_test_libs ascendcl hccl c_sec nnopbase)
  target_link_libraries(model_registry_test PUBLIC ${npu_model_test_libs})

  cc_test(
    NAME
      py_causal_lm_test
    SRCS
      py_causal_lm_test.cpp
    DEPS
      models
      GTest::gtest_main
  )
  target_link_libraries(py_causal_lm_test PUBLIC Python::Python ${npu_model_test_libs})
endif()
