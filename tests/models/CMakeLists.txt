include(cc_test)

add_subdirectory(dit)

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
