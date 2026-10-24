include(cc_test)

add_library(native_library_registry_fixture SHARED
  native_library_registry_fixture.cpp)
target_link_libraries(native_library_registry_fixture PRIVATE
  XLLM::native_extension)
add_dependencies(native_library_registry_fixture xllm_proto)
set_target_properties(native_library_registry_fixture PROPERTIES
  CXX_VISIBILITY_PRESET default
  VISIBILITY_INLINES_HIDDEN OFF)

add_subdirectory(dit)

cc_test(
  NO_NPU_RUNTIME
  NAME model_registry_registration_test
  SRCS model_registry_registration_test.cpp
  DEPS models GTest::gtest_main
)

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

cc_test(
  NO_NPU_RUNTIME
  NAME native_library_integration_test
  SRCS native_library_integration_test.cpp
  DEPS :runtime models GTest::gtest_main
)
if(USE_NPU)
  target_link_libraries(native_library_integration_test PRIVATE
    ascendcl hccl c_sec nnopbase)
endif()
target_compile_definitions(native_library_integration_test PRIVATE
  NATIVE_LIBRARY_FIXTURE="$<TARGET_FILE:native_library_registry_fixture>")
target_link_options(native_library_integration_test PRIVATE -Wl,--export-dynamic)
set_target_properties(native_library_integration_test PROPERTIES ENABLE_EXPORTS ON)
add_dependencies(native_library_integration_test native_library_registry_fixture)
