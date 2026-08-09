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
  target_link_libraries(model_registry_test
                        PUBLIC Python::Python
                               ascendcl
                               hccl
                               c_sec
                               nnopbase)
endif()
