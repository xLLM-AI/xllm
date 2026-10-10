include(cc_test)

cc_test(
  NAME
    kv_cache_test
  SRCS
    kv_cache_test.cpp
  DEPS
    :kv_cache
    GTest::gtest_main
)
target_link_libraries(kv_cache_test
                      PUBLIC
                      Python::Python
                      $<$<BOOL:${USE_NPU}>:ascendcl>
                      $<$<BOOL:${USE_NPU}>:hccl>
                      $<$<BOOL:${USE_NPU}>:c_sec>
                      $<$<BOOL:${USE_NPU}>:nnopbase>)

if(USE_NPU)
  # Exercise the concrete KV storage without distributed runtime initialization.
  cc_test(
    NAME
      kv_cache_memory_registry_test
    SRCS
      kv_cache_memory_registry_test.cpp
    DEPS
      :virtual_memory_kv_cache_storage
      GTest::gtest_main
      glog::glog
  )
  set_tests_properties(
    KVCacheMemoryRegistryTest.ModelsKeepIndependentMappingsAndMetadata
    KVCacheMemoryRegistryTest.ShortageDoesNotPartiallyMapAnotherModel
    KVCacheMemoryRegistryTest.TensorAllocatorUsesItsInjectedRegistry
    PROPERTIES TIMEOUT 60
  )
endif()
