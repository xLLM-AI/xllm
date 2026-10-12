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
      virtual_kv_cache_memory_registry_test
    SRCS
      virtual_kv_cache_memory_registry_test.cpp
    DEPS
      :virtual_memory_kv_cache_storage
      GTest::gtest_main
      glog::glog
  )
  # Each process initializes the NPU runtime and owns the device-0 page pool.
  set_tests_properties(
    VirtualKVCacheMemoryRegistryTest.ModelsKeepIndependentMappingsAndMetadata
    VirtualKVCacheMemoryRegistryTest.ShortageDoesNotPartiallyMapAnotherModel
    VirtualKVCacheMemoryRegistryTest.TensorAllocatorUsesItsInjectedRegistry
    PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
  )
endif()

# Page policy and injected backend regressions do not initialize a device.
cc_test(
  NO_NPU_RUNTIME
  NAME
    virtual_kv_cache_page_allocator_impl_test
  SRCS
    virtual_kv_cache_page_allocator_impl_test.cpp
  DEPS
    :virtual_kv_cache_page_allocator_impl
    GTest::gtest_main
    glog::glog
)
