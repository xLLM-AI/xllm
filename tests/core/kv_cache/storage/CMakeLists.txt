include(cc_test)

cc_test(
  NAME
    linear_state_restore_test
  CPU_ONLY
  SRCS
    linear_state_restore_test.cpp
  DEPS
    :linear_state_restore
    :kv_cache_block
    :static_kv_cache_tensor_allocator
    GTest::gtest_main
)
target_link_libraries(linear_state_restore_test PUBLIC Python::Python)

cc_test(
  CPU_ONLY
  NAME
    static_kv_cache_tensor_allocator_test
  SRCS
    static_kv_cache_tensor_allocator_test.cpp
  DEPS
    :static_kv_cache_tensor_allocator
    GTest::gtest_main
    glog::glog
)


cc_test(
  DISCOVER_TESTS
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
  cc_test(
    NAME
      static_kv_cache_tensor_allocator_npu_test
    SRCS
      static_kv_cache_tensor_allocator_npu_test.cpp
    DEPS
      :static_kv_cache_tensor_allocator
      GTest::gtest_main
  )
  set_tests_properties(
    StaticKVCacheTensorAllocatorNpuTest.KeepsDenseNdViewsInOnePhysicalArena
    PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
  )

  # Exercise the concrete KV storage without distributed runtime initialization.
  cc_test(
    NAME
      virtual_kv_cache_memory_registry_test
    SRCS
      virtual_kv_cache_memory_registry_test.cpp
    DEPS
      :kv_cache
      :virtual_memory_kv_cache_storage
      GTest::gtest_main
      glog::glog
  )
  target_link_libraries(virtual_kv_cache_memory_registry_test
                        PUBLIC
                        Python::Python
                        $<$<BOOL:${USE_NPU}>:ascendcl>
                        $<$<BOOL:${USE_NPU}>:hccl>
                        $<$<BOOL:${USE_NPU}>:c_sec>
                        $<$<BOOL:${USE_NPU}>:nnopbase>)
  # Each process initializes the NPU runtime and owns the device-0 page pool.
  set_tests_properties(
    VirtualKVCacheMemoryRegistryTest.PrepareRejectsDuplicateAndInvalidOffsetsWithoutAllocation
    VirtualKVCacheMemoryRegistryTest.PrepareRejectsExistingMappingsWithoutChangingOwnership
    VirtualKVCacheMemoryRegistryTest.ModelsKeepIndependentMappingsAndMetadata
    VirtualKVCacheMemoryRegistryTest.ShortageDoesNotPartiallyMapAnotherModel
    VirtualKVCacheMemoryRegistryTest.TensorAllocatorUsesItsInjectedRegistry
    VirtualKVCacheMemoryRegistryTest.TransferRejectsBlocksThatDoNotDivideOnePage
    VirtualMLAKVCacheMemoryRegistryTest.LatentAndRopeBoundariesKeepIndependentMappings
    VirtualMLAKVCacheMemoryRegistryTest.TransferOffsetsUseIndependentRoleBlockSizes
    VirtualMLAKVCacheMemoryRegistryTest.ShortageDoesNotPartiallyMapLatentRopeOrAnotherLayer
    VirtualMLAKVCacheMemoryRegistryTest.IncompleteLogicalPagesDoNotChangeExistingMappings
    VirtualIndexedKVCacheMemoryRegistryTest.SparseIndexRegionsKeepLayerIdsAndGroupBoundaries
    VirtualIndexedKVCacheMemoryRegistryTest.TransferOffsetsPreserveSparseIndexOwnership
    VirtualIndexedKVCacheMemoryRegistryTest.ShortageKeepsKeyValueAndIndexMappingsAtomic
    VirtualIndexedKVCacheMemoryRegistryTest.TensorAllocatorSkipsSharedIndexLayers
    VirtualIndexedKVCacheMemoryRegistryTest.DefaultIndexOwnershipMapsAllLayersAndRejectsTransfer
    VirtualIndexedKVCacheMemoryRegistryTest.UnownedIndexWidthDoesNotChangeKVPageGeometry
    VirtualIndexedKVCacheMemoryRegistryTest.AllocateKVCacheUsesSparseVirtualIndexRegions
    VirtualMLAKVCacheMemoryRegistryTest.MappingGroupUsesCommonGeometryForEveryMember
    VirtualMLAKVCacheMemoryRegistryTest.MappingGroupShortageRollsBackEveryMember
    VirtualKVCacheMemoryRegistryTest.MappingGroupsRejectMissingDuplicateAndAlreadyMappedMembers
    VirtualIndexedKVCacheMemoryRegistryTest.SparseKeyValueRegionsPreserveFullAttentionLayerIds
    VirtualIndexedKVCacheMemoryRegistryTest.TensorAllocatorSkipsLinearAttentionLayers
    VirtualIndexedKVCacheMemoryRegistryTest.TensorAllocatorRestrictsIndexToFullAttentionLayers
    PROPERTIES TIMEOUT 60 RUN_SERIAL TRUE
  )
endif()

# Page policy and injected backend regressions do not initialize a device.
cc_test(
  NO_NPU_RUNTIME
  NAME
    vmm_kv_page_allocator_impl_test
  SRCS
    vmm_kv_page_allocator_impl_test.cpp
  DEPS
    :vmm_kv_page_allocator_impl
    :virtual_kv_cache_page_layout
    GTest::gtest_main
    glog::glog
)
