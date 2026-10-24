include(cc_test)

foreach(_fixture valid mismatch backend_mismatch missing_descriptor host_dependent)
  add_library(native_library_${_fixture}_fixture SHARED
    native_library_fixture.cpp)
  target_link_libraries(native_library_${_fixture}_fixture PRIVATE
    XLLM::native_extension)
endforeach()
target_compile_definitions(native_library_mismatch_fixture PRIVATE
  TEST_NATIVE_MISMATCH)
target_compile_definitions(native_library_backend_mismatch_fixture PRIVATE
  TEST_NATIVE_BACKEND_MISMATCH)
target_compile_definitions(native_library_missing_descriptor_fixture PRIVATE
  TEST_NATIVE_NO_DESCRIPTOR)
target_link_libraries(native_library_missing_descriptor_fixture PRIVATE
  native_library_valid_fixture)
target_compile_definitions(native_library_host_dependent_fixture PRIVATE
  TEST_NATIVE_HOST_DEPENDENCY)

add_library(native_library_host_fixture SHARED native_library_host_fixture.cpp)
target_link_libraries(native_library_host_fixture PRIVATE :native_library)
# Plain add_library (outside cc_library) has no COMMON_INCLUDE_DIRS; the
# fixture includes xllm headers through the linked native_library target.
target_include_directories(native_library_host_fixture PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../xllm)
xllm_export_native_host(native_library_host_fixture)

cc_test(
  NO_NPU_RUNTIME
  NAME native_library_test
  SRCS native_library_test.cpp
  DEPS :native_library GTest::gtest_main
)
xllm_export_native_host(native_library_test)
target_compile_definitions(native_library_test PRIVATE
  NATIVE_VALID_FIXTURE="$<TARGET_FILE:native_library_valid_fixture>"
  NATIVE_MISMATCH_FIXTURE="$<TARGET_FILE:native_library_mismatch_fixture>"
  NATIVE_BACKEND_MISMATCH_FIXTURE="$<TARGET_FILE:native_library_backend_mismatch_fixture>"
  NATIVE_NO_DESCRIPTOR_FIXTURE="$<TARGET_FILE:native_library_missing_descriptor_fixture>")
add_dependencies(native_library_test native_library_valid_fixture
  native_library_mismatch_fixture native_library_backend_mismatch_fixture
  native_library_missing_descriptor_fixture)

# This executable must not link the host loader: the fixture models a Python
# extension imported with RTLD_LOCAL and owns the symbols needed by the plugin.
cc_test(
  NO_NPU_RUNTIME
  NAME native_library_local_host_test
  SRCS native_library_local_host_test.cpp
  DEPS GTest::gtest_main ${CMAKE_DL_LIBS}
)
target_compile_definitions(native_library_local_host_test PRIVATE
  NATIVE_HOST_FIXTURE="$<TARGET_FILE:native_library_host_fixture>"
  NATIVE_HOST_DEPENDENT_FIXTURE="$<TARGET_FILE:native_library_host_dependent_fixture>")
add_dependencies(native_library_local_host_test native_library_host_fixture
  native_library_host_dependent_fixture)

cc_test(
  NAME
    common_test
  SRCS
    flash_comm1_context_test.cpp
    metrics_test.cpp
    options_test.cpp
    rate_limiter_test.cpp
  DEPS
    :config
    common
    absl::synchronization
    absl::time
    GTest::gtest_main
    gflags::gflags
    glog::glog
    torch
)
