include(cc_library)

add_subdirectory(sentencepiece)
add_subdirectory(minja)
add_subdirectory(brpc)
add_subdirectory(smhasher/src)
add_subdirectory(cpprestsdk)
set(CPPREST_INCLUDE_DIR ${CMAKE_CURRENT_SOURCE_DIR}/cpprestsdk/Release/include)
set(CPPREST_LIB ${CMAKE_BINARY_DIR}/third_party/cpprestsdk/Release/Binaries/libcpprest.a)

# GCC 12+ strict check workaround for cpprestsdk
if(CMAKE_CXX_COMPILER_ID STREQUAL "GNU" AND 
   CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 12)
    if(TARGET cpprest)
        target_compile_options(cpprest PRIVATE -Wno-restrict -Wno-attributes)
        message(STATUS "Applied GCC 12+ warning suppressions for cpprestsdk")
    endif()
endif()

add_subdirectory(etcd_cpp_apiv3)
add_subdirectory(xxHash)
add_subdirectory(spdlog)

if(USE_NPU)
  add_subdirectory(torch_npu_ops)
endif()
add_subdirectory(Mooncake)

target_include_directories(mooncake_store PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/Mooncake/mooncake-transfer-engine/include
)

find_package(yalantinglibs CONFIG REQUIRED)
if(TARGET ascend_transport)
  target_link_libraries(ascend_transport PUBLIC yalantinglibs::yalantinglibs)
endif()

if(USE_ILU)
  if(TARGET cpprest)
      set_target_properties(cpprest PROPERTIES
          CXX_STANDARD 20
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS OFF
      )
  endif()
  if(TARGET transfer_engine)
    target_compile_options(transfer_engine PRIVATE -std=c++20)
    set_target_properties(transfer_engine PROPERTIES
        CXX_STANDARD 20
        CXX_STANDARD_REQUIRED ON
    )
    message(STATUS "Set C++20 for transfer_engine target")
  endif()
  if(TARGET SMHasherSupport)
      set_target_properties(SMHasherSupport PROPERTIES
          CXX_STANDARD 11
          CXX_STANDARD_REQUIRED ON
          CXX_EXTENSIONS OFF
      )
      message(STATUS "SMHasherSupport target found and configured")
  else()
      message(WARNING "SMHasherSupport target not found after adding smhasher")
  endif()
endif()

## The order of static libraries is important in the CMake build system because the linker processes them strictly from left to right. 
## -> The linker scans each library only once and extracts object files solely to resolve currently undefined symbols.
## -> In modern CMake, the best practice is to explicitly declare a target's dependencies, allowing CMake to automatically determine the correct linking order. 
## -> The following code is added to prevent symbol resolution errors when xllm links against mooncake-related static libraries.
target_link_libraries(mooncake_store PUBLIC transfer_engine cachelib_memory_allocator)
target_link_libraries(cachelib_memory_allocator PRIVATE Folly::folly glog::glog)
