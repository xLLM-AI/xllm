include(cc_library)

add_subdirectory(sentencepiece)
add_subdirectory(minja)

set(_XLLM_BRPC_GIT_DIR_WAS_SET FALSE)
set(_XLLM_BRPC_GIT_WORK_TREE_WAS_SET FALSE)
if(DEFINED ENV{GIT_DIR})
  set(_XLLM_BRPC_GIT_DIR_WAS_SET TRUE)
  set(_XLLM_BRPC_SAVED_GIT_DIR "$ENV{GIT_DIR}")
endif()
if(DEFINED ENV{GIT_WORK_TREE})
  set(_XLLM_BRPC_GIT_WORK_TREE_WAS_SET TRUE)
  set(_XLLM_BRPC_SAVED_GIT_WORK_TREE "$ENV{GIT_WORK_TREE}")
endif()

set(_XLLM_BRPC_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/brpc")
set(_XLLM_BRPC_GIT_DIR "")
set(_XLLM_BRPC_GIT_RESULT 1)
unset(ENV{GIT_DIR})
unset(ENV{GIT_WORK_TREE})
if(EXISTS "${_XLLM_BRPC_SOURCE_DIR}/.git")
  execute_process(
    COMMAND git -C "${_XLLM_BRPC_SOURCE_DIR}" rev-parse --absolute-git-dir
    OUTPUT_VARIABLE _XLLM_BRPC_GIT_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
    RESULT_VARIABLE _XLLM_BRPC_GIT_RESULT
    ERROR_QUIET
  )
endif()

if(_XLLM_BRPC_GIT_RESULT EQUAL 0 AND _XLLM_BRPC_GIT_DIR)
  set(ENV{GIT_DIR} "${_XLLM_BRPC_GIT_DIR}")
  set(ENV{GIT_WORK_TREE} "${_XLLM_BRPC_SOURCE_DIR}")
else()
  set(ENV{GIT_DIR} "${CMAKE_CURRENT_BINARY_DIR}/brpc-no-git")
  unset(ENV{GIT_WORK_TREE})
endif()

add_subdirectory(brpc)

if(_XLLM_BRPC_GIT_DIR_WAS_SET)
  set(ENV{GIT_DIR} "${_XLLM_BRPC_SAVED_GIT_DIR}")
else()
  unset(ENV{GIT_DIR})
endif()
if(_XLLM_BRPC_GIT_WORK_TREE_WAS_SET)
  set(ENV{GIT_WORK_TREE} "${_XLLM_BRPC_SAVED_GIT_WORK_TREE}")
else()
  unset(ENV{GIT_WORK_TREE})
endif()

add_subdirectory(smhasher/src)
add_subdirectory(cpprestsdk)
if(USE_MUSA AND TARGET cpprest)
  # mcc_wrapper diagnoses signed conversions in cpprestsdk headers more
  # aggressively than the host compiler. Keep the suppression local to the
  # third-party target instead of muting warnings for the whole build.
  target_compile_options(cpprest PRIVATE
    -Wno-sign-conversion
    -Wno-error=sign-conversion
  )
endif()
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
