cmake_minimum_required(VERSION 3.26)
set_property(GLOBAL PROPERTY USE_FOLDERS ON)

option(USE_NPU "Enable NPU support" OFF)
option(USE_MLU "Enable MLU support" OFF)
option(USE_ILU "Enable ILU support" OFF)
option(USE_CUDA "Enable CUDA support" OFF)
option(USE_MUSA "Enable MUSA support" OFF)
add_compile_definitions(YLT_ENABLE_IBV)
add_definitions(-DYLT_ENABLE_IBV)
set(YLT_ENABLE_IBV ON)

if(DEVICE_ARCH STREQUAL "ARM")
    set(CMAKE_SYSTEM_PROCESSOR aarch64)
    set(RUST_TARGET aarch64-unknown-linux-gnu)
endif()

if(USE_NPU)
  # Override Mooncake option for mooncake transfer engine
  # CANN 8.5+ migration: ascend_direct_transport replaces ascend_transport
  set(USE_ASCEND_DIRECT ON CACHE BOOL "Enable ADXL engine for Ascend NPU" FORCE)

  set(XLLM_ATB_LAYERS_SOURCE_DIR
      "${CMAKE_SOURCE_DIR}/third_party/xllm_atb_layers"
      CACHE PATH "Path to xllm_atb_layers source tree")
  if(NOT EXISTS "${XLLM_ATB_LAYERS_SOURCE_DIR}")
    message(FATAL_ERROR "xllm_atb_layers source not found: ${XLLM_ATB_LAYERS_SOURCE_DIR}")
  endif()
  message(STATUS "Using xllm_atb_layers source at: ${XLLM_ATB_LAYERS_SOURCE_DIR}")

  execute_process(
    COMMAND git -c "safe.directory=${CMAKE_SOURCE_DIR}/third_party/xllm_ops" -C "${CMAKE_SOURCE_DIR}/third_party/xllm_ops" rev-parse HEAD
    OUTPUT_VARIABLE XLLM_OPS_GIT_HEAD
    OUTPUT_STRIP_TRAILING_WHITESPACE
    ERROR_QUIET
  )
  if(DEFINED ENV{NPU_HOME_PATH} AND NOT "$ENV{NPU_HOME_PATH}" STREQUAL "")
    set(XLLM_NPU_HOME_PATH "$ENV{NPU_HOME_PATH}")
  elseif(DEFINED ENV{ASCEND_HOME_PATH} AND NOT "$ENV{ASCEND_HOME_PATH}" STREQUAL "")
    set(XLLM_NPU_HOME_PATH "$ENV{ASCEND_HOME_PATH}")
  else()
    message(FATAL_ERROR "NPU_HOME_PATH or ASCEND_HOME_PATH is required for NPU build.")
  endif()

  if(DEFINED ENV{ASCEND_OPP_PATH} AND NOT "$ENV{ASCEND_OPP_PATH}" STREQUAL "")
    set(XLLM_ASCEND_OPP_PATH "$ENV{ASCEND_OPP_PATH}")
  else()
    set(XLLM_ASCEND_OPP_PATH "${XLLM_NPU_HOME_PATH}/opp")
  endif()

  if(DEFINED XLLM_ASCEND_OPP_PATH)
    set(XLLM_OPS_MARKER_PATH "${XLLM_ASCEND_OPP_PATH}/vendors/xllm/.xllm_ops_git_head")
  else()
    message(FATAL_ERROR "ASCEND_OPP_PATH is not initialized for NPU build.")
  endif()

  if(NOT DEFINED XLLM_OPS_GIT_HEAD_CACHED OR NOT XLLM_OPS_GIT_HEAD STREQUAL XLLM_OPS_GIT_HEAD_CACHED)
    message(STATUS "xllm_ops git HEAD changed; running precompile via execute_process")
    execute_process(
      COMMAND bash ${CMAKE_SOURCE_DIR}/third_party/xllm_ops/build.sh
      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}/third_party/xllm_ops
      RESULT_VARIABLE XLLM_OPS_RESULT
    )
    if(NOT XLLM_OPS_RESULT EQUAL 0)
      message(FATAL_ERROR "Failed to precompile xllm ops, error code: ${XLLM_OPS_RESULT}")
    endif()

    if(EXISTS "${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop")
      set(XLLM_OPS_MARKER_PATH "${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/.xllm_ops_git_head")
    endif()
    message(STATUS "XLLM_OPS_MARKER_PATH: ${XLLM_OPS_MARKER_PATH}")

    set(XLLM_OPS_GIT_HEAD_CACHED "${XLLM_OPS_GIT_HEAD}" CACHE INTERNAL "" FORCE)
    get_filename_component(XLLM_OPS_MARKER_DIR "${XLLM_OPS_MARKER_PATH}" DIRECTORY)
    file(MAKE_DIRECTORY "${XLLM_OPS_MARKER_DIR}")
    file(WRITE "${XLLM_OPS_MARKER_PATH}" "${XLLM_OPS_GIT_HEAD}\n")
    message(STATUS "xllm ops precompiled and HEAD cache updated")
  else()
    message(STATUS "xllm_ops git HEAD unchanged; skipping precompile")
  endif()
endif()

enable_testing()

if(NOT TARGET all_tests)
  add_custom_target(all_tests)
endif()

if(NOT TARGET export_module)
  add_custom_target(export_module)
endif()

if (CMAKE_BUILD_TYPE STREQUAL "Release")
    add_compile_options(-O3)
endif()

option(USE_MSPTI "Enable MSPTI for NPU Profiling" OFF)
if(USE_MSPTI)
  add_definitions(-DUSE_MSPTI)
endif()

option(USE_CCACHE "Attempt using CCache to wrap the compilation" ON)
option(USE_CXX11_ABI "Use the new C++-11 ABI, which is not backwards compatible." OFF)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -fPIC")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -fPIC")

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_COLOR_DIAGNOSTICS ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)
set(CMAKE_MESSAGE_LOG_LEVEL STATUS)
set(CMAKE_VERBOSE_MAKEFILE ON)

if(POLICY CMP0135)
  # CMP0135: ExternalProject ignores timestamps in archives by default for the URL download method.
  cmake_policy(SET CMP0135 NEW)
endif()

function(parse_make_options options prefix)
  foreach(option ${options})
    string(REGEX REPLACE "(-D|-)" "" option ${option})
    string(REPLACE "=" ";" option ${option})
    list(GET option 0 option_name)
    list(GET option 1 option_value)
    set(${prefix}_${option_name}
        ${option_value}
        PARENT_SCOPE)
  endforeach()
endfunction()

# Set default build type
if(NOT CMAKE_BUILD_TYPE)
  message(STATUS "Build type not set - defaulting to Release")
  set(CMAKE_BUILD_TYPE "Release" 
      CACHE STRING "Choose the type of build from: Debug Release RelWithDebInfo MinSizeRel Coverage." 
      FORCE
  )
endif()

# Convert the bool variable to integer.
if(USE_CXX11_ABI)
  set(USE_CXX11_ABI 1)
  message(STATUS "Using the C++-11 ABI.")
else()
  set(USE_CXX11_ABI 0)
  message(STATUS "Using the pre C++-11 ABI.")
endif()

if(USE_CCACHE)
  find_program(CCACHE_PROGRAM ccache)
  if(CCACHE_PROGRAM)
    set(CMAKE_C_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "C compiler launcher")
    set(CMAKE_CXX_COMPILER_LAUNCHER "${CCACHE_PROGRAM}" CACHE STRING "CXX compiler launcher")
    message(STATUS "Using ccache: ${CCACHE_PROGRAM}")
    if (DEFINED ENV{CCACHE_DIR})
      message(STATUS "Using CCACHE_DIR: $ENV{CCACHE_DIR}")
    endif()
  else()
    message(WARNING "Could not find ccache. Consider installing ccache to speed up compilation.")
  endif()
endif()

# if defined, create and use the default binary cache for vcpkg
if (DEFINED ENV{VCPKG_DEFAULT_BINARY_CACHE})
  file(MAKE_DIRECTORY $ENV{VCPKG_DEFAULT_BINARY_CACHE})
  message(STATUS "Using VCPKG_DEFAULT_BINARY_CACHE: $ENV{VCPKG_DEFAULT_BINARY_CACHE}")
endif()

if (DEFINED ENV{DEPENDENCES_ROOT})
  message(STATUS "Using DEPENDENCES_ROOT: $ENV{DEPENDENCES_ROOT}")
endif()


if(USE_ILU)
    set(CMAKE_MODULE_PATH "${CMAKE_CURRENT_LIST_DIR}/cmake/Modules;${CMAKE_MODULE_PATH}")
    set(CMAKE_EXPORT_COMPILE_COMMANDS   ON)
    set(CMAKE_CUDA_ARCHITECTURES "ivcore11")
    set(WARNINGS_AS_ERRORS OFF)
    if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
      add_definitions(
        -Wno-c++11-narrowing
        -Wno-thread-safety-analysis
      )
  endif()
endif()

# configure vcpkg
# have to set CMAKE_TOOLCHAIN_FILE before first project call.
# if (DEFINED ENV{VCPKG_ROOT} AND NOT DEFINED CMAKE_TOOLCHAIN_FILE)
if (DEFINED ENV{VCPKG_ROOT})
  set(CMAKE_TOOLCHAIN_FILE "$ENV{VCPKG_ROOT}/scripts/buildsystems/vcpkg.cmake"
      CACHE STRING "Vcpkg toolchain file")
  message(STATUS "VCPKG_ROOT found, using vcpkg at $ENV{VCPKG_ROOT}")
else()
  include(FetchContent)
  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(VCPKG_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/vcpkg-src)
  else()
    set(VCPKG_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/vcpkg-src)
  endif()

  if (USE_CXX11_ABI)
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://github.com/microsoft/vcpkg.git"
      GIT_TAG "2024.02.14"
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
  else()
    FetchContent_Declare(vcpkg
      GIT_REPOSITORY "https://gitcode.com/xLLM-AI/vcpkg.git"
      GIT_TAG "ffc42e97c866ce9692f5c441394832b86548422c" #disable cxx11_abi
      SOURCE_DIR ${VCPKG_SOURCE_DIR}
    )
    message(STATUS "Using custom vcpkg with cxx11_abi disabled")
  endif()
  FetchContent_MakeAvailable(vcpkg)

  message(STATUS "Downloading and using vcpkg at ${vcpkg_SOURCE_DIR}")
  set(CMAKE_TOOLCHAIN_FILE ${vcpkg_SOURCE_DIR}/scripts/buildsystems/vcpkg.cmake
      CACHE STRING "Vcpkg toolchain file")
endif()

set(CPPREST_EXCLUDE_WEBSOCKETS ON CACHE BOOL "Exclude websockets functionality." FORCE)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-format-truncation")

list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR}/cmake)
if(USE_CUDA)
  project("xllm" LANGUAGES C CXX CUDA)
  find_package(CUDAToolkit REQUIRED)
elseif(USE_MUSA)
  project("xllm" LANGUAGES C CXX MUSA)
  add_compile_options(
        -Wno-c++11-narrowing
      )
else()
  project("xllm" LANGUAGES C CXX)
endif()

# ---------------------------------------------------------------------------
# ARM architecture optimization (must be after project() so CXX is enabled).
# -march=native lets the compiler use NEON, SVE, LSE atomics, crypto, etc.
# that are available on the local CPU.  Fall back to armv8.2-a if the flag
# is unsupported (cross-compilation scenario).
# armv8.2-a+dotprod+crypto is good for Kunpeng 920, Neoverse N1, etc.
# ---------------------------------------------------------------------------
if(DEVICE_ARCH STREQUAL "ARM")
    include(CheckCXXCompilerFlag)
    check_cxx_compiler_flag("-march=armv8.2-a+dotprod+crypto" HAS_MARCH_CUSTOMIZATION)
   if(HAS_MARCH_CUSTOMIZATION)
       add_compile_options(-march=armv8.2-a+dotprod+crypto)
       message(STATUS "ARM: using -march=armv8.2-a+dotprod+crypto for optimal ISA")
   else()
        add_compile_options(-march=native) 
        message(STATUS "ARM: using -march=native (cross-compile fallback)")
   endif()
endif()

# find_package(CUDAToolkit REQUIRED)

# setup CMake module path, defines path for include() and find_package()
list(APPEND CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
enable_language(Rust)
find_package(Rust REQUIRED)

if(UNIX)
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -Og")
endif()

find_package(Boost REQUIRED)
find_package(Boost REQUIRED COMPONENTS serialization)
find_package(Threads REQUIRED)
# find all dependencies from vcpkg
find_package(glog CONFIG REQUIRED)
find_package(gflags CONFIG REQUIRED)
find_package(leveldb CONFIG REQUIRED)
find_package(OpenSSL REQUIRED)
find_package(absl CONFIG REQUIRED)
find_package(Protobuf CONFIG REQUIRED)
find_package(gRPC CONFIG REQUIRED)
find_package(folly CONFIG REQUIRED)
find_package(GTest CONFIG REQUIRED)
find_package(benchmark CONFIG REQUIRED)
find_package(nlohmann_json CONFIG REQUIRED)
find_package(OpenCV CONFIG REQUIRED)
find_package(Python COMPONENTS Development REQUIRED)
find_package(pybind11 CONFIG REQUIRED)

if (USE_CXX11_ABI)
  # only use jemalloc if using the new C++-11 ABI
  find_package(Jemalloc)
  if(Jemalloc_FOUND)
    link_libraries(Jemalloc::jemalloc)
  endif()
endif()

# Important Note: Always invoke find_package for other dependencies
# before including libtorch, as doing so afterwards may lead to
# unexpected linker errors.
if (DEFINED ENV{LIBTORCH_ROOT})
  find_package(Torch REQUIRED HINTS "$ENV{LIBTORCH_ROOT}")
  message(STATUS "Using libtorch at $ENV{LIBTORCH_ROOT}")
else()
  include(FetchContent)
  if (USE_CXX11_ABI)
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-cxx11-abi-shared-with-deps-2.1.0%2Bcpu.zip")
  else()
    set(LIBTORCH_URL "https://download.pytorch.org/libtorch/cpu/libtorch-shared-with-deps-2.1.0%2Bcpu.zip")
  endif()

  if (DEFINED ENV{DEPENDENCES_ROOT})
    set(LIBTORCH_SOURCE_DIR $ENV{DEPENDENCES_ROOT}/libtorch-src)
  else()
    set(LIBTORCH_SOURCE_DIR ${FETCHCONTENT_BASE_DIR}/libtorch-src)
  endif()

  FetchContent_Declare(libtorch 
    URL ${LIBTORCH_URL} 
    SOURCE_DIR ${LIBTORCH_SOURCE_DIR}
  )
  FetchContent_MakeAvailable(libtorch)
  
  find_package(Torch REQUIRED PATHS ${LIBTORCH_SOURCE_DIR} NO_DEFAULT_PATH)
  message(STATUS "Downloading and using libtorch 2.1.0 for CPU at ${LIBTORCH_SOURCE_DIR}")
endif()

if(USE_NPU)
  add_definitions(-DUSE_NPU)
  add_definitions(-DBUILD_LIBTORCH)
  add_definitions(-DTORCH_SETCUSTOMHANDLER=ON)
  # set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -Wl,--no-undefined")
  set(CMAKE_VERBOSE_MAKEFILE ON)
  add_definitions(-DTORCH_HIGHER_THAN_PTA6)

  add_subdirectory(
    "${XLLM_ATB_LAYERS_SOURCE_DIR}"
    "${CMAKE_BINARY_DIR}/xllm_atb_layers"
  )
  message(STATUS "Configured xllm_atb_layers from source: ${XLLM_ATB_LAYERS_SOURCE_DIR}")

  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      ${Python_INCLUDE_DIRS}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      $ENV{PYTORCH_NPU_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/distributed
      ${XLLM_NPU_HOME_PATH}/include
      $ENV{ATB_HOME_PATH}/include
      ${CMAKE_CURRENT_SOURCE_DIR}/third_party/torch_npu_ops/
  )
  if(EXISTS "${XLLM_ASCEND_OPP_PATH}/vendors/xllm/op_api/include")
    include_directories(SYSTEM
      ${XLLM_ASCEND_OPP_PATH}/vendors/xllm/op_api/include/
    )
  endif()
  if(EXISTS "${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop")
    include_directories(SYSTEM
      ${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/include/aclnnop
    )
  endif()
  # Keep third-party warnings suppressed while providing headers expected by
  # legacy includes like "atb_speed/log.h" from xllm_atb_layers.
  include_directories(SYSTEM
      ${XLLM_ATB_LAYERS_SOURCE_DIR}
      ${XLLM_ATB_LAYERS_SOURCE_DIR}/core/include
  )
  link_directories(
      $ENV{PYTHON_LIB_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/lib
      $ENV{PYTORCH_NPU_INSTALL_PATH}/lib
      ${XLLM_NPU_HOME_PATH}/lib64
      $ENV{ATB_HOME_PATH}/lib
      $ENV{NPU_TOOLKIT_HOME}/lib64
  )
  if(EXISTS "${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/lib")
    link_directories(
      ${XLLM_ASCEND_OPP_PATH}/vendors/custom_xllm_math/op_api/lib/
    )
  endif()
  link_libraries(cust_opapi)
  # avoid conflicts with xllm libruntime.a
  find_library(
    NPU_RUNTIME_SO
    NAMES libruntime.so
    PATHS $ENV{ASCEND_TOOLKIT_HOME}/lib64
    NO_DEFAULT_PATH
  )
  if(NOT NPU_RUNTIME_SO)
      message(FATAL_ERROR "Failed to find npu libruntime.so in ${ASCEND_TOOLKIT_HOME}/lib64")
  endif()
  set(NPU_RUNTIME_SO ${NPU_RUNTIME_SO} CACHE INTERNAL "CACHED ASCEND libruntime.so path")
endif()

if(USE_MLU)
  add_definitions(-DUSE_MLU)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      $ENV{PYTORCH_MLU_INSTALL_PATH}
      $ENV{PYTORCH_MLU_INSTALL_PATH}/../
      $ENV{PYTORCH_MLU_INSTALL_PATH}/csrc
      $ENV{PYTORCH_MLU_INSTALL_PATH}/csrc/include
      $ENV{NEUWARE_HOME}/include
  )

  link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{PYTORCH_MLU_INSTALL_PATH}/csrc/lib
    $ENV{PYTORCH_MLU_INSTALL_PATH}
    $ENV{NEUWARE_HOME}/lib64
  )
endif()


if(USE_MUSA)
  add_definitions(-DUSE_MUSA)
  add_compile_definitions(TORCH_CUDA=1)

  if(NOT DEFINED MUSA_PATH)
    set(MUSA_PATH /usr/local/musa)
  endif()
  list(APPEND CMAKE_MODULE_PATH "${MUSA_PATH}/cmake")
  find_package(MUSA REQUIRED)

  if(NOT DEFINED ENV{MTT_OPLIB_PATH})
    set(ENV{MTT_OPLIB_PATH} $ENV{MUSA_HOME}/tools/MTTOplib)
  endif()

  message(STATUS  "using MTT Oplib at: $ENV{MTT_OPLIB_PATH}")
  set(MTTOplib_DIR $ENV{MTT_OPLIB_PATH}/cmake)
  find_package(MTTOplib REQUIRED)

  find_package(Python COMPONENTS Interpreter REQUIRED)
  list(APPEND CMAKE_PREFIX_PATH $ENV{TORCH_MUSA_PYTHONPATH})
  find_package(TorchMusa REQUIRED CONFIG)
  list(POP_BACK CMAKE_PREFIX_PATH)

  find_package(MKL REQUIRED)
  execute_process(
    COMMAND tvm-ffi-config --includedir
    OUTPUT_VARIABLE tvm_ffi_INCLUDE_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  execute_process(
    COMMAND tvm-ffi-config --libdir
    OUTPUT_VARIABLE tvm_ffi_LIB_DIR
    OUTPUT_STRIP_TRAILING_WHITESPACE
  )
  string(REGEX MATCH "[^\n]*/include[^\n]*" tvm_ffi_INCLUDE_DIR "${tvm_ffi_INCLUDE_DIR}")
  string(REGEX MATCH "[^\n]*/lib[^\n]*" tvm_ffi_LIB_DIR "${tvm_ffi_LIB_DIR}")
  

  include_directories(
      ${MUSA_INCLUDE_DIRS}
      ${TorchMusa_INCLUDE_DIRS}
      ${Python_SITELIB}/torch_musa_compiled/share/torch_musa_codegen

      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      ${MTTOplib_INCLUDE_DIRS}
      ${tvm_ffi_INCLUDE_DIR}
  )

  link_directories(
    ${MUSA_LIB_PATH}
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    ${TorchMusa_LIB_PATH}
    ${MKL_LIB_PATH}
    ${tvm_ffi_LIB_DIR}
    ${TorchMusa_INSTALL_PREFIX}/lib
    )
  set(MUSA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
endif()

if(USE_CUDA)
  message(STATUS "TORCH_CUDA_ARCH_LIST: ${TORCH_CUDA_ARCH_LIST}")
  add_definitions(-DUSE_CUDA)
  add_compile_definitions(TORCH_CUDA=1)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
  )

  link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{CUDA_TOOLKIT_ROOT_DIR}/lib64
  )

  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} -O3)
  # The following definitions must be undefined since half-precision operation is required.
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS}
        -U__CUDA_NO_HALF_OPERATORS__
        -U__CUDA_NO_HALF_CONVERSIONS__
        -U__CUDA_NO_HALF2_OPERATORS__
        -U__CUDA_NO_BFLOAT16_CONVERSIONS__)
  set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS} --use_fast_math -Xfatbin -compress-all)
  message(STATUS "CUDA_NVCC_FLAGS: ${CUDA_NVCC_FLAGS}")

  # find_package(NCCL REQUIRED)

  # find cudnn
  execute_process(COMMAND python -c "import nvidia.cudnn; print(nvidia.cudnn.__file__)" OUTPUT_VARIABLE CUDNN_PYTHON_PATH)
  get_filename_component(CUDNN_ROOT_DIR "${CUDNN_PYTHON_PATH}" DIRECTORY)
  link_directories(
      ${CUDNN_ROOT_DIR}/lib64
      ${CUDNN_ROOT_DIR}/lib
  )
endif()

if(USE_ILU)
  add_definitions(-DUSE_ILU)
  set(CMAKE_VERBOSE_MAKEFILE ON)
  include_directories(
      $ENV{PYTHON_INCLUDE_PATH}
      $ENV{PYTORCH_INSTALL_PATH}/include
      $ENV{PYTORCH_INSTALL_PATH}/include/torch/csrc/api/include
      $ENV{IXFORMER_INSTALL_PATH}/csrc/include/ixformer
  )

  link_directories(
    $ENV{PYTHON_LIB_PATH}
    $ENV{PYTORCH_INSTALL_PATH}/lib
    $ENV{IXFORMER_INSTALL_PATH}
  )
endif()

# check if USE_CXX11_ABI is set correctly
# if (DEFINED USE_CXX11_ABI)
#   parse_make_options(${TORCH_CXX_FLAGS} "TORCH_CXX_FLAGS")
#   if(DEFINED TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI
#      AND NOT ${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} EQUAL ${USE_CXX11_ABI})
#       message(FATAL_ERROR
#           "The libtorch compilation options _GLIBCXX_USE_CXX11_ABI=${TORCH_CXX_FLAGS__GLIBCXX_USE_CXX11_ABI} "
#           "found by CMake conflict with the project setting USE_CXX11_ABI=${USE_CXX11_ABI}.")
#   endif()
# endif()

# carry over torch flags to the rest of the project
message(STATUS "TORCH_CXX_FLAGS: ${TORCH_CXX_FLAGS}")
add_compile_options(${TORCH_CXX_FLAGS})

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -DC10_USE_GLOG")

message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")

# enable testing in this directory so we can do a top-level `make test`.
# this also includes the BUILD_TESTING option, which is on by default.
include(CTest)
include(GoogleTest)
option(BUILD_TESTING "Build the testing tree." ON)

# include current path
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR})
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/xllm/core)
# Note: third_party is not added to COMMON_INCLUDE_DIRS to avoid warnings
# Individual third_party directories are added as SYSTEM headers below
list(APPEND COMMON_INCLUDE_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/third_party/etcd_cpp_apiv3)

# brpc - mark as system headers to suppress warnings from third-party code
set(BRPC_OUTPUT_DIR ${CMAKE_BINARY_DIR}/third_party/brpc/output)
include_directories(SYSTEM ${BRPC_OUTPUT_DIR}/include)
link_directories(${BRPC_OUTPUT_DIR}/lib)

# Mooncake - mark as system headers to suppress warnings from third-party code
# Add third_party as SYSTEM to support #include <Mooncake/...> style includes
# Use absolute path to ensure it works correctly
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake/mooncake-common/include)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake/mooncake-transfer-engine/include)
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/Mooncake/mooncake-store/include)

# CUTLASS - CUDA kernel library for scaled matmul
if(USE_CUDA)
  set(CUTLASS_DIR ${CMAKE_CURRENT_SOURCE_DIR}/third_party/cutlass)
  set(CUTLASS_INCLUDE_DIR ${CUTLASS_DIR}/include)
  set(CUTLASS_TOOLS_UTIL_INCLUDE_DIR ${CUTLASS_DIR}/tools/util/include)
  include_directories(SYSTEM ${CUTLASS_INCLUDE_DIR})
  include_directories(SYSTEM ${CUTLASS_TOOLS_UTIL_INCLUDE_DIR})
endif()

# sentencepiece - mark as system headers to suppress warnings from third-party code
# This must be added before add_subdirectory(third_party) to ensure it takes precedence
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/sentencepiece)
# Also add other commonly used third_party directories as system headers
include_directories(SYSTEM ${CMAKE_CURRENT_SOURCE_DIR}/third_party/minja)
if(USE_MUSA)
  set(CMAKE_MUSA_FLAGS_DEBUG "-g -std=gnu++20 -fPIC --offload-arch=mp_31")
  set(CMAKE_MUSA_FLAGS_RELEASE "-O2 -DNDEBUG -std=gnu++20 -fPIC --offload-arch=mp_31")
endif()

# add subdirectories
# Configure third_party first so xllm can link against real CMake targets.
add_subdirectory(third_party)
if(TARGET brpc-static AND NOT TARGET brpc)
  # Expose a canonical target name used across xllm/CMakeLists.
  add_library(brpc ALIAS brpc-static)
endif()
add_subdirectory(xllm)
