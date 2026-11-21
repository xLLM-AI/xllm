cmake_minimum_required(VERSION 3.16)
project(community_operator_solution LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(SOC_VERSION "Ascend910_9382" CACHE STRING "Target verified SoC")
set(RUN_MODE "npu")
set(ASCEND_CANN_PACKAGE_PATH "$ENV{ASCEND_HOME_PATH}" CACHE PATH "CANN installation")
if(NOT ASCEND_CANN_PACKAGE_PATH)
  message(FATAL_ERROR "Source the CANN environment before building")
endif()
include(${ASCEND_CANN_PACKAGE_PATH}/tools/tikcpp/ascendc_kernel_cmake/ascendc.cmake)
file(GLOB KERNEL_SOURCES CONFIGURE_DEPENDS op_kernel/*.cpp)
ascendc_library(task_kernels SHARED ${KERNEL_SOURCES})

set(CMAKE_SKIP_RPATH FALSE)
add_executable(run_example examples/run_example.cpp)
target_include_directories(run_example PRIVATE include ${ASCEND_CANN_PACKAGE_PATH}/include)
target_link_directories(run_example PRIVATE ${ASCEND_CANN_PACKAGE_PATH}/lib64)
target_link_libraries(run_example PRIVATE task_kernels ascendcl)
set_target_properties(run_example PROPERTIES BUILD_RPATH "$ORIGIN;${ASCEND_CANN_PACKAGE_PATH}/lib64")

# Native PyTorch binding: tensor allocation, device guard and stream tracking in C++.
find_package(Torch REQUIRED PATHS "${TORCH_ROOT}/share/cmake/Torch" NO_DEFAULT_PATH)
add_library(task_torch SHARED op_host/torch_binding.cpp op_host/aclnn_binding.cpp)
target_include_directories(task_torch PRIVATE include ${TORCH_INCLUDE_DIRS} "${TORCH_NPU_ROOT}/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_directories(task_torch PRIVATE "${TORCH_NPU_ROOT}/lib")
target_link_libraries(task_torch PRIVATE task_kernels ${TORCH_LIBRARIES} torch_npu)
set_target_properties(task_torch PROPERTIES BUILD_RPATH "$ORIGIN;${TORCH_ROOT}/lib;${TORCH_NPU_ROOT}/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")
target_compile_definitions(task_torch PRIVATE TASK_INDEX=2)

set_target_properties(task_kernels PROPERTIES OUTPUT_NAME "task2_kernels")
set_target_properties(task_torch PROPERTIES OUTPUT_NAME "task2_torch")

target_link_libraries(task_torch PRIVATE "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so" dl)

add_executable(run_aclnn_example examples/run_aclnn_example.cpp)
target_include_directories(run_aclnn_example PRIVATE "${ACLNN_ROOT}/op_api/include" "${ASCEND_CANN_PACKAGE_PATH}/include")
target_link_libraries(run_aclnn_example PRIVATE "${ACLNN_ROOT}/op_api/lib/libcust_opapi.so" "${ASCEND_CANN_PACKAGE_PATH}/lib64/libnnopbase.so" "${ASCEND_CANN_PACKAGE_PATH}/lib64/libascendcl.so")
set_target_properties(run_aclnn_example PROPERTIES BUILD_RPATH "$ORIGIN/../aclnn_install/vendors/community_task2/op_api/lib;${ASCEND_CANN_PACKAGE_PATH}/lib64")


