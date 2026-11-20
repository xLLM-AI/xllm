# Copyright 2026 The xLLM Authors. Licensed under Apache-2.0.

set(aicpu "${CMAKE_CURRENT_LIST_DIR}/op_kernel_aicpu")
set(cann "${ASCEND_CANN_PACKAGE_PATH}")
find_program(aicpu_cxx NAMES aarch64-target-linux-gnu-g++
    HINTS "${cann}/toolkit/toolchain/hcc/bin" "${cann}/toolchain/hcc/bin"
    NO_DEFAULT_PATH REQUIRED)
find_library(aicpu_context NAMES libaicpu_context.a
    HINTS "${cann}/lib64" "${cann}/ops_base/lib64" NO_DEFAULT_PATH REQUIRED)
find_library(aicpu_protobuf NAMES libbase_ascend_protobuf.a
    HINTS "${cann}/lib64" "${cann}/ops_base/lib64" NO_DEFAULT_PATH REQUIRED)
set(qli_aicpu_library "${CMAKE_CURRENT_BINARY_DIR}/libxllm_kernel_aicpu.so")
add_custom_command(
    OUTPUT "${qli_aicpu_library}"
    COMMAND "${aicpu_cxx}" -std=c++17 -O2 -shared -fPIC -fvisibility=hidden
        -fvisibility-inlines-hidden -fstack-protector-all
        -Dgoogle=ascend_private -DDISABLE_COMPILE_V1 -D_FORTIFY_SOURCE=2 -D_GLIBCXX_USE_CXX11_ABI=1
        "-I${CMAKE_CURRENT_SOURCE_DIR}"
        "-I${cann}/include" "-I${cann}/include/aicpu" "-I${cann}/pkg_inc" "-I${cann}/pkg_inc/base"
        "-I${cann}/pkg_inc/aicpu_common/context/common" "-I${cann}/pkg_inc/aicpu_common/context/cpu_proto"
        "-I${cann}/pkg_inc/aicpu_common/context/utils"
        "${aicpu}/quant_lightning_indexer_metadata_aicpu.cpp"
        -Wl,--whole-archive "${aicpu_context}" "${aicpu_protobuf}" -Wl,--no-whole-archive
        -Wl,-Bsymbolic -Wl,--exclude-libs=libbase_ascend_protobuf.a
        -o "${qli_aicpu_library}"
    DEPENDS
        "${aicpu}/quant_lightning_indexer_metadata_aicpu.cpp"
        "${aicpu}/quant_lightning_indexer_metadata_aicpu.h"
        "${aicpu}/cpu_context_util.h"
        "${qli}/op_kernel/quant_lightning_indexer_metadata.h"
        "${aicpu_context}" "${aicpu_protobuf}"
    VERBATIM
)
add_custom_target(xllm_kernel_aicpu ALL DEPENDS "${qli_aicpu_library}")
set(cpu_destination "packages/vendors/${vendor}/op_impl/cpu")
install(FILES "${qli_aicpu_library}" DESTINATION "${cpu_destination}/aicpu_kernel/impl")
install(FILES "${aicpu}/quant_lightning_indexer_metadata_aicpu.json"
    DESTINATION "${cpu_destination}/config" RENAME cust_aicpu_kernel.json)
