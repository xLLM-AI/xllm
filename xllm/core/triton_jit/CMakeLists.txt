# triton_jit: backend-agnostic runtime Triton JIT runtime.
# The generic core (argument wrapping, signature/cache-key, compilation,
# caching) plus one per-build backend implementation.
#
# Compilation runs in-process via the embedded interpreter, which needs Python
# Development headers/libs. The Python triton-compile helper
# (scripts/triton_compile.py) is imported by the embedded interpreter as the
# module xllm.core.triton_jit.scripts.triton_compile.

include(cc_library)

find_package(Python COMPONENTS Development REQUIRED)

set(TRITON_JIT_DIR ${CMAKE_CURRENT_SOURCE_DIR})

file(GLOB TRITON_JIT_HEADERS
  "${TRITON_JIT_DIR}/include/*.h"
)

set(TRITON_JIT_SRCS
  "${TRITON_JIT_DIR}/src/jit_kernel.cpp"
  "${TRITON_JIT_DIR}/src/backend.cpp"
  "${TRITON_JIT_DIR}/src/spec.cpp"
)

# Only the active backend's header/sources are pulled in, so a non-MLU build is
# not forced to see vendor headers it lacks.
if(USE_MLU)
  list(APPEND TRITON_JIT_HEADERS "${TRITON_JIT_DIR}/include/backends/mlu_backend.h")
  list(APPEND TRITON_JIT_SRCS "${TRITON_JIT_DIR}/src/mlu_backend.cpp")
endif()
# To add a backend: append its header/sources under the matching USE_* guard.

cc_library(
  NAME
    triton_jit
  HDRS
    ${TRITON_JIT_HEADERS}
  SRCS
    ${TRITON_JIT_SRCS}
  DEPS
    torch
    nlohmann_json::nlohmann_json
    pybind11::embed
    Python::Python
)

target_include_directories(triton_jit PUBLIC
  ${TRITON_JIT_DIR}/include
)

target_link_libraries(triton_jit PRIVATE ${CMAKE_DL_LIBS})

if(USE_MLU)
  target_link_libraries(triton_jit PRIVATE cnclep cndrv cnrt torch_mlu torch_mlu_ops)
endif()
