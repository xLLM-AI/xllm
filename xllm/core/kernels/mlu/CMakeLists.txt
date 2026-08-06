include(cc_library)

file(GLOB_RECURSE MLU_HEADER_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.h"
)

file(GLOB_RECURSE MLU_SOURCE_FILES
  "${CMAKE_CURRENT_LIST_DIR}/*.cpp"
)

list(FILTER MLU_HEADER_FILES EXCLUDE REGEX "/tests/")
list(FILTER MLU_SOURCE_FILES EXCLUDE REGEX "/tests/")

cc_library(
  NAME
    mlu_kernels
  HDRS
    ${MLU_HEADER_FILES}
  SRCS
    ${MLU_SOURCE_FILES}
  DEPS
    torch
    cnclep
    torch_mlu_ops
    Python::Python
)

add_subdirectory(tests)
