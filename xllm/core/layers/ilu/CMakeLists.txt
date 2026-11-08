include(cc_library)

cc_library(
  NAME
    ilu_layers
  HDRS
    attention.h
    fused_moe.h
  SRCS
    attention.cpp
    fused_moe.cpp
  DEPS
    :tensor_support
    :common_layers
)
