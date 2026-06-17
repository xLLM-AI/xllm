include(cc_library)

cc_library(
  NAME
    mposition
  HDRS
    mposition.h
  SRCS
    mposition.cpp
  DEPS
    :model
    :common
    torch
)
