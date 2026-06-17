include(cc_library)

cc_library(
  NAME
    encoder_cache
  HDRS
    encoder_cache.h
  SRCS
    encoder_cache.cpp
  DEPS
    glog::glog
    torch
)
