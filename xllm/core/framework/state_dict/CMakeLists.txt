include(cc_library)

include_directories(..)

add_subdirectory(safetensors)

cc_library(
  NAME 
    state_dict
  HDRS
    state_dict.h
    utils.h
  SRCS
    state_dict.cpp
    utils.cpp
  DEPS
    rust_safetensors
    torch
    glog::glog
    Folly::folly
    util
)
