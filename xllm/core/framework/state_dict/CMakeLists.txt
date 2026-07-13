include(cc_library)

include_directories(..)

add_subdirectory(safetensors)

cc_library(
  NAME 
    state_dict
  HDRS
    state_dict.h
    utils.h
    rec_vocab_dict.h
  SRCS
    state_dict.cpp
    utils.cpp
    rec_vocab_dict.cpp
  DEPS
    :config
    rust_safetensors
    torch
    glog::glog
    Folly::folly
    util
)
