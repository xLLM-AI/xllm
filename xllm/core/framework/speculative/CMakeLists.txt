include(cc_library)

cc_library(
  NAME
    speculative
  HDRS
    adaptive_speculative_controller.h
    speculative_profile_registry.h
  SRCS
    adaptive_speculative_controller.cpp
    speculative_profile_registry.cpp
  DEPS
    torch
    glog::glog
    proto::xllm_proto
)
