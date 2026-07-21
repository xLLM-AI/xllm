include(cc_library)

cc_library(
  NAME
    speculative
  HDRS
    adaptive_speculative_controller.h
    speculative_profile_registry.h
    adaptive_pruning_helpers.h
  SRCS
    adaptive_speculative_controller.cpp
    speculative_profile_registry.cpp
    adaptive_pruning_helpers.cpp
  DEPS
    torch
    glog::glog
    proto::xllm_proto
)
