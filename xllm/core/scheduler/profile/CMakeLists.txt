
include(cc_library)

cc_library(
  NAME 
    profile
  HDRS
    graph_warmup.h
    profile_manager.h
    time_predictor.h
  SRCS
    graph_warmup.cpp
    profile_manager.cpp
    time_predictor.cpp
  DEPS
    :batch
    :distributed_runtime
    :request
    :runtime
    glog::glog
    absl::strings
    absl::time
)
