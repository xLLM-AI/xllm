
include(cc_library)

cc_library(
  NAME 
    profile
  HDRS
    decode_graph_warmup_plan.h
    graph_warmup.h
    profile_manager.h
    time_predictor.h
  SRCS
    decode_graph_warmup_plan.cpp
    graph_warmup.cpp
    profile_manager.cpp
    time_predictor.cpp
  DEPS
    :batch
    :distributed_runtime
    :request
    :runtime
    :platform
    glog::glog
    absl::strings
    absl::time
)
