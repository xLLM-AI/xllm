include(cc_library)

cc_library(
  NAME
    xllm_server
  HDRS
    xllm_server.h
    xllm_server_registry.h
  SRCS
    xllm_server.cpp
    xllm_server_registry.cpp
  DEPS
    :common
    :config
    :worker_service
    :collective_service
    brpc
    glog::glog
    protobuf::libprotobuf
)

target_link_libraries(xllm_server PRIVATE
  nlohmann_json::nlohmann_json
  proto::xllm_proto
)
