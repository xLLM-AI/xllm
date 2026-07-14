include(cc_binary)

cc_binary(
  NAME 
    spawn_worker
  HDRS
    spawn_worker_protocol.h
    spawn_worker_server.h
  SRCS
    spawn_worker_server.cpp
    spawn_worker_server_process.cpp
  DEPS
    :distributed_runtime
)

add_dependencies(export_module spawn_worker)
