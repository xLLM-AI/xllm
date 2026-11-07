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
    $<$<BOOL:${USE_NPU}>:npu_startup>
    $<$<BOOL:${USE_NPU}>:pybind11::embed>
)

xllm_export_native_host(spawn_worker)
add_dependencies(export_module spawn_worker)
