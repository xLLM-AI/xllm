# Preserve archive ownership during the directory-only migration. The packed
# input codec is also called by runtime; splitting targets follows separately.
target_sources(runtime PRIVATE
  forward_shared_memory_transport.h
  forward_shared_memory_transport.cpp
)

target_sources(distributed_runtime PRIVATE
  comm_channel.h
  comm_channel.cpp
  shm_channel.h
  shm_channel.cpp
)
