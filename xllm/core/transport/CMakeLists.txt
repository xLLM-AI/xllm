# Keep existing archive ownership while codec dependencies are separated.
target_sources(runtime PRIVATE
  packed_forward_codec.h
  packed_forward_codec.cpp
  forward_shared_memory_transport.h
  forward_shared_memory_transport.cpp
)

target_sources(distributed_runtime PRIVATE
  comm_channel.h
  comm_channel.cpp
  shm_channel.h
  shm_channel.cpp
)
