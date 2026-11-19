# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

# Keep existing archive ownership while codec dependencies are separated.
target_sources(runtime PRIVATE
  packed_forward_codec.h
  packed_forward_codec.cpp
  shared_memory_transport.h
  shared_memory_transport.cpp
)

target_sources(distributed_runtime PRIVATE
  worker_transport.h
  worker_transport.cpp
  comm_channel.h
  rpc_channel.h
  rpc_channel.cpp
  shm_channel.h
  shm_channel.cpp
)
