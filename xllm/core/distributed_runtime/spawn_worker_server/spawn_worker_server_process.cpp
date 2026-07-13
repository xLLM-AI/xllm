/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/jd-opensource/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gflags/gflags.h>
#include <glog/logging.h>

#include <cstdint>
#include <cstdlib>

#include "spawn_worker_server.h"

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @num_decoding_tokens
// @block_size
// @max_tokens_per_batch
// @max_seqs_per_batch
// @enable_shm
// @is_local
// @enable_prefill_sp
// @task_type
// @worker_type
// @enable_speculative_decode
// @num_speculative_tokens
// @speculative_algorithm
// @input_shm_size
// @output_shm_size
// @communication_backend
// @npu_kernel_backend
// @rank_tablefile
// @enable_graph
// @enable_graph_mode_decode_no_padding
// @enable_prefill_piecewise_graph
// @max_tokens_for_graph_mode
// @max_encoder_cache_size
// @dp_size
// @tp_size
// @sp_size
// @cfg_size
int main(int argc, char* argv[]) {
  if (argc < 32) {
    LOG(ERROR)
        << "Spawn worker process receive wrong args. Need 32 args, receive "
        << argc;
    return 1;
  }

  std::string master_node_addr = std::string(argv[1]);
  int32_t local_rank = static_cast<int32_t>(atoi(argv[2]));
  int32_t global_rank = static_cast<int32_t>(atoi(argv[3]));
  int32_t world_size = static_cast<int32_t>(atoi(argv[4]));
  int32_t device_idx = static_cast<int32_t>(atoi(argv[5]));
  int32_t num_decoding_tokens = static_cast<int32_t>(atoi(argv[6]));
  int32_t block_size = static_cast<int32_t>(atoi(argv[7]));
  int32_t max_tokens_per_batch = static_cast<int32_t>(atoi(argv[8]));
  int32_t max_seqs_per_batch = static_cast<int32_t>(atoi(argv[9]));
  int32_t enable_shm = static_cast<int32_t>(atoi(argv[10]));
  int32_t is_local = static_cast<int32_t>(atoi(argv[11]));
  int32_t enable_prefill_sp = static_cast<int32_t>(atoi(argv[12]));
  std::string task_type = std::string(argv[13]);
  std::string worker_type = std::string(argv[14]);
  int32_t enable_speculative_decode = static_cast<int32_t>(atoi(argv[15]));
  int32_t num_speculative_tokens = static_cast<int32_t>(atoi(argv[16]));
  std::string speculative_algorithm = std::string(argv[17]);
  uint64_t input_shm_size = static_cast<uint64_t>(atoll(argv[18]));
  uint64_t output_shm_size = static_cast<uint64_t>(atoll(argv[19]));
  std::string communication_backend = std::string(argv[20]);
  std::string npu_kernel_backend = std::string(argv[21]);
  std::string rank_tablefile = std::string(argv[22]);
  bool enable_graph = static_cast<int32_t>(atoi(argv[23])) > 0;
  bool enable_graph_mode_decode_no_padding =
      static_cast<int32_t>(atoi(argv[24])) > 0;
  bool enable_prefill_piecewise_graph =
      static_cast<int32_t>(atoi(argv[25])) > 0;
  int32_t max_tokens_for_graph_mode = static_cast<int32_t>(atoi(argv[26]));
  int64_t max_encoder_cache_size = static_cast<int64_t>(atoll(argv[27]));
  int32_t dp_size = static_cast<int32_t>(atoi(argv[28]));
  int32_t tp_size = static_cast<int32_t>(atoi(argv[29]));
  int32_t sp_size = static_cast<int32_t>(atoi(argv[30]));
  int32_t cfg_size = static_cast<int32_t>(atoi(argv[31]));

  LOG(INFO) << "Spawn worker: "
            << "master_node_addr = " << master_node_addr
            << ", local_rank = " << local_rank
            << ", world_size = " << world_size
            << ", device_idx = " << device_idx
            << ", num_decoding_tokens = " << num_decoding_tokens
            << ", block_size = " << block_size
            << ", max_tokens_per_batch = " << max_tokens_per_batch
            << ", max_seqs_per_batch = " << max_seqs_per_batch
            << ", enable_shm = " << (enable_shm > 0)
            << ", input_shm_size = " << input_shm_size
            << ", output_shm_size = " << output_shm_size
            << ", is_local = " << (is_local > 0)
            << ", enable_prefill_sp = " << (enable_prefill_sp > 0)
            << ", task_type = " << task_type
            << ", worker_type = " << worker_type
            << ", enable_speculative_decode = "
            << (enable_speculative_decode > 0)
            << ", num_speculative_tokens = " << num_speculative_tokens
            << ", speculative_algorithm = " << speculative_algorithm
            << ", communication_backend = " << communication_backend
            << ", npu_kernel_backend = " << npu_kernel_backend
            << ", rank_tablefile = " << rank_tablefile
            << ", enable_graph = " << enable_graph
            << ", enable_graph_mode_decode_no_padding = "
            << enable_graph_mode_decode_no_padding
            << ", enable_prefill_piecewise_graph = "
            << enable_prefill_piecewise_graph
            << ", max_tokens_for_graph_mode = " << max_tokens_for_graph_mode
            << ", max_encoder_cache_size = " << max_encoder_cache_size
            << ", dp_size = " << dp_size << ", tp_size = " << tp_size
            << ", sp_size = " << sp_size << ", cfg_size = " << cfg_size << "\n";

  xllm::SpawnWorkerServer worker(master_node_addr,
                                 local_rank,
                                 global_rank,
                                 world_size,
                                 device_idx,
                                 num_decoding_tokens,
                                 block_size,
                                 max_tokens_per_batch,
                                 max_seqs_per_batch,
                                 enable_shm > 0,
                                 input_shm_size,
                                 output_shm_size,
                                 is_local > 0,
                                 enable_prefill_sp > 0,
                                 task_type,
                                 worker_type,
                                 enable_speculative_decode > 0,
                                 num_speculative_tokens,
                                 speculative_algorithm,
                                 communication_backend,
                                 npu_kernel_backend,
                                 rank_tablefile,
                                 enable_graph,
                                 enable_graph_mode_decode_no_padding,
                                 enable_prefill_piecewise_graph,
                                 max_tokens_for_graph_mode,
                                 max_encoder_cache_size,
                                 dp_size,
                                 tp_size,
                                 sp_size,
                                 cfg_size);

  worker.run();

  return 0;
}
