/* Copyright 2025-2026 The xLLM Authors.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    https://github.com/xLLM-AI/xllm/blob/main/LICENSE

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>

#include <cstdint>
#include <cstdlib>
#include <optional>
#include <string>
#include <string_view>
#include <utility>

#if defined(USE_NPU)
#include "core/platform/npu/startup.h"
#endif

#include "core/common/native_library.h"
#include "core/distributed_runtime/spawn_worker_server/spawn_worker_protocol.h"
#include "core/distributed_runtime/spawn_worker_server/spawn_worker_server.h"
#include "core/framework/config/model_config.h"
#include "core/framework/transfer/mooncake_transfer_engine.h"

// Worker argv from engine process:
// @master_node_addr
// @local_rank
// @global_rank
// @world_size
// @device_idx
// @block_size
// @max_tokens_per_batch
// @max_seqs_per_batch
// @enable_shm
// @is_local
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
// @cp_size
// @ep_size
// @instance_role
// @indexer_cache_dtype
// @enable_mtp_draft_body_tp1
// @text_encoder_tp_size
// @enable_probabilistic_draft
// @kv_split_size
// @model_impl
// @extension_impl
// @enable_eplb
// @redundant_experts_num
// @use_decode_only_load
// @max_tokens_per_chunk_for_prefill
// @enable_chunked_prefill
// @enable_dp_fair_token_budget
// @vae_size (optional, Python DiT)
// @enable_graph_double_buffer (optional)
// @acl_graph_decode_batch_size_limit (optional)
// @layerwise_split_size (optional)
// @enable_mm_encoder_dp (optional)
// @enable_graph_vmm_pool (optional)
// @enable_prefix_cache (optional)
int main(int argc, char* argv[]) {
  const std::optional<std::string> parsed_indexer_cache_dtype =
      xllm::spawn_worker_protocol::parse_indexer_cache_dtype(argc, argv);
  if (!parsed_indexer_cache_dtype.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid args. Need at least "
               << xllm::spawn_worker_protocol::kMinimumArgumentCount
               << " args, received " << argc;
    return 1;
  }
  if (argc == xllm::spawn_worker_protocol::kMinimumArgumentCount) {
    LOG(WARNING) << "Spawn worker process received legacy args without "
                    "indexer_cache_dtype. Defaulting indexer_cache_dtype to "
                 << xllm::spawn_worker_protocol::kDefaultIndexerCacheDtype
                 << ".";
  }

  std::optional<xllm::EplbWorkerConfig> eplb_worker_config =
      xllm::spawn_worker_protocol::parse_eplb_worker_config(argc, argv);
  if (!eplb_worker_config.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid EPLB args. Expected "
                  "enable_eplb and use_decode_only_load as 0 or 1, and a "
                  "non-negative int32 redundant_experts_num.";
    return 1;
  }

  const std::optional<xllm::spawn_worker_protocol::PrefillConfig>
      prefill_config =
          xllm::spawn_worker_protocol::parse_prefill_config(argc, argv);
  if (!prefill_config.has_value()) {
    LOG(ERROR)
        << "Spawn worker process received invalid prefill args: "
           "expected positive chunk tokens and 0/1 chunked/fair switches.";
    return 1;
  }

  const std::optional<bool> enable_graph_double_buffer =
      xllm::spawn_worker_protocol::parse_graph_double_buffer(argc, argv);
  if (!enable_graph_double_buffer.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid graph double-buffer "
                  "argument: expected 0 or 1.";
    return 1;
  }

  const std::optional<int32_t> graph_decode_batch_size_limit =
      xllm::spawn_worker_protocol::parse_graph_decode_batch_size_limit(argc,
                                                                       argv);
  if (!graph_decode_batch_size_limit.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid graph decode batch "
                  "size limit.";
    return 1;
  }

  const auto topology_config =
      xllm::spawn_worker_protocol::parse_topology_config(argc, argv);
  if (!topology_config.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid topology arguments: "
                  "expected positive layerwise split and 0/1 encoder DP.";
    return 1;
  }

  const std::optional<bool> enable_graph_vmm_pool =
      xllm::spawn_worker_protocol::parse_graph_vmm_pool(argc, argv);
  if (!enable_graph_vmm_pool.has_value()) {
    LOG(ERROR) << "Spawn worker process received invalid graph VMM pool "
                  "argument: expected 0 or 1.";
    return 1;
  }

  // Older launchers omit the prefix-cache setting and retain its enabled
  // default.
  bool enable_prefix_cache = true;
  if (argc > xllm::spawn_worker_protocol::kEnablePrefixCacheArgumentIndex) {
    const std::string_view enabled(
        argv[xllm::spawn_worker_protocol::kEnablePrefixCacheArgumentIndex]);
    if (enabled != "0" && enabled != "1") {
      LOG(ERROR) << "Spawn worker process received invalid prefix-cache "
                    "argument: expected 0 or 1.";
      return 1;
    }
    enable_prefix_cache = enabled == "1";
  }

  xllm::load_native_libraries();
  std::string master_node_addr = std::string(argv[1]);
  int32_t local_rank = static_cast<int32_t>(atoi(argv[2]));
  int32_t global_rank = static_cast<int32_t>(atoi(argv[3]));
  int32_t world_size = static_cast<int32_t>(atoi(argv[4]));
  int32_t device_idx = static_cast<int32_t>(atoi(argv[5]));
  int32_t block_size = static_cast<int32_t>(atoi(argv[6]));
  int32_t max_tokens_per_batch = static_cast<int32_t>(atoi(argv[7]));
  int32_t max_seqs_per_batch = static_cast<int32_t>(atoi(argv[8]));
  int32_t enable_shm = static_cast<int32_t>(atoi(argv[9]));
  int32_t is_local = static_cast<int32_t>(atoi(argv[10]));
  std::string task_type = std::string(argv[11]);
  std::string worker_type = std::string(argv[12]);
  int32_t enable_speculative_decode = static_cast<int32_t>(atoi(argv[13]));
  int32_t num_speculative_tokens = static_cast<int32_t>(atoi(argv[14]));
  std::string speculative_algorithm = std::string(argv[15]);
  uint64_t input_shm_size = static_cast<uint64_t>(atoll(argv[16]));
  uint64_t output_shm_size = static_cast<uint64_t>(atoll(argv[17]));
  std::string communication_backend = std::string(argv[18]);
  std::string npu_kernel_backend = std::string(argv[19]);
  std::string rank_tablefile = std::string(argv[20]);
  bool enable_graph = static_cast<int32_t>(atoi(argv[21])) > 0;
  bool enable_graph_mode_decode_no_padding =
      static_cast<int32_t>(atoi(argv[22])) > 0;
  bool enable_prefill_piecewise_graph =
      static_cast<int32_t>(atoi(argv[23])) > 0;
  int32_t max_tokens_for_graph_mode = static_cast<int32_t>(atoi(argv[24]));
  int64_t max_encoder_cache_size = static_cast<int64_t>(atoll(argv[25]));
  int32_t dp_size = static_cast<int32_t>(atoi(argv[26]));
  int32_t ep_size = static_cast<int32_t>(atoi(argv[31]));
  int32_t cp_size = static_cast<int32_t>(atoi(argv[30]));
  const int32_t kv_split_size =
      argc > xllm::spawn_worker_protocol::kKvSplitSizeArgumentIndex
          ? static_cast<int32_t>(atoi(
                argv[xllm::spawn_worker_protocol::kKvSplitSizeArgumentIndex]))
          : 1;
  int32_t tp_size = static_cast<int32_t>(atoi(argv[27]));
  int32_t sp_size = static_cast<int32_t>(atoi(argv[28]));
  int32_t cfg_size = static_cast<int32_t>(atoi(argv[29]));
  const int32_t vae_size =
      argc > xllm::spawn_worker_protocol::kVaeSizeArgumentIndex
          ? static_cast<int32_t>(
                atoi(argv[xllm::spawn_worker_protocol::kVaeSizeArgumentIndex]))
          : 1;
  const int32_t text_encoder_tp_size =
      argc > xllm::spawn_worker_protocol::kTextEncoderTpSizeArgumentIndex
          ? static_cast<int32_t>(
                atoi(argv[xllm::spawn_worker_protocol::
                              kTextEncoderTpSizeArgumentIndex]))
          : 1;
  std::string instance_role_str = std::string(argv[32]);
  const std::string& indexer_cache_dtype = parsed_indexer_cache_dtype.value();
  const bool enable_probabilistic_draft =
      xllm::spawn_worker_protocol::parse_bool_flag(
          argc,
          argv,
          xllm::spawn_worker_protocol::kEnableProbabilisticDraftArgumentIndex);
  const std::string model_impl =
      xllm::spawn_worker_protocol::parse_model_impl(argc, argv);
  const std::string extension_impl =
      xllm::spawn_worker_protocol::parse_extension_impl(argc, argv);
  const bool enable_mtp_draft_body_tp1 =
      xllm::spawn_worker_protocol::parse_bool_flag(
          argc,
          argv,
          xllm::spawn_worker_protocol::kEnableMtpDraftBodyTp1ArgumentIndex);
  if (world_size < 1 || global_rank < 0 || global_rank >= world_size ||
      cp_size < 1 || ep_size < 1 || text_encoder_tp_size < 1 ||
      kv_split_size < 0 || vae_size < 1 || world_size % vae_size != 0 ||
      (instance_role_str != "DEFAULT" && instance_role_str != "PREFILL" &&
       instance_role_str != "DECODE")) {
    LOG(ERROR) << "Invalid spawn worker topology: global_rank=" << global_rank
               << ", world_size=" << world_size << ", cp_size=" << cp_size
               << ", ep_size=" << ep_size << ", kv_split_size=" << kv_split_size
               << ", instance_role=" << instance_role_str;
    return 1;
  }
  const xllm::InstanceRole instance_role(instance_role_str);

  LOG(INFO)
      << "Spawn worker: "
      << "master_node_addr = " << master_node_addr
      << ", local_rank = " << local_rank << ", world_size = " << world_size
      << ", device_idx = " << device_idx << ", block_size = " << block_size
      << ", max_tokens_per_batch = " << max_tokens_per_batch
      << ", max_seqs_per_batch = " << max_seqs_per_batch
      << ", enable_shm = " << (enable_shm > 0)
      << ", input_shm_size = " << input_shm_size
      << ", output_shm_size = " << output_shm_size
      << ", is_local = " << (is_local > 0) << ", global_rank = " << global_rank
      << ", cp_size = " << cp_size << ", ep_size = " << ep_size
      << ", instance_role = " << instance_role.to_string()
      << ", task_type = " << task_type << ", worker_type = " << worker_type
      << ", enable_speculative_decode = " << (enable_speculative_decode > 0)
      << ", num_speculative_tokens = " << num_speculative_tokens
      << ", speculative_algorithm = " << speculative_algorithm
      << ", communication_backend = " << communication_backend
      << ", npu_kernel_backend = " << npu_kernel_backend
      << ", rank_tablefile = " << rank_tablefile
      << ", enable_graph = " << enable_graph
      << ", enable_graph_double_buffer = " << enable_graph_double_buffer.value()
      << ", acl_graph_decode_batch_size_limit = "
      << graph_decode_batch_size_limit.value()
      << ", enable_graph_mode_decode_no_padding = "
      << enable_graph_mode_decode_no_padding
      << ", enable_prefill_piecewise_graph = " << enable_prefill_piecewise_graph
      << ", max_tokens_for_graph_mode = " << max_tokens_for_graph_mode
      << ", max_encoder_cache_size = " << max_encoder_cache_size
      << ", dp_size = " << dp_size << ", tp_size = " << tp_size
      << ", sp_size = " << sp_size << ", cfg_size = " << cfg_size
      << ", text_encoder_tp_size = " << text_encoder_tp_size
      << ", indexer_cache_dtype = " << indexer_cache_dtype
      << ", enable_mtp_draft_body_tp1 = " << enable_mtp_draft_body_tp1
      << ", enable_probabilistic_draft = " << enable_probabilistic_draft
      << ", kv_split_size = " << kv_split_size
      << ", model_impl = " << model_impl
      << ", extension_impl = " << extension_impl
      << ", enable_eplb = " << eplb_worker_config->enabled
      << ", redundant_experts_num = "
      << eplb_worker_config->redundant_experts_num
      << ", use_decode_only_load = "
      << eplb_worker_config->use_decode_only_load;

#if defined(USE_NPU)
  xllm::ModelConfig::get_instance().model_impl(model_impl);
  pybind11::scoped_interpreter interpreter(/*init_signal_handlers=*/false);
  if (!xllm::initialize_npu_startup(device_idx)) {
    return 1;
  }
  pybind11::gil_scoped_release release;
#endif

  {
    xllm::SpawnWorkerServer worker(master_node_addr,
                                   local_rank,
                                   global_rank,
                                   world_size,
                                   device_idx,
                                   block_size,
                                   indexer_cache_dtype,
                                   max_tokens_per_batch,
                                   max_seqs_per_batch,
                                   enable_shm > 0,
                                   input_shm_size,
                                   output_shm_size,
                                   is_local > 0,
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
                                   ep_size,
                                   cp_size,
                                   kv_split_size,
                                   tp_size,
                                   sp_size,
                                   cfg_size,
                                   vae_size,
                                   text_encoder_tp_size,
                                   instance_role,
                                   enable_mtp_draft_body_tp1,
                                   enable_probabilistic_draft,
                                   model_impl,
                                   extension_impl,
                                   std::move(eplb_worker_config.value()),
                                   prefill_config.value(),
                                   enable_graph_double_buffer.value(),
                                   graph_decode_batch_size_limit.value(),
                                   topology_config.value(),
                                   enable_graph_vmm_pool.value(),
                                   enable_prefix_cache);

    worker.run();
  }

#if defined(USE_NPU)
  xllm::MooncakeTransferEngineCore::get_instance().shutdown();
#endif

  return 0;
}
