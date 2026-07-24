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

#include <folly/init/Init.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/torch.h>

#include <algorithm>
#include <csignal>
#include <filesystem>
#include <memory>
#include <random>

#include "api_service/api_service.h"
#include "core/common/global_flags.h"
#include "core/common/instance_name.h"
#include "core/common/metrics.h"
#include "core/common/options.h"
#include "core/common/types.h"
#include "core/distributed_runtime/dit_master.h"
#include "core/distributed_runtime/master.h"
#include "core/distributed_runtime/vlm_master.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/config_utils.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/distributed_config.h"
#include "core/framework/config/dit_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/help_formatter.h"
#include "core/framework/config/kernel_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/parallel_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/rec_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/service_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/xtensor/global_xtensor.h"
#include "core/framework/xtensor/options.h"
#include "core/framework/xtensor/xtensor_allocator.h"
#include "core/platform/device_name_utils.h"
#include "core/util/net.h"
#include "core/util/utils.h"
#include "function_call/function_call_parser.h"
#include "parser/reasoning_parser.h"
#include "server/xllm_server_registry.h"
using namespace xllm;

static std::atomic<uint32_t> signal_received{0};

namespace {

void initialize_configs() {
  BeamSearchConfig::get_instance().initialize();
  DisaggPDConfig::get_instance().initialize();
  DistributedConfig::get_instance().initialize();
  DiTConfig::get_instance().initialize();
  EPLBConfig::get_instance().initialize();
  ExecutionConfig::get_instance().initialize();
  KernelConfig::get_instance().initialize();
  KVCacheConfig::get_instance().initialize();
  KVCacheStoreConfig::get_instance().initialize();
  LoadConfig::get_instance().initialize();
  ModelConfig::get_instance().initialize();
  ParallelConfig::get_instance().initialize();
  ProfileConfig::get_instance().initialize();
  RecConfig::get_instance().initialize();
  SchedulerConfig::get_instance().initialize();
  ServiceConfig::get_instance().initialize();
  SpeculativeConfig::get_instance().initialize();

  // Reconcile the two per-batch admission caps into a single scheduler view
  // consumed by every downstream user (Master -> Engine -> BlockManagerPool
  // etc.). max_seqs_per_batch bounds the scheduler batch;
  // max_concurrent_requests bounds the service-level admission; the effective
  // batch cap is the tighter of the two. 0 means "unset" on either side; if
  // both are 0 the caller has no way to size batch-bound resources (e.g. the
  // SINGLE block pool) and we fail early.
  {
    SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
    ServiceConfig& service_config = ServiceConfig::get_instance();
    const int32_t scheduler_cap = scheduler_config.max_seqs_per_batch();
    const int32_t service_cap = service_config.max_concurrent_requests();
    int32_t effective_cap = 0;
    if (scheduler_cap > 0 && service_cap > 0) {
      effective_cap = std::min(scheduler_cap, service_cap);
    } else if (scheduler_cap > 0) {
      effective_cap = scheduler_cap;
    } else if (service_cap > 0) {
      effective_cap = service_cap;
    } else {
      LOG(FATAL) << "Both max_seqs_per_batch and max_concurrent_requests are "
                    "0; set at least one to a positive value.";
    }
    scheduler_config.max_seqs_per_batch(effective_cap);
    service_config.max_concurrent_requests(effective_cap);
  }
}

Options create_options(const std::string& instance_name, bool is_local) {
  const ServiceConfig& service_config = ServiceConfig::get_instance();
  const ModelConfig& model_config = ModelConfig::get_instance();
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  const KVCacheStoreConfig& kv_cache_store_config =
      KVCacheStoreConfig::get_instance();
  const BeamSearchConfig& beam_search_config = BeamSearchConfig::get_instance();
  const SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  const ParallelConfig& parallel_config = ParallelConfig::get_instance();
  const EPLBConfig& eplb_config = EPLBConfig::get_instance();
  const DistributedConfig& distributed_config =
      DistributedConfig::get_instance();
  const DisaggPDConfig& disagg_pd_config = DisaggPDConfig::get_instance();
  const SpeculativeConfig& speculative_config =
      SpeculativeConfig::get_instance();
  const ProfileConfig& profile_config = ProfileConfig::get_instance();
  const ExecutionConfig& execution_config = ExecutionConfig::get_instance();
  const KernelConfig& kernel_config = KernelConfig::get_instance();
  const DiTConfig& dit_config = DiTConfig::get_instance();
  const RecConfig& rec_config = RecConfig::get_instance();

#if !defined(USE_NPU)
  CHECK(!speculative_config.enable_mtp_draft_body_tp1())
      << "enable_mtp_draft_body_tp1 is only supported on the NPU backend";
#endif

  Options options;
#if defined(USE_NPU)
  options.npu_kernel_backend(kernel_config.npu_kernel_backend());
  options.enable_flashcomm1(kernel_config.enable_flashcomm1())
      .flashcomm1_min_prefill_tokens(
          kernel_config.flashcomm1_min_prefill_tokens())
      .enable_mmrs_fusion(kernel_config.enable_mmrs_fusion())
      .mmrs_comm_mode(kernel_config.mmrs_comm_mode());
#endif
  options.model_path(model_config.model())
      .model_id(model_config.model_id())
      .task_type(model_config.task())
      .draft_model_path(speculative_config.draft_model())
      .backend(model_config.backend())
      .limit_image_per_prompt(model_config.limit_image_per_prompt())
      .max_encoder_cache_size(model_config.max_encoder_cache_size())
      .block_size(kv_cache_config.block_size())
      .max_cache_size(kv_cache_config.max_cache_size())
      .max_memory_utilization(kv_cache_config.max_memory_utilization())
      .enable_prefix_cache(kv_cache_config.enable_prefix_cache())
      .max_linear_state_cache_slots(
          kv_cache_config.max_linear_state_cache_slots())
      .max_tokens_per_batch(scheduler_config.max_tokens_per_batch())
      .max_seqs_per_batch(scheduler_config.max_seqs_per_batch())
      .max_tokens_per_chunk_for_prefill(
          scheduler_config.max_tokens_per_chunk_for_prefill())
      .num_speculative_tokens(speculative_config.num_speculative_tokens())
      .speculative_algorithm(speculative_config.speculative_algorithm())
      .speculative_suffix_cache_max_depth(
          speculative_config.speculative_suffix_cache_max_depth())
      .speculative_suffix_max_spec_factor(
          speculative_config.speculative_suffix_max_spec_factor())
      .speculative_suffix_max_spec_offset(
          speculative_config.speculative_suffix_max_spec_offset())
      .speculative_suffix_min_token_prob(
          speculative_config.speculative_suffix_min_token_prob())
      .speculative_suffix_max_cached_requests(
          speculative_config.speculative_suffix_max_cached_requests())
      .speculative_suffix_use_tree_spec(
          speculative_config.speculative_suffix_use_tree_spec())
      .enable_mtp_draft_body_tp1(speculative_config.enable_mtp_draft_body_tp1())
      .num_request_handling_threads(
          service_config.num_request_handling_threads())
      .communication_backend(parallel_config.communication_backend())
      .enable_eplb(eplb_config.enable_eplb())
      .redundant_experts_num(eplb_config.redundant_experts_num())
      .eplb_update_interval(eplb_config.eplb_update_interval())
      .eplb_update_threshold(eplb_config.eplb_update_threshold())
      .rank_tablefile(eplb_config.rank_tablefile())
      .expert_parallel_degree(eplb_config.expert_parallel_degree())
      .enable_chunked_prefill(scheduler_config.enable_chunked_prefill())
      .master_node_addr(distributed_config.master_node_addr())
      .instance_role(InstanceRole(disagg_pd_config.instance_role()))
      .transfer_listen_port(
          static_cast<uint16_t>(disagg_pd_config.transfer_listen_port()))
      .nnodes(distributed_config.nnodes())
      .node_rank(distributed_config.node_rank())
      .dp_size(parallel_config.dp_size())
      .cp_size(parallel_config.cp_size())
      .ep_size(parallel_config.ep_size())
      .tp_size(static_cast<int32_t>(parallel_config.tp_size()))
      .sp_size(static_cast<int32_t>(parallel_config.sp_size()))
      .cfg_size(static_cast<int32_t>(parallel_config.cfg_size()))
      .vae_size(static_cast<int32_t>(parallel_config.vae_size()))
      .instance_name(instance_name)
      .enable_disagg_pd(disagg_pd_config.enable_disagg_pd())
      .enable_pd_ooc(disagg_pd_config.enable_pd_ooc())
      .enable_schedule_overlap(scheduler_config.enable_schedule_overlap())
      .kv_cache_transfer_mode(disagg_pd_config.kv_cache_transfer_mode())
      .etcd_addr(distributed_config.etcd_addr())
      .etcd_namespace(distributed_config.etcd_namespace())
      .enable_service_routing(distributed_config.enable_service_routing() ||
                              disagg_pd_config.enable_disagg_pd())
      .tool_call_parser(model_config.tool_call_parser())
      .reasoning_parser(model_config.reasoning_parser())
      .priority_strategy(scheduler_config.priority_strategy())
      .enable_online_preempt_offline(
          scheduler_config.enable_online_preempt_offline())
      .host_blocks_factor(kv_cache_store_config.host_blocks_factor())
      .enable_kvcache_store(kv_cache_store_config.enable_kvcache_store() &&
                            kv_cache_config.enable_prefix_cache() &&
                            (kv_cache_store_config.host_blocks_factor() > 1.0))
      .prefetch_timeout(kv_cache_store_config.prefetch_timeout())
      .prefetch_batch_size(kv_cache_store_config.prefetch_batch_size())
      .layers_wise_copy_batchs(kv_cache_store_config.layers_wise_copy_batchs())
      .store_protocol(kv_cache_store_config.store_protocol())
      .store_master_server_address(
          kv_cache_store_config.store_master_server_address())
      .store_metadata_server(kv_cache_store_config.store_metadata_server())
      .store_local_hostname(kv_cache_store_config.store_local_hostname())
      .enable_multi_stream_parallel(
          parallel_config.enable_multi_stream_parallel())
      .enable_profile_step_time(profile_config.enable_profile_step_time())
      .enable_profile_token_budget(profile_config.enable_profile_token_budget())
      .enable_latency_aware_schedule(
          profile_config.enable_latency_aware_schedule())
      .profile_max_prompt_length(profile_config.profile_max_prompt_length())
      .enable_profile_kv_blocks(profile_config.enable_profile_kv_blocks())
      .disable_ttft_profiling(profile_config.disable_ttft_profiling())
      .enable_forward_interruption(profile_config.enable_forward_interruption())
      .enable_graph(execution_config.enable_graph())
      .enable_graph_mode_decode_no_padding(
          execution_config.enable_graph_mode_decode_no_padding())
      .enable_prefill_piecewise_graph(
          execution_config.enable_prefill_piecewise_graph())
      .max_tokens_for_graph_mode(execution_config.max_tokens_for_graph_mode())
      .max_global_ttft_ms(profile_config.max_global_ttft_ms())
      .max_global_tpot_ms(profile_config.max_global_tpot_ms())
      .max_requests_per_batch(dit_config.max_requests_per_batch())
      .enable_shm(execution_config.enable_shm())
      .input_shm_size(execution_config.input_shm_size())
      .output_shm_size(execution_config.output_shm_size())
      .beam_width(beam_search_config.beam_width())
      .kv_cache_dtype(kv_cache_config.kv_cache_dtype())
      .rec_worker_max_concurrency(
          static_cast<int32_t>(rec_config.rec_worker_max_concurrency()))
      .is_local(is_local);

  return options;
}

}  // namespace

void shutdown_handler(int signal) {
  // TODO: gracefully shutdown the server
  LOG(WARNING) << "Received signal " << signal << ", stopping server...";
  exit(1);
}

void validate_config(const std::string& model_type) {
  ModelConfig& model_config = ModelConfig::get_instance();
  LoadConfig& load_config = LoadConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  DisaggPDConfig& disagg_pd_config = DisaggPDConfig::get_instance();
  SpeculativeConfig& speculative_config = SpeculativeConfig::get_instance();
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();

  if (model_config.backend().empty()) {
    LOG(FATAL) << "Model is not supported currently, model type: "
               << model_type;
  }
  if (model_config.max_encoder_cache_size() < 0) {
    LOG(FATAL) << "max_encoder_cache_size must be >= 0.";
  }
#if defined(USE_MLU)
  // Disable enable_schedule_overlap for VLM models on MLU backend
  if (scheduler_config.enable_schedule_overlap() &&
      model_config.backend() == "vlm") {
    LOG(WARNING) << "enable_schedule_overlap is not supported for VLM models "
                    "on MLU backend. "
                 << "Disabling enable_schedule_overlap.";
    scheduler_config.enable_schedule_overlap(false);
  }
  // TODO: support other block sizes in the future
  if (kv_cache_config.block_size() != 16 && kv_cache_config.block_size() != 1 &&
      model_config.backend() != "dit") {
    LOG(FATAL) << "Currently, block_size must be 16 for MLU backend, we will "
                  "support other block sizes in the future.";
  }
  if (disagg_pd_config.enable_disagg_pd()) {
    if (model_config.backend() != "llm") {
      LOG(FATAL) << "MLU disaggregated PD only supports backend=llm.";
    }
    disagg_pd_config.normalize_mlu(kv_cache_config, scheduler_config);
  }
#endif

#if defined(USE_DCU)
  if (disagg_pd_config.enable_disagg_pd()) {
    if (scheduler_config.enable_schedule_overlap()) {
      LOG(WARNING) << "enable_schedule_overlap is not supported for "
                      "disaggregated PD on DCU backend. "
                   << "Disabling enable_schedule_overlap.";
      scheduler_config.enable_schedule_overlap(false);
    }
    if (model_config.backend() != "llm") {
      LOG(FATAL) << "DCU disaggregated PD only supports backend=llm.";
    }
    disagg_pd_config.normalize_dcu(scheduler_config);
  }
#endif

#if defined(USE_NPU)
  if (speculative_config.num_speculative_tokens() > 0 &&
      execution_config.enable_graph_double_buffer()) {
    LOG(WARNING) << "enable_graph_double_buffer is not compatible with "
                    "speculative decoding. "
                 << "Disabling enable_graph_double_buffer.";
    execution_config.enable_graph_double_buffer(false);
  }
  // enable_xtensor / enable_rolling_load imply enable_manual_loader
  if ((kv_cache_config.enable_xtensor() || load_config.enable_rolling_load()) &&
      !load_config.enable_manual_loader()) {
    LOG(WARNING) << "enable_xtensor or enable_rolling_load requires "
                    "enable_manual_loader; forcing enable_manual_loader=true.";
    load_config.enable_manual_loader(true);
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_cached_layers() < 1) {
    LOG(FATAL) << "rolling_load_num_cached_layers must be >= 1.";
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_rolling_slots() < -1) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be >= -1.";
  }
  if (load_config.enable_rolling_load() &&
      load_config.rolling_load_num_rolling_slots() >= 0 &&
      load_config.rolling_load_num_rolling_slots() >
          load_config.rolling_load_num_cached_layers()) {
    LOG(FATAL) << "rolling_load_num_rolling_slots must be <= "
               << "rolling_load_num_cached_layers.";
  }
#else
  if (kv_cache_config.enable_xtensor()) {
    LOG(FATAL) << "enable_xtensor is only supported on NPU.";
  }
  if (load_config.enable_manual_loader()) {
    LOG(FATAL) << "enable_manual_loader is only supported on NPU.";
  }
  if (load_config.enable_rolling_load()) {
    LOG(FATAL) << "enable_rolling_load is only supported on NPU.";
  }
#endif

  model_config.normalize_cpp_chat_template(model_type);
}

int run() {
  ModelConfig& model_config = ModelConfig::get_instance();
  KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  BeamSearchConfig& beam_search_config = BeamSearchConfig::get_instance();
  SchedulerConfig& scheduler_config = SchedulerConfig::get_instance();
  ParallelConfig& parallel_config = ParallelConfig::get_instance();
  DistributedConfig& distributed_config = DistributedConfig::get_instance();
  ServiceConfig& service_config = ServiceConfig::get_instance();
  ExecutionConfig& execution_config = ExecutionConfig::get_instance();

  // check if model path exists
  if (!std::filesystem::exists(model_config.model())) {
    LOG(FATAL) << "Model path " << model_config.model() << " does not exist.";
  }

  std::filesystem::path model_path =
      std::filesystem::path(model_config.model()).lexically_normal();
  const std::string default_model_name = xllm::util::get_model_name(model_path);

  if (model_config.model_id().empty()) {
    // use last part of the path as model id
    model_config.model_id(default_model_name);
  }

  if (model_config.backend().empty()) {
    model_config.backend(xllm::util::get_model_backend(model_path));
  }

  if (service_config.host().empty()) {
    // set the host to the local IP when the host is empty
    service_config.host(net::get_local_ip_addr());
  }

  const bool is_local =
      !service_config.host().empty() &&
      net::extract_ip(distributed_config.master_node_addr()) ==
          service_config.host();

  LOG(INFO) << "set worker role to "
            << (is_local ? "local worker" : "remote worker");

  // if max_tokens_per_chunk_for_prefill is not set, set its value to
  // max_tokens_per_batch
  if (scheduler_config.max_tokens_per_chunk_for_prefill() < 0) {
    scheduler_config.max_tokens_per_chunk_for_prefill(
        scheduler_config.max_tokens_per_batch());
  }

// disable block copy kernel on unsupported backends
#if !defined(USE_NPU) && !defined(USE_CUDA)
  beam_search_config.enable_block_copy_kernel(false);
#endif
  std::string model_type = "";
  if (model_config.backend() != "dit") {
    model_type = xllm::util::get_model_type(model_path, model_config.backend());
    model_config.tool_call_parser(
        function_call::FunctionCallParser::get_parser_auto(
            model_config.tool_call_parser(), model_type));
    model_config.reasoning_parser(ReasoningParser::get_parser_auto(
        model_config.reasoning_parser(), model_type));
  }

  // validate config before creating master
  validate_config(model_type);

  if (distributed_config.node_rank() == 0 &&
      execution_config.random_seed() < 0) {
    execution_config.random_seed(std::random_device{}() % (1 << 30));
  }

  if (distributed_config.node_rank() == 0) {
    config::dump_startup_config();
  }

  // Create Master
  Options options = create_options(
      service_config.host() + ":" + std::to_string(service_config.port()),
      is_local);

  InstanceName::name()->set_name(options.instance_name().value_or(""));

  // master node
  // init XTensor allocator and PhyPagePool for xtensor mode
  if (kv_cache_config.enable_xtensor()) {
    // Parse devices
    const auto devices = DeviceNameUtils::parse_devices("auto");

    // Initialize XTensorAllocator with first device
    auto& allocator = XTensorAllocator::get_instance();
    allocator.init(devices[0]);

    // Setup distributed XTensor service for multi-GPU/multi-node
    if (distributed_config.nnodes() > 1) {
      xtensor::Options xtensor_options;
      xtensor_options.devices(devices)
          .nnodes(distributed_config.nnodes())
          .node_rank(distributed_config.node_rank());
      allocator.setup_multi_node_xtensor_dist(
          xtensor_options,
          distributed_config.xtensor_master_node_addr(),
          parallel_config.dp_size());
    }

    // Initialize PhyPagePool on all workers
    int64_t num_pages =
        allocator.init_phy_page_pools(kv_cache_config.max_memory_utilization(),
                                      kv_cache_config.max_cache_size());
    if (num_pages <= 0) {
      LOG(FATAL) << "Failed to initialize PhyPagePool";
    }
    LOG(INFO) << "XTensor initialized with " << num_pages << " physical pages";
  }

  std::unique_ptr<Master> master;
  // working node
  if (options.node_rank() != 0) {
    if (model_config.backend() == "dit") {
      master = std::make_unique<DiTAssistantMaster>(options);
    } else if (model_config.backend() == "vlm") {
      master = std::make_unique<VLMAssistantMaster>(options);
    } else {
      master = std::make_unique<LLMAssistantMaster>(options);
    }
  } else {
    // master node
    master = create_master(model_config.backend(), options);
  }
  master->run();

  // supported models
  std::vector<std::string> model_names = {model_config.model_id()};
  std::string model_version = default_model_name;
  std::vector<std::string> model_versions = {model_version};

  if (distributed_config.node_rank() == 0 || kv_cache_config.enable_xtensor()) {
    auto api_service =
        std::make_unique<APIService>(master.get(), model_names, model_versions);
    auto xllm_server =
        ServerRegistry::get_instance().register_server("HttpServer");

    // start brpc server
    if (!xllm_server->start(std::move(api_service))) {
      LOG(ERROR) << "Failed to start brpc server on port "
                 << service_config.port();
      return -1;
    }
  }

  return 0;
}

int main(int argc, char** argv) {
  // Check for --help flag before parsing other flags
  for (int i = 1; i < argc; ++i) {
    std::string arg(argv[i]);
    if (arg == "--help" || arg == "-h") {
      HelpFormatter::print_help();
      return 0;
    }
  }

  FLAGS_alsologtostderr = true;
  FLAGS_minloglevel = 0;
  google::ParseCommandLineFlags(&argc, &argv, true);
  google::InitGoogleLogging("xllm");
  initialize_configs();

  // Check if model path is provided
  if (::xllm::ModelConfig::get_instance().model().empty()) {
    HelpFormatter::print_error("--model flag is required");
    return 1;
  }

  return run();
}
