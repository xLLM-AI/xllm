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

#include "worker_impl.h"

#include <ATen/Parallel.h>
#include <folly/Unit.h>
#include <folly/futures/Future.h>
#include <gflags/gflags.h>
#include <glog/logging.h>
#include <torch/torch.h>

#include <algorithm>
#if defined(USE_NPU)
#include "acl/acl.h"
#include "kernels/npu/xllm_ops/xllm_ops_api.h"
#elif defined(USE_MLU)
#include <framework/core/caching_allocator.h>
#elif defined(USE_CUDA) || defined(USE_ILU)
#include <c10/cuda/CUDACachingAllocator.h>
#endif

#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "common/device_monitor.h"
#include "common/global_flags.h"
#include "common/metrics.h"
#include "core/framework/config/beam_search_config.h"
#include "core/framework/config/disagg_pd_config.h"
#include "core/framework/config/eplb_config.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/load_config.h"
#include "core/framework/config/profile_config.h"
#include "core/framework/config/scheduler_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/kv_cache/kv_cache_estimation.h"
#include "core/platform/platform.h"
#include "core/platform/sleepable_allocator.h"
#if defined(USE_NPU)
#include "platform/npu/device_capture_lock.h"
#elif defined(USE_CUDA) || defined(USE_DCU)
#include "kernels/cuda/cuda_ops_api.h"
#include "platform/cuda_profiler.h"
#include "platform/torch_profiler.h"
#endif
#include "core/distributed_runtime/master.h"
#include "core/runtime/worker_rendezvous.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/kv_cache/linear_state_restore.h"
#include "framework/model/model_input_params.h"
#include "framework/model_loader.h"
#include "framework/parallel_state/npu_cp_ep_padding.h"
#include "framework/sampling/sampler.h"
#include "framework/state_dict/state_dict.h"
#include "framework/xtensor/global_xtensor.h"
#include "framework/xtensor/xtensor_allocator.h"
#include "runtime/cp_input_partition.h"
#if defined(USE_NPU)
#include "layers/npu/loader/rolling_weight_buffer.h"
#endif
#include "util/json_reader.h"
#include "util/tensor_helper.h"
#include "util/threadpool.h"
#include "util/timer.h"
#include "util/utils.h"

#define USE_ASYNC true

namespace xllm {

constexpr uint64_t MBUF_SIZE = 128 * 1024 * 1024;
constexpr uint32_t BATCH_COPY_MAX_SIZE = 4096;
constexpr uint32_t TIMEOUT_S = 60;      // second
constexpr uint32_t TIMEOUT_MS = 60000;  // millisecond

namespace {

// During TP model initialization, each rank loads weights concurrently.
// MoE weight assembly (especially stack/cat on large expert tensors) runs on
// CPU and is backed by ATen intra-op thread pools.
//
// If every TP rank also uses many intra-op threads, we get severe CPU
// oversubscription and memory-bandwidth contention:
//   1) many processes run large stack/cat at the same time
//   2) each process fans out into multiple CPU workers
//   3) host-side contention dominates load time even when I/O is fast
//
// For the weight-loading window only, forcing ATen to 1 thread reduces this
// cross-rank contention and usually lowers end-to-end load latency in TP mode.
// We restore the previous thread count immediately after load_model() returns,
// so runtime compute behavior remains unchanged.
class ScopedAtenLoadThreads {
 public:
  explicit ScopedAtenLoadThreads(int32_t target_threads)
      : prev_threads_(at::get_num_threads()) {
    if (target_threads > 0 && prev_threads_ != target_threads) {
      torch::set_num_threads(target_threads);
      active_ = true;
    }
  }

  ~ScopedAtenLoadThreads() {
    if (active_) {
      torch::set_num_threads(prev_threads_);
    }
  }

  // Non-copyable and non-movable
  ScopedAtenLoadThreads(const ScopedAtenLoadThreads&) = delete;
  ScopedAtenLoadThreads& operator=(const ScopedAtenLoadThreads&) = delete;
  ScopedAtenLoadThreads(ScopedAtenLoadThreads&&) = delete;
  ScopedAtenLoadThreads& operator=(ScopedAtenLoadThreads&&) = delete;

 private:
  int32_t prev_threads_ = 0;
  bool active_ = false;
};

// DFlash draft config lists target_layer_ids as the target-model layer indices
// (0-based) whose output the draft consumes. xLLM's capture hook fires BEFORE
// layer i runs, so capturing layer L's output means putting L+1 in the capture
// set (matched against layer index i in the forward loop). Returns those L+1
// ids.
std::vector<int32_t> read_dflash_capture_layer_ids(
    const std::string& model_weights_path) {
  JsonReader reader;
  const std::string config_path = model_weights_path + "/config.json";
  CHECK(reader.parse(config_path))
      << "Failed to parse DFlash config: " << config_path;
  std::vector<int32_t> capture_layer_ids;
  for (int32_t layer_id : reader.value_or<std::vector<int32_t>>(
           "dflash_config.target_layer_ids", std::vector<int32_t>{})) {
    capture_layer_ids.emplace_back(layer_id + 1);
  }
  return capture_layer_ids;
}

void move_tensor_to_device_if_needed(torch::Tensor& tensor,
                                     const torch::Device& device) {
  if (tensor.defined() && tensor.device() != device) {
    tensor = tensor.to(device, /*non_blocking=*/false).contiguous();
  }
}

// ForwardInput::to(device) returns early when device_tensors_ready is set.
// Nested step_async (e.g. MTP target/draft) can leave CP-remapped control
// tensors on CPU while model tensors are already on NPU.
void ensure_forward_input_device_tensors(ForwardInput& input,
                                         const torch::Device& device) {
  move_tensor_to_device_if_needed(input.token_ids, device);
  move_tensor_to_device_if_needed(input.positions, device);
  move_tensor_to_device_if_needed(
      input.input_params.embedding.mtp_shifted_token_ids, device);
  move_tensor_to_device_if_needed(input.sampling_params.selected_token_idxes,
                                  device);
  move_tensor_to_device_if_needed(input.sampling_params.sample_idxes, device);
  move_tensor_to_device_if_needed(
      input.decoder_sampling_params.selected_token_idxes, device);
  move_tensor_to_device_if_needed(input.decoder_sampling_params.sample_idxes,
                                  device);
}

#if defined(USE_NPU)
void prepare_input_params_for_linear_attention(ModelInputParams& input_params) {
  const std::vector<int32_t>& host_q_seq_lens =
      input_params.attention.host.q_seq_lens;
  const bool has_leading_zero =
      !host_q_seq_lens.empty() && host_q_seq_lens.front() == 0 &&
      host_q_seq_lens.size() ==
          static_cast<size_t>(input_params.meta.num_sequences + 1);
  int64_t batch_size = static_cast<int64_t>(
      has_leading_zero ? host_q_seq_lens.size() - 1 : host_q_seq_lens.size());
  if (batch_size == 0) {
    batch_size = input_params.meta.num_sequences;
  }
  if (batch_size == 0 && input_params.attention.device.block_tables.defined()) {
    batch_size = input_params.attention.device.block_tables.size(0);
  }
  input_params.parallel.query_start_loc.resize(batch_size + 1, 0);
  for (int64_t i = 0; i < batch_size; ++i) {
    int64_t seq_len =
        has_leading_zero
            ? static_cast<int64_t>(host_q_seq_lens[static_cast<size_t>(i + 1)] -
                                   host_q_seq_lens[static_cast<size_t>(i)])
            : static_cast<int64_t>(host_q_seq_lens[static_cast<size_t>(i)]);
    input_params.parallel.query_start_loc[i + 1] =
        input_params.parallel.query_start_loc[i] + seq_len;
  }

  torch::Tensor has_initial_state_tensor =
      input_params.attention.device.kv_cache_tokens_nums > 0;
  torch::Tensor has_initial_state_int64 = has_initial_state_tensor.contiguous()
                                              .view({-1})
                                              .to(torch::kCPU)
                                              .to(torch::kInt64);
  const int64_t has_initial_state_size = has_initial_state_int64.size(0);
  CHECK_GT(has_initial_state_size, 0)
      << "kv_cache_tokens_nums must not be empty for linear attention";
  CHECK(batch_size == has_initial_state_size ||
        batch_size % has_initial_state_size == 0)
      << "kv_cache_tokens_nums size must match or evenly divide active batch "
      << "size, kv_cache_tokens_nums_size=" << has_initial_state_size
      << ", batch_size=" << batch_size;
  if (batch_size == has_initial_state_size) {
    input_params.parallel.has_initial_state = std::vector<int64_t>(
        has_initial_state_int64.data_ptr<int64_t>(),
        has_initial_state_int64.data_ptr<int64_t>() + batch_size);
    return;
  }

  const int64_t repeat_count = batch_size / has_initial_state_size;
  input_params.parallel.has_initial_state.clear();
  input_params.parallel.has_initial_state.reserve(batch_size);
  const int64_t* has_initial_state_ptr =
      has_initial_state_int64.data_ptr<int64_t>();
  for (int64_t i = 0; i < has_initial_state_size; ++i) {
    for (int64_t repeat_idx = 0; repeat_idx < repeat_count; ++repeat_idx) {
      input_params.parallel.has_initial_state.push_back(
          has_initial_state_ptr[i]);
    }
  }
}
#endif

}  // namespace

WorkerImpl::WorkerImpl(const ParallelArgs& parallel_args,
                       const torch::Device& device,
                       const runtime::Options& options)
    : options_(options), device_(device), parallel_args_(parallel_args) {
  if (options_.enable_speculative_decode() &&
      options_.num_decoding_tokens() == 1) {
    is_spec_draft_ = true;
  }

  // first worker is the driver
  driver_ = parallel_args.rank() == 0;
  int32_t tp_size = parallel_args.world_size() /
                    (parallel_args.dp_size() * parallel_args.cp_size());
  dp_driver_ = parallel_args.dp_size() > 1 &&
               parallel_args.rank() % (tp_size * parallel_args.cp_size()) == 0;

  device_.set_device();
  device_.init_device_context();
  threadpool_.schedule([this]() mutable { device_.set_device(); });
  prepare_stream_ = device_.get_stream_from_pool();
  compute_stream_ = device_.current_stream();
  sampler_ = std::make_unique<Sampler>();

#if !defined(USE_NPU) && !defined(USE_CUDA) && !defined(USE_DCU)
  if (::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel()) {
    LOG(WARNING)
        << "enable_block_copy_kernel is only supported on NPU/CUDA/DCU; "
           "forcing enable_block_copy_kernel=false.";
    ::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel(false);
  }
#endif

#if defined(USE_NPU)
  if (::xllm::KVCacheConfig::get_instance().enable_xtensor()) {
    if (!weight_transfer_) {
      weight_transfer_ = std::make_unique<MooncakeWeightTransfer>(
          options_.transfer_listen_port(), device_.unwrap());
    }
    if (!weight_transfer_->initialize()) {
      LOG(ERROR) << "Failed to initialize MooncakeWeightTransfer";
    }
    if (!weight_transfer_->register_global_xtensor()) {
      LOG(ERROR) << "Failed to register GlobalXTensor";
    }
  }
  if (::xllm::LoadConfig::get_instance().enable_rolling_load()) {
    load_stream_ = device_.get_stream_from_pool();
  }
  worker_rendezvous_ =
      std::make_unique<WorkerRendezvous>(kv_cache_transfer_, weight_transfer_);
#else
  worker_rendezvous_ = std::make_unique<WorkerRendezvous>(kv_cache_transfer_);
#endif
}

WorkerImpl::~WorkerImpl() = default;

bool WorkerImpl::allocate_kv_cache_storage(
    const KVCacheShape& kv_cache_shape,
    bool use_huge_page_allocator,
    std::shared_ptr<KVCacheTensorAllocator> tensor_allocator) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  const bool has_grouped_cache = kv_cache_shape.has_grouped_cache_layout();
  if (has_grouped_cache && options_.enable_disagg_pd()) {
    CHECK_EQ(::xllm::ParallelConfig::get_instance().cp_size(), 1)
        << "Grouped KV cache PD does not support context parallelism.";
    CHECK_EQ(::xllm::ParallelConfig::get_instance().kv_split_size_effective(),
             1)
        << "Grouped KV cache PD does not support KV-split.";
    CHECK_EQ(::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
             "LlmDataDist")
        << "Grouped KV cache PD requires LlmDataDist transfer.";
    CHECK_EQ(options_.kv_cache_transfer_mode(), "PUSH")
        << "Grouped KV cache PD requires PUSH transfer mode.";
    CHECK(!options_.enable_pd_ooc())
        << "Grouped KV cache PD does not support PD-OOC yet.";
  }
  if (has_grouped_cache) {
    CHECK(!::xllm::KVCacheConfig::get_instance().enable_xtensor())
        << "Grouped KV cache layout does not support XTensor cache.";
  }
  const auto& args = context_.get_model_args();
  const bool enable_linear_attention = has_linear_attention_layers(args);
  const bool enable_lighting_indexer = args.index_n_heads() > 0;
  CHECK(!(enable_linear_attention && enable_lighting_indexer))
      << "KVCache does not support linear attention and lighting indexer "
      << "simultaneously.";

  const int64_t num_layers = get_num_layers();
  std::vector<bool> indexer_cache_enabled_layers =
      resolve_indexer_cache_enabled_layers(args, num_layers);

  // Check if KV cache quantization is enabled
  // "auto" (default): cache dtype aligns with model dtype (no quantization)
  // "int8": enables INT8 quantization
  const bool enable_kv_cache_quant = options_.kv_cache_dtype() == "int8";

  const bool enable_indexer_cache_quant =
      ::xllm::KVCacheConfig::get_instance().indexer_cache_dtype() == "int8";
  if (enable_lighting_indexer) {
    CHECK_EQ(kv_cache_shape.has_index_cache_scale_shape(),
             enable_indexer_cache_quant)
        << "Indexer cache shape and worker quantization config must agree.";
  }

  if (enable_kv_cache_quant) {
#if !defined(USE_MLU)
    LOG(FATAL) << "KV Cache quantization is only supported on MLU backend. "
               << "Current backend does not support this feature.";
#endif
    // Check for unsupported scenarios
    if (options_.backend() == "vlm") {
      LOG(FATAL) << "KV Cache quantization is not supported for VLM "
                    "(Vision-Language Model) backend.";
    }
    if (options_.enable_disagg_pd()) {
      LOG(FATAL) << "KV Cache quantization is not supported in PD "
                    "disaggregation mode.";
    }
  }

  // Parse mamba_ssm_dtype if specified for linear attention layers.
  torch::ScalarType ssm_dtype = dtype_;
  if (enable_linear_attention) {
    ssm_dtype = resolve_ssm_dtype(args.mamba_ssm_dtype(), dtype_);
  }

  KVCacheCreateOptions create_options;
  create_options.device(device_)
      .dtype(dtype_)
      .ssm_dtype(ssm_dtype)
      .num_layers(num_layers)
      .full_attention_interval(args.full_attention_interval())
      .model_id(options_.model_id())
      .model_type(args.model_type())
      .enable_xtensor(::xllm::KVCacheConfig::get_instance().enable_xtensor())
      .enable_sleep_mode(options_.enable_sleep_mode())
      .enable_linear_attention(enable_linear_attention)
      .enable_lighting_indexer(enable_lighting_indexer)
      .indexer_cache_enabled_layers(std::move(indexer_cache_enabled_layers))
      .enable_kv_cache_quant(enable_kv_cache_quant)
      .enable_indexer_cache_quant(enable_indexer_cache_quant)
      .tensor_allocator(std::move(tensor_allocator))
      .block_size(options_.block_size())
      .head_dim(args.head_dim())
      .index_head_dim(std::max(args.index_head_dim(), 1))
      .window_size(std::max(args.window_size(), 1))
      .compress_ratios(args.compress_ratios());
#if defined(USE_NPU)
  create_options.enable_kv_cache_huge_page_allocator(use_huge_page_allocator);
#endif

  // RL sleep mode: when enable_sleep_mode is set, allocate_kv_caches builds the
  // KV cache over a VMM-backed SleepableAllocator region (see kv_cache.cpp), so
  // sleep()/wake_up() can release / re-acquire it.
  allocate_kv_caches(kv_caches_, kv_cache_shape, create_options);
  init_hierarchy_kv_cache_transfer(kv_cache_shape, create_options);

#if defined(USE_CUDA) || defined(USE_DCU)
  refresh_cuda_block_copy_runtime_state();
#endif

  return true;
}

bool WorkerImpl::allocate_kv_cache(const KVCacheShape& kv_cache_shape) {
  if (!allocate_kv_cache_storage(kv_cache_shape)) {
    return false;
  }

  status_ = Status::READY;
  return true;
}

bool WorkerImpl::allocate_kv_cache_with_transfer(
    const KVCacheShape& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  // create a KVCache for each layer
  const ModelArgs& model_args = context_.get_model_args();
  const int64_t num_layers = model_args.n_layers();
  const bool enable_lighting_indexer = model_args.index_n_heads() > 0;
  kv_cache_transfer_ = KVCacheTransferFactory::create(
      ::xllm::DisaggPDConfig::get_instance().kv_cache_transfer_type(),
      options_.transfer_listen_port(),
      options_.instance_role(),
      device_,
      kv_cache_shape,
      dtype_,
      kv_caches_,
      num_layers,
      [this](const KVCacheShape& shape,
             bool use_huge_page_allocator,
             std::shared_ptr<KVCacheTensorAllocator> tensor_allocator) {
        return this->allocate_kv_cache_storage(
            shape, use_huge_page_allocator, std::move(tensor_allocator));
      },
      enable_lighting_indexer,
      model_args.model_type(),
      options_.model_id());

  status_ = Status::READY;
  return true;
}

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
bool WorkerImpl::allocate_kv_cache_with_transfer(
    std::shared_ptr<KVCacheTransfer> kv_cache_transfer,
    const KVCacheShape& kv_cache_shape) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  CHECK(kv_caches_.empty()) << "KV caches are already initialized.";

  kv_cache_transfer_ = kv_cache_transfer;

  std::shared_ptr<KVCacheTensorAllocator> tensor_allocator;
#if defined(USE_MLU)
  tensor_allocator = mlu_mooncake_tensor_allocator();
#endif
  if (!allocate_kv_cache_storage(kv_cache_shape,
                                 /*use_huge_page_allocator=*/true,
                                 std::move(tensor_allocator))) {
    return false;
  }

  if (is_spec_draft_) {
    kv_cache_transfer_->register_kv_cache_spec(
        kv_caches_, kv_cache_shape, dtype_);
  } else {
    kv_cache_transfer_->register_kv_cache(kv_caches_, kv_cache_shape, dtype_);
  }

  status_ = Status::READY;
  return true;
}
#endif

void WorkerImpl::get_cache_info(uint64_t& cluster_id,
                                std::string& addr,
                                uint16_t& port) {
  cluster_id = 0;
  addr.clear();
#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  if (kv_cache_transfer_) {
    kv_cache_transfer_->get_cache_info(cluster_id, addr);
  }
#endif
  port = options_.transfer_listen_port();
}

bool WorkerImpl::link_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports) {
  return worker_rendezvous_->link_cluster(cluster_ids, addrs, ports);
}

bool WorkerImpl::unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                                const std::vector<std::string>& addrs,
                                const std::vector<uint16_t>& ports) {
  return worker_rendezvous_->unlink_cluster(cluster_ids, addrs, ports);
}

bool WorkerImpl::link_p2p(const std::string& remote_addr) {
  return worker_rendezvous_->link_p2p(remote_addr);
}

bool WorkerImpl::unlink_p2p(const std::string& remote_addr) {
  return worker_rendezvous_->unlink_p2p(remote_addr);
}

std::tuple<int64_t, int64_t> WorkerImpl::estimate_kv_cache_capacity() {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  size_t torch_cache = 0;
  size_t torch_largest_block = 0;
  int32_t device_id = device_.index();
  Device::empty_cache(device_id);
#if defined(USE_NPU)
  // get torch's cache memory size
  c10_npu::NPUCachingAllocator::cacheInfo(
      device_id, &torch_cache, &torch_largest_block);
#endif
  const auto available_memory = device_.free_memory();
  const auto total_memory = device_.total_memory();
  DeviceMonitor::get_instance().set_total_memory(device_id, total_memory);
  DeviceMonitor::get_instance().set_weight_memory(
      device_id, total_memory - available_memory - torch_cache);
  return {available_memory + torch_cache, total_memory};
}

void WorkerImpl::process_group_test() {
  device_.set_device();

  // create random tensors
  const auto options = torch::dtype(torch::kHalf).device(device_);
  torch::Tensor tensor = torch::randn({10, 10}, options);
  // call allreduce
  parallel_state::reduce(tensor, parallel_args_.process_group_);
  // call allgather
  parallel_state::gather(tensor, parallel_args_.process_group_);
}

ForwardInput WorkerImpl::prepare_inputs(Batch& batch) {
  return model_executor_->prepare_inputs(batch);
}

bool WorkerImpl::can_prepare_npu_graph_decode_input(
    const ModelInputParams& input_params) const {
#if defined(USE_NPU)
  return !options_.enable_speculative_decode() &&
         ::xllm::ExecutionConfig::get_instance().enable_graph() &&
         ::xllm::ExecutionConfig::get_instance().enable_graph_double_buffer() &&
         enable_schedule_overlap() && options_.backend() == "llm" &&
         input_params.meta.batch_forward_type.has_decode();
#else
  (void)input_params;
  return false;
#endif
}

bool WorkerImpl::can_prepare_without_compute_stream_wait(
    const ModelInputParams& input_params) const {
#if defined(USE_NPU)
  (void)input_params;
  return !options_.enable_speculative_decode() &&
         ::xllm::ExecutionConfig::get_instance().enable_graph() &&
         ::xllm::ExecutionConfig::get_instance().enable_graph_double_buffer() &&
         options_.backend() == "llm";
#else
  (void)input_params;
  return false;
#endif
}

bool WorkerImpl::can_skip_npu_graph_decode_sync(
    const ModelInputParams& input_params) const {
#if defined(USE_NPU)
  return can_prepare_npu_graph_decode_input(input_params) &&
         input_params.meta.batch_forward_type.is_decode() &&
         options_.kv_cache_transfer_mode() != "PUSH" &&
         !options_.enable_speculative_decode();
#else
  (void)input_params;
  return false;
#endif
}

folly::SemiFuture<std::tuple<int64_t, int64_t>>
WorkerImpl::estimate_kv_cache_capacity_async() {
  folly::Promise<std::tuple<int64_t, int64_t>> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    const auto output = this->estimate_kv_cache_capacity();
    promise.setValue(output);
  });
  return future;
}

void WorkerImpl::update_last_step_output(
    const std::optional<ForwardOutput>& output) {
  if (output.value().sample_output.next_tokens.defined()) {
    last_step_output_ = std::move(output.value());
    last_step_output_valid_ = true;
  } else {
    if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
      last_step_output_ = std::move(output.value());
    }
    last_step_output_valid_ = false;
  }
}

ForwardInput WorkerImpl::update_input_by_last_step_output(
    ForwardInput& inputs) {
#if defined(USE_NPU)
  if (can_prepare_npu_graph_decode_input(inputs.input_params)) {
    xllm::kernel::npu::replace_token(
        inputs.token_ids,
        last_step_output_.sample_output.next_tokens,
        /*synchronize_stream=*/false);
    inputs.input_params.graph.input_tokens_override = inputs.token_ids;
    return inputs;
  }
  xllm::kernel::npu::replace_token(inputs.token_ids,
                                   last_step_output_.sample_output.next_tokens,
                                   /*synchronize_stream=*/true);
#else
  auto& flatten_tokens = inputs.token_ids;
  auto neg_mask = (flatten_tokens < 0);
  auto clamped_neg_indices = torch::clamp(-flatten_tokens, 0);
#if defined(USE_MUSA)
  auto cpu = clamped_neg_indices.cpu() - 1;
  auto replacement =
      last_step_output_.sample_output.next_tokens.index({cpu.musa()});
#else
  auto replacement = last_step_output_.sample_output.next_tokens.index(
      {clamped_neg_indices - 1});
#endif
  inputs.token_ids = torch::where(neg_mask, replacement, flatten_tokens);
#endif
  return inputs;
}

std::optional<ForwardOutput> WorkerImpl::step_for_schedule_overlap(
    const ForwardInput& input) {
  // No linear-state restore here on purpose. LLMWorkerImpl overrides this to
  // copy checkpoints on compute_stream_; speculative/MTP workers keep this base
  // version but run every forward through an inner LLMWorkerImpl built with
  // schedule-overlap off, so the checkpoint copy fires on that inner worker's
  // non-overlap prepare_work_before_execute_on_stream path. The outer
  // speculative worker owns no kv_caches_, so restoring here would be a no-op.
  return step(input);
}

ForwardInput WorkerImpl::update_input_by_last_step_output_for_schedule_overlap(
    ForwardInput& input) {
  return update_input_by_last_step_output(input);
}

#if defined(USE_NPU)
torch::Tensor WorkerImpl::recompute_new_cache_slots(const ForwardInput& input) {
  auto old_cache_slots = input.input_params.attention.device.new_cache_slots;
  int64_t numel = old_cache_slots.numel();
  // The logical block stride that BlockManager hands out is
  // `block_size * kv_split_size_effective` (see llm_engine init). When KV is
  // not split (kv_split_size == 1) the stride collapses back to block_size.
  const int32_t kv_split_size = parallel_args_.kv_split_size_effective();
  const int32_t block_size_total = options_.block_size() * kv_split_size;
  // KV-shard ownership predicate: block whose sub-index inside the logical
  // block matches this rank's KV-split rank (degenerates to "this rank only"
  // when kv_split_size == 1, since sub_block_idx is always 0 there).
  const int32_t owner_kv_split_rank = parallel_args_.kv_split_rank();

  torch::Tensor indices = torch::arange(numel, torch::kCPU);
  torch::Tensor block_offset = indices % block_size_total;
  torch::Tensor sub_block_idx =
      torch::floor_divide(block_offset, options_.block_size());
  torch::Tensor mask = (sub_block_idx == owner_kv_split_rank);
  torch::Tensor valid_indices = torch::nonzero(mask).squeeze();

  torch::Tensor new_cache_slots = torch::full_like(old_cache_slots, -1);
  if (valid_indices.numel() > 0) {
    const torch::Device slots_device = old_cache_slots.device();
    torch::Tensor valid_indices_on_device =
        valid_indices.to(slots_device, /*non_blocking=*/false);
    torch::Tensor old_slotid =
        old_cache_slots.index_select(0, valid_indices_on_device)
            .to(torch::kInt);
    torch::Tensor block_id = torch::floor_divide(old_slotid, block_size_total);
    torch::Tensor block_offset_mod = old_slotid % options_.block_size();
    torch::Tensor new_slotid =
        block_id * options_.block_size() + block_offset_mod;
    new_cache_slots.index_put_({valid_indices_on_device},
                               new_slotid.to(new_cache_slots.scalar_type()));
  }
  return new_cache_slots;
}

torch::Tensor WorkerImpl::compute_in_prefix_slots(const ForwardInput& input) {
  // Derive prefix block count from `kv_cache_tokens_nums` (already-cached
  // tokens at the start of this forward), which covers prefix-cache hits and
  // chunked prefill progression.
  torch::Tensor block_tables = input.input_params.attention.device.block_tables;
  torch::Tensor kv_cache_tokens_nums =
      input.input_params.attention.device.kv_cache_tokens_nums;
  if (block_tables.defined() && !block_tables.device().is_cpu()) {
    block_tables = block_tables.to(torch::kCPU);
  }
  if (kv_cache_tokens_nums.defined() &&
      !kv_cache_tokens_nums.device().is_cpu()) {
    kv_cache_tokens_nums = kv_cache_tokens_nums.to(torch::kCPU);
  }
  const int32_t block_size = options_.block_size();
  // Stride here is the KV-split width (how many ranks the KV is sharded
  // across), NOT cp_size. When kv_split_size == 1 each rank holds the full
  // prefix and we emit ALL prefix blocks; when kv_split_size == cp_size this
  // reduces to the legacy round-robin behavior byte-for-byte.
  const int32_t kv_split_size =
      std::max(1, parallel_args_.kv_split_size_effective());
  const int32_t kv_split_rank = parallel_args_.kv_split_rank();

  CHECK(block_tables.defined() && block_tables.dim() == 2)
      << "block_tables must be a 2D tensor in compute_in_prefix_slots.";
  CHECK_EQ(block_tables.size(0), kv_cache_tokens_nums.numel())
      << "block_tables rows (" << block_tables.size(0)
      << ") must match kv_cache_tokens_nums numel ("
      << kv_cache_tokens_nums.numel() << ").";
  CHECK_GT(block_size, 0);

  const int64_t num_sequences = block_tables.size(0);
  std::vector<int32_t> in_prefix_slots_vec;
  // HCCL AllGather and downstream ATB Gather op cannot accept a [0]-shaped
  // tensor; emit at least one padding slot when the batch contributes none.
  if (num_sequences == 0) {
    in_prefix_slots_vec.push_back(0);
    return torch::tensor(in_prefix_slots_vec, torch::kInt);
  }

  auto block_tables_acc = block_tables.accessor<int32_t, 2>();
  auto kv_cache_tokens_acc = kv_cache_tokens_nums.accessor<int32_t, 1>();

  // Per-rank prefix count = total_prefix_tokens / kv_split_size. This matches
  // the legacy behavior byte-for-byte when kv_split_size == cp_size (rank
  // emits the first `prefix_blocks` entries of block_tables; the ATB reshape
  // assumes a [kv_split_size, local_len, ...] layout for the subsequent
  // AllGather). When kv_split_size == 1 every rank emits the FULL prefix
  // (prefix_blocks == kv_cache_tokens / block_size) and the ATB layer skips
  // the prefix AllGather entirely (see S6).
  // (void)kv_split_rank to silence -Wunused-variable on the kv_split_size==1
  // path; the rank is implicit there.
  (void)kv_split_rank;
  in_prefix_slots_vec.reserve(num_sequences * block_size);
  for (int64_t i = 0; i < num_sequences; ++i) {
    const int32_t prefix_tokens = kv_cache_tokens_acc[i] / kv_split_size;
    const int32_t prefix_blocks = prefix_tokens / block_size;
    if (prefix_blocks <= 0) {
      in_prefix_slots_vec.push_back(0);
      continue;
    }
    for (int32_t j = 0; j < prefix_blocks; j++) {
      const int32_t physical_block = block_tables_acc[i][j];
      const int32_t base_slot = physical_block * block_size;
      for (int32_t k = 0; k < block_size; ++k) {
        in_prefix_slots_vec.push_back(base_slot + k);
      }
    }
  }
  return torch::tensor(in_prefix_slots_vec, torch::kInt);
}
#endif

void WorkerImpl::prepare_work_before_execute(const ForwardInput& input,
                                             ForwardInput& processed_input) {
  prepare_work_before_execute_on_stream(
      input, processed_input, *prepare_stream_);
}

void WorkerImpl::prepare_work_before_execute_on_stream(
    const ForwardInput& input,
    ForwardInput& processed_input,
    Stream& prepare_stream) {
#if defined(USE_NPU)
  // Without device_capture_lock, ACL graph capture will be interrupted by the
  // synchronization H2D of data update streams asynchronously scheduled by
  // other threads, even if the capture and synchronization streams are not
  // the same, and even if capture_mode is set to
  // ACL_MODEL_RI_CAPTURE_MODE_THREAD_LOCAL.
  // The possible reason is that ACL graph capture may use additional
  // auxiliary streams, and these auxiliary streams might be the same as the
  // asynchronously scheduled data update streams.

  std::optional<std::unique_lock<std::mutex>> lock_guard;
  if (::xllm::ExecutionConfig::get_instance().enable_graph()) {
    auto& capture_lock =
        ::xllm::npu::DeviceCaptureLock::get_instance().get_lock(
            device_.index());
    lock_guard.emplace(capture_lock);
  }
#endif
  c10::StreamGuard stream_guard = prepare_stream.set_stream_guard();
  if (enable_schedule_overlap() &&
      !can_prepare_without_compute_stream_wait(input.input_params) &&
      compute_stream_) {
    // MTP updates reuse shared prepare/compute streams and need this ordering;
    // only graph double-buffer decode can prepare the next slot independently.
    prepare_stream.wait_stream(*compute_stream_);
  }
  CHECK(prepare_stream.wait_event(input.metadata_ready_event))
      << "failed to wait input metadata ready event on worker prepare stream";

  // CP partition is now done worker-side (formerly engine-side in
  // LLMEngine::step). torch::Tensor fields are handles, so assigning new
  // tensors to the CP working copy does not mutate `input`.
  // IMPORTANT: every downstream NPU-side prepare call
  // (prepare_cp_prefill_inputs, CpEpPadding, recompute_new_cache_slots,
  // compute_in_prefix_slots) reads from the per-rank slice and therefore MUST
  // consume the CP working copy, not the pre-partition `input.*`. The
  // `!input.cp_partitioned` guard is critical for nested step_async paths (MTP
  // target/draft sub-workers) that re-enter prepare_work_before_execute on
  // already-partitioned device tensors; see ForwardInput::cp_partitioned.
  // Prefill-side CP (partition + ATB cp tensors) applies to PREFILL,
  // CHUNKED_PREFILL, and MIXED. `no_decode()` wrongly excludes MIXED.
  const bool needs_cp_prefill_side =
      parallel_args_.cp_size() > 1 && !Platform::uses_model_cp_partition() &&
      !input.input_params.meta.batch_forward_type.is_decode();
  const bool needs_cp_partition =
      needs_cp_prefill_side && !input.cp_partitioned;
  std::optional<ForwardInput> cp_input;
  if (needs_cp_prefill_side) {
    cp_input.emplace(input);
  }
#if defined(USE_NPU)
  if (needs_cp_prefill_side) {
    ForwardInput& cp_working = *cp_input;
    // RPC packed_input only carries input_host_buffer until unpack; partition
    // and prepare_cp_prefill_inputs need materialized token_ids / seq_lens.
    if (cp_working.input_host_buffer_has_layout &&
        (!cp_working.token_ids.defined() ||
         cp_working.token_ids.numel() == 0)) {
      ForwardInput unpacked;
      if (detail::unpack_from_input_host_buffer(
              cp_working, torch::Device(torch::kCPU), unpacked)) {
        unpacked.cp_partitioned = cp_working.cp_partitioned;
        cp_working = std::move(unpacked);
        cp_working.input_host_buffer_has_layout = false;
      } else {
        LOG(ERROR) << "[CP_PREP] unpack_from_input_host_buffer failed before "
                      "cp_partition (cp_rank="
                   << parallel_args_.cp_rank() << ")";
      }
    }
  }
#endif
  if (needs_cp_partition) {
    ForwardInput& cp_working = *cp_input;
    const int64_t tokens_before =
        cp_working.token_ids.defined() ? cp_working.token_ids.numel() : 0;
    cp::cp_partition_inplace(
        cp_working, parallel_args_.cp_rank(), parallel_args_.cp_size());
    const int64_t tokens_after =
        cp_working.token_ids.defined() ? cp_working.token_ids.numel() : 0;
    // Mark partitioned only when slice materialized (packed RPC used to skip
    // partition on empty token_ids yet still set this flag).
    if (tokens_after > 0) {
      cp_working.cp_partitioned = true;
    } else {
      LOG(ERROR) << "[CP_PREP] cp_partition_inplace produced no tokens "
                    "(before="
                 << tokens_before << " cp_rank=" << parallel_args_.cp_rank()
                 << " host_buffer_has_layout="
                 << cp_working.input_host_buffer_has_layout << ")";
    }
  }
  const ForwardInput& prep_for_device =
      needs_cp_prefill_side ? *cp_input : input;

#if defined(USE_NPU)
  // recompute_new_cache_slots / compute_in_prefix_slots are CP prefill-side
  // prepares that must run EXACTLY ONCE, on the first (outer) pass that owns
  // the partition. They both remap slots from the BlockManager logical space
  // (stride block_size * kv_split_size) into this rank's local physical space,
  // an operation that is NOT idempotent for kv_split_size > 1. Nested MTP
  // target/draft sub-workers re-enter prepare_work_before_execute on the
  // already-partitioned input (cp_partitioned == true) whose new_cache_slots
  // were already remapped by the outer pass; recomputing again would double
  // remap and corrupt the KV slots. Gate on !input.cp_partitioned just like
  // prepare_cp_prefill_inputs below.
  const bool needs_kv_split_prep = needs_cp_prefill_side &&
                                   !input.cp_partitioned &&
                                   util::enable_kvcache_split();
  const bool have_prefix_slots =
      needs_cp_prefill_side && !input.cp_partitioned &&
      (::xllm::KVCacheConfig::get_instance().enable_prefix_cache() ||
       ::xllm::SchedulerConfig::get_instance().enable_chunked_prefill());
#endif

  auto prepare_device_on_stream = [&]() {
    processed_input = prep_for_device.to(device_, dtype_);
    ensure_forward_input_device_tensors(processed_input, device_);

#if defined(USE_NPU)
    CpPrefillInputs tmp_cp_inputs;
    if (needs_cp_prefill_side && !input.cp_partitioned) {
      const ForwardInput& cp_working = *cp_input;
      tmp_cp_inputs = prepare_cp_prefill_inputs(
          parallel_args_.cp_size(),
          cp_working.host_token_ids(),
          cp_working.host_positions(),
          cp_working.input_params.attention.device.q_seq_lens,
          have_prefix_slots,
          cp_working.input_params.attention.host.kv_cache_tokens_nums,
          options_.block_size(),
          parallel_args_.kv_split_size_effective());
      processed_input.input_params.parallel.cp_prefill_inputs =
          tmp_cp_inputs.to(device_);
      CpEpPadding cp_ep_padding(cp_working.host_token_ids(),
                                context_.get_model_args().num_experts_per_tok(),
                                context_.get_parallel_args().mapping_data(),
                                /*device=*/device_,
                                dtype_,
                                /*is_prefill=*/needs_cp_prefill_side);
      processed_input.input_params.parallel.cp_ep_padding_data =
          cp_ep_padding.build();
    }

    if (needs_kv_split_prep) {
      torch::Tensor new_cache_slots = recompute_new_cache_slots(*cp_input);
      processed_input.input_params.attention.device.new_cache_slots =
          new_cache_slots.to(device_);
    }
    if (have_prefix_slots) {
      torch::Tensor in_prefix_slots = compute_in_prefix_slots(*cp_input);
      processed_input.input_params.attention.device.in_prefix_slots =
          in_prefix_slots.to(device_);
    }
#endif

    auto& input_params = processed_input.input_params;

    bool empty_shard = input_params.meta.num_sequences == 0 &&
                       (!processed_input.token_ids.defined() ||
                        processed_input.token_ids.numel() == 0);
    const bool need_fake_input_for_empty_shard =
        empty_shard && !input_params.meta.batch_forward_type.is_empty() &&
        ((context_.get_parallel_args().cp_size() > 1 &&
          !Platform::uses_model_cp_partition()) ||
         (context_.get_parallel_args().dp_size() > 1 ||
          context_.get_parallel_args().ep_size() > 1 ||
          !context_.get_parallel_args().mapping_data().empty()));
    if (need_fake_input_for_empty_shard) {
      auto token_options = processed_input.token_ids.defined()
                               ? processed_input.token_ids.options()
                               : torch::TensorOptions().dtype(torch::kInt32);
      auto position_options = processed_input.positions.defined()
                                  ? processed_input.positions.options()
                                  : torch::TensorOptions().dtype(torch::kInt32);
      processed_input.token_ids =
          torch::ones({1}, token_options.device(device_));
      processed_input.positions =
          torch::zeros({1}, position_options.device(device_));
      empty_shard = false;
    }
    if (empty_shard) {
      return;
    }

    apply_kv_block_swaps(input_params);

#if defined(USE_NPU)
    if (context_.get_model_args().enable_mla() &&
        input_params.meta.batch_forward_type.is_chunked_prefill()) {
      prepare_mla_prefixcache_inputs(input_params);
    }

    if (!context_.get_parallel_args().mapping_data().empty() &&
        !(context_.get_parallel_args().cp_size() > 1) &&
        (context_.get_parallel_args().dp_size() > 1 ||
         context_.get_parallel_args().ep_size() > 1)) {
      torch::Tensor token_size_per_dp_group = torch::tensor(
          processed_input.input_params.parallel.dp_global_token_nums,
          torch::TensorOptions()
              .device(torch::kCPU)
              .dtype(torch::kInt32)
              .pinned_memory(true));
      const bool is_prefill =
          processed_input.input_params.meta.batch_forward_type.no_decode();
      DpEpPadding dp_ep_padding(token_size_per_dp_group,
                                context_.get_model_args().num_experts_per_tok(),
                                context_.get_parallel_args().mapping_data(),
                                device_,
                                dtype_,
                                is_prefill);
      processed_input.input_params.parallel.dp_ep_padding_data =
          dp_ep_padding.build();
      if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
        processed_input.input_params.expert.expert_load_data =
            expert_load_data_;
      }
    }

    if (has_linear_attention_layers(context_.get_model_args())) {
      prepare_input_params_for_linear_attention(processed_input.input_params);
      // Under schedule_overlap chunked prefill the previous chunk's forward
      // runs on compute_stream_ from a worker thread that may not have
      // enqueued its kernels yet when this prepare runs on the main thread.
      // Defer the slot-restore copy to step_for_schedule_overlap (worker
      // thread, on compute_stream_) so stream ordering between chunk N-1
      // writes and chunk N restore is automatic.
      if (!enable_schedule_overlap()) {
        restore_linear_state_slots(
            kv_caches_,
            processed_input.input_params.linear_state_cache_ops,
            processed_input.input_params.parallel.has_initial_state);
      }
    }

    if (can_prepare_npu_graph_decode_input(input_params)) {
      model_executor_->prepare_graph_input(processed_input.token_ids,
                                           processed_input.positions,
                                           kv_caches_,
                                           processed_input.input_params);
    }

#endif
  };

  prepare_device_on_stream();

  StreamEventPtr event = prepare_stream.record_event();
  if (event == nullptr) {
    prepare_stream.synchronize();
  }
  processed_input.metadata_ready_event = event;
}

void WorkerImpl::apply_kv_block_swaps(const ModelInputParams& input_params) {
#if defined(USE_CUDA) || defined(USE_DCU)
  if (::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel() &&
      can_use_cuda_block_copy_kernel(input_params)) {
    execute_cuda_block_copy_kernel(input_params);
    return;
  }
#endif

#if defined(USE_NPU)
  if (input_params.block_copy.swap_blocks.size() == 0 ||
      ::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel()) {
    return;
  }
#elif defined(USE_CUDA) || defined(USE_DCU) || defined(USE_MLU)
  // MLU has no fused block-copy kernel (enable_block_copy_kernel defaults to
  // false), so it always falls through to the torch swap path below. Without
  // this, beam-search copy-on-write blocks are allocated but never populated
  // with the source KV, leaving each diverging beam reading stale pool memory.
  if (input_params.block_copy.swap_blocks.size() == 0) {
    return;
  }
#else
  return;
#endif

#if defined(USE_NPU) || defined(USE_CUDA) || defined(USE_DCU) || \
    defined(USE_MLU)
  std::vector<int64_t> src_indices, dst_indices;
  src_indices.reserve(input_params.block_copy.swap_blocks.size());
  dst_indices.reserve(input_params.block_copy.swap_blocks.size());

  for (const auto& block : input_params.block_copy.swap_blocks) {
    src_indices.push_back(block.src_block_id);
    dst_indices.push_back(block.dst_block_id);
  }

  auto src_tensor =
      torch::tensor(src_indices, torch::dtype(torch::kLong).device(device_));
  auto dst_tensor =
      torch::tensor(dst_indices, torch::dtype(torch::kLong).device(device_));
  for (size_t layer_id = 0; layer_id < kv_caches_.size(); ++layer_id) {
    kv_caches_[layer_id].swap_blocks(src_tensor, dst_tensor);
  }
#endif
}

#if defined(USE_CUDA) || defined(USE_DCU)
void WorkerImpl::refresh_cuda_block_copy_runtime_state() {
  cuda_block_copy_runtime_state_ = {};
  if (!::xllm::BeamSearchConfig::get_instance().enable_block_copy_kernel() ||
      kv_caches_.empty()) {
    return;
  }

  const auto& first_kv_cache = kv_caches_.front();
  auto key_cache = first_kv_cache.get_k_cache();
  auto value_cache = first_kv_cache.get_v_cache();
  if (!key_cache.defined() || !value_cache.defined() || !key_cache.is_cuda() ||
      !value_cache.is_cuda()) {
    return;
  }

  CHECK(key_cache.is_contiguous())
      << "CUDA block copy kernel expects contiguous key cache";
  CHECK(value_cache.is_contiguous())
      << "CUDA block copy kernel expects contiguous value cache";
  CHECK_GT(key_cache.size(0), 0);

  const auto cache_dtype = key_cache.scalar_type();
  std::vector<int64_t> key_cache_ptrs;
  std::vector<int64_t> value_cache_ptrs;
  key_cache_ptrs.reserve(kv_caches_.size());
  value_cache_ptrs.reserve(kv_caches_.size());
  for (const auto& kv_cache : kv_caches_) {
    auto layer_k_cache = kv_cache.get_k_cache();
    auto layer_v_cache = kv_cache.get_v_cache();
    CHECK(layer_k_cache.defined() && layer_v_cache.defined());
    CHECK(layer_k_cache.is_cuda() && layer_v_cache.is_cuda());
    CHECK(layer_k_cache.is_contiguous());
    CHECK(layer_v_cache.is_contiguous());
    CHECK(layer_k_cache.scalar_type() == cache_dtype);
    CHECK(layer_v_cache.scalar_type() == cache_dtype);
    CHECK(layer_k_cache.sizes() == key_cache.sizes());
    CHECK(layer_v_cache.sizes() == value_cache.sizes());
    key_cache_ptrs.push_back(
        reinterpret_cast<int64_t>(layer_k_cache.data_ptr()));
    value_cache_ptrs.push_back(
        reinterpret_cast<int64_t>(layer_v_cache.data_ptr()));
  }

  auto ptr_options =
      torch::TensorOptions().device(device_).dtype(torch::kInt64);
  cuda_block_copy_runtime_state_.k_cache_ptrs_device =
      torch::tensor(key_cache_ptrs, ptr_options);
  cuda_block_copy_runtime_state_.v_cache_ptrs_device =
      torch::tensor(value_cache_ptrs, ptr_options);
  cuda_block_copy_runtime_state_.num_layers = kv_caches_.size();
  cuda_block_copy_runtime_state_.numel_per_block = key_cache[0].numel();
}

bool WorkerImpl::can_use_cuda_block_copy_kernel(
    const ModelInputParams& input_params) const {
  return cuda_block_copy_runtime_state_.valid() &&
         input_params.block_copy.src_block_indices.defined() &&
         input_params.block_copy.dst_block_indices.defined() &&
         input_params.block_copy.cum_sum.defined() &&
         input_params.block_copy.src_block_indices.numel() > 0 &&
         input_params.block_copy.dst_block_indices.numel() > 0 &&
         input_params.block_copy.cum_sum.numel() > 0;
}

void WorkerImpl::execute_cuda_block_copy_kernel(
    const ModelInputParams& input_params) {
  CHECK(!kv_caches_.empty());
  xllm::kernel::cuda::block_copy(
      cuda_block_copy_runtime_state_.k_cache_ptrs_device,
      cuda_block_copy_runtime_state_.v_cache_ptrs_device,
      input_params.block_copy.src_block_indices,
      input_params.block_copy.dst_block_indices,
      input_params.block_copy.cum_sum,
      cuda_block_copy_runtime_state_.numel_per_block,
      kv_caches_.front().get_k_cache().scalar_type());
}
#endif

folly::SemiFuture<std::optional<ForwardOutput>> WorkerImpl::step_async(
    const ForwardInput& input) {
  folly::Promise<std::optional<ForwardOutput>> promise;
  auto future = promise.getSemiFuture();
  // Prepare runs on the worker's own threadpool (the same thread that
  // executes step). Doing prepare on the calling thread instead serialises
  // every worker's input H2D copy through that caller, and in single-node
  // single-process mode each prepare can issue blocking device driver calls
  // through the caching host allocator. While one worker's step is busy-waiting
  // on a collective for its partner, those driver calls share the driver's
  // internal locks with the running collective and the caller never returns to
  // schedule the second worker's step — both devices deadlock. Pinning prepare
  // to the worker's device thread keeps all device work on a single thread.
  threadpool_.schedule([this, input, promise = std::move(promise)]() mutable {
    ForwardInput input_on_device;
    prepare_work_before_execute(input, input_on_device);

    if (hierarchy_kv_cache_transfer_ != nullptr) {
      hierarchy_kv_cache_transfer_->set_layer_synchronizer(
          input_on_device.input_params);
    }

    // run the model on the given input in working thread
    if (!enable_schedule_overlap()) {
      const auto output = this->step(input_on_device);
      promise.setValue(output);
    } else {
      if (last_step_output_valid_ && input_on_device.token_ids.numel() > 0 &&
          input_on_device.input_params.meta.batch_forward_type.has_decode()) {
        // replace step i model input with true output of step i-1
        input_on_device = update_input_by_last_step_output_for_schedule_overlap(
            input_on_device);
      }

      const auto output = this->step_for_schedule_overlap(input_on_device);
      if (output.has_value()) {
        if (is_driver() || ::xllm::EPLBConfig::get_instance().enable_eplb()) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          update_last_step_output(output);
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          update_last_step_output(output);
        }
      } else {
        if (is_driver() || ::xllm::EPLBConfig::get_instance().enable_eplb()) {
          std::unique_lock<std::mutex> lock(mtx_);
          cv_.wait(lock, [this] { return !is_recorded_; });
          last_step_output_valid_ = false;
          is_recorded_ = true;
          cv_.notify_one();
        } else {
          last_step_output_valid_ = false;
        }
      }
      promise.setValue(output);
    }
  });
  return future;
}

ForwardOutput WorkerImpl::get_last_step_result() {
  ForwardOutput output;
  std::unique_lock<std::mutex> lock(mtx_);
  cv_.wait(lock, [this] { return is_recorded_; });
  if (last_step_output_valid_ ||
      ::xllm::EPLBConfig::get_instance().enable_eplb()) {
    output = last_step_output_;
  }
  is_recorded_ = false;
  cv_.notify_one();
  return output;
}

folly::SemiFuture<folly::Unit> WorkerImpl::process_group_test_async() {
  folly::Promise<folly::Unit> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this, promise = std::move(promise)]() mutable {
    this->process_group_test();
    promise.setValue();
  });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::init_model_async(
    const std::string& model_weights_path,
    int32_t random_seed,
    MasterStatus master_status) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([this,
                        model_weights_path,
                        random_seed,
                        master_status,
                        promise = std::move(promise)]() mutable {
    auto status =
        this->init_model(model_weights_path, random_seed, master_status);
    promise.setValue(status);
  });

  return future;
}

bool WorkerImpl::rl_sleep_mode() const {
  return options_.enable_sleep_mode() &&
         !::xllm::KVCacheConfig::get_instance().enable_xtensor();
}

void WorkerImpl::setup_rl_sleep_weights() {
  // Route each decoder layer's contiguous weight buffer (manual loader) into a
  // VMM-backed SleepableAllocator region so rl_sleep() can release the weight
  // HBM. Must run before the layers are constructed (loader mode is chosen in
  // the layer ctor) and before weights are loaded.
  if (!rl_sleep_mode()) {
    return;
  }
  ::xllm::LoadConfig::get_instance().enable_manual_loader(/*enable=*/true);
  SleepableAllocator::get_instance().set_weights_enabled(/*enabled=*/true);
}

bool WorkerImpl::rl_sleep() {
  // Deep sleep only: release physical HBM (weights + KV) via the VMM-backed
  // SleepableAllocator. Contents are discarded; weights are re-loaded and KV is
  // re-prefilled after wakeup.
  SleepableAllocator::get_instance().sleep();
  // Also release the default caching allocator's reserved (but free) blocks,
  // e.g. memory left cached from profiling/warmup activation tensors, so the
  // sleep actually returns the maximum amount of HBM to the driver.
  Device::empty_cache(device_.index());
  return true;
}

bool WorkerImpl::rl_wakeup() {
  // Re-acquire physical HBM via the SleepableAllocator. v1 does a full wake
  // (weights + kv_cache); tag-based partial wake is a follow-up.
  SleepableAllocator::get_instance().wake_up(/*tags=*/{});
  return true;
}

bool WorkerImpl::xtensor_sleep(MasterStatus master_status) {
  // The memory for kvcache and model weights from hbm is released by xtensor;
  if (master_status == MasterStatus::LIGHT_SLEEP) {
    // only load model weights to host memory.
    auto model_loader = ModelLoader::create(model_weights_path_);
    model_->lazy_load_model(std::move(model_loader));
  } else if (master_status == MasterStatus::DEEP_SLEEP) {
    // only release model weights from host memory.
    model_->free_model_weights();
  }
  return true;
}

bool WorkerImpl::sleep(MasterStatus master_status) {
  if (rl_sleep_mode()) {
    return rl_sleep();
  }
  return xtensor_sleep(master_status);
}

bool WorkerImpl::update_weights(const std::string& weights_path) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  const std::string& path =
      weights_path.empty() ? model_weights_path_ : weights_path;
  LOG(INFO) << "Updating weights in place from: " << path;

  auto model_loader = ModelLoader::create(path);

  // Limit ATen threads during weight load (same as initial load) to avoid
  // oversubscription across tensor-parallel workers.
  auto scoped_load_threads =
      std::make_unique<ScopedAtenLoadThreads>(/*target_threads=*/1);

  this->load_model(std::move(model_loader));
  return true;
}

bool WorkerImpl::start_profile() {
#if defined(USE_CUDA)
  const auto& cfg = ProfileConfig::get_instance();
  if (cfg.profile_backend() == "cuda") {
    // Capture-range only; requires the server to run under nsys.
    return CudaProfiler::get_instance().start();
  }
  // Default "torch" backend records in-process via Kineto. CPU-op capture uses
  // thread-local callbacks, so enable it on the compute thread that runs the
  // forward pass rather than on the RPC handler thread.
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule([promise = std::move(promise)]() mutable {
    promise.setValue(TorchProfiler::get_instance().start());
  });
  return std::move(future).get();
#else
  LOG(ERROR) << "Online timeline profiling is only supported on CUDA.";
  return false;
#endif
}

bool WorkerImpl::stop_profile() {
#if defined(USE_CUDA)
  const auto& cfg = ProfileConfig::get_instance();
  if (cfg.profile_backend() == "cuda") {
    return CudaProfiler::get_instance().stop();
  }
  const std::string profile_dir = cfg.profile_dir();
  const int32_t rank = parallel_args_.rank();
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [profile_dir, rank, promise = std::move(promise)]() mutable {
        promise.setValue(TorchProfiler::get_instance().stop(profile_dir, rank));
      });
  return std::move(future).get();
#else
  LOG(ERROR) << "Online timeline profiling is only supported on CUDA.";
  return false;
#endif
}

bool WorkerImpl::wakeup(const WakeupOptions& options) {
  if (rl_sleep_mode()) {
    return rl_wakeup();
  }

  if (!options.remote_addrs.empty()) {
#if defined(USE_NPU)
    return wakeup_from_remote_weights(options);
#endif
    LOG(ERROR) << "Remote weight wakeup only supports npu device.";
    return false;
  }

  return wakeup_local(options);
}

bool WorkerImpl::wakeup_local(const WakeupOptions& options) {
  if (options.master_status == MasterStatus::LIGHT_SLEEP) {
#if defined(USE_NPU)
    if (::xllm::LoadConfig::get_instance().enable_rolling_load() &&
        !is_spec_draft_) {
      // Reuse rolling runtime state and refresh rolling initialization on
      // wakeup without re-reading checkpoint in LIGHT_SLEEP.
      if (!init_rolling_runtime_state()) {
        LOG(ERROR) << "Failed to initialize rolling runtime state on wakeup";
        return false;
      }
    } else {
      model_->reload_model_weights();
    }
#else
    model_->reload_model_weights();
#endif
  } else if (options.master_status == MasterStatus::DEEP_SLEEP) {
    auto model_loader = ModelLoader::create(model_weights_path_);
    this->load_model(std::move(model_loader));
  }
  return true;
}

#if defined(USE_NPU)
bool WorkerImpl::wakeup_from_remote_weights(const WakeupOptions& options) {
  // Prefer segment-based transfer if available, fallback to legacy offsets.
  if (::xllm::LoadConfig::get_instance().enable_rolling_load()) {
    LOG(ERROR) << "Remote weight wakeup does not support "
                  "::xllm::LoadConfig::get_instance().enable_rolling_load()";
    return false;
  }

  bool use_segments = !options.src_weight_segments.empty();
  if (use_segments) {
    if (options.src_weight_segments.size() != options.remote_addrs.size()) {
      LOG(ERROR) << "remote_addrs and src_weight_segments size mismatch: "
                 << options.remote_addrs.size() << " vs "
                 << options.src_weight_segments.size();
      return false;
    }
  } else {
    // Legacy single-offset mode (backward compatibility).
    if (options.src_weight_segments.empty() &&
        options.remote_addrs.size() > 0) {
      LOG(ERROR) << "No weight segments provided for remote wakeup";
      return false;
    }
  }

  auto& allocator = XTensorAllocator::get_instance();
  auto* tensors = allocator.get_model_tensors(options_.model_id());
  if (!tensors || tensors->weight_base_ptr == nullptr ||
      tensors->weight_num_pages == 0) {
    LOG(ERROR) << "Weight region not initialized for model "
               << options_.model_id();
    return false;
  }

  auto& global_xtensor = GlobalXTensor::get_instance();
  if (!global_xtensor.is_initialized()) {
    LOG(ERROR) << "GlobalXTensor not initialized";
    return false;
  }
  if (!weight_transfer_) {
    LOG(ERROR) << "MooncakeWeightTransfer not initialized";
    return false;
  }

  // Destination is always contiguous (local allocation).
  uint64_t dst_base_offset =
      reinterpret_cast<uintptr_t>(tensors->weight_base_ptr) -
      reinterpret_cast<uintptr_t>(global_xtensor.base_vaddr());
  for (size_t i = 0; i < options.remote_addrs.size(); ++i) {
    const auto& segments = options.src_weight_segments[i];
    uint64_t dst_offset = dst_base_offset;
    // Pull each segment from source, writing sequentially to destination.
    for (const auto& seg : segments) {
      if (!weight_transfer_->pull_weights(
              options.remote_addrs[i], seg.offset, dst_offset, seg.size)) {
        LOG(ERROR) << "Failed to pull remote weight segment from "
                   << options.remote_addrs[i] << ", src_offset=" << seg.offset
                   << ", size=" << seg.size;
        return false;
      }
      dst_offset += seg.size;
    }
  }

  model_->reload_model_weights_from_device();
  return true;
}
#endif

// initialize model, cache manager. async call
bool WorkerImpl::init_model(const std::string& model_weights_path,
                            int32_t random_seed,
                            MasterStatus master_status) {
  // set same random seed for all worker
  ::xllm::ExecutionConfig::get_instance().random_seed(random_seed);
  device_.set_seed(random_seed);

  auto model_loader = ModelLoader::create(model_weights_path);
  model_weights_path_ = std::move(model_weights_path);

  auto args = model_loader->model_args();
  auto quant_args = model_loader->quant_args();
  const bool embedding_mode = options_.task_type() == "embed";
  args.embedding_mode(embedding_mode);
  torch::ScalarType dtype = util::parse_dtype(args.dtype(), device_);

  // Draft engine is fed token ids and detokenized by the target, so it loads
  // no tokenizer of its own (not universal: Eagle3 keeps its own draft vocab;
  // see speculative_engine.cpp).
  if (!options_.is_draft_engine()) {
    std::unique_ptr<Tokenizer> tokenizer = model_loader->tokenizer();
    CHECK(tokenizer != nullptr);
    const int64_t tokenizer_vocab_size = tokenizer->vocab_size();
    int64_t model_vocab_size = args.vocab_size();
    // use tokenizer vocab size if model vocab size is not set
    if (model_vocab_size <= 0) {
      LOG(WARNING)
          << "Model vocab size is not set, using tokenizer vocab size: "
          << tokenizer_vocab_size;
      args.vocab_size(tokenizer_vocab_size);
    } else if (tokenizer_vocab_size > model_vocab_size) {
      LOG(WARNING) << "Unsafe vocab mismatch: tokenizer: "
                   << tokenizer_vocab_size << ", model: " << model_vocab_size;
    }
  }

#if defined(USE_NPU)
  if (options_.enable_speculative_decode() &&
      util::is_deepseek_v4_model_type(args.model_type()) &&
      (options_.speculative_algorithm() == "MTP" ||
       options_.speculative_algorithm() == "mtp")) {
    args.num_speculative_tokens(options_.num_speculative_tokens());
  }
  if (options_.speculative_algorithm() == "DFlash") {
    // Both engines capture the same target layers, whose ids live in the draft
    // config: the draft engine reads its own weights path, the target engine
    // reads --draft_model. The draft engine additionally swaps in the
    // DFlashDraftModel body.
    std::string draft_config_path;
    if (options_.is_draft_engine()) {
      LOG(INFO) << "Overriding draft model_type from " << args.model_type()
                << " to DFlashDraftModel for DFlash speculative decoding";
      args.model_type("DFlashDraftModel");
      draft_config_path = model_weights_path_;
    } else {
      CHECK(options_.draft_model_path().has_value())
          << "DFlash requires --draft_model.";
      draft_config_path = options_.draft_model_path().value();
    }
    args.layers_to_capture(read_dflash_capture_layer_ids(draft_config_path));
  } else if (options_.enable_speculative_decode() &&
             ::xllm::SpeculativeConfig::get_instance()
                 .enable_atb_spec_kernel()) {
    args.num_speculative_tokens(options_.num_speculative_tokens());
  } else if (options_.enable_speculative_decode() &&
             options_.num_speculative_tokens() == 0 &&
             args.num_nextn_predict_layers() != 0) {
    const std::string& current_type = args.model_type();
    const char* mtp_model_type = nullptr;
    if (current_type == "qwen3_5_text") {
      mtp_model_type = "qwen3_5_mtp";
    } else if (current_type == "qwen3_5_moe_text") {
      mtp_model_type = "qwen3_5_moe_mtp";
    }
    if (mtp_model_type != nullptr) {
      LOG(INFO) << "Overriding draft model_type from " << current_type << " to "
                << mtp_model_type << " for speculative decoding";
      args.model_type(mtp_model_type);
      const int32_t mtp_layers = args.num_nextn_predict_layers();
      args.n_layers(mtp_layers);
      args.layer_types(std::vector<std::string>(mtp_layers, "full_attention"));
      args.full_attention_interval(1);
    }
  }
  // Eagle3/DFlash targets capture intermediate-layer aux hidden from the layers
  // in layers_to_capture, the model's sole capture signal. Fill the default
  // {2, n/2, n-3} for an Eagle3 target whose config omits the list; DFlash
  // already filled it from the draft config. The DFlash draft body
  // (DFlashDraftModel) consumes context-KV rather than capturing, so exclude
  // it.
  if (options_.enable_speculative_decode() &&
      SpeculativeConfig::requires_aux_hidden_capture(
          options_.speculative_algorithm()) &&
      args.model_type() != "DFlashDraftModel" &&
      args.layers_to_capture().empty()) {
    const int32_t num_layers = static_cast<int32_t>(args.n_layers());
    args.layers_to_capture({2, num_layers / 2, num_layers - 3});
  }
#else
  if (options_.enable_speculative_decode()) {
    args.num_speculative_tokens(options_.num_speculative_tokens());
    // When running speculative decoding, the draft worker reuses the same
    // checkpoint as the target model. The draft worker needs to instantiate
    // the MTP variant, so override the model_type here without mutating the
    // original config.
    if (options_.num_speculative_tokens() == 0 &&
        args.num_nextn_predict_layers() != 0) {
      static const std::unordered_map<std::string, std::string>
          kModelTypeToMtpType = {
              {"deepseek_v3", "deepseek_v3_mtp"},
              {"deepseek_v32", "deepseek_v3_mtp"},
              {"deepseek_v4", "deepseek_v4_mtp"},
              {"glm_moe_dsa", "glm_moe_dsa_mtp"},
              {"joyai_llm_flash", "joyai_llm_flash_mtp"},
              {"mimo", "mimo_mtp"},
          };
      const std::string& current_type = args.model_type();
      auto it = kModelTypeToMtpType.find(current_type);
      if (it != kModelTypeToMtpType.end()) {
        LOG(INFO) << "Overriding draft model_type from " << current_type
                  << " to " << it->second << " for speculative decoding";
        args.model_type(it->second);
      }
    }
  }
#endif

  setup_rl_sleep_weights();

  // create model context
  dtype_ = dtype;
  auto tensor_options = torch::dtype(dtype_).device(device_);
  context_ = ModelContext(parallel_args_, args, quant_args, tensor_options);
  context_.set_model_id(options_.model_id());

  // init model, create model executor
  bool status = this->init_model(context_);
  if (!status) {
    LOG(ERROR) << "init_model failed";
    return false;
  }

  int32_t tp_world_size = parallel_args_.world_size();
  if (parallel_args_.tp_group_) {
    tp_world_size = parallel_args_.tp_group_->world_size();
  }

  std::unique_ptr<ScopedAtenLoadThreads> scoped_load_threads;
  const int32_t prev_threads = torch::get_num_threads();
  LOG(INFO) << "Temporarily setting ATen threads to 1 during weight loading"
            << ", tp_world_size=" << tp_world_size
            << ", prev_threads=" << prev_threads;
  scoped_load_threads =
      std::make_unique<ScopedAtenLoadThreads>(/*target_threads=*/1);

  // In RL sleep mode (see setup_rl_sleep_weights()), the manual loader
  // allocates each layer's weight buffer from the SleepableAllocator (see
  // base_loader.cpp), so the weights become sleepable here without any extra
  // capture step.
  if (master_status == MasterStatus::WAKEUP) {
    this->load_model(std::move(model_loader));
  } else if (master_status == MasterStatus::LIGHT_SLEEP) {
    this->lazy_load_model(std::move(model_loader));
  }

  if (scoped_load_threads) {
    LOG(INFO) << "Weight loading completed, restored ATen threads="
              << torch::get_num_threads();
  }

  status_ = Status::LOADED;
  if (::xllm::EPLBConfig::get_instance().enable_eplb()) {
    // todo: support xtensor
    int32_t num_layers = args.n_layers() - args.first_k_dense_replace();
    int32_t num_device_experts =
        args.n_routed_experts() / context_.get_parallel_args().world_size() +
        ::xllm::EPLBConfig::get_instance().redundant_experts_num();
    expert_load_data_ = torch::zeros({num_layers, num_device_experts})
                            .to(torch::kInt64)
                            .to(device_)
                            .contiguous();
  }
  return true;
}

void WorkerImpl::load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
#if defined(USE_NPU)
  // Rolling mode uses host-pinned weights as the single source of truth:
  // lazy_load_model -> init_rolling_runtime_state() to finish rolling init.
  if (::xllm::LoadConfig::get_instance().enable_rolling_load() &&
      !is_spec_draft_) {
    model_->lazy_load_model(std::move(loader));
    CHECK(init_rolling_runtime_state())
        << "Failed to initialize rolling runtime state during load_model";
    return;
  }
#endif

  model_->load_model(std::move(loader));
}

#if defined(USE_NPU)
bool WorkerImpl::init_rolling_runtime_state() {
  // Draft model (speculative decoding) has only 1 decoder layer, skip rolling
  // load.
  if (!::xllm::LoadConfig::get_instance().enable_rolling_load() ||
      is_spec_draft_) {
    return true;
  }

  CHECK(model_ != nullptr) << "Model is not initialized for rolling load";
  CHECK(load_stream_ != nullptr) << "load_stream_ is null for rolling load";

  // Rolling runtime ownership is moved into model.
  // Worker provides runtime dependencies and delegates
  // initialization/refresh.
  const int32_t n_slots =
      ::xllm::LoadConfig::get_instance().rolling_load_num_cached_layers();
  const int32_t n_rolling_slots =
      ::xllm::LoadConfig::get_instance().rolling_load_num_rolling_slots();
  return model_->init_or_refresh_rolling_runtime(load_stream_.get(),
                                                 compute_stream_.get(),
                                                 n_slots,
                                                 n_rolling_slots,
                                                 options_.model_id());
}
#endif

void WorkerImpl::lazy_load_model(std::unique_ptr<ModelLoader> loader) {
  CHECK(model_ != nullptr) << "Model is not initialized.";
  model_->lazy_load_model(std::move(loader));
}

folly::SemiFuture<bool> WorkerImpl::allocate_kv_cache_async(
    const KVCacheShape& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        const bool success = this->allocate_kv_cache(kv_cache_shape);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::allocate_kv_cache_with_transfer_async(
    const KVCacheShape& kv_cache_shape) {
  folly::Promise<bool> promise;
  auto future = promise.getSemiFuture();
  threadpool_.schedule(
      [this, kv_cache_shape, promise = std::move(promise)]() mutable {
        const bool success =
            this->allocate_kv_cache_with_transfer(kv_cache_shape);
        promise.setValue(success);
      });
  return future;
}

folly::SemiFuture<bool> WorkerImpl::pull_kv_blocks_async(
    uint64_t src_cluster_id,
    const std::string& src_addr,
    const std::vector<uint64_t>& src_blocks,
    const std::vector<uint64_t>& dst_blocks,
    const std::vector<uint64_t>& src_linear_state_ids,
    const std::vector<uint64_t>& dst_linear_state_ids) {
#if defined(USE_NPU) || defined(USE_DCU)
  return kv_cache_transfer_->pull_kv_blocks_async(src_cluster_id,
                                                  src_addr,
                                                  src_blocks,
                                                  dst_blocks,
                                                  src_linear_state_ids,
                                                  dst_linear_state_ids);
#elif defined(USE_MLU)
  (void)src_cluster_id;
  (void)src_addr;
  (void)src_blocks;
  (void)dst_blocks;
  (void)src_linear_state_ids;
  (void)dst_linear_state_ids;
  LOG(FATAL) << "MLU backend does not support PULL kv cache transfer.";
#endif
  return false;
}

uint32_t WorkerImpl::transfer_kv_blocks(
    const uint64_t batch_id,
    const std::vector<BlockTransferInfo>& block_transfer_info) {
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    return hierarchy_kv_cache_transfer_->transfer_kv_blocks(
        batch_id, block_transfer_info);
  }
  return 0;
}

uint32_t WorkerImpl::transfer_kv_blocks(
    const uint64_t batch_id,
    Slice<BlockTransferInfo>& block_transfer_info) {
  if (hierarchy_kv_cache_transfer_ != nullptr) {
    return hierarchy_kv_cache_transfer_->transfer_kv_blocks(
        batch_id, block_transfer_info);
  }
  return 0;
}

int64_t WorkerImpl::get_active_activation_memory() {
  return DeviceMonitor::get_instance()
      .get_device_stats(device_.index())
      .active_activation_memory;
}

// hierarchy temporarily disabled during the block-manager refactor
// void WorkerImpl::init_hierarchy_kv_cache_transfer() {
//   if (options_.host_blocks_factor() > 1 || options_.enable_kvcache_store()) {
//     HierarchyKVCacheTransfer::Options transfer_options;
//     transfer_options
//         .tp_rank(options_.dp_size() > 1
//                      ? options_.node_rank() % options_.dp_size()
//                      : options_.node_rank())
//         .layers(context_.get_model_args().n_layers())
//         .host_blocks_factor(options_.host_blocks_factor())
//         .layers_wise_copy_batchs(options_.layers_wise_copy_batchs())
//         .enable_kvcache_store(options_.enable_kvcache_store())
//         .store_protocol(options_.store_protocol())
//         .store_master_server_address(options_.store_master_server_address())
//         .store_metadata_server(options_.store_metadata_server())
//         .store_local_hostname(options_.store_local_hostname());
//     hierarchy_kv_cache_transfer_ =
//     std::make_unique<HierarchyKVCacheTransfer>(
//         transfer_options, device_, &kv_caches_);
//   }
// }
void WorkerImpl::init_hierarchy_kv_cache_transfer(
    const KVCacheShape& kv_cache_shape,
    const KVCacheCreateOptions& kv_cache_create_options) {
  if (options_.host_blocks_factor() > 1.0) {
    CHECK(!kv_caches_.empty()) << "kv_caches is not initialized.";
    CHECK(hierarchy_kv_cache_transfer_ == nullptr)
        << "Hierarchy KV cache transfer is already initialized.";
    HierarchyKVCacheTransfer::Options transfer_options;
    transfer_options
        .tp_rank(options_.dp_size() > 1
                     ? options_.node_rank() % options_.dp_size()
                     : options_.node_rank())
        .tp_size(options_.world_size() / options_.dp_size())
        .layers(context_.get_model_args().n_layers())
        .host_blocks_factor(options_.host_blocks_factor())
        .layers_wise_copy_batchs(options_.layers_wise_copy_batchs())
        .enable_mla(options_.enable_mla())
        .enable_kvcache_store(false);
    hierarchy_kv_cache_transfer_ =
        std::make_unique<HierarchyKVCacheTransfer>(transfer_options,
                                                   device_,
                                                   &kv_caches_,
                                                   kv_cache_shape,
                                                   kv_cache_create_options);
  }
}
void WorkerImpl::prepare_mla_prefixcache_inputs(
    ModelInputParams& input_params) {
  const bool has_prefixcache_metadata =
      input_params.meta.num_sequences > 0 &&
      input_params.attention.device.kv_cache_tokens_nums.defined() &&
      input_params.attention.device.kv_cache_tokens_nums.numel() > 0 &&
      input_params.attention.device.q_seq_lens.defined() &&
      input_params.attention.device.q_seq_lens.numel() > 0;
  if (!has_prefixcache_metadata) {
    return;
  }
  int32_t sum_prefix =
      input_params.attention.device.kv_cache_tokens_nums.sum().item<int>();
  input_params.attention.device.history_compressed_kv =
      torch::empty({sum_prefix, context_.get_model_args().kv_lora_rank()},
                   torch::TensorOptions().dtype(dtype_).pinned_memory(true))
          .to(device_);

  input_params.attention.device.history_k_rope =
      torch::empty({sum_prefix, context_.get_model_args().qk_rope_head_dim()},
                   torch::TensorOptions().dtype(dtype_).pinned_memory(true))
          .to(device_);
  ;

  input_params.attention.device.ring_cur_seqlen =
      torch::stack({input_params.attention.device.q_seq_lens,
                    input_params.attention.device.q_seq_lens})
          .to(device_);

  input_params.attention.device.ring_cache_seqlen =
      torch::stack(
          {input_params.attention.device.q_seq_lens,
           input_params.attention.device.kv_cache_tokens_nums.to(device_)})
          .to(device_);

  torch::Tensor ring_cur_seqlen_host =
      input_params.attention.device.ring_cur_seqlen.cpu().contiguous();
  torch::Tensor ring_cache_seqlen_host =
      input_params.attention.device.ring_cache_seqlen.cpu().contiguous();
  input_params.attention.host.ring_cur_seqlen = std::vector<int>(
      ring_cur_seqlen_host.data_ptr<int>(),
      ring_cur_seqlen_host.data_ptr<int>() + ring_cur_seqlen_host.numel());
  input_params.attention.host.ring_cache_seqlen = std::vector<int>(
      ring_cache_seqlen_host.data_ptr<int>(),
      ring_cache_seqlen_host.data_ptr<int>() + ring_cache_seqlen_host.numel());
}

int64_t WorkerImpl::get_num_layers() const {
  int64_t num_layers = context_.get_model_args().n_layers();
#if !defined(USE_NPU)
  if (is_spec_draft_) {
    // for MTP draft models, the number of layers is the number of nextn
    // predict layers
    int64_t num_nextn_predict_layers =
        context_.get_model_args().num_nextn_predict_layers();
    if (num_nextn_predict_layers > 0) {
      return num_nextn_predict_layers;
    }
  }
#endif
  return num_layers;
}

}  // namespace xllm
