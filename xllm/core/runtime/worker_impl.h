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

#pragma once

#include <folly/futures/Future.h>
#include <sys/mman.h>
#include <torch/torch.h>

#include <memory>

#include "common/types.h"
#include "executor.h"
#include "forward_params.h"
#include "framework/eplb/eplb_executor.h"
#include "framework/kv_cache/kv_cache_shape.h"
#include "framework/kv_cache_transfer/hierarchy_kv_cache_transfer.h"
#include "framework/kv_cache_transfer/kv_cache_store.h"
#include "framework/kv_cache_transfer/kv_cache_transfer.h"
#include "framework/model/causal_lm.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/sampling/beam_searcher.h"
#include "framework/sampling/sampler.h"
#include "framework/state_dict/state_dict.h"
#include "framework/xtensor/xtensor.h"
#include "options.h"
#include "platform/device.h"
#include "util/threadpool.h"
#if defined(USE_NPU)
#include "framework/kv_cache_transfer/mooncake_weight_transfer.h"
#include "layers/npu/loader/rolling_load_manager.h"
#endif

namespace xllm {

class WorkerRendezvous;

class WorkerImpl {
 public:
  enum Status : int8_t {
    UNINITIALIZED = 0,
    LOADED,
    READY,
  };

  WorkerImpl(const ParallelArgs& parallel_args,
             const torch::Device& device,
             const runtime::Options& options);

  virtual ~WorkerImpl();

  // initialize model, cache manager. blocking call
  virtual bool init_model(ModelContext& context) = 0;

  virtual bool init_model(const std::string& model_weights_path,
                          int32_t random_seed,
                          MasterStatus master_status);

  virtual void load_model(std::unique_ptr<ModelLoader> loader);

  virtual void lazy_load_model(std::unique_ptr<ModelLoader> loader);

  virtual std::tuple<int64_t, int64_t> estimate_kv_cache_capacity();

  // allocate kv cache. blocking call
  virtual bool allocate_kv_cache(const KVCacheShape& kv_cache_shape);

  virtual bool allocate_kv_cache_with_transfer(
      const KVCacheShape& kv_cache_shape);

#if defined(USE_NPU) || defined(USE_MLU) || defined(USE_DCU)
  virtual bool allocate_kv_cache_with_transfer(
      std::shared_ptr<KVCacheTransfer> kv_cache_transfer,
      const KVCacheShape& kv_cache_shape);
#endif

  virtual void get_cache_info(uint64_t& cluster_id,
                              std::string& addr,
                              uint16_t& port);

  virtual bool link_cluster(const std::vector<uint64_t>& cluster_ids,
                            const std::vector<std::string>& addrs,
                            const std::vector<uint16_t>& ports);

  virtual bool unlink_cluster(const std::vector<uint64_t>& cluster_ids,
                              const std::vector<std::string>& addrs,
                              const std::vector<uint16_t>& ports);

  // P2P link for weight transfer
  virtual bool link_p2p(const std::string& remote_addr);
  virtual bool unlink_p2p(const std::string& remote_addr);

  // prepare input for execution
  virtual ForwardInput prepare_inputs(Batch& batch);

  // prepare work before model execution
  virtual void prepare_work_before_execute(const ForwardInput& inputs,
                                           ForwardInput& processed_inputs);
  void prepare_work_before_execute_on_stream(const ForwardInput& input,
                                             ForwardInput& processed_input,
                                             Stream& prepare_stream);

  // Internal helper shared by worker pipelines before model execution.
  virtual void apply_kv_block_swaps(const ModelInputParams& input_params);

  virtual std::optional<ForwardOutput> step(const ForwardInput& inputs) = 0;

  virtual void process_group_test();

  virtual ForwardInput update_input_by_last_step_output(ForwardInput& inputs);

  // initialize model, cache manager. async call
  virtual folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path,
      int32_t random_seed,
      MasterStatus master_status);

  virtual folly::SemiFuture<std::tuple<int64_t, int64_t>>
  estimate_kv_cache_capacity_async();

  // initialize kv cache. async call
  virtual folly::SemiFuture<bool> allocate_kv_cache_async(
      const KVCacheShape& kv_cache_shape);

  virtual folly::SemiFuture<bool> allocate_kv_cache_with_transfer_async(
      const KVCacheShape& kv_cache_shape);

  virtual bool sleep(MasterStatus master_status);

  virtual bool wakeup(const WakeupOptions& options);

  // RL deep-sleep: reload model weights in place from disk (vllm-ascend
  // `reload_weights`). `weights_path` empty = reuse the original model path.
  virtual bool update_weights(const std::string& weights_path);

  // Start/stop online timeline profiling on this worker's device. CUDA only
  // for now; on other backends these are no-ops returning false.
  virtual bool start_profile();

  virtual bool stop_profile();

  virtual folly::SemiFuture<bool> pull_kv_blocks_async(
      uint64_t src_cluster_id,
      const std::string& src_addr,
      const std::vector<uint64_t>& src_blocks,
      const std::vector<uint64_t>& dst_blocks,
      const std::vector<uint64_t>& src_linear_state_ids = {},
      const std::vector<uint64_t>& dst_linear_state_ids = {});

  virtual uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      const std::vector<BlockTransferInfo>& block_transfer_info);

  virtual uint32_t transfer_kv_blocks(
      const uint64_t batch_id,
      Slice<BlockTransferInfo>& block_transfer_info);

  // Run the model on the given input. async call
  // the future returns a successfull status with no meaningful value
  virtual folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs);

  virtual folly::SemiFuture<folly::Unit> process_group_test_async();

  const torch::Device& device() const { return device_.unwrap(); }

  torch::ScalarType dtype() const { return dtype_; }

  int32_t hidden_size() const {
    return context_.get_model_args().hidden_size();
  }

  bool enable_schedule_overlap() const {
    return options_.enable_schedule_overlap_;
  }

  virtual ForwardOutput get_last_step_result();

  bool is_driver() const { return driver_ || dp_driver_; }

  int64_t get_active_activation_memory();

  Status get_status() const { return status_; }

  // model context, includes model args, parallel args and date type etc.
  mutable ModelContext context_;

  OptimizationConfig get_optimization_config() const {
    return context_.get_optimization_config();
  }

 protected:
  void update_last_step_output(const std::optional<ForwardOutput>& output);
  virtual std::optional<ForwardOutput> step_for_schedule_overlap(
      const ForwardInput& input);
  virtual ForwardInput update_input_by_last_step_output_for_schedule_overlap(
      ForwardInput& input);
  // Only used for deepseek chunked prefill ops on npu device
  void prepare_mla_prefixcache_inputs(ModelInputParams& input_params);

  void init_hierarchy_kv_cache_transfer(
      const KVCacheShape& kv_cache_shape,
      const KVCacheCreateOptions& kv_cache_create_options);

  bool can_prepare_npu_graph_decode_input(
      const ModelInputParams& input_params) const;
  bool can_prepare_without_compute_stream_wait(
      const ModelInputParams& input_params) const;
  bool can_skip_npu_graph_decode_sync(
      const ModelInputParams& input_params) const;

  bool allocate_kv_cache_storage(
      const KVCacheShape& kv_cache_shape,
      bool use_huge_page_allocator = false,
      std::shared_ptr<KVCacheTensorAllocator> tensor_allocator = nullptr);

  // Get the effective number of layers based on whether this is a spec draft
  // model
  int64_t get_num_layers() const;

  bool wakeup_local(const WakeupOptions& options);

  // ---- RL deep-sleep path (SleepableAllocator), isolated from the xtensor
  // ---- (PageAllocator) sleep/wakeup path. ----
  // True when this worker uses the RL SleepableAllocator path rather than the
  // xtensor PageAllocator path.
  bool rl_sleep_mode() const;
  // Enable manual-loader weight routing into the SleepableAllocator. Called
  // once during init_model before the layers are constructed.
  void setup_rl_sleep_weights();
  // Deep sleep / wake the SleepableAllocator regions (weights + KV).
  bool rl_sleep();
  bool rl_wakeup();
  // Original xtensor (PageAllocator) sleep path.
  bool xtensor_sleep(MasterStatus master_status);

#if defined(USE_CUDA) || defined(USE_DCU)
  void refresh_cuda_block_copy_runtime_state();
  bool can_use_cuda_block_copy_kernel(
      const ModelInputParams& input_params) const;
  void execute_cuda_block_copy_kernel(const ModelInputParams& input_params);

  struct CudaBlockCopyRuntimeState {
    torch::Tensor k_cache_ptrs_device;
    torch::Tensor v_cache_ptrs_device;
    int64_t num_layers = 0;
    int64_t numel_per_block = 0;

    bool valid() const {
      return k_cache_ptrs_device.defined() && v_cache_ptrs_device.defined() &&
             num_layers > 0 && numel_per_block > 0;
    }
  };
#endif

#if defined(USE_NPU)
  bool wakeup_from_remote_weights(const WakeupOptions& options);
  // Complete rolling initialization by delegating to model-owned rolling
  // runtime (manager + buffer): decoder preload, non-decoder reload, and
  // decoder ATB binding refresh.
  bool init_rolling_runtime_state();

  torch::Tensor recompute_new_cache_slots(const ForwardInput& input);
  torch::Tensor compute_in_prefix_slots(const ForwardInput& input);
#endif

 protected:
  // runtime options
  runtime::Options options_;

  // whether the worker is a driver, who takes care of the sampling
  bool driver_ = false;
  bool dp_driver_ = false;

  // working thread
  // make sure only 1 thread in the pool
  // if enable_schedule_overlap, two step tasks might be dispatched to
  // the task queue, step need to be executed one-by-one
  // INVARIANT: num_threads must remain 1. The linear-state prefix cache (both
  // the in-worker restore on compute_stream_ and the lockless save fast-path
  // in worker_service) relies on FIFO single-thread ordering to chain
  // consecutive step tasks on compute_stream_ without an extra cross-stream
  // barrier. Raising num_threads above 1 would re-introduce the cross-thread
  // race fixed in commit 6e350f8f and silently break linear-state correctness.
  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"WorkerImpl.schedule"};

  // dtype of the model
  torch::ScalarType dtype_;

  // device to run the model on
  Device device_;

  std::unique_ptr<Stream> prepare_stream_;
  std::unique_ptr<Stream> compute_stream_;

  // parallel args of current instance
  ParallelArgs parallel_args_;

  // kv caches
  std::vector<xllm::KVCache> kv_caches_;

  // causal LM model
  std::unique_ptr<CausalLM> model_;

  std::unique_ptr<Executor> model_executor_;

  std::unique_ptr<Sampler> sampler_;

  std::unique_ptr<EplbExecutor> eplb_executor_;

  // params for enable_schedule_overlap case
  // an output to store the result of last step
  ForwardOutput last_step_output_;
  bool last_step_output_valid_ = false;
  std::mutex mtx_;
  std::condition_variable cv_;
  bool is_recorded_ = false;

  InstanceRole instance_role_ = InstanceRole::DEFAULT;

  std::shared_ptr<KVCacheTransfer> kv_cache_transfer_;
  std::unique_ptr<HierarchyKVCacheTransfer> hierarchy_kv_cache_transfer_;
  std::unique_ptr<WorkerRendezvous> worker_rendezvous_;

#if defined(USE_CUDA) || defined(USE_DCU)
  CudaBlockCopyRuntimeState cuda_block_copy_runtime_state_;
#endif

#if defined(USE_NPU)
  std::unique_ptr<MooncakeWeightTransfer> weight_transfer_;
  std::unique_ptr<Stream> load_stream_;
#endif

  bool is_spec_draft_ = false;

  Status status_ = Status::UNINITIALIZED;

  torch::Tensor expert_load_data_;

  std::string model_weights_path_;
};

}  // namespace xllm
