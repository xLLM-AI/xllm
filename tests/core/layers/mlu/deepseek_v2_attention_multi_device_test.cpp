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

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cerrno>
#include <chrono>
#include <csignal>
#include <cstring>
#include <memory>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "core/framework/config/kv_cache_config.h"
#include "framework/batch/batch_forward_type.h"
#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/process_group.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/deepseek_v2_attention.h"
#include "layers/mlu/deepseek_v32_cp_context.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "platform/platform.h"
#include "util/net.h"

namespace xllm {
namespace layer {
namespace test {

namespace {

KVCache create_indexed_kv_cache(torch::Tensor key_cache,
                                torch::Tensor index_cache) {
  return KVCache(IndexedKVCacheTensors{
      KVCacheTensors{key_cache, torch::Tensor()}, index_cache});
}

constexpr int32_t EXIT_CODE_SKIP = 77;

constexpr std::chrono::seconds kChildProcessTimeout{120};
constexpr std::chrono::seconds kChildTerminationGracePeriod{2};
constexpr std::chrono::milliseconds kChildPollInterval{10};

struct ChildWaitResult {
  bool any_failed = false;
  bool any_skipped = false;
  bool timed_out = false;
};

pid_t wait_for_pid(pid_t pid, int* status, int options) {
  pid_t result;
  do {
    result = ::waitpid(pid, status, options);
  } while (result < 0 && errno == EINTR);
  return result;
}

void record_child_status(size_t rank, int status, ChildWaitResult& result) {
  if (WIFEXITED(status)) {
    const int32_t exit_code = WEXITSTATUS(status);
    if (exit_code == EXIT_CODE_SKIP) {
      result.any_skipped = true;
    } else if (exit_code != 0) {
      result.any_failed = true;
      LOG(ERROR) << "Rank " << rank << " failed with code " << exit_code;
    }
    return;
  }

  result.any_failed = true;
  if (WIFSIGNALED(status)) {
    LOG(ERROR) << "Rank " << rank << " terminated by signal "
               << WTERMSIG(status);
  } else {
    LOG(ERROR) << "Rank " << rank << " ended unexpectedly";
  }
}

bool reap_children_until(
    const std::vector<pid_t>& child_pids,
    std::vector<bool>& finished,
    const std::chrono::steady_clock::time_point& deadline) {
  while (std::chrono::steady_clock::now() < deadline) {
    bool all_finished = true;
    for (size_t rank = 0; rank < child_pids.size(); ++rank) {
      if (finished[rank]) {
        continue;
      }

      int status = 0;
      const pid_t wait_result =
          wait_for_pid(child_pids[rank], &status, WNOHANG);
      if (wait_result == child_pids[rank] ||
          (wait_result < 0 && errno == ECHILD)) {
        finished[rank] = true;
        continue;
      }
      if (wait_result < 0) {
        LOG(ERROR) << "Failed to reap rank " << rank << ": "
                   << std::strerror(errno);
      }
      all_finished = false;
    }
    if (all_finished) {
      return true;
    }
    std::this_thread::sleep_for(kChildPollInterval);
  }
  return false;
}

void signal_unfinished_children(const std::vector<pid_t>& child_pids,
                                const std::vector<bool>& finished,
                                int signal_number) {
  for (size_t rank = 0; rank < child_pids.size(); ++rank) {
    if (finished[rank]) {
      continue;
    }
    if (::kill(child_pids[rank], signal_number) != 0 && errno != ESRCH) {
      LOG(ERROR) << "Failed to signal rank " << rank << ": "
                 << std::strerror(errno);
    }
  }
}

void terminate_and_reap_children(const std::vector<pid_t>& child_pids,
                                 std::vector<bool>& finished) {
  signal_unfinished_children(child_pids, finished, SIGTERM);
  const auto term_deadline =
      std::chrono::steady_clock::now() + kChildTerminationGracePeriod;
  if (reap_children_until(child_pids, finished, term_deadline)) {
    return;
  }

  signal_unfinished_children(child_pids, finished, SIGKILL);
  const auto kill_deadline =
      std::chrono::steady_clock::now() + kChildTerminationGracePeriod;
  reap_children_until(child_pids, finished, kill_deadline);
  for (size_t rank = 0; rank < child_pids.size(); ++rank) {
    if (!finished[rank]) {
      LOG(ERROR) << "Unable to reap timed-out rank " << rank;
    }
  }
}

ChildWaitResult wait_for_children(
    const std::vector<pid_t>& child_pids,
    std::chrono::steady_clock::duration timeout = kChildProcessTimeout) {
  ChildWaitResult result;
  std::vector<bool> finished(child_pids.size(), false);
  size_t remaining = child_pids.size();
  const auto deadline = std::chrono::steady_clock::now() + timeout;

  while (remaining > 0 && std::chrono::steady_clock::now() < deadline) {
    bool made_progress = false;
    for (size_t rank = 0; rank < child_pids.size(); ++rank) {
      if (finished[rank]) {
        continue;
      }

      int status = 0;
      const pid_t wait_result =
          wait_for_pid(child_pids[rank], &status, WNOHANG);
      if (wait_result == child_pids[rank]) {
        finished[rank] = true;
        --remaining;
        made_progress = true;
        record_child_status(rank, status, result);
      } else if (wait_result < 0) {
        result.any_failed = true;
        finished[rank] = true;
        --remaining;
        made_progress = true;
        LOG(ERROR) << "Failed waiting for rank " << rank << ": "
                   << std::strerror(errno);
      }
    }
    if (remaining > 0 && !made_progress) {
      std::this_thread::sleep_for(kChildPollInterval);
    }
  }

  if (remaining == 0) {
    return result;
  }

  result.any_failed = true;
  result.timed_out = true;
  const int64_t timeout_ms =
      std::chrono::duration_cast<std::chrono::milliseconds>(timeout).count();
  LOG(ERROR) << "Timed out waiting for " << remaining
             << " multi-device test rank(s) after " << timeout_ms << " ms";
  terminate_and_reap_children(child_pids, finished);
  return result;
}

std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  std::unique_ptr<xllm::ProcessGroup> process_group =
      xllm::create_process_group(rank,
                                 world_size,
                                 world_size,
                                 port,
                                 false,
                                 host,
                                 "attention_multi_device_test_group",
                                 device);
  CHECK(process_group) << "Failed to create the test process group";

  // TCPStore construction does not guarantee that every client has joined
  // before rank 0 can leave a short test. Force one collective rendezvous so
  // the server stays alive until all ranks have connected.
  torch::Tensor rendezvous = torch::ones(
      {1}, torch::TensorOptions().dtype(torch::kFloat32).device(device));
  process_group->allreduce(rendezvous);
  CHECK_EQ(rendezvous.item<float>(), static_cast<float>(world_size))
      << "Multi-device test rendezvous failed";
  return process_group;
}

std::unique_ptr<xllm::ProcessGroup> create_test_self_group(
    int32_t rank,
    int32_t world_size,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(
      rank,
      world_size,
      /*rank_size=*/1,
      /*port=*/0,
      false,
      host,
      "attention_multi_device_single_rank_group_" + std::to_string(rank),
      device);
}

class DistributedAttentionChildContext final {
 public:
  DistributedAttentionChildContext(int32_t rank,
                                   int32_t world_size,
                                   int32_t device_count,
                                   int32_t port,
                                   const std::string& host,
                                   bool enable_context_parallel)
      : xllm_device_(rank % device_count),
        device_(xllm_device_.unwrap()),
        options_(torch::TensorOptions()
                     .dtype(torch::kBFloat16)
                     .device(device_)
                     .requires_grad(false)) {
    xllm_device_.set_device();
    process_group_ =
        create_test_process_group(rank, world_size, port, host, device_);
    self_group_ = create_test_self_group(rank, world_size, host, device_);
    CHECK(process_group_) << "Rank " << rank
                          << ": failed to create process group";
    CHECK(self_group_) << "Rank " << rank << ": failed to create self group";

    parallel_args_ =
        std::make_unique<ParallelArgs>(rank, world_size, process_group_.get());
    parallel_args_->tp_group_ = process_group_.get();
    parallel_args_->single_rank_group_ = self_group_.get();
    parallel_args_->cp_group_ = process_group_.get();
    if (enable_context_parallel) {
      parallel_args_->cp_size() = world_size;
    }
  }

  xllm::Device& xllm_device() { return xllm_device_; }

  const torch::Device& device() const { return device_; }

  ParallelArgs& parallel_args() { return *parallel_args_; }

  const torch::TensorOptions& options() const { return options_; }

 private:
  xllm::Device xllm_device_;
  torch::Device device_;
  std::unique_ptr<xllm::ProcessGroup> process_group_;
  std::unique_ptr<xllm::ProcessGroup> self_group_;
  std::unique_ptr<ParallelArgs> parallel_args_;
  torch::TensorOptions options_;
};

template <typename ChildBody>
int32_t run_distributed_attention_child(int32_t rank,
                                        int32_t world_size,
                                        int32_t port,
                                        const std::string& host,
                                        bool enable_context_parallel,
                                        const ChildBody& child_body) {
  try {
    const int32_t device_count = xllm::Platform::device_count();
    if (device_count < world_size) {
      LOG(WARNING) << "Rank " << rank << ": insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }

    KVCacheConfig::get_instance().block_size(16);
    DistributedAttentionChildContext context(
        rank, world_size, device_count, port, host, enable_context_parallel);
    return child_body(context);
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << rank << ": Exception: " << e.what();
    return 1;
  }
}

void check_tensors_close(const torch::Tensor& actual,
                         const torch::Tensor& expected,
                         double rtol,
                         double atol,
                         const std::string& label) {
  CHECK_EQ(actual.sizes(), expected.sizes())
      << label << " shape mismatch: actual=" << actual.sizes()
      << ", expected=" << expected.sizes();
  torch::Tensor actual_fp32 = actual.to(torch::kFloat32);
  torch::Tensor expected_fp32 = expected.to(torch::kFloat32);
  torch::Tensor diff = torch::abs(actual_fp32 - expected_fp32);
  CHECK(torch::allclose(actual_fp32, expected_fp32, rtol, atol))
      << label << " mismatch. max_diff=" << torch::max(diff).item<float>()
      << ", mean_diff=" << torch::mean(diff).item<float>();
}

torch::Tensor get_full_out(const torch::Tensor& output,
                           xllm::ProcessGroup* tp_group,
                           int64_t hidden_size) {
  if (output.size(-1) == hidden_size) {
    if (tp_group != nullptr && tp_group->world_size() > 1) {
      torch::Tensor reduced = output.clone();
      return xllm::parallel_state::reduce(reduced, tp_group);
    }
    return output;
  }
  CHECK(tp_group != nullptr) << "tp_group is required to gather TP output";
  return xllm::parallel_state::gather(output, tp_group);
}

void check_attn_out_close(const torch::Tensor& sharded_output,
                          const torch::Tensor& full_output,
                          xllm::ProcessGroup* tp_group,
                          int64_t hidden_size,
                          double rtol,
                          double atol,
                          const std::string& label) {
  CHECK_EQ(full_output.size(-1), hidden_size)
      << label << " full-weight output width mismatch";
  check_tensors_close(get_full_out(sharded_output, tp_group, hidden_size),
                      full_output,
                      rtol,
                      atol,
                      label);
}

ModelArgs create_attention_model_args(int64_t q_lora_rank = 0,
                                      bool enable_indexer = false) {
  ModelArgs args;
  args.model_type() = "deepseek_v32";
  args.hidden_size() = 256;
  args.n_heads() = 4;
  args.max_position_embeddings() = 128;
  args.rope_theta() = 10000.0f;
  args.rms_norm_eps() = 1e-6f;
  args.q_lora_rank() = q_lora_rank;
  args.kv_lora_rank() = enable_indexer ? 256 : 128;
  args.qk_nope_head_dim() = 128;
  args.qk_rope_head_dim() = 64;
  args.v_head_dim() = 128;
  args.index_n_heads() = enable_indexer ? 64 : 0;
  args.index_head_dim() = enable_indexer ? 128 : 0;
  args.index_topk() = enable_indexer ? 8 : 0;
  args.sliding_window() = 0;
  args.enable_mla() = true;
  args.rope_scaling_rope_type() = "";
  args.rope_scaling_original_max_position_embeddings() = 128;
  args.rope_scaling_factor() = 1.0f;
  args.rope_extrapolation_factor() = 1.0f;
  args.rope_scaling_attn_factor() = 1.0f;
  args.rope_scaling_beta_fast() = 1;
  args.rope_scaling_beta_slow() = 1;
  args.rope_scaling_mscale() = 1.0f;
  args.rope_scaling_mscale_all_dim() = 1.0f;
  return args;
}

ModelArgs create_glm5_attention_model_args() {
  ModelArgs args =
      create_attention_model_args(/*q_lora_rank=*/64, /*enable_indexer=*/true);
  args.model_type() = "glm_moe_dsa";
  args.qk_nope_head_dim() = 192;
  args.v_head_dim() = 256;
  args.index_n_heads() = 32;
  args.rope_theta() = 1000000.0f;
  args.rope_scaling_rope_type() = "default";
  args.indexer_rope_interleave() = true;
  return args;
}

StateDict create_attention_state_dict(const ModelArgs& args,
                                      const torch::TensorOptions& options) {
  const int64_t hidden_size = args.hidden_size();
  const int64_t num_heads = args.n_heads();
  const int64_t kv_lora_rank = args.kv_lora_rank();
  const int64_t qk_nope_head_dim = args.qk_nope_head_dim();
  const int64_t qk_rope_head_dim = args.qk_rope_head_dim();
  const int64_t v_head_dim = args.v_head_dim();
  const int64_t qk_head_dim = qk_nope_head_dim + qk_rope_head_dim;
  const int64_t q_lora_rank = args.q_lora_rank();
  const int64_t index_n_heads = args.index_n_heads();
  const int64_t index_head_dim = args.index_head_dim();

  std::unordered_map<std::string, torch::Tensor> weights;
  if (q_lora_rank > 0) {
    weights["q_a_proj.weight"] =
        seeded_tensor("attention_multi_device/q_a_proj/weight",
                      {q_lora_rank, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["q_a_layernorm.weight"] =
        seeded_tensor("attention_multi_device/q_a_layernorm/weight",
                      {q_lora_rank},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["q_b_proj.qweight"] =
        seeded_tensor("attention_multi_device/q_b_proj/qweight",
                      {num_heads * qk_head_dim, q_lora_rank},
                      torch::kInt8,
                      options.device());
    weights["q_b_proj.per_channel_scale"] =
        seeded_tensor("attention_multi_device/q_b_proj/per_channel_scale",
                      {num_heads * qk_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
    weights["q_b_proj.smooth"] =
        seeded_tensor("attention_multi_device/q_b_proj/smooth",
                      {q_lora_rank},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
  } else {
    weights["q_proj.qweight"] =
        seeded_tensor("attention_multi_device/q_proj/qweight",
                      {num_heads * qk_head_dim, hidden_size},
                      torch::kInt8,
                      options.device());
    weights["q_proj.per_channel_scale"] =
        seeded_tensor("attention_multi_device/q_proj/per_channel_scale",
                      {num_heads * qk_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
    weights["q_proj.smooth"] =
        seeded_tensor("attention_multi_device/q_proj/smooth",
                      {hidden_size},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.25f);
  }

  weights["kv_a_proj_with_mqa.weight"] =
      seeded_tensor("attention_multi_device/kv_a_proj_with_mqa/weight",
                    {kv_lora_rank + qk_rope_head_dim, hidden_size},
                    torch::kBFloat16,
                    options.device());
  weights["kv_a_layernorm.weight"] =
      seeded_tensor("attention_multi_device/kv_a_layernorm/weight",
                    {kv_lora_rank},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.5f);
  weights["kv_b_proj.weight"] =
      seeded_tensor("attention_multi_device/kv_b_proj/weight",
                    {num_heads * (qk_nope_head_dim + v_head_dim), kv_lora_rank},
                    torch::kBFloat16,
                    options.device());

  weights["o_proj.qweight"] =
      seeded_tensor("attention_multi_device/o_proj/qweight",
                    {hidden_size, num_heads * v_head_dim},
                    torch::kInt8,
                    options.device());
  weights["o_proj.per_channel_scale"] =
      seeded_tensor("attention_multi_device/o_proj/per_channel_scale",
                    {hidden_size},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.25f);
  weights["o_proj.smooth"] =
      seeded_tensor("attention_multi_device/o_proj/smooth",
                    {num_heads * v_head_dim},
                    torch::kFloat32,
                    options.device())
          .abs()
          .add_(0.25f);

  if (index_n_heads > 0) {
    weights["indexer.k_norm.bias"] =
        seeded_tensor("attention_multi_device/indexer/k_norm/bias",
                      {index_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.k_norm.weight"] =
        seeded_tensor("attention_multi_device/indexer/k_norm/weight",
                      {index_head_dim},
                      torch::kFloat32,
                      options.device())
            .abs()
            .add_(0.5f);
    weights["indexer.weights_proj.weight"] =
        seeded_tensor("attention_multi_device/indexer/weights_proj/weight",
                      {index_n_heads, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["indexer.wk.weight"] =
        seeded_tensor("attention_multi_device/indexer/wk/weight",
                      {index_head_dim, hidden_size},
                      torch::kBFloat16,
                      options.device());
    weights["indexer.wq_b.weight"] =
        seeded_tensor("attention_multi_device/indexer/wq_b/weight",
                      {index_n_heads * index_head_dim, q_lora_rank},
                      torch::kBFloat16,
                      options.device());
  }

  return StateDict(std::move(weights));
}

AttentionMetadata create_decode_metadata(const torch::TensorOptions& options) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, 1}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, 0}, int_options);
  metadata.kv_seq_lens = torch::tensor({1}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::tensor({1}, int_options);
  metadata.max_query_len = 1;
  metadata.max_seq_len = 1;
  metadata.total_kv_len = 1;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

// Decode metadata for a bucket of `batch_size` single-token sequences,
// mirroring how the MLU graph pads num_tokens into a fixed bucket. Each
// sequence maps to a distinct KV block so the sparse block table has
// `batch_size` rows.
AttentionMetadata create_batch_decode_metadata(
    const torch::TensorOptions& options,
    int32_t batch_size) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::arange(batch_size + 1, int_options);
  metadata.kv_cu_seq_lens = torch::zeros({batch_size + 1}, int_options);
  metadata.kv_seq_lens = torch::ones({batch_size}, int_options);
  metadata.block_table = torch::zeros({batch_size, 4}, int_options);
  for (int32_t i = 0; i < batch_size; ++i) {
    metadata.block_table[i][0] = i + 1;
  }
  metadata.slot_mapping = torch::arange(1, batch_size + 1, int_options);
  metadata.max_query_len = 1;
  metadata.max_seq_len = 1;
  metadata.total_kv_len = batch_size;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_prefill_metadata(const torch::TensorOptions& options,
                                          int32_t seq_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  metadata.q_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, seq_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({seq_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::arange(seq_len, int_options);
  metadata.max_query_len = seq_len;
  metadata.max_seq_len = seq_len;
  metadata.total_kv_len = seq_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = true;
  metadata.is_chunked_prefill = false;
  metadata.is_dummy = false;
  return metadata;
}

AttentionMetadata create_chunked_metadata(const torch::TensorOptions& options,
                                          int32_t prefix_len,
                                          int32_t chunk_len) {
  AttentionMetadata metadata;
  auto int_options = options.dtype(torch::kInt32);
  const int32_t ctx_len = prefix_len + chunk_len;
  metadata.q_cu_seq_lens = torch::tensor({0, chunk_len}, int_options);
  metadata.kv_cu_seq_lens = torch::tensor({0, ctx_len}, int_options);
  metadata.kv_seq_lens = torch::tensor({ctx_len}, int_options);
  metadata.block_table = torch::zeros({1, 4}, int_options);
  metadata.block_table[0][0] = 1;
  metadata.slot_mapping = torch::arange(prefix_len, ctx_len, int_options);
  metadata.max_query_len = chunk_len;
  metadata.max_seq_len = ctx_len;
  metadata.total_kv_len = ctx_len;
  metadata.compute_dtype = "half";
  metadata.is_prefill = false;
  metadata.is_chunked_prefill = true;
  metadata.is_dummy = false;
  return metadata;
}

KVCache create_decode_kv_cache(const ModelArgs& args,
                               const torch::TensorOptions& options) {
  const int64_t cache_width = args.qk_rope_head_dim() + args.kv_lora_rank();
  torch::Tensor k_cache = seeded_tensor(
      "attention_multi_device/k_cache",
      {8, 1, KVCacheConfig::get_instance().block_size(), cache_width},
      torch::kBFloat16,
      options.device());
  torch::Tensor index_cache = torch::Tensor();
  if (args.index_head_dim() > 0) {
    index_cache = seeded_tensor("attention_multi_device/index_cache",
                                {8,
                                 1,
                                 KVCacheConfig::get_instance().block_size(),
                                 args.index_head_dim()},
                                torch::kBFloat16,
                                options.device());
  }
  return create_indexed_kv_cache(std::move(k_cache), std::move(index_cache));
}

struct AttentionRunResult {
  torch::Tensor local_output;
  torch::Tensor global_output;
  torch::Tensor k_cache;
  torch::Tensor index_cache;
  bool used_sp = false;
  int64_t local_token_num = 0;
};

void check_repl_output_contract(const AttentionRunResult& result,
                                int64_t token_num,
                                int64_t hidden_size,
                                const std::string& label) {
  CHECK(!result.used_sp) << label << " unexpectedly used sequence parallel";
  CHECK_EQ(result.local_token_num, token_num)
      << label << " local token count mismatch";
  CHECK_EQ(result.local_output.sizes(), result.global_output.sizes())
      << label << " local/global output shape mismatch";
  CHECK_EQ(result.local_output.size(0), token_num)
      << label << " output token count mismatch";
  CHECK_EQ(result.local_output.size(1), hidden_size)
      << label << " output hidden size mismatch";
  check_tensors_close(result.local_output,
                      result.global_output,
                      /*rtol=*/0.0,
                      /*atol=*/0.0,
                      label + " local/global");
}

void check_sp_output_contract(const AttentionRunResult& result,
                              int64_t token_num,
                              int64_t hidden_size,
                              int64_t local_token_num,
                              const std::string& label) {
  CHECK(result.used_sp) << label << " did not use sequence parallel";
  CHECK_EQ(result.local_token_num, local_token_num)
      << label << " local token count mismatch";
  CHECK_EQ(result.local_output.size(0), local_token_num)
      << label << " packed local row count mismatch";
  CHECK_EQ(result.local_output.size(1), hidden_size)
      << label << " packed local hidden size mismatch";
  CHECK_EQ(result.global_output.size(0), token_num)
      << label << " restored token count mismatch";
  CHECK_EQ(result.global_output.size(1), hidden_size)
      << label << " restored hidden size mismatch";
}

void check_tensor_same_across_ranks(const torch::Tensor& tensor,
                                    xllm::ProcessGroup* process_group,
                                    const std::string& label) {
  if (process_group == nullptr || process_group->world_size() <= 1) {
    return;
  }

  torch::Tensor gathered =
      xllm::parallel_state::gather(tensor, process_group, /*dim=*/0);
  const int64_t rows = tensor.size(0);
  torch::Tensor ref = gathered.slice(0, 0, rows);
  for (int64_t rank = 1; rank < process_group->world_size(); ++rank) {
    torch::Tensor curr = gathered.slice(0, rank * rows, (rank + 1) * rows);
    check_tensors_close(curr,
                        ref,
                        /*rtol=*/0.0,
                        /*atol=*/0.0,
                        label + " rank " + std::to_string(rank));
  }
}

torch::Tensor run_attention_decode_once(const ModelArgs& args,
                                        const QuantArgs& quant_args,
                                        const ParallelArgs& parallel_args,
                                        const torch::TensorOptions& options,
                                        const StateDict& state_dict,
                                        const torch::Tensor& positions,
                                        const torch::Tensor& hidden_states,
                                        KVCache& kv_cache,
                                        bool enable_full_weight_path,
                                        bool enable_fused_mla_kernel) {
  ParallelArgs effective_parallel_args = parallel_args;
  effective_parallel_args.cp_size() =
      enable_full_weight_path ? parallel_args.world_size() : 1;
  OptimizationConfig optimization_config;
  optimization_config.enable_fused_mla_kernel = enable_fused_mla_kernel;
  optimization_config.enable_fused_indexer_qk = false;
  DeepseekV2Attention attention(
      args, quant_args, effective_parallel_args, options, optimization_config);
  attention->load_state_dict(state_dict);
  AttentionMetadata metadata = create_decode_metadata(options);
  return attention->forward(positions, hidden_states, metadata, kv_cache);
}

AttentionRunResult run_attention_prefill_once(
    const ModelArgs& args,
    const QuantArgs& quant_args,
    const ParallelArgs& parallel_args,
    const torch::TensorOptions& options,
    const StateDict& state_dict,
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const torch::Tensor& hidden_states,
    KVCache& kv_cache,
    bool enable_full_weight_path,
    bool enable_fused_mla_kernel,
    BatchForwardType batch_forward_type = BatchForwardType::PREFILL,
    int32_t prefix_len = 0,
    bool build_cp_context = true) {
  ParallelArgs effective_parallel_args = parallel_args;
  effective_parallel_args.cp_size() =
      enable_full_weight_path ? parallel_args.world_size() : 1;
  OptimizationConfig optimization_config;
  optimization_config.enable_fused_mla_kernel = enable_fused_mla_kernel;
  optimization_config.enable_fused_indexer_qk = false;
  DeepseekV2Attention attention(
      args, quant_args, effective_parallel_args, options, optimization_config);
  attention->load_state_dict(state_dict);
  const int32_t token_num = static_cast<int32_t>(tokens.numel());
  AttentionMetadata metadata =
      batch_forward_type.is_chunked_prefill()
          ? create_chunked_metadata(options, prefix_len, token_num)
          : create_prefill_metadata(options, token_num);
  std::optional<layer::v32_cp::DeepseekV32CPContext> sp_ctx;
  torch::Tensor local_positions = positions;
  torch::Tensor local_hidden_states = hidden_states;
  if (build_cp_context && enable_full_weight_path && args.index_n_heads() > 0) {
    ProcessGroup* cp_group = parallel_args.cp_group_ != nullptr
                                 ? parallel_args.cp_group_
                                 : parallel_args.process_group_;
    sp_ctx = layer::v32_cp::build_deepseek_v32_cp_context(
        effective_parallel_args.cp_size(),
        metadata,
        batch_forward_type,
        tokens,
        cp_group,
        parallel_args.rank(),
        parallel_args.world_size());
    if (sp_ctx.has_value()) {
      local_positions =
          layer::v32_cp::reorder_to_local_shard(positions, sp_ctx.value());
      local_hidden_states =
          layer::v32_cp::reorder_to_local_shard(hidden_states, sp_ctx.value());
    }
  }
  AttentionRunResult result;
  result.local_output = attention->forward(local_positions,
                                           local_hidden_states,
                                           metadata,
                                           kv_cache,
                                           sp_ctx ? &sp_ctx.value() : nullptr);
  result.global_output = result.local_output;
  result.used_sp = sp_ctx.has_value();
  result.local_token_num =
      sp_ctx.has_value() ? sp_ctx->comm_plan.tokens_per_rank.at(sp_ctx->rank)
                         : result.local_output.size(0);
  if (sp_ctx.has_value()) {
    result.global_output = layer::v32_cp::restore_gathered_to_global_order(
        layer::v32_cp::all_gather_across_ranks(result.local_output,
                                               sp_ctx.value()),
        sp_ctx.value());
  }
  result.k_cache = kv_cache.get_k_cache().clone();
  if (kv_cache.get_index_cache().defined()) {
    result.index_cache = kv_cache.get_index_cache().clone();
  }
  return result;
}

int32_t run_attention_test_child(int32_t rank,
                                 int32_t world_size,
                                 int32_t port,
                                 const std::string& host,
                                 bool enable_fused_mla_kernel,
                                 int64_t q_lora_rank) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [enable_fused_mla_kernel,
       q_lora_rank](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args = create_attention_model_args(q_lora_rank);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        torch::Tensor hidden_states =
            seeded_tensor("attention_multi_device/hidden_states",
                          {1, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);

        torch::Tensor full_weight_output =
            run_attention_decode_once(model_args,
                                      quant_args,
                                      parallel_args,
                                      options,
                                      state_dict,
                                      positions,
                                      hidden_states,
                                      full_weight_kv_cache,
                                      true,
                                      enable_fused_mla_kernel);

        xllm_device.synchronize_default_stream();
        CHECK_EQ(full_weight_output.size(0), 1)
            << "DeepseekV2Attention decode replicated token count mismatch";
        CHECK_EQ(full_weight_output.size(1), model_args.hidden_size())
            << "DeepseekV2Attention decode replicated hidden size mismatch";
        check_tensor_same_across_ranks(full_weight_output,
                                       parallel_args.tp_group_,
                                       "DeepseekV2Attention decode replicated");
        return 0;
      });
}

int32_t run_attention_prefill_repl_test_child(int32_t rank,
                                              int32_t world_size,
                                              int32_t port,
                                              const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args = create_attention_model_args();
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t seq_len = 4;
        torch::Tensor tokens =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor hidden_states =
            seeded_tensor("attention_multi_device/prefill_hidden_states",
                          {seq_len, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor positions =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        full_weight_kv_cache.get_k_cache().zero_();

        auto repl_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       tokens,
                                       positions,
                                       hidden_states,
                                       full_weight_kv_cache,
                                       true,
                                       /*enable_fused_mla_kernel=*/
                                       false);

        xllm_device.synchronize_default_stream();
        check_repl_output_contract(repl_result,
                                   seq_len,
                                   model_args.hidden_size(),
                                   "DeepseekV2Attention prefill replicated");
        check_tensor_same_across_ranks(
            repl_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV2Attention prefill replicated output");
        return 0;
      });
}

int32_t run_attention_prefill_fallback_test_child(int32_t rank,
                                                  int32_t world_size,
                                                  int32_t port,
                                                  const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t seq_len = 1;
        torch::Tensor tokens =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/prefill_hidden_states_fallback",
            {seq_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        full_weight_kv_cache.get_k_cache().zero_();

        auto repl_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       tokens,
                                       positions,
                                       hidden_states,
                                       full_weight_kv_cache,
                                       true,
                                       /*enable_fused_mla_kernel=*/
                                       false);

        xllm_device.synchronize_default_stream();
        check_repl_output_contract(repl_result,
                                   seq_len,
                                   model_args.hidden_size(),
                                   "DeepseekV2Attention prefill fallback");
        check_tensor_same_across_ranks(
            repl_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV2Attention prefill fallback output");
        return 0;
      });
}

int32_t run_attention_prefill_sp_test_child(int32_t rank,
                                            int32_t world_size,
                                            int32_t port,
                                            const std::string& host,
                                            bool use_glm5_args = false) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [use_glm5_args, world_size](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            use_glm5_args ? create_glm5_attention_model_args()
                          : create_attention_model_args(/*q_lora_rank=*/64,
                                                        /*enable_indexer=*/
                                                        true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t seq_len = 4;
        torch::Tensor tokens =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor hidden_states = seeded_tensor(
            use_glm5_args ? "attention_multi_device/glm5_prefill_hidden_states"
                          : "attention_multi_device/sp_prefill_hidden_states",
            {seq_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        full_weight_kv_cache.get_k_cache().zero_();

        auto sp_result = run_attention_prefill_once(model_args,
                                                    quant_args,
                                                    parallel_args,
                                                    options,
                                                    state_dict,
                                                    tokens,
                                                    positions,
                                                    hidden_states,
                                                    full_weight_kv_cache,
                                                    true,
                                                    /*enable_fused_mla_kernel=*/
                                                    false);

        xllm_device.synchronize_default_stream();
        check_sp_output_contract(sp_result,
                                 seq_len,
                                 model_args.hidden_size(),
                                 /*local_token_num=*/seq_len / world_size,
                                 use_glm5_args
                                     ? "DeepseekV2Attention glm5 prefill sp"
                                     : "DeepseekV2Attention prefill sp");
        check_tensor_same_across_ranks(
            sp_result.global_output,
            parallel_args.tp_group_,
            use_glm5_args ? "DeepseekV2Attention glm5 prefill sp output"
                          : "DeepseekV2Attention prefill sp output");
        return 0;
      });
}

int32_t run_attention_prefill_sp_baseline_test_child(int32_t rank,
                                                     int32_t world_size,
                                                     int32_t port,
                                                     const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [world_size](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t seq_len = 4;
        torch::Tensor tokens =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/sp_prefill_baseline_hidden_states",
            {seq_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::arange(seq_len, options.dtype(torch::kInt32).device(device));

        KVCache baseline_kv_cache = create_decode_kv_cache(model_args, options);
        baseline_kv_cache.get_k_cache().zero_();
        KVCache sp_kv_cache = create_decode_kv_cache(model_args, options);
        sp_kv_cache.get_k_cache().zero_();

        auto baseline_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       tokens,
                                       positions,
                                       hidden_states,
                                       baseline_kv_cache,
                                       /*enable_full_weight_path=*/true,
                                       /*enable_fused_mla_kernel=*/false,
                                       BatchForwardType::PREFILL,
                                       /*prefix_len=*/0,
                                       /*build_cp_context=*/false);
        auto sp_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       tokens,
                                       positions,
                                       hidden_states,
                                       sp_kv_cache,
                                       /*enable_full_weight_path=*/true,
                                       /*enable_fused_mla_kernel=*/
                                       false);

        xllm_device.synchronize_default_stream();
        check_repl_output_contract(baseline_result,
                                   seq_len,
                                   model_args.hidden_size(),
                                   "DeepseekV2Attention prefill baseline");
        check_sp_output_contract(sp_result,
                                 seq_len,
                                 model_args.hidden_size(),
                                 /*local_token_num=*/seq_len / world_size,
                                 "DeepseekV2Attention prefill sp baseline");
        check_tensors_close(sp_result.global_output,
                            baseline_result.global_output,
                            /*rtol=*/1e-3,
                            /*atol=*/1e-2,
                            "DeepseekV2Attention prefill sp output");
        return 0;
      });
}

int32_t run_attention_chunked_test_child(int32_t rank,
                                         int32_t world_size,
                                         int32_t port,
                                         const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/false,
      [world_size](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        constexpr int32_t prefix_len = 8;
        constexpr int32_t chunk1_len = 4;
        constexpr int32_t chunk2_len = 4;
        torch::Tensor prefix_tokens = torch::arange(
            prefix_len, options.dtype(torch::kInt32).device(device));
        torch::Tensor prefix_hidden_states =
            seeded_tensor("attention_multi_device/chunked_prefix_hidden_states",
                          {prefix_len, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor prefix_positions = torch::arange(
            prefix_len, options.dtype(torch::kInt32).device(device));

        torch::Tensor chunk1_tokens =
            torch::arange(prefix_len,
                          prefix_len + chunk1_len,
                          options.dtype(torch::kInt32).device(device));
        torch::Tensor chunk1_hidden_states = seeded_tensor(
            "attention_multi_device/chunked_suffix1_hidden_states",
            {chunk1_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor chunk1_positions =
            torch::arange(prefix_len,
                          prefix_len + chunk1_len,
                          options.dtype(torch::kInt32).device(device));

        torch::Tensor chunk2_tokens =
            torch::arange(prefix_len + chunk1_len,
                          prefix_len + chunk1_len + chunk2_len,
                          options.dtype(torch::kInt32).device(device));
        torch::Tensor chunk2_hidden_states = seeded_tensor(
            "attention_multi_device/chunked_suffix2_hidden_states",
            {chunk2_len, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor chunk2_positions =
            torch::arange(prefix_len + chunk1_len,
                          prefix_len + chunk1_len + chunk2_len,
                          options.dtype(torch::kInt32).device(device));

        KVCache full_weight_kv_cache =
            create_decode_kv_cache(model_args, options);
        full_weight_kv_cache.get_k_cache().zero_();

        auto sp_prefix_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       prefix_tokens,
                                       prefix_positions,
                                       prefix_hidden_states,
                                       full_weight_kv_cache,
                                       true,
                                       /*enable_fused_mla_kernel=*/
                                       false);
        check_sp_output_contract(sp_prefix_result,
                                 prefix_len,
                                 model_args.hidden_size(),
                                 /*local_token_num=*/prefix_len / world_size,
                                 "DeepseekV2Attention chunked prefix sp");
        check_tensor_same_across_ranks(
            sp_prefix_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV2Attention chunked prefix output");

        auto sp_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       chunk1_tokens,
                                       chunk1_positions,
                                       chunk1_hidden_states,
                                       full_weight_kv_cache,
                                       true,
                                       /*enable_fused_mla_kernel=*/
                                       false,
                                       BatchForwardType::CHUNKED_PREFILL,
                                       prefix_len);

        xllm_device.synchronize_default_stream();
        check_sp_output_contract(sp_result,
                                 chunk1_len,
                                 model_args.hidden_size(),
                                 /*local_token_num=*/chunk1_len / world_size,
                                 "DeepseekV2Attention chunked prefill sp");
        check_tensor_same_across_ranks(
            sp_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV2Attention chunked prefill output");

        auto sp_second_result =
            run_attention_prefill_once(model_args,
                                       quant_args,
                                       parallel_args,
                                       options,
                                       state_dict,
                                       chunk2_tokens,
                                       chunk2_positions,
                                       chunk2_hidden_states,
                                       full_weight_kv_cache,
                                       true,
                                       /*enable_fused_mla_kernel=*/false,
                                       BatchForwardType::CHUNKED_PREFILL,
                                       prefix_len + chunk1_len);

        xllm_device.synchronize_default_stream();
        check_sp_output_contract(
            sp_second_result,
            chunk2_len,
            model_args.hidden_size(),
            /*local_token_num=*/chunk2_len / world_size,
            "DeepseekV2Attention second chunked prefill sp");
        check_tensor_same_across_ranks(
            sp_second_result.global_output,
            parallel_args.tp_group_,
            "DeepseekV2Attention second chunked prefill output");
        return 0;
      });
}

int32_t run_attention_topk_share_test_child(int32_t rank,
                                            int32_t world_size,
                                            int32_t port,
                                            const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        // Mirror the production Full/Shared split. The Shared layer owns no
        // Indexer, so its forward can only succeed by consuming the Full
        // layer's exported sparse metadata.
        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV2Attention full_attention(model_args,
                                           quant_args,
                                           parallel_args,
                                           options,
                                           optimization_config,
                                           /*enable_indexer=*/true);
        DeepseekV2Attention shared_attention(model_args,
                                             quant_args,
                                             parallel_args,
                                             options,
                                             optimization_config,
                                             /*enable_indexer=*/false);
        full_attention->load_state_dict(state_dict);
        shared_attention->load_state_dict(state_dict);
        CHECK(full_attention->named_children().contains("indexer"))
            << "Full layer must own an Indexer";
        CHECK(!shared_attention->named_children().contains("indexer"))
            << "Shared layer must not own an Indexer";

        torch::Tensor hidden_states =
            seeded_tensor("attention_multi_device/topk_share_hidden_states",
                          {1, model_args.hidden_size()},
                          torch::kBFloat16,
                          device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        // Use independent, identically initialized caches so the Shared result
        // is compared with a clean Full-layer recomputation rather than with
        // state mutated by the baseline forward.
        KVCache full_kv_cache = create_decode_kv_cache(model_args, options);
        KVCache shared_kv_cache = create_decode_kv_cache(model_args, options);
        full_kv_cache.get_k_cache().zero_();
        shared_kv_cache.get_k_cache().zero_();

        // Full-layer baseline: independently compute the sparse block table and
        // the attention output.
        AttentionMetadata full_metadata = create_decode_metadata(options);
        DsaTopkTransfer full_transfer = DsaTopkTransfer::capture_output();
        torch::Tensor full_out = full_attention->forward(positions,
                                                         hidden_states,
                                                         full_metadata,
                                                         full_kv_cache,
                                                         /*sp_ctx=*/nullptr,
                                                         &full_transfer);
        xllm_device.synchronize_default_stream();

        const DsaTopkState* full_topk = full_transfer.output();
        CHECK(full_topk != nullptr)
            << "Output layer must publish DSA top-k state";
        const torch::Tensor& full_block_tables = full_topk->block_tables();
        const torch::Tensor& full_context_lens = full_topk->context_lens();
        CHECK(full_block_tables.defined())
            << "Output layer must export a sparse block table";
        CHECK_EQ(full_block_tables.dim(), 2)
            << "sparse block table must be 2D {num_tokens, index_topk}";
        CHECK_EQ(full_block_tables.size(0), 1) << "num_tokens mismatch";
        CHECK_EQ(full_block_tables.size(1), model_args.index_topk())
            << "index_topk mismatch";
        CHECK(full_block_tables.scalar_type() == torch::kInt32)
            << "sparse block table must be int32";
        CHECK(torch::isfinite(full_out.to(torch::kFloat32)).all().item<bool>())
            << "Full layer attention output must be finite";

        // Shared-layer pass: use the production reuse-only relay. Pointer
        // checks below verify the relay does not copy metadata; numerical
        // equivalence with the independently recomputed Full result verifies
        // the no-Indexer Shared path produces the expected attention result.
        AttentionMetadata shared_metadata = create_decode_metadata(options);
        DsaTopkTransfer shared_transfer = DsaTopkTransfer::reuse(*full_topk);
        const DsaTopkState* shared_input = shared_transfer.input();
        CHECK(shared_input != nullptr) << "Shared layer must receive DSA top-k";
        CHECK_EQ(shared_input->block_tables().data_ptr(),
                 full_block_tables.data_ptr())
            << "Shared relay must reuse the Full layer's sparse block table";
        CHECK_EQ(shared_input->context_lens().data_ptr(),
                 full_context_lens.data_ptr())
            << "Shared relay must reuse the Full layer's context lens";
        torch::Tensor shared_out = shared_attention->forward(positions,
                                                             hidden_states,
                                                             shared_metadata,
                                                             shared_kv_cache,
                                                             /*sp_ctx=*/nullptr,
                                                             &shared_transfer);
        xllm_device.synchronize_default_stream();

        CHECK(shared_transfer.output() == nullptr)
            << "A production Shared relay must not self-echo its input";
        check_tensors_close(shared_out,
                            full_out,
                            /*rtol=*/1e-3,
                            /*atol=*/1e-2,
                            "Shared attention vs Full indexer baseline");
        return 0;
      });
}

// Dummy runs (DP sync / MTP step_empty / graph warmup) skip the indexer in
// attention, so an Output layer cannot export a valid sparse block table. This
// pins that contract: a dummy Output-layer forward must not produce a sparse
// {num_tokens, index_topk} table. It is the precondition for the decoder
// bypassing cross-layer sharing on dummy runs, without which a following Shared
// layer would assert on a missing previous top-k.
int32_t run_attention_topk_share_dummy_run_test_child(int32_t rank,
                                                      int32_t world_size,
                                                      int32_t port,
                                                      const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV2Attention attention(model_args,
                                      quant_args,
                                      parallel_args,
                                      options,
                                      optimization_config);
        attention->load_state_dict(state_dict);

        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/topk_share_dummy_hidden_states",
            {1, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::tensor({0}, options.dtype(torch::kInt32).device(device));

        KVCache kv_cache = create_decode_kv_cache(model_args, options);
        kv_cache.get_k_cache().zero_();

        // Output-layer pass on a dummy run: the indexer is skipped, so no fresh
        // sparse block table is computed.
        AttentionMetadata dummy_metadata = create_decode_metadata(options);
        dummy_metadata.is_dummy = true;
        DsaTopkTransfer output_transfer = DsaTopkTransfer::capture_output();
        attention->forward(positions,
                           hidden_states,
                           dummy_metadata,
                           kv_cache,
                           /*sp_ctx=*/nullptr,
                           &output_transfer);
        xllm_device.synchronize_default_stream();

        // A dummy run must not yield a usable sparse top-k table (width
        // index_topk), so a Shared layer cannot legally reuse it. The decoder
        // therefore has to disable cross-layer sharing whenever
        // attn_metadata.is_dummy is set.
        const DsaTopkState* dummy_topk = output_transfer.output();
        const bool produced_sparse_topk =
            dummy_topk != nullptr && dummy_topk->block_tables().dim() == 2 &&
            dummy_topk->block_tables().size(1) == model_args.index_topk();
        CHECK(!produced_sparse_topk) << "Dummy Output-layer run must not "
                                        "export a sparse top-k block table";
        return 0;
      });
}

// Same Full->Shared numerical and relay contract as above, but on a multi-token
// decode bucket (batch_size > 1). This also pins the graph invariant that the
// reused sparse block table's shape strictly equals the captured num_tokens
// bucket, since every layer in one forward processes the same bucketed count.
int32_t run_attention_topk_share_bucket_test_child(int32_t rank,
                                                   int32_t world_size,
                                                   int32_t port,
                                                   const std::string& host) {
  return run_distributed_attention_child(
      rank,
      world_size,
      port,
      host,
      /*enable_context_parallel=*/true,
      [](DistributedAttentionChildContext& context) {
        xllm::Device& xllm_device = context.xllm_device();
        const torch::Device& device = context.device();
        ParallelArgs& parallel_args = context.parallel_args();
        const torch::TensorOptions& options = context.options();
        ModelArgs model_args =
            create_attention_model_args(/*q_lora_rank=*/64,
                                        /*enable_indexer=*/true);
        QuantArgs quant_args = create_default_quant_args();
        StateDict state_dict = create_attention_state_dict(model_args, options);

        OptimizationConfig optimization_config;
        optimization_config.enable_fused_mla_kernel = false;
        optimization_config.enable_fused_indexer_qk = false;
        DeepseekV2Attention full_attention(model_args,
                                           quant_args,
                                           parallel_args,
                                           options,
                                           optimization_config,
                                           /*enable_indexer=*/true);
        DeepseekV2Attention shared_attention(model_args,
                                             quant_args,
                                             parallel_args,
                                             options,
                                             optimization_config,
                                             /*enable_indexer=*/false);
        full_attention->load_state_dict(state_dict);
        shared_attention->load_state_dict(state_dict);
        CHECK(full_attention->named_children().contains("indexer"))
            << "Full layer must own an Indexer";
        CHECK(!shared_attention->named_children().contains("indexer"))
            << "Shared layer must not own an Indexer";

        constexpr int32_t kBucketTokens = 4;
        torch::Tensor hidden_states = seeded_tensor(
            "attention_multi_device/topk_share_bucket_hidden_states",
            {kBucketTokens, model_args.hidden_size()},
            torch::kBFloat16,
            device);
        torch::Tensor positions =
            torch::zeros({kBucketTokens}, options.dtype(torch::kInt32));

        KVCache full_kv_cache = create_decode_kv_cache(model_args, options);
        KVCache shared_kv_cache = create_decode_kv_cache(model_args, options);
        full_kv_cache.get_k_cache().zero_();
        shared_kv_cache.get_k_cache().zero_();

        AttentionMetadata full_metadata =
            create_batch_decode_metadata(options, kBucketTokens);
        DsaTopkTransfer full_transfer = DsaTopkTransfer::capture_output();
        torch::Tensor full_out = full_attention->forward(positions,
                                                         hidden_states,
                                                         full_metadata,
                                                         full_kv_cache,
                                                         /*sp_ctx=*/nullptr,
                                                         &full_transfer);
        xllm_device.synchronize_default_stream();

        const DsaTopkState* full_topk = full_transfer.output();
        CHECK(full_topk != nullptr)
            << "Output layer must publish DSA top-k state";
        const torch::Tensor& full_block_tables = full_topk->block_tables();
        CHECK(full_block_tables.defined())
            << "Output layer must export a sparse block table";
        CHECK_EQ(full_block_tables.size(0), kBucketTokens)
            << "sparse block table row count must equal the decode bucket";
        CHECK_EQ(full_block_tables.size(1), model_args.index_topk())
            << "index_topk mismatch";

        AttentionMetadata shared_metadata =
            create_batch_decode_metadata(options, kBucketTokens);
        DsaTopkTransfer shared_transfer = DsaTopkTransfer::reuse(*full_topk);
        const DsaTopkState* shared_input = shared_transfer.input();
        CHECK(shared_input != nullptr) << "Shared layer must receive DSA top-k";
        CHECK_EQ(shared_input->block_tables().sizes(),
                 full_block_tables.sizes())
            << "Shared layer reuse shape must strictly equal the captured "
               "bucket";
        CHECK_EQ(shared_input->block_tables().data_ptr(),
                 full_block_tables.data_ptr())
            << "Shared relay must reuse the Full layer's sparse block table";
        torch::Tensor shared_out = shared_attention->forward(positions,
                                                             hidden_states,
                                                             shared_metadata,
                                                             shared_kv_cache,
                                                             /*sp_ctx=*/nullptr,
                                                             &shared_transfer);
        xllm_device.synchronize_default_stream();

        CHECK(shared_transfer.output() == nullptr)
            << "A production Shared relay must not self-echo its input";
        check_tensors_close(
            shared_out,
            full_out,
            /*rtol=*/1e-3,
            /*atol=*/1e-2,
            "Bucketed Shared attention vs Full indexer baseline");
        return 0;
      });
}

}  // namespace

TEST(AttentionMultiDeviceProcessTest, TimedOutChildIsTerminatedAndReaped) {
  const pid_t child_pid = ::fork();
  ASSERT_GE(child_pid, 0) << "Failed to fork timeout test child";
  if (child_pid == 0) {
    while (true) {
      ::pause();
    }
  }

  const std::vector<pid_t> child_pids{child_pid};
  const ChildWaitResult result =
      wait_for_children(child_pids, std::chrono::milliseconds{50});
  EXPECT_TRUE(result.any_failed);
  EXPECT_TRUE(result.timed_out);

  int status = 0;
  errno = 0;
  EXPECT_EQ(wait_for_pid(child_pid, &status, WNOHANG), -1);
  EXPECT_EQ(errno, ECHILD) << "Timed-out child was not fully reaped";
}

class AttentionMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    host_ = "127.0.0.1";
  }

  template <typename ChildFn>
  void run_child_test(const ChildFn& child_fn,
                      const std::string& failure_message) {
    const int32_t port = net::get_local_free_port();
    ASSERT_GT(port, 0) << "Failed to allocate a TCP rendezvous port";
    std::vector<pid_t> child_pids;
    child_pids.reserve(world_size_);
    for (int32_t rank = 0; rank < world_size_; ++rank) {
      const pid_t pid = ::fork();
      if (pid == 0) {
        const int32_t exit_code = child_fn(rank, world_size_, port, host_);
        _exit(exit_code);
      }
      if (pid > 0) {
        child_pids.emplace_back(pid);
        continue;
      }

      LOG(ERROR) << "Failed to fork rank " << rank << ": "
                 << std::strerror(errno);
      std::vector<bool> finished(child_pids.size(), false);
      terminate_and_reap_children(child_pids, finished);
      ADD_FAILURE() << "Failed to fork all multi-device test ranks";
      return;
    }

    const ChildWaitResult result = wait_for_children(child_pids);
    if (result.any_skipped && !result.any_failed) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    }
    ASSERT_FALSE(result.any_failed)
        << failure_message
        << (result.timed_out ? " Child process timeout." : "");
  }

  void run_test(bool enable_fused_mla_kernel, int64_t q_lora_rank = 0) {
    run_child_test(
        [enable_fused_mla_kernel, q_lora_rank](int32_t rank,
                                               int32_t world_size,
                                               int32_t port,
                                               const std::string& host) {
          return run_attention_test_child(rank,
                                          world_size,
                                          port,
                                          host,
                                          enable_fused_mla_kernel,
                                          q_lora_rank);
        },
        "Attention multi-device equivalence test failed.");
  }

  void run_prefill_repl_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_prefill_repl_test_child(
              rank, world_size, port, host);
        },
        "Attention multi-device replicated prefill test failed.");
  }

  void run_prefill_fallback_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_prefill_fallback_test_child(
              rank, world_size, port, host);
        },
        "Attention multi-device prefill fallback test failed.");
  }

  void run_prefill_sp_test(bool use_glm5_args = false) {
    run_child_test(
        [use_glm5_args](int32_t rank,
                        int32_t world_size,
                        int32_t port,
                        const std::string& host) {
          return run_attention_prefill_sp_test_child(
              rank, world_size, port, host, use_glm5_args);
        },
        "Attention multi-device SP prefill test failed.");
  }

  void run_chunked_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_chunked_test_child(rank, world_size, port, host);
        },
        "Attention multi-device chunked test failed.");
  }

  void run_prefill_sp_baseline_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_prefill_sp_baseline_test_child(
              rank, world_size, port, host);
        },
        "Attention multi-device SP prefill baseline test failed.");
  }

  void run_topk_share_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_topk_share_test_child(
              rank, world_size, port, host);
        },
        "Attention cross-layer top-k share test failed.");
  }

  void run_topk_share_dummy_run_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_topk_share_dummy_run_test_child(
              rank, world_size, port, host);
        },
        "Attention dummy-run top-k share test failed.");
  }

  void run_topk_share_bucket_test() {
    run_child_test(
        [](int32_t rank,
           int32_t world_size,
           int32_t port,
           const std::string& host) {
          return run_attention_topk_share_bucket_test_child(
              rank, world_size, port, host);
        },
        "Attention cross-layer top-k share bucket test failed.");
  }

  int32_t world_size_;
  std::string host_;
};

TEST_F(AttentionMultiDeviceTest, DecodeReplicatedOutputMatchesTpBaseline) {
  run_test(/*enable_fused_mla_kernel=*/false);
}

TEST_F(AttentionMultiDeviceTest,
       DecodeReplicatedOutputMatchesTpBaselineFusedMLA) {
  run_test(/*enable_fused_mla_kernel=*/true, /*q_lora_rank=*/64);
}

TEST_F(AttentionMultiDeviceTest,
       PrefillReplicatedOutputMatchesTpBaselineWithoutIndexer) {
  run_prefill_repl_test();
}

TEST_F(AttentionMultiDeviceTest,
       PrefillSpLocalPackedOutputRestoresToTpBaseline) {
  run_prefill_sp_test();
}

TEST_F(AttentionMultiDeviceTest, PrefillShortSeqFallsBackToReplicatedPath) {
  run_prefill_fallback_test();
}

TEST_F(AttentionMultiDeviceTest, Glm5PrefillSpRestoresToTpBaseline) {
  run_prefill_sp_test(/*use_glm5_args=*/true);
}

TEST_F(AttentionMultiDeviceTest,
       ChunkedPrefillSpAcrossChunksMatchesTpBaseline) {
  run_chunked_test();
}

TEST_F(AttentionMultiDeviceTest, PrefillSpNonChunkedMatchesReplicatedBaseline) {
  run_prefill_sp_baseline_test();
}

TEST_F(AttentionMultiDeviceTest, DummyRunOutputLayerYieldsNoSparseTopk) {
  run_topk_share_dummy_run_test();
}

TEST_F(AttentionMultiDeviceTest, SharedLayerReusesFullLayerSparseBlockTable) {
  run_topk_share_test();
}

TEST_F(AttentionMultiDeviceTest, SharedLayerReuseShapeMatchesDecodeBucket) {
  run_topk_share_bucket_test();
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
