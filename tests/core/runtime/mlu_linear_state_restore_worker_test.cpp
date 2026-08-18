/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include <c10/core/DeviceGuard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <optional>
#include <utility>
#include <vector>

#include "framework/kv_cache/kv_cache.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "runtime/llm_worker_impl.h"
#include "runtime/options.h"
#include "runtime/worker_impl.h"

namespace xllm {
namespace {

constexpr int64_t kNumSlots = 7;
constexpr int32_t kRestoreSourceSlot = 4;

class LinearStateRestoreWorker final : public WorkerImpl {
 public:
  LinearStateRestoreWorker(const ParallelArgs& parallel_args,
                           const torch::Device& device,
                           const runtime::Options& options,
                           const ModelArgs& model_args)
      : WorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kBFloat16;
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
  }

  bool init_model(ModelContext& /*context*/) override { return true; }

  std::optional<ForwardOutput> step(const ForwardInput& /*input*/) override {
    return std::nullopt;
  }

  void set_kv_caches(std::vector<KVCache> kv_caches) {
    kv_caches_ = std::move(kv_caches);
  }
};

struct LinearCacheTensors {
  torch::Tensor conv;
  torch::Tensor ssm;
};

LinearCacheTensors make_linear_cache(const torch::Device& device,
                                     int64_t ssm_stride,
                                     float sentinel) {
  const torch::TensorOptions conv_options =
      torch::TensorOptions().dtype(torch::kBFloat16).device(device);
  const torch::TensorOptions ssm_options =
      torch::TensorOptions().dtype(torch::kFloat32).device(device);
  return {
      torch::full({kNumSlots, 3, 8}, sentinel, conv_options),
      torch::full({kNumSlots * ssm_stride, 2, 2, 2}, sentinel, ssm_options)};
}

void fill_slot(LinearCacheTensors& cache,
               int64_t slot,
               int64_t ssm_stride,
               float conv_value,
               float ssm_value) {
  cache.conv.select(/*dim=*/0, slot).fill_(conv_value);
  cache.ssm.narrow(/*dim=*/0, slot * ssm_stride, ssm_stride).fill_(ssm_value);
}

void expect_slot_matches(const LinearCacheTensors& cache,
                         int64_t destination,
                         int64_t source,
                         int64_t ssm_stride) {
  EXPECT_TRUE(torch::equal(cache.conv.select(/*dim=*/0, destination),
                           cache.conv.select(/*dim=*/0, source)));
  EXPECT_TRUE(torch::equal(
      cache.ssm.narrow(/*dim=*/0, destination * ssm_stride, ssm_stride),
      cache.ssm.narrow(/*dim=*/0, source * ssm_stride, ssm_stride)));
}

void expect_slot_filled(const LinearCacheTensors& cache,
                        int64_t slot,
                        int64_t ssm_stride,
                        float value) {
  EXPECT_TRUE(
      torch::all(cache.conv.select(/*dim=*/0, slot) == value).item<bool>());
  EXPECT_TRUE(torch::all(cache.ssm.narrow(
                             /*dim=*/0, slot * ssm_stride, ssm_stride) == value)
                  .item<bool>());
}

void expect_slot_equals(const LinearCacheTensors& actual,
                        const LinearCacheTensors& expected,
                        int64_t slot,
                        int64_t ssm_stride) {
  EXPECT_TRUE(torch::equal(actual.conv.select(/*dim=*/0, slot),
                           expected.conv.select(/*dim=*/0, slot)));
  EXPECT_TRUE(torch::equal(
      actual.ssm.narrow(/*dim=*/0, slot * ssm_stride, ssm_stride),
      expected.ssm.narrow(/*dim=*/0, slot * ssm_stride, ssm_stride)));
}

class OverlapLinearStateRestoreWorker final : public LLMWorkerImpl {
 public:
  OverlapLinearStateRestoreWorker(const ParallelArgs& parallel_args,
                                  const torch::Device& device,
                                  const runtime::Options& options,
                                  const ModelArgs& model_args)
      : LLMWorkerImpl(parallel_args, device, options) {
    dtype_ = torch::kBFloat16;
    context_ =
        ModelContext(parallel_args,
                     model_args,
                     QuantArgs(),
                     torch::TensorOptions().dtype(dtype_).device(device));
  }

  void set_kv_caches(std::vector<KVCache> kv_caches) {
    kv_caches_ = std::move(kv_caches);
  }

  void enqueue_previous_chunk_write(
      int64_t source_slot,
      const std::vector<std::pair<float, float>>& layer_values) {
    c10::StreamGuard stream_guard = compute_stream_->set_stream_guard();
    for (size_t layer = 0; layer < layer_values.size(); ++layer) {
      torch::Tensor conv_cache = kv_caches_[layer].get_conv_cache();
      torch::Tensor ssm_cache = kv_caches_[layer].get_ssm_cache();
      const int64_t ssm_stride = ssm_cache.size(0) / conv_cache.size(0);
      conv_cache.select(/*dim=*/0, source_slot)
          .fill_(layer_values[layer].first);
      ssm_cache.narrow(/*dim=*/0, source_slot * ssm_stride, ssm_stride)
          .fill_(layer_values[layer].second);
    }
  }

  std::optional<ForwardOutput> run_overlap_forward(const ForwardInput& input,
                                                   int64_t destination_slot) {
    forward_destination_slot_ = destination_slot;
    return step_for_schedule_overlap(input);
  }

  std::optional<ForwardOutput> execute_no_sync_on_stream(
      const ForwardInput& /*input*/,
      Stream& compute_stream,
      bool /*record_ready_event*/) override {
    c10::StreamGuard stream_guard = compute_stream.set_stream_guard();
    for (KVCache& kv_cache : kv_caches_) {
      torch::Tensor conv_cache = kv_cache.get_conv_cache();
      torch::Tensor ssm_cache = kv_cache.get_ssm_cache();
      if (!conv_cache.defined()) {
        continue;
      }
      const int64_t ssm_stride = ssm_cache.size(0) / conv_cache.size(0);
      conv_cache.select(/*dim=*/0, forward_destination_slot_).add_(2.0f);
      ssm_cache
          .narrow(/*dim=*/0, forward_destination_slot_ * ssm_stride, ssm_stride)
          .add_(4.0f);
    }
    return std::nullopt;
  }

  void synchronize_compute_stream() { compute_stream_->synchronize(); }

 private:
  int64_t forward_destination_slot_ = -1;
};

TEST(MluLinearStateRestoreWorkerTest,
     NonOverlapPreparePublishesRestoredMixedBatch) {
  const torch::Device device("mlu:0");
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  ModelArgs model_args;
  model_args.layer_types({"linear_attention", "linear_attention"});
  runtime::Options runtime_options;
  runtime_options.enable_schedule_overlap(false);
  LinearStateRestoreWorker worker(
      parallel_args, device, runtime_options, model_args);

  constexpr int64_t kFirstStride = 2;
  constexpr int64_t kSecondStride = 3;
  LinearCacheTensors first =
      make_linear_cache(device, kFirstStride, /*sentinel=*/-101.0f);
  LinearCacheTensors second =
      make_linear_cache(device, kSecondStride, /*sentinel=*/-202.0f);
  fill_slot(first,
            kRestoreSourceSlot,
            kFirstStride,
            /*conv_value=*/3.0f,
            /*ssm_value=*/5.0f);
  fill_slot(second,
            kRestoreSourceSlot,
            kSecondStride,
            /*conv_value=*/7.0f,
            /*ssm_value=*/11.0f);
  fill_slot(first,
            /*slot=*/2,
            kFirstStride,
            /*conv_value=*/13.0f,
            /*ssm_value=*/13.0f);
  fill_slot(second,
            /*slot=*/2,
            kSecondStride,
            /*conv_value=*/17.0f,
            /*ssm_value=*/17.0f);
  fill_slot(first,
            /*slot=*/3,
            kFirstStride,
            /*conv_value=*/19.0f,
            /*ssm_value=*/19.0f);
  fill_slot(second,
            /*slot=*/3,
            kSecondStride,
            /*conv_value=*/23.0f,
            /*ssm_value=*/23.0f);

  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{first.conv, first.ssm});
  kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{second.conv, second.ssm});
  worker.set_kv_caches(std::move(kv_caches));

  LinearStateCacheOp restore;
  restore.linear_state_id = 1;
  restore.restore_requested = true;
  restore.restore_src_slot_id = kRestoreSourceSlot;
  LinearStateCacheOp continued;
  continued.linear_state_id = 2;
  LinearStateCacheOp cold;
  cold.linear_state_id = 3;
  cold.reset_requested = true;
  LinearStateCacheOp second_cold;
  second_cold.linear_state_id = 5;
  second_cold.reset_requested = true;

  ForwardInput input;
  input.token_ids = torch::ones(
      {4}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  input.positions = torch::zeros_like(input.token_ids);
  input.device_tensors_ready = true;
  input.input_params.meta.num_sequences = 4;
  input.input_params.meta.batch_id = 42;
  input.input_params.attention.host.q_seq_lens = {0, 1, 2, 3, 4};
  input.input_params.attention.host.kv_cache_tokens_nums = {0, 8, 0, 0};
  input.input_params.linear_state_cache_ops = {
      restore, continued, cold, second_cold};

  ForwardInput processed_input;
  worker.prepare_work_before_execute(input, processed_input);
  ASSERT_NE(processed_input.metadata_ready_event, nullptr);
  Device xllm_device(device);
  std::unique_ptr<Stream> model_stream = xllm_device.current_stream();
  ASSERT_TRUE(model_stream->wait_event(processed_input.metadata_ready_event));
  ASSERT_EQ(model_stream->synchronize(), 0);

  EXPECT_EQ(processed_input.input_params.linear_state_validity_mask,
            std::vector<int64_t>({1, 1, 0, 0}));
  expect_slot_matches(first,
                      /*destination=*/1,
                      kRestoreSourceSlot,
                      kFirstStride);
  expect_slot_matches(second,
                      /*destination=*/1,
                      kRestoreSourceSlot,
                      kSecondStride);
  expect_slot_filled(first, /*slot=*/2, kFirstStride, /*value=*/13.0f);
  expect_slot_filled(second, /*slot=*/2, kSecondStride, /*value=*/17.0f);
  expect_slot_filled(first, /*slot=*/3, kFirstStride, /*value=*/0.0f);
  expect_slot_filled(second, /*slot=*/3, kSecondStride, /*value=*/0.0f);
  expect_slot_filled(first, /*slot=*/5, kFirstStride, /*value=*/0.0f);
  expect_slot_filled(second, /*slot=*/5, kSecondStride, /*value=*/0.0f);
}

TEST(MluLinearStateRestoreWorkerTest,
     NonOverlapPrepareExpandsLogicalRowsForActiveRows) {
  const torch::Device device("mlu:0");
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  ModelArgs model_args;
  model_args.layer_types({"linear_attention"});
  runtime::Options runtime_options;
  runtime_options.enable_schedule_overlap(false);
  LinearStateRestoreWorker worker(
      parallel_args, device, runtime_options, model_args);

  constexpr int64_t kSsmStride = 2;
  LinearCacheTensors cache =
      make_linear_cache(device, kSsmStride, /*sentinel=*/-101.0f);
  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{cache.conv, cache.ssm});
  worker.set_kv_caches(std::move(kv_caches));

  ForwardInput input;
  input.token_ids = torch::ones(
      {6}, torch::TensorOptions().dtype(torch::kInt32).device(device));
  input.positions = torch::zeros_like(input.token_ids);
  input.device_tensors_ready = true;
  input.input_params.meta.num_sequences = 2;
  input.input_params.attention.host.q_seq_lens = {0, 1, 2, 3, 4, 5, 6};
  input.input_params.attention.host.kv_cache_tokens_nums = {0, 8};
  input.input_params.linear_state_cache_ops.resize(/*count=*/6);
  for (int32_t row = 0; row < 6; ++row) {
    LinearStateCacheOp& cache_op =
        input.input_params.linear_state_cache_ops[static_cast<size_t>(row)];
    cache_op.linear_state_id = row + 1;
    cache_op.reset_requested = row < 3;
  }

  ForwardInput processed_input;
  worker.prepare_work_before_execute(input, processed_input);

  EXPECT_EQ(processed_input.input_params.linear_state_validity_mask,
            std::vector<int64_t>({0, 0, 0, 1, 1, 1}));
}

TEST(MluLinearStateRestoreWorkerTest,
     OverlapRestoresAfterPreviousWriteBeforeCurrentForward) {
  const torch::Device device("mlu:0");
  ParallelArgs parallel_args(
      /*rank=*/0, /*world_size=*/1, /*process_group=*/nullptr);
  ModelArgs model_args;
  model_args.layer_types({"linear_attention", "linear_attention"});
  runtime::Options runtime_options;
  runtime_options.enable_schedule_overlap(true);
  OverlapLinearStateRestoreWorker worker(
      parallel_args, device, runtime_options, model_args);

  constexpr int64_t kFirstStride = 2;
  constexpr int64_t kSecondStride = 3;
  LinearCacheTensors first =
      make_linear_cache(device, kFirstStride, /*sentinel=*/-101.0f);
  LinearCacheTensors second =
      make_linear_cache(device, kSecondStride, /*sentinel=*/-202.0f);
  LinearCacheTensors first_baseline =
      make_linear_cache(device, kFirstStride, /*sentinel=*/-101.0f);
  LinearCacheTensors second_baseline =
      make_linear_cache(device, kSecondStride, /*sentinel=*/-202.0f);
  fill_slot(first_baseline,
            /*slot=*/1,
            kFirstStride,
            /*conv_value=*/3.0f,
            /*ssm_value=*/5.0f);
  fill_slot(second_baseline,
            /*slot=*/1,
            kSecondStride,
            /*conv_value=*/7.0f,
            /*ssm_value=*/11.0f);
  first_baseline.conv.select(/*dim=*/0, /*index=*/1).add_(2.0f);
  first_baseline.ssm
      .narrow(/*dim=*/0, /*start=*/kFirstStride, /*length=*/kFirstStride)
      .add_(4.0f);
  second_baseline.conv.select(/*dim=*/0, /*index=*/1).add_(2.0f);
  second_baseline.ssm
      .narrow(/*dim=*/0, /*start=*/kSecondStride, /*length=*/kSecondStride)
      .add_(4.0f);
  fill_slot(first,
            /*slot=*/2,
            kFirstStride,
            /*conv_value=*/13.0f,
            /*ssm_value=*/17.0f);
  fill_slot(second,
            /*slot=*/2,
            kSecondStride,
            /*conv_value=*/19.0f,
            /*ssm_value=*/23.0f);

  std::vector<KVCache> kv_caches;
  kv_caches.emplace_back(LinearAttentionKVCacheTensors{first.conv, first.ssm});
  kv_caches.emplace_back(
      LinearAttentionKVCacheTensors{second.conv, second.ssm});
  worker.set_kv_caches(std::move(kv_caches));

  LinearStateCacheOp restore;
  restore.linear_state_id = 1;
  restore.restore_requested = true;
  restore.restore_src_slot_id = kRestoreSourceSlot;
  LinearStateCacheOp continued;
  continued.linear_state_id = 2;
  LinearStateCacheOp cold;
  cold.linear_state_id = 3;
  cold.reset_requested = true;
  LinearStateCacheOp second_cold;
  second_cold.linear_state_id = 5;
  second_cold.reset_requested = true;

  ForwardInput input;
  input.input_params.meta.batch_id = 43;
  input.input_params.linear_state_cache_ops = {
      restore, continued, cold, second_cold};
  input.input_params.linear_state_validity_mask = {0, 1, 0, 0};

  worker.enqueue_previous_chunk_write(kRestoreSourceSlot,
                                      {{3.0f, 5.0f}, {7.0f, 11.0f}});
  worker.run_overlap_forward(input, /*destination_slot=*/1);
  worker.synchronize_compute_stream();

  EXPECT_EQ(input.input_params.linear_state_validity_mask,
            std::vector<int64_t>({1, 1, 0, 0}));
  expect_slot_equals(first, first_baseline, /*slot=*/1, kFirstStride);
  expect_slot_equals(second, second_baseline, /*slot=*/1, kSecondStride);
  EXPECT_TRUE(torch::all(first.conv.select(/*dim=*/0, /*index=*/2) == 13.0f)
                  .item<bool>());
  EXPECT_TRUE(torch::all(first.ssm.narrow(/*dim=*/0,
                                          /*start=*/2 * kFirstStride,
                                          /*length=*/kFirstStride) == 17.0f)
                  .item<bool>());
  EXPECT_TRUE(torch::all(second.conv.select(/*dim=*/0, /*index=*/2) == 19.0f)
                  .item<bool>());
  EXPECT_TRUE(torch::all(second.ssm.narrow(/*dim=*/0,
                                           /*start=*/2 * kSecondStride,
                                           /*length=*/kSecondStride) == 23.0f)
                  .item<bool>());
  expect_slot_filled(first, /*slot=*/3, kFirstStride, /*value=*/0.0f);
  expect_slot_filled(second, /*slot=*/3, kSecondStride, /*value=*/0.0f);
  expect_slot_filled(first, /*slot=*/5, kFirstStride, /*value=*/0.0f);
  expect_slot_filled(second, /*slot=*/5, kSecondStride, /*value=*/0.0f);
}

}  // namespace
}  // namespace xllm
