/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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
#include <torch/torch.h>

#include <thread>

#include "dit_executor.h"
#include "framework/dit_model_context.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "options.h"
#include "platform/device.h"
#include "util/threadpool.h"
#include "worker_impl.h"

namespace xllm {

class DiTWorkerImpl : public WorkerImpl {
 public:
  DiTWorkerImpl(const ParallelArgs& parallel_args,
                const torch::Device& device,
                const runtime::Options& options);

  ~DiTWorkerImpl() = default;

  // initialize model, cache manager. blocking call
  bool init_model(const std::string& model_weights_path,
                  int32_t random_seed,
                  MasterStatus master_status) override;

  folly::SemiFuture<bool> init_model_async(
      const std::string& model_weights_path,
      int32_t random_seed,
      MasterStatus master_status) override;

  bool init_model(ModelContext& context) override;

  std::optional<ForwardOutput> step(const ForwardInput& inputs) override;

  folly::SemiFuture<std::optional<ForwardOutput>> step_async(
      const ForwardInput& inputs);

  folly::SemiFuture<std::optional<DiTForwardOutput>> step_async(
      const DiTForwardInput& inputs);

  void process_group_test();

  folly::SemiFuture<folly::Unit> process_group_test_async();

  // prepare input for execution
  DiTForwardInput prepare_inputs(DiTBatch& batch);

  int64_t get_active_activation_memory();

 private:
  std::unique_ptr<DiTModel> dit_model_;

  std::unique_ptr<DiTExecutor> dit_model_executor_;

  // model context, includes model args, parallel args and date type etc.
  mutable DiTModelContext dit_context_;

  ThreadPool threadpool_{/*num_threads=*/1,
                         /*cpu_binding=*/false,
                         /*pool_name=*/"DiTWorkerImpl.schedule"};
};

}  // namespace xllm
