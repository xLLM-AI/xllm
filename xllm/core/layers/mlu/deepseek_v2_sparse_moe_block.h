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

#include <torch/torch.h>

#include <functional>

#include "deepseek_v2_attention.h"
#include "framework/model/model_args.h"
#include "framework/model/model_input_params.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/parallel_state/parallel_state.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/common/dp_utils.h"
#include "layers/mlu/deepseek_v32_cp_context.h"
#include "layers/mlu/fused_moe.h"

namespace xllm {
namespace layer {

class DeepseekV2SparseMoEBlockTestPeer;

class DeepseekV2SparseMoEBlockImpl : public torch::nn::Module {
 public:
  struct ExecCfg {
    bool enable_all2all = false;
    bool need_dp_gather = false;
  };

  struct PrepOut {
    torch::Tensor ffn_in;
    torch::Tensor skip_local;
    PaddingInfo pad_info;
    bool need_dp_gather = false;
    bool need_tp_pad = false;
  };

  struct CommFns {
    std::function<bool(ProcessGroup*)> can_keep_local;
    std::function<torch::Tensor(torch::Tensor, ProcessGroup*)> comm;
    std::function<torch::Tensor(torch::Tensor, ProcessGroup*)> reduce;
    std::function<parallel_state::ReduceAsyncCtx(torch::Tensor, ProcessGroup*)>
        launch_reduce;
    std::function<torch::Tensor(parallel_state::ReduceAsyncCtx)> finish_reduce;
  };

  struct ForwardResult {
    torch::Tensor output;
    bool keep_local_output = false;
  };

  DeepseekV2SparseMoEBlockImpl() = default;
  DeepseekV2SparseMoEBlockImpl(const ModelArgs& model_args,
                               const QuantArgs& quant_args,
                               const ParallelArgs& parallel_args,
                               const torch::TensorOptions& options);

  void load_state_dict(const StateDict& state_dict);
  void verify_loaded_weights() const;

  ExecCfg plan_exec(const ModelInputParams& input_params) const;
  PrepOut prep_in(torch::Tensor x,
                  const torch::Tensor& residual,
                  const ModelInputParams& input_params,
                  const ExecCfg& exec,
                  DeepseekV2AttentionImpl::PostAttnLayout attn_layout) const;
  torch::Tensor gather_in(const PrepOut& prep,
                          const ModelInputParams& input_params) const;
  torch::Tensor merge_out(torch::Tensor x,
                          const PrepOut& prep,
                          const ModelInputParams& input_params) const;
  bool has_shared() const;

  ForwardResult forward(torch::Tensor x,
                        bool enable_moe_all2all,
                        const CommFns& comm_fns);
  ForwardResult forward_sp(torch::Tensor x,
                           const v32_cp::DeepseekV32CPContext& sp_ctx,
                           const CommFns& comm_fns);

 private:
  std::pair<torch::Tensor, PaddingInfo> shard_attn_out(
      torch::Tensor x,
      const torch::Tensor& residual,
      int64_t target_tokens,
      DeepseekV2AttentionImpl::PostAttnLayout attn_layout) const;
  torch::Tensor slice_tp_tokens(torch::Tensor x) const;
  ProcessGroup* routed_pg() const;

  ParallelArgs parallel_args_;
  bool enable_deep_ep_ = false;
  FusedMoE moe_{nullptr};

  friend class DeepseekV2SparseMoEBlockTestPeer;
};

TORCH_MODULE(DeepseekV2SparseMoEBlock);

}  // namespace layer
}  // namespace xllm
