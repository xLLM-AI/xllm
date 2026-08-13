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

#include "models/llm/py_causal_lm.h"

#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <algorithm>
#include <memory>
#include <string>
#include <utility>

#include "core/framework/config/execution_config.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/state_dict/state_dict.h"
#include "models/py_model_helper.h"

namespace py = pybind11;

namespace xllm {

PyCausalLM::PyCausalLM(const ModelContext& context)
    : model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      device_(context.get_tensor_options().device()),
      enable_mla_(context.get_model_args().enable_mla()) {
  ensure_python_interpreter();

  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;
  ProcessGroup* dp_group = parallel_args.dp_local_process_group_;
  dp_size_ = (dp_group != nullptr) ? dp_group->world_size() : 1;
  dp_rank_ = (dp_group != nullptr) ? dp_group->rank() : 0;
  ep_size_ = parallel_args.ep_size();
  CHECK(ep_size_ == 1 || ep_size_ == parallel_args.world_size())
      << "Python models support only ep_size=1 or ep_size=world_size.";

  CHECK(parallel_args.moe_tp_group_ != nullptr);
  moe_tp_group_ = parallel_args.moe_tp_group_;
  if (ep_size_ > 1) {
    CHECK(parallel_args.moe_ep_group_ != nullptr);
    moe_ep_group_ = parallel_args.moe_ep_group_;
  }
  moe_tp_size_ =
      (moe_tp_group_ != nullptr) ? moe_tp_group_->world_size() : 1;
  moe_tp_rank_ = (moe_tp_group_ != nullptr) ? moe_tp_group_->rank() : 0;
  ep_rank_ = (moe_ep_group_ != nullptr) ? moe_ep_group_->rank() : 0;
  cp_group_ = parallel_args.cp_group_;
  cp_size_ = std::max<int64_t>(parallel_args.cp_size(), 1);
  cp_rank_ = (cp_group_ != nullptr) ? cp_group_->rank() : 0;

  py::gil_scoped_acquire gil;
  // Skip Python dist.init_process_group — the C++ ProcessGroup (tp_group_,
  // moe_tp_group_, etc.) is already initialized by collective_communicator.cpp
  // and we expose its collectives via xllm_runtime.tp_all_reduce /
  // tp_all_gather. A second HCCL communicator from Python would conflict with
  // the C++ one on the same device (HCCL watchdog timeout 507015).
  // The python_rendezvous check is relaxed: if the host is empty, the Python
  // distributed module simply has no groups — all collectives go through C++.
  (void)parallel_args;
  const std::string module_name = context.get_model_args().model_type().empty()
                                      ? std::string("Qwen3ForCausalLM")
                                      : context.get_model_args().model_type();

  py::module_ registry = py::module_::import("xllm.python.registry");
  py::object model_cls = registry.attr("get_model_class")(py::str(module_name));
  config_dict_ = build_config_dict(parallel_args);
  py_model_ = model_cls(config_dict_);
  py_model_.attr("eval")();
}

PyCausalLM::~PyCausalLM() {
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
  config_dict_ = py::object();
}

py::dict PyCausalLM::build_config_dict(
    const ParallelArgs& parallel_args) const {
  py::dict d;
  PyDictVisitor visitor(d);
  visit_properties(model_args_, visitor);
  visit_properties(parallel_args, visitor);
  d["dtype"] = dtype_to_string(options_);
  d["device"] = c10::str(device_);
  d["tp_size"] = tp_size_;
  d["tp_rank"] = tp_rank_;
  d["dp_size"] = dp_size_;
  d["dp_rank"] = dp_rank_;
  d["moe_tp_size"] = moe_tp_size_;
  d["moe_tp_rank"] = moe_tp_rank_;
  d["ep_size"] = ep_size_;
  d["ep_rank"] = ep_rank_;
  d["cp_size"] = cp_size_;
  d["cp_rank"] = cp_rank_;
  d["enable_graph"] = ExecutionConfig::get_instance().enable_graph();
  d["enable_graph_double_buffer"] =
      ExecutionConfig::get_instance().enable_graph_double_buffer();
  d["python_graph_backend"] =
      ExecutionConfig::get_instance().python_graph_backend();
  return d;
}

void PyCausalLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  auto& state_dicts = loader->get_state_dicts();
  py::module_::import("xllm_weight_loader");

  py::list py_state_dicts;
  for (const auto& sd : state_dicts) {
    py_state_dicts.append(
        py::cast(PyStateDict(sd.get()), py::return_value_policy::move));
  }

  py_model_.attr("load_weights")(py_state_dicts,
                                 static_cast<int32_t>(tp_rank_),
                                 static_cast<int32_t>(tp_size_));
}

ModelOutput PyCausalLM::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) {
  LOG(FATAL) << "PyCausalLM::forward() must not be called directly. "
             << "Python model forward goes through PyExecutorImpl.";
  return ModelOutput(torch::Tensor());
}

torch::Tensor PyCausalLM::logits(const torch::Tensor& hidden_states,
                                 const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object out = py_model_.attr("compute_logits")(hidden_states, selected);
  return out.cast<torch::Tensor>();
}

void PyCausalLM::tp_all_reduce(torch::Tensor& tensor) {
  if (tp_group_ != nullptr) {
    tp_group_->allreduce(tensor);
  }
}

torch::Tensor PyCausalLM::tp_all_gather(const torch::Tensor& tensor, int64_t dim) {
  if (tp_group_ == nullptr) {
    return tensor;
  }
  // allgather_base_sync returns [world_size, *input_shape].
  // We want to concat along `dim`: the result should have shape input_shape
  // but with dim multiplied by world_size.
  auto gathered = tp_group_->allgather_base_sync(tensor);
  int64_t ws = tp_group_->world_size();
  auto in_shape = tensor.sizes().vec();
  int64_t ndim = static_cast<int64_t>(in_shape.size());
  if (dim < 0) {
    dim += ndim;
  }
  // gathered shape: [ws, in_shape...]. Move ws dim to position dim+1, then
  // merge dims dim and dim+1.
  // gathered.transpose(0, dim+1) puts ws at dim+1, in_shape[dim] at 0.
  // But that's complex; simpler: reshape [ws, *in_shape] to [ws, ..., in_dim, ...]
  // then transpose(0, dim+1) then flatten dim and dim+1.
  // For the common 2D case [ws, T, H] with dim=1 (last):
  //   transpose(0,2) -> [H, T, ws] -> permute to [T, ws, H] -> reshape [T, ws*H]
  // General approach: move ws to dim+1, then merge.
  std::vector<int64_t> perm;
  perm.push_back(0);  // will be overwritten
  // Build permutation: [dim, 1, 2, ..., dim-1, 0(ws), dim+1, ...]
  // Actually simplest: contiguous reshape won't work because ws is at front.
  // Use: gathered.reshape({ws, -1}) then transpose(0,1) then reshape.
  // No -- let's do it properly:
  // gathered: [ws, s0, s1, ..., s_{dim}, ..., s_{n-1}]
  // Target: [s0, ..., s_{dim}*ws, ..., s_{n-1}]
  // Step 1: transpose(0, dim+1) -> [s_{dim}, s1, ..., ws, ..., s_{n-1}] -- no
  // Step 2: Actually, just transpose ws to dim+1 position:
  // perm = [1, 2, ..., dim, 0, dim+1, ..., n]  (move 0 to position dim+1)
  perm.clear();
  for (int64_t i = 1; i <= dim; ++i) {
    perm.push_back(i);
  }
  perm.push_back(0);  // ws goes to position dim+1 (0-indexed: dim)
  for (int64_t i = dim + 1; i < ndim + 1; ++i) {  // ndim+1 because gathered has ndim+1 dims
    perm.push_back(i);
  }
  gathered = gathered.permute(perm);
  // Now shape: [s0, ..., s_dim, ws, s_{dim+1}, ...]
  // Merge dim and dim+1 (ws):
  auto out_shape = in_shape;
  out_shape[dim] *= ws;
  return gathered.reshape(out_shape).contiguous();
}

void PyCausalLM::moe_tp_all_reduce(torch::Tensor& tensor) {
  if (moe_tp_group_ != nullptr) {
    moe_tp_group_->allreduce(tensor);
  }
}

void PyCausalLM::moe_ep_all_reduce(torch::Tensor& tensor) {
  if (moe_ep_group_ != nullptr) {
    moe_ep_group_->allreduce(tensor);
  }
}

torch::Tensor PyCausalLM::cp_gather(
    const torch::Tensor& tensor,
    const std::vector<int32_t>& tokens_per_rank) {
  return parallel_state::gather(tensor, cp_group_, tokens_per_rank);
}

}  // namespace xllm
