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

#include "py_causal_lm.h"

#include <c10/util/Exception.h>
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/extension.h>

#include <cmath>
#include <string>
#include <utility>

#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/framework/model_loader.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/state_dict/state_dict.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "py_model_bridge.h"

namespace py = pybind11;

namespace xllm {

namespace {

std::string dtype_to_string(const torch::TensorOptions& options) {
  switch (c10::typeMetaToScalarType(options.dtype())) {
    case torch::kBFloat16:
      return "bfloat16";
    case torch::kFloat16:
      return "float16";
    case torch::kFloat32:
      return "float32";
    case torch::kFloat64:
      return "float64";
    default:
      return "bfloat16";
  }
}

}  // namespace

PyCausalLM::PyCausalLM(const ModelContext& context)
    : model_args_(context.get_model_args()),
      options_(context.get_tensor_options()),
      device_(context.get_tensor_options().device()),
      enable_mla_(context.get_model_args().enable_mla()) {
  // Bring up the embedded interpreter + xllm_models package + xllm_ops library.
  ensure_python_interpreter();

  // Tensor parallelism: read the TP group so per-rank head counts (and the
  // Python-side weight shards, driven by tp_rank/tp_size in the config dict)
  // match the native path. tp_group_ is null / world_size 1 on a single card,
  // degrading cleanly to the TP=1 layout.
  const ParallelArgs& parallel_args = context.get_parallel_args();
  tp_group_ = parallel_args.tp_group_;
  tp_size_ = (tp_group_ != nullptr) ? tp_group_->world_size() : 1;
  tp_rank_ = (tp_group_ != nullptr) ? tp_group_->rank() : 0;

  // Config-only attention module, shared across layers. Constructed here (after
  // the worker's flashinfer workspace init on this thread) with PER-RANK head
  // counts, replicating qwen2_attention.cpp:47-65 so the flashinfer plan
  // matches the q/k/v slices the Python graph produces on this rank.
  const int64_t total_num_heads = model_args_.n_heads();
  const int64_t total_num_kv_heads =
      model_args_.n_kv_heads().value_or(model_args_.n_heads());
  const int64_t head_dim = model_args_.head_dim();
  CHECK(total_num_heads % tp_size_ == 0)
      << "n_heads " << total_num_heads << " not divisible by tp_size "
      << tp_size_;
  const int64_t num_heads = total_num_heads / tp_size_;
  int64_t num_kv_heads;
  if (total_num_kv_heads >= tp_size_) {
    CHECK(total_num_kv_heads % tp_size_ == 0)
        << "n_kv_heads " << total_num_kv_heads << " not divisible by tp_size "
        << tp_size_;
    num_kv_heads = total_num_kv_heads / tp_size_;
  } else {
    CHECK(tp_size_ % total_num_kv_heads == 0)
        << "tp_size " << tp_size_ << " not divisible by n_kv_heads "
        << total_num_kv_heads;
    num_kv_heads = 1;
  }
  const float scale = std::sqrt(1.0f / static_cast<float>(head_dim));
  attn_ = layer::Attention(
      num_heads, head_dim, scale, num_kv_heads, model_args_.sliding_window());
  register_module("attn", attn_);

  py::gil_scoped_acquire gil;
  const std::string module_name =
      context.get_model_args().model_type().empty()
          ? std::string("Qwen3ForCausalLM")
          : context.get_model_args().model_type();

  // Resolve the Python model class via the registry, then instantiate it with
  // the config dict. Keep the object alive for the model's lifetime.
  py::module_ registry = py::module_::import("xllm_models.registry");
  py::object model_cls = registry.attr("get_model_class")(py::str(module_name));
  py_model_ = model_cls(build_config_dict());
  py_model_.attr("eval")();

  forward_batch_cls_ =
      py::module_::import("xllm_models.forward_batch").attr("ForwardBatch");
}

PyCausalLM::~PyCausalLM() {
  // Release Python references under the GIL to avoid refcount races.
  py::gil_scoped_acquire gil;
  py_model_ = py::object();
  forward_batch_cls_ = py::object();
}

py::dict PyCausalLM::build_config_dict() const {
  py::dict d;
  d["model_type"] = model_args_.model_type();
  d["hidden_size"] = model_args_.hidden_size();
  d["num_hidden_layers"] = model_args_.n_layers();
  d["num_attention_heads"] = model_args_.n_heads();
  d["num_key_value_heads"] =
      model_args_.n_kv_heads().value_or(model_args_.n_heads());
  d["head_dim"] = model_args_.head_dim();
  d["intermediate_size"] = model_args_.intermediate_size();
  d["rms_norm_eps"] = model_args_.rms_norm_eps();
  d["rope_theta"] = model_args_.rope_theta();
  d["max_position_embeddings"] = model_args_.max_position_embeddings();
  d["vocab_size"] = model_args_.vocab_size();
  d["tie_word_embeddings"] = model_args_.tie_word_embeddings();
  d["dtype"] = dtype_to_string(options_);
  d["device"] = c10::str(device_);
  d["tp_size"] = tp_size_;
  d["tp_rank"] = tp_rank_;
  return d;
}

void PyCausalLM::load_model(std::unique_ptr<ModelLoader> loader) {
  py::gil_scoped_acquire gil;
  auto& state_dicts = loader->get_state_dicts();

  // GQA KV-head replica coords: when a KV head is shared across ranks
  // (n_kv_heads < tp_size), K/V shard with the reduced (rank, world) so the same
  // KV slice is replicated on the ranks that share it — mirrors
  // qwen2_attention.cpp:57-65 and the native load_tensor_list KV-replica rule.
  const int64_t total_kv_heads =
      model_args_.n_kv_heads().value_or(model_args_.n_heads());
  const int kv_replicas = (total_kv_heads < tp_size_)
                              ? static_cast<int>(tp_size_ / total_kv_heads)
                              : 1;
  const int rank = static_cast<int>(tp_rank_);
  const int world = static_cast<int>(tp_size_);
  const int kv_rank = (kv_replicas > 1) ? rank / kv_replicas : rank;
  const int kv_world = (kv_replicas > 1) ? world / kv_replicas : world;

  // The Python model declares HOW each parameter is assembled from the
  // checkpoint (source names, shard dim, KV-replica flag, concat dim) via
  // sharding_plan(); this bridge stays model-agnostic and only EXECUTES the
  // sharding through the native StateDict::get_sharded_tensor, so the Python
  // graph receives per-rank weights from the single, validated chunk path
  // (no sharding is re-implemented in Python).
  const py::object plan = py_model_.attr("sharding_plan")();

  auto find_owner =
      [&state_dicts](const std::string& name) -> const StateDict* {
    for (const auto& sd : state_dicts) {
      if (sd->has(name)) {
        return sd.get();
      }
    }
    return nullptr;
  };

  py::list assembled;
  for (const auto& entry_handle : plan) {
    const auto entry = entry_handle.cast<py::sequence>();
    const std::string target = entry[0].cast<std::string>();
    const auto sources = entry[1].cast<py::sequence>();
    const int64_t cat_dim = entry[2].cast<int64_t>();

    std::vector<torch::Tensor> parts;
    parts.reserve(py::len(sources));
    for (const auto& source_handle : sources) {
      const auto source = source_handle.cast<py::sequence>();
      const auto candidates = source[0].cast<py::sequence>();
      const int shard_dim = source[1].cast<int>();
      const bool is_kv = source[2].cast<bool>();

      std::string chosen;
      const StateDict* owner = nullptr;
      for (const auto& cand_handle : candidates) {
        const std::string cand = cand_handle.cast<std::string>();
        const StateDict* hit = find_owner(cand);
        if (hit != nullptr) {
          chosen = cand;
          owner = hit;
          break;
        }
      }
      CHECK(owner != nullptr)
          << "PyCausalLM::load_model: no checkpoint tensor found for target '"
          << target << "'";

      torch::Tensor tensor;
      if (shard_dim >= 0) {
        const int r = is_kv ? kv_rank : rank;
        const int w = is_kv ? kv_world : world;
        tensor = owner->get_sharded_tensor(chosen, shard_dim, r, w);
      } else {
        tensor = owner->get_tensor(chosen);  // replicated (full) on every rank
      }
      parts.push_back(std::move(tensor));
    }

    torch::Tensor weight =
        (parts.size() == 1) ? parts.front() : torch::cat(parts, cat_dim);
    assembled.append(py::make_tuple(py::str(target), weight));
  }

  py_model_.attr("load_assembled_weights")(assembled);
}

ModelOutput PyCausalLM::forward(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& parameters) {
  torch::NoGradGuard no_grad;

  // Build the paged-KV attention metadata once per forward (CUDA path mirrors
  // the native qwen3 model), then expose it to the attention op via the
  // thread-local forward context for the duration of the Python call.
  layer::AttentionMetadata attn_metadata =
      layer::AttentionMetadataBuilder::build(parameters, enable_mla_);

  PyForwardContextGuard guard(&attn_metadata, &kv_caches, &attn_, tp_group_);

  py::gil_scoped_acquire gil;
  py::object batch = forward_batch_cls_(py::arg("positions") = positions,
                                        py::arg("native_attention") = false);
  py::object hidden_obj = py_model_.attr("forward")(tokens, positions, batch);
  auto hidden = hidden_obj.cast<torch::Tensor>();
  return ModelOutput(hidden);
}

torch::Tensor PyCausalLM::logits(const torch::Tensor& hidden_states,
                                 const torch::Tensor& seleted_idxes) {
  torch::NoGradGuard no_grad;

  // compute_logits runs lm_head, whose ColumnParallel(gather_output=True) does
  // an all_gather over the vocab dim. That collective reads the TP group from
  // the thread-local forward context via py_all_gather, so establish the
  // context here too (attention/KV state is unused while computing logits, hence
  // null). Without it py_all_gather degrades to identity and the logits keep
  // only this rank's vocab shard — the sampler then never sees the other ranks'
  // tokens (e.g. a tp=2 rank0 sees only vocab [0, vocab/2)).
  PyForwardContextGuard guard(/*attn_metadata=*/nullptr,
                              /*kv_caches=*/nullptr,
                              /*attn=*/nullptr,
                              tp_group_);
  py::gil_scoped_acquire gil;
  py::object selected = seleted_idxes.defined()
                            ? py::object(py::cast(seleted_idxes))
                            : py::object(py::none());
  py::object out =
      py_model_.attr("compute_logits")(hidden_states, selected);
  return out.cast<torch::Tensor>();
}

}  // namespace xllm
