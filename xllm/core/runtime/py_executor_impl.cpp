/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/py_executor_impl.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <memory>
#include <mutex>
#include <optional>
#include <utility>
#include <vector>

#include "common/metrics.h"
#include "core/framework/config/execution_config.h"
#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/mtp_topk_state.h"
#include "core/framework/multimodal/mm_batch_data.h"
#include "core/framework/multimodal/mm_data.h"
#include "core/framework/multimodal/mm_visitor.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/platform/platform.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/runtime/slot_buffer.h"
#include "core/util/pybind_helper.h"
#include "models/llm/py_causal_lm.h"
#include "models/py_model_helper.h"
#include "models/vlm/utils/multimodal_utils.h"
#include "platform/layer_synchronizer.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "platform/npu/device_capture_lock.h"
#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {
namespace {

py::object mtp_topk_indices(const ModelParamsView& params) {
  if (params.mtp_topk_state != nullptr) {
    const auto tensor = params.mtp_topk_state->as_tensor();
    CHECK(tensor.has_value())
        << "Python model requires a single-tensor MTP top-k state";
    return py::cast(*tensor);
  }
  return py::none();
}

py::object dspark_adaptive_graph_input(const ModelParamsView& params) {
  const auto& input = params.graph.dspark_adaptive;
  if (!input.arena.defined()) {
    return py::none();
  }
  return py::module_::import("xllm.python.model_executor.dspark_graph_input")
      .attr("DSparkAdaptiveGraphInputs")(
          py::arg("arena") = input.arena,
          py::arg("control") = input.control,
          py::arg("anchors") = input.anchors,
          py::arg("base_positions") = input.base_positions,
          py::arg("base_kv_lengths") = input.base_kv_lengths,
          py::arg("base_block_table") = input.base_block_table,
          py::arg("offsets") = input.offsets,
          py::arg("owners") = input.owners,
          py::arg("columns") = input.columns,
          py::arg("block_size") = input.block_size,
          py::arg("draft_width") = input.draft_width,
          py::arg("linear_state_indices") =
              input.linear_state_indices.defined()
                  ? py::cast(input.linear_state_indices)
                  : py::none(),
          py::arg("num_accepted_tokens") =
              input.num_accepted_tokens.defined()
                  ? py::cast(input.num_accepted_tokens)
                  : py::none());
}

void register_xllm_runtime_module(py::module_& m) {
  register_attention_metadata_views(m);
  py::class_<ExecutionBatchMetadata>(m, "InputBatchMetadata")
      .def_readonly("num_reqs", &ExecutionBatchMetadata::num_reqs)
      .def_readonly("num_tokens", &ExecutionBatchMetadata::num_tokens)
      .def_readonly("num_scheduled_tokens",
                    &ExecutionBatchMetadata::num_scheduled_tokens)
      .def_readonly("num_computed_tokens",
                    &ExecutionBatchMetadata::num_computed_tokens)
      .def_readonly("query_start_loc", &ExecutionBatchMetadata::query_start_loc)
      .def_readonly("is_prefilling", &ExecutionBatchMetadata::is_prefilling);

  py::class_<LayerSynchronizer, std::shared_ptr<LayerSynchronizer>>(
      m, "HostKVLoadSynchronizer")
      .def("wait_group",
           &LayerSynchronizer::wait_layer_on_current_stream,
           py::call_guard<py::gil_scoped_release>())
      .def("size", &LayerSynchronizer::size);

  m.def("tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("tp_all_gather", [](torch::Tensor tensor, int64_t dim) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      return py_causal_lm->tp_all_gather(tensor, dim);
    }
    return tensor;
  });
  m.def("dp_all_gather",
        [](torch::Tensor tensor,
           const std::vector<int32_t>& execution_token_counts) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          if (py_causal_lm != nullptr) {
            return py_causal_lm->dp_all_gather(tensor, execution_token_counts);
          }
          return tensor;
        });
  m.def("moe_tp_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_tp_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("moe_ep_all_reduce", [](torch::Tensor tensor) {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    if (py_causal_lm != nullptr) {
      py_causal_lm->moe_ep_all_reduce(tensor);
    }
    return tensor;
  });
  m.def("eplb_batch_isend_irecv",
        [](py::list operation_types, py::list tensors, py::list remote_ranks) {
          PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
          CHECK(py_causal_lm != nullptr)
              << "EPLB P2P transfer requires an active PyCausalLM.";
          py_causal_lm->eplb_batch_isend_irecv(
              operation_types, tensors, remote_ranks);
        });
  m.def("eplb_wait_batch_isend_irecv", []() {
    PyCausalLM* py_causal_lm = PyCausalLM::active_instance();
    CHECK(py_causal_lm != nullptr)
        << "EPLB P2P transfer requires an active PyCausalLM.";
    py_causal_lm->eplb_wait_batch_isend_irecv();
  });

#if defined(USE_NPU)
  py::class_<NPULayerSynchronizerImpl,
             std::shared_ptr<NPULayerSynchronizerImpl>>(m, "LayerSynchronizer")
      .def("record_event",
           [](NPULayerSynchronizerImpl& self, int64_t layer_id) {
             int32_t device_id = static_cast<int32_t>(
                 c10_npu::getCurrentNPUStream().device_index());
             return self.record_event(layer_id, device_id);
           });
#endif
}

void ensure_xllm_runtime_module() {
  py::module_ sys = py::module_::import("sys");
  py::dict modules = py::reinterpret_borrow<py::dict>(sys.attr("modules"));
  const py::str module_name("xllm_runtime");
  if (modules.contains(module_name)) {
    return;
  }

  py::object module_object =
      py::module_::import("types").attr("ModuleType")(module_name);
  py::module_ module = py::reinterpret_borrow<py::module_>(module_object);
  register_xllm_runtime_module(module);
  modules[module_name] = module;
}

}  // namespace

PyExecutorImpl::PyExecutorImpl(CausalLM* model,
                               const ModelArgs& args,
                               const torch::Device& device,
                               const runtime::Options& options)
    : py_causal_lm_(dynamic_cast<PyCausalLM*>(model)),
      args_(args),
      dummy_token_count_(dummy_token_count(args)),
      device_(device),
      options_(options),
      enable_mla_(args.enable_mla()) {
  CHECK(py_causal_lm_ != nullptr) << "PyExecutorImpl requires PyCausalLM";
  CHECK_GT(dummy_token_count_, 0);
  // GLM5 KDA targets need fixed-width speculative decoding; adaptive pruning
  // resizes per-step draft widths, which the KDA handoff cannot follow.
  CHECK(!(is_glm5_next_target_model_type(args.model_type()) &&
          (options.enable_adaptive_target_verify() ||
           options.enable_adaptive_draft_step())))
      << "GLM5 KDA does not support adaptive speculative decode; disable "
         "--enable_adaptive_target_verify and --enable_adaptive_draft_step.";

  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  // These values belong to the executor, not the checkpoint. In particular,
  // the MTP draft has different speculative options from its target.
  py::dict executor_config = py_causal_lm_->config_dict().attr("copy")();
  executor_config["enable_disagg_pd"] = options_.enable_disagg_pd();
  executor_config["instance_role"] = options_.instance_role().to_string();
  executor_config["task_type"] = options_.task_type();
  executor_config["max_encoder_cache_size"] = options_.max_encoder_cache_size();
  executor_config["num_speculative_tokens"] = options_.num_speculative_tokens();
  executor_config["mtp_draft_prefix_width"] = options_.mtp_draft_prefix_width();
  executor_config["enable_adaptive_draft_step"] =
      options_.enable_adaptive_draft_step();
  executor_config["enable_adaptive_target_verify"] =
      options_.enable_adaptive_target_verify();
  executor_config["engine_type"] = SpeculativeConfig::uses_ssm_engine(
                                       options_.draft_model_path().value_or(""),
                                       options_.speculative_algorithm(),
                                       options_.num_speculative_tokens())
                                       ? "SSM"
                                       : "LLM";
  executor_config["enable_chunked_prefill"] = options_.enable_chunked_prefill();
  executor_config["enable_prefix_cache"] =
      KVCacheConfig::get_instance().enable_prefix_cache();
  executor_config["enable_schedule_overlap"] =
      options_.enable_schedule_overlap();
  executor_config["enable_unified_mtp_graph"] =
      ExecutionConfig::get_instance().enable_unified_mtp_graph();
  executor_config["is_draft_engine"] = options_.is_draft_engine();
  executor_config["is_mtp_algorithm"] =
      SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm());
  executor_config["is_block_diffusion_algorithm"] =
      SpeculativeConfig::is_block_diffusion_algorithm(
          options_.speculative_algorithm());
  executor_config["requires_aux_hidden_capture"] =
      SpeculativeConfig::requires_aux_hidden_capture(
          options_.speculative_algorithm());
  executor_config["enable_task_pipeline"] = options_.enable_task_pipeline();
  executor_config["block_draft_graph_warmup_max_kv_len"] =
      ExecutionConfig::get_instance().block_draft_graph_warmup_max_kv_len();
  executor_config["enable_graph_mode_decode_no_padding"] =
      ExecutionConfig::get_instance().enable_graph_mode_decode_no_padding();
  executor_config["enable_adaptive_target_verify"] =
      options_.enable_adaptive_target_verify();
  // DFlash2 defaults to eager in the model config. Hybrid worker pipelines
  // explicitly enable its independent full-block ACL runner on the leaf.
  const bool independent_block_graph =
      Platform::is_npu() && is_dflash2_draft_model_type(args_.model_type()) &&
      options_.enable_graph() && options_.num_decoding_tokens() > 1;
  if (options_.enable_task_pipeline() || independent_block_graph) {
    executor_config["enable_graph"] = options_.enable_graph();
    executor_config["python_graph_backend"] =
        options_.enable_graph() ? "aclgraph" : "off";
  }
  py_executor_ = executor_module.attr("ModelExecutor")(
      py_causal_lm_->python_model(),
      executor_config,
      options_.max_seqs_per_batch(),
      options_.num_decoding_tokens(),
      ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit());
  py_causal_lm_->set_context_kv_graph_runner(
      py_executor_.attr("context_kv_graph_runner"));
  requires_input_batch_metadata_ =
      py_executor_.attr("requires_input_batch_metadata").cast<bool>();
  supports_prepared_metadata_ =
      py_executor_.attr("supports_prepared_metadata").cast<bool>();
  supports_layerwise_kv_load_ =
      py_executor_.attr("supports_layerwise_kv_load").cast<bool>();
}

PyExecutorImpl::~PyExecutorImpl() {
  if (adaptive_unified_registry_ && Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    adaptive_unified_registry_.attr("close")();
  }
  clear_python_object(adaptive_unified_registry_);
  if (block_graph_registry_ && Py_IsInitialized()) {
    py::gil_scoped_acquire gil;
    block_graph_registry_.attr("close")();
  }
  clear_python_object(block_graph_registry_);
  if (PyCausalLM::active_instance() == py_causal_lm_) {
    PyCausalLM::set_active_instance(nullptr);
  }
  clear_python_object(py_executor_);
}

bool PyExecutorImpl::supports_adaptive_unified_graph() const {
#if defined(USE_NPU)
  return supports_prepared_metadata_ && options_.enable_graph() &&
         options_.enable_task_pipeline();
#else
  return false;
#endif
}

void PyExecutorImpl::create_adaptive_unified_registry(
    PyExecutorImpl& draft_executor,
    ProcessGroup* sampling_group) {
#if defined(USE_NPU)
  CHECK(supports_adaptive_unified_graph());
  CHECK(draft_executor.supports_adaptive_unified_graph());
  if (adaptive_unified_registry_) {
    CHECK_EQ(adaptive_unified_draft_, &draft_executor);
    CHECK_EQ(adaptive_unified_sampling_group_, sampling_group);
    return;
  }
  auto module = py::module_::import(
      "xllm.python.model_executor.runners.unified_block_acl_graph");
  adaptive_unified_registry_ = module.attr("AdaptiveUnifiedBlockGraphRegistry")(
      py_executor_,
      draft_executor.py_executor_,
      py::cpp_function(
          [this]() { PyCausalLM::set_active_instance(py_causal_lm_); }),
      py::cpp_function([model = draft_executor.py_causal_lm_]() {
        PyCausalLM::set_active_instance(model);
      }),
      py::cpp_function([sampling_group](torch::Tensor arena) {
        // Confidence is local to each TP rank even when the token sampler is
        // greedy. Publish one authoritative compact proposal/confidence arena
        // inside the graph before allocating any verification prefixes.
        if (sampling_group != nullptr && sampling_group->world_size() > 1) {
          sampling_group->broadcast(arena, /*root_rank=*/0);
        }
      }),
      py::cpp_function(
          [device_index = device_.index()](const py::object& runner) {
            // Prepare can take the GIL while holding this device lock.
            py::gil_scoped_release release;
            auto& capture_lock =
                npu::DeviceCaptureLock::get_instance().get_lock(device_index);
            std::lock_guard<std::mutex> lock(capture_lock);
            py::gil_scoped_acquire acquire;
            runner.attr("capture")();
          }));
  adaptive_unified_draft_ = &draft_executor;
  adaptive_unified_sampling_group_ = sampling_group;
#else
  LOG(FATAL) << "Adaptive unified DSpark graphs require NPU.";
#endif
}

void PyExecutorImpl::warmup_adaptive_unified_graph(
    ExecutorImpl& draft_executor,
    const AdaptiveDSparkGraphTask& task,
    ProcessGroup* sampling_group) {
  auto* draft = dynamic_cast<PyExecutorImpl*>(&draft_executor);
  CHECK(draft != nullptr);
  py::gil_scoped_acquire gil;
  create_adaptive_unified_registry(*draft, sampling_group);
  const auto binding = [](SlotBuffer& input) {
    return py::make_tuple(
        input.tokens(),
        input.positions(),
        input.model_params().python_attention_metadata->value());
  };
  adaptive_unified_registry_.attr("warmup")(
      binding(task.target),
      binding(task.draft),
      task.result.tokens,
      task.result.lengths,
      dspark_adaptive_graph_input(task.target.model_params()),
      task.context_slots);
}

torch::Tensor PyExecutorImpl::run_adaptive_unified_graph(
    ExecutorImpl& draft_executor,
    const AdaptiveDSparkGraphTask& task,
    ProcessGroup* sampling_group) {
  CHECK(adaptive_unified_registry_)
      << "Adaptive unified graphs must be captured during startup warmup.";
  CHECK_EQ(adaptive_unified_draft_, &draft_executor);
  CHECK_EQ(adaptive_unified_sampling_group_, sampling_group);
  py::gil_scoped_acquire gil;
  const auto binding = [](SlotBuffer& input) {
    return py::make_tuple(
        input.tokens(),
        input.positions(),
        input.model_params().python_attention_metadata->value());
  };
  return adaptive_unified_registry_
      .attr("execute")(binding(task.target),
                       binding(task.draft),
                       task.result.tokens,
                       task.result.lengths,
                       dspark_adaptive_graph_input(task.target.model_params()),
                       task.context_slots)
      .cast<torch::Tensor>();
}

py::object PyExecutorImpl::mtp_sparse_attention_metadata_view(
    const torch::Tensor& block_table,
    const torch::Tensor& kv_seq_lens,
    const torch::Tensor& slots) const {
  auto metadata = std::make_shared<layer::AttentionMetadata>(
      layer::AttentionMetadataBuilder::build_mtp_sparse_decode(
          block_table, kv_seq_lens, slots, options_.block_size()));
  LlmModelParams storage;
  ModelParamsView params(storage);
  const int32_t rows = static_cast<int32_t>(kv_seq_lens.numel());
  params.parallel.dp_global_token_nums = {rows};
  params.parallel.dp_global_sequence_nums = {rows};
  params.parallel.dp_is_decode = {true};
  return py::cast(
      PyAttentionMetadataView(std::move(metadata), params, dummy_token_count_));
}

py::object PyExecutorImpl::create_mtp_graph_variant_registry(
    PyExecutorImpl& draft_executor,
    int32_t max_variants) {
  py::gil_scoped_acquire gil;
  py::cpp_function draft_activate = py::cpp_function([&draft_executor]() {
    PyCausalLM::set_active_instance(draft_executor.py_causal_lm_);
  });
  py::cpp_function target_activate = py::cpp_function(
      [this]() { PyCausalLM::set_active_instance(py_causal_lm_); });
  py::object capture_runner = py::none();
#if defined(USE_NPU)
  capture_runner =
      py::cpp_function([device_index = device_.index()](
                           const py::object& runner, const py::args& inputs) {
        // Prepare may need the GIL while it holds this device lock. Never
        // wait for the lock with the GIL held. Only cold capture takes it.
        py::gil_scoped_release release;
        auto& capture_lock =
            npu::DeviceCaptureLock::get_instance().get_lock(device_index);
        std::lock_guard<std::mutex> lock(capture_lock);
        py::gil_scoped_acquire acquire;
        runner.attr("capture")(*inputs);
      });
#endif
  return py_executor_.attr("create_mtp_graph_variant_registry")(
      draft_executor.py_executor_,
      py::arg("max_variants") = max_variants,
      py::arg("draft_activate") = draft_activate,
      py::arg("target_activate") = target_activate,
      py::arg("capture_runner") = capture_runner,
      py::arg("draft_metadata_factory") =
          py::cpp_function([&draft_executor](const torch::Tensor& table,
                                             const torch::Tensor& lengths,
                                             const torch::Tensor& slots) {
            return draft_executor.mtp_sparse_attention_metadata_view(
                table, lengths, slots);
          }),
      py::arg("target_metadata_factory") =
          py::cpp_function([this](const torch::Tensor& table,
                                  const torch::Tensor& lengths,
                                  const torch::Tensor& slots) {
            return mtp_sparse_attention_metadata_view(table, lengths, slots);
          }));
}

torch::Tensor PyExecutorImpl::run_unified_block(
    PyExecutorImpl& draft_executor,
    SlotBuffer& target,
    SlotBuffer& draft,
    const TokenResultTensors& result,
    bool dspark,
    ProcessGroup* sampling_group) {
#if defined(USE_NPU)
  py::gil_scoped_acquire gil;
  if (!block_graph_registry_) {
    auto module = py::module_::import(
        "xllm.python.model_executor.runners.unified_block_acl_graph");
    block_graph_registry_ = module.attr("UnifiedBlockGraphRegistry")(
        py_executor_,
        draft_executor.py_executor_,
        py::cpp_function(
            [this]() { PyCausalLM::set_active_instance(py_causal_lm_); }),
        py::cpp_function([model = draft_executor.py_causal_lm_]() {
          PyCausalLM::set_active_instance(model);
        }),
        py::cpp_function([sampling_group](torch::Tensor tokens) {
          if (sampling_group != nullptr && sampling_group->world_size() > 1) {
            sampling_group->broadcast(tokens, /*root_rank=*/0);
          }
        }),
        py::cpp_function(
            [device_index = device_.index()](const py::object& runner) {
              py::gil_scoped_release release;
              auto& capture_lock =
                  npu::DeviceCaptureLock::get_instance().get_lock(device_index);
              std::lock_guard<std::mutex> lock(capture_lock);
              py::gil_scoped_acquire acquire;
              runner.attr("capture")();
            }));
  }
  const auto binding = [](SlotBuffer& input) {
    return py::make_tuple(
        input.tokens(),
        input.positions(),
        input.model_params().python_attention_metadata->value());
  };
  return block_graph_registry_
      .attr("execute")(binding(target),
                       binding(draft),
                       result.tokens,
                       result.lengths,
                       dspark)
      .cast<torch::Tensor>();
#else
  LOG(FATAL) << "Unified block graphs require NPU.";
  return {};
#endif
}

void PyExecutorImpl::bind_kv_caches(std::vector<KVCache>& kv_caches) {
  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py_executor_.attr("bind_kv_caches")(build_python_kv_caches(kv_caches));
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
    if (options_.enable_task_pipeline()) {
      prepared_kv_bindings_.reserve(6 * kv_caches.size());
      for (const auto& kv : kv_caches) {
        prepared_kv_bindings_.emplace_back(kv.get_k_cache());
        prepared_kv_bindings_.emplace_back(kv.get_v_cache());
        prepared_kv_bindings_.emplace_back(kv.get_index_cache());
        prepared_kv_bindings_.emplace_back(kv.get_conv_cache());
        prepared_kv_bindings_.emplace_back(kv.get_ssm_cache());
        prepared_kv_bindings_.emplace_back(
            kv.get_indexer_cache_scale().value_or(torch::Tensor()));
      }
    }
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
    if (options_.enable_task_pipeline()) {
      size_t index = 0;
      for (const auto& kv : kv_caches) {
        for (const auto& current :
             {kv.get_k_cache(),
              kv.get_v_cache(),
              kv.get_index_cache(),
              kv.get_conv_cache(),
              kv.get_ssm_cache(),
              kv.get_indexer_cache_scale().value_or(torch::Tensor())}) {
          const auto& bound = prepared_kv_bindings_[index++];
          CHECK_EQ(current.defined(), bound.defined())
              << "Prepared executor KV binding presence changed.";
          if (!current.defined()) {
            continue;
          }
          CHECK(current.data_ptr() == bound.data_ptr() &&
                current.sizes() == bound.sizes() &&
                current.strides() == bound.strides() &&
                current.scalar_type() == bound.scalar_type() &&
                current.device() == bound.device())
              << "Prepared executor KV binding changed after initialization.";
        }
      }
    }
  }
}

void PyExecutorImpl::prepare_attention_metadata(std::vector<KVCache>& kv_caches,
                                                ModelParamsView& params) {
  CHECK(supports_prepared_metadata_);
  CHECK(params.attn_metadata != nullptr);
  py::gil_scoped_acquire gil;
  bind_kv_caches(kv_caches);
  py::object metadata = py::cast(PyAttentionMetadataView(
      params.attn_metadata, params, dummy_token_count_));
  py_executor_.attr("prepare_metadata")(metadata);
  params.python_attention_metadata =
      std::make_shared<PythonAttentionMetadata>(std::move(metadata));
}

void PyExecutorImpl::warmup_prepared_graph(const torch::Tensor& tokens,
                                           const torch::Tensor& positions,
                                           std::vector<KVCache>& kv_caches,
                                           const ModelParamsView& params) {
  CHECK(params.python_attention_metadata != nullptr);
  py::gil_scoped_acquire gil;
  bind_kv_caches(kv_caches);
  PyCausalLM::set_active_instance(py_causal_lm_);
  py::kwargs graph_kwargs;
  if (params.graph.dspark_adaptive.arena.defined()) {
    graph_kwargs["dspark_adaptive"] = dspark_adaptive_graph_input(params);
  }
  if (requires_input_batch_metadata_) {
    graph_kwargs["input_batch_metadata"] =
        py::cast(ExecutionBatchMetadata(params.meta.execution_batch));
  }
  py_executor_.attr("warmup_prepared_graph")(
      tokens,
      positions,
      params.python_attention_metadata->value(),
      optional_tensor(params.embedding.input_embedding),
      mtp_topk_indices(params),
      optional_tensor(params.expert.expert_load_data),
      optional_tensor(params.expert.eplb_decode_token_mask),
      params.meta.is_graph_warmup,
      **graph_kwargs);
}

std::vector<torch::Tensor> PyExecutorImpl::get_prepared_aux_hidden_states(
    const torch::Tensor& tokens,
    const torch::Tensor& positions,
    const ModelParamsView& params) {
  CHECK(params.python_attention_metadata != nullptr);
  py::gil_scoped_acquire gil;
  return py_executor_
      .attr("get_prepared_aux_hidden_states")(
          tokens, positions, params.python_attention_metadata->value())
      .cast<std::vector<torch::Tensor>>();
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelParamsView& params) {
  torch::NoGradGuard no_grad;
  PyCausalLM::set_active_instance(py_causal_lm_);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (params.python_attention_metadata) {
    CHECK(supports_prepared_metadata_);
    CHECK(attn_metadata != nullptr);
  }
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }
  const bool prepared_graph =
      params.python_attention_metadata && params.enable_graph;
  const bool layerwise_kv_load =
      params.parallel.layer_wise_load_synchronizer != nullptr &&
      supports_layerwise_kv_load_ && !prepared_graph &&
      !params.is_spec_verify &&
      (attn_metadata->is_prefill || attn_metadata->is_chunked_prefill);
  if (params.parallel.layer_wise_load_synchronizer != nullptr) {
    VLOG(1) << "[HostKV][PythonLoad] batch_id=" << params.meta.batch_id
            << ", mode=" << (layerwise_kv_load ? "layerwise" : "wait_all")
            << ", groups="
            << params.parallel.layer_wise_load_synchronizer->size();
    // Captured graphs and models without load hooks retain the per-batch
    // fence. Eager prefill inserts waits at its actual layer consumers.
    if (!layerwise_kv_load) {
      CHECK(params.parallel.layer_wise_load_synchronizer
                ->wait_all_on_current_stream())
          << "Python model Host KV reload failed.";
    }
  }
  py::gil_scoped_acquire gil;

  // Prepared input already bound and checked KV caches during Prepare.
  if (!params.python_attention_metadata) {
    bind_kv_caches(kv_caches);
  }

  py::object py_metadata =
      params.python_attention_metadata
          ? params.python_attention_metadata->value()
          : py::cast(PyAttentionMetadataView(
                attn_metadata, params, dummy_token_count_));
  py::object py_input_batch_metadata =
      requires_input_batch_metadata_
          ? py::cast(ExecutionBatchMetadata(params.meta.execution_batch))
          : py::none();
  if (params.python_attention_metadata) {
    VLOG(1) << "Task pipeline model path="
            << (prepared_graph ? "aclgraph" : "eager")
            << ", batch_id=" << params.meta.batch_id
            << ", model_type=" << args_.model_type()
            << ", tokens=" << tokens.numel();
  }
  if (!prepared_graph) {
    COUNTER_INC(num_model_execution_total_eager);
  }

  py::object input_embedding =
      optional_tensor(params.embedding.input_embedding);
  py::object topk_indices = mtp_topk_indices(params);
  torch::Tensor execution_tokens =
      spec_verify::materialize_graph_speculative_verify_tokens(tokens,
                                                               params.graph);

  // Full encoder outputs belong to the MM items; the shared gather selects
  // this step's chunk rows and builds modality masks before Python merges them.
  if (params.has_multimodal() && params.multimodal().mm_data.valid()) {
    auto& mm_data = params.multimodal().mm_data;
    EncoderInputGatherVisitor input_gather;
    CHECK(mm_data.foreach (input_gather));
    CHECK(input_gather.finish(mm_data));

    torch::Tensor pixel_values =
        mm_data.get<torch::Tensor>("pixel_values").value_or(torch::Tensor());
    torch::Tensor image_grid_thw =
        mm_data.get<torch::Tensor>("image_grid_thw").value_or(torch::Tensor());
    torch::Tensor pixel_values_videos =
        mm_data.get<torch::Tensor>("pixel_values_videos")
            .value_or(torch::Tensor());
    torch::Tensor video_grid_thw =
        mm_data.get<torch::Tensor>("video_grid_thw").value_or(torch::Tensor());

    torch::Tensor input_features =
        mm_data.get<torch::Tensor>("input_features").value_or(torch::Tensor());
    torch::Tensor speech_lengths =
        mm_data.get<torch::Tensor>("speech_lengths").value_or(torch::Tensor());
    CHECK_EQ(input_features.defined(), speech_lengths.defined())
        << "input_features and speech_lengths must be provided together";

    py::object top_model = py_causal_lm_->python_model();
    MMDict encoded;
    if (pixel_values.defined() && image_grid_thw.defined()) {
      torch::Tensor raw = top_model.attr("encode")(pixel_values, image_grid_thw)
                              .cast<torch::Tensor>();
      encoded[get_embedding_key(MMType::IMAGE)] =
          split_by_token_nums(raw, get_mm_token_nums(mm_data, MMType::IMAGE));
    }
    if (pixel_values_videos.defined() && video_grid_thw.defined()) {
      torch::Tensor raw =
          top_model.attr("encode")(pixel_values_videos, video_grid_thw)
              .cast<torch::Tensor>();
      encoded[get_embedding_key(MMType::VIDEO)] =
          split_by_token_nums(raw, get_mm_token_nums(mm_data, MMType::VIDEO));
    }
    if (input_features.defined()) {
      torch::Tensor audio_meta = mm_data.get<torch::Tensor>("audio_encode_meta")
                                     .value_or(torch::Tensor());
      CHECK(audio_meta.defined()) << "Missing audio encoder metadata";
      encoded[get_embedding_key(MMType::AUDIO)] =
          top_model.attr("encode")(input_features, speech_lengths, audio_meta)
              .cast<std::vector<torch::Tensor>>();
    }

    EncoderOutputScatterVisitor scatter(encoded);
    CHECK(mm_data.foreach (scatter));
    CHECK(scatter.finish());
    EncoderEmbeddingGatherVisitor gather(device_,
                                         mm_data.type(),
                                         params.attention.host.kv_seq_lens,
                                         params.attention.host.q_seq_lens);
    CHECK(mm_data.foreach (gather));
    CHECK(gather.finish(mm_data));

    py::object image_embeds = optional_tensor(
        mm_data.get<torch::Tensor>(get_embedding_key(MMType::IMAGE)));
    py::object video_embeds = optional_tensor(
        mm_data.get<torch::Tensor>(get_embedding_key(MMType::VIDEO)));
    py::object audio_embeds = optional_tensor(
        mm_data.get<torch::Tensor>(get_embedding_key(MMType::AUDIO)));
    if (!audio_embeds.is_none()) {
      const auto audio_mask = mm_data.get<torch::Tensor>("audio|mask");
      CHECK(audio_mask.has_value());
      CHECK_EQ(audio_mask->numel(), execution_tokens.numel());
      input_embedding = top_model.attr("get_input_embeddings")(
          execution_tokens,
          image_embeds,
          video_embeds,
          audio_embeds,
          audio_mask->view(execution_tokens.sizes()));
    } else if (!image_embeds.is_none() || !video_embeds.is_none()) {
      // Vision models retain their existing deepstack state and merge
      // signature.
      top_model.attr("get_input_embeddings")(
          execution_tokens, image_embeds, video_embeds);
    }
  }

  // --- mRoPE: collapse [3, N] decode positions to 1-D ---
  // Only PURE decode collapses to 1-D: decode rows are identical
  // (forward_input_builder get_mrope_positions), and mRoPE(p,p,p) == standard
  // RoPE at p, so a single row feeds the captured aclgraph's 1-D
  // static_positions unchanged. Chunked/mixed prefill (is_prefill=false but
  // is_chunked_prefill=true) still needs the full [3, N] for the Python mRoPE
  // path, so it is excluded here. The 2-D shape itself is the mRoPE signal:
  // non-mRoPE models never receive 2-D positions, so no config flag is needed.
  torch::Tensor positions_arg = positions;
  if (positions.dim() == 2 && !attn_metadata->is_prefill &&
      !attn_metadata->is_chunked_prefill) {
    positions_arg = positions.slice(/*dim=*/0, /*start=*/0, /*end=*/1)
                        .squeeze(0)
                        .contiguous();
  }

  py::object py_sync = py::none();
#if defined(USE_NPU)
  if (params.parallel.layer_synchronizer) {
    py_sync = py::cast(params.parallel.layer_synchronizer);
  }
#endif

  py::object expert_load_data = optional_tensor(params.expert.expert_load_data);
  py::object eplb_decode_token_mask =
      optional_tensor(params.expert.eplb_decode_token_mask);

  // Execute: one C++ -> Python call per step. input_embedding stays None for
  // the Qwen3-VL python path (embeddings are merged via the attribute set by
  // get_input_embeddings above), so the runner takes the 2-arg model() branch
  // and Qwen3VLModel.forward reads _inputs_embeds. positions_arg carries the
  // mRoPE [3,N]->1-D decode collapse.
  py::kwargs graph_kwargs;
  if (layerwise_kv_load) {
    graph_kwargs["kv_load_synchronizer"] =
        py::cast(params.parallel.layer_wise_load_synchronizer);
    graph_kwargs["kv_load_layers_per_event"] = params.parallel.layers_per_event;
  }
  if (params.graph.dspark_adaptive.arena.defined()) {
    graph_kwargs["dspark_adaptive"] = dspark_adaptive_graph_input(params);
  }
  py::object hidden_obj =
      py_executor_.attr("execute")(execution_tokens,
                                   positions_arg,
                                   py_metadata,
                                   input_embedding,
                                   py_sync,
                                   topk_indices,
                                   prepared_graph,
                                   expert_load_data,
                                   eplb_decode_token_mask,
                                   params.meta.is_graph_warmup,
                                   py_input_batch_metadata,
                                   **graph_kwargs);
  if (layerwise_kv_load && params.parallel.draft_load_event_index.has_value()) {
    // Target prefill can overlap its own reload, but a following draft body
    // must still observe the composite handle's draft-cache completion.
    py::gil_scoped_release release;
    CHECK(params.parallel.layer_wise_load_synchronizer
              ->wait_layer_on_current_stream(
                  *params.parallel.draft_load_event_index))
        << "Python model draft Host KV reload failed.";
  }
  if (py::isinstance<py::tuple>(hidden_obj)) {
    py::tuple output = hidden_obj.cast<py::tuple>();
    CHECK(output.size() == 2 || output.size() == 3)
        << "Python model tuple output must be (hidden_states, "
           "aux_hidden_states) or (hidden_states, aux_hidden_states, "
           "mtp_topk_indices)";
    ModelOutput model_output(output[0].cast<torch::Tensor>());
    if (!output[1].is_none()) {
      model_output.aux_hidden_states = output[1].cast<torch::Tensor>();
    }
    if (output.size() == 3 && !output[2].is_none()) {
      model_output.mtp_topk_state =
          MtpTopkState::from_tensor(output[2].cast<torch::Tensor>());
    }
    return model_output;
  }
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

}  // namespace xllm
