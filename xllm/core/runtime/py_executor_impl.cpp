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
#include "core/framework/multimodal/multimodal_utils.h"
#include "core/framework/speculative/spec_verify.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/platform/platform.h"
#include "core/runtime/py_attention_metadata.h"
#include "core/util/pybind_helper.h"
#include "models/llm/py_causal_lm.h"
#include "models/py_model_helper.h"
#include "platform/layer_synchronizer.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

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
      device_(device),
      options_(options),
      enable_mla_(args.enable_mla()) {
  CHECK(py_causal_lm_ != nullptr) << "PyExecutorImpl requires PyCausalLM";
  py::gil_scoped_acquire gil;
  ensure_xllm_runtime_module();
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  // These values belong to the executor, not the checkpoint. In particular,
  // the MTP draft has different speculative options from its target.
  py::dict executor_config = py_causal_lm_->config_dict().attr("copy")();
  num_speculative_tokens_ =
      executor_config["num_speculative_tokens"].cast<int32_t>();
  executor_config["enable_disagg_pd"] = options_.enable_disagg_pd();
  executor_config["instance_role"] = options_.instance_role().to_string();
  executor_config["task_type"] = options_.task_type();
  executor_config["max_encoder_cache_size"] = options_.max_encoder_cache_size();
  executor_config["num_speculative_tokens"] = options_.num_speculative_tokens();
  executor_config["enable_chunked_prefill"] = options_.enable_chunked_prefill();
  executor_config["enable_prefix_cache"] =
      KVCacheConfig::get_instance().enable_prefix_cache();
  executor_config["enable_schedule_overlap"] =
      options_.enable_schedule_overlap();
  executor_config["is_draft_model"] = options_.is_draft_model();
  executor_config["is_mtp_algorithm"] =
      SpeculativeConfig::is_mtp_algorithm(options_.speculative_algorithm());
  executor_config["is_block_diffusion_algorithm"] =
      SpeculativeConfig::is_block_diffusion_algorithm(
          options_.speculative_algorithm());
  executor_config["requires_aux_hidden_capture"] =
      SpeculativeConfig::requires_aux_hidden_capture(
          options_.speculative_algorithm());
  executor_config["block_draft_graph_warmup_max_kv_len"] =
      ExecutionConfig::get_instance().block_draft_graph_warmup_max_kv_len();
  executor_config["enable_graph_mode_decode_no_padding"] =
      ExecutionConfig::get_instance().enable_graph_mode_decode_no_padding();
  // DFlash2 defaults to eager in the model config. Enable its independent
  // full-block ACL runner when the worker requests graph execution.
  const bool independent_block_graph =
      Platform::is_npu() && is_dflash2_draft_model_type(args_.model_type()) &&
      options_.enable_graph() && options_.num_speculative_tokens() > 0;
  if (independent_block_graph) {
    executor_config["enable_graph"] = true;
    executor_config["python_graph_backend"] = "aclgraph";
  }
  py_executor_ = executor_module.attr("ModelExecutor")(
      py_causal_lm_->python_model(),
      executor_config,
      options_.max_seqs_per_batch(),
      ExecutionConfig::get_instance().acl_graph_decode_batch_size_limit());
  py_causal_lm_->set_context_kv_graph_runner(
      py_executor_.attr("context_kv_graph_runner"));
  requires_input_batch_metadata_ =
      py_executor_.attr("requires_input_batch_metadata").cast<bool>();
  supports_layerwise_kv_load_ =
      py_executor_.attr("supports_layerwise_kv_load").cast<bool>();
}

PyExecutorImpl::~PyExecutorImpl() {
  if (PyCausalLM::active_instance() == py_causal_lm_) {
    PyCausalLM::set_active_instance(nullptr);
  }
  clear_python_object(py_executor_);
}

void PyExecutorImpl::bind_kv_caches(std::vector<KVCache>& kv_caches) {
  const int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py_executor_.attr("bind_kv_caches")(build_python_kv_caches(kv_caches));
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }
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
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }
  const bool layerwise_kv_load =
      params.parallel.layer_wise_load_synchronizer != nullptr &&
      supports_layerwise_kv_load_ && !params.is_spec_verify &&
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
  if (!enable_mla_ && py_causal_lm_->cp_size() == 1 &&
      py_causal_lm_->kv_split_size() > 1) {
    const KVShardLayout layout(options_.block_size(),
                               py_causal_lm_->kv_split_size(),
                               py_causal_lm_->kv_split_rank());
    attn_metadata->kv_shard_batch_metadata =
        layer::build_kv_shard_batch_metadata(*attn_metadata, layout);
  }
  py::gil_scoped_acquire gil;

  bind_kv_caches(kv_caches);

  py::object py_metadata = py::cast(
      PyAttentionMetadataView(attn_metadata, params, num_speculative_tokens_));
  py::object py_input_batch_metadata =
      requires_input_batch_metadata_
          ? py::cast(ExecutionBatchMetadata(params.meta.execution_batch))
          : py::none();
  COUNTER_INC(num_model_execution_total_eager);

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
    // Models whose image encoder needs every gathered multimodal key (e.g.
    // DeepSeek-OCR's separate global-view and crop tensors) expose
    // encode_with_inputs; they take over the whole batch dict and the
    // per-modality 2-argument encode branches below are skipped. The
    // pixel_values guard keeps chunked prefill safe: later chunks see their
    // image items already embedded, EncoderInputGatherVisitor skips them,
    // and the gathered dict is empty — encoding is skipped, and
    // EncoderEmbeddingGatherVisitor below re-slices the per-item embeddings
    // stored by the first chunk.
    if (py::hasattr(top_model, "encode_with_inputs") &&
        pixel_values.defined()) {
      py::dict inputs;
      for (const auto& [key, value] : mm_data.data()) {
        inputs[py::str(key)] = py::cast(value);
      }
      torch::Tensor raw =
          top_model.attr("encode_with_inputs")(inputs).cast<torch::Tensor>();
      encoded[get_embedding_key(MMType::IMAGE)] =
          split_by_token_nums(raw, get_mm_token_nums(mm_data, MMType::IMAGE));
    } else if (pixel_values.defined() && image_grid_thw.defined()) {
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
      py::object visual_embedding = top_model.attr("get_input_embeddings")(
          execution_tokens, image_embeds, video_embeds);
      if (args_.model_type() == "qwen3_5" ||
          args_.model_type() == "qwen3_5_moe") {
        // Qwen3.5 passes chunk-local visual rows explicitly; no mutable
        // embedding state can survive into subsequent decode steps.
        input_embedding = std::move(visual_embedding);
      }
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

  // Qwen3-VL retains its DeepStack merge state and two-argument decoder.
  // Qwen3.5 visual prefill instead passes its embedding as the third argument.
  // Pure decode has no visual embedding and uses the captured token path.
  py::kwargs graph_kwargs;
  if (layerwise_kv_load) {
    graph_kwargs["kv_load_synchronizer"] =
        py::cast(params.parallel.layer_wise_load_synchronizer);
    graph_kwargs["kv_load_layers_per_event"] = params.parallel.layers_per_event;
  }
  py::object hidden_obj =
      py_executor_.attr("execute")(execution_tokens,
                                   positions_arg,
                                   py_metadata,
                                   input_embedding,
                                   py_sync,
                                   topk_indices,
                                   /*enable_graph=*/false,
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
