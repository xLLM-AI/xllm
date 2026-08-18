/* Copyright 2026 The xLLM Authors.

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

#include "core/runtime/py_executor_impl.h"

#include <Python.h>
#include <glog/logging.h>
#include <pybind11/embed.h>
#include <torch/extension.h>

#include <memory>
#include <optional>
#include <vector>

#include "common/metrics.h"
#include "core/layers/common/attention_metadata.h"
#include "core/layers/common/attention_metadata_builder.h"
#include "core/runtime/py_attention_metadata.h"
#include "models/llm/py_causal_lm.h"

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>

#include "platform/npu/npu_layer_synchronizer.h"
#endif

namespace py = pybind11;

namespace xllm {
namespace {

py::object optional_tensor(const torch::Tensor& tensor) {
  return tensor.defined() ? py::cast(tensor) : py::none();
}

void clear_python_object(py::object& object) {
  if (!object) {
    return;
  }
  if (!Py_IsInitialized()) {
    (void)object.release();
    return;
  }
  py::gil_scoped_acquire gil;
  object = py::object();
}

}  // namespace

PYBIND11_EMBEDDED_MODULE(xllm_runtime, m) {
  register_attention_metadata_views(m);

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
  py::module_::import("xllm_runtime");
  py::module_ executor_module =
      py::module_::import("xllm.python.model_executor.executor");
  py_executor_ =
      executor_module.attr("ModelExecutor")(py_causal_lm_->python_model(),
                                            py_causal_lm_->config_dict(),
                                            options_.max_seqs_per_batch());
}

PyExecutorImpl::~PyExecutorImpl() { clear_python_object(py_executor_); }

ForwardInput PyExecutorImpl::prepare_inputs(Batch& batch) {
  return batch.prepare_forward_input(
      options_.num_decoding_tokens(), 0, args_, options_.cp_size());
}

ModelOutput PyExecutorImpl::run(const torch::Tensor& tokens,
                                const torch::Tensor& positions,
                                std::vector<KVCache>& kv_caches,
                                const ModelInputParams& params) {
  torch::NoGradGuard no_grad;
  COUNTER_INC(num_model_execution_total_eager);

  // Build or reuse attention metadata.
  std::shared_ptr<layer::AttentionMetadata> attn_metadata =
      params.attn_metadata;
  if (!attn_metadata) {
    attn_metadata = std::make_shared<layer::AttentionMetadata>(
        layer::AttentionMetadataBuilder::build(
            params, enable_mla_, std::nullopt, device_));
  }

  py::gil_scoped_acquire gil;

  // Lazy bind KV caches on first call.
  int64_t num_layers = static_cast<int64_t>(kv_caches.size());
  if (!kv_bound_) {
    py::list kv_caches_py;
    for (auto& kv : kv_caches) {
      // Slot order must match ``LayerCache`` on the Python side.
      kv_caches_py.append(py::make_tuple(optional_tensor(kv.get_k_cache()),
                                         optional_tensor(kv.get_v_cache()),
                                         optional_tensor(kv.get_index_cache()),
                                         optional_tensor(kv.get_conv_cache()),
                                         optional_tensor(kv.get_ssm_cache())));
    }
    py_executor_.attr("bind_kv_caches")(kv_caches_py);
    kv_bound_ = true;
    kv_layer_count_ = num_layers;
  } else {
    CHECK_EQ(num_layers, kv_layer_count_)
        << "KV cache layer count changed after initial bind";
  }

  py::object py_metadata =
      py::cast(PyAttentionMetadataView(attn_metadata, params));
  py::object input_embedding = params.embedding.input_embedding.defined()
                                   ? py::cast(params.embedding.input_embedding)
                                   : py::none();

  // --- VLM: vision encode + embedding merge on image/video prefill steps ---
  // On steps carrying multimodal input, ``params.multimodal.mm_data`` holds the
  // batched ``pixel_values`` + ``image_grid_thw`` (still images) and/or
  // ``pixel_values_videos`` + ``video_grid_thw`` (video) — same accessors the
  // C++ Qwen3-VL base uses in qwen3_vl_base.h. Drive the Python model's
  // ``encode`` -> ``get_input_embeddings`` pipeline: the latter scatters each
  // modality's embeddings at its placeholder-token positions and sets
  // ``model._inputs_embeds`` / ``deepstack_input_embeds`` for the runner-driven
  // ``Qwen3VLModel.forward``. Decode steps carry no mm_data, so the attributes
  // stay clear and the aclgraph embed path is used.
  //
  // NOTE: this scatters the FULL image/video embedding into the current
  // forward's tokens, so it assumes every multimodal token is in this batch
  // (i.e. enable_chunked_prefill=False). Chunked prefill — where a chunk
  // boundary can land inside an item's token span — needs item-level scatter
  // (reuse EncoderEmbeddingGatherVisitor + the NPU backend's paged mixed-batch
  // attention, both tracked for a follow-up PR).
  auto& mm_data = params.multimodal.mm_data;
  if (mm_data.valid()) {
    torch::Tensor pixel_values;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values")) {
      pixel_values = res.value();
    }
    torch::Tensor image_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("image_grid_thw")) {
      image_grid_thw = res.value();
    }
    torch::Tensor pixel_values_videos;
    if (const auto& res = mm_data.get<torch::Tensor>("pixel_values_videos")) {
      pixel_values_videos = res.value();
    }
    torch::Tensor video_grid_thw;
    if (const auto& res = mm_data.get<torch::Tensor>("video_grid_thw")) {
      video_grid_thw = res.value();
    }

    if (pixel_values.defined() || pixel_values_videos.defined()) {
      py::object top_model = py_causal_lm_->python_model();
      // encode() moves the tensors onto device internally.
      py::object image_embeds = py::none();
      if (pixel_values.defined() && image_grid_thw.defined()) {
        image_embeds = top_model.attr("encode")(pixel_values, image_grid_thw);
      }
      py::object video_embeds = py::none();
      if (pixel_values_videos.defined() && video_grid_thw.defined()) {
        video_embeds =
            top_model.attr("encode")(pixel_values_videos, video_grid_thw);
      }
      // Sets top_model.model._inputs_embeds + deepstack_input_embeds.
      top_model.attr("get_input_embeddings")(
          tokens, image_embeds, video_embeds);
    }
  }

  // --- mRoPE: collapse [3, N] decode positions to 1-D ---
  // Only PURE decode collapses to 1-D: decode rows are identical
  // (batch_input_builder get_mrope_positions), and mRoPE(p,p,p) == standard
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

  // Execute: one C++ -> Python call per step. input_embedding stays None for
  // the Qwen3-VL python path (embeddings are merged via the attribute set by
  // get_input_embeddings above), so the runner takes the 2-arg model() branch
  // and Qwen3VLModel.forward reads _inputs_embeds. positions_arg carries the
  // mRoPE [3,N]->1-D decode collapse.
  py::object hidden_obj = py_executor_.attr("execute")(
      tokens, positions_arg, py_metadata, input_embedding, py_sync);
  return ModelOutput(hidden_obj.cast<torch::Tensor>());
}

}  // namespace xllm
