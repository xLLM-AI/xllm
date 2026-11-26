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

#include "models/dit/py_dit_model.h"

#include <c10/core/DeviceGuard.h>
#include <glog/logging.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <cstdint>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include "core/framework/config/dit_config.h"
#include "core/framework/model_context/dit_model_context.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/util/pybind_helper.h"
#include "models/py_model_helper.h"

namespace xllm {
namespace py = pybind11;

namespace {
int32_t group_rank(const ProcessGroup* group) {
  return group == nullptr ? 0 : group->rank();
}

int32_t group_size(const ProcessGroup* group) {
  return group == nullptr ? 1 : group->world_size();
}

py::dict build_generation_params(const DiTGenerationParams& params) {
  py::dict result;
  result["width"] = params.width;
  result["height"] = params.height;
  result["num_inference_steps"] = params.num_inference_steps;
  result["true_cfg_scale"] = params.true_cfg_scale;
  result["guidance_scale"] = params.guidance_scale;
  result["num_images_per_prompt"] = params.num_images_per_prompt;
  result["num_videos_per_prompt"] = params.num_videos_per_prompt;
  result["seed"] = params.seed;
  result["seed_is_set"] = params.seed_is_set;
  result["max_sequence_length"] = params.max_sequence_length;
  result["strength"] = params.strength;
  result["enable_cfg_renorm"] = params.enable_cfg_renorm;
  result["cfg_renorm_min"] = params.cfg_renorm_min;
  result["audio_duration_frames"] = params.audio_duration_frames;
  result["audio_steps"] = params.audio_steps;
  result["audio_guidance_method"] = params.audio_guidance_method;
  result["audio_sampling_rate"] = params.audio_sampling_rate;
  result["num_frames"] = params.num_frames;
  result["video_fps"] = params.video_fps;
  result["guidance_scale_2"] = params.guidance_scale_2;
  result["seconds"] = params.seconds;
  result["boundary_ratio"] = params.boundary_ratio;
  result["flow_shift"] = params.flow_shift;
  result["max_new_tokens"] = params.max_new_tokens;
  result["diffusion_steps"] = params.diffusion_steps;
  result["temperature"] = params.temperature;
  result["top_k"] = params.top_k;
  result["top_p"] = params.top_p;
  result["repetition_penalty"] = params.repetition_penalty;
  return result;
}
}  // namespace

PyDiTModel::PyDiTModel(const DiTModelContext& context)
    : options_(context.get_tensor_options()) {
  const auto& parallel = context.get_parallel_args();
  ProcessGroup* const tp = parallel.dit_tp_group_;
  ProcessGroup* const sp = parallel.dit_sp_group_;
  ProcessGroup* const vae = parallel.dit_vae_group_;

  ensure_embedded_python();
  const torch::DeviceGuard device_guard(device());
  py::gil_scoped_acquire gil;
  py::object pipeline_class =
      py::module_::import("xllm.python.dit.registry")
          .attr("get_pipeline_class")(context.model_type());
  pipeline_ = pipeline_class(
      c10::str(device()),
      parallel.rank() == 0,
      group_rank(tp),
      group_size(tp),
      group_rank(sp),
      group_size(sp),
      group_rank(vae),
      group_size(vae),
      py::cpp_function([tp](torch::Tensor input) {
        py::gil_scoped_release release;
        return parallel_state::reduce(input, tp);
      }),
      py::cpp_function([tp](const torch::Tensor& input) {
        py::gil_scoped_release release;
        return parallel_state::gather(input, tp, /*dim=*/-1);
      }),
      py::cpp_function([sp](const torch::Tensor& input) {
        py::gil_scoped_release release;
        return parallel_state::gather(input, sp, /*dim=*/1);
      }),
      py::cpp_function([sp](const torch::Tensor& input, bool scatter_heads) {
        py::gil_scoped_release release;
        const int32_t scatter_dim = scatter_heads ? 2 : 1;
        const int32_t gather_dim = scatter_heads ? 1 : 2;
        return parallel_state::all_to_all_4D(
            input, scatter_dim, gather_dim, /*async_ops=*/false, sp)();
      }),
      py::cpp_function([vae](const torch::Tensor& input) {
        py::gil_scoped_release release;
        if (vae == nullptr || vae->world_size() == 1) {
          return input.unsqueeze(0);
        }
        return vae->allgather_base_sync(input);
      }),
      py::arg("laser_attention_enabled") =
          DiTConfig::get_instance().dit_laser_attention_enabled());
}

PyDiTModel::~PyDiTModel() { clear_python_object(pipeline_); }

void PyDiTModel::load_model(std::unique_ptr<DiTModelLoader> loader) {
  const torch::DeviceGuard device_guard(device());
  using ComponentLoad = std::tuple<std::string, bool, bool>;
  const auto components = [this]() {
    py::gil_scoped_acquire gil;
    ensure_xllm_weight_loader_module();
    return pipeline_.attr("checkpoint_components")()
        .cast<std::vector<ComponentLoad>>();
  }();
  const std::vector<std::unique_ptr<StateDict>> no_weights;
  for (const auto& [name, required, load_weights] : components) {
    if (!loader->has_component(name)) {
      CHECK(!required) << "Missing Python DiT component " << name;
      continue;
    }
    auto component = loader->take_component_loader(name);
    const auto& state_dicts =
        load_weights ? component->get_state_dicts() : no_weights;
    py::gil_scoped_acquire gil;
    py::list shards;
    // Python consumes the borrowed StateDict views within load_component.
    for (const auto& state_dict : state_dicts) {
      shards.append(py::cast(PyStateDict(state_dict.get()),
                             py::return_value_policy::move));
    }
    pipeline_.attr("load_component")(
        name, component->model_weights_path(), shards);
  }
  LOG(INFO) << "Loaded Python DiT model " << loader->get_model_type() << " on "
            << device();
}

DiTForwardOutput PyDiTModel::forward(const DiTModelParams& params) {
  py::gil_scoped_acquire gil;
  py::dict batch;
  batch["batch_size"] = params.batch_size;
  batch["prompts"] = py::cast(params.prompts);
  batch["prompts_2"] = py::cast(params.prompts_2);
  batch["negative_prompts"] = py::cast(params.negative_prompts);
  batch["negative_prompts_2"] = py::cast(params.negative_prompts_2);
  batch["audio_prompt_text"] = params.audio_prompt_text;
  batch["generation_params"] =
      build_generation_params(params.generation_params);

  py::list tensor_sources;
  for (const NamedTensor& source : params.tensor_sources.entries()) {
    tensor_sources.append(
        py::make_tuple(source.name, source.tensor, source.parameters));
  }
  batch["tensor_sources"] = std::move(tensor_sources);

  py::list media_sources;
  for (const MediaNamedTensor& source : params.media_sources.entries()) {
    media_sources.append(py::make_tuple(
        source.name, source.modality, source.tensor, source.parameters));
  }
  batch["media_sources"] = std::move(media_sources);

  auto [tensors, audio_tensors, text_output] =
      pipeline_.attr("forward")(batch)
          .cast<std::tuple<std::vector<torch::Tensor>,
                           std::vector<torch::Tensor>,
                           std::vector<std::string>>>();
  return DiTForwardOutput{.tensors = std::move(tensors),
                          .audio_tensors = std::move(audio_tensors),
                          .text_output = std::move(text_output)};
}

}  // namespace xllm
