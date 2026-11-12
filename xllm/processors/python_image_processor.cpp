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

#include "processors/python_image_processor.h"

#include <glog/logging.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <string>

#include "core/framework/config/model_config.h"
#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm {

namespace {

// Delegates image preprocessing to Python (HF AutoImageProcessor via
// pybind.multimodal.preprocess_tensors) for the Python model-executor path,
// so no per-model C++ image processor is needed. Initialize the Python loader
// with the owning image processor before the service accepts its first image.
class PyImagePreprocess final {
 public:
  static PyImagePreprocess& instance() {
    static PyImagePreprocess ins;
    return ins;
  }

  bool run(const std::vector<torch::Tensor>& images,
           std::vector<MMDataItem>& output_items) {
    py::gil_scoped_acquire gil;
    try {
      // Pass the vector directly: pybind11 (pybind11/stl.h) converts
      // std::vector<torch::Tensor> to a Python list at the call site. (Avoids
      // both the null-handle default py::list pitfall and an explicit cast.)
      py::dict res = preprocess_fn_(images, model_path_);
      // HF returns batched tensors: pixel_values [total_patches, dim],
      // image_grid_thw [num_images, 3]. Split per image to match the C++
      // ImageProcessor contract (one MMDataItem per input image).
      torch::Tensor pixel_values = py::cast<torch::Tensor>(res["pixel_values"]);
      torch::Tensor grid_thw = py::cast<torch::Tensor>(res["image_grid_thw"]);
      torch::Tensor grid = grid_thw.cpu().to(torch::kLong);
      output_items.clear();
      output_items.reserve(grid.size(0));
      int64_t offset = 0;
      for (int64_t i = 0; i < grid.size(0); ++i) {
        int64_t gt = grid[i][0].item<int64_t>();
        int64_t gh = grid[i][1].item<int64_t>();
        int64_t gw = grid[i][2].item<int64_t>();
        int64_t n = gt * gh * gw;
        torch::Tensor pv_i =
            pixel_values.slice(0, offset, offset + n).contiguous();
        torch::Tensor thw_i = grid_thw[i].unsqueeze(0).contiguous();
        output_items.emplace_back(
            MMType::IMAGE,
            MMDict{{"pixel_values", pv_i}, {"image_grid_thw", thw_i}});
        offset += n;
      }
      return true;
    } catch (py::error_already_set& e) {
      LOG(ERROR) << "Python image preprocess failed: " << e.what();
      return false;
    } catch (std::exception& e) {
      LOG(ERROR) << "Python image preprocess failed: " << e.what();
      return false;
    }
  }

 private:
  PyImagePreprocess() {
    // Processor construction runs outside Python, so imports need the GIL.
    py::gil_scoped_acquire gil;
    model_path_ = ::xllm::ModelConfig::get_instance().model();
    // node0 (master+worker) has already imported the `xllm` package via the
    // Python model, so this submodule import is cheap and safe here.
    py::module_ mm = py::module_::import("xllm.pybind.multimodal");
    mm.attr("initialize_image_processor")(model_path_);
    preprocess_fn_ = mm.attr("preprocess_tensors");
  }

  ~PyImagePreprocess() { clear_python_object(preprocess_fn_); }

  std::string model_path_;
  py::object preprocess_fn_;
};

}  // namespace

void initialize_python_image_processor() {
  static_cast<void>(PyImagePreprocess::instance());
}
bool process_python_images(const std::vector<torch::Tensor>& images,
                           std::vector<MMDataItem>& output_items) {
  return PyImagePreprocess::instance().run(images, output_items);
}
}  // namespace xllm
