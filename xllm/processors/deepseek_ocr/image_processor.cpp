/* Copyright 2025-2026 The xLLM Authors.

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

#include "processors/deepseek_ocr/image_processor.h"

#include <pybind11/pybind11.h>
#include <torch/python.h>  // registers the torch::Tensor pybind caster

#include <string>

#include "core/util/pybind_helper.h"

namespace py = pybind11;

namespace xllm {

namespace {

// Resolves xllm.python.models.deepseek_ocr.preprocess_images once and caches
// the callable. The processor runs on the service request path, so the GIL is
// taken per call (and released between requests).
class DeepseekOCRPreprocessFn final {
 public:
  static DeepseekOCRPreprocessFn& instance() {
    static DeepseekOCRPreprocessFn ins;
    return ins;
  }

  py::object fn() const { return fn_; }

 private:
  DeepseekOCRPreprocessFn() {
    py::gil_scoped_acquire gil;
    py::module_ module = py::module_::import("xllm.python.models.deepseek_ocr");
    fn_ = module.attr("preprocess_images");
  }

  ~DeepseekOCRPreprocessFn() { clear_python_object(fn_); }

  py::object fn_;
};

}  // namespace

DeepseekOCRImageProcessor::DeepseekOCRImageProcessor(const ModelArgs& args)
    : base_size_(static_cast<int32_t>(args.mm_image_base_size())),
      crop_size_(static_cast<int32_t>(args.mm_image_size())) {}

bool DeepseekOCRImageProcessor::process(
    const std::vector<torch::Tensor>& images,
    std::vector<MMDataItem>& output_items) const {
  // Images are untrusted request input, so failure is a per-request error
  // rather than a crash. The GIL is held across the whole call (including
  // the handler) so error_already_set::what() never runs lock-free.
  py::gil_scoped_acquire gil;
  try {
    py::object results = DeepseekOCRPreprocessFn::instance().fn()(
        images, /*base_size=*/base_size_, /*crop_size=*/crop_size_);
    output_items.clear();
    output_items.reserve(images.size());
    const size_t num_images = py::len(results);
    for (size_t i = 0; i < num_images; ++i) {
      py::dict item = py::reinterpret_borrow<py::dict>(results[py::int_(i)]);
      MMDict dict;
      for (auto entry : item) {
        const std::string key = entry.first.cast<std::string>();
        dict[key] = entry.second.cast<torch::Tensor>();
      }
      output_items.emplace_back(MMType::IMAGE, std::move(dict));
    }
    return true;
  } catch (const std::exception& e) {
    LOG(ERROR) << "DeepseekOCR image preprocess failed: " << e.what();
    return false;
  }
}

}  // namespace xllm
