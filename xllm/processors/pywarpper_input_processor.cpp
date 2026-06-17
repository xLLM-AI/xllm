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

#include "processors/pywarpper_input_processor.h"

#include <pybind11/embed.h>
#include <torch/extension.h>

#include "butil/synchronization/lock.h"
#include "core/common/global_flags.h"
#include "core/framework/config/model_config.h"

namespace py = pybind11;

namespace xllm {

class __attribute__((visibility("hidden"))) PyWrapperImpl {
 private:
  PyWrapperImpl() {
    py::gil_scoped_acquire gil;

    mm_ = py::module_::import("pybind.multimodal");
    preprocess_ = mm_.attr("preprocess");
  }
  ~PyWrapperImpl() {}

 public:
  static PyWrapperImpl& instance() {
    static PyWrapperImpl ins;
    return ins;
  }

  bool execute(const MMInput& inputs, MMData& data) {
    butil::AutoLock lock_guard(lock_);
    py::gil_scoped_acquire acquire;

    try {
      py::list py_lst;
      for (const auto& item : inputs.items()) {
        py_lst.append(py::bytes(item.raw_data));
      }

      py::dict res =
          preprocess_(py_lst, ::xllm::ModelConfig::get_instance().model());
      data = std::move(MMData(MMType::IMAGE, py::cast<MMDict>(res)));

      return true;
    } catch (std::exception& e) {
      LOG(ERROR) << "python call fail, exception is " << e.what();
      return false;
    }
  }

 private:
  py::module_ mm_;
  py::object preprocess_;

  butil::Lock lock_;
};

PyWarpperInputProcessor::PyWarpperInputProcessor(const ModelArgs&) {
  PyWrapperImpl::instance();
}

bool PyWarpperInputProcessor::process(const MMInput& inputs, MMData& datas) {
  return PyWrapperImpl::instance().execute(inputs, datas);
}

}  // namespace xllm
