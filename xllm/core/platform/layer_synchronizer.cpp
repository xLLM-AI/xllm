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

#include "platform/layer_synchronizer.h"

#if defined(USE_NPU)
#include "platform/npu/npu_layer_synchronizer.h"
#elif defined(USE_MLU)
#include "platform/mlu/mlu_layer_synchronizer.h"
#endif

namespace xllm {

#if defined(USE_NPU) || defined(USE_MLU)

#if defined(USE_NPU)
class NPULayerSynchronizerAdapter final : public LayerSynchronizer {
 public:
  explicit NPULayerSynchronizerAdapter(int64_t num_layers)
      : impl_(num_layers) {}

  bool synchronize_layer(int64_t layer_index) override {
    return impl_.synchronize_layer(layer_index);
  }

  bool record_stream(int64_t layer_index, Stream* stream) override {
    if (stream == nullptr) {
      return false;
    }
    const aclError ret = aclrtRecordEvent(*impl_.get_event(layer_index),
                                          stream->get_stream()->stream());
    if (ret != ACL_SUCCESS) {
      return false;
    }
    impl_.get_event_flag(layer_index)->store(true, std::memory_order_release);
    return true;
  }

  void abort() override { impl_.abort(); }

  uint32_t size() const override {
    return const_cast<NPULayerSynchronizerImpl&>(impl_).get_event_size();
  }

 private:
  NPULayerSynchronizerImpl impl_;
};
#elif defined(USE_MLU)
class MLULayerSynchronizerAdapter final : public LayerSynchronizer {
 public:
  explicit MLULayerSynchronizerAdapter(int64_t num_layers)
      : impl_(num_layers) {}

  bool synchronize_layer(int64_t layer_index) override {
    return impl_.synchronize_layer(layer_index);
  }

  bool record_stream(int64_t layer_index, Stream* stream) override {
    return impl_.record_stream(layer_index, stream);
  }

  void abort() override { impl_.abort(); }

  uint32_t size() const override { return impl_.get_event_size(); }

 private:
  MLULayerSynchronizerImpl impl_;
};
#endif

std::shared_ptr<LayerSynchronizer> create_layer_synchronizer(
    int64_t num_layers) {
#if defined(USE_NPU)
  return std::make_shared<NPULayerSynchronizerAdapter>(num_layers);
#else
  return std::make_shared<MLULayerSynchronizerAdapter>(num_layers);
#endif
}

#else

std::shared_ptr<LayerSynchronizer> create_layer_synchronizer(
    int64_t num_layers) {
  (void)num_layers;
  return nullptr;
}

#endif

}  // namespace xllm
