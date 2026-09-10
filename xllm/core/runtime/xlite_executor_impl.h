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

// Drives XModel::Forward via model->get_xlite_holder() (bypasses
// model_->forward).

#pragma once

#include <glog/logging.h>
#include <xlite/xlite.h>

#include <vector>

#include "core/framework/kv_cache/kv_cache.h"
#include "core/framework/model/causal_lm.h"
#include "core/framework/model/model_input_params.h"
#include "core/framework/model/model_output.h"
#include "core/layers/xlite/xlite_attn_meta_builder.h"
#include "core/layers/xlite/xlite_causal_lm_base.h"
#include "core/layers/xlite/xlite_init_utils.h"
#include "runtime/base_executor_impl.h"
#include "runtime/options.h"

namespace xllm {

class XliteExecutorImpl : public BaseExecutorImpl {
 public:
  XliteExecutorImpl(CausalLM* model,
                    const ModelArgs& args,
                    const torch::Device& device,
                    const runtime::Options& options)
      : BaseExecutorImpl(model, args, device, options),
        model_(model),
        device_(device),
        options_(options) {
    // Sleep mode check lives here rather than XliteCausalLMBase's
    // CheckSupportedConfig because ModelContext does not carry runtime
    // Options. Both sleep paths are rejected: the RL SleepableAllocator path
    // never covers xlite weights, and the xtensor path requires the
    // lazy_load_model / free_model_weights lifecycle xlite does not
    // implement.
    CHECK(!options_.enable_sleep_mode())
        << "Sleep mode is not supported by xlite backend; disable "
           "--enable_sleep_mode or use a non-xlite backend.";
  }

  ModelOutput run(const torch::Tensor& tokens,
                  const torch::Tensor& positions,
                  std::vector<KVCache>& kv_caches,
                  const ModelInputParams& params) override;

 private:
  CausalLM* model_;
  torch::Device device_;
  runtime::Options options_;
  std::vector<std::vector<::XTensor>> kv_buf_;
};

// Header-local so the TU is linked.
REGISTER_EXECUTOR("xlite", XliteExecutorImpl);

}  // namespace xllm