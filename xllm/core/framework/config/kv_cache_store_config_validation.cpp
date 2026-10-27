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

#include "core/framework/config/kv_cache_store_config_validation.h"

#include <glog/logging.h>

#include "core/framework/config/kv_cache_config.h"
#include "core/framework/config/kv_cache_store_config.h"
#include "core/framework/config/model_config.h"
#include "core/framework/config/speculative_config.h"
#include "core/framework/model/model_args.h"
#include "core/framework/model/model_type.h"
#include "core/framework/model_loader/hf_model_args.h"
#include "core/framework/parallel_state/parallel_args.h"
#include "core/platform/platform.h"
#include "core/runtime/options.h"

namespace xllm {

void validate_kv_cache_store_config(const std::string& model_type,
                                    const ModelConfig& model_config,
                                    const KVCacheConfig& kv_cache_config,
                                    const KVCacheStoreConfig& store_config) {
  if (!model_type.empty()) {
    ModelArgs model_args;
    CHECK(load_hf_model_args(
        model_config.model(), model_config.backend(), &model_args))
        << "Failed to load model args from " << model_config.model();
    if (model_args.index_kpool_compress()) {
      CHECK_LE(store_config.host_blocks_factor(), 1.0)
          << "Compressed KPool host offload requires request-state scheduler "
             "support.";
      CHECK(!store_config.enable_kvcache_store())
          << "Compressed KPool external storage is not supported yet.";
    }
  }

  if (kv_cache_config.enable_vmm_kvcache() &&
      store_config.host_blocks_factor() > 1.0) {
    LOG(FATAL) << "Virtual memory does not support host KV cache offload; set "
                  "--host_blocks_factor=0.";
  }

  if (store_config.enable_kvcache_store()) {
    CHECK(kv_cache_config.enable_prefix_cache())
        << "KV cache Store requires --enable_prefix_cache=true.";
    CHECK_GT(store_config.host_blocks_factor(), 1.0)
        << "KV cache Store requires --host_blocks_factor > 1.";
  }
}

void validate_shared_kv_cache_config(const ModelArgs& args,
                                     const runtime::Options& options,
                                     const ParallelArgs& parallel_args) {
  const KVCacheConfig& kv_cache_config = KVCacheConfig::get_instance();
  if (!kv_cache_config.enable_shared_kv_cache()) {
    return;
  }
  const std::string& model_type = args.model_type();
  const std::string& model_impl = ModelConfig::get_instance().model_impl();
  const bool python_model = ModelConfig::is_python_model_impl(model_impl);
  const bool python_vlm =
      python_model && options.backend() == "vlm" && model_type == "qwen3_5";
  CHECK(Platform::is_npu() && (options.backend() == "llm" || python_vlm) &&
        options.task_type() == "generate" &&
        (python_model || model_impl.empty() || model_impl == "native") &&
        options.extension_impl().empty())
      << "Shared KV cache requires a native or Python NPU Torch LLM, or a "
         "Python Qwen3.5 VLM.";
  CHECK(!python_vlm || parallel_args.dp_size() == 1)
      << "Shared Python Qwen3.5 VLM requires a single data-parallel rank.";
  CHECK(!options.enable_prefill_piecewise_graph())
      << "Shared KV cache does not support piecewise graphs.";
  if (python_model) {
    CHECK_EQ(options.npu_kernel_backend(), "TORCH");
    const bool mtp_draft =
        options.is_draft_model() &&
        (model_type == "qwen3_5_mtp" || model_type == "qwen3_5_moe_mtp");
    CHECK(model_type == "qwen3_next" ||
          is_qwen3_5_target_model_type(model_type) || mtp_draft)
        << "Shared Python KV cache requires a supported Qwen hybrid target "
           "or full-attention MTP draft.";
    const bool speculative = options.enable_speculative_decode() ||
                             options.num_speculative_tokens() > 0 ||
                             options.is_draft_model();
    CHECK(
        !speculative ||
        (SpeculativeConfig::is_mtp_algorithm(options.speculative_algorithm()) &&
         (is_qwen3_5_target_model_type(model_type) || mtp_draft)))
        << "Shared Python KV cache supports Qwen3.5 MTP only.";
    CHECK(!options.is_draft_model() ||
          (mtp_draft && !has_linear_attention_layers(args)))
        << "Shared Python KV cache requires a full-attention-only MTP draft.";
  } else {
    CHECK(model_type == "qwen3_next" || model_type == "qwen3_5_text" ||
          model_type == "qwen3_5_moe_text")
        << "Shared native KV cache requires a supported Qwen hybrid model.";
    CHECK(!options.enable_speculative_decode() &&
          options.num_speculative_tokens() == 0 && !options.is_draft_model() &&
          !options.enable_graph())
        << "Shared native KV cache requires ordinary eager decoding.";
  }
  CHECK(!kv_cache_config.enable_vmm_kvcache() && !options.enable_disagg_pd() &&
        options.instance_role() == InstanceRole::DEFAULT &&
        options.host_blocks_factor() == 0.0 && !options.enable_kvcache_store())
      << "Shared KV cache requires local static device storage.";
  CHECK(parallel_args.layerwise_split_size() == 1 &&
        parallel_args.cp_size() == 1 &&
        parallel_args.kv_split_size_effective() == 1)
      << "Shared KV cache does not support layerwise, context or KV splitting.";
  CHECK(args.index_n_heads() == 0 && !args.enable_mla() &&
        !options.enable_mla() && options.kv_cache_dtype() == "auto" &&
        kv_cache_config.indexer_cache_dtype() != "int8")
      << "Shared KV cache requires unquantized ordinary Full and GDN caches.";
}

}  // namespace xllm
