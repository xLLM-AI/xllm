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

#include "core/runtime/xlite_executor_impl.h"

#include <glog/logging.h>

#if defined(USE_NPU)
#include <torch_npu/csrc/core/npu/NPUStream.h>
#endif

#include "core/framework/model/model_output.h"
#include "core/layers/xlite/xlite_causal_lm_base.h"

namespace xllm {

ModelOutput XliteExecutorImpl::run(const torch::Tensor& tokens,
                                   const torch::Tensor& positions,
                                   std::vector<KVCache>& kv_caches,
                                   const ModelInputParams& params) {
  auto* holder = model_->get_xlite_holder();
  CHECK(holder != nullptr) << "xlite backend requires xlite-backed model";
  CHECK(holder->xlite_ready()) << "xlite model not initialized";

  XRuntime& rt = holder->xlite_rt();
  XModel& xlite_model = holder->xlite_model();

  // DP padding: xlite ForwardMoEDispatch/Combine use fixed-shape
  // AllGather/ReduceScatter requiring equal batchedTokens per DP rank. Empty DP
  // shards get padded here.
  const auto& dp_nums = params.parallel.dp_global_token_nums;
  int64_t real_tokens = tokens.size(0);
  int64_t max_tokens = real_tokens;
  int64_t pad_count = 0;
  bool empty_shard = false;
  if (dp_nums.size() > 1) {
    max_tokens = real_tokens;
    for (int32_t v : dp_nums) {
      if (static_cast<int64_t>(v) > max_tokens) {
        max_tokens = static_cast<int64_t>(v);
      }
    }
    // dp_rank = global_rank / attn_tp, attn_tp = world / dp_size.
    const int32_t dp_size = static_cast<int32_t>(dp_nums.size());
    const int32_t world =
        options_.world_size() > 0 ? options_.world_size() : dp_size;
    const int32_t attn_tp = dp_size > 0 ? world / dp_size : 1;
    const int32_t dp_rank =
        attn_tp > 0 ? static_cast<int32_t>(options_.server_idx()) / attn_tp : 0;
    if (dp_rank >= 0 && dp_rank < dp_size) {
      real_tokens = static_cast<int64_t>(dp_nums[dp_rank]);
    }
    if (real_tokens == 0) {
      empty_shard = true;
    }
    if (max_tokens > real_tokens) {
      pad_count = max_tokens - real_tokens;
    }
  }

  // Pad tokens/positions to max with dummy tokens (token_id=0, position=0).
  torch::Tensor run_tokens = tokens;
  torch::Tensor run_positions = positions;
  if (pad_count > 0) {
    auto opts_i = tokens.options();
    auto opts_p = positions.defined()
                      ? positions.options()
                      : torch::TensorOptions().dtype(torch::kInt32);
    torch::Tensor real_tok = real_tokens > 0 ? tokens.slice(0, 0, real_tokens)
                                             : torch::empty({0}, opts_i);
    torch::Tensor real_pos = (real_tokens > 0 && positions.defined())
                                 ? positions.slice(0, 0, real_tokens)
                                 : torch::empty({0}, opts_p);
    torch::Tensor pad_tok = torch::zeros({pad_count}, opts_i);
    torch::Tensor pad_pos = torch::zeros({pad_count}, opts_p);
    run_tokens = real_tokens > 0 ? torch::cat({real_tok, pad_tok}) : pad_tok;
    run_positions = real_tokens > 0 ? torch::cat({real_pos, pad_pos}) : pad_pos;
    run_tokens = run_tokens.contiguous();
    run_positions = run_positions.contiguous();
  }

  ::XModelAttnMeta attn_meta;
  xlite::XliteAttnMetaBuilder::Build(params,
                                     run_positions,
                                     holder->xlite_config().blockSize,
                                     attn_meta,
                                     pad_count);

  ::XTensor x_in, x_out, x_freqs;
  xlite::InitXTensor(x_in, run_tokens);
  torch::Tensor out_slice = holder->xlite_output_buf(run_tokens.size(0));
  xlite::InitXTensor(x_out, out_slice);
  xlite::InitXTensor(x_freqs, holder->xlite_freqs_cis());

  if (kv_buf_.size() != kv_caches.size()) {
    kv_buf_.resize(kv_caches.size());
    for (auto& layer_kv : kv_buf_) {
      layer_kv.resize(2);
    }
  }
  // DSA (GLM-5/5.1): kvCache[layer][2] is indexKCache; framework allocates it
  // when index_n_heads>0. MLA/normal models keep 2 entries (k/v).
  const bool is_dsa = holder->xlite_config().attnType == XMODEL_ATTN_DSA;
  const size_t kv_per_layer = is_dsa ? 3 : 2;
  for (auto& layer_kv : kv_buf_) {
    layer_kv.resize(kv_per_layer);
  }
  for (size_t i = 0; i < kv_caches.size(); ++i) {
    xlite::InitXTensor(kv_buf_[i][0], kv_caches[i].get_k_cache());
    xlite::InitXTensor(kv_buf_[i][1], kv_caches[i].get_v_cache());
    if (is_dsa) {
      xlite::InitXTensor(kv_buf_[i][2], kv_caches[i].get_index_cache());
    }
  }

  std::vector<::XTensor> no_deepstack;
  // xlite Forward takes freqsCis as a vector (CxA/MHC models use multiple sets;
  // GLM-5.x DSA uses only freqsCis[0]). Wrap the single set here.
  std::vector<::XTensor> freqs_cis = {x_freqs};
#if defined(USE_NPU)
  // Hierarchy host-cache restore (host_blocks_factor > 1) installs
  // parallel.layer_wise_load_synchronizer while per-layer H2D copies are in
  // flight; the monolithic forward consumes every layer's KV at once, so wait
  // for all load events first. synchronize_layer() no-ops when no
  // synchronizer is installed.
  for (uint32_t layer = 0; layer < holder->xlite_config().nLayers; ++layer) {
    CHECK(params.synchronize_layer(layer))
        << "Hierarchy KV cache layer load failed; layer " << layer
        << " not ready.";
  }
  aclrtStream ext = c10_npu::getCurrentNPUStream(device_.index()).stream();
  rt.EventWaitCurrStream(ext);
  xlite_model.Forward(
      rt, x_in, attn_meta, kv_buf_, no_deepstack, freqs_cis, x_out);
  rt.EventRecordCurrStream(ext);
  // PUSH-mode KV transfer: push threads spin on per-layer events that the
  // model records during forward (NPULayerSynchronizerImpl). The xlite forward
  // is monolithic with no per-layer hooks, so record every layer's event here
  // after the whole forward; each event then fires with all KV writes
  // complete. Correct, at the cost of forgoing per-layer push overlap. Once
  // xlite exposes per-layer callbacks, record there instead to restore the
  // push/forward overlap.
  if (params.parallel.layer_synchronizer != nullptr) {
    const int64_t num_layers = static_cast<int64_t>(
        params.parallel.layer_synchronizer->get_event_size());
    for (int64_t i = 0; i < num_layers; ++i) {
      CHECK(params.parallel.layer_synchronizer->record_event(
          i, static_cast<int32_t>(device_.index())));
    }
  }
#else
  LOG(FATAL) << "xlite backend requires USE_NPU";
#endif

  // Unpad: empty shard returns undefined; real shard returns first real_tokens
  // rows.
  if (empty_shard) {
    return ModelOutput();
  }
  if (pad_count > 0 && real_tokens > 0) {
    return ModelOutput(out_slice.slice(0, 0, real_tokens));
  }
  return ModelOutput(out_slice);
}

}  // namespace xllm