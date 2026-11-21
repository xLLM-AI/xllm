/* Copyright 2026 The xLLM Authors. SPDX-License-Identifier: Apache-2.0 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <torch/python.h>

#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "core/framework/model/vlm_model_params.h"
#include "core/framework/model_context/model_context.h"
#include "core/framework/parallel_state/process_group.h"
#include "core/util/json_reader.h"
#include "models/vla/gr00t_backbone.h"

namespace xllm {
namespace py = pybind11;
namespace {

class NativeGr00tBackbone final {
 public:
  explicit NativeGr00tBackbone(const std::string& config_json,
                               const std::string& device,
                               const std::string& dtype)
      : device_(device), process_group_(/*rank=*/0, /*world_size=*/1, device_) {
    JsonReader reader;
    CHECK(reader.parse_text(config_json)) << "Invalid backbone config JSON";
    CHECK(kQwen3VLModelArgsLoader(reader, &args_));
    args_.image_token_id() =
        reader.value_or<int32_t>("image_token_id", args_.image_token_id());
    args_.video_token_id() =
        reader.value_or<int32_t>("video_token_id", args_.video_token_id());
    args_.vision_start_token_id() = reader.value_or<int32_t>(
        "vision_start_token_id", args_.vision_start_token_id());
    args_.vision_end_token_id() = reader.value_or<int32_t>(
        "vision_end_token_id", args_.vision_end_token_id());
    if (reader.contains("text_config.rope_parameters.rope_theta")) {
      args_.rope_theta() = reader.value_or<float>(
          "text_config.rope_parameters.rope_theta", args_.rope_theta());
    }
    if (reader.contains("text_config.rope_parameters.mrope_section")) {
      args_.rope_scaling_mrope_section() =
          reader.value_or<std::vector<int64_t>>(
              "text_config.rope_parameters.mrope_section",
              std::vector<int64_t>{});
    }
    CHECK_GT(args_.n_layers(), 0);
    CHECK(device_.is_cuda()) << "GR00T native backbone requires a DCU device";
    CHECK(dtype == "bfloat16" || dtype == "float16" || dtype == "float32");
    const torch::ScalarType scalar_type =
        dtype == "bfloat16"
            ? torch::kBFloat16
            : (dtype == "float16" ? torch::kFloat16 : torch::kFloat32);
    options_ = torch::TensorOptions().device(device_).dtype(scalar_type);
    ParallelArgs parallel_args(/*rank=*/0, /*world_size=*/1, &process_group_);
    parallel_args.tp_group_ = &process_group_;
    parallel_args.single_rank_group_ = &process_group_;
    ModelContext context(parallel_args, args_, QuantArgs(), options_);
    backbone_ = Gr00tBackbone(context);
    backbone_->eval();
  }

  void load_weights(std::unordered_map<std::string, torch::Tensor> weights) {
    torch::NoGradGuard guard;
    backbone_->load_state_dict(StateDict(std::move(weights)));
  }

  torch::Tensor encode(const torch::Tensor& pixels, const torch::Tensor& grid) {
    torch::NoGradGuard guard;
    return backbone_->encode(pixels, grid);
  }

  torch::Tensor embed_tokens(const torch::Tensor& tokens) {
    torch::NoGradGuard guard;
    return backbone_->embed_tokens(tokens);
  }

  torch::Tensor positions(const std::vector<int32_t>& tokens,
                          const torch::Tensor& image_grid) {
    MMData data;
    data.add(MMType::IMAGE, "image_grid_thw", image_grid.cpu());
    Qwen3VLMPositionGenerator generator;
    auto [positions, delta] = generator.generate(tokens, data, args_);
    return positions.to(device_);
  }

  torch::Tensor forward(const torch::Tensor& tokens,
                        const torch::Tensor& positions,
                        const torch::Tensor& embeddings,
                        std::vector<torch::Tensor> deep_stacks) {
    torch::NoGradGuard guard;
    CHECK_EQ(tokens.dim(), 1);
    CHECK_GT(tokens.numel(), 0);
    CHECK_EQ(embeddings.size(0), tokens.numel());
    CHECK_EQ(embeddings.size(1), args_.hidden_size());
    CHECK_EQ(positions.dim(), 2);
    CHECK_EQ(positions.size(0), 3);
    CHECK_EQ(positions.size(1), tokens.numel());
    const int64_t length = tokens.numel();
    VlmModelParams params;
    params.embedding.input_embedding = embeddings;
    params.multimodal.deep_stacks = std::move(deep_stacks);
    params.meta.num_sequences = 1;
    params.meta.q_max_seq_len = length;
    params.meta.kv_max_seq_len = length;
    params.attn_metadata = std::make_shared<layer::AttentionMetadata>();
    auto& metadata = *params.attn_metadata;
    metadata.is_prefill = true;
    metadata.is_chunked_prefill = false;
    metadata.is_dummy = false;
    metadata.prefill_without_cache = true;
    metadata.use_dense_flash_attention = true;
    metadata.max_query_len = length;
    metadata.max_seq_len = length;
    metadata.total_kv_len = length;
    metadata.q_cu_seq_lens = torch::tensor({static_cast<int64_t>(0), length},
                                           options_.dtype(torch::kInt32));
    metadata.kv_cu_seq_lens = metadata.q_cu_seq_lens;
    std::vector<KVCache> caches(args_.n_layers());
    ModelParamsView view(params);
    return backbone_->forward(tokens, positions, caches, view).hidden_states;
  }

 private:
  torch::Device device_;
  ProcessGroup process_group_;
  torch::TensorOptions options_;
  ModelArgs args_;
  Gr00tBackbone backbone_{nullptr};
};

}  // namespace

void register_gr00t_bindings(py::module_& module) {
  py::class_<NativeGr00tBackbone>(module, "Gr00tBackbone")
      .def(py::init<const std::string&,
                    const std::string&,
                    const std::string&>())
      .def("load_weights", &NativeGr00tBackbone::load_weights)
      .def("encode", &NativeGr00tBackbone::encode)
      .def("embed_tokens", &NativeGr00tBackbone::embed_tokens)
      .def("positions", &NativeGr00tBackbone::positions)
      .def("forward", &NativeGr00tBackbone::forward);
}

}  // namespace xllm
