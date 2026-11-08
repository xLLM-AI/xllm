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

#include <ATen/record_function.h>
#include <c10/util/ScopeExit.h>
#include <gtest/gtest.h>
#include <spawn.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdlib>
#include <string_view>
#include <vector>

#include "core/platform/device.h"
#include "models/dit/transformers/transformer_minimax_h3.h"

namespace xllm {
namespace {

int32_t g_rank = 0;
int32_t g_port = 0;

class SequenceParallelProbe final : public dit::SequenceParallelMixin {
 public:
  explicit SequenceParallelProbe(ProcessGroup* group)
      : dit::SequenceParallelMixin(group) {}
  using dit::SequenceParallelMixin::sequence_parallel_forward;
};

TEST(MiniMaxH3AttentionNpuTest, PaddingAndBatchSegmentsPreserveLiveAttention) {
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, g_rank);
  Device(device).set_device();
  auto sp_group = create_process_group(g_rank,
                                       /*world_size=*/2,
                                       /*rank_size=*/2,
                                       g_port,
                                       /*trans=*/false,
                                       "127.0.0.1",
                                       "h3_attention_test",
                                       device);
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs parallel_args(g_rank, /*world_size=*/2, &tp_group);
  parallel_args.dit_tp_group_ = &tp_group;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const ModelContext context(
      parallel_args, ModelArgs(), QuantArgs(), options, /*context=*/nullptr);
  minimax_h3::H3Attention attention(/*hidden_size=*/8,
                                    /*heads=*/2,
                                    /*dim_head=*/128,
                                    /*qk_norm_eps=*/1e-5,
                                    context,
                                    sp_group.get());
  minimax_h3::H3Attention reference(/*hidden_size=*/8,
                                    /*heads=*/2,
                                    /*dim_head=*/128,
                                    /*qk_norm_eps=*/1e-5,
                                    context,
                                    /*sp_group=*/nullptr);
  // Zero Q/K make attention a mean over live V rows. Padding contamination
  // and cross-batch mixing then have a large, predictable effect.
  const StateDict weights({
      {"to_q.weight", torch::zeros({256, 8})},
      {"to_k.weight", torch::zeros({256, 8})},
      {"to_v.weight", torch::eye(8).repeat({32, 1})},
      {"to_out.0.weight",
       torch::cat({torch::eye(8), torch::zeros({8, 248})}, 1)},
      {"norm_q.weight", torch::ones({128})},
      {"norm_k.weight", torch::ones({128})},
  });
  attention->load_state_dict(weights);
  reference->load_state_dict(weights);
  attention->verify_loaded_weights("parallel.");
  reference->verify_loaded_weights("reference.");
  attention->to(device, torch::kBFloat16);
  reference->to(device, torch::kBFloat16);
  minimax_h3::H3RotaryPosEmbed rope(/*rope_freq_dim=*/16);
  SequenceParallelProbe parallel(sp_group.get());
  for (bool overlap : {false, true}) {
    DiTConfig::get_instance().dit_sp_communication_overlap(overlap);
    for (const int64_t length : {63, 64, 65}) {
      SCOPED_TRACE(::testing::Message() << "rank=" << g_rank << " length="
                                        << length << " overlap=" << overlap);
      const int64_t padded_length = length <= 64 ? 64 : 128;
      torch::Tensor positions = torch::zeros({padded_length, 3});
      positions.slice(0, 0, length).select(1, 0).copy_(torch::arange(length));
      auto rotary = rope->forward(positions);
      rotary.first = rotary.first.to(options);
      rotary.second = rotary.second.to(options);
      const torch::Tensor input =
          torch::cat(
              {torch::ones({1, length, 8}), torch::full({1, length, 8}, 5.0F)},
              0)
              .to(options);
      const torch::Tensor expected =
          reference->forward(input,
                             rotary.first.slice(0, 0, length),
                             rotary.second.slice(0, 0, length));
      const auto outputs = parallel.sequence_parallel_forward(
          {{minimax_h3::kPackedSequenceTensorName, {input, 1}}},
          [&](const dit::SequenceParallelTensorMap& inputs) {
            const torch::Tensor& local =
                inputs.at(minimax_h3::kPackedSequenceTensorName).first;
            EXPECT_EQ(local.size(1), padded_length / 2);
            return dit::SequenceParallelTensorMap{
                {minimax_h3::kPackedSequenceTensorName,
                 {attention->forward(local, rotary.first, rotary.second), 1}}};
          },
          minimax_h3::kPackedSequenceAlignment);
      const torch::Tensor actual =
          outputs.at(minimax_h3::kPackedSequenceTensorName).first;
      ASSERT_EQ(actual.sizes(), input.sizes());
      EXPECT_TRUE(torch::allclose(expected, input, 1e-2, 1e-2));
      EXPECT_TRUE(torch::allclose(actual, expected, 1e-2, 1e-2))
          << "maximum error: " << (actual - expected).abs().max().item<float>();
    }
  }
}

TEST(MiniMaxH3AttentionNpuTest, ProjectionConstraintsFailAtModelBoundary) {
  const torch::Device device(torch::kPrivateUse1, g_rank);
  Device(device).set_device();
  ModelArgs model_args;
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text("{}"));
  ASSERT_TRUE(ModelRegistry::get_model_args_loader(
      "MiniMaxH3Transformer3DModel")(reader, &model_args));
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/7, device);
  ParallelArgs args(/*rank=*/0, /*world_size=*/7, &tp_group);
  args.tp_size(7);
  args.dit_tp_group_ = &tp_group;
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const DiTModelContext video_context(args,
                                      {{"transformer", model_args}},
                                      {{"transformer", QuantArgs()}},
                                      options,
                                      DiTCacheConfig(),
                                      "MiniMax-H3");
  ASSERT_DEATH(MiniMaxH3Transformer3DModel model(video_context),
               "MiniMax-H3 video projection dimension");
  ProcessGroup audio_tp_group(/*rank=*/0, /*world_size=*/3, device);
  ParallelArgs audio_args(/*rank=*/0, /*world_size=*/3, &audio_tp_group);
  audio_args.tp_size(3);
  audio_args.dit_tp_group_ = &audio_tp_group;
  model_args.n_heads(6);
  model_args.in_channels(6);
  model_args.wan_patch_size({1, 1, 1});
  const DiTModelContext audio_context(audio_args,
                                      {{"transformer", model_args}},
                                      {{"transformer", QuantArgs()}},
                                      options,
                                      DiTCacheConfig(),
                                      "MiniMax-H3");
  ASSERT_DEATH(MiniMaxH3Transformer3DModel model(audio_context),
               "MiniMax-H3 audio projection dimension");
}

TEST(MiniMaxH3AttentionNpuTest, OutputHeadsProjectAllLocalRowsInFp32) {
  torch::NoGradGuard no_grad;
  const torch::Device device(torch::kPrivateUse1, g_rank);
  Device(device).set_device();
  auto sp_group = create_process_group(
      g_rank, 2, 2, g_port, false, "127.0.0.1", "h3_output_test", device);
  ProcessGroup tp_group(/*rank=*/0, /*world_size=*/1, device);
  ParallelArgs args(g_rank, /*world_size=*/2, &tp_group);
  args.dit_tp_group_ = &tp_group;
  args.dit_sp_group_ = sp_group.get();
  args.sp_size(2);
  ModelArgs model_args;
  JsonReader reader;
  ASSERT_TRUE(reader.parse_text(R"({
    "num_attention_heads": 2, "attention_head_dim": 128,
    "hidden_size": 8, "num_layers": 1, "num_refiner_layers": 1,
    "ffn_dim": 16, "in_channels": 4, "audio_in_channels": 2,
    "patch_size": [1, 1, 1], "text_dim": 8,
    "freq_dim": 2, "time_embed_hidden_dim": 8, "time_embed_dim": 8
  })"));
  ASSERT_TRUE(ModelRegistry::get_model_args_loader(
      "MiniMaxH3Transformer3DModel")(reader, &model_args));
  const auto options =
      torch::TensorOptions().device(device).dtype(torch::kBFloat16);
  const DiTModelContext context(args,
                                {{"transformer", model_args}},
                                {{"transformer", QuantArgs()}},
                                options,
                                DiTCacheConfig(),
                                "MiniMax-H3");
  MiniMaxH3Transformer3DModel model(context);
  for (torch::Tensor parameter : model->parameters()) {
    parameter.fill_(0.125F);
  }
  model->apply_reference_precision(device);
  const auto children = model->named_children();
  const auto* video_head =
      children["proj_out"]->as<layer::ColumnParallelLinearImpl>();
  const auto* audio_head =
      children["audio_proj_out"]->as<layer::ColumnParallelLinearImpl>();
  ASSERT_NE(video_head, nullptr);
  ASSERT_NE(audio_head, nullptr);
  // Observe the real Linear calls without adding a production test interface.
  // Identifying weights excludes other projections with the same dimensions.
  static std::array<const void*, 2> head_weights;
  static std::array<int64_t, 2> calls;
  head_weights = {video_head->weight().data_ptr(),
                  audio_head->weight().data_ptr()};
  calls = {0, 0};
  const at::RecordFunctionGuard recording;
  const at::CallbackHandle callback = at::addThreadLocalCallback(
      at::RecordFunctionCallback([](const at::RecordFunction& function)
                                     -> std::unique_ptr<at::ObserverContext> {
        const std::string_view name(function.name());
        // LibTorch's functional::linear uses addmm for biased 2-D inputs
        // and matmul otherwise; the transposed weight shares its storage.
        if (name == "aten::addmm" || name == "aten::matmul") {
          const auto inputs = function.inputs();
          const size_t input_index = name == "aten::addmm" ? 1 : 0;
          const torch::Tensor& weight = inputs[input_index + 1].toTensor();
          for (size_t head = 0; head < head_weights.size(); ++head) {
            if (weight.data_ptr() == head_weights[head]) {
              const torch::Tensor& input = inputs[input_index].toTensor();
              EXPECT_EQ(input.dim(), 2);
              EXPECT_EQ(input.size(0), 32);
              EXPECT_EQ(input.scalar_type(), torch::kFloat32);
              EXPECT_EQ(weight.scalar_type(), torch::kFloat32);
              ++calls[head];
            }
          }
        }
        return nullptr;
      }).needsInputs(true));
  const auto remove_callback =
      c10::make_scope_exit([callback] { at::removeCallback(callback); });
  const auto indices_options = options.dtype(torch::kLong);
  const torch::Tensor video_indices = torch::tensor({0, 1}, indices_options);
  const torch::Tensor audio_indices = torch::tensor({2}, indices_options);
  const torch::Tensor text_indices = torch::tensor({3, 4, 5}, indices_options);
  const torch::Tensor positions =
      torch::zeros({6, 3}, options.dtype(torch::kFloat32));
  const auto conditioning =
      model->prepare_conditioning(torch::ones({1, 3, 8}, options), positions);
  const auto output =
      model->forward(torch::ones({1, 2, 4}, options),
                     torch::ones({1, 1, 2}, options),
                     conditioning,
                     torch::ones({1}, options.dtype(torch::kFloat32)),
                     torch::zeros({6}, indices_options),
                     positions,
                     torch::zeros({6}, indices_options),
                     video_indices,
                     audio_indices,
                     text_indices);
  EXPECT_EQ(calls[0], 1);
  EXPECT_EQ(calls[1], 1);
  EXPECT_EQ(output.sample.sizes(), (torch::IntArrayRef{1, 2, 4}));
  EXPECT_EQ(output.audio_sample.sizes(), (torch::IntArrayRef{1, 1, 2}));
  EXPECT_TRUE(torch::isfinite(output.sample).all().item<bool>());
  EXPECT_TRUE(torch::isfinite(output.audio_sample).all().item<bool>());
}

}  // namespace
}  // namespace xllm

// A fresh executable prevents ranks from inheriting library worker threads.
// Each rank uses the normal shared Python/NPU test environment.
int main(int argc, char** argv) {
  const bool worker =
      argc >= 4 && std::string_view(argv[argc - 3]) == "--h3-test-worker";
  if (worker) {
    xllm::g_rank = std::atoi(argv[argc - 2]);
    xllm::g_port = std::atoi(argv[argc - 1]);
    argc -= 3;
    argv[argc] = nullptr;
  }
  // InitGoogleTest removes its flags from argv. Preserve them for each rank.
  const std::vector<char*> rank_arguments(argv, argv + argc);
  ::testing::InitGoogleTest(&argc, argv);
  if (worker || ::testing::GTEST_FLAG(list_tests)) {
    return RUN_ALL_TESTS();
  }
  xllm::g_port = 20000 + static_cast<int32_t>(getpid() % 30000);
  std::array<pid_t, 2> children;
  for (int32_t rank : {0, 1}) {
    std::string rank_arg = std::to_string(rank);
    std::string port_arg = std::to_string(xllm::g_port);
    char worker_arg[] = "--h3-test-worker";
    std::vector<char*> arguments = rank_arguments;
    arguments.push_back(worker_arg);
    arguments.push_back(rank_arg.data());
    arguments.push_back(port_arg.data());
    arguments.push_back(nullptr);
    CHECK_EQ(posix_spawn(&children[rank],
                         "/proc/self/exe",
                         nullptr,
                         nullptr,
                         arguments.data(),
                         environ),
             0)
        << "Failed to spawn H3 attention test rank";
  }
  int32_t result = EXIT_SUCCESS;
  for (pid_t child : children) {
    int status = 0;
    CHECK_EQ(waitpid(child, &status, 0), child);
    if (!WIFEXITED(status) || WEXITSTATUS(status) != EXIT_SUCCESS) {
      result = EXIT_FAILURE;
    }
  }
  return result;
}
