/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

// FusedMoE All2All path unit tests
// Tests for the DeepEP communication mode with multi-device setup

#include <glog/logging.h>
#include <gtest/gtest.h>
#include <sys/wait.h>
#include <torch/torch.h>
#include <unistd.h>

#include <cmath>
#include <cstring>
#include <functional>
#include <memory>
#include <vector>

#include "core/framework/config/eplb_config.h"
#include "framework/model/model_args.h"
#include "framework/parallel_state/parallel_args.h"
#include "framework/quant_args.h"
#include "framework/state_dict/state_dict.h"
#include "layers/mlu/fused_moe.h"
#include "layers/mlu/tests_utils.h"
#include "platform/device.h"
#include "util/tensor_helper.h"

#if defined(USE_MLU)
#include "framework/parallel_state/mlu_process_group.h"
#elif defined(USE_CUDA)
#include "framework/parallel_state/cuda_process_group.h"
#endif

namespace xllm {
namespace layer {
namespace test {

// Special exit code definition for skipping test
constexpr int32_t EXIT_CODE_SKIP = 77;

// Helper function to create ProcessGroup
std::unique_ptr<xllm::ProcessGroup> create_test_process_group(
    int32_t rank,
    int32_t world_size,
    int32_t port,
    const std::string& host,
    const torch::Device& device) {
  return xllm::create_process_group(rank,
                                    world_size,
                                    world_size,
                                    port,
                                    false,
                                    host,
                                    "fused_moe_all2all_test_group",
                                    device);
}

struct All2AllTestParams {
  int32_t rank;
  int32_t world_size;
  int32_t port;
  std::string host;
  int32_t device_index;
  // Model parameters
  int64_t hidden_size;
  int64_t intermediate_size;
  int64_t num_experts;
  int64_t top_k;
  int64_t batch_size;
  int64_t seq_len;
  bool is_smoothquant;
  int64_t moe_weight_bits = 8;
  int64_t group_size = 0;
  // Expected output stats (used when perform_precise_validation is true)
  double expected_min = 0.0;
  double expected_max = 0.0;
  double expected_sum = 0.0;
  // When true, validate output against expected_min/max/sum (avoids fragile
  // "any non-zero expected" heuristic; allows validation when all are 0.0).
  bool perform_precise_validation = false;
};

// Helper to create model args
ModelArgs create_model_args(const All2AllTestParams& params) {
  ModelArgs args;
  args.n_routed_experts() = static_cast<int32_t>(params.num_experts);
  args.num_experts_per_tok() = static_cast<int32_t>(params.top_k);
  args.n_group() = 1;
  args.topk_group() = static_cast<int32_t>(params.top_k);
  args.routed_scaling_factor() = 1.0f;
  args.hidden_size() = params.hidden_size;
  args.moe_intermediate_size() = static_cast<int32_t>(params.intermediate_size);
  args.n_shared_experts() = 0;
  args.norm_topk_prob() = true;
  args.hidden_act() = "silu";
  args.scoring_func() = "softmax";
  args.topk_method() = "greedy";
  return args;
}

// Helper to create quant args
QuantArgs create_quant_args(const All2AllTestParams& params) {
  QuantArgs args;
  if (params.is_smoothquant) {
    args.quant_method() = "smoothquant";
    args.bits() = 8;
    args.activation_dynamic() = true;
    args.moe_weight_bits() = params.moe_weight_bits;
    args.group_size() = params.group_size;
  }
  return args;
}

// Helper to create test weights for All2All MoE using seeded tensors
std::unordered_map<std::string, torch::Tensor> create_all2all_test_weights(
    int64_t num_experts,
    int64_t hidden_size,
    int64_t intermediate_size,
    bool is_smoothquant,
    int64_t moe_weight_bits,
    int64_t group_size,
    int64_t world_size,
    const torch::Device& device) {
  std::unordered_map<std::string, torch::Tensor> weight_dict;

  for (size_t expert_id = 0; expert_id < num_experts; ++expert_id) {
    std::string expert_prefix = "experts." + std::to_string(expert_id) + ".";
    std::string seed_prefix =
        "fused_moe_all2all_tests.expert_" + std::to_string(expert_id);

    if (is_smoothquant && moe_weight_bits == 4) {
      CHECK_GT(group_size, 0);
      CHECK_EQ(intermediate_size % world_size, 0);
      const int64_t local_intermediate_size = intermediate_size / world_size;
      test::append_w4a8_expert_weights(weight_dict,
                                       expert_prefix,
                                       seed_prefix,
                                       hidden_size,
                                       intermediate_size,
                                       intermediate_size,
                                       local_intermediate_size,
                                       group_size,
                                       device);
    } else if (is_smoothquant) {
      // Create quantized weights using seeded tensors
      auto gate_weight_fp =
          test::seeded_tensor(seed_prefix + ".gate_proj",
                              {intermediate_size, hidden_size},
                              torch::kBFloat16,
                              device);
      auto gate_qweight = gate_weight_fp.to(torch::kInt8);
      auto gate_scale = test::seeded_tensor(seed_prefix + ".gate_proj.scale",
                                            {intermediate_size},
                                            torch::kFloat32,
                                            device);
      auto gate_smooth = test::seeded_tensor(seed_prefix + ".gate_proj.smooth",
                                             {hidden_size},
                                             torch::kFloat32,
                                             device);

      auto up_weight_fp = test::seeded_tensor(seed_prefix + ".up_proj",
                                              {intermediate_size, hidden_size},
                                              torch::kBFloat16,
                                              device);
      auto up_qweight = up_weight_fp.to(torch::kInt8);
      auto up_scale = test::seeded_tensor(seed_prefix + ".up_proj.scale",
                                          {intermediate_size},
                                          torch::kFloat32,
                                          device);

      auto down_weight_fp =
          test::seeded_tensor(seed_prefix + ".down_proj",
                              {hidden_size, intermediate_size},
                              torch::kBFloat16,
                              device);
      auto down_qweight = down_weight_fp.to(torch::kInt8);
      auto down_scale = test::seeded_tensor(seed_prefix + ".down_proj.scale",
                                            {hidden_size},
                                            torch::kFloat32,
                                            device);
      auto down_smooth = test::seeded_tensor(seed_prefix + ".down_proj.smooth",
                                             {intermediate_size},
                                             torch::kFloat32,
                                             device);

      weight_dict[expert_prefix + "gate_proj.qweight"] = gate_qweight;
      weight_dict[expert_prefix + "gate_proj.per_channel_scale"] = gate_scale;
      weight_dict[expert_prefix + "gate_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "up_proj.qweight"] = up_qweight;
      weight_dict[expert_prefix + "up_proj.per_channel_scale"] = up_scale;
      weight_dict[expert_prefix + "up_proj.smooth"] = gate_smooth;

      weight_dict[expert_prefix + "down_proj.qweight"] = down_qweight;
      weight_dict[expert_prefix + "down_proj.per_channel_scale"] = down_scale;
      weight_dict[expert_prefix + "down_proj.smooth"] = down_smooth;
    } else {
      // Create BF16 weights using seeded tensors
      auto gate_weight = test::seeded_tensor(seed_prefix + ".gate_proj.weight",
                                             {intermediate_size, hidden_size},
                                             torch::kBFloat16,
                                             device);
      auto up_weight = test::seeded_tensor(seed_prefix + ".up_proj.weight",
                                           {intermediate_size, hidden_size},
                                           torch::kBFloat16,
                                           device);
      auto down_weight = test::seeded_tensor(seed_prefix + ".down_proj.weight",
                                             {hidden_size, intermediate_size},
                                             torch::kBFloat16,
                                             device);

      weight_dict[expert_prefix + "gate_proj.weight"] = gate_weight;
      weight_dict[expert_prefix + "up_proj.weight"] = up_weight;
      weight_dict[expert_prefix + "down_proj.weight"] = down_weight;
    }
  }

  // Gate weights (router)
  auto gate_weight = test::seeded_tensor("fused_moe_all2all_tests.gate.weight",
                                         {num_experts, hidden_size},
                                         torch::kBFloat16,
                                         device);
  weight_dict["gate.weight"] = gate_weight;

  return weight_dict;
}

// Child process test function for basic All2All test
int32_t run_all2all_basic_test_child(All2AllTestParams params) {
  try {
    // 0. Set expert_parallel_degree to enable DeepEP
    // This is required for All2All path to work
    EPLBConfig::get_instance().expert_parallel_degree(2);

    // 1. Check devices
    int32_t dev_count = xllm::Device::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    // 2. Set device
    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    // 3. Create ProcessGroup
    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);

    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    // 4. Create ParallelArgs with EP mode
    ParallelArgs parallel_args(
        params.rank, params.world_size, process_group.get());
    parallel_args.moe_ep_group_ = process_group.get();
    parallel_args.ep_size_ = params.world_size;
    parallel_args.moe_tp_group_ = process_group.get();

    // 5. Create tensor options
    // Note: smoothquant mode still uses BF16 for input/output, with int8 for
    // internal quantized computation. Using float32 would cause CNNL GroupGemm
    // to fail with "Data type mismatch" error.
    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);

    // 6. Create model args and quant args
    ModelArgs model_args = create_model_args(params);
    QuantArgs quant_args = create_quant_args(params);

    // 7. Create FusedMoE
    FusedMoE fused_moe(FusedMoEImpl(model_args,
                                    FusedMoEArgs{.is_gated = true},
                                    quant_args,
                                    parallel_args,
                                    options));

    // 8. Create and load weights using seeded tensors
    auto weight_dict = create_all2all_test_weights(params.num_experts,
                                                   params.hidden_size,
                                                   params.intermediate_size,
                                                   params.is_smoothquant,
                                                   params.moe_weight_bits,
                                                   params.group_size,
                                                   params.world_size,
                                                   device);
    StateDict state_dict(weight_dict);
    fused_moe->load_state_dict(state_dict);

    LOG(INFO) << "Rank " << params.rank << ": FusedMoE created and loaded";

    // 9. Create input tensor using seeded tensor for determinism
    int64_t num_tokens = params.batch_size * params.seq_len;
    std::string seed_prefix = params.is_smoothquant
                                  ? "fused_moe_all2all_tests.smoothquant"
                                  : "fused_moe_all2all_tests.basic";
    auto hidden_states = test::seeded_tensor(seed_prefix + ".hidden_states",
                                             {num_tokens, params.hidden_size},
                                             torch::kBFloat16,
                                             device);

    // 10. Run forward with All2All enabled
    auto output =
        fused_moe->forward_experts(hidden_states,
                                   /*enable_all2all_communication=*/true);

    // 11. Verify output
    xllm_device.synchronize_default_stream();

    CHECK_EQ(output.sizes().size(), 2) << "Output should be 2D tensor";
    CHECK_EQ(output.size(0), num_tokens) << "Token count should match";
    CHECK_EQ(output.size(1), params.hidden_size) << "Hidden size should match";

    // Compute and log output stats for setting expected values
    auto flat_output = output.flatten().to(torch::kFloat32).cpu();
    double actual_min = torch::min(flat_output).item<double>();
    double actual_max = torch::max(flat_output).item<double>();
    double actual_sum = torch::sum(flat_output).item<double>();

    LOG(INFO) << "Rank " << params.rank << ": Output stats - "
              << "min=" << actual_min << ", max=" << actual_max
              << ", sum=" << actual_sum;

    // Basic sanity check (will be replaced with precise validation later)
    CHECK_NE(actual_sum, 0.0) << "Output should not be all zeros";

    if (params.perform_precise_validation) {
      test::expect_tensor_stats(output,
                                params.expected_min,
                                params.expected_max,
                                params.expected_sum);
      LOG(INFO) << "Rank " << params.rank
                << ": All2All test passed with precise validation.";
    } else {
      LOG(INFO) << "Rank " << params.rank
                << ": All2All test passed (basic validation only).";
    }

    return 0;

  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Child process test function for SmoothQuant All2All test
int32_t run_all2all_smoothquant_test_child(All2AllTestParams params) {
  params.is_smoothquant = true;
  return run_all2all_basic_test_child(params);
}

int32_t run_all2all_route_guard_test_child(All2AllTestParams params) {
  try {
    EPLBConfig::get_instance().expert_parallel_degree(2);

    int32_t dev_count = xllm::Device::device_count();
    if (dev_count < params.world_size) {
      LOG(WARNING) << "Rank " << params.rank
                   << ": Insufficient devices. Skipping.";
      return EXIT_CODE_SKIP;
    }
    params.device_index = params.rank % dev_count;

    xllm::Device xllm_device(params.device_index);
    xllm_device.set_device();
    torch::Device device = xllm_device.unwrap();

    auto process_group = create_test_process_group(
        params.rank, params.world_size, params.port, params.host, device);
    CHECK(process_group) << "Rank " << params.rank
                         << ": Failed to create ProcessGroup";

    ParallelArgs parallel_args(
        params.rank, params.world_size, process_group.get());
    parallel_args.moe_ep_group_ = process_group.get();
    parallel_args.ep_size_ = params.world_size;
    parallel_args.moe_tp_group_ = process_group.get();

    auto options = torch::TensorOptions()
                       .dtype(torch::kBFloat16)
                       .device(device)
                       .requires_grad(false);

    ModelArgs model_args = create_model_args(params);
    QuantArgs quant_args = create_quant_args(params);
    FusedMoE fused_moe(FusedMoEImpl(model_args,
                                    FusedMoEArgs{.is_gated = true},
                                    quant_args,
                                    parallel_args,
                                    options));

    auto weight_dict = create_all2all_test_weights(params.num_experts,
                                                   params.hidden_size,
                                                   params.intermediate_size,
                                                   params.is_smoothquant,
                                                   params.moe_weight_bits,
                                                   params.group_size,
                                                   params.world_size,
                                                   device);
    StateDict state_dict(weight_dict);
    fused_moe->load_state_dict(state_dict);

    int64_t num_tokens = params.batch_size * params.seq_len;
    auto hidden_states =
        test::seeded_tensor("fused_moe_all2all_tests.guard.hidden_states",
                            {num_tokens, params.hidden_size},
                            torch::kBFloat16,
                            device);
    auto route_info = fused_moe->prep_route(hidden_states);
    fused_moe->forward_experts(hidden_states,
                               /*enable_all2all_communication=*/true,
                               route_info);

    xllm_device.synchronize_default_stream();
    return 0;
  } catch (const std::exception& e) {
    LOG(ERROR) << "Rank " << params.rank << ": Exception: " << e.what();
    return 1;
  }
}

// Multi-process test fixture
class FusedMoEAll2AllMultiDeviceTest : public ::testing::Test {
 protected:
  void SetUp() override {
    world_size_ = 2;
    port_ = 29501;  // Different port from deep_ep_tests
    host_ = "127.0.0.1";
    hidden_size_ = 512;
    intermediate_size_ = 256;
    num_experts_ = 4;
    top_k_ = 2;
    batch_size_ = 2;
    seq_len_ = 4;
  }

  // Runs multi-process test with params built from fixture (optionally
  // is_smoothquant). Use this for basic flow tests.
  void run_test(int32_t (*test_fn)(All2AllTestParams),
                bool is_smoothquant = false) {
    run_test_impl(test_fn, [this, is_smoothquant](int32_t rank) {
      All2AllTestParams params;
      params.rank = rank;
      params.world_size = world_size_;
      params.port = port_;
      params.host = host_;
      params.device_index = -1;
      params.hidden_size = hidden_size_;
      params.intermediate_size = intermediate_size_;
      params.num_experts = num_experts_;
      params.top_k = top_k_;
      params.batch_size = batch_size_;
      params.seq_len = seq_len_;
      params.is_smoothquant = is_smoothquant;
      return params;
    });
  }

  // Runs multi-process test with custom params from factory (e.g. for precise
  // validation tests). params_factory(rank) is called per rank; the returned
  // params should have rank set by the factory or will be overwritten.
  // Named separately to avoid overload ambiguity with run_test(..., bool).
  void run_test_with_params(
      int32_t (*test_fn)(All2AllTestParams),
      std::function<All2AllTestParams(int32_t rank)> params_factory) {
    run_test_impl(test_fn, params_factory);
  }

  void run_fail_test(
      int32_t (*test_fn)(All2AllTestParams),
      std::function<All2AllTestParams(int32_t rank)> params_factory) {
    std::vector<pid_t> child_pids;

    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        All2AllTestParams params = params_factory(rank);
        params.rank = rank;
        params.world_size = world_size_;
        int32_t exit_code = test_fn(params);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_skipped = false;
    bool any_succeeded = false;

    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code == 0) {
          any_succeeded = true;
          LOG(ERROR) << "Rank " << i << " unexpectedly succeeded.";
        }
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_succeeded) << "All2All route guard did not trigger.";
    }
  }

 private:
  void run_test_impl(
      int32_t (*test_fn)(All2AllTestParams),
      std::function<All2AllTestParams(int32_t rank)> params_factory) {
    std::vector<pid_t> child_pids;

    for (int32_t rank = 0; rank < world_size_; ++rank) {
      pid_t pid = fork();
      if (pid == 0) {
        All2AllTestParams params = params_factory(rank);
        params.rank = rank;
        params.world_size = world_size_;
        int32_t exit_code = test_fn(params);
        _exit(exit_code);
      } else if (pid > 0) {
        child_pids.push_back(pid);
      } else {
        LOG(FATAL) << "Failed to fork rank " << rank;
      }
    }

    bool any_failed = false;
    bool any_skipped = false;

    for (size_t i = 0; i < child_pids.size(); ++i) {
      int32_t status;
      waitpid(child_pids[i], &status, 0);
      if (WIFEXITED(status)) {
        int32_t exit_code = WEXITSTATUS(status);
        if (exit_code == EXIT_CODE_SKIP) {
          any_skipped = true;
        } else if (exit_code != 0) {
          any_failed = true;
          LOG(ERROR) << "Rank " << i << " failed with code " << exit_code;
        }
      } else {
        any_failed = true;
        LOG(ERROR) << "Rank " << i << " crashed (signal).";
      }
    }

    if (any_skipped) {
      GTEST_SKIP() << "Test skipped due to insufficient devices.";
    } else {
      ASSERT_FALSE(any_failed) << "FusedMoE All2All Test Failed.";
    }
  }

 protected:
  int32_t world_size_;
  int32_t port_;
  std::string host_;
  int64_t hidden_size_;
  int64_t intermediate_size_;
  int64_t num_experts_;
  int64_t top_k_;
  int64_t batch_size_;
  int64_t seq_len_;
};

TEST_F(FusedMoEAll2AllMultiDeviceTest, BasicAll2AllFlow) {
  run_test(run_all2all_basic_test_child, /*is_smoothquant=*/false);
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, SmoothQuantAll2AllFlow) {
  run_test(run_all2all_smoothquant_test_child, /*is_smoothquant=*/true);
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, BasicAll2AllPreciseValidation) {
  // Expected values obtained from running with seeded tensors:
  // min=835584, max=1.26976e+06, sum=4.20999e+09
  run_test_with_params(run_all2all_basic_test_child, [](int32_t /*rank*/) {
    All2AllTestParams params;
    params.rank = 0;
    params.world_size = 2;
    params.port = 29503;  // Different port to avoid conflicts
    params.host = "127.0.0.1";
    params.device_index = -1;
    params.hidden_size = 512;
    params.intermediate_size = 256;
    params.num_experts = 4;
    params.top_k = 2;
    params.batch_size = 2;
    params.seq_len = 4;
    params.is_smoothquant = false;
    params.expected_min = 835584.0;
    params.expected_max = 1269760.0;
    params.expected_sum = 4209990000.0;
    params.perform_precise_validation = true;
    return params;
  });
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, SmoothQuantAll2AllPreciseValidation) {
  // Expected values obtained from running with seeded tensors:
  // min=0, max=0.104004, sum=3.88289
  run_test_with_params(
      run_all2all_smoothquant_test_child, [](int32_t /*rank*/) {
        All2AllTestParams params;
        params.rank = 0;
        params.world_size = 2;
        params.port = 29504;  // Different port to avoid conflicts
        params.host = "127.0.0.1";
        params.device_index = -1;
        params.hidden_size = 512;
        params.intermediate_size = 256;
        params.num_experts = 4;
        params.top_k = 2;
        params.batch_size = 2;
        params.seq_len = 4;
        params.is_smoothquant = true;
        // Note: min=0 is valid for SmoothQuant due to quantization effects
        params.expected_min = 0.0;
        params.expected_max = 0.104004;
        params.expected_sum = 3.88289;
        params.perform_precise_validation = true;
        return params;
      });
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, W4A8All2AllSmoke) {
  run_test_with_params(run_all2all_smoothquant_test_child,
                       [](int32_t /*rank*/) {
                         All2AllTestParams params;
                         params.rank = 0;
                         params.world_size = 2;
                         params.port = 29505;
                         params.host = "127.0.0.1";
                         params.device_index = -1;
                         params.hidden_size = 512;
                         params.intermediate_size = 256;
                         params.num_experts = 4;
                         params.top_k = 2;
                         params.batch_size = 2;
                         params.seq_len = 4;
                         params.is_smoothquant = true;
                         params.moe_weight_bits = 4;
                         params.group_size = 128;
                         params.perform_precise_validation = false;
                         return params;
                       });
}

TEST_F(FusedMoEAll2AllMultiDeviceTest, ExternalRouteGuard) {
  run_fail_test(run_all2all_route_guard_test_child, [](int32_t /*rank*/) {
    All2AllTestParams params;
    params.rank = 0;
    params.world_size = 2;
    params.port = 29506;
    params.host = "127.0.0.1";
    params.device_index = -1;
    params.hidden_size = 512;
    params.intermediate_size = 256;
    params.num_experts = 4;
    params.top_k = 2;
    params.batch_size = 2;
    params.seq_len = 4;
    params.is_smoothquant = false;
    return params;
  });
}

}  // namespace test
}  // namespace layer
}  // namespace xllm
