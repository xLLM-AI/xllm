/* Copyright 2026 The xLLM Authors. All Rights Reserved.

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

#include "core/framework/parallel_state/dit_sp_qkv_domains.h"

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <array>
#include <chrono>
#include <memory>
#include <set>
#include <stdexcept>
#include <utility>

#include "core/framework/parallel_state/dit_mapping.h"
#include "core/framework/parallel_state/parallel_state.h"
#include "core/framework/parallel_state/process_group.h"

namespace xllm {
namespace {

TEST(DiTSpQkvDomainsTest, UsesAllSixPortsAndOriginalSpMembership) {
  DiTMapping::Options options;
  options.dit_tp_size(2)
      .dit_sp_size(8)
      .dit_dp_size(1)
      .dit_cfg_size(1)
      .dit_vae_size(1)
      .dit_text_encoder_tp_size(1);
  std::set<int32_t> ports;
  for (int32_t rank = 0; rank < 16; ++rank) {
    DiTMapping mapping(16, rank, options);
    const auto specs = build_dit_sp_qkv_domain_specs(mapping, 30075);
    const int32_t group = rank % 2;
    std::vector<int32_t> expected;
    for (int32_t member = group; member < 16; member += 2) {
      expected.push_back(member);
    }
    for (int32_t domain = 0; domain < 3; ++domain) {
      EXPECT_EQ(specs[domain].ranks, expected);
      EXPECT_EQ(specs[domain].local_rank, rank / 2);
      EXPECT_EQ(specs[domain].port, 30140 + domain * 2 + group);
      ports.insert(specs[domain].port);
    }
    EXPECT_EQ(specs[0].name, "sp_q_domain");
    EXPECT_EQ(specs[1].name, "sp_k_domain");
    EXPECT_EQ(specs[2].name, "sp_v_domain");
    EXPECT_THROW(build_dit_sp_qkv_domain_specs(mapping, 65500),
                 std::invalid_argument);
  }
  EXPECT_EQ(ports,
            (std::set<int32_t>{30140, 30141, 30142, 30143, 30144, 30145}));
}

TEST(DiTSpQkvDomainsTest, QualifiedOptionIsIndependentOfNativeOverlap) {
  EXPECT_FALSE(
      make_dit_sp_qkv_communication_options("native", false).create_domains);
  EXPECT_FALSE(
      make_dit_sp_qkv_communication_options("a3_16card_tp2_u8_ref2va_v1", false)
          .create_domains);
  EXPECT_THROW(make_dit_sp_qkv_communication_options("native", true),
               std::invalid_argument);
  // The legacy native overlap flag is deliberately absent from this API.
  EXPECT_TRUE(
      make_dit_sp_qkv_communication_options("a3_16card_tp2_u8_ref2va_v1", true)
          .create_domains);
}

TEST(DiTSpQkvDomainsTest,
     RejectsPartialAliasedAndWrongRankDomainsBeforeLaunch) {
  const torch::Device device(torch::kCPU);
  ProcessGroup sequence(0, 8, device);
  ProcessGroup q(0, 8, device);
  ProcessGroup k(0, 8, device);
  ProcessGroup v(0, 8, device);
  ProcessGroup wrong_rank(1, 8, device);
  EXPECT_THROW(validate_dit_sp_qkv_domains({&q, nullptr, &v}, &sequence),
               std::invalid_argument);
  EXPECT_THROW(validate_dit_sp_qkv_domains({&q, &q, &v}, &sequence),
               std::invalid_argument);
  EXPECT_THROW(validate_dit_sp_qkv_domains({&q, &k, &wrong_rank}, &sequence),
               std::invalid_argument);
  EXPECT_THROW(validate_dit_sp_qkv_domains({&q, &k, &sequence}, &sequence),
               std::invalid_argument);
  EXPECT_NO_THROW(validate_dit_sp_qkv_domains({&q, &k, &v}, &sequence));
}

class DelayedCopyWork final : public c10d::Work {
 public:
  DelayedCopyWork(const torch::Tensor& input, torch::Tensor output)
      : c10d::Work(0, c10d::OpType::ALLTOALL),
        input_(input.getIntrusivePtr()),
        output_(std::move(output)) {}

  bool wait(std::chrono::milliseconds timeout) override {
    auto storage = input_.lock();
    if (!storage) {
      throw std::runtime_error(
          "async all-to-all send staging was released before wait");
    }
    const torch::Tensor input(std::move(storage));
    output_.copy_(input);
    return true;
  }

  bool source_alive() const { return !input_.expired(); }

 private:
  c10::weak_intrusive_ptr<c10::TensorImpl, c10::UndefinedTensorImpl> input_;
  torch::Tensor output_;
};

class DelayedProcessGroup final : public ProcessGroup {
 public:
  DelayedProcessGroup() : ProcessGroup(0, 2, torch::Device(torch::kCPU)) {}

  void all_to_all_single(torch::Tensor output,
                         torch::Tensor input,
                         std::vector<int64_t> output_splits,
                         std::vector<int64_t> input_splits,
                         bool async_op,
                         c10::intrusive_ptr<c10d::Work>* work) override {
    if (!async_op || work == nullptr) {
      throw std::logic_error("lifetime test requires the asynchronous path");
    }
    last_work = c10::make_intrusive<DelayedCopyWork>(input, output);
    *work = last_work;
  }

  c10::intrusive_ptr<DelayedCopyWork> last_work;
};

TEST(DiTSpQkvDomainsTest, AsyncClosureOwnsInternalSendStagingUntilWait) {
  DelayedProcessGroup group;
  torch::Tensor input = torch::arange(64, torch::kFloat32).view({1, 4, 8, 2});
  auto pending = parallel_state::all_to_all_4D(input, 2, 1, true, &group);
  input = torch::Tensor();
  ASSERT_TRUE(group.last_work->source_alive());
  const torch::Tensor output = pending();
  EXPECT_EQ(output.sizes(), c10::IntArrayRef({1, 8, 4, 2}));
  pending = nullptr;
  EXPECT_FALSE(group.last_work->source_alive());
}

}  // namespace
}  // namespace xllm
