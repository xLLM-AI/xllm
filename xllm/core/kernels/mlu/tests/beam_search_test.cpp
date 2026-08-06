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

#include <framework/core/MLUStream.h>
#include <framework/core/device.h>
#include <framework/core/stream_guard.h>
#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <tuple>
#include <vector>

#include "common/global_flags.h"
#include "kernels/mlu/mlu_ops_api.h"

namespace xllm::kernel::mlu {
namespace {

constexpr int32_t kSequenceSentinel = -777;
constexpr int32_t kReservedSentinel = 919;

struct FlagGuard {
  explicit FlagGuard(bool value) : old(FLAGS_enable_topk_sorted) {
    FLAGS_enable_topk_sorted = value;
  }
  ~FlagGuard() { FLAGS_enable_topk_sorted = old; }
  bool old;
};

struct Expected {
  torch::Tensor scores;
  torch::Tensor tokens;
  torch::Tensor indices;
  torch::Tensor sequence;
};

struct State {
  State(int64_t batch, int64_t beam, int64_t rounds, uint32_t step)
      : batch(batch), beam(beam), rounds(rounds), step(step) {
    auto fp32 = torch::TensorOptions().dtype(torch::kFloat32);
    auto int32 = torch::TensorOptions().dtype(torch::kInt32);
    const int64_t rows = batch * beam;

    acc_cpu = (-0.13f * torch::arange(rows, fp32)).view({rows, 1});
    sequence_cpu =
        torch::arange(batch * beam * rounds, int32).view({batch, beam, rounds});
    if (step == 0) {
      tokens_cpu =
          (100 + torch::arange(batch * beam, int32)).view({batch, beam});
      probs_cpu =
          (-0.07f * torch::arange(batch * beam, fp32)).view({batch, beam});
    } else {
      tokens_cpu = (200 + torch::arange(rows * beam, int32)).view({rows, beam});
      auto order =
          torch::remainder(torch::arange(rows * beam, fp32) * 7, rows * beam);
      probs_cpu = (-0.031f * order).view({rows, beam});
    }

    const auto device = torch::Device("mlu:0");
    acc = acc_cpu.to(device);
    sequence = sequence_cpu.to(device);
    tokens = tokens_cpu.to(device);
    probs = probs_cpu.to(device);
    out_scores = torch::full({rows, 1}, 42.0f, fp32.device(device));
    out_tokens = torch::full({rows, 1}, -1, int32.device(device));
    out_indices = torch::full({rows, 1}, -1, int32.device(device));
    reserved = torch::full({rows, 1}, kReservedSentinel, int32.device(device));
    out_sequence = torch::full(
        {batch, beam, rounds}, kSequenceSentinel, int32.device(device));
  }

  Expected reference(bool sorted = true) const {
    Expected expected;
    expected.sequence =
        torch::full({batch, beam, rounds},
                    kSequenceSentinel,
                    torch::TensorOptions().dtype(torch::kInt32));
    if (step == 0) {
      expected.scores = probs_cpu.reshape({-1, 1}).clone();
      expected.tokens = tokens_cpu.reshape({-1, 1}).clone();
      expected.indices =
          torch::arange(beam, torch::TensorOptions().dtype(torch::kInt32))
              .unsqueeze(0)
              .expand({batch, beam})
              .reshape({-1, 1})
              .clone();
      expected.sequence.select(2, 0).copy_(tokens_cpu);
      return expected;
    }

    auto combined = (acc_cpu + probs_cpu).view({batch, beam * beam});
    auto topk = torch::topk(combined,
                            beam,
                            /*dim=*/1,
                            /*largest=*/true,
                            /*sorted=*/sorted);
    auto scores = std::get<0>(topk);
    auto indices = std::get<1>(topk);
    if (step < static_cast<uint32_t>(rounds - 1)) {
      auto order = indices.argsort(1, false);
      scores = scores.gather(1, order);
      indices = indices.gather(1, order);
    }
    auto selected_tokens =
        tokens_cpu.view({batch, beam * beam}).gather(1, indices);
    auto parent = torch::div(indices, beam, "floor");
    auto history_idx = parent.unsqueeze(-1).expand({batch, beam, rounds});
    auto selected_history = sequence_cpu.gather(1, history_idx);
    expected.sequence.slice(2, 0, step)
        .copy_(selected_history.slice(2, 0, step));
    expected.sequence.select(2, step).copy_(selected_tokens);
    expected.scores = scores.reshape({-1, 1});
    expected.tokens = selected_tokens.reshape({-1, 1});
    expected.indices = indices.to(torch::kInt32).reshape({-1, 1});
    return expected;
  }

  void run(uint32_t returns = 0) {
    beam_search(acc,
                sequence,
                tokens,
                probs,
                out_scores,
                out_tokens,
                out_indices,
                reserved,
                out_sequence,
                batch,
                returns,
                step);
  }

  int64_t batch;
  int64_t beam;
  int64_t rounds;
  uint32_t step;
  torch::Tensor acc_cpu;
  torch::Tensor sequence_cpu;
  torch::Tensor tokens_cpu;
  torch::Tensor probs_cpu;
  torch::Tensor acc;
  torch::Tensor sequence;
  torch::Tensor tokens;
  torch::Tensor probs;
  torch::Tensor out_scores;
  torch::Tensor out_tokens;
  torch::Tensor out_indices;
  torch::Tensor reserved;
  torch::Tensor out_sequence;
};

void expect_result(const State& state, const Expected& expected) {
  EXPECT_TRUE(
      torch::allclose(state.out_scores.cpu(), expected.scores, 1e-5, 1e-6));
  EXPECT_TRUE(torch::equal(state.out_tokens.cpu(), expected.tokens));
  EXPECT_TRUE(torch::equal(state.out_indices.cpu(), expected.indices));
  EXPECT_TRUE(torch::equal(state.out_sequence.cpu(), expected.sequence));
  EXPECT_TRUE(
      torch::all(state.reserved.cpu() == kReservedSentinel).item<bool>());
}

class BeamSearchTest : public ::testing::Test {
 protected:
  void SetUp() override {
    if (torch_mlu::device_count() == 0) {
      GTEST_SKIP() << "MLU is not available";
    }
    FLAGS_enable_topk_sorted = true;
  }
};

TEST_F(BeamSearchTest, InitializesFirstRound) {
  State state(/*batch=*/3, /*beam=*/4, /*rounds=*/5, /*step=*/0);
  auto expected = state.reference();
  auto score_storage = state.out_scores.storage().unsafeGetStorageImpl();
  auto token_storage = state.out_tokens.storage().unsafeGetStorageImpl();
  auto index_storage = state.out_indices.storage().unsafeGetStorageImpl();
  auto sequence_storage = state.out_sequence.storage().unsafeGetStorageImpl();

  state.run(/*returns=*/0);

  expect_result(state, expected);
  EXPECT_EQ(state.out_scores.storage().unsafeGetStorageImpl(), score_storage);
  EXPECT_EQ(state.out_tokens.storage().unsafeGetStorageImpl(), token_storage);
  EXPECT_EQ(state.out_indices.storage().unsafeGetStorageImpl(), index_storage);
  EXPECT_EQ(state.out_sequence.storage().unsafeGetStorageImpl(),
            sequence_storage);
}

TEST_F(BeamSearchTest, HandlesSingleRound) {
  State state(/*batch=*/1, /*beam=*/2, /*rounds=*/1, /*step=*/0);
  auto expected = state.reference();
  state.run(/*returns=*/2);
  expect_result(state, expected);
}

TEST_F(BeamSearchTest, OrdersMiddleRoundByFlatIndex) {
  State state(/*batch=*/3, /*beam=*/4, /*rounds=*/5, /*step=*/2);
  auto expected = state.reference();
  state.run(/*returns=*/9);
  expect_result(state, expected);

  auto indices = state.out_indices.cpu().view({state.batch, state.beam});
  EXPECT_TRUE(
      torch::all(indices.slice(1, 1) >= indices.slice(1, 0, -1)).item<bool>());
}

TEST_F(BeamSearchTest, SortsFinalRoundByScore) {
  State state(/*batch=*/3, /*beam=*/4, /*rounds=*/3, /*step=*/2);
  auto expected = state.reference();
  state.run(/*returns=*/4);
  expect_result(state, expected);

  auto scores = state.out_scores.cpu().view({state.batch, state.beam});
  EXPECT_TRUE(
      torch::all(scores.slice(1, 0, -1) >= scores.slice(1, 1)).item<bool>());
}

TEST_F(BeamSearchTest, SupportsNegativeInfinity) {
  State state(/*batch=*/1, /*beam=*/4, /*rounds=*/3, /*step=*/1);
  state.probs_cpu[0][0] = -std::numeric_limits<float>::infinity();
  state.probs_cpu[1][1] = -std::numeric_limits<float>::infinity();
  state.probs = state.probs_cpu.to(torch::Device("mlu:0"));
  auto expected = state.reference();
  state.run();
  expect_result(state, expected);
}

TEST_F(BeamSearchTest, PreservesAssociationsForTies) {
  State state(/*batch=*/1, /*beam=*/4, /*rounds=*/3, /*step=*/2);
  state.acc_cpu.zero_();
  state.probs_cpu.fill_(-1.0f);
  state.acc = state.acc_cpu.to(torch::Device("mlu:0"));
  state.probs = state.probs_cpu.to(torch::Device("mlu:0"));
  state.run(/*returns=*/4);

  auto indices = state.out_indices.cpu().view({1, state.beam}).to(torch::kLong);
  auto expected_tokens =
      state.tokens_cpu.view({1, state.beam * state.beam}).gather(1, indices);
  auto parent = torch::div(indices, state.beam, "floor");
  auto history_idx = parent.unsqueeze(-1).expand({1, state.beam, state.rounds});
  auto expected_history = state.sequence_cpu.gather(1, history_idx);
  EXPECT_TRUE(torch::equal(state.out_tokens.cpu().view({1, state.beam}),
                           expected_tokens));
  EXPECT_TRUE(torch::equal(state.out_sequence.cpu().slice(2, 0, state.step),
                           expected_history.slice(2, 0, state.step)));
  EXPECT_TRUE(torch::allclose(state.out_scores.cpu(),
                              torch::full_like(state.out_scores.cpu(), -1.0f)));
}

TEST_F(BeamSearchTest, UnsortedModeReturnsBestSet) {
  FlagGuard guard(false);
  State state(/*batch=*/3, /*beam=*/4, /*rounds=*/3, /*step=*/2);
  auto expected = state.reference(/*sorted=*/false);
  state.run(/*returns=*/4);

  auto actual_idx = state.out_indices.cpu().view({state.batch, state.beam});
  auto expected_idx = expected.indices.view({state.batch, state.beam});
  auto actual_order = std::get<1>(actual_idx.sort(1));
  auto expected_order = std::get<1>(expected_idx.sort(1));
  EXPECT_TRUE(torch::equal(actual_idx.gather(1, actual_order),
                           expected_idx.gather(1, expected_order)));
  EXPECT_TRUE(torch::allclose(
      state.out_scores.cpu()
          .view({state.batch, state.beam})
          .gather(1, actual_order),
      expected.scores.view({state.batch, state.beam}).gather(1, expected_order),
      1e-5,
      1e-6));
  EXPECT_TRUE(torch::equal(state.out_tokens.cpu()
                               .view({state.batch, state.beam})
                               .gather(1, actual_order),
                           expected.tokens.view({state.batch, state.beam})
                               .gather(1, expected_order)));
}

TEST_F(BeamSearchTest, CarriesStateAcrossRounds) {
  State first(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/0);
  first.run(/*returns=*/2);

  State middle(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/1);
  middle.acc_cpu = first.out_scores.cpu();
  middle.sequence_cpu = first.out_sequence.cpu();
  middle.acc = first.out_scores;
  middle.sequence = first.out_sequence;
  auto middle_expected = middle.reference();
  middle.run(/*returns=*/2);
  expect_result(middle, middle_expected);

  State last(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/2);
  last.acc_cpu = middle.out_scores.cpu();
  last.sequence_cpu = middle.out_sequence.cpu();
  last.acc = middle.out_scores;
  last.sequence = middle.out_sequence;
  auto last_expected = last.reference();
  last.run(/*returns=*/2);
  expect_result(last, last_expected);
}

TEST_F(BeamSearchTest, CarriesBusinessShapeAcrossThreeRounds) {
  FlagGuard guard(true);
  constexpr int64_t kBatchSize = 4;
  constexpr int64_t kBeamWidth = 256;
  constexpr int64_t kTotalRounds = 3;

  State first(kBatchSize, kBeamWidth, kTotalRounds, /*step=*/0);
  auto first_expected = first.reference();
  first.run(kBeamWidth);
  expect_result(first, first_expected);
  EXPECT_EQ(first.out_sequence.sizes(),
            torch::IntArrayRef({kBatchSize, kBeamWidth, kTotalRounds}));
  EXPECT_EQ(first.out_scores.sizes(),
            torch::IntArrayRef({kBatchSize * kBeamWidth, 1}));

  State middle(kBatchSize, kBeamWidth, kTotalRounds, /*step=*/1);
  middle.acc_cpu = first.out_scores.cpu();
  middle.sequence_cpu = first.out_sequence.cpu();
  middle.acc = first.out_scores;
  middle.sequence = first.out_sequence;
  auto middle_expected = middle.reference();
  middle.run(kBeamWidth);
  expect_result(middle, middle_expected);
  EXPECT_EQ(middle.out_sequence.sizes(),
            torch::IntArrayRef({kBatchSize, kBeamWidth, kTotalRounds}));
  EXPECT_EQ(middle.out_scores.sizes(),
            torch::IntArrayRef({kBatchSize * kBeamWidth, 1}));
  auto middle_indices = middle.out_indices.cpu().view({kBatchSize, kBeamWidth});
  EXPECT_TRUE(torch::equal(
      torch::div(middle_indices, kBeamWidth, "floor"),
      torch::div(middle_expected.indices.view({kBatchSize, kBeamWidth}),
                 kBeamWidth,
                 "floor")));
  EXPECT_TRUE(
      torch::all(middle_indices.slice(1, 1) >= middle_indices.slice(1, 0, -1))
          .item<bool>());

  State last(kBatchSize, kBeamWidth, kTotalRounds, /*step=*/2);
  last.acc_cpu = middle.out_scores.cpu();
  last.sequence_cpu = middle.out_sequence.cpu();
  last.acc = middle.out_scores;
  last.sequence = middle.out_sequence;
  auto last_expected = last.reference();
  last.run(kBeamWidth);
  expect_result(last, last_expected);
  auto last_indices = last.out_indices.cpu().view({kBatchSize, kBeamWidth});
  EXPECT_TRUE(torch::equal(
      torch::div(last_indices, kBeamWidth, "floor"),
      torch::div(last_expected.indices.view({kBatchSize, kBeamWidth}),
                 kBeamWidth,
                 "floor")));
  auto last_scores = last.out_scores.cpu().view({kBatchSize, kBeamWidth});
  EXPECT_TRUE(torch::all(last_scores.slice(1, 0, -1) >= last_scores.slice(1, 1))
                  .item<bool>());
  EXPECT_EQ(last.out_sequence.sizes(),
            torch::IntArrayRef({kBatchSize, kBeamWidth, kTotalRounds}));
  EXPECT_EQ(last.out_scores.sizes(),
            torch::IntArrayRef({kBatchSize * kBeamWidth, 1}));
}

TEST_F(BeamSearchTest, HandlesSingleBeam) {
  State state(/*batch=*/3, /*beam=*/1, /*rounds=*/2, /*step=*/1);
  auto expected = state.reference();
  state.run(/*returns=*/1);
  expect_result(state, expected);
}

TEST_F(BeamSearchTest, SupportsLargeBeams) {
  for (int64_t beam : {64, 128}) {
    State state(/*batch=*/1, beam, /*rounds=*/2, /*step=*/1);
    auto expected = state.reference();
    state.run(beam);
    expect_result(state, expected);
  }
}

TEST_F(BeamSearchTest, UsesCurrentStream) {
  auto stream = torch_mlu::getStreamFromPool(/*isHighPriority=*/false, 0);
  State* state_ptr = nullptr;
  Expected expected;
  {
    torch_mlu::mlu::MLUStreamGuard guard(stream);
    state_ptr = new State(/*batch=*/3, /*beam=*/4, /*rounds=*/3, /*step=*/1);
    expected = state_ptr->reference();
    state_ptr->run(/*returns=*/4);
  }
  stream.synchronize();
  expect_result(*state_ptr, expected);
  delete state_ptr;
}

TEST_F(BeamSearchTest, RejectsWidenedFinal) {
  State state(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/2);
  EXPECT_THROW(state.run(/*returns=*/3), c10::Error);
}

TEST_F(BeamSearchTest, AllowsWideNonFinalAndZeroReturns) {
  State wide(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/1);
  auto wide_expected = wide.reference();
  EXPECT_NO_THROW(wide.run(/*returns=*/9));
  expect_result(wide, wide_expected);

  State zero(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/2);
  auto zero_expected = zero.reference();
  EXPECT_NO_THROW(zero.run(/*returns=*/0));
  expect_result(zero, zero_expected);
}

TEST_F(BeamSearchTest, RejectsInvalidDtypes) {
  {
    State state(1, 2, 3, 1);
    state.probs = state.probs.to(torch::kFloat16);
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.tokens = state.tokens.to(torch::kInt64);
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.out_indices = state.out_indices.to(torch::kInt64);
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.sequence = state.sequence.to(torch::kInt64);
    EXPECT_THROW(state.run(), c10::Error);
  }
}

TEST_F(BeamSearchTest, RejectsInvalidShapes) {
  {
    State state(1, 2, 3, 1);
    state.acc = torch::empty({2, 2}, state.acc.options());
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.tokens = torch::empty({2, 3}, state.tokens.options());
    state.probs = torch::empty({2, 3}, state.probs.options());
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.out_tokens = torch::empty({2, 2}, state.out_tokens.options());
    EXPECT_THROW(state.run(), c10::Error);
  }
}

TEST_F(BeamSearchTest, RejectsNonContiguousTensor) {
  State state(/*batch=*/2, /*beam=*/3, /*rounds=*/3, /*step=*/0);
  state.tokens = torch::empty({state.beam, state.batch}, state.tokens.options())
                     .transpose(0, 1);
  ASSERT_FALSE(state.tokens.is_contiguous());
  EXPECT_THROW(state.run(), c10::Error);
}

TEST_F(BeamSearchTest, RejectsStorageAliases) {
  {
    State state(1, 2, 3, 1);
    state.out_scores = state.acc;
    EXPECT_THROW(state.run(), c10::Error);
  }
  {
    State state(1, 2, 3, 1);
    state.out_indices = state.out_tokens;
    EXPECT_THROW(state.run(), c10::Error);
  }
}

TEST_F(BeamSearchTest, RejectsDifferentDevices) {
  if (torch_mlu::device_count() < 2) {
    GTEST_SKIP() << "two MLU devices are required";
  }
  State state(1, 2, 3, 1);
  state.tokens = state.tokens.to(torch::Device("mlu:1"));
  EXPECT_THROW(state.run(), c10::Error);
}

TEST_F(BeamSearchTest, RejectsZeroBatch) {
  auto fp32 = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(torch::Device("mlu:0"));
  auto int32 = fp32.dtype(torch::kInt32);
  EXPECT_THROW(beam_search(torch::empty({0, 1}, fp32),
                           torch::empty({0, 1, 1}, int32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1}, fp32),
                           torch::empty({0, 1}, fp32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1, 1}, int32),
                           /*batch_size=*/0,
                           /*num_return_sequences=*/0,
                           /*current_step=*/0),
               c10::Error);
}

TEST_F(BeamSearchTest, RejectsZeroBeamOrRounds) {
  auto fp32 = torch::TensorOptions()
                  .dtype(torch::kFloat32)
                  .device(torch::Device("mlu:0"));
  auto int32 = fp32.dtype(torch::kInt32);
  EXPECT_THROW(beam_search(torch::empty({0, 1}, fp32),
                           torch::empty({1, 0, 1}, int32),
                           torch::empty({1, 0}, int32),
                           torch::empty({1, 0}, fp32),
                           torch::empty({0, 1}, fp32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1}, int32),
                           torch::empty({0, 1}, int32),
                           torch::empty({1, 0, 1}, int32),
                           1,
                           0,
                           0),
               c10::Error);
  EXPECT_THROW(beam_search(torch::empty({1, 1}, fp32),
                           torch::empty({1, 1, 0}, int32),
                           torch::empty({1, 1}, int32),
                           torch::empty({1, 1}, fp32),
                           torch::empty({1, 1}, fp32),
                           torch::empty({1, 1}, int32),
                           torch::empty({1, 1}, int32),
                           torch::empty({1, 1}, int32),
                           torch::empty({1, 1, 0}, int32),
                           1,
                           0,
                           0),
               c10::Error);
}

TEST_F(BeamSearchTest, RejectsStepPastRounds) {
  State state(/*batch=*/1, /*beam=*/2, /*rounds=*/3, /*step=*/1);
  state.step = 3;
  EXPECT_THROW(state.run(), c10::Error);
}

}  // namespace
}  // namespace xllm::kernel::mlu
