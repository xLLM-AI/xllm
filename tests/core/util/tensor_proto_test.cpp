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

#include <gtest/gtest.h>
#include <torch/torch.h>

#include <cstdint>
#include <limits>
#include <string>

#include "core/util/binary_payload.h"
#include "core/util/utils.h"
#include "tensor.pb.h"

namespace xllm {
namespace {

TEST(TensorProtoTest, BinaryTensorRoundTripPreservesValues) {
  const torch::Tensor input = torch::tensor({{1.25f, -2.5f}, {3.75f, 4.5f}},
                                            torch::dtype(torch::kFloat32));
  proto::Tensor proto_tensor;
  std::string payload = "prefix";

  ASSERT_TRUE(util::torch_to_proto(input, &proto_tensor, payload));
  const torch::Tensor output = util::proto_to_torch(proto_tensor, payload);

  ASSERT_TRUE(output.defined());
  EXPECT_TRUE(torch::equal(output, input));
  EXPECT_EQ(proto_tensor.parameters().at("offset").int64_param(), 6);
  EXPECT_EQ(proto_tensor.parameters().at("len").int64_param(),
            input.numel() * input.element_size());
}

TEST(TensorProtoTest, BinaryTensorsSharePayloadWithIndependentOffsets) {
  const torch::Tensor first =
      torch::tensor({1, 2}, torch::dtype(torch::kInt32));
  const torch::Tensor second =
      torch::tensor({3.0, 4.0}, torch::dtype(torch::kFloat64));
  proto::Tensor first_proto;
  proto::Tensor second_proto;
  std::string payload;

  ASSERT_TRUE(util::torch_to_proto(first, &first_proto, payload));
  ASSERT_TRUE(util::torch_to_proto(second, &second_proto, payload));

  EXPECT_EQ(first_proto.parameters().at("offset").int64_param(), 0);
  EXPECT_EQ(second_proto.parameters().at("offset").int64_param(),
            first.numel() * first.element_size());
  EXPECT_TRUE(torch::equal(util::proto_to_torch(first_proto, payload), first));
  EXPECT_TRUE(
      torch::equal(util::proto_to_torch(second_proto, payload), second));
}

TEST(TensorProtoTest, SegmentedBinaryPayloadCopiesDirectlyToTensor) {
  const torch::Tensor input = torch::tensor({1.25f, -2.5f});
  proto::Tensor proto_tensor;
  std::string flat_payload;
  ASSERT_TRUE(util::torch_to_proto(input, &proto_tensor, flat_payload));

  butil::IOBuf first;
  butil::IOBuf second;
  first.append(flat_payload.data(), 3);
  second.append(flat_payload.data() + 3, flat_payload.size() - 3);
  butil::IOBuf segmented;
  segmented.append(first);
  segmented.append(second);

  const torch::Tensor output =
      util::proto_to_torch(proto_tensor, BinaryPayload(std::move(segmented)));

  ASSERT_TRUE(output.defined());
  EXPECT_TRUE(torch::equal(output, input));
}

TEST(TensorProtoTest, ContentsTensorRemainsSupportedWithPayloadOverload) {
  const torch::Tensor input = torch::tensor({5.0f, 6.0f});
  proto::Tensor proto_tensor;
  ASSERT_TRUE(util::torch_to_proto(input, &proto_tensor));

  const torch::Tensor output = util::proto_to_torch(proto_tensor, "unused");

  ASSERT_TRUE(output.defined());
  EXPECT_TRUE(torch::equal(output, input));
}

TEST(TensorProtoTest, RejectsMalformedBinaryDescriptors) {
  proto::Tensor proto_tensor;
  proto_tensor.set_datatype("FP32");
  proto_tensor.add_shape(2);
  (*proto_tensor.mutable_parameters())["is_binary"].set_bool_param(true);
  (*proto_tensor.mutable_parameters())["offset"].set_int64_param(1);
  (*proto_tensor.mutable_parameters())["len"].set_int64_param(
      std::numeric_limits<int64_t>::max());

  EXPECT_FALSE(util::proto_to_torch(proto_tensor, std::string(8, 0)).defined());

  (*proto_tensor.mutable_parameters())["offset"].set_int64_param(0);
  (*proto_tensor.mutable_parameters())["len"].set_int64_param(sizeof(float));
  EXPECT_FALSE(util::proto_to_torch(proto_tensor, std::string(8, 0)).defined());

  proto_tensor.mutable_parameters()->erase("offset");
  EXPECT_FALSE(util::proto_to_torch(proto_tensor, std::string(8, 0)).defined());
}

}  // namespace
}  // namespace xllm
