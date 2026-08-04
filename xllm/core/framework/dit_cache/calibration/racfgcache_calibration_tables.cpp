/* Copyright 2025 The xLLM Authors. All Rights Reserved.

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

#include "racfgcache_calibration_tables.h"

#include <cmath>
#include <cstddef>
#include <limits>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>

namespace xllm {
namespace {

using RhoTableBuilder = torch::Tensor (*)();

inline float NaN() { return std::numeric_limits<float>::quiet_NaN(); }

// Build a square rho table with shape [steps, steps].
torch::Tensor make_square_table(int64_t steps,
                                const std::vector<float>& values) {
  CHECK(steps > 0) << "steps must be positive, got " << steps;
  CHECK(values.size() == static_cast<size_t>(steps * steps))
      << "rho table size mismatch, expected " << steps * steps << ", got "
      << values.size();

  return torch::tensor(values, torch::TensorOptions().dtype(torch::kFloat32))
      .view({steps, steps})
      .contiguous()
      .clone();
}

// Registered hardcoded tables
// Add new tables by:
//
// 1. defining a new build_xxx() function
// 2. adding one entry to GetRhoRegistry()

torch::Tensor build_qwen_image_edit_plus_cfg4_steps40() {
  static torch::Tensor table = make_square_table(
      40,
      {NaN(),   0.7310f, 0.8270f, 0.8686f, 0.8948f, 0.9094f, 0.9218f, 0.9295f,
       0.9373f, 0.9442f, 0.9500f, 0.9553f, 0.9587f, 0.9630f, 0.9662f, 0.9694f,
       0.9723f, 0.9749f, 0.9773f, 0.9795f, 0.9818f, 0.9836f, 0.9853f, 0.9868f,
       0.9883f, 0.9898f, 0.9909f, 0.9921f, 0.9932f, 0.9941f, 0.9951f, 0.9957f,
       0.9966f, 0.9974f, 0.9980f, 0.9988f, 0.9994f, 1.0002f, 1.0019f, 1.0050f,
       NaN(),   NaN(),   0.6363f, 0.7810f, 0.8425f, 0.8759f, 0.9009f, 0.9150f,
       0.9273f, 0.9381f, 0.9461f, 0.9536f, 0.9580f, 0.9632f, 0.9672f, 0.9707f,
       0.9742f, 0.9770f, 0.9796f, 0.9818f, 0.9841f, 0.9859f, 0.9874f, 0.9888f,
       0.9901f, 0.9915f, 0.9926f, 0.9935f, 0.9945f, 0.9953f, 0.9961f, 0.9966f,
       0.9974f, 0.9980f, 0.9985f, 0.9991f, 0.9996f, 1.0005f, 1.0022f, 1.0054f,
       NaN(),   NaN(),   NaN(),   0.5467f, 0.7181f, 0.8004f, 0.8535f, 0.8786f,
       0.9002f, 0.9190f, 0.9311f, 0.9421f, 0.9488f, 0.9562f, 0.9616f, 0.9665f,
       0.9709f, 0.9745f, 0.9777f, 0.9804f, 0.9829f, 0.9850f, 0.9868f, 0.9882f,
       0.9897f, 0.9913f, 0.9923f, 0.9935f, 0.9945f, 0.9953f, 0.9961f, 0.9966f,
       0.9973f, 0.9981f, 0.9985f, 0.9992f, 0.9997f, 1.0005f, 1.0023f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   0.4969f, 0.6716f, 0.7815f, 0.8289f,
       0.8667f, 0.8951f, 0.9137f, 0.9292f, 0.9388f, 0.9487f, 0.9558f, 0.9621f,
       0.9676f, 0.9719f, 0.9756f, 0.9788f, 0.9818f, 0.9841f, 0.9860f, 0.9877f,
       0.9893f, 0.9910f, 0.9922f, 0.9934f, 0.9945f, 0.9953f, 0.9961f, 0.9967f,
       0.9974f, 0.9981f, 0.9986f, 0.9992f, 0.9998f, 1.0006f, 1.0023f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4771f, 0.6648f, 0.7509f,
       0.8174f, 0.8629f, 0.8908f, 0.9134f, 0.9259f, 0.9390f, 0.9486f, 0.9567f,
       0.9636f, 0.9688f, 0.9731f, 0.9769f, 0.9803f, 0.9829f, 0.9850f, 0.9870f,
       0.9886f, 0.9905f, 0.9917f, 0.9930f, 0.9942f, 0.9950f, 0.9959f, 0.9964f,
       0.9973f, 0.9980f, 0.9985f, 0.9992f, 0.9997f, 1.0005f, 1.0023f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4765f, 0.6389f,
       0.7531f, 0.8145f, 0.8585f, 0.8910f, 0.9094f, 0.9266f, 0.9389f, 0.9490f,
       0.9579f, 0.9642f, 0.9697f, 0.9740f, 0.9779f, 0.9810f, 0.9836f, 0.9856f,
       0.9876f, 0.9897f, 0.9911f, 0.9924f, 0.9937f, 0.9946f, 0.9955f, 0.9962f,
       0.9970f, 0.9978f, 0.9983f, 0.9990f, 0.9996f, 1.0005f, 1.0023f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4385f,
       0.6383f, 0.7278f, 0.8023f, 0.8565f, 0.8829f, 0.9081f, 0.9250f, 0.9379f,
       0.9497f, 0.9577f, 0.9647f, 0.9699f, 0.9746f, 0.9785f, 0.9815f, 0.9839f,
       0.9864f, 0.9886f, 0.9902f, 0.9917f, 0.9931f, 0.9941f, 0.9951f, 0.9959f,
       0.9968f, 0.9976f, 0.9982f, 0.9990f, 0.9996f, 1.0004f, 1.0023f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       0.4180f, 0.6034f, 0.7215f, 0.8044f, 0.8456f, 0.8825f, 0.9067f, 0.9246f,
       0.9397f, 0.9502f, 0.9593f, 0.9658f, 0.9713f, 0.9758f, 0.9793f, 0.9823f,
       0.9849f, 0.9874f, 0.9892f, 0.9910f, 0.9925f, 0.9936f, 0.9947f, 0.9955f,
       0.9966f, 0.9974f, 0.9980f, 0.9988f, 0.9995f, 1.0004f, 1.0023f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   0.4438f, 0.6026f, 0.7271f, 0.7919f, 0.8471f, 0.8820f, 0.9070f,
       0.9275f, 0.9409f, 0.9521f, 0.9602f, 0.9670f, 0.9724f, 0.9766f, 0.9800f,
       0.9832f, 0.9860f, 0.9881f, 0.9900f, 0.9917f, 0.9930f, 0.9942f, 0.9951f,
       0.9961f, 0.9971f, 0.9978f, 0.9986f, 0.9994f, 1.0003f, 1.0022f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   0.4017f, 0.6266f, 0.7222f, 0.8027f, 0.8508f, 0.8841f,
       0.9121f, 0.9289f, 0.9434f, 0.9536f, 0.9622f, 0.9687f, 0.9738f, 0.9777f,
       0.9815f, 0.9847f, 0.9870f, 0.9891f, 0.9910f, 0.9924f, 0.9937f, 0.9947f,
       0.9958f, 0.9969f, 0.9976f, 0.9985f, 0.9992f, 1.0003f, 1.0022f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   0.4453f, 0.5923f, 0.7231f, 0.7986f, 0.8504f,
       0.8889f, 0.9125f, 0.9317f, 0.9447f, 0.9557f, 0.9638f, 0.9700f, 0.9747f,
       0.9791f, 0.9828f, 0.9855f, 0.9879f, 0.9901f, 0.9917f, 0.9931f, 0.9942f,
       0.9955f, 0.9966f, 0.9974f, 0.9983f, 0.9991f, 1.0002f, 1.0022f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   0.4092f, 0.6031f, 0.7228f, 0.8052f,
       0.8584f, 0.8902f, 0.9172f, 0.9340f, 0.9479f, 0.9579f, 0.9653f, 0.9710f,
       0.9763f, 0.9807f, 0.9838f, 0.9864f, 0.9889f, 0.9907f, 0.9924f, 0.9936f,
       0.9950f, 0.9962f, 0.9971f, 0.9980f, 0.9989f, 1.0000f, 1.0021f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4017f, 0.6019f, 0.7334f,
       0.8134f, 0.8612f, 0.8973f, 0.9194f, 0.9373f, 0.9502f, 0.9595f, 0.9667f,
       0.9730f, 0.9781f, 0.9818f, 0.9849f, 0.9879f, 0.9898f, 0.9917f, 0.9931f,
       0.9945f, 0.9958f, 0.9969f, 0.9979f, 0.9988f, 1.0000f, 1.0020f, 1.0055f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.3901f, 0.6117f,
       0.7383f, 0.8139f, 0.8672f, 0.8994f, 0.9227f, 0.9401f, 0.9517f, 0.9610f,
       0.9686f, 0.9750f, 0.9792f, 0.9830f, 0.9863f, 0.9886f, 0.9907f, 0.9923f,
       0.9939f, 0.9954f, 0.9965f, 0.9976f, 0.9986f, 0.9999f, 1.0019f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4308f,
       0.6192f, 0.7413f, 0.8231f, 0.8700f, 0.9035f, 0.9262f, 0.9416f, 0.9536f,
       0.9632f, 0.9708f, 0.9760f, 0.9806f, 0.9844f, 0.9872f, 0.9897f, 0.9914f,
       0.9933f, 0.9949f, 0.9962f, 0.9973f, 0.9984f, 0.9998f, 1.0019f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       0.4524f, 0.6337f, 0.7564f, 0.8279f, 0.8762f, 0.9078f, 0.9293f, 0.9446f,
       0.9569f, 0.9662f, 0.9727f, 0.9780f, 0.9825f, 0.9856f, 0.9885f, 0.9905f,
       0.9926f, 0.9944f, 0.9958f, 0.9971f, 0.9982f, 0.9996f, 1.0018f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   0.4232f, 0.6396f, 0.7544f, 0.8324f, 0.8793f, 0.9100f, 0.9314f,
       0.9475f, 0.9593f, 0.9676f, 0.9743f, 0.9798f, 0.9837f, 0.9870f, 0.9893f,
       0.9917f, 0.9938f, 0.9953f, 0.9967f, 0.9980f, 0.9995f, 1.0017f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   0.4490f, 0.6393f, 0.7659f, 0.8381f, 0.8835f, 0.9133f,
       0.9358f, 0.9510f, 0.9616f, 0.9698f, 0.9765f, 0.9812f, 0.9851f, 0.9878f,
       0.9906f, 0.9929f, 0.9947f, 0.9963f, 0.9976f, 0.9993f, 1.0016f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   0.4237f, 0.6431f, 0.7648f, 0.8406f, 0.8864f,
       0.9180f, 0.9391f, 0.9533f, 0.9641f, 0.9725f, 0.9781f, 0.9829f, 0.9862f,
       0.9895f, 0.9921f, 0.9940f, 0.9959f, 0.9973f, 0.9991f, 1.0015f, 1.0054f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   0.4558f, 0.6601f, 0.7805f, 0.8502f,
       0.8960f, 0.9246f, 0.9437f, 0.9575f, 0.9678f, 0.9747f, 0.9805f, 0.9845f,
       0.9883f, 0.9912f, 0.9934f, 0.9954f, 0.9969f, 0.9989f, 1.0014f, 1.0053f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4558f, 0.6737f, 0.7860f,
       0.8590f, 0.9014f, 0.9293f, 0.9476f, 0.9613f, 0.9702f, 0.9771f, 0.9819f,
       0.9866f, 0.9899f, 0.9924f, 0.9948f, 0.9966f, 0.9986f, 1.0013f, 1.0053f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4770f, 0.6794f,
       0.8035f, 0.8691f, 0.9088f, 0.9343f, 0.9524f, 0.9643f, 0.9730f, 0.9791f,
       0.9846f, 0.9885f, 0.9914f, 0.9941f, 0.9961f, 0.9984f, 1.0012f, 1.0053f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.4699f,
       0.6968f, 0.8121f, 0.8755f, 0.9139f, 0.9397f, 0.9560f, 0.9677f, 0.9752f,
       0.9819f, 0.9867f, 0.9902f, 0.9932f, 0.9955f, 0.9981f, 1.0011f, 1.0052f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       0.4956f, 0.7028f, 0.8182f, 0.8815f, 0.9205f, 0.9439f, 0.9598f, 0.9699f,
       0.9784f, 0.9844f, 0.9887f, 0.9922f, 0.9948f, 0.9977f, 1.0009f, 1.0052f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   0.4918f, 0.7106f, 0.8276f, 0.8906f, 0.9265f, 0.9490f, 0.9628f,
       0.9740f, 0.9815f, 0.9867f, 0.9911f, 0.9941f, 0.9973f, 1.0007f, 1.0051f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   0.5015f, 0.7317f, 0.8442f, 0.9012f, 0.9347f, 0.9541f,
       0.9685f, 0.9783f, 0.9846f, 0.9897f, 0.9932f, 0.9969f, 1.0006f, 1.0051f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   0.5291f, 0.7533f, 0.8578f, 0.9116f, 0.9407f,
       0.9610f, 0.9733f, 0.9816f, 0.9880f, 0.9922f, 0.9964f, 1.0004f, 1.0051f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   0.5507f, 0.7709f, 0.8700f, 0.9187f,
       0.9489f, 0.9668f, 0.9777f, 0.9856f, 0.9908f, 0.9957f, 1.0001f, 1.0050f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.5809f, 0.7915f, 0.8818f,
       0.9310f, 0.9574f, 0.9723f, 0.9827f, 0.9892f, 0.9950f, 0.9999f, 1.0050f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.6050f, 0.8052f,
       0.8975f, 0.9415f, 0.9641f, 0.9783f, 0.9869f, 0.9939f, 0.9994f, 1.0049f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.6376f,
       0.8337f, 0.9148f, 0.9514f, 0.9719f, 0.9836f, 0.9925f, 0.9990f, 1.0048f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       0.6782f, 0.8583f, 0.9272f, 0.9616f, 0.9791f, 0.9907f, 0.9984f, 1.0047f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   0.7215f, 0.8841f, 0.9450f, 0.9720f, 0.9882f, 0.9976f, 1.0046f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   0.7453f, 0.9067f, 0.9592f, 0.9842f, 0.9965f, 1.0045f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   0.7948f, 0.9312f, 0.9766f, 0.9945f, 1.0042f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   0.8412f, 0.9607f, 0.9912f, 1.0039f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.9070f, 0.9836f, 1.0032f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.9564f, 1.0019f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   0.9973f,
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),
       NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN(),   NaN()});
  return table;
}

// Registry of exact hardcoded matches.
// Future extension only requires adding new entries here.
const std::unordered_map<RhoTableSpec, RhoTableBuilder, RhoTableSpecHash>&
GetRhoRegistry() {
  static const std::
      unordered_map<RhoTableSpec, RhoTableBuilder, RhoTableSpecHash>
          registry = {
              {RhoTableSpec{"qwen_image_edit_plus", 4.0f, 40},
               &build_qwen_image_edit_plus_cfg4_steps40},
          };
  return registry;
}

}  // namespace

std::size_t RhoTableSpecHash::operator()(const RhoTableSpec& spec) const {
  std::size_t h1 = std::hash<std::string>{}(spec.model_name);
  std::size_t h2 =
      std::hash<int>{}(static_cast<int>(std::round(spec.cfg_scale * 1000.0f)));
  std::size_t h3 = std::hash<int>{}(spec.infer_steps);

  // A simple hash combine.
  std::size_t seed = h1;
  seed ^= h2 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  seed ^= h3 + 0x9e3779b9 + (seed << 6) + (seed >> 2);
  return seed;
}

torch::Tensor get_hardcoded_rho_table(const RhoTableSpec& spec) {
  const auto& registry = GetRhoRegistry();
  auto it = registry.find(spec);
  if (it == registry.end()) {
    return torch::Tensor();
  }
  return it->second();
}

bool has_hardcoded_rho_table(const RhoTableSpec& spec) {
  const auto& registry = GetRhoRegistry();
  return registry.find(spec) != registry.end();
}

std::string to_string(const RhoTableSpec& spec) {
  std::ostringstream oss;
  oss << "{model_name=" << spec.model_name << ", cfg_scale=" << spec.cfg_scale
      << ", infer_steps=" << spec.infer_steps << "}";
  return oss.str();
}

}  // namespace xllm