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

#include "core/framework/multimodal/mm_codec.h"

#include <gtest/gtest.h>

#include <cstdint>
#include <string>

#include "core/common/message.h"

namespace xllm {
namespace {

std::string make_wav(int32_t sample_rate, int32_t num_samples) {
  std::string wav;
  auto append = [&wav](uint32_t value, int bytes) {
    for (int byte = 0; byte < bytes; ++byte) {
      wav.push_back(static_cast<char>(value >> (8 * byte)));
    }
  };
  const uint32_t data_size = num_samples * 2;
  wav.append("RIFF");
  append(36 + data_size, 4);
  wav.append("WAVEfmt ");
  append(16, 4);
  append(1, 2);
  append(1, 2);
  append(sample_rate, 4);
  append(sample_rate * 2, 4);
  append(2, 2);
  append(16, 2);
  wav.append("data");
  append(data_size, 4);
  wav.append(data_size, '\0');
  return wav;
}

}  // namespace

TEST(FFmpegAudioDecoderTest, DurationBudgetBoundsDecoding) {
  const std::string wav = make_wav(/*sample_rate=*/8000, /*num_samples=*/8000);
  // The near-end limit also covers resampler output rejected during flush.
  for (double budget : {0.5, 0.999}) {
    torch::Tensor tensor;
    AudioMetadata meta;
    FFmpegAudioDecoder decoder(budget);
    EXPECT_FALSE(decoder.decode(wav, tensor, meta));
    EXPECT_FALSE(tensor.defined());
    EXPECT_GT(meta.duration, budget);
  }
  {
    torch::Tensor tensor;
    AudioMetadata meta;
    FFmpegAudioDecoder decoder(/*max_duration_s=*/1.0);
    ASSERT_TRUE(decoder.decode(wav, tensor, meta));
    EXPECT_EQ(tensor.numel(), 16000);
    EXPECT_DOUBLE_EQ(meta.duration, 1.0);
  }
  {
    torch::Tensor tensor;
    AudioMetadata meta;
    FFmpegAudioDecoder decoder;
    ASSERT_TRUE(decoder.decode(wav, tensor, meta));
    EXPECT_EQ(tensor.numel(), 16000);
    EXPECT_DOUBLE_EQ(meta.duration, 1.0);
  }
}

}  // namespace xllm

namespace xllm {

// ============================================================
// ImageURL / VideoURL / AudioURL struct tests
// ============================================================

TEST(ImageUrlTest, DefaultHeadersEmpty) {
  ImageURL url;
  url.url = "http://example.com/img.jpg";
  EXPECT_TRUE(url.headers.empty());
  EXPECT_EQ("http://example.com/img.jpg", url.url);
}

TEST(ImageUrlTest, SetHeaders) {
  ImageURL url;
  url.url = "http://example.com/img.jpg";
  url.headers["Authorization"] = "Bearer abc";
  url.headers["Referer"] = "https://ref.com";
  EXPECT_EQ(2, url.headers.size());
  EXPECT_EQ("Bearer abc", url.headers["Authorization"]);
  EXPECT_EQ("https://ref.com", url.headers["Referer"]);
}

TEST(VideoUrlTest, DefaultHeadersEmpty) {
  VideoURL url;
  url.url = "http://example.com/video.mp4";
  EXPECT_TRUE(url.headers.empty());
}

TEST(AudioUrlTest, DefaultHeadersEmpty) {
  AudioURL url;
  url.url = "http://example.com/audio.mp3";
  EXPECT_TRUE(url.headers.empty());
}

// ============================================================
// MMContent struct with headers
// ============================================================

TEST(MMContentTest, ImageUrlWithHeaders) {
  ImageURL img;
  img.url = "http://example.com/img.jpg";
  img.headers["Authorization"] = "Bearer test";

  MMContent content("image_url", img);
  EXPECT_EQ("image_url", content.type);
  EXPECT_EQ("http://example.com/img.jpg", content.image_url.url);
  EXPECT_EQ(1, content.image_url.headers.size());
  EXPECT_EQ("Bearer test", content.image_url.headers["Authorization"]);
}

TEST(MMContentTest, ImageUrlWithoutHeaders) {
  ImageURL img;
  img.url = "http://example.com/img.jpg";

  MMContent content("image_url", img);
  EXPECT_TRUE(content.image_url.headers.empty());
}

}  // namespace xllm
