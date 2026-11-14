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
#include <torch/torch.h>

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <string>
#include <vector>

extern "C" {
#include <libavformat/avformat.h>
#include <libavutil/mem.h>
}

#include "core/common/message.h"
#include "core/framework/multimodal/dit_media_encoder.h"

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

namespace {

// Inspect the actual MP4 timing table through FFmpeg, not the input metadata.
void expect_cfr_packets(const std::string& encoded,
                        int32_t expected_frames,
                        AVRational frame_rate) {
  struct Input {
    const std::string* bytes;
    int64_t offset;
  } input{&encoded, 0};
  auto read = [](void* opaque, uint8_t* buffer, int size) -> int {
    auto* input = static_cast<Input*>(opaque);
    const int64_t count = std::min<int64_t>(
        size, static_cast<int64_t>(input->bytes->size()) - input->offset);
    if (count <= 0) {
      return AVERROR_EOF;
    }
    std::memcpy(buffer, input->bytes->data() + input->offset, count);
    input->offset += count;
    return static_cast<int>(count);
  };
  auto seek = [](void* opaque, int64_t offset, int whence) -> int64_t {
    auto* input = static_cast<Input*>(opaque);
    const int64_t size = static_cast<int64_t>(input->bytes->size());
    if (whence == AVSEEK_SIZE) {
      return size;
    }
    const int mode = whence & ~AVSEEK_FORCE;
    int64_t next = offset;
    if (mode == SEEK_CUR) {
      next += input->offset;
    } else if (mode == SEEK_END) {
      next += size;
    } else if (mode != SEEK_SET) {
      return AVERROR(EINVAL);
    }
    if (next < 0 || next > size) {
      return AVERROR(EINVAL);
    }
    input->offset = next;
    return next;
  };
  auto free_io = [](AVIOContext* io) {
    if (io != nullptr) {
      av_freep(&io->buffer);
      avio_context_free(&io);
    }
  };
  auto close_format = [](AVFormatContext* format) {
    avformat_close_input(&format);
  };
  auto free_packet = [](AVPacket* packet) { av_packet_free(&packet); };
  auto* buffer = static_cast<uint8_t*>(av_malloc(4096));
  ASSERT_NE(buffer, nullptr);
  std::unique_ptr<AVIOContext, decltype(free_io)> io(
      avio_alloc_context(buffer, 4096, 0, &input, read, nullptr, seek),
      free_io);
  ASSERT_NE(io, nullptr);
  AVFormatContext* raw_format = avformat_alloc_context();
  ASSERT_NE(raw_format, nullptr);
  raw_format->pb = io.get();
  raw_format->flags |= AVFMT_FLAG_CUSTOM_IO;
  ASSERT_EQ(avformat_open_input(&raw_format, nullptr, nullptr, nullptr), 0);
  std::unique_ptr<AVFormatContext, decltype(close_format)> format(raw_format,
                                                                  close_format);
  ASSERT_GE(avformat_find_stream_info(format.get(), nullptr), 0);
  const int video_index =
      av_find_best_stream(format.get(), AVMEDIA_TYPE_VIDEO, -1, -1, nullptr, 0);
  ASSERT_GE(video_index, 0);
  const AVStream* stream = format->streams[video_index];
  EXPECT_EQ(av_cmp_q(stream->avg_frame_rate, frame_rate), 0);
  EXPECT_EQ(av_cmp_q(stream->r_frame_rate, frame_rate), 0);
  EXPECT_EQ(stream->nb_frames, expected_frames);
  const int64_t frame_duration =
      av_rescale_q(1, av_inv_q(frame_rate), stream->time_base);
  ASSERT_GT(frame_duration, 0);
  EXPECT_EQ(stream->duration, expected_frames * frame_duration);
  std::unique_ptr<AVPacket, decltype(free_packet)> packet(av_packet_alloc(),
                                                          free_packet);
  ASSERT_NE(packet, nullptr);
  std::vector<int64_t> presentation_times;
  presentation_times.reserve(expected_frames);
  int64_t last_packet_duration = 0;
  while (av_read_frame(format.get(), packet.get()) >= 0) {
    if (packet->stream_index == video_index) {
      EXPECT_EQ(packet->duration, frame_duration);
      EXPECT_EQ(packet->flags & AV_PKT_FLAG_DISCARD, 0);
      presentation_times.emplace_back(packet->pts);
      last_packet_duration = packet->duration;
    }
    av_packet_unref(packet.get());
  }
  EXPECT_EQ(presentation_times.size(), expected_frames);
  EXPECT_EQ(last_packet_duration, frame_duration);
  std::sort(presentation_times.begin(), presentation_times.end());
  for (size_t i = 0; i < presentation_times.size(); ++i) {
    EXPECT_EQ(presentation_times[i], static_cast<int64_t>(i) * frame_duration);
  }
}

void expect_full_video_round_trip(const std::string& encoded,
                                  int32_t expected_frames,
                                  double expected_fps) {
  FFmpegVideoDecoder decoder;
  torch::Tensor video;
  VideoMetadata metadata;
  ASSERT_TRUE(decoder.decode(encoded, video, metadata));
  EXPECT_EQ(metadata.total_num_frames, expected_frames);
  EXPECT_EQ(video.size(0), expected_frames);
  EXPECT_DOUBLE_EQ(metadata.fps, expected_fps);
  expect_cfr_packets(encoded, expected_frames, av_d2q(expected_fps, 1000000));
}

TEST(MMCodecTest, CfrTailPreservesAll124FramesAndStereoAudioAt24Fps) {
  DiTForwardOutput output;
  output.tensors = {torch::zeros({1, 124, 3, 16, 16}, torch::kFloat32)};
  output.audio_tensors = {torch::zeros({1, 2, 165334}, torch::kFloat32)};
  DiTEncodedMedia encoded;
  ASSERT_TRUE(
      encode_minimax_h3_video_output(output, 24.0, 32000, encoded).ok());
  expect_full_video_round_trip(encoded.data, 124, 24.0);
  FFmpegAudioDecoder decoder;
  torch::Tensor audio;
  AudioMetadata metadata;
  ASSERT_TRUE(decoder.decode(encoded.data, audio, metadata, 32000, 2));
  EXPECT_EQ(metadata.sample_rate, 32000);
  EXPECT_EQ(metadata.num_channels, 2);
  EXPECT_EQ(audio.size(0), 2);
  EXPECT_GE(audio.size(1), 165334);
  EXPECT_LT(audio.size(1), 165334 + 1024);
}

TEST(MMCodecTest, CfrTailPreservesAll124VideoOnlyFramesAt24Fps) {
  FFmpegVideoEncoder encoder;
  std::string encoded;
  ASSERT_TRUE(
      encoder.encode(torch::zeros({124, 3, 16, 16}), 24.0, "mp4", encoded));
  expect_full_video_round_trip(encoded, 124, 24.0);
}

TEST(MMCodecTest, CfrTailPreservesFractionalFrameRate) {
  FFmpegVideoEncoder encoder;
  std::string encoded;
  const double fps = 30000.0 / 1001.0;
  ASSERT_TRUE(
      encoder.encode(torch::zeros({7, 3, 16, 16}), fps, "mp4", encoded));
  expect_full_video_round_trip(encoded, 7, fps);
}

TEST(MMCodecTest, H3EncoderProducesDecodableVideoAndStereoAudio) {
  DiTForwardOutput output;
  output.tensors = {torch::zeros({1, 4, 3, 16, 16}, torch::kFloat32)};
  output.audio_tensors = {torch::zeros({1, 2, 4096}, torch::kFloat32)};
  DiTEncodedMedia encoded;
  const Status status =
      encode_minimax_h3_video_output(output, 25.0, 32000, encoded);
  ASSERT_TRUE(status.ok()) << status.message();
  ASSERT_FALSE(encoded.data.empty());
  EXPECT_EQ(encoded.mime_type, "video/mp4");
  EXPECT_EQ(encoded.container, "mp4");
  EXPECT_EQ(encoded.width, 16);
  EXPECT_EQ(encoded.height, 16);
  EXPECT_EQ(encoded.num_frames, 4);
  EXPECT_DOUBLE_EQ(encoded.fps, 25.0);
  EXPECT_EQ(encoded.audio_sample_rate, 32000);
  EXPECT_EQ(encoded.audio_channels, 2);
  FFmpegVideoDecoder video_decoder;
  torch::Tensor video;
  VideoMetadata video_metadata;
  ASSERT_TRUE(video_decoder.decode(encoded.data, video, video_metadata));
  EXPECT_TRUE(video.defined());
  EXPECT_GT(video.numel(), 0);
  EXPECT_EQ(video_metadata.total_num_frames, 4);
  EXPECT_DOUBLE_EQ(video_metadata.fps, 25.0);
  expect_cfr_packets(encoded.data, 4, AVRational{25, 1});
  FFmpegAudioDecoder audio_decoder;
  torch::Tensor audio;
  AudioMetadata audio_metadata;
  ASSERT_TRUE(audio_decoder.decode(encoded.data,
                                   audio,
                                   audio_metadata,
                                   /*target_sr=*/32000,
                                   /*target_channels=*/2));
  EXPECT_TRUE(audio.defined());
  EXPECT_GT(audio.numel(), 0);
  EXPECT_EQ(audio_metadata.sample_rate, 32000);
  EXPECT_EQ(audio_metadata.num_channels, 2);
}

TEST(MMCodecTest, H3EncodingFailureDoesNotReplaceDestination) {
  DiTForwardOutput output;
  output.tensors = {torch::zeros({1, 4, 3, 16, 16})};
  DiTEncodedMedia encoded;
  encoded.data = "previous";
  EXPECT_FALSE(
      encode_minimax_h3_video_output(output, 25.0, 32000, encoded).ok());
  EXPECT_EQ(encoded.data, "previous");
  output.audio_tensors = {torch::zeros({1, 2, 4096})};
  EXPECT_FALSE(
      encode_minimax_h3_video_output(output, 0.0, 32000, encoded).ok());
  EXPECT_FALSE(
      encode_minimax_h3_video_output(
          output, std::numeric_limits<double>::quiet_NaN(), 32000, encoded)
          .ok());
  EXPECT_FALSE(encode_minimax_h3_video_output(output, 25.0, 0, encoded).ok());
  output.tensors = {torch::zeros({4, 3, 16, 16})};
  EXPECT_FALSE(
      encode_minimax_h3_video_output(output, 25.0, 32000, encoded).ok());
  EXPECT_EQ(encoded.data, "previous");
}

}  // namespace
}  // namespace xllm
