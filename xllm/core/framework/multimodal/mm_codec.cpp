/* Copyright 2025 The xLLM Authors. All Rights Reserved.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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
#include "mm_codec.h"

#include <algorithm>
#include <cmath>

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/channel_layout.h>
#include <libavutil/opt.h>
#include <libavutil/samplefmt.h>
#include <libswresample/swresample.h>
#include <libswscale/swscale.h>
}

namespace xllm {

namespace {

struct MemCtx {
  const uint8_t* mem_ptr;
  int64_t size;
  int64_t offset;
};

struct Reader {
  // AVIO read callback
  static int32_t read(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemCtx*>(opaque);
    if (mc->offset < 0) {
      return AVERROR(EINVAL);
    }
    int64_t remain = mc->size - mc->offset;
    int64_t n = std::min(remain, static_cast<int64_t>(buf_size));
    if (n <= 0) {
      return AVERROR_EOF;
    }
    std::memcpy(buf, mc->mem_ptr + mc->offset, static_cast<size_t>(n));
    mc->offset += n;
    return static_cast<int32_t>(n);
  }

  // AVIO seek callback
  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemCtx*>(opaque);

    if (whence == AVSEEK_SIZE) {
      return mc->size;
    }

    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->offset + offset;
        break;
      case SEEK_END:
        pos = mc->size + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }

    if (pos < 0 || pos > mc->size) {
      return AVERROR(EINVAL);
    }

    mc->offset = pos;
    return pos;
  }
};

// Downstream multimodal preprocess expects a 3-channel RGB image. For BGRA
// input, blend alpha onto a white background instead of dropping alpha
// directly, so transparent regions keep visually stable colors.
void blend_bgra_to_rgb_with_white_background(const cv::Mat& image,
                                             cv::Mat& rgb_image) {
  rgb_image = cv::Mat(image.rows, image.cols, CV_8UC3);

  for (int32_t row = 0; row < image.rows; ++row) {
    const cv::Vec4b* src_row = image.ptr<cv::Vec4b>(row);
    cv::Vec3b* dst_row = rgb_image.ptr<cv::Vec3b>(row);

    for (int32_t col = 0; col < image.cols; ++col) {
      const float alpha = src_row[col][3] / 255.0f;
      const float blue = src_row[col][0];
      const float green = src_row[col][1];
      const float red = src_row[col][2];

      dst_row[col][0] = static_cast<uint8_t>(
          std::round(red * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][1] = static_cast<uint8_t>(
          std::round(green * alpha + 255.0f * (1.0f - alpha)));
      dst_row[col][2] = static_cast<uint8_t>(
          std::round(blue * alpha + 255.0f * (1.0f - alpha)));
    }
  }
}

// OpenCV decodes different image formats into different channel layouts:
// 1 channel -> GRAY, 3 channels -> BGR, 4 channels -> BGRA. Normalize them
// here so the decoder path always returns a 3-channel RGB image.
void convert_decoded_image_to_rgb(const cv::Mat& image, cv::Mat& rgb_image) {
  const int32_t channels = image.channels();

  if (channels == 4) {
    blend_bgra_to_rgb_with_white_background(image, rgb_image);
  } else if (channels == 3) {
    cv::cvtColor(image, rgb_image, cv::COLOR_BGR2RGB);
  } else if (channels == 1) {
    cv::cvtColor(image, rgb_image, cv::COLOR_GRAY2RGB);
  } else {
    LOG(FATAL) << "unsupported channel count: " << channels;
  }
}

}  // namespace

class MemoryMediaReader {
 public:
  MemoryMediaReader(const uint8_t* data, size_t size) {
    mc_.mem_ptr = data;
    if (size > static_cast<size_t>(INT64_MAX)) {
      LOG(FATAL) << "MemCtx size too large";
    }
    mc_.size = static_cast<int64_t>(size);
    mc_.offset = 0;
  }

  ~MemoryMediaReader() {
    if (frm_) {
      av_frame_free(&frm_);
    }
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      avformat_close_input(&fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    } else if (avio_buf_) {
      av_freep(reinterpret_cast<void**>(&avio_buf_));
    }
  }

  bool init(AVMediaType type) {
    fmt_ctx_ = avformat_alloc_context();
    if (!fmt_ctx_) {
      return false;
    }
    constexpr int32_t avio_buf_sz = 1 << 16;
    avio_buf_ =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
    if (!avio_buf_) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(
        avio_buf_, avio_buf_sz, 0, &mc_, &Reader::read, nullptr, &Reader::seek);
    if (!avio_ctx_) {
      return false;
    }
    avio_buf_ = nullptr;

    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    if (avformat_open_input(&fmt_ctx_, nullptr, nullptr, nullptr) < 0) {
      return false;
    }

    if (avformat_find_stream_info(fmt_ctx_, nullptr) < 0) {
      return false;
    }

    stream_index_ = av_find_best_stream(fmt_ctx_, type, -1, -1, nullptr, 0);
    if (stream_index_ < 0) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    const AVCodec* codec = avcodec_find_decoder(st->codecpar->codec_id);
    if (!codec) {
      return false;
    }

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    if (avcodec_parameters_to_context(codec_ctx_, st->codecpar) < 0 ||
        avcodec_open2(codec_ctx_, codec, nullptr) < 0) {
      return false;
    }

    pkt_ = av_packet_alloc();
    frm_ = av_frame_alloc();
    if (!pkt_ || !frm_) {
      return false;
    }

    return true;
  }

  bool decode() {
    CHECK(fmt_ctx_ && codec_ctx_ && pkt_ && frm_) << "ffmpeg init failed";
    CHECK_GE(stream_index_, 0) << "stream index not found";

    // read packets, send to decoder, pull frames
    while (av_read_frame(fmt_ctx_, pkt_) >= 0) {
      if (pkt_->stream_index == stream_index_) {
        if (avcodec_send_packet(codec_ctx_, pkt_) == 0) {
          while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
            // handle_frame: video->push RGB frame, audio->append PCM
            if (!handle_frame(frm_)) {
              av_packet_unref(pkt_);
              return false;
            }
          }
        }
      }
      av_packet_unref(pkt_);
    }

    // flush decoder at end of stream
    avcodec_send_packet(codec_ctx_, nullptr);
    while (avcodec_receive_frame(codec_ctx_, frm_) == 0) {
      if (!handle_frame(frm_)) {
        return false;
      }
    }

    return true;
  }

  // video->RGB tensor, audio->PCM samples
  virtual bool handle_frame(AVFrame* f) = 0;

 protected:
  AVFormatContext* fmt_ctx_ = nullptr;
  uint8_t* avio_buf_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVFrame* frm_ = nullptr;
  MemCtx mc_{nullptr, 0, 0};
  int32_t stream_index_ = -1;
};

class MemoryVideoReader : public MemoryMediaReader {
 public:
  MemoryVideoReader(const uint8_t* data, size_t size)
      : MemoryMediaReader(data, size) {}

  ~MemoryVideoReader() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (rgb_frame_) {
      av_frame_free(&rgb_frame_);
    }
  }

  bool init(VideoMetadata& metadata) {
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_VIDEO)) {
      return false;
    }

    // init VideoMetadata
    AVStream* st = fmt_ctx_->streams[stream_index_];
    AVRational r =
        st->avg_frame_rate.num ? st->avg_frame_rate : st->r_frame_rate;
    metadata.fps = (r.num && r.den) ? av_q2d(r) : 0.0;
    metadata.total_num_frames = 0;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, VideoMetadata& metadata) {
    CHECK(frames_.empty()) << "frames is not cleared before read";

    if (!decode()) {
      return false;
    }
    if (frames_.empty()) {
      return false;
    }

    tensor = torch::stack(frames_);  // [T,C,H,W]
    metadata.total_num_frames = static_cast<int32_t>(frames_.size());
    metadata.duration =
        (metadata.fps > 0.0)
            ? static_cast<double>(metadata.total_num_frames) / metadata.fps
            : 0.0;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    // init colorspace converter once based on first frame
    if (!sws_ctx_) {
      sws_ctx_ = sws_getContext(f->width,
                                f->height,
                                static_cast<AVPixelFormat>(f->format),
                                f->width,
                                f->height,
                                AV_PIX_FMT_RGB24,
                                SWS_BILINEAR,
                                nullptr,
                                nullptr,
                                nullptr);
      if (!sws_ctx_) {
        return false;
      }
    }

    // use an FFmpeg-allocated frame so sws_scale writes into a buffer with the
    // correct padded linesize
    if (!rgb_frame_) {
      rgb_frame_ = av_frame_alloc();
      if (!rgb_frame_) {
        return false;
      }
    }

    // (re)allocate the RGB buffer when input changes
    if (rgb_frame_->width != f->width || rgb_frame_->height != f->height ||
        rgb_frame_->format != AV_PIX_FMT_RGB24 || !rgb_frame_->data[0]) {
      av_frame_unref(rgb_frame_);
      rgb_frame_->format = AV_PIX_FMT_RGB24;
      rgb_frame_->width = f->width;
      rgb_frame_->height = f->height;
      if (av_frame_get_buffer(rgb_frame_, 0) < 0) {
        return false;
      }
    }
    if (av_frame_make_writable(rgb_frame_) < 0) {
      return false;
    }

    // convert the current decoded frame into RGB24
    if (sws_scale(sws_ctx_,
                  f->data,
                  f->linesize,
                  0,
                  f->height,
                  rgb_frame_->data,
                  rgb_frame_->linesize) != f->height) {
      return false;
    }

    // build CHW uint8 tensor
    const int64_t H = f->height;
    const int64_t W = f->width;
    const int64_t src_ls = rgb_frame_->linesize[0];

    auto rgb = torch::from_blob(rgb_frame_->data[0],
                                {3, H, W},  // [C,H,W]
                                {1, src_ls, 3},
                                torch::TensorOptions().dtype(torch::kUInt8))
                   .contiguous();

    frames_.emplace_back(rgb.clone());
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* rgb_frame_ = nullptr;
  std::vector<torch::Tensor> frames_;
};

class MemoryAudioReader : public MemoryMediaReader {
 public:
  MemoryAudioReader(const uint8_t* data, size_t size, int64_t target_sr = 16000)
      : MemoryMediaReader(data, size) {
    target_sr_ = target_sr;
  }

  ~MemoryAudioReader() {
    if (swr_ctx_) {
      swr_free(&swr_ctx_);
    }
  }

  bool init(AudioMetadata& metadata) {
    if (!MemoryMediaReader::init(AVMEDIA_TYPE_AUDIO)) {
      return false;
    }

    AVStream* st = fmt_ctx_->streams[stream_index_];
    codec_ctx_->pkt_timebase = st->time_base;

    // setup resampler
    swr_ctx_ = swr_alloc();
    if (!swr_ctx_) {
      return false;
    }

    AVChannelLayout in_layout;
    if (av_channel_layout_copy(&in_layout, &codec_ctx_->ch_layout) < 0) {
      return false;
    }

    AVChannelLayout out_layout;
    av_channel_layout_default(&out_layout, target_ch_);

    if (swr_alloc_set_opts2(&swr_ctx_,
                            &out_layout,
                            AV_SAMPLE_FMT_FLT,
                            target_sr_,
                            &in_layout,
                            codec_ctx_->sample_fmt,
                            codec_ctx_->sample_rate,
                            0,
                            nullptr) < 0) {
      av_channel_layout_uninit(&out_layout);
      av_channel_layout_uninit(&in_layout);
      return false;
    }

    av_channel_layout_uninit(&out_layout);
    av_channel_layout_uninit(&in_layout);

    // if downmixing stereo -> mono, use customized remix matrix (L+R)/2
    int32_t in_ch = codec_ctx_->ch_layout.nb_channels;
    if (target_ch_ == 1 && in_ch == 2) {
      constexpr double matrix[2] = {0.5, 0.5};
      if (swr_set_matrix(swr_ctx_, matrix, in_ch) < 0) {
        return false;
      }
    }

    if (swr_init(swr_ctx_) < 0) {
      return false;
    }

    // init AudioMetadata
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    metadata.duration = 0.0;
    return true;
  }

  bool read(torch::Tensor& tensor, AudioMetadata& metadata) {
    CHECK(swr_ctx_) << "SwrContext is null";
    CHECK(pcm_.empty()) << "PCM buffer is not cleared before read";

    if (!decode()) {
      return false;
    }

    // flush resampler buffered samples after decode
    while (true) {
      if (resample_to_pcm(nullptr, 0) <= 0) {
        break;
      }
    }

    if (pcm_.empty()) {
      return false;
    }

    // build output tensor and compute metadata
    if (target_ch_ == 1) {
      tensor = torch::from_blob(pcm_.data(),
                                {static_cast<int32_t>(pcm_.size())},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .clone();
      metadata.duration = static_cast<double>(pcm_.size()) / target_sr_;
    } else {
      int32_t T =
          static_cast<int32_t>(pcm_.size() / static_cast<size_t>(target_ch_));
      tensor = torch::from_blob(pcm_.data(),
                                {T, target_ch_},
                                torch::TensorOptions().dtype(torch::kFloat32))
                   .permute({1, 0})
                   .clone();
      metadata.duration = static_cast<double>(T) / target_sr_;
    }
    metadata.sample_rate = target_sr_;
    metadata.num_channels = target_ch_;
    return true;
  }

  bool handle_frame(AVFrame* f) override {
    return resample_to_pcm((const uint8_t**)f->extended_data, f->nb_samples) >=
           0;
  }

  int32_t resample_to_pcm(const uint8_t** in_data, int32_t nb_samples) {
    int32_t out_nb = swr_get_out_samples(swr_ctx_, nb_samples);
    if (out_nb < 0) {
      return out_nb;
    }
    if (out_nb == 0) {
      return 0;
    }

    std::vector<float> out_buf(static_cast<size_t>(out_nb) *
                               static_cast<size_t>(target_ch_));
    uint8_t* out_data[1] = {reinterpret_cast<uint8_t*>(out_buf.data())};

    // convert input frame samples to target format
    int32_t converted =
        swr_convert(swr_ctx_, out_data, out_nb, in_data, nb_samples);
    if (converted < 0) {
      return converted;
    }
    if (converted == 0) {
      return 0;
    }

    // append converted samples to pcm buffer
    const int64_t n = static_cast<int64_t>(converted * target_ch_);
    pcm_.reserve(pcm_.size() + static_cast<size_t>(n));
    pcm_.insert(pcm_.end(), out_buf.data(), out_buf.data() + n);
    return converted;
  }

 private:
  SwrContext* swr_ctx_ = nullptr;
  int32_t target_sr_ = 16000;
  int32_t target_ch_ = 1;
  std::vector<float> pcm_;
};

bool OpenCVImageDecoder::decode(const std::string& raw_data, torch::Tensor& t) {
  cv::Mat buffer(1, raw_data.size(), CV_8UC1, (void*)raw_data.data());
  if (raw_data.empty()) {
    LOG(ERROR) << "opencv image decode got empty data";
    return false;
  }
  cv::Mat image = cv::imdecode(buffer, cv::IMREAD_UNCHANGED);
  if (image.empty()) {
    LOG(INFO) << "opencv image decode failed";
    return false;
  }

  cv::Mat rgb_image;
  convert_decoded_image_to_rgb(image, rgb_image);

  torch::Tensor tensor =
      torch::from_blob(rgb_image.data,
                       {rgb_image.rows, rgb_image.cols, 3},
                       torch::TensorOptions().dtype(torch::kUInt8));

  t = tensor.permute({2, 0, 1}).clone();  // [C, H, W]
  return true;
}

bool OpenCVImageEncoder::encode(const torch::Tensor& t, std::string& raw_data) {
  if (!valid(t)) {
    return false;
  }

  auto img = t.permute({1, 2, 0}).contiguous();
  cv::Mat mat(img.size(0), img.size(1), CV_32FC3, img.data_ptr<float>());

  cv::Mat mat_8u;
  mat.convertTo(mat_8u, CV_8UC3, 255.0);

  // rgb -> bgr
  cv::cvtColor(mat_8u, mat_8u, cv::COLOR_RGB2BGR);

  std::vector<uchar> data;
  if (!cv::imencode(".png", mat_8u, data)) {
    LOG(ERROR) << "image encode faild";
    return false;
  }

  raw_data.assign(data.begin(), data.end());
  return true;
}

bool OpenCVImageEncoder::valid(const torch::Tensor& t) {
  if (t.dim() != 3 || t.size(0) != 3) {
    LOG(ERROR) << "input tensor must be 3HW  tensor";
    return false;
  }

  if (t.scalar_type() != torch::kFloat32 || !t.device().is_cpu()) {
    LOG(ERROR) << "tensor must be cpu float32";
    return false;
  }

  return true;
}

bool FFmpegVideoDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                VideoMetadata& metadata) {
  MemoryVideoReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size());

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "video decode faild";
    return false;
  }
  return true;
}

bool FFmpegAudioDecoder::decode(const std::string& raw_data,
                                torch::Tensor& t,
                                AudioMetadata& metadata,
                                int64_t target_sr) {
  MemoryAudioReader reader(reinterpret_cast<const uint8_t*>(raw_data.data()),
                           raw_data.size(),
                           target_sr);

  if (!reader.init(metadata) || !reader.read(t, metadata)) {
    LOG(INFO) << "audio decode faild";
    return false;
  }
  return true;
}

// ---- MemoryMediaWriter (in-memory encoding base class) ----

namespace {

struct MemWriteCtx {
  std::vector<uint8_t>* buf;
  int64_t pos;
};

struct Writer {
  static int32_t write(void* opaque, uint8_t* buf, int32_t buf_size) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    int64_t end_pos = mc->pos + buf_size;
    if (end_pos > static_cast<int64_t>(mc->buf->size())) {
      mc->buf->resize(static_cast<size_t>(end_pos), 0);
    }
    std::memcpy(mc->buf->data() + mc->pos, buf, static_cast<size_t>(buf_size));
    mc->pos = end_pos;
    return buf_size;
  }

  static int64_t seek(void* opaque, int64_t offset, int32_t whence) {
    auto* mc = static_cast<MemWriteCtx*>(opaque);
    if (whence == AVSEEK_SIZE) {
      return static_cast<int64_t>(mc->buf->size());
    }
    int64_t pos = 0;
    switch (whence) {
      case SEEK_SET:
        pos = offset;
        break;
      case SEEK_CUR:
        pos = mc->pos + offset;
        break;
      case SEEK_END:
        pos = static_cast<int64_t>(mc->buf->size()) + offset;
        break;
      default:
        return AVERROR(EINVAL);
    }
    if (pos < 0) {
      return AVERROR(EINVAL);
    }
    mc->pos = pos;
    return pos;
  }
};

}  // namespace

class MemoryMediaWriter {
 public:
  MemoryMediaWriter() = default;

  virtual ~MemoryMediaWriter() {
    if (pkt_) {
      av_packet_free(&pkt_);
    }
    if (codec_ctx_) {
      avcodec_free_context(&codec_ctx_);
    }
    if (fmt_ctx_) {
      if (!finished_) {
        av_write_trailer(fmt_ctx_);
      }
      avformat_free_context(fmt_ctx_);
    }
    if (avio_ctx_) {
      av_freep(&avio_ctx_->buffer);
      avio_context_free(&avio_ctx_);
    }
  }

 protected:
  bool init(const char* format,
            AVCodecID codec_id,
            int32_t width,
            int32_t height,
            double fps,
            AVPixelFormat pix_fmt,
            AVDictionary** opts = nullptr) {
    const AVCodec* codec = avcodec_find_encoder(codec_id);
    if (!codec) {
      LOG(ERROR) << "MemoryMediaWriter: encoder not found, codec_id="
                 << avcodec_get_name(codec_id);
      return false;
    }

    constexpr int32_t avio_buf_sz = 1 << 16;
    uint8_t* avio_buf =
        static_cast<uint8_t*>(av_malloc(static_cast<size_t>(avio_buf_sz)));
    if (!avio_buf) {
      return false;
    }

    avio_ctx_ = avio_alloc_context(avio_buf,
                                   avio_buf_sz,
                                   1,
                                   &write_ctx_,
                                   nullptr,
                                   &Writer::write,
                                   &Writer::seek);
    if (!avio_ctx_) {
      av_freep(reinterpret_cast<void**>(&avio_buf));
      return false;
    }
    avio_ctx_->seekable = AVIO_SEEKABLE_NORMAL;

    const AVOutputFormat* fmt = av_guess_format(format, nullptr, nullptr);
    if (!fmt) {
      LOG(ERROR) << "MemoryMediaWriter: no muxer for " << format;
      return false;
    }

    if (avformat_alloc_output_context2(&fmt_ctx_, fmt, nullptr, nullptr) < 0 ||
        !fmt_ctx_) {
      return false;
    }
    fmt_ctx_->pb = avio_ctx_;
    fmt_ctx_->flags |= AVFMT_FLAG_CUSTOM_IO;

    codec_ctx_ = avcodec_alloc_context3(codec);
    if (!codec_ctx_) {
      return false;
    }

    codec_ctx_->width = width;
    codec_ctx_->height = height;
    codec_ctx_->time_base = {1, static_cast<int32_t>(std::llround(fps))};
    codec_ctx_->framerate = {static_cast<int32_t>(std::llround(fps)), 1};
    codec_ctx_->pix_fmt = pix_fmt;

    if (fmt_ctx_->oformat->flags & AVFMT_GLOBALHEADER) {
      codec_ctx_->flags |= AV_CODEC_FLAG_GLOBAL_HEADER;
    }

    if (avcodec_open2(codec_ctx_, codec, opts) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avcodec_open2 failed for "
                 << codec->name;
      return false;
    }

    stream_ = avformat_new_stream(fmt_ctx_, nullptr);
    if (!stream_) {
      return false;
    }
    stream_->time_base = codec_ctx_->time_base;
    if (avcodec_parameters_from_context(stream_->codecpar, codec_ctx_) < 0) {
      return false;
    }

    if (avformat_write_header(fmt_ctx_, nullptr) < 0) {
      LOG(ERROR) << "MemoryMediaWriter: avformat_write_header failed";
      return false;
    }

    pkt_ = av_packet_alloc();
    if (!pkt_) {
      return false;
    }

    LOG(INFO) << "MemoryMediaWriter: initialized " << codec->name << " ["
              << format << "] " << width << "x" << height << " @ " << fps
              << " fps";
    return true;
  }

  bool send_frame(AVFrame* frame) {
    if (avcodec_send_frame(codec_ctx_, frame) < 0) {
      return false;
    }
    return drain_packets();
  }

  bool finish() {
    avcodec_send_frame(codec_ctx_, nullptr);
    if (!drain_packets()) {
      return false;
    }
    av_write_trailer(fmt_ctx_);
    finished_ = true;
    return true;
  }

  std::vector<uint8_t> take_output() { return std::move(out_buf_); }

  AVCodecContext* codec_ctx() { return codec_ctx_; }
  AVStream* stream() { return stream_; }

  bool drain_packets() {
    while (avcodec_receive_packet(codec_ctx_, pkt_) == 0) {
      av_packet_rescale_ts(pkt_, codec_ctx_->time_base, stream_->time_base);
      pkt_->stream_index = stream_->index;
      if (av_interleaved_write_frame(fmt_ctx_, pkt_) < 0) {
        av_packet_unref(pkt_);
        return false;
      }
      av_packet_unref(pkt_);
    }
    return true;
  }

  AVFormatContext* fmt_ctx_ = nullptr;
  AVIOContext* avio_ctx_ = nullptr;
  AVCodecContext* codec_ctx_ = nullptr;
  AVPacket* pkt_ = nullptr;
  AVStream* stream_ = nullptr;
  MemWriteCtx write_ctx_{&out_buf_, 0};
  std::vector<uint8_t> out_buf_;
  bool finished_ = false;
};

class MemoryVideoWriter final : public MemoryMediaWriter {
 public:
  MemoryVideoWriter() = default;

  ~MemoryVideoWriter() {
    if (sws_ctx_) {
      sws_freeContext(sws_ctx_);
    }
    if (yuv_frame_) {
      av_frame_free(&yuv_frame_);
    }
  }

  bool write(const torch::Tensor& video,
             double fps,
             const std::string& format,
             std::string& raw_data) {
    if (video.dim() != 4 || video.size(1) != 3) {
      LOG(ERROR) << "MemoryVideoWriter: expects [T,C,H,W] with C=3, got "
                 << video.sizes();
      return false;
    }
    if (video.scalar_type() != torch::kFloat32 || !video.device().is_cpu()) {
      LOG(ERROR) << "MemoryVideoWriter: expects cpu float32 tensor";
      return false;
    }

    const int64_t T = video.size(0);
    const int64_t H = video.size(2);
    const int64_t W = video.size(3);
    if (T == 0 || H == 0 || W == 0) {
      LOG(ERROR) << "MemoryVideoWriter: empty dimensions T=" << T << " H=" << H
                 << " W=" << W;
      return false;
    }

    AVCodecID codec_id;
    AVPixelFormat pix_fmt;
    AVDictionary* opts = nullptr;

    if (format == "avi") {
      codec_id = AV_CODEC_ID_MJPEG;
      pix_fmt = AV_PIX_FMT_YUVJ420P;
    } else {
      const AVCodec* x264_codec = avcodec_find_encoder_by_name("libx264");

      if (x264_codec) {
        codec_id = x264_codec->id;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "crf", "18", 0);
        av_dict_set(&opts, "preset", "medium", 0);
        av_dict_set(&opts, "profile", "high", 0);
        av_dict_set(&opts, "level", "4.1", 0);
        LOG(INFO) << "Using libx264 H.264 encoder with CRF=18";
      } else {
        codec_id = AV_CODEC_ID_MPEG4;
        pix_fmt = AV_PIX_FMT_YUV420P;
        av_dict_set(&opts, "mbd", "2", 0);
        LOG(WARNING) << "libx264 not available, using MPEG4 fallback";
      }
    }

    if (!init(format.c_str(),
              codec_id,
              static_cast<int32_t>(W),
              static_cast<int32_t>(H),
              fps,
              pix_fmt,
              &opts)) {
      if (opts) av_dict_free(&opts);
      return false;
    }
    if (opts) av_dict_free(&opts);

    sws_ctx_ = sws_getContext(static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              AV_PIX_FMT_RGB24,
                              static_cast<int32_t>(W),
                              static_cast<int32_t>(H),
                              pix_fmt,
                              SWS_BILINEAR,
                              nullptr,
                              nullptr,
                              nullptr);
    if (!sws_ctx_) {
      LOG(ERROR) << "MemoryVideoWriter: sws_getContext failed";
      return false;
    }

    yuv_frame_ = av_frame_alloc();
    if (!yuv_frame_) {
      return false;
    }
    yuv_frame_->format = pix_fmt;
    yuv_frame_->width = static_cast<int32_t>(W);
    yuv_frame_->height = static_cast<int32_t>(H);
    if (av_frame_get_buffer(yuv_frame_, 0) < 0) {
      return false;
    }

    auto video_acc = video.accessor<float, 4>();
    const int64_t stride = W * 3;
    std::vector<uint8_t> rgb_buf(static_cast<size_t>(H * stride));
    int64_t pts = 0;

    for (int64_t t = 0; t < T; ++t) {
      for (int64_t y = 0; y < H; ++y) {
        for (int64_t x = 0; x < W; ++x) {
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 0)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][0][y][x] * 255.0f, 0.0f, 255.0f));
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 1)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][1][y][x] * 255.0f, 0.0f, 255.0f));
          rgb_buf[static_cast<size_t>(y * stride + x * 3 + 2)] =
              static_cast<uint8_t>(
                  std::clamp(video_acc[t][2][y][x] * 255.0f, 0.0f, 255.0f));
        }
      }

      const uint8_t* src_data[1] = {rgb_buf.data()};
      int32_t src_linesize[1] = {static_cast<int32_t>(stride)};

      if (av_frame_make_writable(yuv_frame_) < 0) {
        return false;
      }
      sws_scale(sws_ctx_,
                src_data,
                src_linesize,
                0,
                static_cast<int32_t>(H),
                yuv_frame_->data,
                yuv_frame_->linesize);
      yuv_frame_->pts = pts++;

      if (!send_frame(yuv_frame_)) {
        return false;
      }
    }

    if (!finish()) {
      return false;
    }

    auto out = take_output();
    raw_data.assign(out.begin(), out.end());

    LOG(INFO) << "MemoryVideoWriter: encoded " << T << " frames (" << W << "x"
              << H << ") at " << fps << " fps [" << format << "], output "
              << out.size() << " bytes";
    return true;
  }

 private:
  SwsContext* sws_ctx_ = nullptr;
  AVFrame* yuv_frame_ = nullptr;
};

bool FFmpegVideoEncoder::encode(const torch::Tensor& video,
                                double fps,
                                const std::string& format,
                                std::string& raw_data) {
  MemoryVideoWriter writer;
  return writer.write(video, fps, format, raw_data);
}

}  // namespace xllm
