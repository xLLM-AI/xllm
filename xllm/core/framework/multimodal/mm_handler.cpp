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

#include "mm_handler.h"

#include <butil/base64.h>
#include <butil/strings/string_number_conversions.h>
#include <glog/logging.h>

#include <fstream>
#include <iterator>

#include "common/global_flags.h"
#include "core/framework/config/model_config.h"
#include "core/util/http_downloader.h"
#include "mm_codec.h"
#include "mm_embedding_handler.h"

namespace xllm {

MMErrCode MMHandlerBase::process(const MMContent& content,
                                 MMInputItem& input,
                                 MMPayload& payload) {
  MMErrCode code = this->load(content, input, payload);
  if (code != MMErrCode::SUCCESS) return code;

  code = this->decode(input);
  if (code != MMErrCode::SUCCESS) return code;

  return MMErrCode::SUCCESS;
}

MMErrCode MMHandlerBase::load_from_dataurl(const std::string& url,
                                           std::string& raw_data,
                                           MMPayload& payload) {
  size_t pos = url.find_first_of(';');
  if (pos == std::string::npos) return MMErrCode::LOAD_DATA_ERR;

  butil::StringPiece sub(url, pos + 1);
  pos = sub.find_first_of(',');
  if (pos == std::string::npos) return MMErrCode::LOAD_DATA_ERR;

  butil::StringPiece type(sub, 0, pos);
  butil::StringPiece data(sub, pos + 1);

  if (type == "base64") {
    if (!butil::Base64Decode(data, &raw_data)) {
      return MMErrCode::LOAD_DATA_ERR;
    }
    return MMErrCode::SUCCESS;
  } else if (type == "binary") {
    size_t len = 0;
    bool res = butil::StringToSizeT(data, &len);
    if (res) {
      if (!payload.get(raw_data, len)) {
        LOG(ERROR) << "load data from binary url failed, url is: " << url;
        return MMErrCode::LOAD_DATA_ERR;
      }
      return MMErrCode::SUCCESS;
    } else {
      LOG(ERROR) << " data url is invalid, url is " << url;
      return MMErrCode::LOAD_DATA_ERR;
    }
  } else {
    LOG(ERROR) << " data url is invalid, url is " << url;
    return MMErrCode::LOAD_DATA_ERR;
  }
}

MMErrCode MMHandlerBase::load_from_local(const std::string& url,
                                         std::string& data) {
  std::string path = url;
  const std::string prefix = "file://";
  if (path.compare(0, prefix.size(), prefix) == 0) {
    path = path.substr(prefix.size());
  }

  std::ifstream in(path, std::ios::binary);
  if (!in) {
    LOG(ERROR) << "failed to open local file: " << path;
    return MMErrCode::LOAD_LOCAL_ERR;
  }

  data.assign(std::istreambuf_iterator<char>(in),
              std::istreambuf_iterator<char>());
  return MMErrCode::SUCCESS;
}

MMErrCode MMHandlerBase::load_from_http(const std::string& url,
                                        std::string& data) {
  BRpcDownloader helper_;
  if (!helper_.fetch_data(url, data)) {
    return MMErrCode::LOAD_HTTP_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMErrCode ImageHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();

  const auto& image_url = content.image_url;
  const auto& url = image_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::IMAGE;
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::IMAGE;
    return this->load_from_http(url, input.raw_data);
  } else {
    // treat as local path or file:// url
    input.type = MMType::IMAGE;
    if (this->load_from_local(url, input.raw_data) == MMErrCode::SUCCESS) {
      return MMErrCode::SUCCESS;
    }
    LOG(ERROR) << " image url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode ImageHandler::decode(MMInputItem& input) {
  OpenCVImageDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_image)) {
    return MMErrCode::DECODE_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMErrCode VideoHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();

  const auto& video_url = content.video_url;
  const auto& url = video_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::VIDEO;
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::VIDEO;
    return this->load_from_http(url, input.raw_data);
  } else {
    LOG(ERROR) << " video url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode VideoHandler::decode(MMInputItem& input) {
  if (::xllm::ModelConfig::get_instance().use_audio_in_video()) {
    FFmpegAudioDecoder audio_decoder;
    if (audio_decoder.decode(
            input.raw_data, input.decode_audio, input.audio_meta)) {
      input.type |= MMType::AUDIO;
    } else {
      LOG(ERROR) << "decode audio in video failed";
      return MMErrCode::DECODE_ERR;
    }
  }

  FFmpegVideoDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_video, input.video_meta)) {
    return MMErrCode::DECODE_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMErrCode AudioHandler::load(const MMContent& content,
                             MMInputItem& input,
                             MMPayload& payload) {
  input.clear();

  const auto& audio_url = content.audio_url;
  const auto& url = audio_url.url;

  if (url.compare(0, dataurl_prefix_.size(), dataurl_prefix_) ==
      0) {  // data url

    input.type = MMType::AUDIO;
    return this->load_from_dataurl(url, input.raw_data, payload);
  } else if (url.compare(0, httpurl_prefix_.size(), httpurl_prefix_) ==
             0) {  // http url

    input.type = MMType::AUDIO;
    return this->load_from_http(url, input.raw_data);
  } else {
    LOG(ERROR) << " audio url is invalid, url is " << url;
    return MMErrCode::INVALID_URL_ERR;
  }
}

MMErrCode AudioHandler::decode(MMInputItem& input) {
  FFmpegAudioDecoder decoder;
  if (!decoder.decode(input.raw_data, input.decode_audio, input.audio_meta)) {
    return MMErrCode::DECODE_ERR;
  }
  return MMErrCode::SUCCESS;
}

MMHandlerSet::MMHandlerSet() {
  handlers_["image_url"] = std::make_unique<ImageHandler>();
  handlers_["video_url"] = std::make_unique<VideoHandler>();
  handlers_["audio_url"] = std::make_unique<AudioHandler>();
  handlers_["image_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::IMAGE);
  handlers_["video_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::VIDEO);
  handlers_["audio_embedding"] =
      std::make_unique<MMEmbeddingHandler>(MMType::AUDIO);
}

MMHandlerSet::~MMHandlerSet() {}

MMErrCode MMHandlerSet::process(const std::string& type,
                                const MMContent& content,
                                MMInputItem& input,
                                MMPayload& payload) {
  auto itor = handlers_.find(type);
  if (itor == handlers_.end()) {
    return MMErrCode::HANDLER_ERR;
  }

  auto& handler = itor->second;
  return handler->process(content, input, payload);
}

}  // namespace xllm
