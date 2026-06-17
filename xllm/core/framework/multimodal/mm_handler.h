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

#pragma once

#include <string>
#include <vector>

#include "core/common/message.h"
#include "core/common/types.h"
#include "mm_input.h"

namespace xllm {

struct MMInputItem;
struct MMPayload;
struct MMInput;

class MMHandlerBase {
 public:
  MMHandlerBase() = default;
  virtual ~MMHandlerBase() = default;

  MMErrCode process(const MMContent& content,
                    MMInputItem& input,
                    MMPayload& payload);

  virtual MMErrCode load(const MMContent& content,
                         MMInputItem& input,
                         MMPayload& payload) = 0;

  virtual MMErrCode decode(MMInputItem& input) = 0;

 protected:
  MMErrCode load_from_dataurl(const std::string& url,
                              std::string& data,
                              MMPayload& payload);

  MMErrCode load_from_local(const std::string& url, std::string& data);

  MMErrCode load_from_http(const std::string& url, std::string& data);

 protected:
  std::string httpurl_prefix_{"http"};
};

class ImageHandler : public MMHandlerBase {
 public:
  ImageHandler() = default;
  ~ImageHandler() = default;

  MMErrCode load(const MMContent& content,
                 MMInputItem& input,
                 MMPayload& payload) override;
  MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:image"};
};

class VideoHandler : public MMHandlerBase {
 public:
  VideoHandler() = default;
  ~VideoHandler() = default;

  MMErrCode load(const MMContent& content,
                 MMInputItem& input,
                 MMPayload& payload) override;
  MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:video"};
};

class AudioHandler : public MMHandlerBase {
 public:
  AudioHandler() = default;
  ~AudioHandler() = default;

  MMErrCode load(const MMContent& content,
                 MMInputItem& input,
                 MMPayload& payload) override;
  MMErrCode decode(MMInputItem& input) override;

 private:
  std::string dataurl_prefix_{"data:audio"};
};

class MMHandlerSet {
 public:
  MMHandlerSet();
  ~MMHandlerSet();

  MMErrCode process(const std::string& type,
                    const MMContent& content,
                    MMInputItem& input,
                    MMPayload& payload);

 private:
  std::unordered_map<std::string, std::unique_ptr<MMHandlerBase>> handlers_;
};

}  // namespace xllm
