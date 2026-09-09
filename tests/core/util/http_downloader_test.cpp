/* Copyright 2025-2026 The xLLM Authors.

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

#include "core/util/http_downloader.h"

#include <gtest/gtest.h>

#include "core/common/message.h"

namespace xllm {

// ============================================================
// parse_headers_json tests (pure function, no caching)
// ============================================================

TEST(ParseHeadersJsonTest, EmptyStringReturnsEmptyMap) {
  auto result = parse_headers_json("");
  EXPECT_TRUE(result.empty());
}

TEST(ParseHeadersJsonTest, ValidJsonParsesCorrectly) {
  auto result = parse_headers_json(
      R"({"Authorization":"Bearer token123","Referer":"https://example.com"})");
  EXPECT_EQ(2, result.size());
  EXPECT_EQ("Bearer token123", result["Authorization"]);
  EXPECT_EQ("https://example.com", result["Referer"]);
}

TEST(ParseHeadersJsonTest, SingleHeader) {
  auto result = parse_headers_json(R"({"X-Custom":"value"})");
  EXPECT_EQ(1, result.size());
  EXPECT_EQ("value", result["X-Custom"]);
}

TEST(ParseHeadersJsonTest, InvalidJsonReturnsEmptyMap) {
  auto result = parse_headers_json("not-valid-json");
  EXPECT_TRUE(result.empty());
}

TEST(ParseHeadersJsonTest, EmptyJsonObject) {
  auto result = parse_headers_json("{}");
  EXPECT_TRUE(result.empty());
}

TEST(ParseHeadersJsonTest, HeaderValueWithSpecialChars) {
  auto result = parse_headers_json(
      R"({"Authorization":"Bearer eyJhbGciOiJIUzI1NiJ9.abc.xyz"})");
  EXPECT_EQ(1, result.size());
  EXPECT_EQ("Bearer eyJhbGciOiJIUzI1NiJ9.abc.xyz", result["Authorization"]);
}

TEST(ParseHeadersJsonTest, MultipleCustomHeaders) {
  auto result = parse_headers_json(
      R"({"X-Key-1":"val1","X-Key-2":"val2","X-Key-3":"val3"})");
  EXPECT_EQ(3, result.size());
  EXPECT_EQ("val1", result["X-Key-1"]);
  EXPECT_EQ("val2", result["X-Key-2"]);
  EXPECT_EQ("val3", result["X-Key-3"]);
}

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

// ============================================================
// Header merge order: global < request
// ============================================================

TEST(HeaderMergeTest, RequestHeadersOverrideGlobal) {
  // Simulate the merge order used in BRpcDownloader::download():
  // 1) global defaults first
  // 2) per-request headers second (SetHeader overrides)

  std::unordered_map<std::string, std::string> global;
  global["Authorization"] = "Bearer global";
  global["Referer"] = "https://global-ref.com";

  std::unordered_map<std::string, std::string> request;
  request["Authorization"] = "Bearer specific";  // overrides global
  request["X-Custom"] = "custom-value";          // request-only

  // Merge: global first, then request
  std::unordered_map<std::string, std::string> merged;
  for (const auto& [k, v] : global) {
    merged[k] = v;
  }
  for (const auto& [k, v] : request) {
    merged[k] = v;  // request overrides
  }

  // Request value wins for shared key
  EXPECT_EQ("Bearer specific", merged["Authorization"]);
  // Global value preserved for non-overlapping key
  EXPECT_EQ("https://global-ref.com", merged["Referer"]);
  // Request-only key present
  EXPECT_EQ("custom-value", merged["X-Custom"]);
  EXPECT_EQ(3, merged.size());
}

TEST(HeaderMergeTest, EmptyRequestHeadersKeepsGlobal) {
  std::unordered_map<std::string, std::string> global;
  global["Authorization"] = "Bearer global";

  std::unordered_map<std::string, std::string> request;  // empty

  std::unordered_map<std::string, std::string> merged;
  for (const auto& [k, v] : global) merged[k] = v;
  for (const auto& [k, v] : request) merged[k] = v;

  EXPECT_EQ(1, merged.size());
  EXPECT_EQ("Bearer global", merged["Authorization"]);
}

TEST(HeaderMergeTest, EmptyGlobalHeadersKeepsRequest) {
  std::unordered_map<std::string, std::string> global;  // empty

  std::unordered_map<std::string, std::string> request;
  request["Authorization"] = "Bearer request";

  std::unordered_map<std::string, std::string> merged;
  for (const auto& [k, v] : global) merged[k] = v;
  for (const auto& [k, v] : request) merged[k] = v;

  EXPECT_EQ(1, merged.size());
  EXPECT_EQ("Bearer request", merged["Authorization"]);
}

}  // namespace xllm
