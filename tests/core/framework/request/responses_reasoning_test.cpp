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

#include <gtest/gtest.h>

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "parser/detector_registry.h"
#include "parser/reasoning_parser.h"

namespace xllm {
namespace {

std::pair<std::string, std::string> parse_chunks(
    const std::vector<std::string>& chunks,
    bool stream_reasoning,
    bool force_reasoning) {
  ReasoningParser parser("qwen3",
                         stream_reasoning,
                         force_reasoning,
                         std::nullopt,
                         /*lossless=*/true);
  std::string normal;
  std::string reasoning;
  auto append = [&](const ReasoningResult& result) {
    normal += result.normal_text.value_or("");
    reasoning += result.reasoning_text.value_or("");
  };
  for (const std::string& chunk : chunks) {
    append(parser.parse_stream_chunk(chunk));
  }
  append(parser.finish_stream());
  ReasoningResult second_finish = parser.finish_stream();
  EXPECT_FALSE(second_finish.normal_text.has_value());
  EXPECT_FALSE(second_finish.reasoning_text.has_value());
  return {normal, reasoning};
}

TEST(ResponsesReasoningTest, EverySplitPreservesReasoningAndAnswerBytes) {
  const std::string text = "<think> why\nnow </think> answer\n";
  for (bool stream : {false, true}) {
    for (size_t split = 0; split <= text.size(); ++split) {
      SCOPED_TRACE(split);
      SCOPED_TRACE(stream);
      const auto [normal, reasoning] = parse_chunks(
          {text.substr(0, split), text.substr(split)}, stream, false);
      EXPECT_EQ(normal, " answer\n");
      EXPECT_EQ(reasoning, " why\nnow ");
    }
    std::vector<std::string> bytes;
    bytes.reserve(text.size());
    for (char byte : text) {
      bytes.emplace_back(1, byte);
    }
    EXPECT_EQ(
        parse_chunks(bytes, stream, false),
        std::make_pair(std::string(" answer\n"), std::string(" why\nnow ")));
  }
}

TEST(ResponsesReasoningTest, PartialEndAfterContentDoesNotLeakIntoReasoning) {
  ReasoningParser parser("qwen3",
                         /*stream_reasoning=*/true,
                         /*force_reasoning=*/true,
                         std::nullopt,
                         /*lossless=*/true);
  ReasoningResult first = parser.parse_stream_chunk("why</th");
  EXPECT_EQ(first.reasoning_text, "why");
  EXPECT_FALSE(first.normal_text.has_value());
  ReasoningResult second = parser.parse_stream_chunk("ink>answer");
  EXPECT_EQ(second.normal_text, "answer");
  EXPECT_FALSE(second.reasoning_text.has_value());
  EXPECT_FALSE(parser.finish_stream().reasoning_text.has_value());
}

TEST(ResponsesReasoningTest, EofFlushesUnfinishedDelimitersLiterallyOnce) {
  EXPECT_EQ(parse_chunks({"normal<thi"}, true, false),
            std::make_pair(std::string("normal<thi"), std::string()));
  for (bool stream : {false, true}) {
    EXPECT_EQ(parse_chunks({"why</th"}, stream, true),
              std::make_pair(std::string(), std::string("why</th")));
    EXPECT_EQ(parse_chunks({"<think>", "why<thi"}, stream, false),
              std::make_pair(std::string(), std::string("why<thi")));
  }
}

TEST(ResponsesReasoningTest,
     NonStreamingAndStreamingUseTheSameMarkerSemantics) {
  for (bool force : {false, true}) {
    for (const std::string& text :
         {std::string("<think> why </think> answer\n"),
          std::string("why</think>answer<think>literal"),
          std::string("normal<thi"),
          std::string("why</th")}) {
      SCOPED_TRACE(text);
      SCOPED_TRACE(force);
      ReasoningParser parser("qwen3",
                             /*stream_reasoning=*/true,
                             force,
                             std::nullopt,
                             /*lossless=*/true);
      ReasoningResult result = parser.parse_non_stream(text);
      EXPECT_EQ(parse_chunks({text}, true, force),
                std::make_pair(result.normal_text.value_or(""),
                               result.reasoning_text.value_or("")));
    }
  }
}

TEST(ResponsesReasoningTest, DefaultParserKeepsLegacyNonStreamingWhitespace) {
  ReasoningParser parser("qwen3");
  const ReasoningResult result =
      parser.parse_non_stream("<think> why\nnow </think> answer\n");
  EXPECT_EQ(result.reasoning_text, "why\nnow ");
  EXPECT_EQ(result.normal_text, "answer");
  EXPECT_EQ(parser.parse_non_stream("<think> why\n").reasoning_text, "why");
  EXPECT_EQ(parser.parse_non_stream("  answer\n").normal_text, "  answer\n");
}

TEST(ResponsesReasoningTest, DefaultParserKeepsLegacyStreamingWhitespace) {
  ReasoningParser parser("qwen3");
  const ReasoningResult partial = parser.parse_stream_chunk("<thi");
  EXPECT_FALSE(partial.normal_text.has_value());
  EXPECT_FALSE(partial.reasoning_text.has_value());
  EXPECT_EQ(parser.parse_stream_chunk("nk> why\n").reasoning_text, " why\n");
  const ReasoningResult end =
      parser.parse_stream_chunk("now </think> answer\n");
  EXPECT_EQ(end.reasoning_text, "now ");
  EXPECT_EQ(end.normal_text, "answer");
  EXPECT_EQ(parser.parse_stream_chunk(" follow\n").normal_text, " follow\n");
}

TEST(ResponsesReasoningTest, AutoResolutionCanValidateWithoutStartupFatal) {
  DetectorRegistry& registry = DetectorRegistry::get_instance();
  EXPECT_EQ(registry.resolve_parser_name("glm4_moe"), "glm45");
  EXPECT_EQ(registry.get_parser_name_by_model_type("glm4_moe"), "glm45");
  EXPECT_FALSE(registry.resolve_parser_name("unsupported").has_value());
  EXPECT_TRUE(registry.has_detector("qwen3"));
  EXPECT_FALSE(registry.has_detector("unsupported"));
}

TEST(ResponsesReasoningTest, PromptMetadataOverridesThinkingOnlyDefault) {
  ReasoningParser thinking("qwen3-thinking",
                           /*stream_reasoning=*/true,
                           /*force_reasoning=*/false,
                           std::nullopt,
                           /*lossless=*/true);
  EXPECT_TRUE(thinking.initially_in_reasoning());
  EXPECT_EQ(thinking.parse_stream_chunk("why").reasoning_text, "why");

  ReasoningParser answer("qwen3-thinking",
                         /*stream_reasoning=*/true,
                         /*force_reasoning=*/false,
                         /*initial_reasoning=*/false,
                         /*lossless=*/true);
  EXPECT_FALSE(answer.initially_in_reasoning());
  EXPECT_EQ(answer.parse_stream_chunk("answer").normal_text, "answer");
  ReasoningParser forced("glm5",
                         /*stream_reasoning=*/true,
                         /*force_reasoning=*/false,
                         /*initial_reasoning=*/true,
                         /*lossless=*/true);
  EXPECT_TRUE(forced.initially_in_reasoning());
  EXPECT_EQ(forced.parse_stream_chunk("why").reasoning_text, "why");
}

TEST(ResponsesReasoningTest, NonStreamingDoesNotMutateLaterParserInitialState) {
  ReasoningParser parser("qwen3",
                         /*stream_reasoning=*/true,
                         /*force_reasoning=*/true,
                         std::nullopt,
                         /*lossless=*/true);
  EXPECT_EQ(parser.parse_non_stream("why</think>answer").normal_text, "answer");
  EXPECT_EQ(parser.parse_non_stream("why</th").reasoning_text, "why</th");
}

}  // namespace
}  // namespace xllm
