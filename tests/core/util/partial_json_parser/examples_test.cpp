#include <gtest/gtest.h>

#include <cmath>

#include "core/util/partial_json_parser/options.h"
#include "core/util/partial_json_parser/parser.h"

using namespace partial_json_parser;

// Test string parsing
TEST(PartialJsonParserTest, TestStr) {
  // Test incomplete string with STR option
  EXPECT_EQ(parse_malformed_string("\"", STR), "\"\"");

  // Test incomplete string without STR option should throw
  EXPECT_THROW(parse_malformed_string("\"", static_cast<TypeOptions>(~STR)),
               MalformedJSONException);

  // Test escaped backslash
  EXPECT_EQ(parse_malformed_string("\"\\\\", STR), "\"\\\\\"");

  // Test unicode escape sequences
  EXPECT_EQ(parse_malformed_string("\"\\\\u", STR), "\"\\\\u\"");
  EXPECT_EQ(parse_malformed_string("\"\\\\U\\\\u", STR), "\"\\\\U\\\\u\"");
}

// Test array parsing
TEST(PartialJsonParserTest, TestArr) {
  // Test incomplete array with ARR option only
  EXPECT_EQ(parse_malformed_string("[\"", ARR), "[]");

  // Test incomplete array with both ARR and STR options
  EXPECT_EQ(parse_malformed_string("[\"", static_cast<TypeOptions>(ARR | STR)),
            "[\"\"]");

  // Test various incomplete arrays without proper options should throw
  EXPECT_THROW(parse_malformed_string("[", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("[\"", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("[\"\",", STR), MalformedJSONException);
}

// Test object parsing
TEST(PartialJsonParserTest, TestObj) {
  // Test incomplete object with OBJ option only
  EXPECT_EQ(parse_malformed_string("{\"\": \"", OBJ), "{}");

  // Test incomplete object with both OBJ and STR options
  EXPECT_EQ(
      parse_malformed_string("{\"\": \"", static_cast<TypeOptions>(OBJ | STR)),
      "{\"\": \"\"}");

  // Test various incomplete objects without proper options should throw
  EXPECT_THROW(parse_malformed_string("{", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{\"", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{\"\"]", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{\"\":", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{\"\":\"", STR), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{\"\":\"\"]", STR),
               MalformedJSONException);
}

// Test singleton values
TEST(PartialJsonParserTest, TestSingletons) {
  // Test null
  EXPECT_EQ(parse_malformed_string("n", NULL_TYPE), "null");
  EXPECT_THROW(
      parse_malformed_string("n", static_cast<TypeOptions>(~NULL_TYPE)),
      MalformedJSONException);

  // Test boolean true
  EXPECT_EQ(parse_malformed_string("t", BOOL), "true");
  EXPECT_THROW(parse_malformed_string("t", static_cast<TypeOptions>(~BOOL)),
               MalformedJSONException);

  // Test boolean false
  EXPECT_EQ(parse_malformed_string("f", BOOL), "false");
  EXPECT_THROW(parse_malformed_string("f", static_cast<TypeOptions>(~BOOL)),
               MalformedJSONException);

  // Test Infinity
  EXPECT_EQ(parse_malformed_string("I", INF), "Infinity");
  EXPECT_THROW(
      parse_malformed_string("I", static_cast<TypeOptions>(~INFINITY_TYPE)),
      MalformedJSONException);

  // Test negative Infinity
  EXPECT_EQ(parse_malformed_string("-I", INF), "-Infinity");
  EXPECT_THROW(
      parse_malformed_string("-I", static_cast<TypeOptions>(~NEG_INFINITY)),
      MalformedJSONException);

  // Test NaN
  EXPECT_EQ(parse_malformed_string("N", NAN_TYPE), "NaN");
  EXPECT_THROW(parse_malformed_string("N", static_cast<TypeOptions>(~NAN_TYPE)),
               MalformedJSONException);
}

// Test number parsing
TEST(PartialJsonParserTest, TestNum) {
  // Test complete numbers (should work without NUM option)
  EXPECT_EQ(parse_malformed_string("0", static_cast<TypeOptions>(~NUM)), "0");
  EXPECT_EQ(parse_malformed_string("-1.25e+4", static_cast<TypeOptions>(~NUM)),
            "-1.25e+4");

  // Test incomplete numbers (need NUM option)
  EXPECT_EQ(parse_malformed_string("-1.25e+", NUM), "-1.25");
  EXPECT_EQ(parse_malformed_string("-1.25e", NUM), "-1.25");
}

// Test error cases
TEST(PartialJsonParserTest, TestError) {
  // Test unexpected characters
  EXPECT_THROW(parse_malformed_string("a", ALL), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("{0", ALL), MalformedJSONException);
  EXPECT_THROW(parse_malformed_string("--", ALL), MalformedJSONException);
}

// Test basic functionality
TEST(PartialJsonParserTest, TestBasicFunctionality) {
  // Test basic incomplete JSON structures
  EXPECT_EQ(parse_malformed_string("[", ALL), "[]");
  EXPECT_EQ(parse_malformed_string("[0.", ALL), "[0]");
  EXPECT_EQ(parse_malformed_string("{\"key\": ", ALL), "{}");
  EXPECT_EQ(parse_malformed_string("t", ALL), "true");

  // Test with restricted options
  EXPECT_EQ(parse_malformed_string("[1", static_cast<TypeOptions>(~NUM)), "[]");
  EXPECT_EQ(parse_malformed_string("1", static_cast<TypeOptions>(~NUM)), "1");

  // Test error case
  EXPECT_THROW(parse_malformed_string("-", ALL), MalformedJSONException);
}

// Test complex nested structures
TEST(PartialJsonParserTest, TestComplexStructures) {
  // Test nested array with incomplete string
  std::string result1 =
      parse_malformed_string("[\"a\", \"b", static_cast<TypeOptions>(~STR));
  EXPECT_EQ(result1, "[\"a\"]");

  // Test nested object with incomplete values
  std::string result2 =
      parse_malformed_string("{\"key1\": 123, \"key2\": \"val", ALL);
  EXPECT_EQ(result2, "{\"key1\": 123, \"key2\": \"val\"}");

  // Test deeply nested structure
  std::string result3 =
      parse_malformed_string("{\"arr\": [1, 2, {\"nested\":", ALL);
  EXPECT_EQ(result3, "{\"arr\": [1, 2, {}]}");
}

// Test edge cases
TEST(PartialJsonParserTest, TestEdgeCases) {
  // Test empty string
  EXPECT_THROW(parse_malformed_string("", ALL), MalformedJSONException);

  // Test whitespace only
  EXPECT_THROW(parse_malformed_string("   ", ALL), MalformedJSONException);

  // Test single characters
  EXPECT_EQ(parse_malformed_string("n", NULL_TYPE), "null");
  EXPECT_EQ(parse_malformed_string("t", BOOL), "true");
  EXPECT_EQ(parse_malformed_string("f", BOOL), "false");

  // Test numbers with trailing operators
  EXPECT_EQ(parse_malformed_string("123.", NUM), "123");
  EXPECT_EQ(parse_malformed_string("123e", NUM), "123");
  EXPECT_EQ(parse_malformed_string("123e+", NUM), "123");
  EXPECT_EQ(parse_malformed_string("123e-", NUM), "123");
}

// Test format parameter
TEST(PartialJsonParserTest, TestFormatParameter) {
  std::string result1 = parse_malformed_string("{\"foo\":\"bar", ALL, false);
  EXPECT_EQ(result1, "{\"foo\":\"bar\"}");

  std::string result2 = parse_malformed_string("{\"foo\":\"bar", ALL, true);
  EXPECT_EQ(result2, "{\n \"foo\": \"bar\"\n}");
}

// Test Go example
TEST(PartialJsonParserTest, TestGoExample) {
  // Test the example: `{"foo":"bar`
  std::string result = parse_malformed_string("{\"foo\":\"bar", ALL);
  EXPECT_EQ(result, "{\"foo\":\"bar\"}");

  // Test the array example: `["a",{"a":123`
  std::string result2 = parse_malformed_string(
      "[\"a\",{\"a\":123", static_cast<TypeOptions>(NUM | ARR | OBJ));
  EXPECT_EQ(result2, "[\"a\",{\"a\":123}]");
}