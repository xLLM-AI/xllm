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

/**
 * Partial JSON Parser - C++ Implementation
 *
 * Based on:
 * - Python: https://github.com/promplate/partial-json-parser
 * - Go: https://github.com/blaze2305/partial-json-parser
 *
 * Thanks to the original authors for their excellent work.
 */

#include "core/util/partial_json_parser/parser.h"

#include <algorithm>
#include <cctype>
#include <cmath>
#include <nlohmann/json.hpp>
#include <sstream>

namespace partial_json_parser {

std::string format_json(const std::string& json_string) {
  try {
    auto json = nlohmann::json::parse(json_string);
    return json.dump(1);
  } catch (const std::exception& e) {
    return json_string;
  }
}

std::string parse_malformed_string(const std::string& malformed,
                                   TypeOptions options,
                                   bool format) {
  std::string str = malformed;

  // Trim whitespace
  str.erase(str.begin(),
            std::find_if(str.begin(), str.end(), [](unsigned char ch) {
              return !std::isspace(ch);
            }));
  str.erase(std::find_if(str.rbegin(),
                         str.rend(),
                         [](unsigned char ch) { return !std::isspace(ch); })
                .base(),
            str.end());

  if (str.empty()) {
    throw MalformedJSONException("string is empty; cannot parse");
  }

  std::string json_string = parse_json(malformed, options);

  if (format) {
    return format_json(json_string);
  } else {
    return json_string;
  }
}

int32_t skip_blank(const std::string& text, int32_t index) {
  int32_t i = index;
  while (
      i < static_cast<int>(text.length()) &&
      (std::isspace(static_cast<unsigned char>(text[i])) || text[i] == '\n')) {
    i += 1;
  }
  return i;
}

std::string parse_json(const std::string& json_string, TypeOptions allowed) {
  int32_t i = skip_blank(json_string, 0);
  std::string value = json_string.substr(i);

  JsonCompletion completion = complete_any(value, allowed, true);

  if (completion.index == 0 && completion.string.empty()) {
    throw MalformedJSONException("no valid JSON content found");
  }

  return value.substr(0, completion.index) + completion.string;
}

JsonCompletion complete_any(const std::string& json_string,
                            TypeOptions allowed,
                            bool top_level) {
  if (json_string.empty()) {
    throw MalformedJSONException("empty string");
  }

  char first_char = json_string[0];

  switch (first_char) {
    case '"':
      return complete_string(json_string, allowed);
    case '[':
      return complete_array(json_string, allowed);
    case '{':
      return complete_object(json_string, allowed);
    case '-':  // handles negative numbers
      if (json_string.length() == 1) {
        throw MalformedJSONException("cannot parse singular '-'");
      } else if (json_string.length() > 1 &&
                 json_string[1] != 'I') {  // not negative infinity
        return complete_number(json_string, allowed, top_level);
      }
      break;
    default:
      if (std::isdigit(static_cast<unsigned char>(first_char))) {
        return complete_number(json_string, allowed, top_level);
      }
      break;
  }

  // Handle NULL
  if (json_string.substr(
          0, std::min(4, static_cast<int>(json_string.length()))) == "null") {
    return JsonCompletion(4, "");
  }

  if (json_string.length() < 4 &&
      std::string("null").substr(0, json_string.length()) == json_string) {
    if ((NULL_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "null");
    }
    throw MalformedJSONException("cannot parse null with given options");
  }

  // Handle boolean true
  if (json_string.substr(
          0, std::min(4, static_cast<int>(json_string.length()))) == "true") {
    return JsonCompletion(4, "");
  }

  if (json_string.length() < 4 &&
      std::string("true").substr(0, json_string.length()) == json_string) {
    if ((BOOL | allowed) == allowed) {
      return JsonCompletion(0, "true");
    }
    throw MalformedJSONException("cannot parse bool with given options");
  }

  // Handle boolean false
  if (json_string.substr(
          0, std::min(5, static_cast<int>(json_string.length()))) == "false") {
    return JsonCompletion(5, "");
  }

  if (json_string.length() < 5 &&
      std::string("false").substr(0, json_string.length()) == json_string) {
    if ((BOOL | allowed) == allowed) {
      return JsonCompletion(0, "false");
    }
    throw MalformedJSONException("cannot parse bool with given options");
  }

  // Handle infinity
  if (json_string.substr(0,
                         std::min(8, static_cast<int>(json_string.length()))) ==
      "Infinity") {
    return JsonCompletion(8, "");
  }

  if (json_string.length() < 8 &&
      std::string("Infinity").substr(0, json_string.length()) == json_string) {
    if ((INFINITY_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "Infinity");
    }
    throw MalformedJSONException("cannot parse Infinity with given options");
  }

  // Handle negative infinity
  if (json_string.substr(0,
                         std::min(9, static_cast<int>(json_string.length()))) ==
      "-Infinity") {
    return JsonCompletion(9, "");
  }

  if (json_string.length() < 9 &&
      std::string("-Infinity").substr(0, json_string.length()) == json_string) {
    if ((NEG_INFINITY | allowed) == allowed) {
      return JsonCompletion(0, "-Infinity");
    }
    throw MalformedJSONException("cannot parse -Infinity with given options");
  }

  // Handle NaN
  if (json_string.substr(
          0, std::min(3, static_cast<int>(json_string.length()))) == "NaN") {
    return JsonCompletion(3, "");
  }

  if (json_string.length() < 3 &&
      std::string("NaN").substr(0, json_string.length()) == json_string) {
    if ((NAN_TYPE | allowed) == allowed) {
      return JsonCompletion(0, "NaN");
    }
    throw MalformedJSONException("cannot parse NaN with given options");
  }

  throw MalformedJSONException(std::string("MalformedJSON(unexpected char ") +
                               first_char + ")");
}

JsonCompletion complete_string(const std::string& json_string,
                               TypeOptions allowed) {
  if (json_string.empty() || json_string[0] != '"') {
    throw MalformedJSONException("string must start with quote");
  }

  int32_t index = 1;
  bool char_escaped = false;
  int32_t string_length = static_cast<int32_t>(json_string.length());

  while (index < string_length && (json_string[index] != '"' || char_escaped)) {
    if (json_string[index] == '\\') {
      char_escaped = !char_escaped;
    } else {
      char_escaped = false;
    }
    index += 1;
  }

  if (index < string_length) {
    return JsonCompletion(index + 1, "");
  }

  if ((STR | allowed) != allowed) {
    throw MalformedJSONException("cannot complete malformed json");
  }

  // Handle unicode and hex strings
  // Handle \uXXXX
  size_t u_index = json_string.rfind("\\u");
  if (u_index != std::string::npos) {
    if (static_cast<int>(u_index) + 6 == string_length) {
      return JsonCompletion(static_cast<int>(u_index) + 6, "\"");
    }
    return JsonCompletion(static_cast<int>(u_index) + 2, "\"");
  }

  // Handle \UXXXXXXXX
  size_t U_index = json_string.rfind("\\U");
  if (U_index != std::string::npos) {
    if (static_cast<int>(U_index) + 10 == string_length) {
      return JsonCompletion(static_cast<int>(U_index) + 10, "\"");
    }
    return JsonCompletion(static_cast<int>(U_index) + 2, "\"");
  }

  // Handle \xXX
  size_t x_index = json_string.rfind("\\x");
  if (x_index != std::string::npos) {
    if (static_cast<int>(x_index) + 4 == string_length) {
      return JsonCompletion(static_cast<int>(x_index) + 4, "\"");
    }
    return JsonCompletion(static_cast<int>(x_index) + 2, "\"");
  }

  if (char_escaped) {
    return JsonCompletion(index - 1, "\"");
  }

  return JsonCompletion(index, "\"");
}

JsonCompletion complete_array(const std::string& json_string,
                              TypeOptions allowed) {
  int32_t i = 1;
  int32_t j = 1;
  int32_t length = static_cast<int32_t>(json_string.length());

  while (j < length) {
    j = skip_blank(json_string, j);
    if (j >= length) {
      break;
    }

    if (json_string[j] == ']') {
      return JsonCompletion(j + 1, "");
    }

    try {
      JsonCompletion result =
          complete_any(json_string.substr(j), allowed, false);

      // If the string in the result has some char in it, complete the array
      if (!result.string.empty()) {
        if ((ARR | allowed) == allowed) {
          return JsonCompletion(j + result.index, result.string + "]");
        }
        throw MalformedJSONException("cannot parse array with given options");
      }

      // First item in array is fine, check other items
      j += result.index;
      i = j;

      j = skip_blank(json_string, j);
      if (j >= length) {
        break;
      }

      if (json_string[j] == ',') {
        j += 1;
      } else if (json_string[j] == ']') {
        return JsonCompletion(j + 1, "");
      } else {
        throw MalformedJSONException(
            std::string("MalformedJSON(expected \",\" or \"]\" got ") +
            json_string[j] + ")");
      }
    } catch (const MalformedJSONException&) {
      // Can't complete the array, make it empty
      if ((ARR | allowed) == allowed) {
        return JsonCompletion(i, "]");
      }
      throw MalformedJSONException("cannot parse array with given options");
    }
  }

  // Reached end of string, close array at last known good point
  if ((ARR | allowed) == allowed) {
    return JsonCompletion(i, "]");
  }
  throw MalformedJSONException("cannot parse array with given options");
}

JsonCompletion complete_object(const std::string& json_string,
                               TypeOptions allowed) {
  int32_t i = 1;
  int32_t j = 1;
  int32_t length = static_cast<int32_t>(json_string.length());

  while (j < length) {
    j = skip_blank(json_string, j);
    if (j >= length) {
      break;
    }

    if (json_string[j] == '}') {
      return JsonCompletion(j + 1, "");
    }

    try {
      JsonCompletion key = complete_string(json_string.substr(j), allowed);
      if (!key.string.empty()) {
        // Can't parse the key or key is incomplete
        if ((OBJ | allowed) == allowed) {
          return JsonCompletion(i, "}");
        }
        throw MalformedJSONException("cannot parse object with given options");
      }

      // Move index by key length
      j += key.index;

      j = skip_blank(json_string, j);
      if (j >= length) {
        break;
      }

      if (json_string[j] != ':') {
        throw MalformedJSONException(
            std::string("MalformedJSON( expected \":\" got ") + json_string[j] +
            ")");
      }
      j += 1;

      j = skip_blank(json_string, j);
      if (j >= length) {
        break;
      }

      JsonCompletion result =
          complete_any(json_string.substr(j), allowed, false);

      // If the string in the result has some char in it, complete the object
      if (!result.string.empty()) {
        if ((OBJ | allowed) == allowed) {
          return JsonCompletion(j + result.index, result.string + "}");
        }
        throw MalformedJSONException("cannot parse object with given options");
      }

      // First key-value pair is fine, check other items
      j += result.index;
      i = j;

      j = skip_blank(json_string, j);
      if (j >= length) {
        break;
      }

      if (json_string[j] == ',') {
        j += 1;
      } else if (json_string[j] == '}') {
        return JsonCompletion(j + 1, "");
      } else {
        throw MalformedJSONException(
            std::string("MalformedJSON(expected \",\" or \"}\" got ") +
            json_string[j] + ")");
      }
    } catch (const MalformedJSONException& e) {
      // Check if this is a case where the key is not a valid string start
      // For cases like "{0", we should throw the exception
      if (j < length && json_string[j] != '"' && json_string[j] != '}') {
        // This is an invalid key (not starting with quote), re-throw exception
        throw e;
      }

      // For other cases (like incomplete but valid objects), return empty
      // object
      if ((OBJ | allowed) == allowed) {
        return JsonCompletion(i, "}");
      }
      throw MalformedJSONException("cannot parse object with given options");
    }
  }

  // Reached end of string, close object at last known good point
  if ((OBJ | allowed) == allowed) {
    return JsonCompletion(i, "}");
  }
  throw MalformedJSONException("cannot parse object with given options");
}

JsonCompletion complete_number(const std::string& json_string,
                               TypeOptions allowed,
                               bool top_level) {
  int32_t i = 1;
  int32_t length = static_cast<int32_t>(json_string.length());

  // Move forwards while we still have numbers, including exponents and decimals
  while (i < length &&
         (std::isdigit(static_cast<unsigned char>(json_string[i])) ||
          json_string[i] == '.' || json_string[i] == '+' ||
          json_string[i] == '-' || json_string[i] == 'e' ||
          json_string[i] == 'E')) {
    i += 1;
  }

  bool special_num = false;
  // no boundary check initially
  while (i >= 1 && (json_string[i - 1] == '.' || json_string[i - 1] == '-' ||
                    json_string[i - 1] == '+' || json_string[i - 1] == 'e' ||
                    json_string[i - 1] == 'E')) {
    i -= 1;
    special_num = true;
    // If we've gone to position 0, we need to check if we can continue
    if (i == 0) {
      throw MalformedJSONException("string index out of range");
    }
  }

  if (special_num || (i == length && !top_level)) {
    if ((NUM | allowed) == allowed) {
      return JsonCompletion(i, "");
    }
    throw MalformedJSONException("cannot parse number with given options");
  }

  return JsonCompletion(i, "");
}

}  // namespace partial_json_parser