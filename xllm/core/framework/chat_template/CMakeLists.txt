include(cc_library)

# Base class (header-only, no circular deps)
cc_library (
  NAME
    chat_template_base
  HDRS
    chat_template.h
  DEPS
    :tokenizer
    nlohmann_json::nlohmann_json
)

cc_library (
  NAME
    jinja_chat_template
  HDRS
    jinja_chat_template.h
  SRCS
    jinja_chat_template.cpp
  DEPS
    :chat_template_base
    :minja
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
)

cc_library (
  NAME
    deepseek_v32_cpp_template
  HDRS
    deepseek_v32_cpp_template.h
  SRCS
    deepseek_v32_cpp_template.cpp
  DEPS
    :chat_template_base
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
    absl::strings
)

cc_library (
  NAME
    deepseek_v4_cpp_template
  HDRS
    deepseek_v4_cpp_template.h
  SRCS
    deepseek_v4_cpp_template.cpp
  DEPS
    :chat_template_base
    :tokenizer
    nlohmann_json::nlohmann_json
    glog::glog
    absl::strings
)

# Factory (depends on all implementations)
cc_library (
  NAME
    chat_template
  SRCS
    chat_template.cpp
  DEPS
    :chat_template_base
    :jinja_chat_template
    :deepseek_v32_cpp_template
    :deepseek_v4_cpp_template
    nlohmann_json::nlohmann_json
    glog::glog
    gflags
)
