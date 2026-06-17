include(cc_library)

# Define the library
cc_library(
  NAME 
    reasoning
  HDRS
    reasoning_detector.h
    reasoning_parser.h
    detector_registry.h
  SRCS
    reasoning_detector.cpp
    reasoning_parser.cpp
    detector_registry.cpp
  DEPS
    absl::strings
    glog::glog
)

