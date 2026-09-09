/* Copyright 2025-2026 The xLLM Authors.
Copyright 2024 The ScaleLLM Authors. All Rights Reserved.

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

#pragma once

#include <glog/logging.h>

#include <vector>

namespace xllm {

template <typename T>
class Slice final {
 public:
  Slice() = default;

  Slice(const T* data, size_t size) : data_(data), size_(size) {}

  // it is on purpose to allow implicit conversion from vector to slice
  Slice(const std::vector<T>& data) : data_(data.data()), size_(data.size()) {}

  Slice(const std::vector<T>& data, size_t size)
      : data_(data.data()), size_(size) {
    CHECK_LE(size, data.size());
  }

  // iterator for the slice
  const T* begin() const { return data_; }
  const T* end() const { return data_ + size_; }
  const T* cbegin() const { return data_; }
  const T* cend() const { return data_ + size_; }

  // get the size of the slice
  size_t size() const { return size_; }

  // check if the slice is empty
  bool empty() const { return size_ == 0; }

  // get the data pointer
  const T* data() const { return data_; }

  // index operator
  const T& operator[](size_t i) const { return data_[i]; }

  const T& front() const { return data_[0]; }

  const T& back() const { return data_[size_ - 1]; }

  // get a sub slice
  Slice<T> slice(size_t start) const {
    CHECK_LE(start, size_);
    return {data_ + start, size_ - start};
  }

  Slice<T> slice(size_t start, size_t end) const {
    CHECK(start <= end && end <= size_);
    return {data_ + start, end - start};
  }

  // it is safe to allow implicit conversion to vector
  operator std::vector<T>() const { return {data_, data_ + size_}; }

 private:
  const T* data_ = nullptr;
  size_t size_ = 0;
};

// help comparison operators between slices and std::vector
template <typename T>
inline bool operator==(const Slice<T>& lhs, const std::vector<T>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.data() == rhs.data() ||
          std::equal(lhs.begin(), lhs.end(), rhs.begin()));
}

template <typename T>
inline bool operator==(const std::vector<T>& lhs, const Slice<T>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.data() == rhs.data() ||
          std::equal(lhs.begin(), lhs.end(), rhs.begin()));
}

template <typename T>
inline bool operator==(const Slice<T>& lhs, const Slice<T>& rhs) {
  return lhs.size() == rhs.size() &&
         (lhs.data() == rhs.data() ||
          std::equal(lhs.begin(), lhs.end(), rhs.begin()));
}

template <typename T>
inline std::ostream& operator<<(std::ostream& os, const Slice<T>& slice) {
  os << "Slice<" << typeid(T).name() << ">("
     << "data=" << static_cast<const void*>(slice.data()) << ", "
     << "size=" << slice.size() << "): [";

  if (slice.empty()) {
    os << "]";
    return os;
  }

  constexpr size_t kMaxPrintElements = 10;
  size_t print_count = std::min(slice.size(), kMaxPrintElements);

  for (size_t i = 0; i < print_count; ++i) {
    if (i > 0) {
      os << ", ";
    }

    if constexpr (std::is_same_v<T, char>) {
      os << "\"" << std::string(slice.data() + i, 1) << "\"";
    } else if constexpr (std::is_pointer_v<T>) {
      os << static_cast<const void*>(slice[i]);
    } else {
      os << slice[i];
    }
  }

  if (slice.size() > kMaxPrintElements) {
    os << ", ... (" << slice.size() - kMaxPrintElements << " more elements)";
  }

  os << "]";
  return os;
}
}  // namespace xllm
