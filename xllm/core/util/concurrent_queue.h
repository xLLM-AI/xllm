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

#pragma once

#include <absl/synchronization/mutex.h>
#include <absl/types/optional.h>

#include <optional>
#include <queue>

#if __has_attribute(guarded_by)
#define GUARDED_BY(x) __attribute__((guarded_by(x)))
#else
#define GUARDED_BY(x)
#endif

namespace xllm {

// a simple thread-safe queue that supports multiple producers and multiple
// consumers concurrently the queue is implemented as a queue with condition
// variable and mutex lock
template <typename T>
class ConcurrentQueue {
 public:
  // constructor
  ConcurrentQueue() = default;

  explicit ConcurrentQueue(size_t capacity) : capacity_(capacity) {}

  // destructor
  ~ConcurrentQueue() = default;

  // push an element to the queue
  void push(T value) {
    absl::MutexLock lock(&mutex_);
    if (capacity_ > 0) {
      auto not_full = [this]() { return queue_.size() < capacity_; };
      mutex_.Await(absl::Condition(&not_full));
    }
    queue_.push(std::move(value));
  }

  template <typename... Args>
  void emplace(Args&&... args) {
    absl::MutexLock lock(&mutex_);
    if (capacity_ > 0) {
      auto not_full = [this]() { return queue_.size() < capacity_; };
      mutex_.Await(absl::Condition(&not_full));
    }
    queue_.emplace(std::forward<Args>(args)...);
  }

  // pop an element from the queue, block if the queue is empty
  T pop() {
    absl::MutexLock lock(&mutex_);

    auto not_empty = [this]() { return !queue_.empty(); };
    mutex_.Await(absl::Condition(&not_empty));

    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }
  // pop an element from the queue, block if the queue is empty, with timeout
  absl::optional<T> pop(absl::Duration timeout) {
    absl::MutexLock lock(&mutex_);

    auto not_empty = +[](std::queue<T>* q) { return !q->empty(); };

    if (mutex_.AwaitWithTimeout(absl::Condition(not_empty, &queue_), timeout)) {
      T value = std::move(queue_.front());
      queue_.pop();
      return value;
    }

    return absl::nullopt;
  }

  absl::optional<T> try_pop() {
    absl::MutexLock lock(&mutex_);
    if (queue_.empty()) {
      return std::nullopt;
    }
    T value = std::move(queue_.front());
    queue_.pop();
    return value;
  }

  // Non-blocking pop into an output reference. Avoids constructing an
  // `absl::optional<T>` for move-only `T` (e.g. `folly::Function<void()>`).
  // Returns true if an element was popped, false if the queue was empty.
  bool try_pop(T& out) {
    absl::MutexLock lock(&mutex_);
    if (queue_.empty()) {
      return false;
    }
    out = std::move(queue_.front());
    queue_.pop();
    return true;
  }

  // return the size of the queue
  size_t size() {
    absl::MutexLock lock(&mutex_);
    return queue_.size();
  }

  // return true if the queue is empty
  bool empty() {
    absl::MutexLock lock(&mutex_);
    return queue_.empty();
  }

 private:
  // the underlying queue
  std::queue<T> queue_ GUARDED_BY(mutex_);
  // mutex lock for the queue
  absl::Mutex mutex_;

  // maximum capacity of the queue, 0 means no limit.
  // when the queue is full, push will block
  size_t capacity_ = 0;
};
}  // namespace xllm
