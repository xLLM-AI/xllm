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
#include <algorithm>
#include <deque>
#include <functional>
#include <memory>
#include <queue>
#include <set>

#include "common/macros.h"
#include "framework/request/priority_comparator.h"
#include "framework/request/request.h"

namespace xllm {

// for Encapsulate and support Iterator pattern
class BaseIterator {
 public:
  virtual ~BaseIterator() = default;
  virtual std::shared_ptr<Request> operator*() const = 0;
  virtual void operator++() = 0;
  virtual bool operator!=(const BaseIterator& other) const = 0;
  virtual std::unique_ptr<BaseIterator> clone() const = 0;
};

template <typename Iterator>
class ConcreteIterator : public BaseIterator {
  Iterator iter_;

 public:
  explicit ConcreteIterator(Iterator iter) : iter_(iter) {}

  std::shared_ptr<Request> operator*() const override { return *iter_; }

  void operator++() override { ++iter_; }

  bool operator!=(const BaseIterator& other) const override {
    const auto* derived = dynamic_cast<const ConcreteIterator*>(&other);
    return derived && iter_ != derived->iter_;
  }

  std::unique_ptr<BaseIterator> clone() const override {
    return std::make_unique<ConcreteIterator>(iter_);
  }
};

class RequestPriorityQueue {
 public:
  using Comparator = std::function<bool(const std::shared_ptr<Request>&,
                                        const std::shared_ptr<Request>&)>;

  class Iterator {
    std::unique_ptr<BaseIterator> itr_;

   public:
    explicit Iterator(std::unique_ptr<BaseIterator> itr)
        : itr_(std::move(itr)) {}

    std::shared_ptr<Request> operator*() const { return **itr_; }

    Iterator& operator++() {
      ++*itr_;
      return *this;
    }

    bool operator!=(const Iterator& other) const {
      return itr_->operator!=(*other.itr_);
    }
  };
  virtual void push(std::shared_ptr<Request> req) = 0;
  virtual void push(std::shared_ptr<Request> req, bool if_back) = 0;
  virtual void pop_top() = 0;
  virtual void pop_back() = 0;
  virtual std::shared_ptr<Request> top() const = 0;
  virtual std::shared_ptr<Request> back() const = 0;
  virtual bool empty() const = 0;
  virtual size_t size() const = 0;
  virtual std::unique_ptr<RequestPriorityQueue> clone() const = 0;
  virtual ~RequestPriorityQueue() = default;
  virtual bool supports_sort() const { return false; }
  virtual void sort(const Comparator&) {}

  virtual Iterator begin() const = 0;
  virtual Iterator end() const = 0;
  virtual Iterator rbegin() const = 0;
  virtual Iterator rend() const = 0;
};

class HeapQueue final : public RequestPriorityQueue {
 private:
  using QueueType = std::priority_queue<std::shared_ptr<Request>,
                                        std::vector<std::shared_ptr<Request>>,
                                        Comparator>;
  QueueType queue_;

 public:
  explicit HeapQueue(Comparator lower_priority_comparator)
      : queue_(std::move(lower_priority_comparator)) {}

  void push(std::shared_ptr<Request> req) override { queue_.push(req); }
  void push(std::shared_ptr<Request> req, bool if_back) override {
    // Heap queue has no front/back insertion semantics.
    UNUSED_PARAMETER(if_back);
    queue_.push(req);
  }
  void pop_top() override { queue_.pop(); }
  void pop_back() override { NOT_IMPLEMENTED(); }
  std::shared_ptr<Request> top() const override { return queue_.top(); }
  std::shared_ptr<Request> back() const override {
    NOT_IMPLEMENTED();
    return nullptr;
  }
  bool empty() const override { return queue_.empty(); }
  size_t size() const override { return queue_.size(); }
  std::unique_ptr<RequestPriorityQueue> clone() const override {
    return std::make_unique<HeapQueue>(*this);
  }
  bool supports_sort() const override { return false; }
  void sort(const Comparator&) override { NOT_IMPLEMENTED(); }

  Iterator begin() const override {
    NOT_IMPLEMENTED();
    return Iterator(nullptr);
  }

  Iterator end() const override {
    NOT_IMPLEMENTED();
    return Iterator(nullptr);
  }

  Iterator rbegin() const override {
    NOT_IMPLEMENTED();
    return Iterator(nullptr);
  }

  Iterator rend() const override {
    NOT_IMPLEMENTED();
    return Iterator(nullptr);
  }
};

class SetQueue final : public RequestPriorityQueue {
 private:
  using QueueType = std::set<std::shared_ptr<Request>, Comparator>;
  Comparator lower_priority_comparator_;
  QueueType queue_;

 public:
  explicit SetQueue(Comparator lower_priority_comparator)
      : lower_priority_comparator_(std::move(lower_priority_comparator)),
        queue_(lower_priority_comparator_) {}

  void push(std::shared_ptr<Request> req) override { queue_.insert(req); }
  void push(std::shared_ptr<Request> req, bool if_back) override {
    UNUSED_PARAMETER(req);
    UNUSED_PARAMETER(if_back);
    NOT_IMPLEMENTED();
  }
  void pop_top() override { queue_.erase(queue_.begin()); }
  void pop_back() override { queue_.erase(std::prev(queue_.end())); }
  std::shared_ptr<Request> top() const override { return *queue_.begin(); }
  std::shared_ptr<Request> back() const override { return *queue_.rbegin(); }
  bool empty() const override { return queue_.empty(); }
  size_t size() const override { return queue_.size(); }
  std::unique_ptr<RequestPriorityQueue> clone() const override {
    auto copied = std::make_unique<SetQueue>(lower_priority_comparator_);
    for (const auto& request : queue_) {
      copied->push(request);
    }
    return copied;
  }

  Iterator begin() const override {
    return Iterator(std::make_unique<ConcreteIterator<QueueType::iterator>>(
        queue_.begin()));
  }

  Iterator end() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::iterator>>(queue_.end()));
  }

  Iterator rbegin() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::reverse_iterator>>(
            queue_.rbegin()));
  }

  Iterator rend() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<QueueType::reverse_iterator>>(
            queue_.rend()));
  }
};

class DequeQueue final : public RequestPriorityQueue {
  // use deque to implement FCFS queue for insert and evict effeciency
 private:
  std::deque<std::shared_ptr<Request>> queue_;

 public:
  void push(std::shared_ptr<Request> req) override { queue_.push_front(req); }

  void push(std::shared_ptr<Request> req, bool if_back) override {
    if (if_back) {
      queue_.push_back(req);
    } else {
      queue_.push_front(req);
    }
  }

  void pop_top() override { queue_.pop_front(); }
  void pop_back() override { queue_.pop_back(); }
  std::shared_ptr<Request> top() const override { return queue_.front(); }
  std::shared_ptr<Request> back() const override { return queue_.back(); }
  bool empty() const override { return queue_.empty(); }
  size_t size() const override { return queue_.size(); }
  std::unique_ptr<RequestPriorityQueue> clone() const override {
    return std::make_unique<DequeQueue>(*this);
  }
  bool supports_sort() const override { return true; }
  void sort(const Comparator& comparator) override {
    std::stable_sort(queue_.begin(), queue_.end(), comparator);
  }

  Iterator begin() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<decltype(queue_)::const_iterator>>(
            queue_.begin()));
  }

  Iterator end() const override {
    return Iterator(
        std::make_unique<ConcreteIterator<decltype(queue_)::const_iterator>>(
            queue_.end()));
  }

  Iterator rbegin() const override {
    return Iterator(std::make_unique<
                    ConcreteIterator<decltype(queue_)::const_reverse_iterator>>(
        queue_.rbegin()));
  }

  Iterator rend() const override {
    return Iterator(std::make_unique<
                    ConcreteIterator<decltype(queue_)::const_reverse_iterator>>(
        queue_.rend()));
  }
};
}  // namespace xllm
