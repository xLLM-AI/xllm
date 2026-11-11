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

// ref to:
// https://
// github.com/snowflakedb/ArcticInference/blob/main/csrc/suffix_decoding/suffix_tree.h
#pragma once

#include <deque>
#include <memory>
#include <span>
#include <utility>
#include <vector>

#include "core/util/int32_map.h"

namespace xllm {

struct Group;

struct Node {
  // Number of suffixes from the root that end at or pass through this node.
  int64_t count = 0;

  // Token referenced by this node. Node can refer to a sequence of tokens,
  // this is just the ID of the first token.
  int32_t token = 0;

  // Number of tokens in this node.
  int32_t length = 0;

  // Reference sequence ID and starting index for the tokens in this node.
  // Implements the path compression optimization to achieve O(N) memory.
  int32_t ref_seq = 0;
  int32_t ref_idx = -1;

  // This map tracks all the suffixes that end at this node. Maps seq_id to
  // the end index of that suffix (may be truncated due to tree depth). Used
  // to find a new ref_seq and ref_idx if the reference sequence is deleted.
  Int32Map<int32_t> endpoints;

  // Pointer to parent node.
  Node* parent = nullptr;

  // Children nodes, the key should always be the first token of the child.
  Int32Map<std::unique_ptr<Node>> children;

  // All the children of each node are kept in order of decreasing count in
  // a doubly linked list for efficient speculation. head_child points to the
  // first child (highest count) and tail_child points to the last child.
  Node* head_child = nullptr;
  Node* tail_child = nullptr;

  // Pointers to the next and previous siblings in the doubly linked list.
  Node* next_sibling = nullptr;
  Node* prev_sibling = nullptr;

  // To enable efficient reordering of the siblings list when counts change,
  // nodes with the same count are grouped together. Each node holds a shared
  // pointer to its group, and the groups also form a doubly linked list.
  std::shared_ptr<Group> group = nullptr;

  Node() = default;

  Node(int64_t count,
       int32_t token,
       int32_t length,
       int32_t ref_seq,
       int32_t ref_idx)
      : count(count),
        token(token),
        length(length),
        ref_seq(ref_seq),
        ref_idx(ref_idx) {}
};

struct Group {
  // Pointer to the head node of this group. All nodes before the head node
  // have a strictly higher count, and all nodes after the head node have a
  // lower or equal count.
  Node* head = nullptr;

  // Pointers to the next and previous groups in the doubly linked list.
  Group* next = nullptr;
  Group* prev = nullptr;

  Group(Node* head) : head(head) {}
};

struct Draft {
  // The token ids of the speculation draft.
  std::vector<int32_t> token_ids;

  // Floating point score of the draft (sum of all probs).
  float score = 0.0;
};

class SuffixTree {
 public:
  SuffixTree(int32_t max_depth);

  // Append a new element to the sequence with id seq_id.
  void append(int32_t seq_id, int32_t token);

  // Append multiple new elements to the sequence with id seq_id.
  void extend(int32_t seq_id, std::span<const int32_t> tokens);

  std::span<const int32_t> tail(int32_t seq_id, size_t n) const;

  // Remove the sequence with id seq_id.
  void remove(int32_t seq_id);

  Draft speculate(std::span<const int32_t> context, int32_t max_tokens);

 private:
  // Maximum depth of the suffix tree.
  int32_t max_depth_;

  // The root node of the suffix tree.
  std::unique_ptr<Node> root_;

  // Mapping from seq id to its sequence of tokens (vectors of int32_t).
  Int32Map<std::vector<int32_t>> seqs_;

  // For each sequence, a sliding window of active nodes. Maintains at most
  // max_depth_ active nodes for each sequence. Queue is shifted when a new
  // token is added to the sequence. Each active node is in the queue for at
  // most max_depth_ iterations before being removed.
  Int32Map<std::deque<Node*>> active_nodes_;

  std::pair<Node*, int32_t> match_context(std::span<const int32_t> context);

  Draft speculate_path(Node* node, int32_t idx, int32_t max_spec_tokens);
};

}  // namespace xllm
