# Copyright 2025-2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/jd-opensource/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Attention metadata passed explicitly from C++ to the Python forward.

The C++ ``PyCausalLM::forward`` builds this from the internal
``layer::AttentionMetadata`` struct and passes it as a dict argument to the
Python model.  ``AttentionMetadata`` wraps that dict with typed attribute
access so the Python attention layer has a clean interface.
"""

from __future__ import annotations

from typing import Optional

import torch


class AttentionMetadata:
    """Typed wrapper over the metadata dict built by C++ PyCausalLM::forward."""

    __slots__ = (
        "slot_mapping",
        "paged_kv_indptr",
        "paged_kv_indices",
        "paged_kv_last_page_len",
        "qo_indptr",
        "q_cu_seq_lens",
        "kv_cu_seq_lens",
        "is_prefill",
        "is_chunked_prefill",
        "enable_cuda_graph",
        "use_tensor_core",
    )

    def __init__(self, meta_dict: dict) -> None:
        self.slot_mapping: torch.Tensor = meta_dict["slot_mapping"]
        self.paged_kv_indptr: torch.Tensor = meta_dict["paged_kv_indptr"]
        self.paged_kv_indices: torch.Tensor = meta_dict["paged_kv_indices"]
        self.paged_kv_last_page_len: torch.Tensor = meta_dict[
            "paged_kv_last_page_len"
        ]
        self.qo_indptr: Optional[torch.Tensor] = meta_dict.get("qo_indptr")
        self.q_cu_seq_lens: Optional[torch.Tensor] = meta_dict.get(
            "q_cu_seq_lens"
        )
        self.kv_cu_seq_lens: Optional[torch.Tensor] = meta_dict.get(
            "kv_cu_seq_lens"
        )
        self.is_prefill: bool = meta_dict["is_prefill"]
        self.is_chunked_prefill: bool = meta_dict["is_chunked_prefill"]
        self.enable_cuda_graph: bool = meta_dict["enable_cuda_graph"]
        self.use_tensor_core: bool = meta_dict.get("use_tensor_core", False)
