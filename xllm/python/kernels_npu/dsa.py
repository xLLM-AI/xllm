# Copyright 2026 The xLLM Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://github.com/xLLM-AI/xllm/blob/main/LICENSE
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""DeepSeek-V4 compressed KV and shared sparse attention kernels."""

from xllm_kernel.ops.attention.sparse_attention import (
    compressor as compressor,
)
from xllm_kernel.ops.attention.sparse_attention import (
    sparse_attn_sharedkv as sparse_attn_sharedkv,
)
from xllm_kernel.ops.attention.sparse_attention import (
    sparse_attn_sharedkv_metadata as sparse_attn_sharedkv_metadata,
)
