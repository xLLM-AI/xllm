"""Standalone DSV4 sparse metadata probe; run one case per process."""

import os
import sys

import torch
import torch_npu


torch_npu.npu.set_device(0)
device = torch.device("npu:0")
sys.path.insert(0, "/mnt/sfs_turbo/tongpan/xllm/build/lib.linux-aarch64-cpython-311")
from xllm.python import kernels  # noqa: E402


kv_lens = [int(value) for value in os.environ["KV_LENS"].split(",")]
q_lens = [
    int(value)
    for value in os.environ.get(
        "Q_LENS", ",".join("1" for _ in kv_lens)
    ).split(",")
]
if len(q_lens) != len(kv_lens):
    raise ValueError("Q_LENS and KV_LENS must have the same number of entries")
batch = len(kv_lens)
cu_q = torch.zeros(batch + 1, dtype=torch.int32, device=device)
cu_q[1:] = torch.tensor(q_lens, dtype=torch.int32, device=device).cumsum(0)
seq_kv = torch.tensor(kv_lens, dtype=torch.int32, device=device)
empty = torch.empty(0, dtype=torch.int32, device=device)
ratio = int(os.environ.get("RATIO", "1"))
has_cmp = ratio > 1
cmp_topk = 512 if ratio == 4 else 0
max_q = max(q_lens)
args = (
    8, 1, 1088, cu_q, cu_q if max_q > 1 else empty, empty, empty,
    seq_kv, batch, max_q, max(kv_lens), 0, cmp_topk, ratio, 4, 3,
    127, 0, "TND", "PA_ND", True, has_cmp,
)
call_path = os.environ.get("CALL_PATH", "direct")
if call_path == "direct":
    import xllm_runtime

    result = xllm_runtime.dsv4_sparse_attn_sharedkv_metadata(*args)
elif call_path == "torch_ops":
    result = kernels.sparse_attn_sharedkv_metadata(*args)
else:
    raise ValueError(f"unsupported CALL_PATH={call_path!r}")
torch_npu.npu.synchronize()
print(
    f"OK q_lens={q_lens} kv_lens={kv_lens} ratio={ratio} "
    f"call_path={call_path} result={tuple(result.shape)}"
)
