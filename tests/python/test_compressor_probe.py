"""Minimal compressor probe: reproduce the C4 compressor call with the exact
params from [CMPDUMP L2] and try 2D vs 3D x to isolate blockDim=0.

Params (from real run dump):
  x=[84,4096] bf16 (2D) or [1,84,4096] (3D)
  wkv=[256,4096] wgate=[256,4096] norm_weight=[128] ape=[4,256] (f32, cast bf16 for wkv/wgate/norm)
  rope_sin=[21,64] rope_cos=[21,64] bf16
  kv_state=[50,128,1024] score_state=[50,128,1024] f32
  kv_bt=[[1]] score_bt=[[1]] int32
  cu_seqlens=[0,84] int32
  start_pos=[0] int32
  cmp_ratio=4 coff=2 rope_head_dim=64 rotary_mode=2
"""
import torch
import torch_npu

torch_npu.npu.set_device(0)
dev = torch.device("npu:0")
import sys
sys.path.insert(0, "/mnt/sfs_turbo/tongpan/xllm/build/lib.linux-aarch64-cpython-311")
from xllm.python import kernels

M = 84
HIDDEN = 4096
bf16 = torch.bfloat16
f32 = torch.float32

x2d = torch.randn(M, HIDDEN, dtype=bf16, device=dev)
x3d = torch.randn(1, M, HIDDEN, dtype=bf16, device=dev)
wkv = torch.randn(256, HIDDEN, dtype=f32, device=dev).to(bf16)
wgate = torch.randn(256, HIDDEN, dtype=f32, device=dev).to(bf16)
norm_weight = torch.randn(128, dtype=f32, device=dev).to(bf16)
ape = torch.randn(4, 256, dtype=f32, device=dev)
rope_sin = torch.randn(21, 64, dtype=bf16, device=dev)
rope_cos = torch.randn(21, 64, dtype=bf16, device=dev)
kv_state = torch.randn(50, 128, 1024, dtype=f32, device=dev)
score_state = torch.randn(50, 128, 1024, dtype=f32, device=dev)
kv_bt = torch.tensor([[1]], dtype=torch.int32, device=dev)
score_bt = torch.tensor([[1]], dtype=torch.int32, device=dev)
cu_seqlens = torch.tensor([0, 84], dtype=torch.int32, device=dev)
start_pos = torch.tensor([0], dtype=torch.int32, device=dev)


def try_compress(label, x):
    try:
        out, _, _, _, _ = kernels.compressor(
            x=x, wkv=wkv, wgate=wgate, kv_state=kv_state, score_state=score_state,
            ape=ape, norm_weight=norm_weight, rope_sin=rope_sin, rope_cos=rope_cos,
            kv_block_table=kv_bt, score_block_table=score_bt, cu_seqlens=cu_seqlens,
            seqused=None, start_pos=start_pos, rope_head_dim=64, cmp_ratio=4, coff=2,
            norm_eps=1e-6, rotary_mode=2, enable_grad=False,
        )
        print(f"[OK] {label}: out={out.shape} {out.dtype}")
    except Exception as e:
        msg = str(e)
        acl = next((l.strip() for l in msg.splitlines() if 'blockDim' in l or 'EZ' in l or 'Invalid' in l or 'Check failed' in l), msg[:160])
        print(f"[FAIL] {label}: {acl}")


print("=== compressor probe (real params) ===")
try_compress("x=2D [84,4096]", x2d)
try_compress("x=3D [1,84,4096]", x3d)
# Variants: larger kv_bt (more columns)
kv_bt2 = torch.tensor([[1, 1, 1, 1]], dtype=torch.int32, device=dev)
try_compress("x=2D kv_bt=[1,4]", x2d)  # will use kv_bt2 via closure? no—re-call
# start_pos=None
try:
    out, *_ = kernels.compressor(x=x2d, wkv=wkv, wgate=wgate, kv_state=kv_state,
        score_state=score_state, ape=ape, norm_weight=norm_weight, rope_sin=rope_sin,
        rope_cos=rope_cos, kv_block_table=kv_bt, score_block_table=score_bt,
        cu_seqlens=cu_seqlens, seqused=None, start_pos=None, rope_head_dim=64,
        cmp_ratio=4, coff=2, norm_eps=1e-6, rotary_mode=2, enable_grad=False)
    print(f"[OK] start_pos=None: out={out.shape}")
except Exception as e:
    print(f"[FAIL] start_pos=None: {next((l.strip() for l in str(e).splitlines() if 'blockDim' in l or 'EZ' in l), str(e)[:120])}")
print("=== done ===")
