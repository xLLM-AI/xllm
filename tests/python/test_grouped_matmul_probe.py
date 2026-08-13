"""Probe: does NZ format / contiguous state cause the '4096 vs 512' error?

Real w2 prints as [32,256,4096] (correct) but the kernel reads dim 512
(=2*inter, the w13 out). Hypothesis: ALLOW_INTERNAL_FORMAT=1 auto-converts
the loaded int8 weight to FRACTAL_NZ, and the kernel mis-reads its dims.
Test w2 as ND vs NZ, and as .contiguous() vs transposed-view.
"""
import torch
import torch_npu

torch_npu.npu.set_device(0)
dev = torch.device("npu:0")
_FRACTAL_NZ = 29

G = 32
counts = [0] * 32
counts[2] = 4; counts[3] = 2; counts[5] = 1; counts[10] = 3; counts[15] = 2; counts[20] = 1
M = sum(counts)
K = 256
N = 4096
group_list = torch.tensor(counts, dtype=torch.int64, device=dev)
act_pt = torch.rand(M, dtype=torch.float32, device=dev) * 0.5 + 0.75
act_i8 = torch.randint(-127, 127, (M, K), dtype=torch.int8, device=dev)
w2_scale_bf16 = (torch.rand(G, N, dtype=torch.float32, device=dev) * 0.01 + 0.001).to(torch.bfloat16)

def try_w2(label, w2):
    try:
        out = torch.ops.npu.npu_grouped_matmul(
            x=[act_i8], weight=[w2], scale=[w2_scale_bf16], per_token_scale=[act_pt],
            split_item=2, group_list_type=1, group_type=0, group_list=group_list,
            output_dtype=torch.bfloat16)[0]
        print(f"[OK] {label}: out={out.shape} fmt={torch_npu.get_npu_format(w2)}")
    except Exception as e:
        acl = next((l.strip() for l in str(e).splitlines() if 'dim' in l.lower() or 'AclNN' in l), str(e)[:140])
        print(f"[FAIL] {label}: {acl} fmt={torch_npu.get_npu_format(w2)}")

print(f"=== NZ/contiguous probe: G={G} M={M} K={K} N={N} ===")
# 1. Fresh ND contiguous [G,K,N]
w2_nd = torch.randint(-127, 127, (G, K, N), dtype=torch.int8, device=dev)
try_w2("fresh ND [G,K,N]", w2_nd)
# 2. Built as [G,N,K] then transposed+contiguous (mimics process_weights_after_loading)
w2_oi = torch.randint(-127, 127, (G, N, K), dtype=torch.int8, device=dev)
w2_t = w2_oi.transpose(1, 2).contiguous()
try_w2("transposed-from-[G,N,K] .contiguous()", w2_t)
# 3. Transposed but NOT contiguous (view)
w2_tv = w2_oi.transpose(1, 2)
try_w2("transposed NOT contiguous (view)", w2_tv)
# 4. Force NZ format on the transposed weight
w2_nz = torch_npu.npu_format_cast(w2_t.clone(), _FRACTAL_NZ)
try_w2("transposed + NZ format", w2_nz)
# 5. NZ on fresh ND
w2_nd_nz = torch_npu.npu_format_cast(w2_nd.clone(), _FRACTAL_NZ)
try_w2("fresh ND + NZ format", w2_nd_nz)
print("=== done ===")
