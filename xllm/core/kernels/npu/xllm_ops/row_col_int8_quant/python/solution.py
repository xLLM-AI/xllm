import os
from pathlib import Path
_vendor=Path(__file__).resolve().parents[1]/"aclnn_install/vendors/community_task2"
_paths=os.environ.get("ASCEND_CUSTOM_OPP_PATH","").split(":")
if _vendor.is_dir() and str(_vendor) not in _paths:
    os.environ["ASCEND_CUSTOM_OPP_PATH"]=":".join([str(_vendor)]+[p for p in _paths if p])
"""Ascend C kernel bindings. Metadata on host; tensor computation on NPU."""
from pathlib import Path
import ctypes as C
import functools
import math
import torch
import torch_npu
OP_INDEX = 2
ROOT = Path(__file__).resolve().parents[1]

@functools.lru_cache(None)
def library():
    path = ROOT / 'build/libtask2_kernels.so'
    if not path.is_file():
        raise RuntimeError('Run bash build.sh first')
    return C.CDLL(str(path))

def function(name, pointers, scalars):
    fn = getattr(library(), name)
    fn.argtypes = [C.c_uint32, C.c_void_p] + [C.c_void_p] * pointers + scalars
    fn.restype = None
    return fn

class Prepared:

    def __init__(self, inputs, outputs, workspace, launches):
        self.inputs, self.outputs, self.workspace, self.launches = (inputs, outputs, workspace, launches)
        self.workspace_bytes = sum((t.numel() * t.element_size() for t in workspace))

    def __call__(self):
        with torch.npu.device(self.inputs[0].device):
            stream = torch.npu.current_stream()
            for fn, blocks, pointers, attrs in self.launches:
                fn(blocks, stream.npu_stream, *[t.data_ptr() for t in pointers], *attrs)
            for t in (*self.inputs, *self.outputs, *self.workspace):
                t.record_stream(stream)
        return self.outputs

def prepare(inputs, attributes=None):
    attrs = validate_attributes(attributes)
    inputs = tuple(inputs)
    expected = {1: 5, 2: 1, 3: 3}[OP_INDEX]
    if len(inputs) != expected:
        raise ValueError('wrong input count')
    x = inputs[0]
    if not all((isinstance(t, torch.Tensor) for t in inputs)):
        raise ValueError('tensor inputs required')
    if x.device.type != 'npu':
        raise ValueError('NPU tensors required')
    if any((t.device != x.device or t.dtype != x.dtype or (not t.is_contiguous()) for t in inputs)):
        raise ValueError('inputs must share device/dtype and be contiguous')
    allowed = (torch.float16,) if OP_INDEX == 2 else (torch.float16, torch.bfloat16)
    if x.dtype not in allowed:
        raise ValueError('unsupported dtype')
    if x.ndim != 2:
        raise ValueError('x must be a matrix')
    m, n = x.shape
    if not 1 <= m <= 4096 or not 2 <= n <= 4096 or n % 2:
        raise ValueError('shape outside current contract')
    qr, qc = (torch.empty(x.shape, dtype=torch.int8, device=x.device) for _ in range(2))
    r = torch.empty((m,), dtype=torch.float32, device=x.device)
    c = torch.empty((n,), dtype=torch.float32, device=x.device)
    rt, ct = ((m + 63) // 64, (n + 255) // 256)
    pr = torch.empty((rt * ct * 64,), dtype=torch.float32, device=x.device)
    pc = torch.empty((rt * ct * 256,), dtype=torch.float32, device=x.device)
    stats = function('row_col_stats_launch', 3, [C.c_uint32, C.c_uint32])
    merge = function('row_col_merge_launch', 4, [C.c_uint32, C.c_uint32])
    quant = function('row_col_quant_launch', 5, [C.c_uint32, C.c_uint32])
    calls = [(stats, min(48, rt * ct), (x, pr, pc), (m, n)), (merge, min(48, rt + ct), (pr, pc, r, c), (m, n)), (quant, min(48, rt * ct), (x, r, c, qr, qc), (m, n))]
    outputs = (qr, qc, r, c)
    workspace = (pr, pc)
    return Prepared(inputs, outputs, workspace, calls)

@functools.lru_cache(None)
def native_function():
    torch.ops.load_library(str(ROOT / 'build/libtask2_torch.so'))
    return torch.ops.cann_community.row_col_int8_quant

def run(inputs, attributes=None):
    attrs = validate_attributes(attributes)
    return tuple(native_function()(inputs, float(attrs.get('epsilon', 1e-05))))

def validate_attributes(attributes):
    attrs = {} if attributes is None else attributes
    if not isinstance(attrs, dict):
        raise ValueError('attributes must be a dictionary')
    allowed = set()
    if set(attrs) - allowed:
        raise ValueError('unknown attributes')
    if OP_INDEX == 1 and 'epsilon' in attrs:
        if isinstance(attrs['epsilon'], bool) or not isinstance(attrs['epsilon'], (float, int)):
            raise ValueError('epsilon must be a real scalar')
    return attrs

def run_aclnn(inputs, attributes=None):
    attrs=validate_attributes(attributes)
    inputs=tuple(inputs)
    if len(inputs)!=1:raise ValueError('wrong input count')
    if not all(isinstance(t,torch.Tensor) for t in inputs):raise ValueError('tensor inputs required')
    fn,library=_aclnn_function()
    return tuple(fn(inputs,float(attrs.get('epsilon',1e-5)),library))

run_direct = run
run = run_aclnn

def aclnn_workspace_bytes():
    native_function()
    return int(torch.ops.cann_community.row_col_int8_quant_workspace_bytes())

@functools.lru_cache(None)
def _aclnn_function():
    native_function()
    library=ROOT/'aclnn_install/vendors/community_task2/op_api/lib/libcust_opapi.so'
    if not library.is_file():raise RuntimeError('Run python3 tools/build_aclnn.py first')
    return torch.ops.cann_community.row_col_int8_quant_aclnn,str(library)
