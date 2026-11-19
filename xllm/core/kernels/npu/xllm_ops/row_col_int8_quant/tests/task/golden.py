"""Independent CPU oracle. No solution code is imported."""
import numpy as np

def to_dtype(value, dtype):
    if dtype == "float16":
        return np.asarray(value, dtype=np.float16)
    if dtype == "bfloat16":
        from ml_dtypes import bfloat16
        return np.asarray(value, dtype=bfloat16)
    return np.asarray(value, dtype=np.dtype(dtype))

def calculate(inputs, attributes):
    x=inputs[0].astype(np.float32)
    r=np.max(np.abs(x),axis=1)
    c=np.max(np.abs(x),axis=0)
    def quant(scale):
        ratio=np.zeros_like(scale)
        np.divide(np.float32(127),scale,out=ratio,where=scale!=0)
        return np.clip(np.rint(x*ratio),-127,127).astype(np.int8)
    return quant(r[:,None]),quant(c[None,:]),r,c

def calc_expect_func(input_data, output_desc=None, attr=None):
    attrs = {} if attr is None else ({x["name"]: x["value"] for x in attr} if isinstance(attr, list) else attr)
    return list(calculate(input_data, attrs))
