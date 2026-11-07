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
    qkv,pk,pv=inputs
    b,h,l,d=pk.shape
    q,k,v=[np.array(a.reshape(b,1,h,d).transpose(0,2,1,3),copy=True,order="C") for a in np.split(qkv,3,axis=-1)]
    return q,np.concatenate((pk,k),axis=2),np.concatenate((pv,v),axis=2)

def calc_expect_func(input_data, output_desc=None, attr=None):
    attrs = {} if attr is None else ({x["name"]: x["value"] for x in attr} if isinstance(attr, list) else attr)
    return list(calculate(input_data, attrs))
