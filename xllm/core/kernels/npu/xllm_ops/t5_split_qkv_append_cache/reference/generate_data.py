"""Deterministic test inputs; relative paths only; supports raw .bin export."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from golden import to_dtype

ROOT = Path(__file__).resolve().parent

def generate(case):
    seed = int.from_bytes(hashlib.sha256(("20261014/" + case["case_name"]).encode()).digest()[:8], "little")
    rng = np.random.default_rng(seed)
    inputs = []
    for desc in case["input_desc"]:
        value = rng.normal(size=desc["shape"]).astype(np.float32)
        if desc["name"].startswith("gamma"):
            value = rng.uniform(-2, 2, desc["shape"]).astype(np.float32)
        inputs.append(to_dtype(value, desc["data_type"]))
    kind = case.get("kind", "random")
    if kind == "constant": inputs[0][...] = 3
    if kind == "zero_gamma": inputs[1][...] = 0
    if kind == "offset":
        inputs[0][...] = 8192
        inputs[0][:, -1] = 8256
    if kind == "zeros": inputs[0][...] = 0
    if kind == "zero_axes":
        inputs[0][0, :] = 0
        inputs[0][:, 0] = 0
    if kind == "extreme":
        inputs[0][...] = np.nextafter(np.float16(0), np.float16(1))
        inputs[0][-1,-1] = 65504
        inputs[0][0,0] = -65504
    if kind == "ties_row":
        inputs[0][...] = [[127,.5,1.5,2.5,-.5,-1.5,-2.5,0]]
    if kind == "ties_col":
        inputs[0][...] = np.repeat(np.array([127,.5,1.5,2.5,-.5,-1.5,-2.5,0])[:,None], 2, 1)
    if kind == "bits":
        for i, a in enumerate(inputs):
            a.view(np.uint16).reshape(-1)[:] = (np.arange(a.size,dtype=np.uint32)+i*7919).astype(np.uint16)
    if kind == "ln_limits":
        inputs[0][...] = 10000
        inputs[0][:,::2] = -10000
        for a in inputs[1:]:a[...] = 8
        inputs[3][...] = -8
    if kind == "ln_tiny":
        inputs[0][...] = 0
        inputs[0][:,::2] = np.float32(2**-20)
    if kind == "subnormal":
        inputs[0][...] = np.nextafter(np.float16(0),np.float16(1))
        inputs[0][:,::2] *= -1
    if kind == "signed_zero":
        inputs[0][...] = np.float16(0)
        inputs[0][:,::2] = np.float16(-0.)
    return inputs

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir",type=Path,default=ROOT/"generated_data")
    parser.add_argument("--case",action="append")
    args = parser.parse_args()
    cases = json.loads((ROOT/"case.json").read_text())
    if args.case:
        names = {c["case_name"] for c in cases}
        if set(args.case)-names: parser.error("unknown case name")
    args.output_dir.mkdir(parents=True,exist_ok=True)
    manifest=[]
    for case in cases:
        if args.case and case["case_name"] not in args.case: continue
        values=generate(case)
        files=[]
        for desc,a in zip(case["input_desc"],values):
            filename=case["case_name"]+"_"+desc["name"]+".bin"
            data=a.tobytes()
            (args.output_dir/filename).write_bytes(data)
            files.append({"name":desc["name"],"file":filename,"bytes":len(data),"sha256":hashlib.sha256(data).hexdigest()})
        manifest.append({"case_name":case["case_name"],"inputs":files})
    (args.output_dir/"manifest.json").write_text(json.dumps(manifest,indent=2)+"\n")
    print(f"Generated {len(manifest)} cases")

if __name__ == "__main__": main()
