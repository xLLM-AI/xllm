"""Accuracy runner. Fails rather than claiming success without an NPU."""
import argparse, importlib.util, json, sys, time, traceback
from pathlib import Path
import numpy as np
import torch, torch_npu

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"))
from solution import OP_INDEX,prepare,run,run_direct
from baseline import candidates

def load_package(path):
    sys.path.insert(0,str(path))
    import golden,generate_data
    return golden,generate_data

def device_inputs(case,arrays,device):
    result=[]
    for d,a in zip(case["input_desc"],arrays):
        dtype={"float16":torch.float16,"bfloat16":torch.bfloat16}[d["data_type"]]
        value=torch.from_numpy(a.view(np.int16).copy()).to(device).view(dtype)
        result.append(value)
    return result

def cpu_array(t):
    if t.dtype==torch.bfloat16:
        from ml_dtypes import bfloat16
        return t.view(torch.int16).cpu().numpy().view(bfloat16)
    return t.cpu().numpy()

def compare(case,arrays,actual,expected):
    if len(actual)!=len(expected): raise AssertionError("wrong output count")
    records=[]
    for i,(t,e) in enumerate(zip(actual,expected)):
        if not isinstance(t,torch.Tensor) or t.device.type!="npu" or t.device.index!=torch.npu.current_device(): raise AssertionError("output must be on the selected NPU")
        if not t.is_contiguous(): raise AssertionError("output is not contiguous")
        if list(t.shape)!=case["output_desc"][i]["shape"]: raise AssertionError("wrong output shape")
        a=cpu_array(t)
        if a.dtype!=e.dtype: raise AssertionError("output dtype mismatch")
        record={"output":case["output_desc"][i]["name"],"elements":int(e.size)}
        if OP_INDEX==3:
            mismatch=int(np.count_nonzero(a.view(np.uint16)!=e.view(np.uint16)))
            record["bit_mismatches"]=mismatch
            if mismatch: raise AssertionError(record)
        elif OP_INDEX==2:
            if i>=2:
                if not np.array_equal(a.view(np.uint32),e.view(np.uint32)): raise AssertionError("absmax must be bit-exact")
            else:
                scale=expected[2][:,None] if i==0 else expected[3][None,:]
                ratio=np.zeros_like(scale)
                np.divide(np.float32(127),scale,out=ratio,where=scale!=0)
                scaled=arrays[0].astype(np.float32)*ratio
                near=np.abs(scaled-np.floor(scaled)-np.float32(.5))<=np.float32(2e-5)
                delta=np.abs(a.astype(np.int16)-e.astype(np.int16))
                allowed=(delta==0)|((delta<=1)&near)
                if case["kind"].startswith("ties") or case["kind"]=="zeros": allowed=delta==0
                record.update(mismatches=int(np.count_nonzero(delta)),near_half_elements=int(np.count_nonzero(near)))
                if np.any(a==-128) or not allowed.all(): raise AssertionError(record)
        else:
            af,ef=a.astype(np.float64),e.astype(np.float64)
            tolerance=2e-3 if a.dtype==np.float16 else 2e-2
            error=np.abs(af-ef)
            record.update(max_abs_error=float(error.max()),max_scaled_error=float((error/(tolerance+tolerance*np.abs(ef))).max()))
            if not np.isfinite(af).all() or not np.all(error<=tolerance+tolerance*np.abs(ef)): raise AssertionError(record)
            if case["kind"]=="constant" or arrays[0].shape[1]==1:
                beta=arrays[2 if i==0 else 4]
                if not np.array_equal(a,np.broadcast_to(beta,a.shape)): raise AssertionError("constant-row result differs from beta")
        records.append(record)
    return records


def validate_ownership(inputs, outputs):
    if not all(isinstance(t,torch.Tensor) for t in outputs):
        raise AssertionError('outputs must be tensors')
    if any(t.device!=inputs[0].device for t in outputs):
        raise AssertionError('outputs must be on the input NPU device')
    input_storage={t.untyped_storage().data_ptr() for t in inputs if t.numel()}
    output_storage=[t.untyped_storage().data_ptr() for t in outputs if t.numel()]
    if len(set(output_storage))!=len(output_storage) or input_storage.intersection(output_storage):
        raise AssertionError('output shares storage with input or another output')

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-dir",type=Path,required=True)
    p.add_argument("--device",type=int,default=0)
    p.add_argument("--case",action="append")
    p.add_argument("--report",type=Path,default=ROOT/"task_submission/accuracy.json")
    args=p.parse_args()
    args.report.parent.mkdir(parents=True,exist_ok=True)
    result={"status":"running","npu_executed":False,"cases":[]}
    args.report.write_text(json.dumps(result))
    try:
        torch.npu.set_device(args.device)
        golden,data=load_package(args.task_dir.resolve())
        cases=json.loads((args.task_dir/"case.json").read_text())
        if args.case and set(args.case)-{c["case_name"] for c in cases}: raise ValueError("unknown case")
        for case in cases:
            if args.case and case["case_name"] not in args.case: continue
            arrays=data.generate(case)
            attrs={x["name"]:x["value"] for x in case["attr_desc"]}
            expected=golden.calculate(arrays,attrs)
            inputs=device_inputs(case,arrays,f"npu:{args.device}")
            plan=prepare(inputs,attrs)
            outputs=plan()
            torch.npu.synchronize()
            validate_ownership(inputs,outputs)
            checks=compare(case,arrays,outputs,expected)
            native_outputs=run(inputs,attrs)
            validate_ownership(inputs,native_outputs)
            native_checks=compare(case,arrays,native_outputs,expected)
            direct_native=run_direct(inputs,attrs)
            validate_ownership(inputs,direct_native)
            direct_native_checks=compare(case,arrays,direct_native,expected)
            for src,original in zip(inputs,arrays):
                if not np.array_equal(cpu_array(src).view(np.uint16),original.view(np.uint16)):
                    raise AssertionError("input mutated")
            addresses=[t.data_ptr() for t in outputs]
            if len(addresses)!=len(set(addresses)) or set(addresses)&{t.data_ptr() for t in inputs}:
                raise AssertionError("output aliases another tensor")
            baselines={}
            if case["performance"]:
                for name,fn in candidates(inputs,attrs).items():
                    try:
                        baseline_outputs=fn(); torch.npu.synchronize()
                        compare(case,arrays,baseline_outputs,expected)
                        baselines[name]="passed"
                    except Exception as exc: baselines[name]="failed: "+str(exc)
                if not any(x=="passed" for x in baselines.values()): raise AssertionError("no valid performance baseline")
            record={"case_name":case["case_name"],"status":"passed","outputs":checks,
                    "workspace_bytes":plan.workspace_bytes,"native_outputs":native_checks,"direct_native_outputs":direct_native_checks,"baselines":baselines}
            result["cases"].append(record)
            result["npu_executed"]=True
            print(json.dumps(record,ensure_ascii=False),flush=True)
        if not result["cases"]: raise AssertionError("no cases executed")
        result["status"]="passed"
    except Exception as exc:
        result["status"]="failed"; result["error"]=str(exc)
        raise
    finally:
        result["finished_unix"]=time.time()
        args.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__": main()
