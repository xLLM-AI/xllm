"""API-to-completion latency and device-event observation; no invented baseline."""
import argparse,json,statistics,sys,time
from pathlib import Path
import torch,torch_npu
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"python"));sys.path.insert(0,str(ROOT/"tests"))
from solution import prepare,run
from baseline import candidates
from run_tests import load_package,device_inputs,compare

def measure(fn,warmup,iterations,rounds):
    for _ in range(warmup): fn()
    torch.npu.synchronize()
    wall=[]; events=[]
    for _ in range(rounds):
        start,end=(torch.npu.Event(enable_timing=True) for _ in range(2))
        torch.npu.synchronize(); before=time.perf_counter_ns()
        start.record()
        for _ in range(iterations): fn()
        end.record(); end.synchronize()
        wall.append((time.perf_counter_ns()-before)/iterations/1000)
        events.append(start.elapsed_time(end)*1000/iterations)
    return {"wall_us_rounds":wall,"device_event_us_rounds":events,
            "wall_us_median":statistics.median(wall),"device_event_us_median":statistics.median(events)}

def main():
    p=argparse.ArgumentParser()
    p.add_argument("--task-dir",type=Path,required=True);p.add_argument("--device",type=int,default=0)
    p.add_argument("--warmup",type=int,default=50);p.add_argument("--iterations",type=int,default=200)
    p.add_argument("--rounds",type=int,default=5)
    p.add_argument("--report",type=Path,default=ROOT/"task_submission/performance.json")
    args=p.parse_args()
    if min(args.warmup,args.iterations,args.rounds)<1: p.error("counts must be positive")
    report={"status":"running","npu_executed":False,"cases":[],"protocol":vars(args).copy(),
      "timing_scope":"Both run APIs allocate fresh outputs; input transfers and golden comparisons excluded. Device events include dispatch gaps; these are not pure kernel profiler times."}
    report["protocol"]={k:str(v) if isinstance(v,Path) else v for k,v in report["protocol"].items()}
    args.report.parent.mkdir(parents=True,exist_ok=True);args.report.write_text(json.dumps(report))
    try:
        torch.npu.set_device(args.device)
        golden,data=load_package(args.task_dir.resolve())
        for case in json.loads((args.task_dir/"case.json").read_text()):
            if not case["performance"]: continue
            arrays=data.generate(case);attrs={a["name"]:a["value"] for a in case["attr_desc"]}
            expected=golden.calculate(arrays,attrs);inputs=device_inputs(case,arrays,f"npu:{args.device}")
            custom=lambda:run(inputs,attrs)
            compare(case,arrays,custom(),expected)
            functions={"custom":custom};rejected={}
            for name,fn in candidates(inputs,attrs).items():
                try: compare(case,arrays,fn(),expected);functions[name]=fn
                except Exception as exc: rejected[name]=str(exc)
            if len(functions)==1: raise RuntimeError("no valid baseline")
            numbers={name:measure(fn,args.warmup,args.iterations,args.rounds) for name,fn in functions.items()}
            fastest=min(v["wall_us_median"] for k,v in numbers.items() if k!="custom")
            record={"case_name":case["case_name"],"results":numbers,"rejected_baselines":rejected,
                    "wall_speedup":fastest/numbers["custom"]["wall_us_median"]}
            report["cases"].append(record);report["npu_executed"]=True
            print(json.dumps(record),flush=True)
        if not report["cases"]: raise RuntimeError("no performance cases")
        report["status"]="passed"
    except Exception as exc:
        report.update(status="failed",error=str(exc));raise
    finally: args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+"\n")

if __name__=="__main__":main()
