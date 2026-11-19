"""NPU metadata rejection, independent streams, repeated calls and allocator checks."""
import argparse
import gc
import json
import sys
from pathlib import Path
import torch
import torch_npu

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'python'))
sys.path.insert(0,str(ROOT/'tests'))
from solution import OP_INDEX,prepare,run,aclnn_workspace_bytes
from baseline import candidates
from run_tests import load_package,device_inputs,compare


def rejects(fn):
    try:
        fn()
    except (ValueError,RuntimeError):
        return
    raise AssertionError('invalid metadata was accepted')


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--task-dir',type=Path,required=True)
    parser.add_argument('--device',type=int,default=0)
    parser.add_argument('--report',type=Path,default=ROOT/'task_submission/engineering.json')
    args=parser.parse_args()
    report={'status':'running','npu_executed':False,'negative_checks':0,'streams':[], 'memory':[]}
    args.report.parent.mkdir(parents=True,exist_ok=True)
    args.report.write_text(json.dumps(report))
    try:
        torch.npu.set_device(args.device)
        golden,data=load_package(args.task_dir.resolve())
        cases=json.loads((args.task_dir/'case.json').read_text())
        selected=[c for c in cases if c['performance']]
        stream_cases=[selected[0],selected[-1]]
        streams=[torch.npu.Stream() for _ in range(2)]
        saved=[]
        for stream,case in zip(streams,stream_cases):
            arrays=data.generate(case)
            attrs={a['name']:a['value'] for a in case['attr_desc']}
            inputs=device_inputs(case,arrays,f'npu:{args.device}')
            stream.wait_stream(torch.npu.current_stream())
            with torch.npu.stream(stream):
                plan=prepare(inputs,attrs)
                for _ in range(10): outputs=plan()
                native=run(inputs,attrs)
            saved.append((case,arrays,attrs,inputs,plan,outputs,native))
        torch.npu.synchronize()
        for case,arrays,attrs,inputs,plan,outputs,native in saved:
            compare(case,arrays,outputs,golden.calculate(arrays,attrs))
            compare(case,arrays,native,golden.calculate(arrays,attrs))
            report['streams'].append({'case_name':case['case_name'],'status':'passed','calls':10})
        report['npu_executed']=True
        # Drop input references before completion; force allocator reuse from a
        # different stream to validate the native stream-recording contract.
        case=stream_cases[0];arrays=data.generate(case)
        attrs={a['name']:a['value'] for a in case['attr_desc']}
        with torch.npu.stream(streams[0]):
            temporary_inputs=device_inputs(case,arrays,f'npu:{args.device}')
        streams[1].wait_stream(streams[0])
        with torch.npu.stream(streams[1]):
            early_outputs=run(temporary_inputs,attrs)
        shapes=[(t.shape,t.dtype) for t in temporary_inputs]
        del temporary_inputs
        with torch.npu.stream(streams[0]):
            for _ in range(30):
                garbage=[torch.empty(shape,dtype=dtype,device=f'npu:{args.device}').fill_(7) for shape,dtype in shapes]
        del garbage
        torch.npu.synchronize()
        compare(case,arrays,early_outputs,golden.calculate(arrays,attrs))
        del early_outputs
        report['early_release_cross_stream']='passed'
        # Keep eight calls live while another 32 calls release outputs early.
        # Interleave same-stream allocation/writes to exercise FIFO-safe reuse.
        lifetime_case=selected[min(2,len(selected)-1)]
        lifetime_arrays=data.generate(lifetime_case)
        lifetime_attrs={a['name']:a['value'] for a in lifetime_case['attr_desc']}
        with torch.npu.stream(streams[1]):
            lifetime_inputs=device_inputs(lifetime_case,lifetime_arrays,f'npu:{args.device}')
            held=[run(lifetime_inputs,lifetime_attrs) for _ in range(8)]
            storage_ids=[t.untyped_storage().data_ptr() for output in held for t in output]
            assert len(storage_ids)==len(set(storage_ids)), 'live calls share output storage'
            for _ in range(32):
                released=run(lifetime_inputs,lifetime_attrs)
                del released
                pressure=[torch.empty_like(t).fill_(7) for t in held[0]]
                del pressure
        torch.npu.synchronize()
        expected_lifetime=golden.calculate(lifetime_arrays,lifetime_attrs)
        for output in held:compare(lifetime_case,lifetime_arrays,output,expected_lifetime)
        report['output_lifetime']={'status':'passed','case_name':lifetime_case['case_name'],
            'retained_calls':8,'released_calls':32,'same_stream_pressure_calls':32,
            'distinct_live_output_storages':True}
        del held,output,lifetime_inputs
        # Use metadata changes only; no invalid kernel is launched.
        case,arrays,attrs,inputs,plan,outputs,native=saved[0]
        checks=[lambda:prepare(inputs[:-1],attrs),
                lambda:prepare([inputs[0].float(),*inputs[1:]],attrs)]
        if OP_INDEX==1:
            checks.extend([lambda:prepare(inputs,{'epsilon':0}),
                           lambda:prepare(inputs,{'epsilon':float('nan')}),
                           lambda:prepare([inputs[0],inputs[1][:-1],*inputs[2:]],attrs),
                           lambda:prepare([inputs[0].t(),*inputs[1:]],attrs)])
        elif OP_INDEX==2:
            checks.extend([lambda:prepare([inputs[0][:,:3]],attrs),
                           lambda:prepare([inputs[0].t()],attrs),
                           lambda:prepare([inputs[0][:0]],attrs)])
        else:
            checks.extend([lambda:prepare(inputs,{**attrs,'headNum':attrs['headNum']+1}),
                           lambda:prepare([inputs[0],inputs[1],inputs[2][:,:-1]],attrs)])
        for check in checks:
            rejects(check);report['negative_checks']+=1
        rejects(lambda:run(inputs[:-1],attrs)); report['negative_checks']+=1
        del saved,plan,outputs,native
        torch.npu.synchronize();gc.collect()
        for case in selected:
            arrays=data.generate(case);attrs={a['name']:a['value'] for a in case['attr_desc']}
            inputs=device_inputs(case,arrays,f'npu:{args.device}')
            # Warm up registration and device runtime before the memory baseline.
            p=prepare(inputs,attrs);p();torch.npu.synchronize();del p;gc.collect()
            before=torch.npu.memory_allocated()
            torch.npu.reset_peak_memory_stats()
            plan=prepare(inputs,attrs)
            explicit=sum(t.numel()*t.element_size() for t in plan.outputs)+plan.workspace_bytes
            plan();torch.npu.synchronize()
            peak=torch.npu.max_memory_allocated()
            workspace=plan.workspace_bytes
            del plan;gc.collect();torch.npu.synchronize()
            after=torch.npu.memory_allocated()
            for _ in range(20):
                plan=prepare(inputs,attrs);plan();torch.npu.synchronize();del plan
            gc.collect();torch.npu.synchronize()
            repeated=torch.npu.memory_allocated()
            record={'case_name':case['case_name'],'explicit_output_and_workspace_bytes':explicit,
                    'workspace_bytes':workspace,'allocator_before':before,'allocator_peak':peak,
                    'allocator_after':after,'allocator_after_20_calls':repeated,
                    'scope':'PyTorch live tensor allocator only; not total device HBM or driver memory'}
            # Include native entry point and valid baseline peaks with fresh outputs.
            for name,fn in {'native':lambda:run(inputs,attrs),**candidates(inputs,attrs)}.items():
                value=fn();torch.npu.synchronize();del value;gc.collect()
                start=torch.npu.memory_allocated();torch.npu.reset_peak_memory_stats()
                value=fn();torch.npu.synchronize()
                peak_delta=torch.npu.max_memory_allocated()-start
                del value;gc.collect();torch.npu.synchronize()
                if torch.npu.memory_allocated()!=start: raise AssertionError('native/baseline allocation leak')
                record[name+'_allocator_peak_delta']=peak_delta
                if name=='native':
                    api_workspace=aclnn_workspace_bytes()
                    record['direct_workspace_bytes']=record['workspace_bytes']
                    record['native_workspace_bytes']=api_workspace
                    record['workspace_bytes']=api_workspace
                    record['explicit_output_and_workspace_bytes']=explicit-workspace+api_workspace
                    for _ in range(20):
                        value=fn();torch.npu.synchronize();del value
                    gc.collect();torch.npu.synchronize()
                    record['native_allocator_after_20_calls']=torch.npu.memory_allocated()
                    record['native_allocator_before_20_calls']=start
                    if record['native_allocator_after_20_calls']!=start:raise AssertionError('native repeated allocation leak')
            report['memory'].append(record)
            print(json.dumps(record),flush=True)
            if after>before or repeated>before: raise AssertionError('live tensor allocation did not return to baseline')
        if OP_INDEX==3:
            case=next(c for c in cases if c['input_desc'][1]['shape'][2]==31)
            arrays=data.generate(case);attrs={a['name']:a['value'] for a in case['attr_desc']}
            inputs=device_inputs(case,arrays,f'npu:{args.device}')
            first=run(inputs,attrs)
            second=run((inputs[0],first[1],first[2]),attrs)
            expected1=golden.calculate(arrays,attrs)
            expected2=golden.calculate((arrays[0],expected1[1],expected1[2]),attrs)
            import copy
            second_case=copy.deepcopy(case)
            for desc in second_case['output_desc'][1:]:desc['shape'][2]+=1
            compare(second_case,arrays,second,expected2)
            report['sequential_cache_append']='passed'
        report['status']='passed'
        print(json.dumps({'status':'passed','negative_checks':report['negative_checks'],'streams':report['streams']}),flush=True)
    except Exception as exc:
        report.update(status='failed',error=str(exc));raise
    finally:
        args.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')


if __name__=='__main__':main()
