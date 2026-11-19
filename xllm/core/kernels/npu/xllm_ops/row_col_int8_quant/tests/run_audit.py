"""Additional NPU contract checks: invalid metadata, ownership, seeds and offsets."""
from pathlib import Path
import copy,json,sys
import numpy as np
import torch,torch_npu
ROOT=Path(__file__).resolve().parents[1]
sys.path[:0]=[str(ROOT/'python'),str(ROOT/'tests'),str(ROOT/'tests/task')]
from solution import OP_INDEX,prepare,run,run_direct
from run_tests import load_package,device_inputs,compare,validate_ownership,cpu_array
import verify

def reject(fn):
    try:fn()
    except (ValueError,TypeError,RuntimeError,AssertionError,NotImplementedError):return
    raise AssertionError('invalid result or metadata was accepted')

def main():
    torch.npu.set_device(0)
    path=ROOT/'task_submission/audit.json'
    result={'status':'running','npu_executed':False,'negative_cases':[], 'randomized_cases':[], 'verifier_checks':[]}
    try:
        golden,data=load_package(ROOT/'tests/task')
        cases=json.loads((ROOT/'tests/task/case.json').read_text())
        case=next(c for c in cases if c['performance'])
        arrays=data.generate(case);attrs={a['name']:a['value'] for a in case['attr_desc']}
        inputs=device_inputs(case,arrays,'npu:0')
        def empty(shape,dtype=None):return torch.empty(shape,dtype=dtype or inputs[0].dtype,device='npu:0')
        bad=[('input_count',inputs[:-1],attrs),('cpu_input',[inputs[0].cpu(),*inputs[1:]],attrs),
             ('float32',[inputs[0].float(),*inputs[1:]],attrs),('unknown_attribute',inputs,{'unknown':1}),
             ('attributes_type',inputs,[])]
        if OP_INDEX==1:
            n,h=inputs[0].shape
            bad += [('rank',[inputs[0].reshape(-1),*inputs[1:]],attrs),('empty_n',[empty((0,h)),*inputs[1:]],attrs),
                    ('large_n',[empty((4097,h)),*inputs[1:]],attrs),('large_h',[empty((1,8193)),*inputs[1:]],attrs),
                    ('affine_length',[inputs[0],inputs[1][:-1],*inputs[2:]],attrs),
                    ('affine_dtype',[inputs[0],inputs[1].float(),*inputs[2:]],attrs),
                    ('noncontiguous',[empty((n,h*2))[:,::2],*inputs[1:]],attrs)]
            for v in [0,-1,1e-8,.1,float('nan'),float('inf'),True,'1e-5']:
                bad.append(('epsilon_'+repr(v),inputs,{'epsilon':v}))
        elif OP_INDEX==2:
            bad += [('rank',[empty((2,2,2))],attrs),('empty_m',[empty((0,2))],attrs),('empty_n',[empty((2,0))],attrs),
                    ('odd_n',[empty((2,3))],attrs),('large_m',[empty((4097,2))],attrs),('large_n',[empty((2,4098))],attrs),
                    ('bf16',[inputs[0].bfloat16()],attrs),('noncontiguous',[empty((4,8))[:,::2]],attrs)]
        else:
            b,h,l,d=inputs[1].shape
            bad += [('cache_rank',[inputs[0],empty((2,3)),inputs[2]],attrs),
                    ('cache_mismatch',[inputs[0],inputs[1],empty((b,h,l+1,d))],attrs),
                    ('packed_mismatch',[empty((b,1,3*h*d+1)),*inputs[1:]],attrs),
                    ('packed_tokens',[empty((b,2,3*h*d)),*inputs[1:]],attrs),
                    ('cache_dtype',[inputs[0],inputs[1].float(),inputs[2]],attrs),
                    ('noncontiguous',[empty((b,1,6*h*d))[:,:,::2],*inputs[1:]],attrs)]
            for key in ['headNum','headDim']:
                for v in [-1,0,1.5,True,'8',attrs[key]+1]:bad.append((key+'_'+repr(v),inputs,{**attrs,key:v}))
            for b1,h1,l1,d1 in [(0,1,0,1),(65,1,0,1),(1,129,0,1),(1,1,8193,1),(1,1,0,257)]:
                bad.append(('shape_limit_'+str((b1,h1,l1,d1)),[empty((b1,1,3*h1*d1)),empty((b1,h1,l1,d1)),empty((b1,h1,l1,d1))],{}))
        for label,values,attributes in bad:
            for entry,fn in [('prepare',prepare),('run',run),('run_direct',run_direct)]:
                reject(lambda:fn(values,attributes))
                result['negative_cases'].append({'case':label,'entry':entry,'status':'passed'})
        outputs=run(inputs,attrs);torch.npu.synchronize()
        expected=golden.calculate(arrays,attrs)
        reject(lambda:verify.compare(case,arrays,[t.cpu() for t in outputs],expected))
        result['verifier_checks'].append('CPU output rejected')
        reject(lambda:verify.validate_ownership(inputs,[inputs[0],*outputs[1:]]))
        result['verifier_checks'].append('input storage reuse rejected')
        if len(outputs)>=2 and outputs[0].dtype==outputs[1].dtype:
            elements=max(outputs[0].numel(),outputs[1].numel())
            storage=torch.empty(elements*2,dtype=outputs[0].dtype,device='npu:0')
            reject(lambda:verify.validate_ownership(inputs,[storage[:outputs[0].numel()].reshape(outputs[0].shape),storage[1:outputs[1].numel()+1].reshape(outputs[1].shape),*outputs[2:]]))
            result['verifier_checks'].append('offset overlapping storage rejected')
        # New deterministic seeds and valid nonzero storage offsets, separate from public cases.
        for index in range(12):
            c=copy.deepcopy(cases[index%len(cases)])
            # Bound audit-only cases; public cases already cover the maximum sizes.
            if sum(np.prod(d['shape']) for d in c['input_desc'])>2**20:c=copy.deepcopy(cases[2])
            c['case_name']+='__extra_seed_'+str(index);c['kind']='random';c['performance']=False
            arrays=data.generate(c);attrs={a['name']:a['value'] for a in c['attr_desc']}
            original=device_inputs(c,arrays,'npu:0')
            values=[]
            for t in original:
                padded=torch.empty(t.numel()+16,dtype=t.dtype,device=t.device)
                view=padded[1:1+t.numel()].view(t.shape);view.copy_(t);values.append(view)
            expected=golden.calculate(arrays,attrs)
            for name,fn in [('run',run),('run_direct',run_direct)]:
                output=fn(values,attrs);torch.npu.synchronize()
                validate_ownership(values,output);compare(c,arrays,output,expected)
                for t,a in zip(values,arrays):assert np.array_equal(cpu_array(t).view(np.uint16),a.view(np.uint16))
                result['randomized_cases'].append({'case_name':c['case_name'],'entry':name,'status':'passed'})
        result.update(status='passed',npu_executed=True)
    except Exception as exc:
        result.update(status='failed',error=str(exc));raise
    finally:
        path.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
        print(json.dumps(result,ensure_ascii=False),flush=True)

if __name__=='__main__':main()
