# -*- coding: utf-8 -*-
# Listing fixtures and reading help do not require the device libraries.
if __name__ == '__main__':
    import sys
    if '--help' in sys.argv or '-h' in sys.argv or '--list' in sys.argv:
        import argparse, json
        from pathlib import Path
        parser=argparse.ArgumentParser(description='Generate the task fixtures.')
        parser.add_argument('--output-dir',default='generated_data')
        parser.add_argument('--case',action='append',default=[])
        parser.add_argument('--list',action='store_true')
        parser.add_argument('--device',choices=['cpu','npu'],default='npu')
        args=parser.parse_args()
        contract=json.loads(Path(__file__).with_name('case.json').read_text())['input_cases']
        cases=contract if isinstance(contract,list) else contract['generation_cases']
        unknown=set(args.case)-{c['id'] for c in cases}
        if unknown:parser.error('Unknown fixture: '+', '.join(sorted(unknown)))
        for case in cases:
            if not args.case or case['id'] in args.case:print(case['id'])
        raise SystemExit(0)

"""Deterministic fixtures; no candidate code is used to create expected values."""
import random
TASK = 'packed'
def generate(case):
    if 'values' in case: return case['values'],case.get('attributes',{})
    rng=random.Random(case.get('seed',20261030)); l=case.get('layers',2); b=case.get('blocks',31)
    perm=list(range(l*b)); rng.shuffle(perm); m=[perm[i*b:(i+1)*b] for i in range(l)]
    if case.get('single'): m=m[0]
    if TASK=='slot':
        block=case.get('block_size',128); n=case['tokens']; x=[rng.randrange(b*block) for _ in range(n)]
        if n>1: x[0]=-1;x[-1]=-7
        return [x,m],{'block_size':block}
    if TASK in ('dense','packed'):
        r,c=case['requests'],case['columns']; x=[[rng.randrange(b) for _ in range(c)] for _ in range(r)]
        if r and c: x[0][0]=-1;x[-1][-1]=-9
        if TASK=='dense': return [x,m],{}
        counts=[rng.randrange(c+1) for _ in range(r)]; cu=[0]
        for n in counts: cu.append(cu[-1]+n)
        if r>1:
            counts[1]=0;cu=[0]
            for n in counts: cu.append(cu[-1]+n)
        return [x,m,cu],{}
    tile=case['tile_size']; r=case['requests']; mx=case['max_tiles']; offsets=[]; ip=[0]; lens=[]
    for j in range(r):
        length=max(0,(mx-1)*tile-j%(tile+1)); lens.append(length)
        data=list(range(j*mx*tile,j*mx*tile+length))
        if j%3==1: data=[v+1 for v in data]
        if j%3==2 and len(data)>2: data[len(data)//2]+=3
        offsets.extend(data); ip.append(len(offsets))
    return [offsets,ip,lens,[i*mx for i in range(r+1)]],{'tile_size':tile,'max_tiles_per_seq':mx}

TASK_NUMBER = 6


def main():
    import argparse, json
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Generate the task fixtures.')
    parser.add_argument('--output-dir', default='generated_data')
    parser.add_argument('--case', action='append', default=[])
    parser.add_argument('--list', action='store_true')
    parser.add_argument('--device', choices=['cpu','npu'], default='npu')
    args = parser.parse_args()
    contract = json.loads(Path(__file__).with_name('case.json').read_text())['input_cases']
    cases = contract if isinstance(contract,list) else contract['generation_cases']
    available = {c['id'] for c in cases}
    unknown = set(args.case)-available
    if unknown: parser.error('Unknown fixture: '+', '.join(sorted(unknown)))
    cases = [c for c in cases if not args.case or c['id'] in args.case]
    if args.list:
        for c in cases: print(c['id'])
        return
    out = Path(args.output_dir); out.mkdir(parents=True,exist_ok=True)
    manifest = []
    models = {}
    for case in cases:
        if TASK_NUMBER >= 11:
            import torch
            dtype = case['dtype']
            if dtype not in models:
                model, provenance = make_model(dtype,args.device)
                models[dtype] = model
                torch.save({'state_dict':model.state_dict(),'provenance':provenance},out/('model_'+dtype+'.pt'))
            data = generate(case, models[dtype], args.device)
        else:
            data = generate(case)
        name = case['id'] + ('.json' if TASK_NUMBER in (4,5,6,10) else '.pt')
        if TASK_NUMBER in (4,5,6,10):
            (out/name).write_text(json.dumps({'values':data[0],'attributes':data[1]},ensure_ascii=False)+'\n')
        else:
            import torch
            torch.save(data,out/name)
        manifest.append({'id':case['id'],'file':name})
    (out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2)+'\n')

if __name__ == '__main__':
    main()
