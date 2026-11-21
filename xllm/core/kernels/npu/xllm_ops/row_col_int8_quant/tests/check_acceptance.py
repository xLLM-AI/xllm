"""Fail closed against every public acceptance requirement and actual reports."""
from pathlib import Path
import hashlib,json,math,statistics
ROOT=Path(__file__).resolve().parents[1];sub=ROOT/'task_submission';task=ROOT/'tests/task'
def read(name):return json.loads((sub/(name+'.json')).read_text())
from validate_coverage import validate_coverage
coverage_counts=validate_coverage(ROOT)
accuracy=read('accuracy');public=read('public_accuracy');engineering=read('engineering');performance=read('performance');audit=read('audit')
cases=json.loads((task/'case.json').read_text());limits=json.loads((task/'performance_baseline.json').read_text())
for r in [accuracy,public,engineering,performance,audit]:assert r['status']=='passed' and r['npu_executed']
expected={c['case_name'] for c in cases}
assert len(expected)==len(cases)>0
for r in [accuracy,public]:
 assert {c['case_name'] for c in r['cases']}==expected and len(r['cases'])==len(cases)
 assert all(c['status']=='passed' for c in r['cases'])
assert all(c.get('native_outputs') and c.get('direct_native_outputs') for c in accuracy['cases'])
assert engineering['early_release_cross_stream']=='passed'
life=engineering['output_lifetime']
assert life['status']=='passed' and life['distinct_live_output_storages']
assert life['retained_calls']>=8 and life['released_calls']>=32 and life['same_stream_pressure_calls']>=32
assert engineering['negative_checks']>=5 and len(engineering['streams'])==2
assert all(s['status']=='passed' for s in engineering['streams'])
assert len(audit['negative_cases'])>=39 and all(c['status']=='passed' for c in audit['negative_cases'])
assert {c['entry'] for c in audit['negative_cases']}=={'prepare','run','run_direct'}
assert len(audit['randomized_cases'])==24 and all(c['status']=='passed' for c in audit['randomized_cases'])
assert len(audit['verifier_checks'])==3
if cases[0]['op_name']=='T5SplitQkvAppendCache':assert engineering['sequential_cache_append']=='passed'
measured={c['case_name']:c for c in performance['cases']}
performance_names={c['case_name'] for c in cases if c['performance']}
assert set(measured)==performance_names=={c['case_name'] for c in engineering['memory']}=={c['case_name'] for c in limits['cases']}
for name in ['warmup','iterations','rounds']:assert performance['protocol'][name]==limits[name]
assert [limits[x] for x in ['warmup','iterations','rounds']]==[50,200,5]
checks=[]
for c in limits['cases']:
 assert 0<c['max_custom_over_fastest_valid_baseline']<=1.0, 'performance regressions are forbidden'
 r=measured[c['case_name']]
 valid=next(x for x in accuracy['cases'] if x['case_name']==c['case_name'])['baselines']
 assert set(r['results'])-{'custom'}=={k for k,v in valid.items() if v=='passed'}
 for item in r['results'].values():
  assert len(item['wall_us_rounds'])==5 and all(v>0 and math.isfinite(v) for v in item['wall_us_rounds'])
  assert math.isclose(statistics.median(item['wall_us_rounds']),item['wall_us_median'])
 custom=r['results']['custom']['wall_us_median'];base=min(v['wall_us_median'] for k,v in r['results'].items() if k!='custom')
 ratio=custom/base
 assert ratio<=c['max_custom_over_fastest_valid_baseline'],(c['case_name'],ratio,c['max_custom_over_fastest_valid_baseline'])
 checks.append({'case_name':c['case_name'],'ratio':ratio,'limit':c['max_custom_over_fastest_valid_baseline'],'status':'passed'})
for m in engineering['memory']:
 assert m['allocator_before']==m['allocator_after']==m['allocator_after_20_calls']
 assert m['native_allocator_before_20_calls']==m['native_allocator_after_20_calls']
 assert m['native_allocator_peak_delta']<=m['explicit_output_and_workspace_bytes']+16384
 case=next(c for c in cases if c['case_name']==m['case_name'])
 if case['op_name']=='RowColInt8Quant':
  rows,cols=case['input_desc'][0]['shape'];bound=16777216+4*((rows+63)//64)*((cols+255)//256)*320
 else:bound=0
 assert m['native_workspace_bytes']<=bound
 assert m['workspace_bytes']==m['native_workspace_bytes']
for f in ['build.log','example.log','aclnn_example.log','2.2 精度自验证日志.log','3.2 性能自验证日志.log','4.2 内存自验证日志.log','audit.log','public_accuracy.log']:
 assert (sub/f).stat().st_size>0
assert 'ACLNN NPU example passed' in (sub/'aclnn_example.log').read_text()
result={'status':'passed','primary_entry':'run() -> generated ACLNN API','public_cases':len(cases),'performance_cases':len(checks),'memory_cases':len(engineering['memory']),'negative_checks':len(audit['negative_cases']),'randomized_entry_checks':len(audit['randomized_cases']),'verifier_checks':len(audit['verifier_checks']),'performance_checks':checks}
result['coverage_counts']=coverage_counts
(sub/'acceptance.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(result,ensure_ascii=False))
