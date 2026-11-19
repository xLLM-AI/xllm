"""Verify the frozen task coverage contract against recorded results (no NPU imports)."""
from pathlib import Path
import hashlib,json

def _unique(rows,key,label):
    keys=[key(row) for row in rows]
    if len(keys)!=len(set(keys)):raise AssertionError(label+': duplicate records cannot count as independent cases')
    return set(keys)

def _require(actual,required,label):
    missing=set(required)-set(actual)
    if missing:raise AssertionError(label+': missing '+repr(sorted(missing)))

def case_digest(case):
    return hashlib.sha256(json.dumps(case,sort_keys=True,ensure_ascii=False,separators=(',',':')).encode()).hexdigest()

def validate_coverage(root):
    root=Path(root);task=root/'tests/task';sub=root/'task_submission'
    spec=json.loads((task/'coverage_requirements.json').read_text())
    read=lambda n:json.loads((sub/(n+'.json')).read_text())
    cases=json.loads((task/'case.json').read_text())
    by={c['case_name']:c for c in cases}
    _unique(cases,lambda c:c['case_name'],'case.json')
    _require(by,spec['public_cases'],'public input definitions')
    for name,digest in spec['public_cases'].items():
        if case_digest(by[name])!=digest:raise AssertionError('required input shape/dtype/attributes/data pattern changed: '+name)
    for kind in ['accuracy','public_accuracy']:
        r=read(kind)
        if r['status']!='passed' or not r['npu_executed']:raise AssertionError(kind+': no passing NPU run')
        keys=_unique(r['cases'],lambda c:c['case_name'],kind)
        _require(keys,by,kind)
        if any(c['status']!='passed' for c in r['cases']):raise AssertionError(kind+': failed record')
    for kind,field in [('performance','cases'),('engineering','memory')]:
        r=read(kind)
        if r['status']!='passed' or not r['npu_executed']:raise AssertionError(kind+': no passing NPU run')
        keys=_unique(r[field],lambda c:c['case_name'],kind)
        _require(keys,spec['performance_case_ids'],kind)
    a=read('audit');e=read('engineering')
    if a['status']!='passed' or not a['npu_executed']:raise AssertionError('audit: no passing NPU run')
    negative=_unique(a['negative_cases'],lambda r:(r['case'],r['entry']),'negative entry checks')
    required={(name,entry) for name in spec['negative_scenarios'] for entry in spec['negative_entries']}
    _require(negative,required,'negative entry checks')
    random=_unique(a['randomized_cases'],lambda r:(r['case_name'],r['entry']),'extra input checks')
    _require(random,{(name,entry) for name in spec['extra_input_case_ids'] for entry in spec['extra_entries']},'extra input checks')
    if any(r['status']!='passed' for r in a['negative_cases']+a['randomized_cases']):raise AssertionError('audit: failed detail')
    verified=_unique(a['verifier_checks'],lambda r:r,'verifier self checks')
    _require(verified,spec['verifier_checks'],'verifier self checks')
    streams=_unique(e['streams'],lambda r:r['case_name'],'stream scenarios')
    _require(streams,spec['stream_case_ids'],'stream scenarios')
    if any(r['status']!='passed' or r['calls']<spec['min_prepared_calls_per_stream'] for r in e['streams']):raise AssertionError('stream calls/status below contract')
    if e['negative_checks']<spec['min_engineering_metadata_checks']:raise AssertionError('engineering metadata checks below contract')
    if e['early_release_cross_stream']!='passed':raise AssertionError('early release not passed')
    if spec['sequential_cache_append'] and e.get('sequential_cache_append')!='passed':raise AssertionError('cache append sequence not passed')
    return {'public_input_cases':len(by),'extra_input_cases':len({r['case_name'] for r in a['randomized_cases']}),
            'valid_input_cases':len(by)+len({r['case_name'] for r in a['randomized_cases']}),
            'negative_input_scenarios':len({r['case'] for r in a['negative_cases']}),
            'negative_entry_checks':len(negative),'extra_entry_checks':len(random),
            'performance_cases':len(read('performance')['cases']),'memory_cases':len(e['memory']),
            'note':'Performance/memory reuse public input cases; repetitions and entry paths are not extra independent input cases.'}
