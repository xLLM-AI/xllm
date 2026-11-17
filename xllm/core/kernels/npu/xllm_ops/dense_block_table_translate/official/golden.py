# -*- coding: utf-8 -*-
"""CPU 标杆与 NPU 基线接口。"""
import ast
import hashlib
import importlib.abc
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import types
TASK_NUMBER = 4
API_EXPORTS = {'calculate': ['public_api', 'calculate']}
REFERENCE_SOURCES = {'public_api.py': '"""Independent Python scalar oracle; input/output are lists, never the candidate."""\n'
                  "TASK = 'dense'\n"
                  'def calculate(values, attrs, single=False):\n'
                  "    if TASK == 'flags':\n"
                  "        offsets,indptr,lens,tileptr=values; tile=attrs['tile_size']; "
                  "mx=attrs['max_tiles_per_seq']\n"
                  '        result=[0]*(len(lens)*mx)\n'
                  '        for r,length in enumerate(lens):\n'
                  '            for t in range(mx):\n'
                  '                begin=t*tile\n'
                  '                if begin+tile>length: continue\n'
                  '                data=offsets[indptr[r]+begin:indptr[r]+begin+tile]\n'
                  '                result[tileptr[r]+t]=int(data[0]%tile==0 and all(data[i]-data[i-1]==1 for '
                  'i in range(1,tile)))\n'
                  '        return result\n'
                  '    x,m=values[:2]; rows=[m] if single else m\n'
                  '    outputs=[]\n'
                  '    for row in rows:\n'
                  "        if TASK=='dense': result=[[row[v] if v>=0 else -1 for v in req] for req in x]\n"
                  "        elif TASK=='slot':\n"
                  "            b=attrs['block_size']; result=[row[v//b]*b+v%b if v>=0 else -1 for v in x]\n"
                  '        else:\n'
                  '            cu=values[2]; result=[]\n'
                  '            for r,req in enumerate(x): result.extend(row[v] if v>=0 else -1 for v in '
                  'req[:cu[r+1]-cu[r]])\n'
                  '        outputs.append(result)\n'
                  '    return outputs[0] if single else outputs\n',
 'baseline.py': '"""Ordinary PyTorch NPU vector operations, frozen comparison baseline."""\n'
                'import torch\n'
                "TASK = 'dense'\n"
                'def prepare(inputs, attrs, info):\n'
                "    if TASK=='packed':\n"
                "        x=inputs[0]; cu=info['values'][2]\n"
                '        # The original caller supplies counts. Build layout once, outside both hot paths.\n'
                '        ids=[r*x.shape[1]+c for r in range(x.shape[0]) for c in range(cu[r+1]-cu[r])]\n'
                '        ids=torch.tensor(ids,dtype=torch.int64,device=x.device)\n'
                "    elif TASK=='flags':\n"
                "        tile=attrs['tile_size']; mx=attrs['max_tiles_per_seq']; rows=info['rows']\n"
                '        tile_ids=torch.arange(mx,device=inputs[0].device).reshape(1,mx,1)\n'
                '        lane=torch.arange(tile,device=inputs[0].device).reshape(1,1,tile)\n'
                '        pos=tile_ids*tile+lane\n'
                '    def run():\n'
                "        if TASK in ('dense','slot','packed'):\n"
                "            x,m=inputs[:2]; m=m.reshape(info['layers'],info['blocks'])\n"
                "            if TASK=='packed': x=x.reshape(-1).index_select(0,ids)\n"
                '            flat=x.reshape(-1).to(torch.int64)\n'
                "            if TASK=='slot':\n"
                "                b=attrs['block_size']; safe=flat.clamp_min(0); logical=safe//b\n"
                '                result=m.index_select(1,logical)*b+(safe%b).reshape(1,-1)\n'
                '            else: result=m.index_select(1,flat.clamp_min(0))\n'
                '            result=torch.where(flat.reshape(1,-1)<0,-1,result).to(torch.int32)\n'
                "            shape=tuple(x.shape) if info['single'] else (info['layers'],*x.shape)\n"
                '            return result.reshape(shape)\n'
                '        offsets,ip,lens,tp=inputs\n'
                '        if rows==0: return torch.empty((0,),dtype=torch.uint8,device=offsets.device)\n'
                '        if offsets.numel()==0: return '
                'torch.zeros((rows*mx,),dtype=torch.uint8,device=offsets.device)\n'
                '        address=ip[:-1].to(torch.int64).reshape(rows,1,1)+pos\n'
                '        '
                'data=offsets.index_select(0,address.clamp_max(offsets.numel()-1).reshape(-1)).reshape(rows,mx,tile)\n'
                '        first=data[:,:,:1]; consecutive=(data==first+lane).all(dim=-1)\n'
                '        full=(tile_ids.reshape(1,mx)+1)*tile<=lens.reshape(rows,1)\n'
                '        return (consecutive & (first.squeeze(-1)%tile==0) & '
                'full).to(torch.uint8).reshape(-1)\n'
                '    return run\n',
 'contract.py': '"""Metadata/value validation occurs in prepare; kernels never calculate on CPU."""\n'
                'import torch\n'
                "TASK = 'dense'\n"
                'I32 = 2**31-1\n'
                '\n'
                'def validate(inputs, attributes):\n'
                '    if not isinstance(inputs, (tuple,list)) or not isinstance(attributes,dict):\n'
                '        raise TypeError("inputs must be sequence; attributes must be dict")\n'
                "    expected = 4 if TASK == 'flags' else 3 if TASK == 'packed' else 2\n"
                '    if len(inputs) != expected: raise ValueError("input count")\n'
                "    allowed = {'block_size'} if TASK == 'slot' else {'tile_size','max_tiles_per_seq'} if "
                "TASK == 'flags' else set()\n"
                '    if set(attributes) != allowed: raise ValueError("attributes missing or unknown")\n'
                '    for value in attributes.values():\n'
                '        if type(value) is not int or value <= 0: raise ValueError("attributes must be '
                'positive integers")\n'
                '    for x in inputs:\n'
                '        if not isinstance(x,torch.Tensor) or x.dtype != torch.int32: raise TypeError("all '
                'inputs int32")\n'
                '        if x.device.type != \'npu\': raise ValueError("NPU inputs required")\n'
                '        if not x.is_contiguous(): raise ValueError("contiguous inputs required; storage '
                'offset is supported")\n'
                '        if x.device != inputs[0].device: raise ValueError("same device required")\n'
                '        if x.numel() > I32: raise ValueError("input indexing exceeds int32")\n'
                '    # These readbacks are a one-time preflight, excluded from prepared hot-path timing.\n'
                '    values = [x.detach().cpu().tolist() for x in inputs]\n'
                "    if TASK in ('dense','slot','packed'):\n"
                '        x,m = inputs[:2]; xv,mv = values[:2]\n'
                '        if m.ndim not in (1,2) or m.shape[-1] < 1: raise ValueError("M shape [B] or [L,B], '
                'B>=1")\n'
                '        if m.ndim == 2 and m.shape[0] < 1: raise ValueError("L>=1")\n'
                '        rows = [mv] if m.ndim == 1 else mv\n'
                '        if any(v<0 for row in rows for v in row): raise ValueError("M must be '
                'nonnegative")\n'
                "        if TASK == 'slot':\n"
                '            if x.ndim != 1: raise ValueError("slots shape [N]")\n'
                "            b=attributes['block_size']\n"
                '            if b>I32: raise ValueError("block_size int32 overflow")\n'
                '            if any(v>=0 and v//b>=m.shape[-1] for v in xv): raise ValueError("logical slot '
                'out of range")\n'
                '            if any(v*b+b-1>I32 for row in rows for v in row): raise ValueError("translated '
                'slot int32 overflow")\n'
                '        else:\n'
                '            if x.ndim != 2: raise ValueError("logical table shape [R,C]")\n'
                "            if TASK == 'packed':\n"
                '                cu=inputs[2]; cv=values[2]\n'
                '                if cu.ndim!=1 or len(cv)!=x.shape[0]+1 or cv[0]!=0: raise '
                'ValueError("invalid cu shape/start")\n'
                '                if any(cv[i]>cv[i+1] or cv[i+1]-cv[i]>x.shape[1] for i in '
                'range(x.shape[0])): raise ValueError("invalid cu counts")\n'
                '                active=[row[:cv[r+1]-cv[r]] for r,row in enumerate(xv)]\n'
                '            else: active=xv\n'
                '            if any(v>=m.shape[-1] for row in active for v in row): raise '
                'ValueError("logical block out of range")\n'
                "        output_count = len(rows) * (values[2][-1] if TASK == 'packed' else x.numel())\n"
                '        if output_count > I32: raise ValueError("output indexing exceeds int32")\n'
                "        return {'values':values,'layers':len(rows),'blocks':m.shape[-1], "
                "'single':m.ndim==1}\n"
                '    offsets,indptr,lens,tileptr=inputs; ov,iv,lv,tv=values\n'
                '    if any(x.ndim!=1 for x in inputs): raise ValueError("flags inputs must be vectors")\n'
                '    if len(iv)!=len(lv)+1 or len(tv)!=len(lv)+1 or iv[0]!=0: raise ValueError("indptr '
                'lengths/start")\n'
                "    tile=attributes['tile_size']; mx=attributes['max_tiles_per_seq']\n"
                '    if tile>4096: raise ValueError("tile_size supported range 1..4096")\n'
                '    if len(lv)*mx > I32: raise ValueError("output indexing exceeds int32")\n'
                '    if iv[-1]!=len(ov) or any(iv[i]>iv[i+1] or lv[i]<0 or lv[i]>iv[i+1]-iv[i] for i in '
                'range(len(lv))): raise ValueError("offset segments/lens")\n'
                '    if tv != [i*mx for i in range(len(lv)+1)]: raise ValueError("dense tile output '
                'required")\n'
                '    if any((n+tile-1)//tile>mx for n in lv) or any(v<0 for v in ov): raise '
                'ValueError("capacity or offsets")\n'
                "    return {'values':values,'rows':len(lv)}\n"}
_SESSION = None
def _write_sources(root, sources):
    for name, source in sources.items():
        relative = Path(name)
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('invalid embedded source path')
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(source)

class _ImportNames(ast.NodeTransformer):
    def __init__(self, prefix, names): self.prefix, self.names = prefix, names

    def visit_Import(self, node):
        result = []
        for alias in node.names:
            if alias.name.split('.')[0] in self.names:
                if '.' in alias.name:
                    result.append(ast.Import(names=[ast.alias(self.prefix+'.'+alias.name, alias.asname or alias.name.split('.')[0])]))
                else:
                    result.append(ast.ImportFrom(module=self.prefix, names=[ast.alias(alias.name, alias.asname)], level=0))
            else: result.append(ast.Import(names=[alias]))
        return [ast.copy_location(x, node) for x in result]

    def visit_ImportFrom(self, node):
        if not node.level and node.module and node.module.split('.')[0] in self.names:
            node.module = self.prefix + '.' + node.module
        return node

class _ReferenceLoader(importlib.abc.MetaPathFinder, importlib.abc.Loader):
    """Isolate absolute imports; function bodies and native source stay unchanged."""
    def __init__(self, root):
        self.root = root
        self.prefix = '_task_reference_' + hashlib.sha256(str(root).encode()).hexdigest()[:16]
        self.names = {p.stem for p in root.glob('*.py')}
        package = types.ModuleType(self.prefix)
        package.__path__ = [str(root)]
        sys.modules[self.prefix] = package
        sys.meta_path.insert(0, self)

    def find_spec(self, fullname, path=None, target=None):
        if fullname.startswith(self.prefix+'.'):
            name = fullname[len(self.prefix)+1:]
            source = self.root / (name.replace('.', '/') + '.py')
            if source.is_file(): return importlib.util.spec_from_loader(fullname, self, origin=str(source))
        return None

    def create_module(self, spec): return None

    def exec_module(self, module):
        path = Path(module.__spec__.origin)
        module.__file__ = str(path)
        tree = ast.parse(path.read_text(), str(path))
        tree = ast.fix_missing_locations(_ImportNames(self.prefix, self.names).visit(tree))
        exec(compile(tree, str(path), 'exec'), module.__dict__)

    def get(self, name): return importlib.import_module(self.prefix+'.'+name)

def reference(name):
    """Load a frozen oracle/baseline by its documented module name."""
    global _SESSION
    if _SESSION is None:
        temp = tempfile.TemporaryDirectory(prefix='task-reference-')
        root = Path(temp.name)
        _write_sources(root, REFERENCE_SOURCES)
        _SESSION = (temp, _ReferenceLoader(root))
    return _SESSION[1].get(name)

def __getattr__(name):
    if name in API_EXPORTS:
        module, attribute = API_EXPORTS[name]
        return getattr(reference(module), attribute)
    raise AttributeError(name)
FUNCTIONAL_SOURCES = {'functional_checks.py': '"""功能与工程检查；性能计时由提交者自行实现。"""\n'
                         'import argparse, datetime, gc, hashlib, importlib.metadata, json, platform, '
                         'statistics, time\n'
                         'from pathlib import Path\n'
                         'import torch, torch_npu, triton\n'
                         'from generate_data import generate, TASK\n'
                         'from golden import calculate\n'
                         'from solution import prepare\n'
                         'from baseline import prepare as prepare_baseline\n'
                         'ROOT = Path(__file__).resolve().parent\n'
                         '\n'
                         'def make_inputs(case, values):\n'
                         '    out = []\n'
                         '    for (i, value) in enumerate(values):\n'
                         '        t = torch.tensor(value, dtype=torch.int32)\n'
                         "        if i == 0 and TASK in ('dense', 'packed') and ('requests' in case):\n"
                         "            t = t.reshape(case['requests'], case['columns'])\n"
                         "        if case.get('offset'):\n"
                         '            storage = torch.empty((t.numel() + 17,), dtype=torch.int32, '
                         "device='npu:0')\n"
                         '            v = storage[1:t.numel() + 1].reshape(t.shape)\n'
                         '            v.copy_(t)\n'
                         '            t = v\n'
                         '        else:\n'
                         "            t = t.to('npu:0')\n"
                         '        out.append(t)\n'
                         '    return out\n'
                         '\n'
                         'def snapshot(inputs):\n'
                         '    return [x.cpu().clone() for x in inputs]\n'
                         '\n'
                         'def sync():\n'
                         '    torch.npu.synchronize()\n'
                         '\n'
                         'def fp():\n'
                         '    return {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in '
                         "sorted(ROOT.iterdir()) if p.suffix in ('.py', '.json', '.cpp') and p.name != "
                         "'result.json'}\n"
                         '\n'
                         'def environment():\n'
                         '    versions = {}\n'
                         "    for name in ['torch', 'torch-npu', 'triton-ascend']:\n"
                         '        try:\n'
                         '            versions[name] = importlib.metadata.version(name)\n'
                         '        except importlib.metadata.PackageNotFoundError:\n'
                         '            versions[name] = None\n'
                         "    return {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(), "
                         "'hostname': platform.node(), 'python': platform.python_version(), 'device': "
                         "torch.npu.get_device_name(0), 'versions': versions, 'triton_target': "
                         "str(triton.runtime.driver.active.get_current_target()), 'source_sha256': fp()}\n"
                         '\n'
                         'def verify_case(case):\n'
                         '    (values, attrs) = generate(case)\n'
                         '    inputs = make_inputs(case, values)\n'
                         '    before = snapshot(inputs)\n'
                         '    prepared = prepare(inputs, attrs)\n'
                         '    baseline = prepare_baseline(inputs, attrs, prepared.info)\n'
                         "    expected_list = calculate(values, attrs, prepared.info.get('single', False))\n"
                         "    if 'expected' in case:\n"
                         "        assert expected_list == case['expected'], 'independent fixture oracle "
                         "error'\n"
                         "    expected = torch.tensor(expected_list, dtype=torch.uint8 if TASK == 'flags' "
                         'else torch.int32).reshape(prepared.shape)\n'
                         '    actual = prepared.run()\n'
                         '    reference = baseline()\n'
                         '    sync()\n'
                         '    actual_cpu = actual.cpu()\n'
                         '    reference_cpu = reference.cpu()\n'
                         "    checks = {'candidate_mismatches': int((actual_cpu != expected).sum()), "
                         "'baseline_mismatches': int((reference_cpu != expected).sum()), 'shape': "
                         "list(actual.shape) == list(expected.shape), 'dtype': actual.dtype == "
                         "expected.dtype, 'npu': actual.device.type == 'npu', 'contiguous': "
                         "actual.is_contiguous(), 'inputs_unchanged': all((torch.equal(x.cpu(), b) for (x, "
                         "b) in zip(inputs, before))), 'independent_storage': all((actual.numel() == 0 or "
                         'x.numel() == 0 or actual.untyped_storage().data_ptr() != '
                         'x.untyped_storage().data_ptr() for x in inputs))}\n'
                         "    assert checks['candidate_mismatches'] == 0 and checks['baseline_mismatches'] "
                         "== 0 and all((v for (k, v) in checks.items() if not k.endswith('mismatches'))), "
                         'checks\n'
                         "    return ({'id': case['id'], 'status': 'PASS', 'checks': checks, 'shape': "
                         "list(actual.shape), 'input_shapes': [list(x.shape) for x in inputs], 'attributes': "
                         'attrs}, prepared, baseline)\n'
                         '\n'
                         'def negative_tests(case):\n'
                         '    (values, attrs) = generate(case)\n'
                         '    inputs = make_inputs(case, values)\n'
                         '    results = []\n'
                         '\n'
                         '    def check(name, x, a):\n'
                         '        try:\n'
                         '            prepare(x, a)\n'
                         '        except (TypeError, ValueError) as e:\n'
                         "            results.append({'id': name, 'status': 'PASS', 'exception': "
                         'type(e).__name__})\n'
                         '            return\n'
                         "        raise AssertionError('invalid input accepted: ' + name)\n"
                         "    check('input_count', inputs[:-1], attrs)\n"
                         "    check('unknown_attribute', inputs, dict(attrs, unknown=1))\n"
                         "    check('cpu_input', [inputs[0].cpu(), *inputs[1:]], attrs)\n"
                         "    check('dtype', [inputs[0].to(torch.int64), *inputs[1:]], attrs)\n"
                         "    noncontig = torch.empty((inputs[0].numel() * 2,), device='npu:0', "
                         'dtype=torch.int32)[::2].reshape(inputs[0].shape)\n'
                         '    noncontig.copy_(inputs[0])\n'
                         "    check('noncontiguous', [noncontig, *inputs[1:]], attrs)\n"
                         "    if TASK in ('dense', 'slot', 'packed'):\n"
                         '        badm = inputs[1].clone()\n'
                         '        badm.reshape(-1)[0] = -1\n'
                         "        check('negative_mapping', [inputs[0], badm, *inputs[2:]], attrs)\n"
                         '        badx = inputs[0].clone()\n'
                         '        badx.reshape(-1)[0] = 2147483647\n'
                         "        check('logical_address_out_of_range', [badx, *inputs[1:]], attrs)\n"
                         "        if TASK == 'slot':\n"
                         "            check('zero_block_size', inputs, {'block_size': 0})\n"
                         "            check('bool_block_size', inputs, {'block_size': True})\n"
                         '            badm = inputs[1].clone()\n'
                         '            badm.reshape(-1)[0] = 2147483647\n'
                         "            check('output_int32_overflow', [inputs[0], badm], attrs)\n"
                         "        if TASK == 'packed':\n"
                         '            badcu = inputs[2].clone()\n'
                         '            badcu[1] = -1\n'
                         "            check('nonmonotonic_cu', [*inputs[:2], badcu], attrs)\n"
                         '    else:\n'
                         "        check('zero_tile', inputs, dict(attrs, tile_size=0))\n"
                         "        check('oversize_tile', inputs, dict(attrs, tile_size=4097))\n"
                         '        badtp = inputs[3].clone()\n'
                         '        badtp[1] = 0\n'
                         "        check('non_dense_output_offsets', [*inputs[:3], badtp], attrs)\n"
                         '        badlen = inputs[2].clone()\n'
                         '        badlen[0] = 2147483647\n'
                         "        check('length_exceeds_segment', [*inputs[:2], badlen, inputs[3]], attrs)\n"
                         '    return results\n'
                         '\n'
                         'def engineering(prepared, baseline):\n'
                         '    streams = [torch.npu.Stream(), torch.npu.Stream()]\n'
                         '    outputs = []\n'
                         '    for s in streams:\n'
                         '        s.wait_stream(torch.npu.current_stream())\n'
                         '    for s in streams:\n'
                         '        with torch.npu.stream(s):\n'
                         '            outputs.append([prepared.run() for _ in range(3)])\n'
                         '    sync()\n'
                         '    gold = outputs[0][0].cpu()\n'
                         '    assert all((torch.equal(x.cpu(), gold) for row in outputs for x in row))\n'
                         '    ptrs = [x.untyped_storage().data_ptr() for row in outputs for x in row]\n'
                         '    assert len(ptrs) == len(set(ptrs))\n'
                         '    del outputs\n'
                         '    gc.collect()\n'
                         '    sync()\n'
                         '    for _ in range(3):\n'
                         '        temp = prepared.run()\n'
                         '    del temp\n'
                         '    sync()\n'
                         '    gc.collect()\n'
                         '    before = torch.npu.memory_allocated()\n'
                         '    torch.npu.reset_peak_memory_stats()\n'
                         '    y = prepared.run()\n'
                         '    sync()\n'
                         '    peak = torch.npu.max_memory_allocated() - before\n'
                         '    output_bytes = y.numel() * y.element_size()\n'
                         '    del y\n'
                         '    gc.collect()\n'
                         '    sync()\n'
                         '    for _ in range(20):\n'
                         '        y = prepared.run()\n'
                         '    del y\n'
                         '    gc.collect()\n'
                         '    sync()\n'
                         '    after = torch.npu.memory_allocated()\n'
                         "    assert after == before, ('active tensor leak', before, after)\n"
                         '    for _ in range(3):\n'
                         '        temp = baseline()\n'
                         '    del temp\n'
                         '    gc.collect()\n'
                         '    sync()\n'
                         '    base_before = torch.npu.memory_allocated()\n'
                         '    torch.npu.reset_peak_memory_stats()\n'
                         '    base_y = baseline()\n'
                         '    sync()\n'
                         '    base_peak = torch.npu.max_memory_allocated() - base_before\n'
                         '    base_output_bytes = base_y.numel() * base_y.element_size()\n'
                         '    del base_y\n'
                         '    gc.collect()\n'
                         '    sync()\n'
                         '    base_after = torch.npu.memory_allocated()\n'
                         '    assert base_after == base_before\n'
                         "    return {'two_streams': 'PASS', 'retained_outputs_independent': 'PASS', "
                         "'repeat20_released': 'PASS', 'active_before_bytes': before, 'active_after_bytes': "
                         "after, 'peak_increment_bytes': peak, 'output_bytes': output_bytes, "
                         "'baseline_peak_increment_bytes': base_peak, 'baseline_output_bytes': "
                         "base_output_bytes, 'baseline_active_before_bytes': base_before, "
                         "'baseline_active_after_bytes': base_after, 'candidate_minus_baseline_peak_bytes': "
                         "peak - base_peak, 'peak_definition': 'Warmed PyTorch active allocation for one "
                         'complete prepared call on identical inputs; excludes persistent prepared '
                         "input/layout buffers in both; not total HBM or all runtime scratch'}\n"
                         '\n'
                         'def dynamic_values(prepared):\n'
                         '    """Exercise valid value mutations without re-prepare; no timing claim '
                         'here."""\n'
                         '    inputs = prepared.inputs\n'
                         '    original = snapshot(inputs)\n'
                         '    records = []\n'
                         '\n'
                         '    def check(name):\n'
                         '        values = [x.cpu().tolist() for x in inputs]\n'
                         '        expected = torch.tensor(calculate(values, prepared.attributes, '
                         "prepared.info['single']), dtype=torch.int32).reshape(prepared.shape)\n"
                         '        output = prepared.run()\n'
                         '        sync()\n'
                         '        mismatches = int((output.cpu() != expected).sum())\n'
                         '        assert mismatches == 0, (name, mismatches)\n'
                         "        records.append({'id': name, 'status': 'PASS', 'candidate_mismatches': "
                         "mismatches, 'compared_with': 'independent CPU oracle; outside performance "
                         "timing'})\n"
                         '    try:\n'
                         "        for (index, name) in ((0, 'logical_values_changed'), (1, "
                         "'mapping_values_changed')):\n"
                         '            values = original[index].tolist()\n'
                         '            changed = [row[::-1] for row in values] if original[index].ndim == 2 '
                         'else values[::-1]\n'
                         '            inputs[index].copy_(torch.tensor(changed, dtype=torch.int32))\n'
                         '            check(name)\n'
                         '            inputs[index].copy_(original[index])\n'
                         '            sync()\n'
                         "        if prepared.n * prepared.info['layers'] >= 32768:\n"
                         "            page = prepared.attributes.get('block_size', 1)\n"
                         '            limit = (2147483647 - (page - 1)) // page\n'
                         '            replacement = torch.full(original[1].shape, limit, dtype=torch.int32)\n'
                         '            inputs[1].copy_(replacement)\n'
                         "            if TASK == 'slot':\n"
                         '                addresses = original[0].clone()\n'
                         '                addresses[0] = page - 1\n'
                         '                inputs[0].copy_(addresses)\n'
                         "            check('large_path_int32_output_boundary')\n"
                         '            inputs[0].copy_(original[0])\n'
                         '            inputs[1].copy_(original[1])\n'
                         '            sync()\n'
                         "        if TASK == 'packed':\n"
                         '            cu = original[2].tolist()\n'
                         '            counts = [b - a for (a, b) in zip(cu, cu[1:])]\n'
                         '            width = inputs[0].shape[1]\n'
                         '            pair = next(((i, j) for (i, a) in enumerate(counts) for (j, b) in '
                         'enumerate(counts) if i != j and a > 0 and (b < width)), None)\n'
                         '            if pair is not None:\n'
                         '                counts[pair[0]] -= 1\n'
                         '                counts[pair[1]] += 1\n'
                         '                updated = [0]\n'
                         '                for count in counts:\n'
                         '                    updated.append(updated[-1] + count)\n'
                         '                inputs[2].copy_(torch.tensor(updated, dtype=torch.int32))\n'
                         "                check('cu_counts_changed_same_total')\n"
                         '    finally:\n'
                         '        for (x, before) in zip(inputs, original):\n'
                         '            x.copy_(before)\n'
                         '        sync()\n'
                         '    return records\n'
                         '\n'
                         'def main():\n'
                         "    ap=argparse.ArgumentParser(description='功能及工程检查，不执行性能验收')\n"
                         "    ap.add_argument('--output',required=True)\n"
                         '    args=ap.parse_args()\n'
                         '    path=Path(args.output)\n'
                         '    path.parent.mkdir(parents=True,exist_ok=True)\n'
                         "    report={'task':TASK,'scope':'functional','status':'RUNNING','cases':[], "
                         "'negative':[], 'dynamic_updates':{}, 'engineering_by_case':{}, "
                         "'storage_rebindings':{}}\n"
                         '    try:\n'
                         '        torch.npu.set_device(0)\n'
                         "        assert triton.runtime.driver.active.get_current_target().backend=='npu'\n"
                         "        report['environment']=environment()\n"
                         "        cases=json.loads((ROOT/'case.json').read_text())\n"
                         '        for case in cases:\n'
                         '            record,prepared,baseline=verify_case(case)\n'
                         "            report['cases'].append(record)\n"
                         '            print(json.dumps(record),flush=True)\n'
                         "            if case.get('performance'):\n"
                         "                report['dynamic_updates'][case['id']]=dynamic_values(prepared)\n"
                         '                '
                         "report['engineering_by_case'][case['id']]=engineering(prepared,baseline)\n"
                         "        report['negative']=negative_tests(cases[0])\n"
                         "        report['status']='PASS'\n"
                         '    except Exception as exc:\n'
                         "        report['status']='FAIL';report['error']=repr(exc)\n"
                         '        raise\n'
                         '    finally:\n'
                         "        path.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\\n')\n"
                         "if __name__=='__main__':main()\n",
 'golden.py': '"""Independent Python scalar oracle; input/output are lists, never the candidate."""\n'
              "TASK = 'dense'\n"
              'def calculate(values, attrs, single=False):\n'
              "    if TASK == 'flags':\n"
              "        offsets,indptr,lens,tileptr=values; tile=attrs['tile_size']; "
              "mx=attrs['max_tiles_per_seq']\n"
              '        result=[0]*(len(lens)*mx)\n'
              '        for r,length in enumerate(lens):\n'
              '            for t in range(mx):\n'
              '                begin=t*tile\n'
              '                if begin+tile>length: continue\n'
              '                data=offsets[indptr[r]+begin:indptr[r]+begin+tile]\n'
              '                result[tileptr[r]+t]=int(data[0]%tile==0 and all(data[i]-data[i-1]==1 for i '
              'in range(1,tile)))\n'
              '        return result\n'
              '    x,m=values[:2]; rows=[m] if single else m\n'
              '    outputs=[]\n'
              '    for row in rows:\n'
              "        if TASK=='dense': result=[[row[v] if v>=0 else -1 for v in req] for req in x]\n"
              "        elif TASK=='slot':\n"
              "            b=attrs['block_size']; result=[row[v//b]*b+v%b if v>=0 else -1 for v in x]\n"
              '        else:\n'
              '            cu=values[2]; result=[]\n'
              '            for r,req in enumerate(x): result.extend(row[v] if v>=0 else -1 for v in '
              'req[:cu[r+1]-cu[r]])\n'
              '        outputs.append(result)\n'
              '    return outputs[0] if single else outputs\n',
 'baseline.py': '"""Ordinary PyTorch NPU vector operations, frozen comparison baseline."""\n'
                'import torch\n'
                "TASK = 'dense'\n"
                'def prepare(inputs, attrs, info):\n'
                "    if TASK=='packed':\n"
                "        x=inputs[0]; cu=info['values'][2]\n"
                '        # The original caller supplies counts. Build layout once, outside both hot paths.\n'
                '        ids=[r*x.shape[1]+c for r in range(x.shape[0]) for c in range(cu[r+1]-cu[r])]\n'
                '        ids=torch.tensor(ids,dtype=torch.int64,device=x.device)\n'
                "    elif TASK=='flags':\n"
                "        tile=attrs['tile_size']; mx=attrs['max_tiles_per_seq']; rows=info['rows']\n"
                '        tile_ids=torch.arange(mx,device=inputs[0].device).reshape(1,mx,1)\n'
                '        lane=torch.arange(tile,device=inputs[0].device).reshape(1,1,tile)\n'
                '        pos=tile_ids*tile+lane\n'
                '    def run():\n'
                "        if TASK in ('dense','slot','packed'):\n"
                "            x,m=inputs[:2]; m=m.reshape(info['layers'],info['blocks'])\n"
                "            if TASK=='packed': x=x.reshape(-1).index_select(0,ids)\n"
                '            flat=x.reshape(-1).to(torch.int64)\n'
                "            if TASK=='slot':\n"
                "                b=attrs['block_size']; safe=flat.clamp_min(0); logical=safe//b\n"
                '                result=m.index_select(1,logical)*b+(safe%b).reshape(1,-1)\n'
                '            else: result=m.index_select(1,flat.clamp_min(0))\n'
                '            result=torch.where(flat.reshape(1,-1)<0,-1,result).to(torch.int32)\n'
                "            shape=tuple(x.shape) if info['single'] else (info['layers'],*x.shape)\n"
                '            return result.reshape(shape)\n'
                '        offsets,ip,lens,tp=inputs\n'
                '        if rows==0: return torch.empty((0,),dtype=torch.uint8,device=offsets.device)\n'
                '        if offsets.numel()==0: return '
                'torch.zeros((rows*mx,),dtype=torch.uint8,device=offsets.device)\n'
                '        address=ip[:-1].to(torch.int64).reshape(rows,1,1)+pos\n'
                '        '
                'data=offsets.index_select(0,address.clamp_max(offsets.numel()-1).reshape(-1)).reshape(rows,mx,tile)\n'
                '        first=data[:,:,:1]; consecutive=(data==first+lane).all(dim=-1)\n'
                '        full=(tile_ids.reshape(1,mx)+1)*tile<=lens.reshape(rows,1)\n'
                '        return (consecutive & (first.squeeze(-1)%tile==0) & '
                'full).to(torch.uint8).reshape(-1)\n'
                '    return run\n',
 'contract.py': '"""Metadata/value validation occurs in prepare; kernels never calculate on CPU."""\n'
                'import torch\n'
                "TASK = 'dense'\n"
                'I32 = 2**31-1\n'
                '\n'
                'def validate(inputs, attributes):\n'
                '    if not isinstance(inputs, (tuple,list)) or not isinstance(attributes,dict):\n'
                '        raise TypeError("inputs must be sequence; attributes must be dict")\n'
                "    expected = 4 if TASK == 'flags' else 3 if TASK == 'packed' else 2\n"
                '    if len(inputs) != expected: raise ValueError("input count")\n'
                "    allowed = {'block_size'} if TASK == 'slot' else {'tile_size','max_tiles_per_seq'} if "
                "TASK == 'flags' else set()\n"
                '    if set(attributes) != allowed: raise ValueError("attributes missing or unknown")\n'
                '    for value in attributes.values():\n'
                '        if type(value) is not int or value <= 0: raise ValueError("attributes must be '
                'positive integers")\n'
                '    for x in inputs:\n'
                '        if not isinstance(x,torch.Tensor) or x.dtype != torch.int32: raise TypeError("all '
                'inputs int32")\n'
                '        if x.device.type != \'npu\': raise ValueError("NPU inputs required")\n'
                '        if not x.is_contiguous(): raise ValueError("contiguous inputs required; storage '
                'offset is supported")\n'
                '        if x.device != inputs[0].device: raise ValueError("same device required")\n'
                '        if x.numel() > I32: raise ValueError("input indexing exceeds int32")\n'
                '    # These readbacks are a one-time preflight, excluded from prepared hot-path timing.\n'
                '    values = [x.detach().cpu().tolist() for x in inputs]\n'
                "    if TASK in ('dense','slot','packed'):\n"
                '        x,m = inputs[:2]; xv,mv = values[:2]\n'
                '        if m.ndim not in (1,2) or m.shape[-1] < 1: raise ValueError("M shape [B] or [L,B], '
                'B>=1")\n'
                '        if m.ndim == 2 and m.shape[0] < 1: raise ValueError("L>=1")\n'
                '        rows = [mv] if m.ndim == 1 else mv\n'
                '        if any(v<0 for row in rows for v in row): raise ValueError("M must be '
                'nonnegative")\n'
                "        if TASK == 'slot':\n"
                '            if x.ndim != 1: raise ValueError("slots shape [N]")\n'
                "            b=attributes['block_size']\n"
                '            if b>I32: raise ValueError("block_size int32 overflow")\n'
                '            if any(v>=0 and v//b>=m.shape[-1] for v in xv): raise ValueError("logical slot '
                'out of range")\n'
                '            if any(v*b+b-1>I32 for row in rows for v in row): raise ValueError("translated '
                'slot int32 overflow")\n'
                '        else:\n'
                '            if x.ndim != 2: raise ValueError("logical table shape [R,C]")\n'
                "            if TASK == 'packed':\n"
                '                cu=inputs[2]; cv=values[2]\n'
                '                if cu.ndim!=1 or len(cv)!=x.shape[0]+1 or cv[0]!=0: raise '
                'ValueError("invalid cu shape/start")\n'
                '                if any(cv[i]>cv[i+1] or cv[i+1]-cv[i]>x.shape[1] for i in '
                'range(x.shape[0])): raise ValueError("invalid cu counts")\n'
                '                active=[row[:cv[r+1]-cv[r]] for r,row in enumerate(xv)]\n'
                '            else: active=xv\n'
                '            if any(v>=m.shape[-1] for row in active for v in row): raise '
                'ValueError("logical block out of range")\n'
                "        output_count = len(rows) * (values[2][-1] if TASK == 'packed' else x.numel())\n"
                '        if output_count > I32: raise ValueError("output indexing exceeds int32")\n'
                "        return {'values':values,'layers':len(rows),'blocks':m.shape[-1], "
                "'single':m.ndim==1}\n"
                '    offsets,indptr,lens,tileptr=inputs; ov,iv,lv,tv=values\n'
                '    if any(x.ndim!=1 for x in inputs): raise ValueError("flags inputs must be vectors")\n'
                '    if len(iv)!=len(lv)+1 or len(tv)!=len(lv)+1 or iv[0]!=0: raise ValueError("indptr '
                'lengths/start")\n'
                "    tile=attributes['tile_size']; mx=attributes['max_tiles_per_seq']\n"
                '    if tile>4096: raise ValueError("tile_size supported range 1..4096")\n'
                '    if len(lv)*mx > I32: raise ValueError("output indexing exceeds int32")\n'
                '    if iv[-1]!=len(ov) or any(iv[i]>iv[i+1] or lv[i]<0 or lv[i]>iv[i+1]-iv[i] for i in '
                'range(len(lv))): raise ValueError("offset segments/lens")\n'
                '    if tv != [i*mx for i in range(len(lv)+1)]: raise ValueError("dense tile output '
                'required")\n'
                '    if any((n+tile-1)//tile>mx for n in lv) or any(v<0 for v in ov): raise '
                'ValueError("capacity or offsets")\n'
                "    return {'values':values,'rows':len(lv)}\n",
 'generate_data.py': '"""Deterministic fixtures; no candidate code is used to create expected values."""\n'
                     'import random\n'
                     "TASK = 'dense'\n"
                     'def generate(case):\n'
                     "    if 'values' in case: return case['values'],case.get('attributes',{})\n"
                     "    rng=random.Random(case.get('seed',20261030)); l=case.get('layers',2); "
                     "b=case.get('blocks',31)\n"
                     '    perm=list(range(l*b)); rng.shuffle(perm); m=[perm[i*b:(i+1)*b] for i in range(l)]\n'
                     "    if case.get('single'): m=m[0]\n"
                     "    if TASK=='slot':\n"
                     "        block=case.get('block_size',128); n=case['tokens']; x=[rng.randrange(b*block) "
                     'for _ in range(n)]\n'
                     '        if n>1: x[0]=-1;x[-1]=-7\n'
                     "        return [x,m],{'block_size':block}\n"
                     "    if TASK in ('dense','packed'):\n"
                     "        r,c=case['requests'],case['columns']; x=[[rng.randrange(b) for _ in range(c)] "
                     'for _ in range(r)]\n'
                     '        if r and c: x[0][0]=-1;x[-1][-1]=-9\n'
                     "        if TASK=='dense': return [x,m],{}\n"
                     '        counts=[rng.randrange(c+1) for _ in range(r)]; cu=[0]\n'
                     '        for n in counts: cu.append(cu[-1]+n)\n'
                     '        if r>1:\n'
                     '            counts[1]=0;cu=[0]\n'
                     '            for n in counts: cu.append(cu[-1]+n)\n'
                     '        return [x,m,cu],{}\n'
                     "    tile=case['tile_size']; r=case['requests']; mx=case['max_tiles']; offsets=[]; "
                     'ip=[0]; lens=[]\n'
                     '    for j in range(r):\n'
                     '        length=max(0,(mx-1)*tile-j%(tile+1)); lens.append(length)\n'
                     '        data=list(range(j*mx*tile,j*mx*tile+length))\n'
                     '        if j%3==1: data=[v+1 for v in data]\n'
                     '        if j%3==2 and len(data)>2: data[len(data)//2]+=3\n'
                     '        offsets.extend(data); ip.append(len(offsets))\n'
                     '    return [offsets,ip,lens,[i*mx for i in '
                     "range(r+1)]],{'tile_size':tile,'max_tiles_per_seq':mx}\n"}

def verify_functional(implementation, output):
    import os, subprocess
    implementation=Path(implementation).resolve()
    if not (implementation/'solution.py').is_file():
        raise ValueError('提交工程根目录须包含 solution.py')
    output=Path(output).resolve()
    if output.exists():
        raise FileExistsError('结果文件已存在，请指定新文件')
    with tempfile.TemporaryDirectory(prefix='task-functional-') as tmp:
        root=Path(tmp)
        _write_sources(root,FUNCTIONAL_SOURCES)
        cases=json.loads(Path(__file__).with_name('case.json').read_text())['input_cases']
        (root/'case.json').write_text(json.dumps(cases,ensure_ascii=False))
        env=os.environ.copy()
        env['PYTHONPATH']=os.pathsep.join([str(root),str(implementation),env.get('PYTHONPATH','')])
        subprocess.run([sys.executable,str(root/'functional_checks.py'),'--output',str(output)],cwd=implementation,env=env,check=True)

def main():
    import argparse
    parser=argparse.ArgumentParser(description='标杆 calculate(values, attrs, single=False)；基线 reference("baseline").prepare(inputs, attrs, info)')
    parser.add_argument('--verify-functional',action='store_true',help='执行功能及工程检查，不执行性能验收')
    parser.add_argument('--implementation',help='已完成构建的提交工程根目录')
    parser.add_argument('--output',help='功能检查结果 JSON 文件；须不存在')
    args=parser.parse_args()
    if args.verify_functional:
        if not args.implementation or not args.output:
            parser.error('--verify-functional 需要 --implementation 和 --output')
        verify_functional(args.implementation,args.output)
    else:
        if args.implementation or args.output:parser.error('请同时指定 --verify-functional')
        parser.print_help()

if __name__ == "__main__":
    main()
