"""P3 gate oracle on the frozen P1 L4 runs: separate gate errors from generation errors.

O1 replaces only the target preference's gate decision with its label; O2 also
suppresses every companion. Retrieval, rendering, generator, selector and judge are
the P1 ones. A side whose oracle context equals the recorded L4 context reuses the
P1 result unchanged (temperature 0, same prompt); only changed sides call the API.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P1 = ROOT/'runs/spm/p1'
SETTINGS = ('small', 'strong')


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text())


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


CODE = ('spm/pairbench.py', 'spm/pairset.py', 'spm/pipeline.py', 'spm/api.py', 'scripts/p3_run.py')


def prepare(args):
    from spm import pairbench
    out = Path(args.output)
    if (out/'plan.json').exists():
        raise ValueError('Refuse to overwrite an existing P3 plan')
    p1 = read(P1/'plan.json')
    samples = {s: sorted(p.stem for p in (P1/'samples'/s).glob('*.json')) for s in SETTINGS}
    plan = dict(p1_plan_sha256=sha(P1/'plan.json'), seed=p1['seed'], oracles=list(pairbench.ORACLES),
                samples=samples, sample_sha256={s: {i: sha(P1/'samples'/s/f'{i}.json') for i in ids} for s, ids in samples.items()},
                models=p1['models'], spm_config=p1['spm_config'], code={p: sha(ROOT/p) for p in CODE},
                budget=dict(max_calls=args.max_calls, max_tokens=args.max_tokens),
                note='Prepared after observing P1 results; P3 is a diagnostic decomposition, not a new comparison. '
                     'P1 samples are inputs and are never modified.')
    write(out/'plan.json', plan)
    print(json.dumps({s: len(v) for s, v in samples.items()}))


def check_plan(out):
    plan = read(out/'plan.json')
    for p, digest in plan['code'].items():
        if sha(ROOT/p) != digest:
            raise ValueError(f'Code changed since plan: {p}')
    if sha(P1/'plan.json') != plan['p1_plan_sha256']:
        raise ValueError('P1 plan changed')
    return plan


async def run_setting(setting, plan, out, args, failures):
    from spm.api import Gateway, APIError
    from spm.config import SPMConfig
    from spm.models import Decision
    from spm import pairbench
    sys.path.insert(0, str(ROOT/'scripts'))
    import p1_run
    eps = p1_run.gateway_endpoints(setting, args.builder_env)
    if {r: dict(model=e.model, base_url=e.base_url) for r, e in eps.items()} != plan['models'][setting]:
        raise ValueError(f'Endpoints differ from P1 for {setting}')
    hosts = {urlparse(eps['judge'].base_url).netloc: args.siliconflow_concurrency}
    if setting == 'strong':
        hosts[urlparse(eps['chat'].base_url).netloc] = args.zhipu_concurrency
    api = Gateway(eps, out/'api_cache.sqlite', concurrency=args.concurrency, max_calls=args.max_calls,
                  max_tokens=args.max_tokens, timeout=240, provider_limits=hosts)
    cfg = SPMConfig(**plan['spm_config'])
    by_id = {r['id']: r for r in read(read(P1/'plan.json')['items_file'])}
    try:
        async def one(identity):
            path = out/'samples'/setting/f'{identity}.json'
            if path.exists():
                return
            source = P1/'samples'/setting/f'{identity}.json'
            if sha(source) != plan['sample_sha256'][setting][identity]:
                raise ValueError(f'P1 sample changed: {setting}/{identity}')
            record, item = read(source), by_id[identity]
            units = pairbench.read_units(P1/'memory'/setting/f'{identity}.sqlite')
            result = dict(id=identity, setting=setting, category=record['category'], oracles={})
            try:
                for mode in pairbench.ORACLES:
                    sides = {}
                    for side in pairbench.SIDES:
                        s = record['spm'][side]
                        decisions = pairbench.oracle_decisions(s['decisions'], s['candidates'], record['target_index'], side, mode)
                        context = pairbench.render_oracle(units, decisions, cfg)
                        if context == s['context']:
                            sides[side] = dict(record['methods']['L4'][side], reused_from_p1=True)
                            continue
                        picked, answer = await asyncio.gather(pairbench.select(api, 'chat', item, side, context, plan['seed']),
                                                              pairbench.generate(api, 'chat', item, side, context))
                        grade = await pairbench.grade(api, item, side, answer)
                        sides[side] = dict(selected=picked, answer=answer, grade=grade.model_dump(), reused_from_p1=False, context=context)
                    result['oracles'][mode] = sides
            except APIError as error:
                failures[f'{setting}/{identity}'] = str(error)[:300]
                return
            write(path, result)
        ids = plan['samples'][setting]
        for chunk in range(0, len(ids), args.batch):
            await asyncio.gather(*(one(i) for i in ids[chunk:chunk+args.batch]))
            print(f'{setting}: {min(chunk+args.batch, len(ids))}/{len(ids)}', flush=True)
    finally:
        usage = await api.close()
        write(out/'usage'/f'{setting}_{usage["run_id"]}.json', usage)
        print(setting, json.dumps({k: usage[k] for k in ('http_requests', 'known_tokens', 'unknown_usage_requests')}), flush=True)


def attribution(p1_records):
    """Classify each L4 generation pair from P1 by whether the gate was right on both sides."""
    from spm import pairbench
    rows = dict(success=0, gate_error=0, generation_error=0, hedged=0)
    detail = dict(gate_error_negative_applied=0, gate_error_positive_suppressed=0)
    for r in p1_records:
        g = {s: r['methods']['L4'][s]['grade'] for s in pairbench.SIDES}
        if g['positive']['hedged'] or g['negative']['hedged']:
            rows['hedged'] += 1
            continue
        ok = pairbench.generation_metrics([g])['pair_success'] == 1
        gate = {s: pairbench.gate_correct(r, s) for s in pairbench.SIDES}
        if ok:
            rows['success'] += 1
        elif not all(gate.values()):
            rows['gate_error'] += 1
            detail['gate_error_negative_applied'] += not gate['negative']
            detail['gate_error_positive_suppressed'] += not gate['positive']
        else:
            rows['generation_error'] += 1
    failed = rows['gate_error'] + rows['generation_error']
    return dict(rows, **detail, gate_share_of_failures=rows['gate_error']/failed if failed else None)


def analyze(out, plan):
    from spm import pairbench
    summary = dict(note=plan['note'], settings={})
    for setting in SETTINGS:
        records = [read(p) for p in sorted((out/'samples'/setting).glob('*.json'))]
        if not records:
            continue
        p1 = {r['id']: r for r in (read(P1/'samples'/setting/f'{x["id"]}.json') for x in records)}
        ids = [r['id'] for r in records]
        variants = dict(L4={i: p1[i]['methods']['L4'] for i in ids})
        for mode in plan['oracles']:
            variants[mode] = {r['id']: r['oracles'][mode] for r in records}
        block = dict(items=len(records), methods={}, vs_L4={}, attribution_p1_L4=attribution(list(p1.values())))
        for name, table in variants.items():
            sel = [{s: table[i][s]['selected'] for s in pairbench.SIDES} for i in ids]
            gen = [{s: table[i][s]['grade'] for s in pairbench.SIDES} for i in ids]
            block['methods'][name] = dict(selection=pairbench.selection_metrics(sel), generation=pairbench.generation_metrics(gen),
                                          sides_regenerated=sum(not table[i][s].get('reused_from_p1', True) for i in ids for s in pairbench.SIDES))
        for mode in plan['oracles']:
            for metric in ('pair_success', 'use_both', 'use_neg'):
                gen_m = lambda rows, k=metric: pairbench.generation_metrics(rows).get(k)
                a = [{s: variants[mode][i][s]['grade'] for s in pairbench.SIDES} for i in ids]
                b = [{s: variants['L4'][i][s]['grade'] for s in pairbench.SIDES} for i in ids]
                block['vs_L4'][f'{mode}_{metric}'] = dict(delta=gen_m(a)-gen_m(b),
                                                         ci95=pairbench.bootstrap_pair(a, b, ids, gen_m, gen_m, seed=plan['seed']))
        summary['settings'][setting] = block
    failures = read(out/'failures.json') if (out/'failures.json').exists() else {}
    summary['failures'] = failures
    write(out/'summary.json', summary)
    for setting, block in summary['settings'].items():
        print(f'\n[{setting}] items={block["items"]}  attribution(P1 L4)={json.dumps(block["attribution_p1_L4"])}')
        for name, m in block['methods'].items():
            g = m['generation']
            print(f'  {name:3s} regen={m["sides_regenerated"]:3d} genPS={g["pair_success"]:.3f} use+={g["use_pos"]:.3f} '
                  f'use-={g["use_neg"]:.3f} both={g["use_both"]:.3f} hedged={g["hedged_excluded"]} selPS={m["selection"]["pair_success"]:.3f}')
        for k, v in block['vs_L4'].items():
            print(f'  {k:18s} {v["delta"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]')
    print('failures', len(failures))


async def run(args):
    out = Path(args.output)
    plan = check_plan(out)
    failures = {}
    try:
        await asyncio.gather(*(run_setting(s, plan, out, args, failures) for s in args.settings))
    finally:
        write(out/'failures.json', failures)
    analyze(out, plan)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--max-calls', type=int, default=6000)
    q.add_argument('--max-tokens', type=int, default=8_000_000)
    r = sub.add_parser('run')
    r.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    r.add_argument('--batch', type=int, default=16)
    r.add_argument('--concurrency', type=int, default=60)
    r.add_argument('--zhipu-concurrency', type=int, default=48)
    r.add_argument('--siliconflow-concurrency', type=int, default=12)
    r.add_argument('--max-calls', type=int, default=6000)
    r.add_argument('--max-tokens', type=int, default=8_000_000)
    r.add_argument('--builder-env', default='.env zhipu')
    a = sub.add_parser('analyze')
    for x in (q, r, a):
        x.add_argument('--output', default='runs/spm/p3')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'analyze':
        out = Path(args.output)
        analyze(out, read(out/'plan.json'))
    else:
        asyncio.run(run(args))
