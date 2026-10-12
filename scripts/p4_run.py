"""P4: situational gate prompt (v6_situational) on the frozen P1 pair set.

Only the gate prompt changes relative to P1 L4 (v6_compact): the same given-preference
stores (copied, not modified), retrieval, rendering, generator, selector and judge.
A side whose new context equals the P1 L4 context reuses the P1 result (temperature 0,
same prompt). The paired comparison is L4S vs P1 L4 on the P1-completed items.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
P1 = ROOT/'runs/spm/p1'
P3 = ROOT/'runs/spm/p3'
ANNOTATION = ROOT/'runs/spm/p1_annotation/calibrated_p1.json'
SETTINGS = ('small', 'strong')
CODE = ('spm/pairbench.py', 'spm/pipeline.py', 'spm/prompts.py', 'spm/config.py', 'spm/api.py', 'scripts/p4_run.py')
# Decision rule, fixed before any P4 prediction.
RULE = dict(primary='generation pair_success, L4S - L4, judge grades, paired bootstrap 95% CI',
            improve='CI lower bound > 0 and positive-side generation Use+ drops by less than 0.05',
            harm='CI upper bound < 0 or positive-side generation Use+ drops by 0.05 or more',
            otherwise='inconclusive', max_use_pos_drop=.05)


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


def copy_sqlite(source, target):
    target.parent.mkdir(parents=True, exist_ok=True)
    src = sqlite3.connect(f'file:{source}?mode=ro', uri=True)
    dst = sqlite3.connect(target)
    src.backup(dst)
    dst.close()
    src.close()


def prepare(args):
    from spm.config import VARIANTS
    out = Path(args.output)
    if (out/'plan.json').exists():
        raise ValueError('Refuse to overwrite an existing P4 plan')
    samples = {s: sorted(p.stem for p in (P1/'samples'/s).glob('*.json')) for s in SETTINGS}
    for setting, ids in samples.items():
        for identity in ids:
            copy_sqlite(P1/'memory'/setting/f'{identity}.sqlite', out/'memory'/setting/f'{identity}.sqlite')
    if not (out/'api_cache.sqlite').exists():
        copy_sqlite(P3/'api_cache.sqlite', out/'api_cache.sqlite')
    p1 = read(P1/'plan.json')
    plan = dict(p1_plan_sha256=sha(P1/'plan.json'), seed=p1['seed'], samples=samples,
                sample_sha256={s: {i: sha(P1/'samples'/s/f'{i}.json') for i in ids} for s, ids in samples.items()},
                models=p1['models'], spm_config=VARIANTS['v6_situational'].dump(), baseline_config=p1['spm_config'],
                code={p: sha(ROOT/p) for p in CODE}, rule=RULE,
                budget=dict(max_calls=args.max_calls, max_tokens=args.max_tokens),
                note='Designed after P1/P3 showed gate errors dominate L4 failures. The new gate prompt was written '
                     'from the P1 situation categories, so P4 tests prompt-level gate repair on the same item '
                     'distribution, not out-of-distribution generalization.')
    write(out/'plan.json', plan)
    print(json.dumps({s: len(v) for s, v in samples.items()}))


def check_plan(out):
    plan = read(out/'plan.json')
    for p, digest in plan['code'].items():
        if sha(ROOT/p) != digest:
            raise ValueError(f'Code changed since plan: {p}')
    return plan


async def run_setting(setting, plan, out, args, failures):
    from spm.api import Gateway, APIError
    from spm.config import SPMConfig
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
            notes, target = pairbench.memory_pool(item, plan['seed'])
            if target != record['target_index']:
                raise ValueError(f'Memory pool order differs from P1: {identity}')
            result = dict(id=identity, setting=setting, category=record['category'], sides={}, spm={})
            try:
                spm = await pairbench.spm_contexts(api, item, notes, cfg, out/'memory'/setting)
                for side in pairbench.SIDES:
                    s = spm[side]
                    result['spm'][side] = dict(decisions=s['decisions'], candidates=s['candidates'], context=s['context'],
                        target_decision=next((d for d, c in zip(s['decisions'], s['candidates']) if c == target), None))
                    if s['context'] == record['spm'][side]['context']:
                        result['sides'][side] = dict(record['methods']['L4'][side], reused_from_p1=True)
                        continue
                    picked, answer = await asyncio.gather(pairbench.select(api, 'chat', item, side, s['context'], plan['seed']),
                                                          pairbench.generate(api, 'chat', item, side, s['context']))
                    grade = await pairbench.grade(api, item, side, answer)
                    result['sides'][side] = dict(selected=picked, answer=answer, grade=grade.model_dump(), reused_from_p1=False)
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


def gate_rates(spm_by_side):
    rows = list(spm_by_side)
    applied = lambda side: sum((r[side]['target_decision'] or {}).get('decision') == 'apply' for r in rows)/len(rows)
    return dict(positive_apply=applied('positive'), negative_apply=applied('negative'))


def analyze(out, plan):
    from spm import pairbench
    p_zero = {float(k): v for k, v in read(ANNOTATION)['p_human_zero_given_judge'].items()} if ANNOTATION.exists() else None
    summary = dict(rule=plan['rule'], note=plan['note'], settings={})
    for setting in SETTINGS:
        records = [read(p) for p in sorted((out/'samples'/setting).glob('*.json'))]
        if not records:
            continue
        ids = [r['id'] for r in records]
        p1 = {i: read(P1/'samples'/setting/f'{i}.json') for i in ids}
        tables = dict(L4={i: p1[i]['methods']['L4'] for i in ids}, L4S={r['id']: r['sides'] for r in records})
        gen = {m: [{s: t[i][s]['grade'] for s in pairbench.SIDES} for i in ids] for m, t in tables.items()}
        sel = {m: [{s: t[i][s]['selected'] for s in pairbench.SIDES} for i in ids] for m, t in tables.items()}
        block = dict(items=len(records), planned=len(plan['samples'][setting]), methods={}, deltas={},
                     gate=dict(L4=gate_rates(p1[i]['spm'] for i in ids), L4S=gate_rates(r['spm'] for r in records)),
                     sides_regenerated=sum(not r['sides'][s]['reused_from_p1'] for r in records for s in pairbench.SIDES))
        for m in tables:
            block['methods'][m] = dict(selection=pairbench.selection_metrics(sel[m]), generation=pairbench.generation_metrics(gen[m]),
                                       calibrated=pairbench.calibrated_generation(gen[m], p_zero) if p_zero else None)
        for metric in ('pair_success', 'use_both', 'use_neg', 'use_pos'):
            fn = lambda rows, k=metric: pairbench.generation_metrics(rows).get(k)
            block['deltas'][metric] = dict(delta=fn(gen['L4S'])-fn(gen['L4']),
                                           ci95=pairbench.bootstrap_pair(gen['L4S'], gen['L4'], ids, fn, fn, seed=plan['seed']))
        if p_zero:
            fn = lambda rows: pairbench.calibrated_generation(rows, p_zero)['pair_success']
            block['deltas']['calibrated_pair_success'] = dict(delta=fn(gen['L4S'])-fn(gen['L4']),
                ci95=pairbench.bootstrap_pair(gen['L4S'], gen['L4'], ids, fn, fn, seed=plan['seed']))
        lo, hi = block['deltas']['pair_success']['ci95']
        drop = -block['deltas']['use_pos']['delta']
        rule = plan['rule']
        block['verdict'] = ('improve' if lo > 0 and drop < rule['max_use_pos_drop'] else
                            'harm' if hi < 0 or drop >= rule['max_use_pos_drop'] else 'inconclusive')
        summary['settings'][setting] = block
    failures = read(out/'failures.json') if (out/'failures.json').exists() else {}
    summary['failures'] = failures
    write(out/'summary.json', summary)
    for setting, b in summary['settings'].items():
        print(f'\n[{setting}] items={b["items"]}/{b["planned"]} regenerated sides={b["sides_regenerated"]} verdict={b["verdict"]}')
        print('  gate on target', json.dumps(b['gate']))
        for m, x in b['methods'].items():
            g, c = x['generation'], x['calibrated'] or {}
            print(f'  {m:3s} genPS={g["pair_success"]:.3f} use+={g["use_pos"]:.3f} use-={g["use_neg"]:.3f} '
                  f'both={g["use_both"]:.3f} hedged={g["hedged_excluded"]} selPS={x["selection"]["pair_success"]:.3f} '
                  f'calibratedPS={c.get("pair_success", float("nan")):.3f}')
        for k, v in b['deltas'].items():
            print(f'  {k:24s} {v["delta"]:+.3f} [{v["ci95"][0]:+.3f}, {v["ci95"][1]:+.3f}]')
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
    q.add_argument('--max-calls', type=int, default=8000)
    q.add_argument('--max-tokens', type=int, default=12_000_000)
    r = sub.add_parser('run')
    r.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    r.add_argument('--batch', type=int, default=16)
    r.add_argument('--concurrency', type=int, default=60)
    r.add_argument('--zhipu-concurrency', type=int, default=48)
    r.add_argument('--siliconflow-concurrency', type=int, default=12)
    r.add_argument('--max-calls', type=int, default=8000)
    r.add_argument('--max-tokens', type=int, default=12_000_000)
    r.add_argument('--builder-env', default='.env zhipu')
    a = sub.add_parser('analyze')
    for x in (q, r, a):
        x.add_argument('--output', default='runs/spm/p4')
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'analyze':
        out = Path(args.output)
        analyze(out, read(out/'plan.json'))
    else:
        asyncio.run(run(args))
