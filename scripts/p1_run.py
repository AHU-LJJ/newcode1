"""P1 (judgement != behaviour) on the self-built paired set.

Methods L0 (no memory), L1 (all memories), P (all memories + applicability
instruction), L4 (SPM v6_compact gate over the same pool). Two settings:
small (Qwen2.5-7B everywhere) and strong (GLM-5.3-Flash answers/selects/gates,
Qwen still normalizes). Judge DeepSeek-V3.2. `prepare` freezes items, code,
prompts, models and thresholds before any prediction; `run` refuses a changed plan.
"""
import argparse
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sys
from urllib.parse import urlparse

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20261021
METHODS = ('L0', 'L1', 'P', 'L4')
SETTINGS = ('small', 'strong')
GATES = dict(H2_pass_gap=.20, H2_fail_gap=.10)


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


def gateway_endpoints(setting, builder_env):
    from spm import pairset
    eps = pairset.build_endpoints(ROOT/'.env', ROOT/builder_env)
    roles = dict(chat=eps['chat'], cheap=eps['cheap'], judge=eps['judge'], embed=eps['embed'])
    if setting == 'strong':
        roles['chat'] = eps['builder']
    return roles


def spm_config():
    from spm.config import VARIANTS
    return VARIANTS['v6_compact']


def prepare(args):
    from spm import pairbench, pairset
    out = Path(args.output)
    if (out/'plan.json').exists():
        raise ValueError('Refuse to overwrite an existing P1 plan')
    items_path = Path(args.items)
    items = [r for r in read(items_path) if r['accepted']]
    if args.limit:
        items = items[:args.limit]
    models = {s: {r: dict(model=e.model, base_url=e.base_url) for r, e in gateway_endpoints(s, args.builder_env).items()}
              for s in SETTINGS}
    plan = dict(seed=SEED, methods=list(METHODS), settings=list(SETTINGS), items=[r['id'] for r in items],
                items_file=str(items_path), items_sha256=sha(items_path), models=models, spm_config=spm_config().dump(),
                gates=GATES, budget=dict(max_calls=args.max_calls, max_tokens=args.max_tokens),
                code={p: sha(ROOT/p) for p in ('spm/pairbench.py', 'spm/pairset.py', 'spm/pipeline.py', 'spm/api.py',
                                               'spm/prompts.py', 'spm/config.py', 'scripts/p1_run.py')},
                prompts=dict(select=pairbench.SELECT, answer=pairbench.ANSWER, judge=pairbench.JUDGE,
                             p_instruction=pairbench.P_INSTRUCTION),
                note='Prepared before any P1 prediction. Items built by GLM-5.3-Flash and verified by DeepSeek-V3.2; '
                     'the strong setting therefore answers items written by the same model family.')
    write(out/'plan.json', plan)
    print(json.dumps(dict(items=len(items), output=str(out)), indent=1))


def check_plan(out, args):
    plan = read(out/'plan.json')
    for p, digest in plan['code'].items():
        if sha(ROOT/p) != digest:
            raise ValueError(f'Code changed since plan: {p}')
    if sha(plan['items_file']) != plan['items_sha256']:
        raise ValueError('Items file changed since plan')
    return plan


async def run_setting(setting, plan, items, out, args, failures):
    from spm.api import Gateway, APIError
    from spm import pairbench
    eps = gateway_endpoints(setting, args.builder_env)
    if {r: dict(model=e.model, base_url=e.base_url) for r, e in eps.items()} != plan['models'][setting]:
        raise ValueError(f'Endpoints differ from plan for {setting}')
    # One cache DB for both settings so the Qwen TPM limiter coordinates them.
    # Zhipu allows 50 concurrent; SiliconFlow DeepSeek returned 429s and transport
    # failures at higher fan-out in the smoke test, so it gets its own smaller cap.
    hosts = dict(zhipu=urlparse(eps['chat'].base_url).netloc if setting == 'strong' else None,
                 siliconflow=urlparse(eps['judge'].base_url).netloc)
    limits = {hosts['siliconflow']: args.siliconflow_concurrency}
    if hosts['zhipu']:
        limits[hosts['zhipu']] = args.zhipu_concurrency
    api = Gateway(eps, out/'api_cache.sqlite', concurrency=args.concurrency[setting],
                  max_calls=args.max_calls, max_tokens=args.max_tokens, timeout=240, provider_limits=limits)
    cfg = spm_config()
    try:
        async def one(item):
            path = out/'samples'/setting/f'{item["id"]}.json'
            if path.exists():
                return
            notes, target = pairbench.memory_pool(item, SEED)
            record = dict(id=item['id'], setting=setting, category=item['slot']['category'],
                          domain=item['slot']['domain'], target_index=target, methods={})
            try:
                spm = await pairbench.spm_contexts(api, item, notes, cfg, out/'memory'/setting)
                record['spm'] = {s: dict(decisions=v['decisions'], candidates=v['candidates'],
                                         target_decision=next((d for d, c in zip(v['decisions'], v['candidates']) if c == target), None),
                                         context=v['context']) for s, v in spm.items()}
                async def method(name):
                    sides = {}
                    for side in pairbench.SIDES:
                        memory = pairbench.context_for(name, notes, spm[side]['context'] if name == 'L4' else None)
                        picked, answer = await asyncio.gather(pairbench.select(api, 'chat', item, side, memory, SEED),
                                                              pairbench.generate(api, 'chat', item, side, memory))
                        grade = await pairbench.grade(api, item, side, answer)
                        sides[side] = dict(selected=picked, answer=answer, grade=grade.model_dump())
                    record['methods'][name] = sides
                await asyncio.gather(*(method(m) for m in METHODS))
            except APIError as error:
                failures[f'{setting}/{item["id"]}'] = str(error)[:300]
                return
            write(path, record)
        done = 0
        for chunk in range(0, len(items), args.batch):
            await asyncio.gather(*(one(i) for i in items[chunk:chunk+args.batch]))
            done += len(items[chunk:chunk+args.batch])
            print(f'{setting}: {done}/{len(items)}', flush=True)
    finally:
        usage = await api.close()
        write(out/'usage'/f'{setting}_{usage["run_id"]}.json', usage)
        print(setting, json.dumps({k: usage[k] for k in ('http_requests', 'known_tokens', 'unknown_usage_requests')}), flush=True)


def analyze(out, plan):
    from spm import pairbench
    summary = dict(settings={}, gates=plan['gates'])
    for setting in plan['settings']:
        records = [read(p) for p in sorted((out/'samples'/setting).glob('*.json'))]
        ids = [r['id'] for r in records]
        block = dict(items=len(records), methods={})
        for name in plan['methods']:
            sel = [{s: r['methods'][name][s]['selected'] for s in pairbench.SIDES} for r in records]
            gen = [{s: r['methods'][name][s]['grade'] for s in pairbench.SIDES} for r in records]
            block['methods'][name] = dict(selection=pairbench.selection_metrics(sel), generation=pairbench.generation_metrics(gen))
        # H2: P selection pair success vs P generation pair success, paired over the same items.
        sel = [{s: r['methods']['P'][s]['selected'] for s in pairbench.SIDES} for r in records]
        gen = [{s: r['methods']['P'][s]['grade'] for s in pairbench.SIDES} for r in records]
        a = lambda rows: pairbench.selection_metrics(rows)['pair_success']
        b = lambda rows: pairbench.generation_metrics(rows).get('pair_success')
        gap = a(sel) - b(gen) if records else None
        ci = pairbench.bootstrap_pair(sel, gen, ids, a, b, seed=plan['seed']) if records else None
        verdict = None if gap is None else ('pass' if gap >= plan['gates']['H2_pass_gap'] else
                                            'fail' if gap < plan['gates']['H2_fail_gap'] else 'inconclusive')
        block['H2'] = dict(selection_pair_success=a(sel) if records else None,
                           generation_pair_success=b(gen) if records else None, gap=gap, gap_95ci=ci, verdict=verdict)
        gate = [r['spm'][s]['target_decision'] for r in records for s in pairbench.SIDES]
        block['L4_gate_on_target'] = dict(
            positive_apply=sum((r['spm']['positive']['target_decision'] or {}).get('decision') == 'apply' for r in records)/max(1, len(records)),
            negative_apply=sum((r['spm']['negative']['target_decision'] or {}).get('decision') == 'apply' for r in records)/max(1, len(records)),
            target_not_retrieved=sum(g is None for g in gate))
        by_category = {}
        for r in records:
            by_category.setdefault(r['category'], []).append(r)
        block['P_by_category'] = {c: dict(n=len(rs), selection=pairbench.selection_metrics([{s: x['methods']['P'][s]['selected'] for s in pairbench.SIDES} for x in rs])['pair_success'],
                                          generation=pairbench.generation_metrics([{s: x['methods']['P'][s]['grade'] for s in pairbench.SIDES} for x in rs]).get('pair_success'))
                                  for c, rs in sorted(by_category.items())}
        summary['settings'][setting] = block
    failures = read(out/'failures.json') if (out/'failures.json').exists() else {}
    summary['failures'] = len(failures)
    write(out/'summary.json', summary)
    print(json.dumps(summary, indent=1))


async def run(args):
    out = Path(args.output)
    plan = check_plan(out, args)
    by_id = {r['id']: r for r in read(plan['items_file'])}
    items = [by_id[i] for i in plan['items']]
    failures_path = out/'failures.json'
    failures = {}  # rebuilt each run; completed samples are skipped, failed ones retried
    try:
        await asyncio.gather(*(run_setting(s, plan, items, out, args, failures) for s in args.settings))
    finally:
        write(failures_path, failures)
    analyze(out, plan)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    q = sub.add_parser('prepare')
    q.add_argument('--items', default='runs/spm/p1_pairs/items.json')
    q.add_argument('--limit', type=int, default=0)
    q.add_argument('--max-calls', type=int, default=30000)
    q.add_argument('--max-tokens', type=int, default=40_000_000)
    for q in (q, sub.add_parser('run'), sub.add_parser('analyze')):
        q.add_argument('--output', default='runs/spm/p1')
        q.add_argument('--builder-env', default='.env zhipu')
    r = sub.choices['run']
    r.add_argument('--settings', nargs='+', choices=SETTINGS, default=list(SETTINGS))
    r.add_argument('--batch', type=int, default=16)
    r.add_argument('--small-concurrency', type=int, default=24)
    r.add_argument('--strong-concurrency', type=int, default=60)
    r.add_argument('--zhipu-concurrency', type=int, default=48, help='Per strong gateway; Zhipu limit is 50')
    r.add_argument('--siliconflow-concurrency', type=int, default=12, help='Per gateway, both settings run at once')
    r.add_argument('--max-calls', type=int, default=30000)
    r.add_argument('--max-tokens', type=int, default=40_000_000)
    args = p.parse_args()
    if args.command == 'prepare':
        prepare(args)
    elif args.command == 'analyze':
        out = Path(args.output)
        analyze(out, read(out/'plan.json'))
    else:
        args.concurrency = dict(small=args.small_concurrency, strong=args.strong_concurrency)
        asyncio.run(run(args))
