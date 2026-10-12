"""Build the P1 self-made paired-situation set (no system predictions are made here).

Stages: GLM generation -> rule screen -> embedding dedup and contamination check
against CUPID/BenchPreS preferences -> DeepSeek semantic verification -> Qwen
context-free screening. Every stage result is kept; nothing is silently dropped.
"""
import argparse
import asyncio
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
OUT = ROOT/'runs/spm/p1_pairs'
SEED = 20261020


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix+'.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2)+'\n')
    temporary.replace(path)


def external_preferences():
    """Test-set preference texts the synthetic items must not duplicate."""
    import pyarrow.parquet as pq
    texts = set()
    for r in pq.read_table(ROOT/'datasets/benchpres/data/test-00000-of-00001.parquet').to_pylist():
        texts.update(r['preference_attribute'])
    for r in pq.read_table(ROOT/'datasets/cupid/test.parquet').to_pylist():
        texts.add(r['current_contextual_preference'])
    return sorted(t for t in texts if t and t.strip())


async def main(args):
    from spm.api import Gateway, APIError
    from spm import pairset
    slots = pairset.pilot_slots(args.limit, SEED) if args.limit else pairset.slots()
    eps = pairset.build_endpoints(ROOT/'.env', ROOT/args.builder_env if args.builder_env else None)
    manifest = dict(seed=SEED, slots=len(slots),
                    models={r: dict(model=eps[r].model, base_url=eps[r].base_url) for r in ('builder', 'verifier', 'cheap', 'embed')},
                    code=dict(pairset=sha(ROOT/'spm/pairset.py'), script=sha(Path(__file__))),
                    thresholds=dict(dedup_cosine=args.dedup, contamination_cosine=args.contamination, screen_max_follow=1))
    out = Path(args.output)
    if (out/'manifest.json').exists() and json.loads((out/'manifest.json').read_text()) != manifest:
        raise ValueError('Existing build has different code/config; choose a new output directory')
    write(out/'manifest.json', manifest)
    api = Gateway(eps, out/'api_cache.sqlite', concurrency=args.concurrency, max_calls=args.max_calls,
                  max_tokens=args.max_tokens, timeout=180)
    status = {}
    try:
        async def guarded(identity, stage, coroutine):
            try:
                return await coroutine
            except APIError as error:
                status[identity] = dict(stage=stage, error=str(error)[:300])
                return None
        drafts = await asyncio.gather(*(guarded(s['id'], 'build', pairset.build(api, s)) for s in slots))
        items = {s['id']: dict(slot=s, draft=d) for s, d in zip(slots, drafts) if d is not None}
        print(f'built {len(items)}/{len(slots)}', flush=True)
        for identity, item in items.items():
            issues = pairset.rule_issues(item['draft'])
            item['rule_issues'] = issues
            if issues:
                status[identity] = dict(stage='rules', issues=issues)
        alive = [i for i in items if i not in status]
        texts = [items[i]['draft'].preference for i in alive]
        external = external_preferences()
        vectors = await api.embed(texts)
        kept, dropped = pairset.dedup(alive, vectors, args.dedup)
        for identity, info in dropped.items():
            status[identity] = dict(stage='dedup', **info)
        if external:
            ext = await api.embed(external)
            index = {i: n for n, i in enumerate(alive)}
            for identity in kept:
                sims = ext @ vectors[index[identity]]
                best = int(sims.argmax())
                items[identity]['max_external_cosine'] = float(sims[best])
                if sims[best] > args.contamination:
                    status[identity] = dict(stage='contamination', cosine=float(sims[best]), external=external[best][:200])
        alive = [i for i in kept if i not in status]
        print(f'after rules/dedup/contamination {len(alive)}', flush=True)
        verdicts = await asyncio.gather(*(guarded(i, 'verify', pairset.verify(api, items[i]['draft'])) for i in alive))
        for identity, verdict in zip(alive, verdicts):
            if verdict is None:
                continue
            items[identity]['verdict'] = verdict.model_dump()
            if not verdict.passed():
                status[identity] = dict(stage='verify', failed=[k for k, v in verdict.model_dump().items() if v is False])
        alive = [i for i in alive if i not in status]
        print(f'after verification {len(alive)}', flush=True)
        orders = {i: pairset.screening_orders(i, SEED) for i in alive}
        picks = await asyncio.gather(*(guarded(i, 'screen', pairset.screen(api, items[i]['draft'], orders[i])) for i in alive))
        for identity, chosen in zip(alive, picks):
            if chosen is None:
                continue
            items[identity]['screen'] = dict(orders=orders[identity], picks=chosen)
            if not pairset.screen_passed(chosen):
                status[identity] = dict(stage='screen', picks=chosen)
        accepted = [i for i in alive if i not in status]
        print(f'accepted {len(accepted)}', flush=True)
        records = []
        for identity, item in items.items():
            records.append(dict(id=identity, slot=item['slot'], draft=item['draft'].model_dump(),
                                rule_issues=item.get('rule_issues'), verdict=item.get('verdict'),
                                screen=item.get('screen'), max_external_cosine=item.get('max_external_cosine'),
                                accepted=identity in accepted, rejection=status.get(identity)))
        write(out/'items.json', records)
        failures = {i: v for i, v in status.items() if i not in items}
        write(out/'build_failures.json', failures)
        funnel = dict(slots=len(slots), built=len(items))
        for stage in ('rules', 'dedup', 'contamination', 'verify', 'screen'):
            funnel['rejected_'+stage] = sum(v['stage'] == stage and 'error' not in v for v in status.values())
        funnel['api_failures'] = {stage: sum(v['stage'] == stage and 'error' in v for v in status.values())
                                  for stage in ('build', 'verify', 'screen')}
        funnel['accepted'] = len(accepted)
        by_category = {}
        for identity in accepted:
            category = items[identity]['slot']['category']
            by_category[category] = by_category.get(category, 0)+1
        write(out/'summary.json', dict(funnel=funnel, accepted_by_category=by_category))
        print(json.dumps(dict(funnel=funnel, accepted_by_category=by_category), indent=1), flush=True)
    finally:
        usage = await api.close()
        write(out/'usage.json', usage)
        print(json.dumps({k: usage[k] for k in ('http_requests', 'known_tokens', 'unknown_usage_requests')}), flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default=str(OUT))
    p.add_argument('--limit', type=int, default=0, help='Evenly spaced slot subset for a pilot; 0 = all 320')
    p.add_argument('--builder-env', default='.env zhipu', help='Separate provider file for the builder; empty = SiliconFlow')
    p.add_argument('--concurrency', type=int, default=48, help='HTTP slots shared by all roles; Zhipu allows 50')
    p.add_argument('--max-calls', type=int, default=2500)
    p.add_argument('--max-tokens', type=int, default=6_000_000)
    p.add_argument('--dedup', type=float, default=.92)
    p.add_argument('--contamination', type=float, default=.90)
    asyncio.run(main(p.parse_args()))
