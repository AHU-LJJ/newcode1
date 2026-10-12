"""Offline P1 analysis beyond the frozen summary (no API calls).

Adds paired selection-vs-generation gaps for every method, and an EXPLORATORY
strict-uptake sensitivity (a side counts as "used" only at uptake 1). The strict
threshold was chosen after inspecting samples, so it is reported as exploratory.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def strict(grade):
    """Map uptake 0.5 to 0 so only explicit/central use counts."""
    return dict(grade, uptake=1 if grade['uptake'] == 1 else 0)


def gap_block(records, method, seed, grade_map=None):
    from spm import pairbench
    sel = [{s: r['methods'][method][s]['selected'] for s in pairbench.SIDES} for r in records]
    gen = [{s: (grade_map or (lambda g: g))(r['methods'][method][s]['grade']) for s in pairbench.SIDES} for r in records]
    ids = [r['id'] for r in records]
    a = lambda rows: pairbench.selection_metrics(rows)['pair_success']
    b = lambda rows: pairbench.generation_metrics(rows).get('pair_success')
    g = pairbench.generation_metrics(gen)
    return dict(selection_pair_success=a(sel), generation_pair_success=g.get('pair_success'),
                gap=a(sel) - g['pair_success'] if g.get('pairs') else None,
                gap_95ci=pairbench.bootstrap_pair(sel, gen, ids, a, b, seed=seed),
                generation_use_both=g.get('use_both'), generation_use_neg=g.get('use_neg'),
                hedged_excluded=g.get('hedged_excluded'))


def main(args):
    out = Path(args.output)
    plan = json.loads((out/'plan.json').read_text())
    report = dict(note='Offline analysis of frozen P1 samples. strict_uptake is exploratory (defined after sample inspection).',
                  settings={})
    for setting in plan['settings']:
        records = [json.loads(p.read_text()) for p in sorted((out/'samples'/setting).glob('*.json'))]
        if not records:
            continue
        block = dict(items=len(records), primary={}, strict_uptake={})
        for method in plan['methods']:
            block['primary'][method] = gap_block(records, method, plan['seed'])
            block['strict_uptake'][method] = gap_block(records, method, plan['seed'], strict)
        report['settings'][setting] = block
    failures = json.loads((out/'failures.json').read_text()) if (out/'failures.json').exists() else {}
    missing = {}
    for key, error in failures.items():
        setting = key.split('/', 1)[0]
        reason = 'HTTP 402' if '402' in error else error.split(':', 1)[0]
        missing.setdefault(setting, {}).setdefault(reason, 0)
        missing[setting][reason] += 1
    report['missing'] = {s: dict(planned=len(plan['items']), completed=report['settings'].get(s, {}).get('items', 0),
                                 by_reason=missing.get(s, {})) for s in plan['settings']}
    (out/'report_analysis.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(report['missing'], indent=1))
    for setting, block in report['settings'].items():
        print(f'\n[{setting}] items={block["items"]}')
        for kind in ('primary', 'strict_uptake'):
            for method, m in block[kind].items():
                ci = m['gap_95ci']
                print(f'  {kind:13s} {method:3s} sel_PS={m["selection_pair_success"]:.3f} gen_PS={m["generation_pair_success"]:.3f} '
                      f'gap={m["gap"]:+.3f} ci=[{ci[0]:+.3f},{ci[1]:+.3f}] use_both={m["generation_use_both"]:.3f}')


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--output', default='runs/spm/p1')
    main(p.parse_args())
