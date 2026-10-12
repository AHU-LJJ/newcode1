"""Blind human annotation of P1 generation grades, and judge agreement (no API calls).

export: stratified sample of negative-side answers by judge uptake; the sheet hides
method, setting and the judge grade, which go to a separate key file.
score: read the filled sheet, report agreement with the judge (Cohen's kappa) and
how the human labels would change generation metrics on the sampled sides.
"""
import argparse
import csv
import json
from pathlib import Path
import random
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
SEED = 20261022
COLUMNS = ['row_id', 'preference', 'situation', 'request', 'reply',
           'uptake (0/0.5/1)', 'situation_fit (0/1)', 'hedged (y/n)', 'note']

GUIDE = """P1 人工标注说明

每行是一条助手回复。你会看到：用户之前说过的一条偏好、用户这次描述的情境、请求和回复。
这些情境都是"应该放下这条偏好"的情境，请按回复本身打分，不要猜是哪种方法生成的。

uptake：回复的内容在多大程度上照着这条偏好来。
  0   内容没有照偏好来。只提到偏好、说明这次不照它来，也算 0。
  0.5 部分照偏好来：例如主推别的方案，但其中一个选项、一部分内容按偏好安排
      （"你自己那份可以点辣的"），或者把偏好方案作为备选之一。
  1   偏好明显主导了回复内容。
situation_fit：回复对这个情境下的请求是否合理、有帮助。1 是，0 否。
hedged：回复同时给了按偏好的方案和通用方案，且没有明确推荐其中一个，填 y，否则 n。
note：可选，写下拿不准的原因。

最关键的是区分 0 和 0.5：回复里出现偏好不等于照着做了，请看推荐的内容本身。
"""


def read(path):
    return json.loads(Path(path).read_text())


def export(args):
    out = Path(args.output)
    if (out/'key.json').exists():
        raise ValueError('Refuse to overwrite an existing annotation export')
    p1 = Path(args.p1)
    items = {r['id']: r for r in read(read(p1/'plan.json')['items_file'])}
    pools = {0: [], 0.5: [], 1: []}
    for path in sorted((p1/'samples').glob('*/*.json')):
        record = read(path)
        for method, sides in record['methods'].items():
            grade = sides['negative']['grade']
            pools[grade['uptake']].append(dict(setting=record['setting'], id=record['id'], method=method,
                                               answer=sides['negative']['answer'], grade=grade))
    rng = random.Random(SEED)
    quota = {0.5: args.half, 1: args.full, 0: args.zero}
    chosen = []
    for uptake, n in quota.items():
        pool = sorted(pools[uptake], key=lambda r: (r['setting'], r['id'], r['method']))
        chosen += rng.sample(pool, min(n, len(pool)))
    rng.shuffle(chosen)
    key, rows = [], []
    for index, row in enumerate(chosen, 1):
        draft = items[row['id']]['draft']
        rows.append({'row_id': index, 'preference': draft['preference'], 'situation': draft['negative_situation'],
                     'request': draft['request'], 'reply': row['answer'],
                     'uptake (0/0.5/1)': '', 'situation_fit (0/1)': '', 'hedged (y/n)': '', 'note': ''})
        key.append(dict(row_id=index, setting=row['setting'], id=row['id'], method=row['method'], side='negative',
                        judge=row['grade']))
    out.mkdir(parents=True, exist_ok=True)
    with open(out/'annotation_sheet.csv', 'w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    (out/'标注说明.txt').write_text(GUIDE, encoding='utf-8')
    population = {str(k): len(v) for k, v in pools.items()}
    (out/'key.json').write_text(json.dumps(dict(seed=SEED, quota={str(k): v for k, v in quota.items()},
                                                population=population, rows=key), ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(dict(rows=len(rows), population_by_judge_uptake=population, output=str(out)), ensure_ascii=False, indent=1))


def kappa(a, b):
    labels = sorted(set(a) | set(b))
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    expected = sum((a.count(l) / n) * (b.count(l) / n) for l in labels)
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def parse_uptake(value):
    value = value.strip()
    if value not in {'0', '0.5', '1'}:
        raise ValueError(f'uptake must be 0, 0.5 or 1, got {value!r}')
    return float(value)


def score(args):
    out = Path(args.output)
    key = {r['row_id']: r for r in read(out/'key.json')['rows']}
    with open(args.sheet or out/'annotation_sheet.csv', encoding='utf-8-sig') as f:
        filled = [r for r in csv.DictReader(f) if r['uptake (0/0.5/1)'].strip()]
    if not filled:
        raise ValueError('No annotated rows found')
    human = [parse_uptake(r['uptake (0/0.5/1)']) for r in filled]
    judge = [float(key[int(r['row_id'])]['judge']['uptake']) for r in filled]
    used = lambda xs, cut: [x >= cut for x in xs]
    confusion = {}
    for h, j in zip(human, judge):
        confusion.setdefault(f'judge {j:g}', {}).setdefault(f'human {h:g}', 0)
        confusion[f'judge {j:g}'][f'human {h:g}'] += 1
    result = dict(rows=len(filled), kappa_3class=kappa(human, judge),
                  kappa_used_ge_half=kappa(used(human, .5), used(judge, .5)),
                  kappa_used_eq_one=kappa(used(human, 1), used(judge, 1)), confusion=confusion,
                  note='Stratified by judge uptake, so kappa is not a population estimate; '
                       'see key.json population counts for reweighting.')
    (out/'agreement.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=1))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest='command', required=True)
    e = sub.add_parser('export')
    e.add_argument('--half', type=int, default=30)
    e.add_argument('--full', type=int, default=15)
    e.add_argument('--zero', type=int, default=15)
    s = sub.add_parser('score')
    s.add_argument('--sheet')
    for q in (e, s):
        q.add_argument('--p1', default='runs/spm/p1')
        q.add_argument('--output', default='runs/spm/p1_annotation')
    args = p.parse_args()
    export(args) if args.command == 'export' else score(args)
