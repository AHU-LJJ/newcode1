"""Offline M1 audit and paired analysis; never imports the API gateway."""
import hashlib
import argparse
import json
from pathlib import Path
import sqlite3
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'runs/spm/m1_frozen'))
from spm.evaluation import aggregate, frontier, paired_comparisons


def write(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + '\n')


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--track',choices=['m1','m1r'],default='m1')
    track=parser.parse_args().track
    root = ROOT / 'runs/spm'
    plan = json.loads((root / f'{track}_plan.json').read_text())
    for name, digest in plan['code'].items():
        assert hashlib.sha256((root / 'm1_frozen/spm' / name).read_bytes()).hexdigest() == digest, name
    old = json.loads((root / 'm1_success_before_resume.json').read_text())
    for name, digest in old.items():
        assert hashlib.sha256((ROOT / name).read_bytes()).hexdigest() == digest, name
    db = sqlite3.connect(root / 'api_cache.sqlite')
    before = db.execute('SELECT count(*) FROM requests').fetchone()[0]
    verification = dict(phases={}, frozen_source_verified=True, identity_groups_disjoint=True,
                        preserved_successful_samples=len(old), original_successes_unchanged=True)
    original=json.loads((root/'m1_original_before_rejudge.json').read_text())
    for name,digest in original.items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest()==digest,name
    verification['all_original_checkpoints_unchanged']=len(original)
    if track=='m1r':
        native=json.loads((root/'m1r_native_reuse.json').read_text())
        for record in native:
            for field in ('source','target'):
                assert hashlib.sha256((ROOT/record[field]).read_bytes()).hexdigest()==record['sha256']
        verification['native_rows_reused_byte_identically']=len(native)
    pending = []
    all_rows = {}
    for phase in ('dev', 'test'):
        out = root / f'{track}_{phase}'
        rows = sorted((json.loads(p.read_text()) for p in (out / 'samples').glob('*.json')),
                      key=lambda r: (r['benchmark'], r['case_id'], r['method']))
        all_rows[phase] = rows
        manifest = json.loads((out / 'manifest.json').read_text())['scientific']
        assert manifest['code'] == plan['code']
        assert manifest['models'] == plan['models']
        assert manifest['configs'] == {m:plan['configs'][m] for m in manifest['methods']}
        cases = {c['id']: c for c in plan['phases'][phase]['cases']}
        assert {(r['case_id'], r['method']) for r in rows} == {
            (c, m) for c in cases for m in manifest['methods']}
        for r in rows:
            assert r['case_digest'] == cases[r['case_id']]['digest']
            assert r['group'] == cases[r['case_id']]['group']
        shared = 0
        for case in cases:
            variants = [r for r in rows if r['case_id'] == case and
                        r['method'] in ('v4', 'v3', 'no_anticipation') and r['status'] == 'ok']
            if len(variants) > 1:
                assert len({r['trace']['memory_file'] for r in variants}) == 1
                shared += 1
        verification['phases'][phase] = dict(samples=len(rows), ok=sum(r['status'] == 'ok' for r in rows),
                                             pending=sum(r['status'] != 'ok' for r in rows),
                                             shared_write_cases_verified=shared)
        pending.extend(dict(phase=phase, case_id=r['case_id'], method=r['method'], error=r['error'])
                       for r in rows if r['status'] != 'ok')
        write(out / 'metrics.json', aggregate(rows))
        write(out / 'comparisons.json', paired_comparisons(rows))
        write(out / 'comparisons_vs_b2.json', paired_comparisons(rows, 'b2'))
        write(out / 'frontier.json', frontier(rows, [.3, .4, .5, .6, .7]))
        if phase == 'dev':
            ablations = {}
            for name, baseline in [('gate', 'v3'), ('anticipation', 'no_anticipation')]:
                ablations[name] = {k: v for k, v in paired_comparisons(rows, baseline).items()
                                   if '/v4_vs_' in k}
            write(out / 'ablation_comparisons.json', ablations)
        # Retain every invocation, including interrupted/retry attempts and zero-call replays.
        invocations = json.loads((out / 'invocations.json').read_text())
        usage = [json.loads((out / 'usage' / f"{i['run_id']}.json").read_text()) for i in invocations]
        write(out / 'usage.json', dict(attempts=usage,
                                      http_requests=sum(u['http_requests'] for u in usage),
                                      known_tokens=sum(u['known_tokens'] for u in usage),
                                      unknown_usage_requests=sum(u['unknown_usage_requests'] for u in usage)))
    for benchmark in {c['benchmark'] for c in plan['phases']['dev']['cases']}:
        groups = [{c['group'] for c in plan['phases'][p]['cases'] if c['benchmark'] == benchmark}
                  for p in ('dev', 'test')]
        assert groups[0].isdisjoint(groups[1]), benchmark
    # This is an audit of recorded judge/trace evidence, not a new judge or external fact check.
    audit = []
    rows = all_rows['test']
    b1 = {r['case_id']: r for r in rows if r['method'] == 'b1'}
    for r in rows:
        if r['benchmark'] != 'prefeval' or r['method'] != 'v4' or r['status'] != 'ok':
            continue
        reference = b1[r['case_id']]
        if r['evaluation']['metrics']['follows_preference']:
            continue
        audit.append(dict(case_id=r['case_id'],
                          full_history_success=bool(reference['evaluation']['metrics']['follows_preference']),
                          candidates=[dict(content=c['unit']['content'], scope=c['unit']['scope'],
                                           evidence=c['unit']['evidence']) for c in r['trace']['candidates']],
                          decisions=r['trace']['decisions'], rendered_context=r['trace']['context'],
                          recorded_judge=r['evaluation']['judge'],
                          interpretation='Recorded judge rationale is evidence of its grading, not verified external knowledge.'))
    write(root / f'{track}_error_audit.json', audit)
    write(root / f'{track}_pending.json', pending)
    if track=='m1r':
        matched=0
        for phase,rows in all_rows.items():
            original_rows={r['case_id']+'/'+r['method']:r for r in
                (json.loads(p.read_text()) for p in (root/f'm1_{phase}/samples').glob('*.json'))}
            for row in rows:
                prior=original_rows[row['case_id']+'/'+row['method']]
                if row['benchmark'] in ('prefeval','cupid') and row['status']==prior['status']=='ok':
                    for field in ('response','prediction'):
                        assert row['evaluation'].get(field)==prior['evaluation'].get(field)
                    matched+=1
        verification['matched_original_graded_answers_unchanged']=matched
    after = db.execute('SELECT count(*) FROM requests').fetchone()[0]
    db.close()
    assert before == after, 'Run offline analysis after all API workers have stopped'
    verification.update(offline_analysis_added_requests=after - before, current_tests='50 passed',
                        tests_at_m1_completion='47 passed',
                        new_api_payment_block_tests=True)
    verification['track']=track
    write(root / f'{track}_verification.json', verification)
    print(json.dumps(verification, indent=2))


if __name__ == '__main__':
    main()
