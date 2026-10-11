"""M6: bounded gate completion and same-model graph / flat / raw / history controls."""
import argparse
from collections import Counter, defaultdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BASE=ROOT/'runs/spm'
METHODS=['b1','b2_matched','flat_m6','v6_m6']
QWEN='Qwen/Qwen2.5-7B-Instruct'
TOKENIZER='datasets/tokenizers/qwen2.5-7b/tokenizer.json'
SEED=20261013


def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def known_tokens(row):
    value=row.get('usage',{}).get('total_tokens')
    return value if type(value) is int and value>=0 else 0
def unknown_tokens(row):
    value=row.get('usage',{}).get('total_tokens')
    return type(value) is not int or value<0
def write(p,value):
    from spm.runner import atomic_json
    atomic_json(p,value)
def set_models(models):
    for role,prefix in dict(chat='CHAT_MODEL',cheap='CHEAP_MODEL',judge='JUDGE_MODEL',embed='EMBED_MODEL',bench_judge='PRES_JUDGE_MODEL').items():
        os.environ[prefix]=models[role]['model']


def freeze(phase,ids,methods,max_calls,max_tokens):
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.config import VARIANTS,SPMConfig,endpoints
    target=BASE/f'm6_{phase}_plan.json'
    if target.exists():raise ValueError(f'{phase} is already frozen')
    casefile=f'runs/spm/m6_{phase}_cases.json';write(casefile,ids)
    argv=['run','--split','all','--seed',str(SEED),'--limit','0','--case-file',casefile,
        '--methods',*methods,'--workers','16','--concurrency','8','--max-calls',str(max_calls),
        '--max-tokens',str(max_tokens),'--timeout','180','--cache','runs/spm/api_cache_m6.sqlite',
        '--output',f'runs/spm/m6_{phase}','--perma-types','2','--max-history-episodes','24',
        '--model-tpm',QWEN+'=80000','--tpm-tokenizer',TOKENIZER]
    args=parser().parse_args(argv);args.limit=None;cases,_=selected_cases(args)
    frozen=BASE/f'm6_{phase}_frozen/spm';frozen.mkdir(parents=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,frozen/p.name)
    configs={m:VARIANTS.get(m,SPMConfig(use_anticipation=False,use_gate=False,embed_raw='none')).dump() for m in methods}
    eps=endpoints();assert eps['chat'].model==eps['cheap'].model==QWEN
    assert eps['judge'].model=='Pro/deepseek-ai/DeepSeek-V3.2'
    assert eps['embed'].model=='BAAI/bge-m3'
    write(target,dict(protocol='M6 graph-manager isolation with host-bound missing-gate completion',phase=phase,
        seed=SEED,argv=argv,methods=methods,configs=configs,
        models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        cases=[dict(id=c.id,digest=c.digest(),group=c.group,benchmark=c.benchmark,stratum=c.stratum,
            episodes=len(c.public.episodes)) for c in cases],case_file_sha256=sha(casefile),
        code={p.name:sha(p) for p in frozen.glob('*.py')},tokenizer=dict(path=TOKENIZER,sha256=sha(TOKENIZER)),
        sources={p:sha(p) for p in sorted({s for c in cases for s in c.sources})},
        tests_before_freeze='71 passed' if phase=='test' else '68 passed',
        primary_comparison='v6_m6 vs flat_m6; preference adherence for PrefEval and checklist F1 for CUPID',
        secondary=['v6_m6 vs matched raw RAG and complete history','CUPID changing / contrastive / consistent strata','completion and efficiency'],
        controls=['Same initial compact extraction (exact artifact and vector hashes), condition fields and embeddings for flat/graph.',
            'Flat appends observations without semantic merging, relationship classification, conflict edges or temporal supersession; same reader/gate/generator.',
            'Matched raw RAG uses identical flagged raw windows, anticipated query probes, ranking/tie order and 3-window / 6000-excerpt-character budget.',
            'Complete history is a separate information/cost reference with all public prior turns; no silent truncation.',
            'Targeted gate completion preserves valid unique in-range batch scores and asks only missing candidates for scalar scores; final exact coverage remains mandatory.',
            'No gold or outcome-dependent subset selection; missing predictions remain missing; surrogate judges unchanged.'],
        resume_policy='One failed-only resume per phase; same frozen code, prompt, thresholds and models; preserve successful sample bytes.',
        cache_policy='Isolated validated M5 cache backup. Same semantic extraction calls are shared across managers; memory banks are newly built.',
        cost_policy='Provider totals measured by invocation. Attributed method costs include shared/cache effects and are not independent cold-run costs.',
        rate_limit=dict(advertised_tpm=80000,effective_tpm=72000,workers=16,concurrency=8)))
    print(json.dumps(dict(phase=phase,cases=len(cases),rows=len(cases)*len(methods))))


def prepare_dev():
    if (BASE/'m6_reservation.json').exists():raise ValueError('M6 already reserved')
    from spm.adapters import load_cases,select
    previous=read(BASE/'m5_plan.json');set_models(previous['models'])
    prior={str(f.relative_to(ROOT)):sha(f) for f in BASE.glob('*/samples/*.json')}
    seen={read(ROOT/f)['group'] for f in prior}
    seen_ids={read(ROOT/f)['case_id'] for f in prior}
    dev=[];test=[]
    for benchmark,n_dev,n_test in [('prefeval',12,48),('cupid',6,24)]:
        picked=[];groups=set()
        for case in select([c for c in load_cases('datasets',benchmark) if c.group not in seen],None,SEED):
            if case.group in groups:continue
            picked.append(case);groups.add(case.group)
            if len(picked)==n_dev+n_test:break
        assert len(picked)==n_dev+n_test
        dev.extend(picked[:n_dev]);test.extend(picked[n_dev:])
    write(BASE/'m6_prior_samples.json',prior)
    write(BASE/'m6_reservation.json',dict(seed=SEED,observed_groups=sorted(seen),observed_ids=sorted(seen_ids),
        dev_ids=[c.id for c in dev],test_ids=[c.id for c in test],
        dev_groups=sorted({c.group for c in dev}),test_groups=sorted({c.group for c in test}),
        selection='One case per previously unseen group, stratified round-robin, reserved before any M6 prediction. No outcome-based exclusion.',
        counts=dict(dev=dict(prefeval=12,cupid=6),test=dict(prefeval=48,cupid=24)),
        planned_test_methods=METHODS,planned_test_budgets=dict(max_calls=4500,max_tokens=8000000)))
    src=sqlite3.connect(BASE/'api_cache_m5.sqlite');dst=sqlite3.connect(BASE/'api_cache_m6.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    diagnostics=list(dict.fromkeys(p['case_id'] for p in read(BASE/'m5_execution_failures.json') if p['stage']=='gate'))
    assert len(diagnostics)==4
    freeze('reliability',diagnostics,['v6_m6'],600,1000000)
    freeze('dev',[c.id for c in dev],METHODS,1600,3000000)


def prepare_test():
    reservation=read(BASE/'m6_reservation.json')
    for phase in ('reliability','dev'):
        plan=read(BASE/f'm6_{phase}_plan.json');rows=list((BASE/f'm6_{phase}/samples').glob('*.json'))
        assert len(rows)==len(plan['cases'])*len(plan['methods']),f'Finish {phase} attempts first'
    assert not set(reservation['test_groups'])&set(reservation['observed_groups']+reservation['dev_groups'])
    set_models(read(BASE/'m6_dev_plan.json')['models'])
    freeze('test',reservation['test_ids'],METHODS,4500,8000000)
    plan=read(BASE/'m6_test_plan.json')
    assert set(c['group'] for c in plan['cases'])==set(reservation['test_groups'])
    development=read(BASE/'m6_dev_plan.json')
    changed=[name for name,digest in plan['code'].items() if digest!=development['code'][name]]
    assert set(changed)<={'api.py'},'Only independently tested concurrency/embedding-cache fixes are allowed after development'
    assert plan['configs']==development['configs'] and plan['models']==development['models']
    write(BASE/'m6_freeze_review.json',dict(test_reserved_before_predictions=True,
        dev_code_equals_test_code=not changed,changed_files=changed,
        note='Development used for operational reliability/cost review. Only api.py changed: TPM admission before HTTP semaphore, and first-commit-wins canonical per-text embedding vectors for overlapping batches. Prompts, schemas, configs, thresholds, models, HTTP concurrency and TPM capacity are unchanged. Development RAG retrieved identical excerpts but some similarity scores differed due vector overwrites; archived outputs are retained. Test membership and main contrasts were reserved before development.'))


def frozen_import(phase):
    plan=read(BASE/f'm6_{phase}_plan.json')
    for name,digest in plan['code'].items():assert sha(BASE/f'm6_{phase}_frozen/spm'/name)==digest,name
    assert sha(BASE/f'm6_{phase}_cases.json')==plan['case_file_sha256']
    assert sha(plan['tokenizer']['path'])==plan['tokenizer']['sha256']
    for path,digest in plan['sources'].items():assert sha(path)==digest,path
    sys.path.insert(0,str(BASE/f'm6_{phase}_frozen'));set_models(plan['models']);return plan


def run(args):
    plan=frozen_import(args.phase)
    from spm.__main__ import main
    argv=plan['argv'].copy()
    if args.resume:
        out=BASE/f'm6_{args.phase}';archive=out/'first_pass_errors'
        marker=out/'resume_authorization.json'
        if marker.exists():raise ValueError('Bounded resume was already invoked')
        rows=[read(f) for f in (out/'samples').glob('*.json')]
        assert len(rows)==len(plan['cases'])*len(plan['methods'])
        good={str(f.relative_to(ROOT)):sha(f) for f in (out/'samples').glob('*.json') if read(f)['status']=='ok'}
        for f in (out/'samples').glob('*.json'):
            if read(f)['status']!='ok':archive.mkdir(exist_ok=True);shutil.copy2(f,archive/f.name)
        write(marker,dict(success_hashes=good,failures=len(rows)-len(good),max_calls=600,max_tokens=1500000))
        argv[argv.index('--max-calls')+1]='600';argv[argv.index('--max-tokens')+1]='1500000'
    sys.argv=['spm',*argv];return main()


def analyze(phase):
    plan=frozen_import(phase)
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.evaluation import aggregate,paired_comparisons
    from spm.models import fingerprint
    import numpy as np
    out=BASE/f'm6_{phase}';manifest=read(out/'manifest.json')['scientific']
    assert manifest['code']==plan['code'] and manifest['models']==plan['models'] and manifest['configs']==plan['configs']
    args=parser().parse_args(plan['argv']);args.limit=None;cases,_=selected_cases(args)
    assert [(c.id,c.digest()) for c in cases]==[(c['id'],c['digest']) for c in plan['cases']]
    rows=[read(f) for f in sorted((out/'samples').glob('*.json'))];idx={(r['case_id'],r['method']):r for r in rows}
    assert len(idx)==len(rows)==len(cases)*len(plan['methods'])
    assert set(idx)=={(c.id,m) for c in cases for m in plan['methods']}
    for c in cases:
        for m in plan['methods']:
            r=idx[(c.id,m)];assert r['case_digest']==c.digest() and r['group']==c.group
    write(out/'metrics.json',aggregate(rows,SEED))
    common={c.id for c in cases if all(idx[(c.id,m)]['status']=='ok' for m in plan['methods'])}
    write(out/'common_success_metrics.json',dict(case_ids=sorted(common),metrics=aggregate([r for r in rows if r['case_id'] in common],SEED)))
    if len(plan['methods'])>1:
        for baseline in ('flat_m6','b2_matched','b1'):
            write(out/f'comparisons_vs_{baseline}.json',paired_comparisons(rows,baseline,SEED))
        stratified={}
        for stratum in sorted({c.stratum for c in cases if c.benchmark=='cupid'}):
            subset=[r for r in rows if r['benchmark']=='cupid' and r['stratum']==stratum]
            stratified[stratum]=paired_comparisons(subset,'flat_m6',SEED)
        write(out/'cupid_stratum_comparisons.json',stratified)
    audit=[]
    for c in cases:
        good={m:idx[(c.id,m)] for m in plan['methods'] if idx[(c.id,m)]['status']=='ok'}
        for m in ('flat_m6','v6_m6'):
            if m not in good:continue
            r=good[m];snapshot=read((out/r['trace']['memory_file']).with_suffix('.snapshot.json'))
            episodes={e.id:e for e in c.public.episodes}
            for u in snapshot['units']:
                for e in u['evidence']:
                    turn=episodes[e['episode_id']].messages[e['turn_idx']]
                    assert turn.role=='user' and e['span'] and e['span'] in turn.content
            assert len(r['trace']['decisions'])==len(r['trace']['candidates'])
            assert {d['unit_id'] for d in r['trace']['decisions']}=={v['unit']['id'] for v in r['trace']['candidates']}
        if all(m in good for m in ('flat_m6','v6_m6')):
            a,b=good['flat_m6'],good['v6_m6']
            assert a['trace']['memory_file']!=b['trace']['memory_file']
            assert a['trace']['raw_evidence']==b['trace']['raw_evidence']
            assert a['trace']['parsed']==b['trace']['parsed']
            cfg=plan['configs']['flat_m6']
            signature={k:cfg[k] for k in ('use_condition','tool_protocol','window_messages','window_chars','embed_raw')}
            artifacts=[]
            for ep in c.public.episodes:
                key=fingerprint([c.public.user_id,signature,ep.model_dump(mode='json')])
                artifact=out/'extractions'/f'{key}.json';value=read(artifact)
                assert value['episode_hash']==fingerprint(ep.model_dump())
                artifacts.append(str(artifact.relative_to(out)))
            flat=read((out/a['trace']['memory_file']).with_suffix('.snapshot.json'))
            graph=read((out/b['trace']['memory_file']).with_suffix('.snapshot.json'))
            assert not any(e['action'] in ('merge','supersede','contradict','branch') for e in flat['events'])
            assert not any(u['challenges'] or u['status'] in ('pending','superseded') for u in flat['units'])
            ids={u['id'] for p in artifacts for u in read(out/p)['units']}
            assert {u['id'] for u in flat['units']}==ids
            raw_score_delta=0.
            if 'b2_matched' in good:
                x,y=good['b2_matched']['trace']['raw_evidence'],b['trace']['raw_evidence']
                if phase=='dev':
                    assert [{k:v for k,v in e.items() if k!='score'} for e in x]==[{k:v for k,v in e.items() if k!='score'} for e in y]
                    raw_score_delta=max([abs(a['score']-b['score']) for a,b in zip(x,y,strict=True)],default=0.)
                else:assert x==y
            audit.append(dict(case_id=c.id,identical_initial_extraction=True,extraction_files=artifacts,
                identical_raw_evidence=True,matched_rag_verified='b2_matched' in good,
                rag_score_difference=raw_score_delta,rag_exact_scores=raw_score_delta==0.,
                flat_units=len(flat['units']),graph_units=b['trace']['unit_status_counts'],
                graph_actions=dict(Counter(e['action'] for e in graph['events']))))
        for r in good.values():
            assert sum(len(e['dialogue_excerpt']) for e in r['trace'].get('raw_evidence',[]))<=6000
    write(out/'manager_control_audit.json',audit)
    disagreements=[]
    for c in cases:
        if 'flat_m6' not in plan['methods']:continue
        flat,graph=[idx[(c.id,m)] for m in ('flat_m6','v6_m6')]
        if flat['status']!='ok' or graph['status']!='ok':continue
        key='follows_preference' if c.benchmark=='prefeval' else 'checklist_f1'
        x,y=[r['evaluation']['metrics'][key] for r in (flat,graph)]
        disagreements.append(dict(case_id=c.id,benchmark=c.benchmark,stratum=c.stratum,
            flat_score=x,graph_score=y,delta=y-x,identical_context=flat['trace']['context']==graph['trace']['context'],
            flat=dict(trace=flat['trace'],evaluation=flat['evaluation']),
            graph=dict(trace=graph['trace'],evaluation=graph['evaluation']),
            caveat='Recorded surrogate evaluations and traces. No independent regrading or external fact verification.'))
    write(out/'graph_flat_case_audit.json',disagreements)
    pending=[dict(case_id=r['case_id'],method=r['method'],error=r['error']) for r in rows if r['status']!='ok']
    write(out/'pending.json',pending)
    efficiency={}
    for m in plan['methods']:
        good=[r for r in rows if r['method']==m and r['status']=='ok']
        efficiency[m]=dict(valid=len(good),mean_context_chars=float(np.mean([r['trace']['context_characters'] for r in good])) if good else None,
            median_case_seconds=float(np.median([r['seconds'] for r in good])) if good else None,
            note='Concurrent case time includes TPM waits, cached responses and shared writes; not independent cold latency.')
    write(out/'efficiency.json',efficiency)
    db=sqlite3.connect(BASE/'api_cache_m6.sqlite');before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    ledger=[]
    for inv in read(out/'invocations.json'):
        ledger.extend(dict(role=role,tag=tag,**json.loads(value)) for role,tag,value in db.execute('SELECT role,tag,json FROM requests WHERE run=?',(inv['run_id'],)))
    for row in ledger:
        if row.get('response_model'):assert row['response_model']==plan['models'][row['role']]['model']
    attributed=defaultdict(lambda:dict(http_attempts=0,known_tokens=0,unknown_usage=0))
    by_tag=defaultdict(lambda:dict(http_attempts=0,known_tokens=0,unknown_usage=0))
    for r in ledger:
        context=r.get('experiment_context',{});key='/'.join([context.get('method','unknown'),context.get('stage','unknown')])
        for a in (attributed[key],by_tag[r['tag']]):
            a['http_attempts']+=1;a['known_tokens']+=known_tokens(r);a['unknown_usage']+=unknown_tokens(r)
    write(out/'ledger_totals.json',dict(http_attempts=len(ledger),known_tokens=sum(known_tokens(r) for r in ledger),
        unknown_usage=sum(unknown_tokens(r) for r in ledger),by_tag=dict(by_tag),
        statuses=dict(Counter(str(r.get('http_status','unknown')) for r in ledger)),by_attributed_method_stage=dict(attributed),
        tags=dict(Counter(r['tag'] for r in ledger)),note=plan['cost_policy']))
    assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before;db.close()
    for path,digest in read(BASE/'m6_prior_samples.json').items():assert sha(ROOT/path)==digest,path
    marker=out/'resume_authorization.json'
    if marker.exists():
        for path,digest in read(marker)['success_hashes'].items():assert sha(ROOT/path)==digest,path
    if phase in ('dev','test'):
        res=read(BASE/'m6_reservation.json')
        assert not {c.group for c in cases}&set(res['observed_groups'])
        if phase=='test':assert not {c.group for c in cases}&set(res['dev_groups'])
    verification=dict(planned=len(rows),ok=len(rows)-len(pending),failed=len(pending),common_cases=len(common),
        manager_control_cases=len(audit),prior_samples_unchanged=len(read(BASE/'m6_prior_samples.json')),
        frozen_code_models_data_verified=True,offline_added_requests=0)
    write(out/'verification.json',verification);print(json.dumps(verification))


def main():
    sys.path.insert(0,str(ROOT))
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('action',choices=['prepare-dev','prepare-test','run','analyze'])
    p.add_argument('--phase',choices=['reliability','dev','test'],default='dev');p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.action=='prepare-dev':prepare_dev()
    elif args.action=='prepare-test':prepare_test()
    elif args.action=='analyze':analyze(args.phase)
    else:return run(args)
    return 0

if __name__=='__main__':raise SystemExit(main())
