"""M7: frozen, fresh-group conservative / original graph / flat comparison."""
import argparse
from collections import Counter,defaultdict
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
METHODS=['flat_m6','v6_m6','v6_m7']
SEED=20261014
TOKENIZER='datasets/tokenizers/qwen2.5-7b/tokenizer.json'
QWEN='Qwen/Qwen2.5-7B-Instruct'


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path,value):
    from spm.runner import atomic_json
    atomic_json(path,value)
def set_models(models):
    for role,prefix in dict(chat='CHAT_MODEL',cheap='CHEAP_MODEL',judge='JUDGE_MODEL',embed='EMBED_MODEL',bench_judge='PRES_JUDGE_MODEL').items():
        os.environ[prefix]=models[role]['model']
def known_tokens(row):
    value=row.get('usage',{}).get('total_tokens')
    return value if type(value) is int and value>=0 else 0
def unknown_tokens(row):
    value=row.get('usage',{}).get('total_tokens')
    return type(value) is not int or value<0


def freeze(phase,ids,calls,tokens):
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.config import VARIANTS,endpoints
    target=BASE/f'm7_{phase}_plan.json'
    if target.exists():raise ValueError('Refuse to overwrite frozen plan')
    casefile=f'runs/spm/m7_{phase}_cases.json';write(casefile,ids)
    argv=['run','--split','all','--seed',str(SEED),'--limit','0','--case-file',casefile,
        '--methods',*METHODS,'--workers','16','--concurrency','8','--max-calls',str(calls),
        '--max-tokens',str(tokens),'--timeout','180','--cache','runs/spm/api_cache_m7.sqlite',
        '--output',f'runs/spm/m7_{phase}','--perma-types','2','--max-history-episodes','24',
        '--model-tpm',QWEN+'=80000','--tpm-tokenizer',TOKENIZER]
    args=parser().parse_args(argv);args.limit=None;cases,_=selected_cases(args)
    frozen=BASE/f'm7_{phase}_frozen/spm';frozen.mkdir(parents=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,frozen/p.name)
    eps=endpoints()
    assert eps['chat'].model==eps['cheap'].model==QWEN
    assert eps['judge'].model=='Pro/deepseek-ai/DeepSeek-V3.2' and eps['embed'].model=='BAAI/bge-m3'
    write(target,dict(protocol='M7 conservative updates: same target/facet + grounded explicit revision',
        phase=phase,seed=SEED,argv=argv,methods=METHODS,configs={m:VARIANTS[m].dump() for m in METHODS},
        models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        cases=[dict(id=c.id,digest=c.digest(),group=c.group,benchmark=c.benchmark,stratum=c.stratum,episodes=len(c.public.episodes)) for c in cases],
        code={p.name:sha(p) for p in frozen.glob('*.py')},case_file_sha256=sha(casefile),
        tokenizer=dict(path=TOKENIZER,sha256=sha(TOKENIZER)),
        sources={p:sha(p) for p in sorted({s for c in cases for s in c.sources})},
        primary_comparison='v6_m7 vs v6_m6; PrefEval adherence and CUPID checklist F1',
        secondary=['v6_m7 vs flat_m6','CUPID native strata','context compression, retained units, operation checks, completion, cost'],
        freezing='Both stages frozen before any M7 API prediction. Development is operational validation only; no outcome-driven tuning.',
        tests_before_freeze='82 passed',resume_policy='One bounded failed-only resume per stage; 800 calls / 2000000 tokens; successful bytes unchanged.',
        cost_policy='Invocation provider usage. Shared/cache method attribution is not independent cold-run cost.',
        rate_limit=dict(advertised_tpm=80000,effective_tpm=72000,workers=16,concurrency=8)))
    print(json.dumps(dict(phase=phase,cases=len(cases),rows=len(cases)*len(METHODS))))


def prepare():
    if (BASE/'m7_reservation.json').exists():raise ValueError('M7 already reserved')
    from spm.adapters import load_cases,select
    set_models(read(BASE/'m6_test_plan.json')['models'])
    prior={str(f.relative_to(ROOT)):sha(f) for f in BASE.glob('*/samples/*.json')}
    observed={read(ROOT/f)['group'] for f in prior}
    # Include previously reserved but possibly unexecuted members, independently of scores.
    for path in BASE.glob('*_reservation.json'):
        r=read(path)
        for key in ('observed_groups','dev_groups','test_groups'):observed.update(r.get(key,[]))
    dev=[];test=[]
    for benchmark,n_dev,n_test in [('prefeval',12,48),('cupid',6,24)]:
        chosen=[];groups=set()
        pool=[c for c in load_cases('datasets',benchmark) if c.group not in observed and len(c.public.episodes)<=24]
        for case in select(pool,None,SEED):
            if case.group in groups:continue
            chosen.append(case);groups.add(case.group)
            if len(chosen)==n_dev+n_test:break
        assert len(chosen)==n_dev+n_test
        dev+=chosen[:n_dev];test+=chosen[n_dev:]
    write(BASE/'m7_prior_samples.json',prior)
    write(BASE/'m7_reservation.json',dict(seed=SEED,observed_groups=sorted(observed),
        dev_ids=[c.id for c in dev],test_ids=[c.id for c in test],
        dev_groups=sorted({c.group for c in dev}),test_groups=sorted({c.group for c in test}),
        selection='One case per fresh identity group, stratified round-robin; both stages reserved before M7 predictions.',
        counts=dict(dev=dict(prefeval=12,cupid=6),test=dict(prefeval=48,cupid=24))))
    src=sqlite3.connect(BASE/'api_cache_m6.sqlite');dst=sqlite3.connect(BASE/'api_cache_m7.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    freeze('dev',[c.id for c in dev],2000,4000000)
    freeze('test',[c.id for c in test],5000,10000000)
    a,b=[read(BASE/f'm7_{p}_plan.json') for p in ('dev','test')]
    assert a['code']==b['code'] and a['configs']==b['configs'] and a['models']==b['models']
    assert not set(a['cases'][i]['group'] for i in range(len(a['cases'])))&set(c['group'] for c in b['cases'])


def frozen_import(phase):
    plan=read(BASE/f'm7_{phase}_plan.json')
    for name,digest in plan['code'].items():assert sha(BASE/f'm7_{phase}_frozen/spm'/name)==digest,name
    assert sha(BASE/f'm7_{phase}_cases.json')==plan['case_file_sha256']
    assert sha(plan['tokenizer']['path'])==plan['tokenizer']['sha256']
    for path,digest in plan['sources'].items():assert sha(path)==digest,path
    sys.path.insert(0,str(BASE/f'm7_{phase}_frozen'));set_models(plan['models'])
    return plan


def run(phase,resume):
    plan=frozen_import(phase)
    from spm.__main__ import main
    argv=plan['argv'].copy()
    if resume:
        out=BASE/f'm7_{phase}';marker=out/'resume_authorization.json'
        if marker.exists():raise ValueError('Bounded resume already invoked')
        rows=list((out/'samples').glob('*.json'))
        assert len(rows)==len(plan['cases'])*len(METHODS)
        successes={str(f.relative_to(ROOT)):sha(f) for f in rows if read(f)['status']=='ok'}
        archive=out/'first_pass_errors';archive.mkdir(exist_ok=True)
        for f in rows:
            if read(f)['status']!='ok':shutil.copy2(f,archive/f.name)
        write(marker,dict(success_hashes=successes,failures=len(rows)-len(successes),max_calls=800,max_tokens=2000000))
        argv[argv.index('--max-calls')+1]='800';argv[argv.index('--max-tokens')+1]='2000000'
    sys.argv=['spm',*argv]
    return main()


def analyze(phase):
    plan=frozen_import(phase)
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.evaluation import aggregate,paired_comparisons
    from spm.models import fingerprint
    from statistics import mean,median
    out=BASE/f'm7_{phase}';manifest=read(out/'manifest.json')['scientific']
    assert all(manifest[k]==plan[k] for k in ('code','models','configs'))
    args=parser().parse_args(plan['argv']);args.limit=None;cases,_=selected_cases(args)
    assert [(c.id,c.digest()) for c in cases]==[(c['id'],c['digest']) for c in plan['cases']]
    rows=[read(f) for f in sorted((out/'samples').glob('*.json'))]
    idx={(r['case_id'],r['method']):r for r in rows}
    assert len(idx)==len(rows)==len(cases)*len(METHODS)
    assert set(idx)=={(c.id,m) for c in cases for m in METHODS}
    for c in cases:
        for m in METHODS:
            r=idx[(c.id,m)];assert r['case_digest']==c.digest() and r['group']==c.group
    write(out/'metrics.json',aggregate(rows,SEED))
    common={c.id for c in cases if all(idx[(c.id,m)]['status']=='ok' for m in METHODS)}
    write(out/'common_success_metrics.json',dict(case_ids=sorted(common),metrics=aggregate([r for r in rows if r['case_id'] in common],SEED)))
    for baseline in ('v6_m6','flat_m6'):
        write(out/f'comparisons_vs_{baseline}.json',paired_comparisons(rows,baseline,SEED))
    write(out/'cupid_stratum_comparisons.json',{s:paired_comparisons([r for r in rows if r['benchmark']=='cupid' and r['stratum']==s],'v6_m6',SEED)
        for s in sorted({c.stratum for c in cases if c.benchmark=='cupid'})})
    audit=[];disagreements=[];operations=[]
    for c in cases:
        good={m:idx[(c.id,m)] for m in METHODS if idx[(c.id,m)]['status']=='ok'}
        snapshots={}
        for m,r in good.items():
            snapshot=read((out/r['trace']['memory_file']).with_suffix('.snapshot.json'));snapshots[m]=snapshot
            eps={e.id:e for e in c.public.episodes}
            for u in snapshot['units']:
                for e in u['evidence']:
                    turn=eps[e['episode_id']].messages[e['turn_idx']]
                    assert turn.role=='user' and e['span'] and e['span'] in turn.content
            assert len(r['trace']['decisions'])==len(r['trace']['candidates'])
            assert {d['unit_id'] for d in r['trace']['decisions']}=={v['unit']['id'] for v in r['trace']['candidates']}
            assert sum(len(e['dialogue_excerpt']) for e in r['trace'].get('raw_evidence',[]))<=6000
        if len(good)==len(METHODS):
            reference=good['flat_m6']['trace']
            assert len({r['trace']['memory_file'] for r in good.values()})==len(METHODS)
            assert all(r['trace']['raw_evidence']==reference['raw_evidence'] and r['trace']['parsed']==reference['parsed'] for r in good.values())
            cfg=plan['configs']['flat_m6'];signature={k:cfg[k] for k in ('use_condition','tool_protocol','window_messages','window_chars','embed_raw')}
            artifacts=[]
            for ep in c.public.episodes:
                key=fingerprint([c.public.user_id,signature,ep.model_dump(mode='json')])
                p=out/'extractions'/f'{key}.json';value=read(p)
                assert value['episode_hash']==fingerprint(ep.model_dump())
                artifacts.append(str(p.relative_to(out)))
            flat=snapshots['flat_m6']
            assert not any(e['action'] in ('merge','supersede','contradict','branch','update_check') for e in flat['events'])
            assert not any(u['challenges'] or u['status'] in ('pending','superseded') for u in flat['units'])
            assert {u['id'] for u in flat['units']}=={u['id'] for p in artifacts for u in read(out/p)['units']}
            audit.append(dict(case_id=c.id,identical_initial_extraction=True,extraction_files=artifacts,
                identical_raw_evidence_and_scores=True,identical_query_analysis=True,
                units={m:good[m]['trace']['unit_status_counts'] for m in METHODS},
                actions={m:dict(Counter(e['action'] for e in snapshots[m]['events'])) for m in METHODS}))
        if 'v6_m7' in good:
            snapshot=snapshots['v6_m7'];units={u['id']:u for u in snapshot['units']}
            checks=[e for e in snapshot['events'] if e['action']=='update_check']
            for event in snapshot['events']:
                if event['action']!='supersede':continue
                matches=[e for e in checks if e['new']==event['new'] and e['old']==event['old'] and e['episode']==event['episode'] and e['check']['explicit_revision']]
                assert matches,'Every supersession requires a grounded explicit revision check'
                witness=matches[-1]['check'];new=units[event['new']];old=units[event['old']]
                assert witness['same_target'] and witness['same_facet'] and witness['mutually_exclusive']
                original=[e for e in new['evidence'] if e['episode_id']==event['episode'] and witness['revision_quote'] in e['span']]
                assert original,'Revision quote must belong to the new unit, not just any USER source'
                ep=eps[event['episode']]
                old_turns=[e['turn_idx'] for e in old['evidence'] if e['episode_id']==ep.id]
                assert ep.order>old['last_order'] or (ep.order==old['last_order'] and old_turns and any(e['turn_idx']>max(old_turns) for e in original))
            operations.append(dict(case_id=c.id,actions=dict(Counter(e['action'] for e in snapshot['events'])),
                rejected_relations=sum(e['initial_relation']!=e['effective_relation'] for e in checks),
                checks=checks,caveat='Grounded algorithm checks; model semantics have not been independently regraded.'))
        if 'v6_m7' in good and 'v6_m6' in good:
            key='follows_preference' if c.benchmark=='prefeval' else 'checklist_f1'
            x,y=[good[m]['evaluation']['metrics'][key] for m in ('v6_m6','v6_m7')]
            disagreements.append(dict(case_id=c.id,benchmark=c.benchmark,stratum=c.stratum,old_score=x,new_score=y,delta=y-x,
                identical_context=good['v6_m6']['trace']['context']==good['v6_m7']['trace']['context'],
                original=dict(trace=good['v6_m6']['trace'],evaluation=good['v6_m6']['evaluation']),
                conservative=dict(trace=good['v6_m7']['trace'],evaluation=good['v6_m7']['evaluation'])))
    write(out/'manager_control_audit.json',audit);write(out/'conservative_operations.json',operations)
    write(out/'conservative_old_case_audit.json',disagreements)
    pending=[dict(case_id=r['case_id'],method=r['method'],error=r['error']) for r in rows if r['status']!='ok']
    write(out/'pending.json',pending)
    efficiency={}
    for benchmark in ('prefeval','cupid'):
        efficiency[benchmark]={}
        for m in METHODS:
            good=[r for r in rows if r['benchmark']==benchmark and r['method']==m and r['case_id'] in common]
            efficiency[benchmark][m]=dict(cases=len(good),mean_context_chars=mean(r['trace']['context_characters'] for r in good) if good else None,
                median_case_seconds=median(r['seconds'] for r in good) if good else None,
                mean_candidates=mean(len(r['trace']['candidates']) for r in good) if good else None,
                mean_applied=mean(sum(d['decision']=='apply' for d in r['trace']['decisions']) for r in good) if good else None,
                caveat='Identical common-success cases; concurrent time includes shared writes/cache/TPM waits.')
    write(out/'matched_efficiency.json',efficiency)
    db=sqlite3.connect(BASE/'api_cache_m7.sqlite');before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    ledger=[]
    for inv in read(out/'invocations.json'):
        ledger.extend(dict(role=role,tag=tag,**json.loads(value)) for role,tag,value in db.execute('SELECT role,tag,json FROM requests WHERE run=?',(inv['run_id'],)))
    for row in ledger:
        if row.get('response_model'):assert row['response_model']==plan['models'][row['role']]['model']
    by_tag=defaultdict(lambda:dict(http_attempts=0,known_tokens=0,unknown_usage=0))
    for r in ledger:
        a=by_tag[r['tag']];a['http_attempts']+=1;a['known_tokens']+=known_tokens(r);a['unknown_usage']+=unknown_tokens(r)
    write(out/'ledger_totals.json',dict(http_attempts=len(ledger),known_tokens=sum(known_tokens(r) for r in ledger),
        unknown_usage=sum(unknown_tokens(r) for r in ledger),by_tag=dict(by_tag),
        statuses=dict(Counter(str(r.get('http_status','unknown')) for r in ledger)),note=plan['cost_policy']))
    assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before;db.close()
    prior=read(BASE/'m7_prior_samples.json')
    for path,digest in prior.items():assert sha(ROOT/path)==digest,path
    marker=out/'resume_authorization.json'
    if marker.exists():
        for path,digest in read(marker)['success_hashes'].items():assert sha(ROOT/path)==digest,path
    res=read(BASE/'m7_reservation.json')
    assert not {c.group for c in cases}&set(res['observed_groups'])
    if phase=='test':assert not {c.group for c in cases}&set(res['dev_groups'])
    verification=dict(planned=len(rows),ok=len(rows)-len(pending),failed=len(pending),common_cases=len(common),
        manager_control_cases=len(audit),conservative_operation_cases=len(operations),
        prior_samples_unchanged=len(prior),frozen_code_models_data_verified=True,offline_added_requests=0)
    write(out/'verification.json',verification);print(json.dumps(verification))


def main():
    sys.path.insert(0,str(ROOT))
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action',choices=['prepare','run','analyze'])
    parser.add_argument('--phase',choices=['dev','test'],default='dev');parser.add_argument('--resume',action='store_true')
    args=parser.parse_args()
    if args.action=='prepare':prepare()
    elif args.action=='analyze':analyze(args.phase)
    else:return run(args.phase,args.resume)
    return 0


if __name__=='__main__':raise SystemExit(main())
