"""Freeze and run source-bound compact v6, with an isolated raw-evidence ablation."""
import argparse
from collections import Counter
import hashlib
import json
import os
from pathlib import Path
import shutil
import sqlite3
import sys

ROOT=Path(__file__).resolve().parents[1]
os.chdir(ROOT)
BASE=ROOT/'runs/spm';PLAN=BASE/'m5_plan.json'
METHODS=['v6_compact_no_raw','v6_compact']
QWEN='Qwen/Qwen2.5-7B-Instruct';TOKENIZER='datasets/tokenizers/qwen2.5-7b/tokenizer.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,value):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def set_models(models):
    prefixes=dict(chat='CHAT_MODEL',cheap='CHEAP_MODEL',judge='JUDGE_MODEL',embed='EMBED_MODEL',bench_judge='PRES_JUDGE_MODEL')
    for role,key in prefixes.items():os.environ[key]=models[role]['model']

def prepare():
    if PLAN.exists():raise ValueError('M5 already frozen; do not overwrite')
    sys.path.insert(0,str(ROOT))
    from spm.adapters import load_cases,select
    from spm.config import VARIANTS,endpoints
    from spm.__main__ import parser
    from spm.runner import selected_cases
    prior={};seen_groups=set();seen_ids=set()
    for f in BASE.glob('*/samples/*.json'):
        row=read(f);prior[str(f.relative_to(ROOT))]=sha(f)
        seen_groups.add(row['group']);seen_ids.add(row['case_id'])
    write(BASE/'m5_prior_samples.json',prior)
    seed=20261012;fresh=[]
    for b,count in [('prefeval',24),('cupid',12)]:
        pool=[c for c in load_cases('datasets',b) if c.group not in seen_groups]
        groups=set()
        for c in select(pool,None,seed):
            if c.group in groups:continue
            fresh.append(c);groups.add(c.group)
            if len(groups)==count:break
        assert len(groups)==count
    previous=read(BASE/'m4_plan.json')
    diagnostic=previous['phases']['test']['cases'];phases={}
    for phase,ids in [('diagnostic',[c['id'] for c in diagnostic]),('test',[c.id for c in fresh])]:
        casefile=f'runs/spm/m5_{phase}_cases.json';write(casefile,ids)
        argv=['run','--split','all','--seed',str(seed),'--limit','0','--case-file',casefile,
              '--methods',*METHODS,'--workers','16','--concurrency','8','--max-calls','2000',
              '--max-tokens','4000000','--timeout','180','--cache','runs/spm/api_cache_m5.sqlite',
              '--output',f'runs/spm/m5_{phase}','--perma-types','2','--max-history-episodes','24',
              '--model-tpm',QWEN+'=80000','--tpm-tokenizer',TOKENIZER]
        args=parser().parse_args(argv);args.limit=None;cases,_=selected_cases(args)
        if phase=='diagnostic':assert [(c.id,c.digest()) for c in cases]==[(c['id'],c['digest']) for c in diagnostic]
        else:assert not {c.group for c in cases}&seen_groups and not {c.id for c in cases}&seen_ids
        phases[phase]=dict(argv=argv,methods=METHODS,case_file_sha256=sha(casefile),
                          cases=[dict(id=c.id,digest=c.digest(),group=c.group,benchmark=c.benchmark,stratum=c.stratum) for c in cases])
    freeze=BASE/'m5_frozen/spm';freeze.mkdir(parents=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,freeze/p.name)
    src=sqlite3.connect(BASE/'api_cache_m4.sqlite');dst=sqlite3.connect(BASE/'api_cache_m5.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    set_models(previous['models']);eps=endpoints()
    assert eps['chat'].model==eps['cheap'].model==QWEN
    write(PLAN,dict(protocol='M5 compact source-bound v6 reliability diagnostic plus predeclared fresh-group raw ablation',
        seed=seed,phases=phases,models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        configs={m:VARIANTS[m].dump() for m in METHODS},code={p.name:sha(p) for p in freeze.glob('*.py')},
        tokenizer=dict(path=TOKENIZER,sha256=sha(TOKENIZER)),reference_plan_sha256=sha(BASE/'m4_plan.json'),
        previously_observed_groups=sorted(seen_groups),previously_observed_ids=sorted(seen_ids),
        tests_before_freeze='63 passed',
        hypotheses=['Local source/index selection will improve end-to-end completion on previously observed M4 questions.',
                    'At identical compact writes and gates, bounded raw evidence may improve preference following and CUPID inference on new groups.'],
        changes=['USER text is partitioned into literal sentence/newline excerpts at most 600 characters; model selects source_id; host binds exact span and original turn.',
                 'Given preferences normalized one at a time; host binds full input as explicit evidence. No applicability labels are exposed.',
                 'Relations and gate return local integer indices; exact-once coverage and all existing semantic guards remain.',
                 'No invalid output fallback, no strong-model writer, no relaxed evidence validation, no reinforcement learning.',
                 'Raw evidence cap stays 6000 characters / three retrieved windows, disabled for given BenchPreS preferences.'],
        rate_limit=dict(advertised_tpm=80000,effective_tpm=72000,concurrency=8,workers=16),
        cache='Isolated M4 cache backup; fresh compact banks. Valid identical embeddings/judge calls can reuse cache; legacy generation not reused as compact generation.',
        holdout='Fresh test includes only PrefEval and CUPID. No unseen BenchPreS profiles or PERMA sd_clean type2 users within 24-episode cap remain.',
        policy='Freeze both phases before predictions; do not tune on diagnostic outputs; execution failures remain missing and full metrics null.'))
    print(json.dumps({p:len(v['cases'])*len(METHODS) for p,v in phases.items()}))

def frozen_import():
    plan=read(PLAN)
    assert sha(BASE/'m4_plan.json')==plan['reference_plan_sha256']
    assert sha(plan['tokenizer']['path'])==plan['tokenizer']['sha256']
    for name,digest in plan['code'].items():assert sha(BASE/'m5_frozen/spm'/name)==digest,name
    for phase,p in plan['phases'].items():assert sha(BASE/f'm5_{phase}_cases.json')==p['case_file_sha256']
    sys.path.insert(0,str(BASE/'m5_frozen'));return plan

def run(args):
    plan=frozen_import();set_models(plan['models'])
    from spm.__main__ import main
    argv=plan['phases'][args.phase]['argv'].copy()
    for name in ('workers','concurrency','max_calls','max_tokens','timeout'):
        value=getattr(args,name)
        if value is not None:argv[argv.index('--'+name.replace('_','-'))+1]=str(value)
    sys.argv=['spm',*argv];return main()

def analyze():
    plan=frozen_import()
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.evaluation import aggregate,paired_comparisons
    verification={};pending=[];all_ledger=[]
    db=sqlite3.connect(BASE/'api_cache_m5.sqlite');before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    for phase,p in plan['phases'].items():
        out=BASE/f'm5_{phase}';manifest=read(out/'manifest.json')['scientific']
        assert manifest['models']==plan['models'] and manifest['code']==plan['code'] and manifest['configs']==plan['configs']
        for path,digest in manifest['sources'].items():assert sha(path)==digest
        args=parser().parse_args(p['argv']);args.limit=None;cases,_=selected_cases(args)
        assert [(c.id,c.digest(),c.group) for c in cases]==[(c['id'],c['digest'],c['group']) for c in p['cases']]
        rows=[read(f) for f in sorted((out/'samples').glob('*.json'))];idx={(r['case_id'],r['method']):r for r in rows}
        assert len(idx)==len(rows)
        assert len(rows)==len(cases)*len(METHODS), 'Finish all planned attempts before analysis; checkpoint success alone is not full coverage'
        assert set(idx)<={(c.id,m) for c in cases for m in METHODS}
        for c in cases:
            for m in METHODS:
                r=idx.get((c.id,m))
                if r:assert r['case_digest']==c.digest() and r['group']==c.group
                if not r or r['status']!='ok':pending.append(dict(phase=phase,case_id=c.id,method=m,error=r['error'] if r else 'not checkpointed'))
        write(out/'metrics.json',aggregate(rows,plan['seed']))
        common={c.id for c in cases if all(idx.get((c.id,m),{}).get('status')=='ok' for m in METHODS)}
        write(out/'common_success_metrics.json',dict(case_ids=sorted(common),metrics=aggregate([r for r in rows if r['case_id'] in common],plan['seed'])))
        write(out/'comparisons_raw.json',paired_comparisons(rows,METHODS[0],plan['seed']))
        audit=[]
        for c in cases:
            if c.id not in common:continue
            a,b=[idx[(c.id,m)] for m in METHODS]
            assert a['trace']['memory_file']==b['trace']['memory_file']
            assert a['trace']['decisions']==b['trace']['decisions']
            n=sum(len(e['dialogue_excerpt']) for e in b['trace']['raw_evidence']);assert n<=6000
            assert not a['trace']['raw_evidence']
            if c.public.given:assert n==0 and a['trace']['context']==b['trace']['context']
            audit.append(dict(case_id=c.id,same_write_and_gate=True,raw_chars=n))
        write(out/'gate_audit.json',audit)
        writes=[]
        for c in cases:
            r=next((idx.get((c.id,m)) for m in METHODS if idx.get((c.id,m),{}).get('status')=='ok'),None)
            if not r:continue
            snapshot=read((out/r['trace']['memory_file']).with_suffix('.snapshot.json'))
            episodes={ep.id:ep for ep in c.public.episodes}
            evidence_count=0
            for unit in snapshot['units']:
                for evidence in unit['evidence']:
                    ep=episodes[evidence['episode_id']];turn=ep.messages[evidence['turn_idx']]
                    assert turn.role=='user' and evidence['span'] and evidence['span'] in turn.content
                    evidence_count+=1
            writes.append(dict(case_id=c.id,benchmark=c.benchmark,stored_units=len(snapshot['units']),
                               grounded_user_evidence=evidence_count,empty_bank=not snapshot['units']))
        write(out/'write_quality_audit.json',writes)
        errors=[]
        for r in rows:
            if r['status']=='ok' and r['benchmark']=='prefeval' and not r['evaluation']['metrics']['follows_preference']:
                errors.append(dict(case_id=r['case_id'],method=r['method'],trace=r['trace'],
                    evaluation=r['evaluation'],caveat='Recorded surrogate judge decisions; outside facts not independently verified.'))
        write(out/'prefeval_error_audit.json',errors)
        if phase=='diagnostic':
            old=[dict(read(f),method='legacy_v6') for f in (BASE/'m4_test/samples').glob('*.json') if read(f)['method']=='v6']
            old_index={r['case_id']:r for r in old}
            transitions=[]
            for c in cases:
                a=old_index[c.id];b=idx[(c.id,'v6_compact')]
                assert a['case_digest']==b['case_digest']
                transitions.append(dict(case_id=c.id,benchmark=c.benchmark,legacy_status=a['status'],compact_status=b['status']))
            write(out/'completion_transitions.json',transitions)
            combined=old+[dict(r,method='compact_v6') for r in rows if r['method']=='v6_compact']
            write(out/'comparisons_legacy.json',paired_comparisons(combined,'legacy_v6',plan['seed']))
            shared_old={r['case_id'] for r in old if r['status']=='ok'}&{r['case_id'] for r in rows if r['method']=='v6_compact' and r['status']=='ok'}
            write(out/'legacy_common_metrics.json',dict(case_ids=sorted(shared_old),metrics=aggregate([r for r in combined if r['case_id'] in shared_old],plan['seed'])))
        ledger=[]
        for invocation in read(out/'invocations.json'):
            ledger.extend(dict(role=role,tag=tag,**json.loads(value)) for role,tag,value in db.execute('SELECT role,tag,json FROM requests WHERE run=?',(invocation['run_id'],)))
        for r in ledger:
            if r.get('response_model'):assert r['response_model']==plan['models'][r['role']]['model']
        all_ledger.extend(ledger)
        usage=dict(http_attempts=len(ledger),known_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in ledger),
                   unknown_usage=sum(type(r.get('usage',{}).get('total_tokens')) is not int for r in ledger),
                   statuses=dict(Counter(str(r.get('http_status','unknown')) for r in ledger)),
                   response_models=dict(Counter(r.get('response_model','unknown') for r in ledger)),
                   schema_failures=dict(sum((Counter(read(out/'usage'/f"{i['run_id']}.json").get('schema_failures',{})) for i in read(out/'invocations.json')),Counter())))
        usage['tpm_throttles']=[read(out/'usage'/f"{i['run_id']}.json").get('tpm_throttle') for i in read(out/'invocations.json')]
        write(out/'ledger_totals.json',usage)
        verification[phase]=dict(planned=len(cases)*len(METHODS),checkpointed=len(rows),ok=sum(r['status']=='ok' for r in rows),common_cases=len(common),gate_audit_cases=len(audit))
    for path,digest in read(BASE/'m5_prior_samples.json').items():assert sha(ROOT/path)==digest,path
    saved=read(BASE/'m5_success_before_resume.json')
    for path,digest in saved.items():assert sha(ROOT/path)==digest,path
    assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before;db.close()
    write(BASE/'m5_pending.json',pending)
    write(BASE/'m5_execution_failures.json',[dict(p,stage=(p['error'].get('message','').split(':')[0]
          if isinstance(p['error'],dict) else 'not checkpointed')) for p in pending])
    write(BASE/'m5_verification.json',dict(phases=verification,prior_samples_unchanged=len(read(BASE/'m5_prior_samples.json')),
          successful_checkpoints_before_resume_unchanged=len(saved),
          frozen_code_and_models_verified=True,source_hashes_verified=True,offline_added_requests=0,
          fresh_test_groups_verified=not {c['group'] for c in plan['phases']['test']['cases']}&set(plan['previously_observed_groups'])))
    print(json.dumps(verification,indent=2));print('Pending',len(pending))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','analyze']);p.add_argument('--phase',choices=['diagnostic','test'],default='diagnostic')
    for name in ('workers','concurrency','max-calls','max-tokens','timeout'):p.add_argument('--'+name,type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare();return 0
    if args.action=='analyze':analyze();return 0
    return run(args)
if __name__=='__main__':raise SystemExit(main())
