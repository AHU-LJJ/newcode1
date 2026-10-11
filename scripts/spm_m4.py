"""Frozen Qwen-7B v4/v5/v6 model comparison on the same M3 test questions."""
import argparse
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
PLAN=BASE/'m4_plan.json'
QWEN='Qwen/Qwen2.5-7B-Instruct'
TOKENIZER='datasets/tokenizers/qwen2.5-7b/tokenizer.json'

def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return json.loads(Path(p).read_text())
def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')

def set_models(models=None):
    reference=read(BASE/'m3_plan.json')['models'] if models is None else models
    keys=dict(chat='CHAT_MODEL',cheap='CHEAP_MODEL',judge='JUDGE_MODEL',embed='EMBED_MODEL',bench_judge='PRES_JUDGE_MODEL')
    for role,prefix in keys.items():os.environ[prefix]=reference[role]['model']
    os.environ['CHAT_MODEL']=QWEN;os.environ['CHEAP_MODEL']=QWEN

def prepare():
    if PLAN.exists():raise ValueError('M4 already frozen; do not overwrite')
    sys.path.insert(0,str(ROOT))
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.config import endpoints,VARIANTS
    previous=read(BASE/'m3_plan.json')
    assert read(BASE/'m3_pending.json')==[]
    # Only transport, CLI and operational runner additions differ from M3.
    for name,digest in previous['code'].items():
        if name not in ('api.py','__main__.py','runner.py'):assert sha(ROOT/'spm'/name)==digest,name
    ids=[c['id'] for c in previous['phases']['test']['cases']]
    casefile='runs/spm/m4_test_cases.json';write(casefile,ids)
    argv=['run','--split','all','--seed',str(previous['seed']),'--limit','0','--case-file',casefile,
          '--methods','v4','v5','v6','--workers','12','--concurrency','4','--max-calls','2000',
          '--max-tokens','5000000','--timeout','180','--cache','runs/spm/api_cache_m4.sqlite',
          '--output','runs/spm/m4_test','--perma-types','2','--max-history-episodes','24',
          '--model-tpm',QWEN+'=80000','--tpm-tokenizer',TOKENIZER]
    args=parser().parse_args(argv);args.limit=None;cases,_=selected_cases(args)
    expected=previous['phases']['test']['cases']
    assert [(c.id,c.digest(),c.group) for c in cases]==[(c['id'],c['digest'],c['group']) for c in expected]
    freeze=BASE/'m4_frozen/spm';freeze.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,freeze/p.name)
    src=sqlite3.connect(BASE/'api_cache_m3.sqlite');dst=sqlite3.connect(BASE/'api_cache_m4.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    assert not (BASE/'m4_test/memory').exists(), 'Qwen must build its own memory banks'
    prior={str(f.relative_to(ROOT)):sha(f) for d in BASE.iterdir() if d.is_dir() and not d.name.startswith('m4')
           for f in (d/'samples').glob('*.json')}
    write(BASE/'m4_prior_samples.json',prior)
    set_models();eps=endpoints()
    assert eps['judge'].model=='Pro/deepseek-ai/DeepSeek-V3.2'
    for r,e in eps.items():assert e.base_url==previous['models'][r]['base_url'],r
    write(PLAN,dict(protocol='M4 fixed-question end-to-end small-model comparison; no fresh holdout claim',
        seed=previous['seed'],phases=dict(test=dict(argv=argv,methods=['v4','v5','v6'],cases=expected,
            case_file_sha256=sha(casefile))),
        code={p.name:sha(p) for p in freeze.glob('*.py')},
        models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        configs={m:VARIANTS[m].dump() for m in ('v4','v5','v6')},
        reference_plan_sha256=sha(BASE/'m3_plan.json'),tokenizer=dict(path=TOKENIZER,sha256=sha(TOKENIZER)),
        rate_limit=dict(advertised_tpm=80000,effective_tpm=72000,concurrency=4,workers=12,
            reservation='ChatML tokenizer count + max completion + 256; refund with known provider usage + 256; retain usage 60s after completion; unknown usage keeps full reservation.',
            scope='Shared provider/model rolling window persisted in this experiment cache; serialize invocations.'),
        tests_before_freeze='58 passed',
        cache='Isolated M3 cache backup. BGE-M3 embeddings and identical judge payloads may reuse validated cache. Qwen model payloads have distinct keys. No prior memory banks copied.',
        differences=['CHAT_MODEL and CHEAP_MODEL both Qwen: extraction, relations, analysis, gates, inference and generation.',
                     'Judge stays DeepSeek-V3.2, embeddings stay BGE-M3; same thresholds, prompts, source cases and raw evidence bounds as M3.',
                     'Only operational TPM admission code changed; no small-model-specific prompt/schema relaxation or fallback to a large writing model.'],
        cautions=['Same 59 M3 test questions, previously observed; paired model comparison, not a new holdout.',
                  'Qwen rebuilds the 39 conversational histories; BenchPreS has 20 given-preference questions and disables raw evidence.',
                  'PERMA only three short-history type2 users; BenchPreS ten seen identities; no full-benchmark claim.',
                  'Execution failures are missing, not wrong; report complete and common-success metrics separately.',
                  'No b1/b2 Qwen baselines in this requested v4/v5/v6 comparison; DeepSeek b1/b2 are not small-model baselines.']))
    print(json.dumps(dict(plan=str(PLAN),cases=len(cases),jobs=3*len(cases),effective_tpm=72000),indent=2))

def frozen_import():
    plan=read(PLAN)
    assert sha(BASE/'m3_plan.json')==plan['reference_plan_sha256']
    for name,digest in plan['code'].items():assert sha(BASE/'m4_frozen/spm'/name)==digest,name
    assert sha(plan['tokenizer']['path'])==plan['tokenizer']['sha256']
    for p in plan['phases'].values():
        file=p['argv'][p['argv'].index('--case-file')+1]
        assert sha(file)==p['case_file_sha256']
    sys.path.insert(0,str(BASE/'m4_frozen'))
    return plan

def run(args):
    plan=frozen_import();set_models(plan['models'])
    from spm.__main__ import main
    argv=plan['phases']['test']['argv'].copy()
    for name in ('workers','concurrency','max_calls','max_tokens','timeout'):
        value=getattr(args,name)
        if value is not None:argv[argv.index('--'+name.replace('_','-'))+1]=str(value)
    sys.argv=['spm',*argv]
    return main()

def analyze():
    plan=frozen_import()
    from spm.evaluation import aggregate,paired_comparisons,frontier
    from spm.__main__ import parser
    from spm.runner import selected_cases
    out=BASE/'m4_test';manifest=read(out/'manifest.json')['scientific']
    assert manifest['models']==plan['models'] and manifest['code']==plan['code']
    assert manifest['configs']==plan['configs']
    args=parser().parse_args(plan['phases']['test']['argv']);args.limit=None
    cases,_=selected_cases(args)
    expected=plan['phases']['test']['cases']
    assert [(c.id,c.digest(),c.group) for c in cases]==[(c['id'],c['digest'],c['group']) for c in expected]
    for p,digest in manifest['sources'].items():assert sha(p)==digest
    db=sqlite3.connect(BASE/'api_cache_m4.sqlite');before=db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]
    rows=[read(f) for f in sorted((out/'samples').glob('*.json'))]
    idx={(r['case_id'],r['method']):r for r in rows}
    assert len(idx)==len(rows)
    wanted={(c['id'],m) for c in expected for m in plan['configs']}
    assert set(idx)<=wanted
    pending=[]
    for c in expected:
        for m in plan['configs']:
            r=idx.get((c['id'],m))
            if r is None:pending.append(dict(case_id=c['id'],method=m,error='not checkpointed'));continue
            assert r['case_digest']==c['digest'] and r['group']==c['group']
            if r['status']!='ok':pending.append(dict(case_id=c['id'],method=m,error=r['error']))
    write(out/'metrics.json',aggregate(rows,plan['seed']))
    common_ids={c['id'] for c in expected if all(idx.get((c['id'],m),{}).get('status')=='ok' for m in plan['configs'])}
    common_rows=[r for r in rows if r['case_id'] in common_ids]
    write(out/'common_success_metrics.json',dict(case_ids=sorted(common_ids),
          metrics=aggregate(common_rows,plan['seed']),
          caveat='Same successful questions across all three Qwen methods. Selected after execution; missingness may be nonrandom.'))
    for baseline in ('v4','v5'):write(out/f'comparisons_vs_{baseline}.json',paired_comparisons(rows,baseline,plan['seed']))
    write(out/'frontier.json',frontier(rows))
    prior_rows=[read(f) for f in (BASE/'m3_test/samples').glob('*.json')]
    for method in plan['configs']:
        combined=[dict(r,method='qwen_'+method) for r in rows if r['method']==method]
        combined += [dict(r,method='deepseek_'+method) for r in prior_rows if r['method']==method]
        write(out/f'comparisons_model_{method}.json',paired_comparisons(combined,'deepseek_'+method,plan['seed']))
    audit=[];shared=0
    for c in expected:
        rs=[idx.get((c['id'],m)) for m in ('v4','v5','v6')]
        good=[r for r in rs if r and r['status']=='ok']
        if len(good)>1:
            assert len({r['trace']['memory_file'] for r in good})==1;shared+=1
        a,b=rs[1:]
        if a and b and a['status']==b['status']=='ok':
            assert a['trace']['decisions']==b['trace']['decisions']
            raw_chars=sum(len(e['dialogue_excerpt']) for e in b['trace']['raw_evidence'])
            assert raw_chars<=6000
            if c['benchmark']=='benchpres':assert raw_chars==0 and a['trace']['context']==b['trace']['context']
            audit.append(dict(case_id=c['id'],benchmark=c['benchmark'],same_gates=True,raw_evidence_chars=raw_chars,
                metrics_v5=a['evaluation']['metrics'],metrics_v6=b['evaluation']['metrics']))
    write(out/'gate_audit.json',audit)
    errors=[]
    for r in rows:
        if r['benchmark']=='prefeval' and r['status']=='ok' and not r['evaluation']['metrics']['follows_preference']:
            errors.append(dict(case_id=r['case_id'],method=r['method'],
                candidates=r['trace']['candidates'],decisions=r['trace']['decisions'],
                rendered_context=r['trace']['context'],response=r['evaluation']['response'],
                recorded_judge=r['evaluation']['judge'],
                other_method_metrics={m:idx[(r['case_id'],m)]['evaluation']['metrics']
                    for m in plan['configs'] if idx.get((r['case_id'],m),{}).get('status')=='ok'},
                caveat='Recorded surrogate judge reasons; external brands/works not independently checked.'))
    write(out/'error_audit.json',errors)
    ledger=[]
    for invocation in read(out/'invocations.json'):
        ledger.extend(dict(role=r,tag=t,**json.loads(j)) for r,t,j in db.execute(
            'SELECT role,tag,json FROM requests WHERE run=?',(invocation['run_id'],)))
    from collections import Counter
    for r in ledger:
        if r.get('response_model') is not None:
            assert r['response_model']==plan['models'][r['role']]['model'],r['role']
    usage=dict(invocations=len(read(out/'invocations.json')),http_attempts=len(ledger),
        known_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in ledger),
        unknown_usage=sum(type(r.get('usage',{}).get('total_tokens')) is not int for r in ledger),
        http_status=dict(Counter(str(r.get('http_status','unknown')) for r in ledger)),
        response_models=dict(Counter(r['response_model'] for r in ledger if r.get('response_model'))),
        by_role={role:dict(requests=sum(r['role']==role for r in ledger),
                          known_tokens=sum(r.get('usage',{}).get('total_tokens',0) for r in ledger if r['role']==role))
                 for role in sorted({r['role'] for r in ledger})})
    known_qwen=sorted((r['timestamp'],r['usage']['total_tokens']) for r in ledger
                     if r['role'] in ('chat','cheap') and type(r.get('usage',{}).get('total_tokens')) is int)
    from datetime import datetime
    window=[];peak=0
    for stamp,tokens in known_qwen:
        ts=datetime.fromisoformat(stamp).timestamp()
        window=[(t,n) for t,n in window if ts-t<60];window.append((ts,tokens))
        peak=max(peak,sum(n for t,n in window))
    usage['observed_known_qwen_peak_60s_tokens']=peak
    usage['rate_audit_caveat']='Measured provider tokens by completion timestamp; excludes unknown-usage attempts, tokenizer headroom and external same-account calls.'
    schema_failures=Counter();throttles=[]
    for invocation in read(out/'invocations.json'):
        saved=out/'usage'/f"{invocation['run_id']}.json"
        if saved.exists():
            u=read(saved);schema_failures.update(u.get('schema_failures') or {})
            if u.get('tpm_throttle'):throttles.append(u['tpm_throttle'])
    usage['schema_failures']=dict(schema_failures)
    usage['tpm_throttles']=throttles
    write(out/'ledger_totals.json',usage)
    assert db.execute('SELECT COUNT(*) FROM requests').fetchone()[0]==before;db.close()
    old=read(BASE/'m4_prior_samples.json')
    for f,digest in old.items():assert sha(ROOT/f)==digest
    verify=dict(planned=len(wanted),checkpointed=len(rows),ok=sum(r['status']=='ok' for r in rows),
                frozen_code_verified=True,same_m3_questions_verified=True,models_verified=True,
                provider_response_models_verified=True,
                fresh_holdout=False,source_hashes_verified=True,shared_write_cases=shared,
                raw_gate_audit_cases=len(audit),prior_sample_files_unchanged=len(old),offline_analysis_added_requests=0)
    verify['common_success_cases']=len(common_ids)
    execution_failures=[]
    for p in pending:
        message=p['error'].get('message','') if isinstance(p['error'],dict) else str(p['error'])
        prefix=message.split(':',1)[0]
        stage=prefix if prefix in ('extract','relations_batch','relation','normalize','analyze','gate','infer','generate','mcq','pref_judge','cupid_judge') else 'other'
        execution_failures.append(dict(**p,stage=stage))
    write(out/'execution_failure_audit.json',execution_failures)
    preserved=BASE/'m4_success_before_resume.json'
    if preserved.exists():
        hashes=read(preserved)
        for f,digest in hashes.items():assert sha(ROOT/f)==digest
        verify['successful_checkpoints_before_resume_unchanged']=len(hashes)
    write(BASE/'m4_pending.json',pending);write(BASE/'m4_verification.json',verify)
    print(json.dumps(verify,indent=2));print('Pending:',len(pending))

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','analyze'])
    for name in ('workers','concurrency','max-calls','max-tokens','timeout'):p.add_argument('--'+name,type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare();return 0
    if args.action=='analyze':analyze();return 0
    return run(args)

if __name__=='__main__':raise SystemExit(main())
