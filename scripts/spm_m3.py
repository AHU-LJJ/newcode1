"""Freeze, run and audit M3: an answer-filter gate on previously unseen groups."""
import argparse
import ast
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
PLAN=BASE/'m3_plan.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare():
    if PLAN.exists():raise ValueError('M3 already frozen; do not overwrite its plan')
    sys.path.insert(0,str(ROOT))
    from collections import Counter
    from spm.adapters import load_cases,select,describe
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.config import VARIANTS,SPMConfig,endpoints
    observed={};seen=set();seen_ids=set()
    for directory in BASE.iterdir():
        if directory.is_dir() and not directory.name.startswith('m3'):
            for f in (directory/'samples').glob('*.json'):
                row=json.loads(f.read_text());seen.add(row['group']);seen_ids.add(row['case_id'])
                observed[str(f.relative_to(ROOT))]=sha(f)
    write(BASE/'m3_prior_samples.json',observed)
    seed=20261011;test=[];pools={}
    for b in ('prefeval','cupid','benchpres','perma'):
        all_cases=load_cases('datasets',b,'sd_clean',[2] if b=='perma' else [1,2,3])
        eligible=[c for c in all_cases if (c.id not in seen_ids if b=='benchpres' else c.group not in seen)
                  and (b!='perma' or len(c.public.episodes)<=24)]
        pools[b]=describe(eligible)
        count={'prefeval':24,'cupid':12,'benchpres':20,'perma':3}[b]
        picked=[];group_counts=Counter()
        for c in select(eligible,None,seed):
            if group_counts[c.group]>=(2 if b=='benchpres' else 1):continue
            picked.append(c);group_counts[c.group]+=1
            if len(picked)==count:break
        if len(picked)!=count:raise ValueError(f'Insufficient eligible {b} cases')
        if b=='benchpres':assert len(group_counts)==10
        test+=picked
    old=json.loads((BASE/'m2_plan.json').read_text())
    dev_ids=[c['id'] for c in old['phases']['dev']['cases']]+['cupid/506']
    phases={}
    for phase,ids,methods in (
        ('dev',dev_ids,['v5','v6']),('test',[c.id for c in test],['v4','v5','v6','b1','b2'])):
        casefile=f'runs/spm/m3_{phase}_cases.json';write(casefile,ids)
        argv=['run','--split','all','--seed',str(seed),'--limit','0','--case-file',casefile,
              '--methods',*methods,'--workers','64','--concurrency',('16' if phase=='dev' else '32'),
              '--max-calls',('1000' if phase=='dev' else '2500'),
              '--max-tokens',('1500000' if phase=='dev' else '5000000'),'--timeout','180',
              '--cache','runs/spm/api_cache_m3.sqlite','--output',f'runs/spm/m3_{phase}']
        if phase=='test':argv+=['--perma-types','2','--max-history-episodes','24']
        args=parser().parse_args(argv);args.limit=None;cases,_=selected_cases(args)
        phases[phase]=dict(argv=argv,methods=methods,case_file_sha256=sha(casefile),
            cases=[dict(id=c.id,group=c.group,digest=c.digest(),benchmark=c.benchmark,
                        stratum=c.stratum,episodes=len(c.public.episodes)) for c in cases])
    fresh={c.group for c in test if c.benchmark!='benchpres'}
    assert not fresh&seen and not {c.id for c in test}&seen_ids
    freeze=BASE/'m3_frozen/spm';freeze.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,freeze/p.name)
    src=sqlite3.connect(BASE/'api_cache_m2.sqlite');dst=sqlite3.connect(BASE/'api_cache_m3.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    dest=BASE/'m3_dev/memory';dest.mkdir(parents=True,exist_ok=True);index={}
    for phase in ('dev','test'):
        source=BASE/f'm2_{phase}/memory'
        index.update(json.loads((source/'index.json').read_text()))
        for p in source.iterdir():
            if p.name!='index.json' and p.is_file():shutil.copy2(p,dest/p.name)
    write(dest/'index.json',index)
    os.environ['JUDGE_MODEL']='Pro/deepseek-ai/DeepSeek-V3.2';eps=endpoints()
    write(PLAN,dict(protocol='M3 predeclared new-gate replication and bounded raw-evidence ablation',
        seed=seed,phases=phases,code={p.name:sha(p) for p in freeze.glob('*.py')},
        models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        configs={m:VARIANTS.get(m,SPMConfig(use_anticipation=False,use_gate=False,embed_raw='none')).dump()
                 for m in ('v4','v5','v6','b1','b2')},
        previously_observed_groups=sorted(seen),previously_observed_ids=sorted(seen_ids),
        fresh_test_groups=sorted(fresh),public_eligible_pools=pools,
        user_selected_default='v5',tests_before_freeze='53 passed',
        hypothesis='v5 replicates answer-filter gating; adding at most 6000 characters of up to three retrieved historical dialogue excerpts (v6) may recover extraction/scope omissions.',
        changes=['CLI and SPMConfig default to v5; explicit v4 retains the original gate prompt.',
                 'v6 appends raw evidence without extra LLM calls or changed gate decisions; given-preference BenchPreS disables raw evidence.',
                 'Grounding repair includes short cited USER source text or matching USER indices; no automatic span rewrite, same strict grounding.',
                 'No threshold/model changes or reinforcement learning.'],
        cache='Isolated backup of M2 API cache. Dev copies completed M2 banks; new test builds its own banks.',
        cautions=['Dev reuses 39 observed development cases plus formerly failed cupid/506 as a diagnostic case.',
                  'PrefEval/CUPID/PERMA test groups exclude every previously observed group; no test ID was previously used.',
                  'BenchPreS has no unused identities; it is separately reported as new question IDs from ten previously seen profiles, two questions/profile.',
                  'PERMA test is three short-history sd_clean type2 users (public history cap 24), not the full interactive benchmark.',
                  'Same-model judge and API surrogate metrics; BenchPreS cannot test raw-evidence leakage because it has no raw dialogue.',
                  'New extraction repair can affect failed writes; existing validated cache and completed dev banks are reused.']))
    print(json.dumps(dict(plan=str(PLAN),jobs={p:len(v['cases'])*len(v['methods']) for p,v in phases.items()},
                         fresh_non_bench_groups=len(fresh),eligible=pools),ensure_ascii=False,indent=2))


def frozen_import():
    plan=json.loads(PLAN.read_text())
    for name,digest in plan['code'].items():
        assert sha(BASE/'m3_frozen/spm'/name)==digest,name
    for phase in plan['phases'].values():
        casefile=phase['argv'][phase['argv'].index('--case-file')+1]
        assert sha(casefile)==phase['case_file_sha256']
    sys.path.insert(0,str(BASE/'m3_frozen'))
    return plan


def run(args):
    plan=frozen_import()
    os.environ['JUDGE_MODEL']='Pro/deepseek-ai/DeepSeek-V3.2'
    from spm.__main__ import main
    argv=plan['phases'][args.phase]['argv'].copy()
    for name in ('workers','concurrency','max_calls','max_tokens','timeout'):
        value=getattr(args,name)
        if value is not None:argv[argv.index('--'+name.replace('_','-'))+1]=str(value)
    sys.argv=['spm',*argv]
    return main()


def analyze():
    plan=frozen_import()
    from spm.evaluation import aggregate,paired_comparisons,frontier
    from spm.runner import selected_cases
    from spm.__main__ import parser
    verify=dict(frozen_code_verified=True,fresh_test_groups_disjoint=True,phases={})
    def gate_literal(path):
        tree=ast.parse(path.read_text())
        return next(ast.literal_eval(n.value) for n in tree.body if isinstance(n,ast.Assign)
                    and any(isinstance(t,ast.Name) and t.id=='GATE' for t in n.targets))
    assert gate_literal(BASE/'m1_frozen/spm/prompts.py')==gate_literal(BASE/'m3_frozen/spm/prompts.py')
    verify['legacy_gate_prompt_identical']=True
    assert not set(plan['fresh_test_groups'])&set(plan['previously_observed_groups'])
    observed=set()
    for directory in BASE.iterdir():
        if directory.is_dir() and not directory.name.startswith('m3'):
            for f in (directory/'samples').glob('*.json'):
                observed.add(json.loads(f.read_text())['group'])
    assert not observed&set(plan['fresh_test_groups'])
    verify['all_prior_sample_groups_checked']=len(observed)
    db=sqlite3.connect(BASE/'api_cache_m3.sqlite')
    before=db.execute('SELECT count(*) FROM requests').fetchone()[0]
    pending=[]
    for phase,p in plan['phases'].items():
        out=BASE/f'm3_{phase}'
        if not (out/'manifest.json').exists():continue
        manifest=json.loads((out/'manifest.json').read_text())['scientific']
        assert manifest['code']==plan['code'] and manifest['models']==plan['models']
        assert manifest['configs']=={m:plan['configs'][m] for m in p['methods']}
        args=parser().parse_args(p['argv']);args.limit=None
        actual,_=selected_cases(args)
        assert [(c.id,c.digest(),c.group) for c in actual]==[(c['id'],c['digest'],c['group']) for c in p['cases']]
        rows=[json.loads(f.read_text()) for f in sorted((out/'samples').glob('*.json'))]
        indexed={(r['case_id'],r['method']):r for r in rows}
        expected={(c['id'],m) for c in p['cases'] for m in p['methods']}
        assert set(indexed)<=expected and len(rows)==len(indexed)
        for c in p['cases']:
            for m in p['methods']:
                r=indexed.get((c['id'],m))
                if r is None:pending.append(dict(phase=phase,case_id=c['id'],method=m,error='not yet checkpointed'));continue
                assert r['case_digest']==c['digest'] and r['group']==c['group']
                if r['status']!='ok':pending.append(dict(phase=phase,case_id=c['id'],method=m,error=r['error']))
        shared=0
        for c in p['cases']:
            variants=[indexed[(c['id'],m)] for m in ('v4','v5','v6') if (c['id'],m) in indexed and indexed[(c['id'],m)]['status']=='ok']
            if len(variants)>1:
                assert len({r['trace']['memory_file'] for r in variants})==1;shared+=1
        write(out/'metrics.json',aggregate(rows,plan['seed']))
        for baseline in ('b1','b2','v4','v5'):
            write(out/f'comparisons_vs_{baseline}.json',paired_comparisons(rows,baseline,plan['seed']))
        write(out/'frontier.json',frontier(rows))
        audit=[]
        for c in p['cases']:
            pair=[indexed.get((c['id'],m)) for m in ('v5','v6')]
            if not all(r and r['status']=='ok' for r in pair):continue
            a,b=pair
            da={d['unit_id']:d for d in a['trace']['decisions']}
            dbb={d['unit_id']:d for d in b['trace']['decisions']}
            assert da==dbb, 'Raw-evidence ablation must leave gate decisions unchanged'
            changes=[dict(unit_id=k,old=da[k],new=dbb[k]) for k in da if da[k]['decision']!=dbb[k]['decision']]
            audit.append(dict(case_id=c['id'],benchmark=c['benchmark'],group=c['group'],
                candidate_ids=list(da),changed_decisions=changes,
                same_rendered_context=a['trace']['context']==b['trace']['context'],
                metrics_v5=a['evaluation']['metrics'],metrics_v6=b['evaluation']['metrics'],
                raw_evidence_chars=sum(len(e['dialogue_excerpt']) for e in b['trace']['raw_evidence']),
                recorded_judge_v5=a['evaluation'].get('judge'),recorded_judge_v6=b['evaluation'].get('judge'),
                caveat='Recorded surrogate scores, not an independent fact check.'))
        write(out/'gate_audit.json',audit)
        errors=[]
        for r in rows:
            if r['benchmark']=='prefeval' and r['method'] in ('v4','v5','v6') and r['status']=='ok' and not r['evaluation']['metrics']['follows_preference']:
                errors.append(dict(case_id=r['case_id'],method=r['method'],
                    candidates=r['trace']['candidates'],decisions=r['trace']['decisions'],
                    rendered_context=r['trace']['context'],response=r['evaluation']['response'],
                    recorded_judge=r['evaluation']['judge'],
                    baseline_success={m:indexed[(r['case_id'],m)]['evaluation']['metrics']['follows_preference']
                        for m in ('b1','b2') if indexed.get((r['case_id'],m),{}).get('status')=='ok'},
                    caveat='Recorded judge rationale is not independently verified external knowledge.'))
        write(out/'error_audit.json',errors)
        verify['phases'][phase]=dict(planned=len(expected),checkpointed=len(rows),ok=sum(r['status']=='ok' for r in rows),shared_write_cases=shared)
        # Recover usage from persistent ledger, including cancelled invocations.
        invocations=json.loads((out/'invocations.json').read_text())
        ledger=[]
        for invocation in invocations:
            records=[json.loads(x[0]) for x in db.execute('SELECT json FROM requests WHERE run=?',(invocation['run_id'],))]
            ledger.extend(records)
        write(out/'ledger_totals.json',dict(invocations=len(invocations),http_attempts=len(ledger),
             known_tokens=sum(x.get('usage',{}).get('total_tokens',0) for x in ledger),
             unknown_usage=sum(type(x.get('usage',{}).get('total_tokens')) is not int for x in ledger)))
    assert db.execute('SELECT count(*) FROM requests').fetchone()[0]==before
    db.close();verify['offline_analysis_added_requests']=0
    for f,d in json.loads((BASE/'m1_original_before_rejudge.json').read_text()).items():assert sha(ROOT/f)==d
    verify['original_m1_files_unchanged']=True
    prior=json.loads((BASE/'m3_prior_samples.json').read_text())
    for f,d in prior.items():assert sha(ROOT/f)==d
    verify['prior_sample_files_unchanged']=len(prior)
    resumed=BASE/'m3_success_before_resume.json'
    if resumed.exists():
        preserved=json.loads(resumed.read_text())
        for f,d in preserved.items():assert sha(ROOT/f)==d
        verify['successful_checkpoints_before_resume_unchanged']=len(preserved)
    assert not {c['id'] for c in plan['phases']['test']['cases']}&set(plan['previously_observed_ids'])
    verify['all_new_test_question_ids']=True
    verify['benchpres_test']='question holdout from ten seen identities; no identity generalization claim'
    write(BASE/'m3_pending.json',pending);write(BASE/'m3_verification.json',verify)
    print(json.dumps(verify,indent=2));print('Pending:',len(pending))


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('action',choices=['prepare','run','analyze'])
    p.add_argument('--phase',choices=['dev','test'])
    for name in ('workers','concurrency','max-calls','max-tokens','timeout'):p.add_argument('--'+name,type=int)
    args=p.parse_args()
    if args.action=='prepare':prepare();return 0
    if args.action=='analyze':analyze();return 0
    if not args.phase:p.error('--phase required for run')
    return run(args)

if __name__=='__main__':raise SystemExit(main())
