"""Freeze, run and audit M2: an answer-filter gate on previously unseen groups."""
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
PLAN=BASE/'m2_plan.json'


def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()


def write(p,v):
    Path(p).parent.mkdir(parents=True,exist_ok=True)
    Path(p).write_text(json.dumps(v,ensure_ascii=False,indent=2,allow_nan=False)+'\n')


def prepare():
    if PLAN.exists():raise ValueError('M2 already frozen; do not overwrite its plan')
    sys.path.insert(0,str(ROOT))
    from spm.adapters import load_cases,select,describe
    from spm.__main__ import parser
    from spm.runner import selected_cases
    from spm.config import VARIANTS,SPMConfig,endpoints
    old=json.loads((BASE/'m1_plan.json').read_text())
    seen={c['group'] for phase in old['phases'].values() for c in phase['cases']}
    for f in (BASE/'m0_final/samples').glob('*.json'):seen.add(json.loads(f.read_text())['group'])
    test=[];pools={}
    seed=20261010
    for b in ('prefeval','cupid','benchpres','perma'):
        cases=load_cases('datasets',b,'sd_clean',[2] if b=='perma' else [1,2,3])
        eligible=[c for c in cases if c.group not in seen and (b!='perma' or len(c.public.episodes)<=24)]
        pools[b]=describe(eligible)
        count=3 if b=='perma' else 12
        chosen=[];groups=set()
        for c in select(eligible,None,seed):
            if b!='benchpres' and c.group in groups:continue
            chosen.append(c);groups.add(c.group)
            if len(chosen)==count:break
        if len(chosen)!=count:raise ValueError(f'Insufficient unseen {b} cases')
        if b=='benchpres' and len(groups)!=len({c.group for c in eligible}):
            raise ValueError('BenchPreS sample must cover both remaining identities')
        test+=chosen
    phases={}
    for phase,ids,methods in (
        ('dev',[c['id'] for c in old['phases']['dev']['cases']],['v4','v5','v3','b1','b2']),
        ('test',[c.id for c in test],['v4','v5','b1','b2'])):
        casefile=f'runs/spm/m2_{phase}_cases.json';write(casefile,ids)
        argv=['run','--split','all','--seed',str(seed),'--limit','0','--case-file',casefile,
              '--methods',*methods,'--workers','48','--concurrency','16','--max-calls','1500',
              '--max-tokens','3000000','--timeout','180','--cache','runs/spm/api_cache_m2.sqlite',
              '--output',f'runs/spm/m2_{phase}']
        if phase=='test':argv+=['--perma-types','2','--max-history-episodes','24']
        args=parser().parse_args(argv);args.limit=None
        cases,_=selected_cases(args)
        phases[phase]=dict(argv=argv,methods=methods,case_file_sha256=sha(casefile),
            cases=[dict(id=c.id,group=c.group,digest=c.digest(),benchmark=c.benchmark,
                        stratum=c.stratum,episodes=len(c.public.episodes)) for c in cases])
    groups={c.group for c in test}
    assert not groups&seen
    freeze=BASE/'m2_frozen/spm';freeze.mkdir(parents=True,exist_ok=True)
    for p in (ROOT/'spm').glob('*.py'):shutil.copy2(p,freeze/p.name)
    src=sqlite3.connect(BASE/'api_cache.sqlite');dst=sqlite3.connect(BASE/'api_cache_m2.sqlite')
    try:src.backup(dst)
    finally:src.close();dst.close()
    shutil.copytree(BASE/'m1_dev/memory',BASE/'m2_dev/memory')
    os.environ['JUDGE_MODEL']='Pro/deepseek-ai/DeepSeek-V3.2'
    eps=endpoints()
    write(PLAN,dict(protocol='M2 predeclared answer-filter gate experiment; fresh grouped test',
        seed=seed,phases=phases,code={p.name:sha(p) for p in freeze.glob('*.py')},
        models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        configs={m:VARIANTS.get(m,SPMConfig(use_anticipation=False,use_gate=False,embed_raw='none')).dump()
                 for m in ('v4','v5','v3','b1','b2')},
        previously_observed_groups=sorted(seen),fresh_test_groups=sorted(groups),public_eligible_pools=pools,
        hypothesis='Within supported scope/conditions, applicability is whether a preference changes or filters a good answer. Acceptable alternatives do not imply suppression.',
        changes=['v5 adds a generic answer-filter definition to the original gate prompt; v4 remains legacy.',
                 'Schema repair feedback includes short error messages; strict judge validation remains.',
                 'Exact case-list selection recorded in manifests.'],
        cache='SQLite backup of validated M1 cache; M2 writes an isolated copy. Dev reuses original write banks.',
        cautions=['Development reuses previously observed M1 development cases; the hypothesis was informed by M1 including its old test.',
                  'Fresh test excludes every M0/M1 observed identity/problem group; selection uses public strata/history length only.',
                  'BenchPreS fresh test has only two remaining identities; PERMA is three short-history type2 cases.',
                  'Generator and judge are the same model; API surrogate metrics are not official full benchmarks.',
                  'No RL and no threshold tuning: apply=.5 suppress=.2.']))
    print(json.dumps(dict(plan=str(PLAN),jobs={p:len(v['cases'])*len(v['methods']) for p,v in phases.items()},
                         fresh_test_groups=len(groups),eligible=pools),ensure_ascii=False,indent=2))


def frozen_import():
    plan=json.loads(PLAN.read_text())
    for name,digest in plan['code'].items():
        assert sha(BASE/'m2_frozen/spm'/name)==digest,name
    for phase in plan['phases'].values():
        casefile=phase['argv'][phase['argv'].index('--case-file')+1]
        assert sha(casefile)==phase['case_file_sha256']
    sys.path.insert(0,str(BASE/'m2_frozen'))
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
    assert gate_literal(BASE/'m1_frozen/spm/prompts.py')==gate_literal(BASE/'m2_frozen/spm/prompts.py')
    verify['legacy_gate_prompt_identical']=True
    assert not set(plan['fresh_test_groups'])&set(plan['previously_observed_groups'])
    observed=set()
    for directory in BASE.iterdir():
        if directory.is_dir() and not directory.name.startswith('m2'):
            for f in (directory/'samples').glob('*.json'):
                observed.add(json.loads(f.read_text())['group'])
    assert not observed&set(plan['fresh_test_groups'])
    verify['all_prior_sample_groups_checked']=len(observed)
    db=sqlite3.connect(BASE/'api_cache_m2.sqlite')
    before=db.execute('SELECT count(*) FROM requests').fetchone()[0]
    pending=[]
    for phase,p in plan['phases'].items():
        out=BASE/f'm2_{phase}'
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
            variants=[indexed[(c['id'],m)] for m in ('v4','v5','v3') if (c['id'],m) in indexed and indexed[(c['id'],m)]['status']=='ok']
            if len(variants)>1:
                assert len({r['trace']['memory_file'] for r in variants})==1;shared+=1
        write(out/'metrics.json',aggregate(rows,plan['seed']))
        for baseline in ('b1','b2','v4'):
            write(out/f'comparisons_vs_{baseline}.json',paired_comparisons(rows,baseline,plan['seed']))
        write(out/'frontier.json',frontier(rows))
        audit=[]
        for c in p['cases']:
            pair=[indexed.get((c['id'],m)) for m in ('v4','v5')]
            if not all(r and r['status']=='ok' for r in pair):continue
            a,b=pair
            da={d['unit_id']:d for d in a['trace']['decisions']}
            dbb={d['unit_id']:d for d in b['trace']['decisions']}
            assert set(da)==set(dbb), 'Gate comparison must use identical candidates'
            changes=[dict(unit_id=k,old=da[k],new=dbb[k]) for k in da if da[k]['decision']!=dbb[k]['decision']]
            audit.append(dict(case_id=c['id'],benchmark=c['benchmark'],group=c['group'],
                candidate_ids=list(da),changed_decisions=changes,
                same_rendered_context=a['trace']['context']==b['trace']['context'],
                metrics_v4=a['evaluation']['metrics'],metrics_v5=b['evaluation']['metrics'],
                recorded_judge_v4=a['evaluation'].get('judge'),recorded_judge_v5=b['evaluation'].get('judge'),
                caveat='Recorded surrogate scores, not an independent fact check.'))
        write(out/'gate_audit.json',audit)
        errors=[]
        for r in rows:
            if r['benchmark']=='prefeval' and r['method'] in ('v4','v5') and r['status']=='ok' and not r['evaluation']['metrics']['follows_preference']:
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
    write(BASE/'m2_pending.json',pending);write(BASE/'m2_verification.json',verify)
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
