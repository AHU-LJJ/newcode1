"""Serial M7 stages with one bounded failed-only resume and offline validation."""
import json
import subprocess
import sys
import spm_m7 as exp


def invoke(arguments,log):
    with open(log,'a') as output:
        result=subprocess.run([sys.executable,*arguments],cwd=exp.ROOT,stdout=output,stderr=subprocess.STDOUT)
    return result.returncode


def finish(phase):
    out=exp.BASE/f'm7_{phase}';plan=exp.read(exp.BASE/f'm7_{phase}_plan.json')
    rows=[exp.read(f) for f in (out/'samples').glob('*.json')]
    assert len(rows)==len(plan['cases'])*len(plan['methods'])
    pending=[r for r in rows if r['status']!='ok']
    billing=[r for r in pending if r['error']['type']=='AccountUnavailable']
    marker=out/'resume_authorization.json'
    if pending and not billing and not marker.exists():
        print(json.dumps(dict(phase=phase,action='bounded failed-only resume',failed=len(pending))),flush=True)
        status=invoke(['scripts/spm_m7.py','run','--phase',phase,'--resume'],exp.BASE/f'm7_{phase}_resume.log')
        if status not in (0,1):return False
        rows=[exp.read(f) for f in (out/'samples').glob('*.json')]
        billing=[r for r in rows if r['status']!='ok' and r['error']['type']=='AccountUnavailable']
    for script in ('spm_m7.py','spm_m7_replay.py'):
        args=[f'scripts/{script}',*(['analyze'] if script=='spm_m7.py' else []),'--phase',phase]
        if invoke(args,exp.BASE/f'm7_{phase}_verification.log')!=0:
            print(json.dumps(dict(phase=phase,blocked='offline verification failed',script=script)),flush=True)
            return False
    if invoke(['scripts/spm_m7_report.py'],exp.BASE/'m7_report.log')!=0:return False
    print(json.dumps(dict(phase=phase,action='verified',verification=exp.read(out/'verification.json'))),flush=True)
    if billing:
        print(json.dumps(dict(phase=phase,blocked='API account unavailable',affected=len(billing))),flush=True)
        return False
    return True


def main():
    for phase in ('dev','test'):
        out=exp.BASE/f'm7_{phase}'
        if (out/'verification.json').exists() and (out/'replay_verification.json').exists():continue
        print(json.dumps(dict(phase=phase,action='start frozen three-method comparison')),flush=True)
        status=invoke(['scripts/spm_m7.py','run','--phase',phase],exp.BASE/f'm7_{phase}.log')
        if status not in (0,1):return status
        if not finish(phase):return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
