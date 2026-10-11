"""Serial, bounded M6 continuation after the first development invocation finishes."""
import json
from pathlib import Path
import subprocess
import sys
import spm_m6 as exp


def invoke(arguments,log):
    with open(log,'a') as output:
        result=subprocess.run([sys.executable,*arguments],cwd=exp.ROOT,stdout=output,stderr=subprocess.STDOUT)
    return result.returncode


def finish(phase):
    out=exp.BASE/f'm6_{phase}'
    plan=exp.read(exp.BASE/f'm6_{phase}_plan.json')
    rows=[exp.read(f) for f in (out/'samples').glob('*.json')]
    assert len(rows)==len(plan['cases'])*len(plan['methods']),f'{phase}: missing planned checkpoints'
    pending=[r for r in rows if r['status']!='ok']
    billing=[r for r in pending if r['error']['type']=='AccountUnavailable']
    marker=out/'resume_authorization.json'
    if pending and not billing and not marker.exists():
        print(json.dumps(dict(phase=phase,action='bounded failed-only resume',failed=len(pending))),flush=True)
        status=invoke(['scripts/spm_m6.py','run','--phase',phase,'--resume'],exp.BASE/f'm6_{phase}_resume.log')
        if status not in (0,1):return False
        updated=[exp.read(f) for f in (out/'samples').glob('*.json')]
        billing=[r for r in updated if r['status']!='ok' and r['error']['type']=='AccountUnavailable']
    for script in ('spm_m6.py','spm_m6_replay.py'):
        arguments=[f'scripts/{script}',*(['analyze'] if script=='spm_m6.py' else []),'--phase',phase]
        if invoke(arguments,exp.BASE/f'm6_{phase}_verification.log')!=0:
            print(json.dumps(dict(phase=phase,blocked='offline verification failed',script=script)),flush=True)
            return False
    if invoke(['scripts/spm_m6_report.py'],exp.BASE/'m6_report.log')!=0:return False
    print(json.dumps(dict(phase=phase,action='verified',verification=exp.read(out/'verification.json'))),flush=True)
    if billing:
        print(json.dumps(dict(phase=phase,blocked='API account unavailable',affected=len(billing))),flush=True)
        return False
    return True


def main():
    dev=exp.BASE/'m6_dev'
    if not ((dev/'verification.json').exists() and (dev/'replay_verification.json').exists()):
        if not finish('dev'):return 1
    else:
        print(json.dumps(dict(phase='dev',action='use completed offline verification',verification=exp.read(dev/'verification.json'))),flush=True)
    if not (exp.BASE/'m6_test_plan.json').exists():
        if invoke(['scripts/spm_m6.py','prepare-test'],exp.BASE/'m6_freeze_test.log')!=0:return 1
    out=exp.BASE/'m6_test'
    if not (out/'verification.json').exists():
        print(json.dumps(dict(phase='test',action='start frozen 72-case four-method comparison')),flush=True)
        status=invoke(['scripts/spm_m6.py','run','--phase','test'],exp.BASE/'m6_test.log')
        if status not in (0,1):return status
        if not finish('test'):return 1
    return 0


if __name__=='__main__':raise SystemExit(main())
