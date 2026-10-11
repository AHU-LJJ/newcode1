"""Check successful checkpoint replay with networking replaced by a failing mock."""
import asyncio
import json
import httpx
import spm_m3

plan=spm_m3.frozen_import()
from spm.api import Gateway
from spm.config import SPMConfig,endpoints
from spm.__main__ import parser
from spm.runner import Experiment,selected_cases

async def main():
    phases={}
    spm_m3.os.environ['JUDGE_MODEL']='Pro/deepseek-ai/DeepSeek-V3.2'
    for phase,p in plan['phases'].items():
        out=spm_m3.BASE/f'm3_{phase}'
        files=list((out/'samples').glob('*.json'))
        before={str(f):spm_m3.sha(f) for f in files}
        rows=[json.loads(f.read_text()) for f in files]
        args=parser().parse_args(p['argv']);args.limit=None
        cases,_=selected_cases(args);lookup={c.id:c for c in cases}
        api=Gateway(endpoints(),spm_m3.BASE/'api_cache_m3.sqlite')
        await api.http.aclose()
        def offline(request):raise AssertionError('Checkpoint replay must not send any network request')
        api.http=httpx.AsyncClient(transport=httpx.MockTransport(offline))
        exp=Experiment(api,out,{m:SPMConfig(**plan['configs'][m]) for m in p['methods']})
        for r in rows:
            if r['status']=='ok':await exp.run_case(lookup[r['case_id']],r['method'])
        usage=await api.close()
        assert usage['http_requests']==0
        assert before=={str(f):spm_m3.sha(f) for f in files}
        phases[phase]=dict(successes_replayed=len(exp.rows),failed_checkpoints_excluded=sum(r['status']!='ok' for r in rows),
                           sample_files_unchanged=len(files),http_requests=0)
    spm_m3.write(spm_m3.BASE/'m3_replay_verification.json',dict(phases=phases,
        caveat='Verifies successful checkpoint skipping, not new model generation or independent rescoring. Network replaced by a failing mock.'))
    print(json.dumps(phases,indent=2))

asyncio.run(main())
