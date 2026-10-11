"""Verify successful checkpoints skip networking and preserve sample bytes."""
import asyncio
import json
import httpx
import spm_m5
plan=spm_m5.frozen_import();spm_m5.set_models(plan['models'])
from spm.api import Gateway
from spm.config import SPMConfig,endpoints
from spm.__main__ import parser
from spm.runner import Experiment,selected_cases
from spm.pipeline import SPM
from spm.store import Store
async def main():
    result={}
    for phase,p in plan['phases'].items():
        out=spm_m5.BASE/f'm5_{phase}';files=list((out/'samples').glob('*.json'));before={str(f):spm_m5.sha(f) for f in files}
        rows=[spm_m5.read(f) for f in files]
        args=parser().parse_args(p['argv']);args.limit=None;cases,_=selected_cases(args);lookup={c.id:c for c in cases}
        api=Gateway(endpoints(),spm_m5.BASE/'api_cache_m5.sqlite')
        await api.http.aclose()
        def offline(request):raise AssertionError('Checkpoint replay must send no request')
        api.http=httpx.AsyncClient(transport=httpx.MockTransport(offline))
        async def no_request(*args,**kwargs):raise AssertionError('Offline replay requires a validated cache hit')
        api._request=no_request
        exp=Experiment(api,out,{m:SPMConfig(**plan['configs'][m]) for m in p['methods']})
        for r in rows:
            if r['status']=='ok':await exp.run_case(lookup[r['case_id']],r['method'])
        reconstructed=0
        for r in rows:
            if r['status']!='ok':continue
            case=lookup[r['case_id']];store=Store(out/r['trace']['memory_file'],case.public.user_id)
            try:
                before_store=store.snapshot()
                memory=SPM(case.public.user_id,store,api,SPMConfig(**plan['configs'][r['method']]))
                context,trace=await memory.query(case.public.query,force_all=case.public.given)
                assert context==r['trace']['context']
                assert all(value==r['trace'][key] for key,value in trace.items())
                assert store.snapshot()==before_store
                reconstructed+=1
            finally:store.close()
        usage=await api.close();assert usage['http_requests']==0
        assert before=={str(f):spm_m5.sha(f) for f in files}
        result[phase]=dict(successes_replayed=len(exp.rows),contexts_reconstructed=reconstructed,
            failed_excluded=sum(r['status']!='ok' for r in rows),sample_files_unchanged=len(files),http_requests=0)
    spm_m5.write(spm_m5.BASE/'m5_replay_verification.json',dict(phases=result,caveat='Checkpoint skipping and context reconstruction using validated API cache only; no independent generation or rescoring. Request transport disabled.'))
    print(json.dumps(result,indent=2))
asyncio.run(main())
