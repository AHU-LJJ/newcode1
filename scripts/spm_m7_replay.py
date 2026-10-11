"""Exact offline reconstruction of successful M7 contexts and cached evaluations."""
import argparse
import asyncio
import json
import httpx
import spm_m7 as exp


async def main(phase):
    plan=exp.frozen_import(phase)
    from spm.api import Gateway
    from spm.config import SPMConfig,endpoints
    from spm.__main__ import parser
    from spm.runner import Experiment,selected_cases
    from spm.evaluation import evaluate
    out=exp.BASE/f'm7_{phase}';files=list((out/'samples').glob('*.json'))
    before={str(f):exp.sha(f) for f in files};rows=[exp.read(f) for f in files]
    args=parser().parse_args(plan['argv']);args.limit=None;cases,_=selected_cases(args);lookup={c.id:c for c in cases}
    api=Gateway(endpoints(),exp.BASE/'api_cache_m7.sqlite')
    await api.http.aclose()
    def reject(request):raise AssertionError('Offline replay cannot send HTTP')
    api.http=httpx.AsyncClient(transport=httpx.MockTransport(reject))
    async def no_request(*args,**kwargs):raise AssertionError('Validated cache hit required')
    api._request=no_request
    runner=Experiment(api,out,{m:SPMConfig(**c) for m,c in plan['configs'].items()})
    for identity,entry in runner.memory_index.items():
        ready=asyncio.get_running_loop().create_future();ready.set_result((str(out/entry['path']),entry['user']))
        runner.prepared[identity]=ready
    memory_files={str(f):exp.sha(f) for f in (out/'memory').glob('*.json')}
    reconstructed=evaluations=0
    for row in rows:
        if row['status']!='ok':continue
        case=lookup[row['case_id']];await runner.run_case(case,row['method'])
        context,trace=await runner.context(case.public,row['method'])
        trace['context']=context;trace['context_characters']=len(context)
        assert context==row['trace']['context'] and trace==row['trace'],row['case_id']+'/'+row['method']
        reconstructed+=1
        assert await evaluate(case,row['method'],context,trace,api)==row['evaluation']
        evaluations+=1
    usage=await api.close();assert usage['http_requests']==0
    assert before=={str(f):exp.sha(f) for f in files}
    assert memory_files=={str(f):exp.sha(f) for f in (out/'memory').glob('*.json')}
    result=dict(phase=phase,successes_replayed=len(runner.rows),contexts_reconstructed=reconstructed,evaluations_replayed=evaluations,
        failed_excluded=sum(r['status']!='ok' for r in rows),sample_files_unchanged=len(files),http_requests=0,
        caveat='Exact context/trace reconstruction and original generation/grades from validated cache; no independent generation or regrading.')
    exp.write(out/'replay_verification.json',result);print(json.dumps(result))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['dev','test'],required=True)
    args=parser.parse_args();asyncio.run(main(args.phase))
