"""Optional offline write replay capturing exact update pairs for development diagnosis."""
import argparse
import asyncio
import json
from pathlib import Path
import tempfile

import httpx
import spm_m7 as exp


async def main(phase):
    plan=exp.frozen_import(phase)
    from spm.api import Gateway
    from spm.config import SPMConfig,endpoints
    from spm.__main__ import parser
    from spm.runner import Experiment,selected_cases
    args=parser().parse_args(plan['argv']);args.limit=None;cases,_=selected_cases(args)
    out=exp.BASE/f'm7_{phase}'
    rows={r['case_id']:r for r in [exp.read(f) for f in (out/'samples').glob('*.json')]
          if r['method']=='v6_m7' and r['status']=='ok'}
    original_files={str(f):exp.sha(f) for f in (out/'memory').glob('*.json')}
    api=Gateway(endpoints(),exp.BASE/'api_cache_m7.sqlite');await api.http.aclose()
    def reject(request):raise AssertionError('Offline writes cannot send HTTP')
    api.http=httpx.AsyncClient(transport=httpx.MockTransport(reject))
    async def no_request(*args,**kwargs):raise AssertionError('Original validated write cache required')
    api._request=no_request
    original_ask=api.ask;captured=[];case_id=None
    async def record(*args,**kwargs):
        result=await original_ask(*args,**kwargs)
        tag=args[4] if len(args)>4 else kwargs.get('tag','misc')
        if tag=='update_check':
            value=args[2] if len(args)>2 else kwargs['value']
            captured.append(dict(case_id=case_id,payload=value,checks=result.model_dump(mode='json')))
        return result
    api.ask=record;rebuilt=0
    try:
        with tempfile.TemporaryDirectory(prefix='caps-m7-write-',dir='/tmp') as temp:
            runner=Experiment(api,Path(temp),{'v6_m7':SPMConfig(**plan['configs']['v6_m7'])})
            for case in cases:
                if case.id not in rows:continue
                case_id=case.id
                path,_=await runner.memory(case.public,'v6_m7')
                expected=(out/rows[case.id]['trace']['memory_file']).with_suffix('.snapshot.json')
                actual=Path(path).with_suffix('.snapshot.json')
                assert exp.read(actual)==exp.read(expected),case.id
                rebuilt+=1
        usage=api.usage();assert usage['http_requests']==0
        assert original_files=={str(f):exp.sha(f) for f in (out/'memory').glob('*.json')}
        exp.write(out/'write_replay_pairs.json',captured)
        result=dict(phase=phase,successful_conservative_banks_rebuilt=rebuilt,
            cached_update_batches=len(captured),cached_update_pairs=sum(len(c['payload']['pairs']) for c in captured),
            snapshots_equal=True,original_memory_files_unchanged=True,http_requests=0,
            caveat='Additional offline development diagnostic. Exact original meanings/evidence captured; no independent semantic labels.')
        exp.write(out/'write_replay_verification.json',result);print(json.dumps(result))
    finally:await api.close()


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--phase',choices=['dev','test'],default='dev')
    args=parser.parse_args();asyncio.run(main(args.phase))
