import asyncio
import json
from dataclasses import replace

import httpx
import numpy as np
import pytest

from spm.adapters import MemoryInput, episode
from spm.api import Gateway
from spm.baselines import FlatMemory, raw_context
from spm.config import Endpoint, VARIANTS
from spm.models import IndexedGate, ParsedQuery
from spm.pipeline import Candidate, SPM
from spm.store import Store
from test_spm_compact import API, fields


def test_targeted_gate_retains_valid_scores_and_only_completes_missing(tmp_path):
    async def run():
        responses=[]
        def transport(request):
            payload=json.loads(request.content);value=json.loads(payload['messages'][1]['content'])
            responses.append(value)
            if 'target_candidate' in value:
                result=dict(p_apply=.1,reason='unrelated omitted candidate')
            else:
                result=dict(decisions=[dict(index=0,p_apply=.8,reason='original valid decision')])
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(result)))],usage=dict(total_tokens=1)))
        endpoint=Endpoint('mock','SECRET','https://example.com/v1')
        api=Gateway(dict(chat=endpoint),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        store=Store(tmp_path/'memory.sqlite','u')
        base_api=API();memory=SPM('u',store,base_api,VARIANTS['v6_m6'])
        await memory.install_given(episode([dict(role='user',content='Likes coffee'),dict(role='user',content='Likes tea')]))
        memory.api=api;units=store.all_units()
        decisions,scores=await memory.gate('coffee',ParsedQuery(),[Candidate(u,.8) for u in units],[])
        assert len(responses)==2 and responses[1]['target_candidate']['index']==1
        assert [d.p_apply for d in decisions]==[.8,.1]
        assert scores[0]['reason']=='original valid decision'
        assert memory.gate_coverage['completed_indices']==[1]
        assert not api.schema_failures['gate']
        # Validated cached partial + scalar results replay without HTTP.
        await memory.gate('coffee',ParsedQuery(),[Candidate(u,.8) for u in units],[])
        assert len(responses)==2
        await api.close();store.close()
    asyncio.run(run())


@pytest.mark.parametrize('indices',[[0,0],[0,99]])
def test_targeted_gate_never_accepts_duplicate_or_unknown(tmp_path,indices):
    class InvalidAPI(API):
        async def ask(self,role,system,value,schema,tag,validate=None,**kwargs):
            if tag=='gate':
                result=IndexedGate(decisions=[dict(index=i,p_apply=.8,reason='invalid') for i in indices])
                validate(result);return result
            if tag=='gate_complete':raise AssertionError('Must reject before scalar completion')
            return await super().ask(role,system,value,schema,tag,validate,**kwargs)
    async def run():
        store=Store(tmp_path/'s.sqlite','u');api=InvalidAPI();memory=SPM('u',store,api,VARIANTS['v6_m6'])
        await memory.install_given(episode([dict(role='user',content='coffee'),dict(role='user',content='tea')]))
        with pytest.raises(ValueError,match='unique and supplied'):
            await memory.gate('coffee',ParsedQuery(),[Candidate(u,.8) for u in store.all_units()],[])
        store.close()
    asyncio.run(run())


def test_flat_preserves_conflicting_observations_and_matches_raw_rag(tmp_path):
    class QueryAPI(API):
        async def ask(self,role,system,value,schema,tag,validate=None,**kwargs):
            if tag=='analyze':return ParsedQuery(probes=['drink'])
            return await super().ask(role,system,value,schema,tag,validate,**kwargs)
    async def run():
        api=QueryAPI();cfg=VARIANTS['flat_m6'];store=Store(tmp_path/'flat.sqlite','u')
        memory=FlatMemory('u',store,api,cfg)
        eps=[episode([dict(role='user',content='I prefer coffee.')],0),episode([dict(role='user',content='I dislike coffee.')],1)]
        for i,ep in enumerate(eps):
            data=await memory.prepare(ep)
            if i:data['units'][0].polarity='avoid';data['units'][0].content='The user dislikes coffee'
            await memory.ingest(ep,data)
        assert len(store.all_units())==2
        assert {u.status for u in store.all_units()}=={'active'}
        assert not any(tag.startswith('relation') for tag,_ in api.calls)
        assert cfg.write_signature()!=VARIANTS['v6_m6'].write_signature()
        query=await memory.analyze('drink');_,raw=await memory.retrieve(query)
        from spm.pipeline import render_raw_evidence
        expected,evidence=render_raw_evidence(raw,eps,cfg)
        actual,trace=await raw_context('b2_matched',MemoryInput('u',tuple(eps),'drink'),api,VARIANTS['b2_matched'])
        assert actual==expected and trace['raw_evidence']==evidence
        assert len(''.join(e['dialogue_excerpt'] for e in evidence))<=6000
        store.close()
    asyncio.run(run())


def test_tpm_wait_does_not_occupy_judge_http_slot(tmp_path):
    async def run():
        limited=Endpoint('limited','SECRET','https://example.com/v1')
        judge=Endpoint('judge','SECRET','https://example.com/v1')
        api=Gateway(dict(chat=limited,judge=judge),tmp_path/'cache.sqlite',concurrency=1,tpm_limits={'limited':10000})
        entered=asyncio.Event();release=asyncio.Event()
        original_acquire=api.limiter.acquire
        async def blocked(service,model,tokens):
            entered.set();await release.wait()
            return await original_acquire(service,model,tokens)
        api.limiter.acquire=blocked
        def transport(request):
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content='answer'))],usage=dict(total_tokens=1)))
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        task=asyncio.create_task(api.ask('chat','instructions',dict(request='x'),tag='chat'))
        await entered.wait()
        try:
            result=await asyncio.wait_for(api.ask('judge','instructions',dict(request='y'),tag='judge'),1.)
            assert result=='answer' and not task.done()
        finally:release.set()
        assert await task=='answer'
        assert api.limiter.reservations==1
        await api.close()
    asyncio.run(run())


def test_cancelled_http_queue_refunds_unsent_tpm_reservation(tmp_path):
    async def run():
        endpoint=Endpoint('limited','SECRET','https://example.com/v1')
        api=Gateway(dict(chat=endpoint),tmp_path/'cache.sqlite',concurrency=1,tpm_limits={'limited':10000})
        await api.sem.acquire()
        acquired=asyncio.Event();original=api.limiter.acquire
        async def observed(service,model,tokens):
            result=await original(service,model,tokens);acquired.set();return result
        api.limiter.acquire=observed
        async def queued():
            async with api._admission('test','limited',dict(messages=[],max_tokens=1000)):
                raise AssertionError('No free HTTP slot')
        task=asyncio.create_task(queued());await acquired.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        # A zero-usage refund retains only the existing 256-token framing buffer.
        assert api.db.execute('SELECT tokens FROM tpm_reservations').fetchone()[0]==256
        assert not api.db.execute('SELECT id FROM requests').fetchall()
        api.sem.release();await api.close()
    asyncio.run(run())


def test_overlapping_embedding_batches_share_one_canonical_vector(tmp_path):
    async def run():
        endpoint=Endpoint('embedding','SECRET','https://example.com/v1')
        api=Gateway(dict(embed=endpoint),tmp_path/'cache.sqlite')
        first_entered=asyncio.Event();second_entered=asyncio.Event()
        async def transport(request):
            payload=json.loads(request.content);texts=payload['input']
            if 'first' in texts:
                first_entered.set();await second_entered.wait();await asyncio.sleep(.02)
                coordinate=.05
            else:second_entered.set();coordinate=.1
            vector=[1.,coordinate]+[0.]*1022
            return httpx.Response(200,json=dict(data=[dict(index=i,embedding=vector) for i,_ in enumerate(texts)],usage=dict(total_tokens=1)))
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        first=asyncio.create_task(api.embed(['shared','first']));await first_entered.wait()
        second=asyncio.create_task(api.embed(['shared','second']))
        x,y=await asyncio.gather(first,second)
        assert np.array_equal(x[0],y[0])
        cached=await api.embed(['shared']);assert np.array_equal(cached[0],x[0])
        assert api.calls==2
        await api.close()
    asyncio.run(run())
