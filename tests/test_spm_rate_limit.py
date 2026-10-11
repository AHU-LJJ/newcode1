import asyncio
import json
import sqlite3

import httpx
import pytest

from spm.api import Gateway
from spm.config import Endpoint
from spm.rate_limit import TokenCounter, WindowLimiter


class FakeTime:
    def __init__(self):self.now=1000.
    def clock(self):return self.now
    async def sleep(self,seconds):self.now+=seconds


def test_rate_window_persists_across_gateways_and_refunds_known_usage(tmp_path):
    async def scenario():
        clock=FakeTime();path=tmp_path/'cache.sqlite'
        a=sqlite3.connect(path);b=sqlite3.connect(path)
        first=WindowLimiter(a,{'qwen':1000},clock=clock.clock,sleep=clock.sleep)
        second=WindowLimiter(b,{'qwen':1000},clock=clock.clock,sleep=clock.sleep)
        lease=await first.acquire('same-provider-model','qwen',600)
        first.finish(lease,100)
        # Another model/service does not consume this model's quota.
        await second.acquire('other-provider-model','qwen',600)
        await second.acquire('same-provider-model','qwen',600)
        assert clock.now==1060. and second.wait_seconds==60
        assert second.capacity['qwen']==900
        a.close();b.close()
    asyncio.run(scenario())


def test_unknown_and_abandoned_requests_keep_conservative_reservations(tmp_path):
    async def scenario():
        clock=FakeTime();db=sqlite3.connect(tmp_path/'cache.sqlite')
        limiter=WindowLimiter(db,{'qwen':1000},lease_seconds=120,clock=clock.clock,sleep=clock.sleep)
        unknown=await limiter.acquire('service','qwen',600)
        limiter.finish(unknown)
        assert db.execute('SELECT tokens FROM tpm_reservations').fetchone()[0]==600
        abandoned=await limiter.acquire('service','qwen',600)
        assert clock.now==1060.
        await limiter.acquire('service','qwen',600)
        assert clock.now==1180.  # In-flight lease outlives a single minute.
        assert abandoned is not None
        with pytest.raises(ValueError,match='Single request'):
            await limiter.acquire('service','qwen',901)
        assert await limiter.acquire('service','unlimited-model',100000) is None
        db.close()
    asyncio.run(scenario())


def test_cancelled_rate_wait_does_not_leave_an_extra_lease(tmp_path):
    async def scenario():
        db=sqlite3.connect(tmp_path/'cache.sqlite');waiting=asyncio.Event()
        async def sleep(_):waiting.set();await asyncio.Future()
        limiter=WindowLimiter(db,{'qwen':1000},sleep=sleep)
        await limiter.acquire('service','qwen',600)
        task=asyncio.create_task(limiter.acquire('service','qwen',600))
        await waiting.wait();task.cancel()
        with pytest.raises(asyncio.CancelledError):await task
        assert db.execute('SELECT COUNT(*) FROM tpm_reservations').fetchone()[0]==1
        db.close()
    asyncio.run(scenario())


def test_gateway_limits_all_roles_and_retries_but_skips_cache_hits(tmp_path,monkeypatch):
    async def scenario():
        async def sleep(_):pass
        monkeypatch.setattr('spm.api.asyncio.sleep',sleep)
        ep=Endpoint('qwen','SECRET','https://example.com/v1')
        api=Gateway(dict(chat=ep,cheap=ep),tmp_path/'cache.sqlite',tpm_limits={'qwen':20000})
        await api.http.aclose();seen=[]
        def transport(request):
            seen.append(json.loads(request.content))
            if len(seen)==1:return httpx.Response(429)
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content='ok'))],
                                                usage=dict(total_tokens=10)))
        api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        await api.ask('chat','s','a',tag='generate')
        await api.ask('cheap','s','b',tag='extract')
        await api.ask('chat','s','a',tag='generate')
        assert len(seen)==3 and api.limiter.reservations==3
        assert api.db.execute('SELECT COUNT(DISTINCT service) FROM tpm_reservations').fetchone()[0]==1
        assert api.usage()['tpm_throttle']['underestimated_requests']==0
        await api.close()
    asyncio.run(scenario())


def test_token_reservation_counts_completion_and_unicode():
    counter=TokenCounter()
    p=dict(messages=[dict(role='user',content='偏好 🎵')],max_tokens=20)
    before=counter(p);p['max_tokens']+=100
    assert counter(p)==before+100
    assert counter({'input':['偏好']})==len('偏好'.encode())+256


def test_qwen_sdk_defaults_share_role_quota_without_cli(tmp_path):
    import httpx,json,asyncio
    from spm.api import Gateway
    from spm.config import Endpoint
    async def run():
        model='Qwen/Qwen2.5-7B-Instruct';ep=Endpoint(model,'SECRET','https://example.com/v1')
        api=Gateway(dict(chat=ep,cheap=ep),tmp_path/'cache.sqlite',max_output=128)
        await api.http.aclose()
        def response(request):
            return httpx.Response(200,json=dict(model=model,choices=[dict(finish_reason='stop',message=dict(content='answer'))],usage=dict(total_tokens=10)))
        api.http=httpx.AsyncClient(transport=httpx.MockTransport(response))
        assert await api.ask('chat','system','question',tag='reader')=='answer'
        assert await api.ask('cheap','system','question',tag='writer')=='answer'
        assert api.limiter.capacity[model]==72000
        assert api.db.execute('SELECT COUNT(DISTINCT service),COUNT(*) FROM tpm_reservations').fetchone()==(1,2)
        assert api.limiter.reservations==2
        await api.close()
    asyncio.run(run())
