import asyncio
import numpy as np
import pytest
from pydantic import ValidationError

from spm.adapters import episode
from spm.config import VARIANTS, SPMConfig
from spm.models import Condition
from spm.pipeline import SPM, make_unit
from spm.small_protocol import WireExtraction, WireQuery, SmallRawUnit, TypedUnit, query_mentions
from spm.store import Store
from spm.typed_graph import TypedGraphMemory, TypedRelations
from spm.runner import Experiment


def vec():
    v=np.zeros(1024,dtype=np.float32);v[0]=1;return v


def unit(ep,content,target='coffee',facet='enjoyment',polarity='prefer',turn=0,expression='explicit',condition=None):
    raw=SmallRawUnit(content=content,polarity=polarity,scope='drinks',domain='food',
        nature='taste',expression=expression,target=target,facet=facet,
        condition=condition or Condition(),span=ep.messages[turn].content,turn_idx=turn)
    return make_unit('u',raw,ep,VARIANTS['typed_m8'])


class API:
    def __init__(self,r=1,s=0):self.r=r;self.s=s;self.calls=[]
    async def ask(self,role,system,value,schema,tag,validate=None,**kwargs):
        self.calls.append(value)
        assert tag=='typed_relations'
        out=TypedRelations(relations=[dict(index=p['index'],r=self.r,s=self.s) for p in value['pairs']])
        if validate:validate(out)
        return out


@pytest.mark.parametrize('axis',['target','facet','condition'])
def test_different_axes_preserve_both_without_relation_http(tmp_path,axis):
    async def run():
        api=API();store=Store(tmp_path/'s.sqlite','u');memory=TypedGraphMemory('u',store,api,VARIANTS['typed_m8'])
        ep0=episode([dict(role='user',content='I like coffee')],0)
        ep1=episode([dict(role='user',content='I dislike tea')],1)
        a=unit(ep0,'Likes coffee')
        b=unit(ep1,'Avoids tea',polarity='avoid',**{'target':'tea'} if axis=='target' else
            {'facet':'temperature'} if axis=='facet' else {'condition':Condition(entity='Alice')})
        await memory._upsert(a,vec(),ep0)
        await memory._upsert(b,vec(),ep1)
        assert len(store.all_units())==2 and not api.calls
        assert not any(e['action']=='supersede' for e in store.snapshot()['events'])
        store.close()
    asyncio.run(run())


def test_implicit_change_replaces_older_same_object_facet(tmp_path):
    async def run():
        store=Store(tmp_path/'s.sqlite','u');api=API()
        memory=TypedGraphMemory('u',store,api,VARIANTS['typed_m8'])
        oldep=episode([dict(role='user',content='Coffee tastes wonderful')],0)
        newep=episode([dict(role='user',content='Coffee is bitter and unpleasant to me')],1)
        a=unit(oldep,'Enjoys coffee');b=unit(newep,'Dislikes coffee',polarity='avoid',expression='feedback')
        await memory._upsert(a,vec(),oldep);await memory._upsert(b,vec(),newep)
        assert store.get(a.id).status=='superseded' and store.get(b.id).status=='active'
        saved=store.get(b.id);assert isinstance(saved,TypedUnit) and saved.target=='coffee'
        event=next(e for e in store.snapshot()['events'] if e['action']=='typed_check')
        assert event['new_meaning']['sources'][event['source']]['span']==newep.messages[0].content
        assert len(api.calls)==1
        store.close()
    asyncio.run(run())


@pytest.mark.parametrize('expression,order,expected',[('explicit',0,'pending'),('choice',1,'pending'),('explicit',1,'active')])
def test_replacement_requires_causal_current_assertion(tmp_path,expression,order,expected):
    async def run():
        store=Store(tmp_path/'s.sqlite','u');memory=TypedGraphMemory('u',store,API(),VARIANTS['typed_m8'])
        ep0=episode([dict(role='user',content='I like coffee but dislike it sometimes')],0)
        ep1=ep0 if order==0 else episode([dict(role='user',content='I dislike coffee')],1)
        a=unit(ep0,'Likes coffee');b=unit(ep1,'Dislikes coffee',polarity='avoid',expression=expression)
        await memory._upsert(a,vec(),ep0);await memory._upsert(b,vec(),ep1)
        assert store.get(b.id).status==expected
        assert (store.get(a.id).status=='superseded')==(expected=='active')
        store.close()
    asyncio.run(run())


def test_unresolved_conflict_does_not_replace_by_repetition(tmp_path):
    async def run():
        store=Store(tmp_path/'s.sqlite','u');api=API(2,None)
        memory=TypedGraphMemory('u',store,api,VARIANTS['typed_m8'])
        eps=[episode([dict(role='user',content=s)],i) for i,s in enumerate(['I like coffee','I dislike coffee','I dislike coffee'])]
        a=unit(eps[0],'Likes coffee');b=unit(eps[1],'Dislikes coffee',polarity='avoid');c=unit(eps[2],'Dislikes coffee',polarity='avoid')
        await memory._upsert(a,vec(),eps[0]);await memory._upsert(b,vec(),eps[1]);await memory._upsert(c,vec(),eps[2])
        assert store.get(a.id).status=='active' and store.get(b.id).status=='pending'
        assert store.get(b.id).n_support==2
        store.close()
    asyncio.run(run())


def test_invalid_revision_source_cannot_authorize_replacement(tmp_path):
    async def run():
        store=Store(tmp_path/'s.sqlite','u');memory=TypedGraphMemory('u',store,API(1,9),VARIANTS['typed_m8'])
        ep=episode([dict(role='user',content='I like coffee')],0)
        a=unit(ep,'Likes coffee');b=unit(ep,'Avoids coffee',polarity='avoid')
        await memory._upsert(a,vec(),ep)
        with pytest.raises(ValueError,match='NEW USER source'):await memory._upsert(b,vec(),ep)
        assert len(store.all_units())==1
        store.close()
    asyncio.run(run())


def test_numeric_labels_are_strict_and_larger_content_is_preserved():
    item=dict(s=0,c='x'*900,p=1,n=0,e=3,scope='food',domain='food',target='Coffee',facet='Flavor')
    out=WireExtraction(units=[item]);assert len(out.units[0].c)==900
    for field,value in [('p','avoid'),('n',False),('s','0'),('e',7)]:
        with pytest.raises(ValidationError):WireExtraction(units=[dict(item,**{field:value})])
    with pytest.raises(ValidationError):WireQuery(refs=['0'])
    with pytest.raises(ValidationError):WireQuery(refs=list(range(13)))


def test_query_references_select_literal_mentions_without_copying_names():
    async def run():
        text='Write to Alice Chen about North Star and "quiet mode".'
        mentions=query_mentions(text);assert all(s in text for s in mentions)
        alice=mentions.index('Alice Chen')
        class QueryAPI:
            async def ask(self,role,system,value,schema,tag,validate):
                out=WireQuery(refs=[alice],primary=alice,probes=['tone']);validate(out);return out
        memory=SPM('u',None,QueryAPI(),VARIANTS['v6_m8'])
        q=await memory.analyze(text);assert q.entities==['alice chen'] and q.situation.entity=='alice chen'
        assert q.probes[0]==text
    asyncio.run(run())


def test_old_writer_and_protocol_remain_separate(tmp_path):
    old=VARIANTS['v6_m6'];new=VARIANTS['v6_m8'];typed=VARIANTS['typed_m8']
    assert old.tool_protocol=='compact' and old.update_policy=='legacy'
    assert len({str(c.write_signature()) for c in (old,new,typed)})==3
    assert Experiment(None,tmp_path,{'typed_m8':typed}).memory_class('typed_m8') is TypedGraphMemory
    with pytest.raises(ValueError):SPM('u',None,None,typed)
    with pytest.raises(ValueError):SPMConfig(tool_protocol='small',update_policy='typed',memory_manager='flat')
