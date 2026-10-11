import asyncio
from dataclasses import replace
import numpy as np
import pytest
from pydantic import ValidationError
from spm.adapters import episode
from spm.config import VARIANTS
from spm.models import (Condition, SourceExtraction, IndexedGate, IndexedRelations,
                        PreferenceFields, ParsedQuery)
from spm.pipeline import SPM, source_dialogue, validate_indices, Candidate
from spm.store import Store


def fields(content='The user prefers coffee'):
    return dict(content=content,polarity='prefer',scope='drinks',domain='food',nature='taste',condition={})


class API:
    def __init__(self):self.calls=[];self.invalid=False
    async def embed(self,texts):
        v=np.zeros((len(texts),1024),dtype=np.float32);v[:,0]=1;return v
    async def ask(self,role,system,value,schema,tag,validate=None,**kwargs):
        self.calls.append((tag,value))
        if tag=='extract':
            sources=[e for m in value['messages'] if m['role']=='user' for e in m['excerpts'] if 'source_id' in e]
            out=SourceExtraction(units=[dict(**fields(),source_id=999 if self.invalid else sources[0]['source_id'],expression='explicit')])
        elif tag=='normalize':out=PreferenceFields(**fields(value['preference']))
        elif tag=='relations_batch':out=IndexedRelations(relations=[dict(index=p['index'],relation='contradicts') for p in reversed(value['pairs'])])
        elif tag=='relation':out=IndexedRelations(relations=[dict(index=p['index'],relation='contradicts') for p in value['existing']])
        elif tag=='gate':out=IndexedGate(decisions=[dict(index=p['index'],p_apply=.9,reason='relevant') for p in reversed(value['candidates'])])
        else:raise AssertionError(tag)
        if validate:validate(out)
        return out


def test_literal_source_partition_and_invalid_source_never_writes(tmp_path):
    source='I prefer coffee.\n'+('long '*300)+' Tea is fine.'
    chunk=[dict(turn_idx=0,role='assistant',content='You love wine.'),dict(turn_idx=1,role='user',content=source)]
    messages,sources=source_dialogue(chunk)
    assert ''.join(e['text'] for e in messages[1]['excerpts'])==source
    assert all(s['turn_idx']==1 and s['span'] in source and len(s['span'])<=600 for s in sources)
    assert 'source_id' not in messages[0]
    async def run():
        store=Store(tmp_path/'s.sqlite','u');api=API();memory=SPM('u',store,api,VARIANTS['v6_compact'])
        ep=episode([dict(role=m['role'],content=m['content']) for m in chunk])
        prepared=await memory.prepare(ep)
        evidence=prepared['units'][0].evidence[0]
        assert evidence.turn_idx==1 and evidence.span=='I prefer coffee.'
        api.invalid=True
        with pytest.raises(ValueError,match='supplied USER source_id'):await memory.ingest(ep)
        assert not store.all_units() and not store.db.execute('SELECT id FROM episodes WHERE done=1').fetchall()
        store.close()
    asyncio.run(run())


def test_index_validation_rejects_duplicates_unknown_and_noninteger():
    for indices in ([0,0],[0,2],[0],[]):
        out=IndexedRelations(relations=[dict(index=i,relation='same') for i in indices])
        with pytest.raises(ValueError):validate_indices(out.relations,2)
    for value in ('0',False,0.0,-1):
        with pytest.raises(ValidationError):IndexedGate(decisions=[dict(index=value,p_apply=.8,reason='x')])


def test_compact_given_normalization_and_canonical_gate_mapping(tmp_path):
    async def run():
        store=Store(tmp_path/'s.sqlite','Alice');api=API();memory=SPM('Alice',store,api,VARIANTS['v6_compact'])
        ep=episode([dict(role='user',content='Likes coffee'),dict(role='user',content='Avoids tea')])
        await memory.install_given(ep)
        units=store.all_units()
        assert len(units)==2 and len([c for c in api.calls if c[0]=='normalize'])==2
        for unit in units:
            e=unit.evidence[0];assert e.span==ep.messages[e.turn_idx].content and e.expression=='explicit'
        units[0].condition.entity='bob'
        decisions,scores=await memory.gate('A drink for Alice',ParsedQuery(entities=['alice']),
                                         [Candidate(u,.8) for u in units],[])
        assert {d.unit_id for d in decisions}=={u.id for u in units}
        assert decisions[0].decision=='suppress' and decisions[1].decision=='apply'
        sent=api.calls[-1][1]['candidates']
        assert all('id' not in c['unit'] for c in sent)
        assert scores[0]['unit_id']==units[1].id
        store.close()
    asyncio.run(run())


def test_compact_relation_indices_preserve_negation_and_share_write_bank(tmp_path):
    async def run():
        cfg=VARIANTS['v6_compact'];other=VARIANTS['v6_compact_no_raw']
        assert cfg.write_signature()==other.write_signature()
        assert cfg.write_signature()!=VARIANTS['v6'].write_signature()
        store=Store(tmp_path/'s.sqlite','u');api=API();memory=SPM('u',store,api,cfg)
        first=episode([dict(role='user',content='I prefer coffee')],0)
        data=await memory.prepare(first);await memory.ingest(first,data)
        second=episode([dict(role='user',content='I dislike coffee')],1)
        data=await memory.prepare(second)
        data['units'][0].polarity='avoid';data['units'][0].content='The user dislikes coffee'
        relations=await memory._batch_relations(data['units'],data['vectors'])
        assert list(relations.values())==['contradicts']
        data['relations']=relations;await memory.ingest(second,data)
        assert store.all_units()[0].polarity=='avoid'
        assert len(store.all_units(statuses=("active","candidate","pending","superseded")))==2
        store.close()
    asyncio.run(run())


def test_compact_source_repair_rejects_hallucinated_id_and_binds_literal(tmp_path):
    import httpx,json
    from spm.api import Gateway
    from spm.config import Endpoint
    async def run():
        payloads=[]
        def transport(request):
            payload=json.loads(request.content)
            if request.url.path.endswith('embeddings'):
                return httpx.Response(200,json=dict(data=[dict(index=i,embedding=[1.]+[0.]*1023) for i,_ in enumerate(payload['input'])],usage=dict(total_tokens=1)))
            payloads.append(payload)
            result=dict(units=[dict(**fields(),source_id=99 if len(payloads)==1 else 0,expression='explicit')])
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(result)))],usage=dict(total_tokens=1)))
        ep=Endpoint('test','SECRET','https://example.com/v1');api=Gateway(dict(cheap=ep,embed=ep),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        store=Store(tmp_path/'bank.sqlite','u');memory=SPM('u',store,api,VARIANTS['v6_compact'])
        data=await memory.prepare(episode([dict(role='user',content='I prefer coffee.')]))
        assert data['units'][0].evidence[0].span=='I prefer coffee.'
        assert api.schema_failures['extract']==1
        assert 'Do not add span or turn_idx' in payloads[1]['messages'][-1]['content']
        await api.close();store.close()
    asyncio.run(run())
