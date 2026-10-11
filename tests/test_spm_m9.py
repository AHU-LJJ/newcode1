import asyncio
import numpy as np
import pytest
from pydantic import ValidationError

from spm.adapters import episode
from spm.boolean_protocol import extraction_schema,query_schema,extract_flags,analyze_flags,gate_schema,dense_gate
from spm.config import VARIANTS
from spm.pipeline import SPM, make_unit, Candidate
from spm.models import ParsedQuery, Condition, RawUnit


def observation(**changes):
    item=dict(source_id=0,content='The user dislikes coffee.',avoid=True,require=False,
        constraint=False,style=False,hypothesis=False,scope='drinks',domain='food',target='coffee',facet='enjoyment')
    return dict(item,**changes)


@pytest.mark.parametrize('field',['avoid','require','constraint','style','hypothesis'])
def test_semantic_flags_reject_category_numbers_and_strings(field):
    schema=extraction_schema(2)
    for value in [1,'false','taste']:
        with pytest.raises(ValidationError):schema.model_validate({'units':[observation(**{field:value})]})


def test_source_bounds_are_in_schema_and_do_not_coerce_indices():
    schema=extraction_schema(2)
    assert schema.model_json_schema()['$defs']['BoundFlagObservation']['properties']['source_id']['maximum']==1
    for index in [-1,2,'0',True]:
        with pytest.raises(ValidationError):schema.model_validate({'units':[observation(source_id=index)]})


@pytest.mark.parametrize('count',[0,1,4])
def test_mention_bounds_and_no_mention_schema(count):
    schema=query_schema(count)
    if not count:
        assert 'entity_indices' not in schema.model_json_schema()['properties']
        with pytest.raises(ValidationError):schema.model_validate({'entity_indices':[0]})
    else:
        item=schema.model_json_schema()['properties']['entity_indices']['items']
        assert item['minimum']==0 and item['maximum']==count-1
        with pytest.raises(ValidationError):schema.model_validate({'entity_indices':[count]})


@pytest.mark.parametrize('hypothesis',[False,True])
def test_flags_preserve_literal_source_and_hypothesis(hypothesis):
    async def run():
        ep=episode([dict(role='assistant',content='You might like coffee.'),dict(role='user',content='Coffee tastes bitter to me.')])
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                assert tag=='extract_flags'
                assert all('turn_idx' not in m for m in value['messages'])
                out=schema.model_validate({'units':[observation(hypothesis=hypothesis)]});validate(out);return out
        memory=SPM('u',None,Fake(),VARIANTS['v6_m8'])
        chunk=[dict(turn_idx=i,**m.model_dump()) for i,m in enumerate(ep.messages)]
        out=await extract_flags(memory,chunk,ep);raw=out.units[0]
        assert raw.span==ep.messages[1].content and raw.turn_idx==1
        assert raw.polarity=='avoid' and raw.nature=='taste'
        assert raw.expression==('choice' if hypothesis else 'explicit')
        typed=make_unit('u',raw,ep,VARIANTS['v6_m8']);assert typed.target=='coffee'
    asyncio.run(run())


def test_no_mentions_still_analyzes_probes_without_inventing_entities():
    async def run():
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                assert not value['mentions'] and 'entity_indices' not in schema.model_json_schema()['properties']
                out=schema.model_validate({'probes':['tone'],'situation':{'channel':'chat_with_ai'}});validate(out);return out
        memory=SPM('u',None,Fake(),VARIANTS['v6_m8'])
        q=await analyze_flags(memory,'write a friendly reply to the team')
        assert not q.entities and q.situation.entity is None and q.probes[1]=='tone'
    asyncio.run(run())


def test_avoid_and_require_cannot_both_authorize_one_observation():
    async def run():
        ep=episode([dict(role='user',content='I dislike coffee.')])
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                out=schema.model_validate({'units':[observation(require=True)]});validate(out);return out
        memory=SPM('u',None,Fake(),VARIANTS['v6_m8'])
        with pytest.raises(ValueError,match='atomic observation'):await extract_flags(memory,[dict(turn_idx=0,role='user',content=ep.messages[0].content)],ep)
    asyncio.run(run())


def test_flag_protocol_is_routed_and_preserves_extraction(tmp_path):
    async def run():
        ep=episode([dict(role='user',content='I dislike coffee.')])
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                assert tag=='extract_flags'
                out=schema.model_validate({'units':[observation()]});validate(out);return out
            async def embed(self,texts):
                vectors=np.zeros((len(texts),1024),dtype=np.float32);vectors[:,0]=1;return vectors
        cfg=VARIANTS['v6_m9'];memory=SPM('u',None,Fake(),cfg)
        data=await memory.prepare(ep)
        assert data['units'][0].target=='coffee' and data['units'][0].polarity=='avoid'
        assert data['units'][0].evidence[0].span==ep.messages[0].content
        assert cfg.write_signature()!=VARIANTS['v6_m8'].write_signature()
    asyncio.run(run())


@pytest.mark.parametrize('selected,valid',[(0,False),(1,True)])
def test_literal_copy_of_other_user_turn_requires_correct_source(selected,valid):
    async def run():
        ep=episode([dict(role='user',content='Help me write a menu.'),dict(role='assistant',content='Try coffee.'),
            dict(role='user',content='I avoid coffee.')])
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                out=schema.model_validate({'units':[observation(source_id=selected,content='I avoid coffee.')]})
                validate(out);return out
        memory=SPM('u',None,Fake(),VARIANTS['v6_m9'])
        chunk=[dict(turn_idx=i,**m.model_dump()) for i,m in enumerate(ep.messages)]
        if valid:
            data=await extract_flags(memory,chunk,ep);assert data.units[0].turn_idx==2
        else:
            with pytest.raises(ValueError,match='different USER turn'):await extract_flags(memory,chunk,ep)
    asyncio.run(run())


@pytest.mark.parametrize('scores',[[],[.5],[.5,.6,.7],[False,.5],['.5',.5],[float('nan'),.5],[1.1,.5]])
def test_dense_gate_requires_exact_finite_numeric_coverage(scores):
    with pytest.raises(ValidationError):gate_schema(2).model_validate({'scores':scores})


def test_dense_gate_binds_order_and_keeps_entity_and_conflict_guards():
    async def run():
        cfg=VARIANTS['v6_m9'];ep=episode([dict(role='user',content='I like coffee and avoid tea.')])
        def candidate(content,polarity='prefer',entity=None):
            raw=RawUnit(content=content,polarity=polarity,scope='drinks',domain='food',nature='taste',
                expression='explicit',span=ep.messages[0].content,turn_idx=0,condition=Condition(entity=entity))
            return Candidate(make_unit('u',raw,ep,cfg),.8)
        a=candidate('Likes coffee');b=candidate('Avoids tea','avoid');c=candidate('Alice likes sweet tea',entity='Alice')
        b.unit.status='pending';b.unit.challenges=[a.unit.id]
        class Fake:
            async def ask(self,role,system,value,schema,tag,**kwargs):
                assert tag=='gate_dense' and kwargs['max_tokens']==512
                assert 'a short reason' not in system
                assert [item['index'] for item in value['candidates']]==[0,1,2]
                return schema.model_validate({'scores':[.9,.8,1]})
        memory=SPM('u',None,Fake(),cfg)
        q=ParsedQuery(situation=Condition(entity='Bob'),entities=['bob'],probes=['drinks'])
        decisions,scores=await memory.gate('Suggest drinks for Bob',q,[a,b,c],[])
        assert [s['unit_id'] for s in scores]==[a.unit.id,b.unit.id,c.unit.id]
        assert [s['p_apply'] for s in scores]==[.9,.8,1.]
        assert [d.decision for d in decisions]==['uncertain','uncertain','suppress']
        assert memory.gate_coverage==dict(expected=3,initial_returned=3,completed_indices=[])
    asyncio.run(run())


@pytest.mark.parametrize('correct_primary',[True,False])
def test_named_query_indices_bind_literal_mentions_and_check_primary(correct_primary):
    async def run():
        text='Write to Alice Chen.'
        class Fake:
            async def ask(self,role,system,value,schema,tag,validate):
                assert tag=='analyze_flags'
                idx=next(m['index'] for m in value['mentions'] if m['text']=='Alice Chen')
                out=schema.model_validate({'entity_indices':[idx],
                    'primary_index':idx if correct_primary else 0,'probes':['tone']})
                validate(out);return out
        memory=SPM('u',None,Fake(),VARIANTS['v6_m9'])
        if correct_primary:
            q=await memory.analyze(text);assert q.entities==['alice chen'] and q.situation.entity=='alice chen'
        else:
            with pytest.raises(ValueError,match='must belong'):await memory.analyze(text)
    asyncio.run(run())
