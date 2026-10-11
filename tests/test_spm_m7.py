import asyncio
from dataclasses import replace

import numpy as np
import pytest

from spm.adapters import episode
from spm.config import SPMConfig,VARIANTS
from spm.conservative import ConservativeMemory
from spm.models import Condition,RawUnit,UpdateChecks
from spm.pipeline import SPM,make_unit
from spm.runner import Experiment
from spm.store import Store


def vector():
    v=np.zeros(1024,dtype=np.float32);v[0]=1
    return v


def observation(content,ep,cfg,polarity='prefer'):
    raw=RawUnit(content=content,polarity=polarity,scope='drinks',domain='food',nature='taste',
        expression='explicit',condition=Condition(),span=ep.messages[0].content,turn_idx=0)
    return make_unit('u',raw,ep,cfg)


class CheckAPI:
    def __init__(self,**check):self.check=check;self.calls=[]
    async def ask(self,role,system,value,schema,tag,validate,**kwargs):
        self.calls.append((tag,value))
        assert tag=='update_check'
        out=UpdateChecks(checks=[dict(
                dict(index=p['index'],same_target=False,same_facet=False,equivalent=False,
                    mutually_exclusive=False,explicit_revision=False,reason='evidence check'),**self.check)
                for p in value['pairs']])
        validate(out)
        return out


@pytest.mark.parametrize('label',['same','contradicts'])
def test_complementary_observations_survive_wrong_relation(tmp_path,label):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u');api=CheckAPI()
        memory=ConservativeMemory('u',store,api,cfg)
        ep=episode([dict(role='user',content='I enjoy cycling and avoid water sports.')],0)
        a=observation('Enjoys cycling',ep,cfg);b=observation('Avoids water sports',ep,cfg,'avoid')
        await memory._upsert(a,vector(),ep,{})
        await memory._upsert(b,vector(),ep,{(b.id,a.id):label})
        assert len(store.all_units())==2
        assert not any(u.status in ('pending','superseded') for u in store.all_units(None))
        assert store.get(a.id).evidence==a.evidence
        assert len(api.calls)==1
        store.close()
    asyncio.run(run())


def test_real_revision_supersedes_only_older_observation(tmp_path):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u')
        api=CheckAPI(same_target=True,same_facet=True,mutually_exclusive=True,explicit_revision=True,
            revision_source=0,revision_quote='I no longer like coffee')
        memory=ConservativeMemory('u',store,api,cfg)
        oldep=episode([dict(role='user',content='I like coffee')],0)
        ep=episode([dict(role='user',content='I no longer like coffee')],1)
        a=observation('Likes coffee',oldep,cfg);b=observation('Avoids coffee',ep,cfg,'avoid')
        await memory._upsert(a,vector(),oldep,{})
        await memory._upsert(b,vector(),ep,{(b.id,a.id):'contradicts'})
        assert store.get(a.id).status=='superseded'
        assert store.get(b.id).status=='active'
        store.close()
    asyncio.run(run())


def test_repeated_weak_conflict_does_not_promote_pending(tmp_path):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u')
        api=CheckAPI(same_target=True,same_facet=True,mutually_exclusive=True)
        memory=ConservativeMemory('u',store,api,cfg)
        ep0=episode([dict(role='user',content='I like coffee')],0)
        ep1=episode([dict(role='user',content='Coffee tastes bad')],1)
        ep2=episode([dict(role='user',content='Coffee tastes bad')],2)
        a=observation('Likes coffee',ep0,cfg);b=observation('Avoids coffee',ep1,cfg,'avoid')
        await memory._upsert(a,vector(),ep0,{})
        await memory._upsert(b,vector(),ep1,{(b.id,a.id):'contradicts'})
        await memory._upsert(observation('Avoids coffee',ep2,cfg,'avoid'),vector(),ep2,{})
        assert store.get(a.id).status=='active'
        assert store.get(b.id).status=='pending' and store.get(b.id).n_support==2
        assert store.get(b.id).challenges==[a.id]
        store.close()
    asyncio.run(run())


def test_equivalence_merges_evidence_without_extra_calls(tmp_path):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u')
        api=CheckAPI(same_target=True,same_facet=True,equivalent=True)
        memory=ConservativeMemory('u',store,api,cfg)
        ep0=episode([dict(role='user',content='I like coffee')],0)
        ep1=episode([dict(role='user',content='Coffee is my favorite drink')],1)
        a=observation('Likes coffee',ep0,cfg);b=observation('Coffee is a favorite',ep1,cfg)
        await memory._upsert(a,vector(),ep0,{})
        await memory._verify_pairs([(b,a),(b,a)])
        await memory._upsert(b,vector(),ep1,{(b.id,a.id):'same'})
        assert len(api.calls)==1
        assert len(store.all_units())==1 and store.get(a.id).n_support==2
        assert len(store.get(a.id).evidence)==2
        store.close()
    asyncio.run(run())


@pytest.mark.parametrize('change',[
    dict(revision_quote='invented evidence'),dict(revision_source=4),
    dict(same_target=False),dict(equivalent=True),
])
def test_invalid_revision_witness_rejected(tmp_path,change):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u')
        fields=dict(same_target=True,same_facet=True,mutually_exclusive=True,explicit_revision=True,
            revision_source=0,revision_quote='I no longer like coffee')
        fields.update(change);api=CheckAPI(**fields);memory=ConservativeMemory('u',store,api,cfg)
        ep=episode([dict(role='user',content='I no longer like coffee')],0)
        a=observation('Likes coffee',ep,cfg);b=observation('Avoids coffee',ep,cfg,'avoid')
        with pytest.raises(ValueError):await memory._verify_pairs([(b,a)])
        assert not memory.update_checks
        store.close()
    asyncio.run(run())


def test_same_turn_revision_cannot_reverse_source_order(tmp_path):
    async def run():
        cfg=VARIANTS['v6_m7'];store=Store(tmp_path/'s.sqlite','u')
        api=CheckAPI(same_target=True,same_facet=True,mutually_exclusive=True,explicit_revision=True,
            revision_source=0,revision_quote='I no longer like coffee')
        memory=ConservativeMemory('u',store,api,cfg)
        ep=episode([dict(role='user',content='I no longer like coffee')],0)
        a=observation('Likes coffee',ep,cfg);b=observation('Avoids coffee',ep,cfg,'avoid')
        await memory._upsert(a,vector(),ep,{})
        await memory._upsert(b,vector(),ep,{(b.id,a.id):'contradicts'})
        assert store.get(a.id).status=='active' and store.get(b.id).status=='pending'
        store.close()
    asyncio.run(run())


def test_writer_identity_and_direct_sdk_guard(tmp_path):
    old=VARIANTS['v6_m6'];cfg=VARIANTS['v6_m7']
    assert 'update_policy' not in old.write_signature()
    assert cfg.write_signature()!=old.write_signature()
    assert replace(cfg,gate_completion='strict').write_signature()==cfg.write_signature()
    assert Experiment(None,tmp_path,{'v6_m7':cfg}).memory_class('v6_m7') is ConservativeMemory
    with pytest.raises(ValueError):SPM('u',None,None,cfg)
    with pytest.raises(ValueError):SPMConfig(memory_manager='flat',update_policy='conservative')
