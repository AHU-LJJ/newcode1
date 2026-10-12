import asyncio
from dataclasses import replace
import json
from pathlib import Path
import httpx
import numpy as np
import pytest
from pydantic import ValidationError
from spm.adapters import episode,load_cases,partition,select,MemoryInput,Case
from spm.api import APIError,AccountUnavailable,BudgetExceeded,Gateway
from spm.config import Endpoint,SPMConfig,VARIANTS
from spm.evaluation import aggregate,bench_scores,checklist_metrics,frontier,paired_comparisons
from spm.models import Condition,Extraction,GateScores,ParsedQuery,RawUnit,Relations,BatchRelations
from spm.models import SourceExtraction,IndexedGate,IndexedRelations,PreferenceFields
from spm.pipeline import SPM,make_unit,Candidate,validate_coverage
from spm.runner import Experiment,selected_cases
from spm.store import Store


def vec():
    v=np.zeros(1024,dtype=np.float32);v[0]=1;return v


def raw(content='Likes coffee',entity=None,span='I like coffee',expression='explicit'):
    return RawUnit(content=content,polarity='prefer',scope='drinks',domain='food',nature='taste',
                   expression=expression,condition=Condition(entity=entity),span=span,turn_idx=0)


class FakeAPI:
    def __init__(self):self.output={};self.calls=[];self.relation='contradicts';self.fail=False
    async def embed(self,texts):return np.stack([vec() for _ in texts]) if texts else np.empty((0,1024))
    async def ask(self,role,system,value,schema=None,tag='misc',validate=None,**kwargs):
        self.calls.append((tag,value))
        if self.fail:raise APIError('simulated failure')
        if tag=='relation':
            result=(IndexedRelations(relations=[dict(index=u['index'],relation=self.relation) for u in value['existing']]) if schema==IndexedRelations
                    else Relations(relations=[dict(unit_id=u['id'],relation=self.relation) for u in value['existing']]))
        elif tag=='relations_batch':
            result=(IndexedRelations(relations=[dict(index=p['index'],relation=self.relation) for p in value['pairs']]) if schema==IndexedRelations
                    else BatchRelations(relations=[dict(new_id=p['new']['id'],unit_id=p['existing']['id'],relation=self.relation) for p in value['pairs']]))
        elif tag=='extract':
            if schema==SourceExtraction:
                sources=[e for m in value['messages'] if m['role']=='user' for e in m['excerpts'] if 'source_id' in e]
                result=SourceExtraction(units=[dict(**raw().model_dump(exclude={'span','turn_idx'}),source_id=sources[0]['source_id'])]) if sources else SourceExtraction()
            else:result=Extraction(units=[raw(span=value['messages'][0]['content'])])
        elif tag=='analyze':result=ParsedQuery(situation=Condition(entity='Alice'),entities=['Alice'],probes=['drinks']*6)
        elif tag=='gate':
            result=(IndexedGate(decisions=[dict(index=c['index'],p_apply=.9,reason='relevant') for c in value['candidates']]) if schema==IndexedGate
                    else GateScores(decisions=[dict(unit_id=c['unit']['id'],p_apply=.9,reason='relevant') for c in value['candidates']]))
        elif tag=='normalize' and schema==PreferenceFields:
            result=PreferenceFields(**self.output[tag].units[0].model_dump(exclude={'span','turn_idx','expression'}))
        else:result=self.output[tag]
        if validate:validate(result)
        return result


def test_gate_schema_and_coverage():
    for probability in (-.1,1.1,float('nan'),float('inf')):
        with pytest.raises(ValidationError):GateScores(decisions=[dict(unit_id='x',p_apply=probability,reason='a')])
    out=GateScores(decisions=[dict(unit_id='x',p_apply=.5,reason='a')])
    with pytest.raises(ValueError):validate_coverage(out.decisions,['x','y'])
    with pytest.raises(ValueError):validate_coverage(out.decisions*2,['x'])


def test_store_immutability_order_user_and_vectors(tmp_path):
    s=Store(tmp_path/'u.sqlite','u');ep=episode([dict(role='user',content='a')])
    assert not s.stage(ep)
    with pytest.raises(ValueError):s.stage(episode([dict(role='user',content='b')],1))
    changed=ep.model_copy(update={'messages':[]})
    with pytest.raises(ValueError):s.stage(changed)
    s.db.execute('UPDATE episodes SET done=1');s.db.commit()
    assert s.stage(ep)
    with pytest.raises(ValueError):Store(tmp_path/'u.sqlite','other')
    with pytest.raises(ValueError):s.put_vector('unit','a',np.ones(3))
    s.close()


def test_negation_pending_distinct_support_and_replay(tmp_path):
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');api=FakeAPI();spm=SPM('u',s,api)
        for order in (0,1):
            ep=episode([dict(role='user',content='I like coffee')],order)
            u=make_unit('u',raw(),ep,spm.cfg)
            await spm._upsert(u,vec(),ep)
        s.db.commit();old=s.all_units()[0]
        assert old.n_support==2
        ep=episode([dict(role='user',content='Coffee tastes bad')],2)
        new=make_unit('u',raw('Dislikes coffee',span='Coffee tastes bad',expression='feedback'),ep,spm.cfg)
        await spm._upsert(new,vec(),ep);s.db.commit()
        assert len(s.all_units())==2
        pending=s.get(new.id);assert pending.status=='pending' and pending.challenges==[old.id]
        await spm._upsert(new,vec(),ep);s.db.commit()
        assert s.get(new.id).n_support==1 and s.get(new.id).status=='pending'
        ep2=episode([dict(role='user',content='Coffee tastes bad')],3)
        await spm._upsert(make_unit('u',raw('Dislikes coffee',span='Coffee tastes bad',expression='feedback'),ep2,spm.cfg),vec(),ep2)
        s.db.commit()
        assert s.get(old.id).status=='superseded'
        assert len(s.all_units())==1 and s.get(new.id).n_support==2
        assert any(t[0]=='relation' for t in api.calls)  # Identical embeddings did not merge negation.
        s.close()
    asyncio.run(scenario())


def test_context_branch_and_no_anticipation(tmp_path):
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');api=FakeAPI();spm=SPM('u',s,api,replace(SPMConfig(),use_anticipation=False))
        for i,name in enumerate(('Alice','Bob')):
            ep=episode([dict(role='user',content='I like coffee')],i)
            await spm._upsert(make_unit('u',raw(entity=name),ep,spm.cfg),vec(),ep)
        s.db.commit();assert len(s.all_units())==2
        assert not any(t[0]=='relation' for t in api.calls)
        query=await spm.analyze('A drink for Alice')
        assert query.probes==['A drink for Alice'] and query.entities==['alice']
        spm.cfg=SPMConfig();query=await spm.analyze('A drink for Alice');assert len(query.probes)<=6
        candidates=[Candidate(u,.9) for u in s.all_units()]
        decisions,_=await spm.gate('A drink for Alice',query,candidates,[])
        mapping={u.id:u for u in s.all_units()}
        assert all(d.decision=='suppress' for d in decisions if mapping[d.unit_id].condition.entity=='bob')
        s.close()
    asyncio.run(scenario())


def test_pending_gate_guards_and_render(tmp_path):
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');api=FakeAPI();spm=SPM('u',s,api)
        a=make_unit('u',raw(),episode([dict(role='user',content='I like coffee')]),spm.cfg)
        b=make_unit('u',raw('Dislikes coffee'),episode([dict(role='user',content='I like coffee')],1),spm.cfg)
        b.status='pending';b.challenges=[a.id]
        c=[Candidate(a,.9),Candidate(b,.9)]
        decisions,_=await spm.gate('coffee',ParsedQuery(probes=['coffee']),c,[])
        assert all(d.decision=='uncertain' for d in decisions)
        assert not spm.render(c,decisions)
        s.close()
    asyncio.run(scenario())


def test_failed_ingestion_retry_and_completed_idempotency(tmp_path):
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');api=FakeAPI();spm=SPM('u',s,api)
        ep=episode([dict(role='user',content='I like coffee')]);api.fail=True
        with pytest.raises(APIError):await spm.ingest(ep)
        assert not s.all_units()
        api.fail=False;await spm.ingest(ep)
        before=len(api.calls);await spm.ingest(ep)
        assert len(api.calls)==before and len(s.all_units())==1
        assert s.all_units()[0].n_support==1
        s.close()
    asyncio.run(scenario())


def test_gateway_coalescing_schema_repair_embedding_and_budget(tmp_path):
    async def scenario():
        endpoint=Endpoint('test','SECRET','https://example.com/v1');calls=[]
        async def transport(request):
            data=json.loads(request.content);calls.append(data)
            await asyncio.sleep(.01)
            if request.url.path.endswith('embeddings'):
                rows=[dict(index=i,embedding=(vec()*2).tolist()) for i,_ in enumerate(data['input'])]
                return httpx.Response(200,json=dict(data=list(reversed(rows)),usage=dict(total_tokens=3,prompt_tokens=3)))
            content='{"decisions": []}' if len(calls)==1 else '{"decisions": [{"unit_id":"x","p_apply":0.9,"reason":"ok"}]}'
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content=content))],usage=dict(total_tokens=10,prompt_tokens=8,completion_tokens=2)))
        api=Gateway(dict(chat=endpoint,embed=endpoint),tmp_path/'cache.sqlite',max_calls=10,max_tokens=100000)
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        def coverage(out):validate_coverage(out.decisions,['x'])
        results=await asyncio.gather(*(api.ask('chat','system',{'x':1},GateScores,'gate',coverage) for _ in range(8)))
        assert len(calls)==2 and api.schema_failures['gate']==1
        assert all(r.decisions[0].p_apply==.9 for r in results)
        await api.ask('chat','system',{'x':1},GateScores,'gate',coverage);assert len(calls)==2
        vectors=await api.embed(['a','b','a']);assert vectors.shape==(3,1024) and np.allclose(np.linalg.norm(vectors,axis=1),1)
        await api.embed(['b','a']);assert len(calls)==3
        assert api.usage()['http_requests']==3 and api.usage()['known_tokens']==23
        api.max_calls=3
        with pytest.raises(BudgetExceeded):await api.ask('chat','system',{'x':2},GateScores,'gate',coverage)
        assert len(calls)==3
        await api.close()
    asyncio.run(scenario())


def test_gateway_provider_error_redacts_secrets(tmp_path):
    async def scenario():
        endpoint=Endpoint('test','TOP_SECRET','https://example.com/v1');api=Gateway(dict(chat=endpoint),tmp_path/'cache.sqlite',retries=1)
        await api.http.aclose()
        api.http=httpx.AsyncClient(transport=httpx.MockTransport(lambda _:httpx.Response(401,text='TOP_SECRET')))
        with pytest.raises(APIError) as error:await api.ask('chat','s','u')
        assert 'TOP_SECRET' not in str(error.value) and 'TOP_SECRET' not in json.dumps(api.usage())
        assert api.usage()['http_requests']==1
        await api.close()
    asyncio.run(scenario())


def test_local_adapters_gold_isolation_splits_and_temporal_protocol():
    root=Path(__file__).parents[1]/'datasets'
    if not (root/'cupid'/'test.parquet').exists():pytest.skip('Local benchmark datasets not installed')
    cupid=load_cases(root,'cupid');c=cupid[0]
    assert c.public.provenance==('prior_interactions.*.dialogue.role/content','current_request')
    assert all(set(m.model_dump())=={'role','content'} for e in c.public.episodes for m in e.messages)
    dev=partition(cupid,'dev');test=partition(cupid,'test')
    assert not {c.group for c in dev}&{c.group for c in test}
    perma=load_cases(root,'perma')
    assert all(not c.public.episodes for c in perma if c.stratum=='type1')
    assert all(e.time[:10]<=json.loads(Path(c.sources[1]).read_text())['question_date'][:10] for c in perma for e in c.public.episodes)
    prefs=load_cases(root,'prefeval')
    assert len(prefs)==3000 and len({c.stratum for c in select(prefs,3)})==3
    assert all(c.public.episodes[0].messages[0].content==c.gold['preference'] for c in prefs if c.stratum=='explicit')
    assert not {c.group for c in partition(prefs,'dev')}&{c.group for c in partition(prefs,'test')}
    bench=load_cases(root,'benchpres');assert len(bench)==390
    assert all(c.public.given and len(c.public.episodes[0].messages)==5 and 'UserKnowledgeMemories' not in c.public.query for c in bench)


def test_matching_frontier_and_failure_accounting():
    m=checklist_metrics([[0,1],[0]],2)
    assert m['checklist_f1']==1
    assert checklist_metrics([[0],[0]],1)['matched']==1
    trace=dict(given_indices=[0,2]);gold=dict(labels=[True,False,False])
    metrics,gate=bench_scores(trace,gold,'b2')
    assert metrics['gate_AAR']==1 and metrics['gate_MR']==.5
    good=dict(benchmark='benchpres',method='v4',status='ok',group='a',stratum='finance',evaluation=dict(metrics=metrics,gate_evaluation=gate))
    failed=dict(benchmark='benchpres',method='v4',status='error',group='b',stratum='finance')
    summary=aggregate([good,failed])['benchpres/v4']
    assert summary['failed']==1 and summary['complete_run_metrics'] is None
    assert frontier([good],[.5])[0]['gate_MR']==.5


def test_batch_relations_and_transaction_rollback(tmp_path):
    class TwoUnits(FakeAPI):
        async def ask(self,role,system,value,schema=None,tag='misc',validate=None,**kwargs):
            if tag!='extract':return await super().ask(role,system,value,schema,tag,validate,**kwargs)
            self.calls.append((tag,value))
            units=[raw('Likes coffee',span='coffee'),raw('Likes tea',span='tea')]
            result=(SourceExtraction(units=[dict(**u.model_dump(exclude={'span','turn_idx'}),source_id=0) for u in units])
                    if schema==SourceExtraction else Extraction(units=units))
            if validate:validate(result)
            return result
    class FailingSPM(SPM):
        async def _upsert(self,new,vector,ep,known_relations=None):
            await super()._upsert(new,vector,ep,known_relations)
            if new.content=='Likes tea':raise RuntimeError('failed after partial update')
    async def scenario():
        s=Store(tmp_path/'rollback.sqlite','u');api=TwoUnits();api.relation='compatible'
        spm=FailingSPM('u',s,api);ep=episode([dict(role='user',content='I like coffee and tea')])
        with pytest.raises(RuntimeError):await spm.ingest(ep)
        assert not s.all_units(None)
        assert s.db.execute('SELECT COUNT(*) FROM vecs').fetchone()[0]==0
        assert s.db.execute('SELECT COUNT(*) FROM events').fetchone()[0]==0
        assert s.db.execute('SELECT done FROM episodes').fetchone()[0]==0
        assert sum(tag=='relations_batch' for tag,_ in api.calls)==1
        assert not any(tag=='relation' for tag,_ in api.calls)
        await SPM('u',s,api).ingest(ep)
        assert len(s.all_units())==2
        s.close()
    asyncio.run(scenario())


def test_prefix_copy_preserves_past_and_reuses_extraction(tmp_path):
    async def scenario():
        api=FakeAPI();cfg=SPMConfig(use_gate=False,use_anticipation=False)
        exp=Experiment(api,tmp_path,dict(b4=cfg))
        episodes=tuple(episode([dict(role='user',content='I like coffee')],i) for i in range(3))
        first=MemoryInput('u',episodes[:2],'coffee')
        later=MemoryInput('u',episodes,'coffee')
        (short,user),(same,_) = await asyncio.gather(exp.memory(first,'b4'),exp.memory(first,'b4'))
        assert short==same
        long,_=await exp.memory(later,'b4')
        assert sum(tag=='extract' for tag,_ in api.calls)==3
        a=Store(short,user);b=Store(long,user)
        assert len(a.episodes())==2 and len(b.episodes())==3
        assert len(a.all_units())==2 and len(b.all_units())==3
        a.close();b.close()
        other,_=await exp.memory(MemoryInput('other',episodes[:1],'coffee'),'b4')
        assert other!=short
        assert sum(tag=='extract' for tag,_ in api.calls)==4
    asyncio.run(scenario())


def test_sample_resume_skips_successful_case(tmp_path):
    async def scenario():
        api=FakeAPI();exp=Experiment(api,tmp_path,dict(b1=SPMConfig()))
        public=MemoryInput('u',(episode([dict(role='user',content='I like factual prose')]),),'write a letter',given=True)
        case=Case('bench/one','benchpres','u','finance',public,dict(labels=[True]))
        await exp.run_case(case,'b1')
        path=next((tmp_path/'samples').glob('*.json'));first=json.loads(path.read_text())
        await exp.run_case(case,'b1');second=json.loads(path.read_text())
        assert first==second and first['attempts']==1 and first['evaluation']['metrics']['gate_AAR']==1
        assert not api.calls
    asyncio.run(scenario())


def test_annotations_sentinels_are_never_memory_inputs(tmp_path):
    import pyarrow as pa
    import pyarrow.parquet as pq
    (tmp_path/'cupid').mkdir()
    source=dict(persona_id='u',instance_type='changing',current_request='Write to Board X',
        current_context_factor='SECRET_FACTOR',current_contextual_preference='SECRET_PREFERENCE',
        current_checklist=['SECRET_CHECKLIST'],prior_interactions=[dict(context_factor='SECRET_OLD_FACTOR',
        contextual_preference='SECRET_OLD_PREFERENCE',dialogue=[dict(role='user',content='Use short sentences for Board X')])])
    pq.write_table(pa.Table.from_pylist([source]),tmp_path/'cupid'/'test.parquet')
    case=load_cases(tmp_path,'cupid')[0]
    assert 'SECRET_' not in json.dumps(case.public.dump())
    assert 'SECRET_' in json.dumps(case.gold)
    perma=tmp_path/'perma';tasks=perma/'tasks'/'u';meta=perma/'evaluation'/'u'/'meta'/'overall'
    tasks.mkdir(parents=True);meta.mkdir(parents=True)
    item=dict(task_id='task',type=2,context=[[[dict(role='user',content='I like tea')],'2023-01-01 Morning']],
              preferences=['SECRET_PROFILE'],affinity_links=['SECRET_LINK'],user_use_topic_dialog='SECRET_SIMULATOR')
    (tasks/'input_data_c.json').write_text(json.dumps(dict(overall=[item])))
    (meta/'task_2.json').write_text(json.dumps(dict(question='What drink?',question_date='2023-01-02',options='A: tea\nB: coffee',gold_label='A')))
    case=load_cases(tmp_path,'perma')[0]
    assert 'SECRET_' not in json.dumps(case.public.dump()) and 'gold_label' not in json.dumps(case.public.dump())


def test_extractor_rejects_assistant_evidence(tmp_path):
    class AssistantEvidence(FakeAPI):
        async def ask(self,role,system,value,schema=None,tag='misc',validate=None,**kwargs):
            if tag=='extract':
                result=(SourceExtraction(units=[dict(**raw().model_dump(exclude={'span','turn_idx'}),source_id=1)])
                        if schema==SourceExtraction else Extraction(units=[raw(span='I like coffee').model_copy(update=dict(turn_idx=1))]))
                if validate:validate(result)
                return result
            return await super().ask(role,system,value,schema,tag,validate,**kwargs)
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');spm=SPM('u',s,AssistantEvidence())
        ep=episode([dict(role='user',content='Recommend a drink'),dict(role='assistant',content='I like coffee')])
        with pytest.raises(ValueError):await spm.ingest(ep)
        assert not s.all_units()
        s.close()
    asyncio.run(scenario())


def test_truncated_response_expands_budget_and_caches_valid_only(tmp_path):
    async def scenario():
        endpoint=Endpoint('model','SECRET','https://example.com/v1');requests=[]
        def transport(request):
            data=json.loads(request.content);requests.append(data)
            truncated=len(requests)==1
            content='' if truncated else '{"decisions":[]}'
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='length' if truncated else 'stop',message=dict(content=content))],
                usage=dict(prompt_tokens=10,completion_tokens=64 if truncated else 5,total_tokens=74 if truncated else 15)))
        api=Gateway(dict(chat=endpoint),tmp_path/'cache.sqlite',max_output=64)
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        await api.ask('chat','system','query',GateScores,'gate')
        assert [r['max_tokens'] for r in requests]==[64,128]
        await api.ask('chat','system','query',GateScores,'gate');assert len(requests)==2
        assert api.usage()['schema_failures']['gate']==1
        await api.close()
    asyncio.run(scenario())


def test_inflight_request_is_persisted_before_network_completion(tmp_path):
    async def scenario():
        endpoint=Endpoint('model','SECRET','https://example.com/v1');started=asyncio.Event()
        async def transport(request):
            started.set();await asyncio.Event().wait()
        api=Gateway(dict(chat=endpoint),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        task=asyncio.create_task(api.ask('chat','s','u'))
        await started.wait()
        row=json.loads(api.db.execute('SELECT json FROM requests').fetchone()[0])
        assert row['reserved_tokens']>0 and not row['usage_complete']
        # Recovered usage remains explicit about unknown billing after interruption.
        usage=api.usage('different');assert usage['recovered_from_ledger']
        task.cancel();await asyncio.gather(task,return_exceptions=True)
        usage=await api.close()
        assert usage['http_requests']==1 and not usage['by_tag']['chat:misc']['usage_complete']
    asyncio.run(scenario())


def test_preclassified_future_relations_do_not_write_future_preferences(tmp_path):
    class Named(FakeAPI):
        async def ask(self,role,system,value,schema=None,tag='misc',validate=None,**kwargs):
            if tag=='extract':
                if schema==SourceExtraction:
                    source=value['messages'][0]['excerpts'][0]
                    result=SourceExtraction(units=[dict(**raw(content=source['text']).model_dump(exclude={'span','turn_idx'}),source_id=source['source_id'])])
                else:
                    message=value['messages'][0]['content']
                    result=Extraction(units=[raw(content=message,span=message)])
                if validate:validate(result)
                return result
            return await super().ask(role,system,value,schema,tag,validate,**kwargs)
    async def scenario():
        s=Store(tmp_path/'s.sqlite','u');api=Named();spm=SPM('u',s,api)
        episodes=[episode([dict(role='user',content=x)],i) for i,x in enumerate(['I prefer coffee','I prefer tea'])]
        prepared=await asyncio.gather(*(spm.prepare(ep) for ep in episodes))
        units=[u for data in prepared for u in data['units']]
        relations=await spm._batch_relations(units,np.concatenate([data['vectors'] for data in prepared]))
        for data in prepared:data['relations']=relations
        await spm.ingest(episodes[0],prepared[0])
        assert len(s.all_units())==1 and s.all_units()[0].content=='I prefer coffee'
        assert len(s.episodes())==1
        await spm.ingest(episodes[1],prepared[1])
        assert len(s.all_units())==1 and s.all_units()[0].content=='I prefer tea'
        assert len(s.all_units(None))==2
        assert not any(tag=='relation' for tag,_ in api.calls)
        s.close()
    asyncio.run(scenario())


def test_given_preference_owner_is_not_a_context_entity(tmp_path):
    async def scenario():
        s=Store(tmp_path/'s.sqlite','Jeffery Day');api=FakeAPI();spm=SPM('Jeffery Day',s,api)
        ep=episode([dict(role='user',content='Jeffery likes bold formatting')])
        api.output['normalize']=Extraction(units=[raw(content='The user likes bold formatting',entity='Jeffery',span='Jeffery likes bold formatting')])
        with pytest.raises(ValueError):await spm.install_given(ep)
        assert not s.all_units()
        api.output['normalize'].units[0].condition.entity=None
        await spm.install_given(ep)
        assert len(s.all_units())==1 and s.all_units()[0].condition.entity is None
        assert api.calls[-1][1]['owner_name']=='Jeffery Day'
        s.close()
    asyncio.run(scenario())


def test_query_waits_for_failed_write_rollback(tmp_path):
    ready,release=None,None
    class PausedFailure(SPM):
        async def _upsert(self,new,vector,ep,known_relations=None):
            await super()._upsert(new,vector,ep,known_relations)
            ready.set();await release.wait()
            raise RuntimeError('fail after partial write')
    async def scenario():
        nonlocal ready,release
        ready,release=asyncio.Event(),asyncio.Event()
        s=Store(tmp_path/'s.sqlite','u');api=FakeAPI();spm=PausedFailure('u',s,api)
        writing=asyncio.create_task(spm.ingest(episode([dict(role='user',content='I like coffee')])))
        await ready.wait()
        assert len(s.all_units())==1  # The connection can see its own uncommitted write.
        reading=asyncio.create_task(spm.query('Coffee for Alice'))
        await asyncio.sleep(.01)
        assert not reading.done() and not any(tag=='analyze' for tag,_ in api.calls)
        release.set()
        with pytest.raises(RuntimeError):await writing
        context,trace=await reading
        assert not context and trace['candidates']==[] and not s.all_units()
        s.close()
    asyncio.run(scenario())


def test_read_ablations_share_memory_but_write_ablations_do_not(tmp_path):
    async def scenario():
        api=FakeAPI();api.relation='same'
        configs={m:VARIANTS[m] for m in ['v4','v3','no_anticipation','no_condition']}
        exp=Experiment(api,tmp_path,configs)
        public=MemoryInput('u',(episode([dict(role='user',content='I like coffee')]),),'coffee')
        paths=await asyncio.gather(*(exp.memory(public,m) for m in ['v4','v3','no_anticipation']))
        assert len({p for p,_ in paths})==1
        assert sum(tag=='extract' for tag,_ in api.calls)==1
        other,_=await exp.memory(public,'no_condition')
        assert other!=paths[0][0] and sum(tag=='extract' for tag,_ in api.calls)==2
        profile=json.loads(Path(paths[0][0]).with_suffix('.profile.json').read_text())
        assert profile['episodes_written']==1 and profile['prepare_seconds']>=0
        assert VARIANTS['v4'].write_signature()!=replace(VARIANTS['v4'],sim_cand=.8).write_signature()
    asyncio.run(scenario())


def test_paired_bootstrap_uses_same_cases_and_reports_failures():
    def row(case,method,score,group,status='ok'):
        return dict(case_id=case,case_digest=case,benchmark='perma',method=method,group=group,status=status,
                    evaluation=dict(metrics=dict(mcq_accuracy=score)))
    rows=[row('a','v4',1,'u'),row('a','b1',0,'u'),row('b','v4',0,'v'),row('b','b1',1,'v'),
          row('c','v4',1,'w','error'),row('c','b1',1,'w')]
    out=paired_comparisons(rows,resamples=100)['perma/v4_vs_b1']
    assert out['paired_cases']==2 and out['excluded_or_failed_cases']==1 and out['groups']==2
    assert out['metrics']['mcq_accuracy']['delta']==0
    assert out['metrics']['mcq_accuracy']['paired_cluster_95ci']==[-1,1]
    changed=[dict(r,case_digest='wrong') if r['case_id']=='a' and r['method']=='v4' else r for r in rows]
    with pytest.raises(ValueError):paired_comparisons(changed,resamples=100)


def test_paired_bench_rates_are_micro_not_mean_ratios():
    rows=[]
    for case,positive,applied in [('a',1,1),('b',4,0)]:
        for method in ['v4','b1']:
            metrics=dict(appropriate_total=positive,appropriate_applied=applied if method=='v4' else positive,
                         inappropriate_total=1,inappropriate_applied=0)
            rows.append(dict(case_id=case,case_digest=case,benchmark='benchpres',method=method,group='one',status='ok',
                             evaluation=dict(metrics=metrics)))
    out=paired_comparisons(rows)['benchpres/v4_vs_b1']['metrics']['gate_AAR']
    assert out['method_score']==.2 and out['delta']==-.8 and out['paired_cluster_95ci'] is None


def test_resource_filter_is_public_and_applied_before_sampling(tmp_path,monkeypatch):
    from types import SimpleNamespace
    import spm.runner as runner
    short=Case('short','perma','u','type2',MemoryInput('u',(),'q'),{'labels':['A']})
    long=Case('long','perma','u','type2',MemoryInput('u',tuple(episode([dict(role='user',content='x')],i) for i in range(4)),'q'),{'labels':['B']})
    monkeypatch.setattr(runner,'load_cases',lambda *args:[short,long])
    args=SimpleNamespace(benchmark='perma',datasets=tmp_path,perma_mode='sd_clean',perma_types=[2],
                         split='all',seed=42,limit=10,perma_limit=1,max_history_episodes=2)
    selected,inventory=selected_cases(args)
    assert selected==[short] and inventory['perma']['resource_eligible']['cases']==1
    assert inventory['perma']['eligible']['cases']==2


def test_payment_block_skips_network_and_budget_but_keeps_valid_cache(tmp_path):
    async def scenario():
        endpoint=Endpoint('model','SECRET','https://example.com/v1');calls=[]
        def transport(request):
            data=json.loads(request.content);calls.append(data)
            if data['messages'][-1]['content']=='known':
                return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content='OK'))],usage=dict(total_tokens=1)))
            return httpx.Response(402,text='SECRET: insufficient balance')
        api=Gateway(dict(chat=endpoint,cheap=endpoint),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        assert await api.ask('chat','s','known')=='OK'
        with pytest.raises(AccountUnavailable):await api.ask('chat','s','new')
        bound=api.tokens_bound
        with pytest.raises(AccountUnavailable):await api.ask('cheap','s','another')
        assert len(calls)==2 and api.tokens_bound==bound
        assert await api.ask('chat','s','known')=='OK' and len(calls)==2
        assert api.usage()['unknown_usage_requests']==1
        assert 'SECRET' not in json.dumps(api.usage())
        await api.close()
    asyncio.run(scenario())


def test_payment_block_does_not_block_other_model_or_account(tmp_path):
    async def scenario():
        first=Endpoint('model','SECRET1','https://example.com/v1')
        other_key=Endpoint('model','SECRET2','https://example.com/v1')
        other_model=Endpoint('other','SECRET1','https://example.com/v1')
        def transport(request):
            payload=json.loads(request.content)
            if payload['model']=='model' and request.headers['authorization']=='Bearer SECRET1':return httpx.Response(402)
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content='OK'))],usage=dict(total_tokens=1)))
        api=Gateway(dict(chat=first,cheap=other_key,judge=other_model),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        with pytest.raises(AccountUnavailable):await api.ask('chat','s','q')
        assert await api.ask('cheap','s','q')=='OK' and await api.ask('judge','s','q')=='OK'
        assert api.usage()['http_requests']==3
        await api.close()
    asyncio.run(scenario())


def test_schema_repair_explains_inconsistent_grade_without_dumping_input(tmp_path):
    from spm.evaluation import PrefGrade
    async def scenario():
        requests=[]
        def transport(request):
            data=json.loads(request.content);requests.append(data)
            grade=dict(follows_preference=len(requests)==1,preference_unaware=False,
                       inconsistent=False,hallucinated_preference=True,unhelpful=False,
                       reason='PRIVATE_VALIDATION_INPUT')
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(grade)))],
                usage=dict(total_tokens=10,prompt_tokens=8,completion_tokens=2)))
        api=Gateway(dict(judge=Endpoint('test','SECRET','https://example.com/v1')),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        result=await api.ask('judge','system','question',PrefGrade,'pref_judge')
        assert not result.follows_preference and result.hallucinated_preference
        feedback=requests[1]['messages'][-1]['content']
        assert 'Success requires all four failure flags false' in feedback
        assert 'PRIVATE_VALIDATION_INPUT' not in feedback and "'input':" not in feedback
        assert len(requests)==2
        await api.close()
    asyncio.run(scenario())


def test_gate_policy_changes_only_read_prompt_and_shares_written_memory(tmp_path):
    from spm import prompts
    class PromptAPI(FakeAPI):
        def __init__(self):super().__init__();self.systems=[]
        async def ask(self,role,system,value,*args,**kwargs):
            self.systems.append(system)
            return await super().ask(role,system,value,*args,**kwargs)
    async def scenario():
        api=PromptAPI();configs={m:VARIANTS[m] for m in ('v4','v5')}
        assert configs['v4'].write_signature()==configs['v5'].write_signature()
        assert configs['v4'].gate_policy=='legacy'
        with pytest.raises(ValueError):SPMConfig(gate_policy='unknown')
        ep=episode([dict(role='user',content='I like coffee')])
        public=MemoryInput(user_id='u',episodes=[ep],query='coffee',given=False,provenance=[])
        exp=Experiment(api,tmp_path/'run',configs)
        old=await exp.memory(public,'v4');new=await exp.memory(public,'v5')
        assert old==new
        store=Store(old[0],'u')
        candidates=[Candidate(u,.9) for u in store.all_units()]
        for method in ('v4','v5'):
            await SPM('u',store,api,configs[method]).gate('coffee',ParsedQuery(probes=['coffee']),candidates,[])
        assert api.systems[-2:]==[prompts.GATE,prompts.GATE_ANSWER_FILTER]
        store.close()
    asyncio.run(scenario())


def test_case_file_is_exact_and_rejects_filtered_or_duplicate_ids(tmp_path):
    from spm.__main__ import parser
    dataset_root=tmp_path/'datasets'
    explicit=dataset_root/'PrefEval'/'explicit_preference'
    explicit.mkdir(parents=True)
    (explicit/'synthetic.json').write_text(json.dumps([
        dict(question=f'Synthetic request {i}',preference='Use a concise answer') for i in range(3)
    ]))
    args=parser().parse_args(['run','--benchmark','prefeval','--split','all','--limit','0',
                              '--datasets',str(dataset_root)])
    args.limit=None
    cases=load_cases(dataset_root,'prefeval')
    ids=[cases[-1].id,cases[0].id]
    f=tmp_path/'cases.json';f.write_text(json.dumps(ids));args.case_file=str(f)
    picked,_=selected_cases(args)
    assert [c.id for c in picked]==ids
    args.limit=1
    with pytest.raises(ValueError,match='missing'):selected_cases(args)
    f.write_text(json.dumps([ids[0],ids[0]]))
    with pytest.raises(ValueError,match='unique'):selected_cases(args)


def test_new_default_matches_compact_v6_and_legacy_remains_explicit():
    from spm.__main__ import parser
    assert parser().parse_args(['run']).methods==['v6_compact']
    assert SPMConfig().dump()==VARIANTS['v6_compact'].dump()
    assert VARIANTS['v6'].tool_protocol=='legacy' and VARIANTS['v6'].use_raw_evidence
    assert VARIANTS['v4'].gate_policy=='legacy'
    assert VARIANTS['v5'].write_signature()==VARIANTS['v6'].write_signature()
    # The situational gate is opt-in, read-only, and shares written memory with v6_compact.
    assert VARIANTS['v6_situational'].gate_policy=='situational'
    assert VARIANTS['v6_situational'].write_signature()==VARIANTS['v6_compact'].write_signature()
    assert VARIANTS['v6_situational'].dump()=={**VARIANTS['v6_compact'].dump(),'gate_policy':'situational'}


def test_situational_gate_prompt_extends_answer_filter():
    from spm import prompts
    assert prompts.GATE_SITUATIONAL.startswith(prompts.GATE_ANSWER_FILTER)
    assert 'Topical relevance is not applicability' in prompts.GATE_SITUATIONAL


def test_raw_evidence_recovers_empty_extraction_with_bound_and_no_given_leak(tmp_path):
    class Empty(FakeAPI):
        async def ask(self,role,system,value,schema=None,tag='misc',validate=None,**kwargs):
            if tag=='extract':return Extraction(units=[])
            if tag=='analyze':return ParsedQuery(probes=[value['request']])
            return await super().ask(role,system,value,schema,tag,validate,**kwargs)
    async def scenario():
        store=Store(tmp_path/'bank.sqlite','u');api=Empty()
        ep=episode([dict(role='user',content='I only choose German authors. '+('evidence '*20))])
        memory=SPM('u',store,api,VARIANTS['v5']);await memory.ingest(ep)
        assert not store.all_units()
        before=store.snapshot()
        old,trace=await memory.query('Suggest a novel');assert old==''
        memory.cfg=replace(VARIANTS['v6'],raw_evidence_chars=80)
        new,trace=await memory.query('Suggest a novel')
        assert 'Only USER statements' in new and 'German authors' in new
        assert sum(len(x['dialogue_excerpt']) for x in trace['raw_evidence'])<=80
        assert trace['raw_evidence'][0]['truncated'] and trace['raw_evidence'][0]['order']==ep.order
        assert not trace['candidates'] and not trace['decisions']
        given,trace=await memory.query('Suggest a novel',force_all=True)
        assert given=='' and not trace['raw_evidence']
        assert store.snapshot()==before
        assert not any(tag=='gate' for tag,_ in api.calls)
        store.close()
    asyncio.run(scenario())


def test_grounding_repair_supplies_source_without_accepting_paraphrased_evidence(tmp_path):
    async def scenario():
        requests=[];source='Cross-referencing with printed editions is crucial.'
        def transport(request):
            payload=json.loads(request.content);requests.append(payload)
            if request.url.path.endswith('embeddings'):
                return httpx.Response(200,json=dict(data=[dict(index=i,embedding=vec().tolist()) for i,_ in enumerate(payload['input'])],usage=dict(total_tokens=1)))
            repair=sum(r.get('response_format') is not None for r in requests)>1
            unit=raw('Requires cross-referencing',span=source if repair else 'Cross-referencing with at least two printed editions is crucial.').model_dump(mode='json')
            return httpx.Response(200,json=dict(choices=[dict(finish_reason='stop',message=dict(content=json.dumps(dict(units=[unit]))))],usage=dict(total_tokens=1)))
        endpoint=Endpoint('test','SECRET','https://example.com/v1')
        api=Gateway(dict(cheap=endpoint,embed=endpoint),tmp_path/'cache.sqlite')
        await api.http.aclose();api.http=httpx.AsyncClient(transport=httpx.MockTransport(transport))
        store=Store(tmp_path/'bank.sqlite','u');spm=SPM('u',store,api,VARIANTS['v6'])
        result=await spm.prepare(episode([dict(role='user',content=source)]))
        assert result['units'][0].evidence[0].span==source
        feedback=requests[1]['messages'][-1]['content']
        assert source in feedback and 'omit the unit' in feedback
        assert api.schema_failures['extract']==1
        await api.close();store.close()
    asyncio.run(scenario())
