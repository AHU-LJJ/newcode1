"""Concurrent cases, sequential user histories, atomic checkpoints and resume."""
import asyncio
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import sqlite3
import sys
import httpx
import numpy
import pyarrow
import pydantic
import time
from .adapters import describe,load_cases,partition,select
from .api import Gateway,REQUEST_CONTEXT
from .baselines import EntityMemory,FlatMemory,raw_context
from .config import endpoints,VARIANTS,SPMConfig
from .evaluation import aggregate,evaluate,frontier,paired_comparisons
from .models import fingerprint
from .pipeline import SPM
from .conservative import ConservativeMemory
from .typed_graph import TypedGraphMemory
from .store import Store


def atomic_json(path,value):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(path.suffix+'.tmp')
    tmp.write_text(json.dumps(value,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
    tmp.replace(path)


def file_hash(path):
    digest=hashlib.sha256()
    with open(path,'rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''):digest.update(block)
    return digest.hexdigest()


def selected_cases(args):
    case_file=getattr(args,'case_file',None)
    requested=json.loads(Path(case_file).read_text()) if case_file else None
    if requested is not None and (not isinstance(requested,list) or not requested or
            any(not isinstance(x,str) for x in requested) or len(set(requested))!=len(requested)):
        raise ValueError('Case file must be a nonempty list of unique case IDs')
    benches=['perma','cupid','benchpres'] if args.benchmark=='all' else [args.benchmark]
    cases=[];inventory={}
    for bench in benches:
        all_cases=load_cases(args.datasets,bench,args.perma_mode,args.perma_types)
        eligible=partition(all_cases,args.split,args.seed)
        cap=getattr(args,'max_history_episodes',None)
        resource_pool=[c for c in eligible if cap is None or len(c.public.episodes)<=cap]
        if requested is not None:resource_pool=[c for c in resource_pool if c.id in requested]
        limit=getattr(args,'perma_limit',None) if bench=='perma' else None
        limit=args.limit if limit is None else (limit or None)
        picked=select(resource_pool,limit,args.seed)
        inventory[bench]=dict(local=describe(all_cases),eligible=describe(eligible),selected=describe(picked))
        inventory[bench]['resource_eligible']=describe(resource_pool)
        cases.extend(picked)
    if requested is not None:
        mapping={c.id:c for c in cases}
        if set(mapping)!=set(requested):raise ValueError('Requested case IDs missing after selection; check split, limits and resource cap')
        cases=[mapping[x] for x in requested]
    return cases,inventory


class Experiment:
    def __init__(self,api,out,configs,workers=8):
        self.api,self.out,self.configs=api,Path(out),configs
        self.workers=workers;self.prepared={};self.rows=[]
        self.timelines={};self.memory_profiles={}
        index=self.out/'memory'/'index.json'
        self.memory_index=json.loads(index.read_text()) if index.exists() else {}

    def memory_class(self,method):
        if method=='b4':return EntityMemory
        if self.configs[method].update_policy=='conservative':return ConservativeMemory
        if self.configs[method].update_policy=='typed':return TypedGraphMemory
        return FlatMemory if self.configs[method].memory_manager=='flat' else SPM

    def record_extraction(self,public,cfg,episode,data):
        # Manager-independent input audit. API caching gives both writers the same
        # extraction; validate exact semantic fields, literal evidence and vectors.
        signature={k:getattr(cfg,k) for k in ('use_condition','tool_protocol','window_messages','window_chars','embed_raw')}
        key=fingerprint([public.user_id,signature,episode.model_dump(mode='json')])
        artifact=dict(signature=signature,user=public.user_id,episode_hash=data['episode_hash'],
            units=[u.model_dump(mode='json') for u in data['units']],raw_texts=data['raw_texts'],
            vectors_sha256=hashlib.sha256(numpy.asarray(data['vectors'],dtype='<f4').tobytes()).hexdigest(),
            raw_vectors_sha256=hashlib.sha256(numpy.asarray(data['raw_vectors'],dtype='<f4').tobytes()).hexdigest())
        path=self.out/'extractions'/f'{key}.json'
        if path.exists():
            if json.loads(path.read_text())!=artifact:raise ValueError('Matched extraction changed between managers')
        else:atomic_json(path,artifact)
        return str(path.relative_to(self.out))

    async def memory(self,public,method):
        cfg=self.configs[method]
        write_signature=cfg.write_signature('entity' if method=='b4' else 'spm')
        identity=fingerprint([write_signature,public.user_id,public.given,
                              [e.model_dump(mode='json') for e in public.episodes]])
        if identity not in self.prepared:
            async def build():
                user=public.user_id
                path=self.out/'memory'/f'{identity}.sqlite'
                lock=self.timelines.setdefault((fingerprint(write_signature),user),asyncio.Lock())
                async with lock:
                    return await build_ordered(path,user)
            async def build_ordered(path,user):
                started=time.monotonic()
                profile=dict(write_signature=write_signature,episodes_total=len(public.episodes),
                             prepare_seconds=0.,preclassify_seconds=0.,commit_seconds=0.,
                             install_given_seconds=0.,prefix_episodes=0)
                ids=[ep.id for ep in public.episodes]
                start=0
                if identity not in self.memory_index:
                    prefixes=[entry for entry in self.memory_index.values() if entry['user']==user and entry.get('write_signature')==write_signature
                              and len(entry['episodes'])<=len(ids) and ids[:len(entry['episodes'])]==entry['episodes']]
                    if prefixes and not path.exists():
                        source=max(prefixes,key=lambda e:len(e['episodes']))
                        path.parent.mkdir(parents=True,exist_ok=True)
                        src=sqlite3.connect(self.out/source['path']);dest=sqlite3.connect(path)
                        try:src.backup(dest)
                        finally:src.close();dest.close()
                        profile['prefix_episodes']=len(source['episodes'])
                if identity in self.memory_index:start=len(ids)
                store=Store(path,user)
                klass=self.memory_class(method)
                spm=klass(user,store,self.api,cfg)
                request_token=REQUEST_CONTEXT.set(dict(REQUEST_CONTEXT.get(),stage='write',memory_manager=cfg.memory_manager))
                try:
                    done={r[0] for r in store.db.execute('SELECT id FROM episodes WHERE done=1')}
                    remaining=[ep for ep in public.episodes[start:] if ep.id not in done]
                    profile['episodes_already_done']=len(done)
                    profile['episodes_written']=len(remaining)
                    if public.given:
                        stage=time.monotonic()
                        for ep in remaining:await spm.install_given(ep)
                        profile['install_given_seconds']=time.monotonic()-stage
                    else:
                        for start in range(0,len(remaining),8):
                            block=remaining[start:start+8]
                            stage=time.monotonic()
                            prepared=await asyncio.gather(*(spm.prepare(ep) for ep in block))
                            for ep,data in zip(block,prepared,strict=True):
                                self.record_extraction(public,cfg,ep,data)
                            profile['prepare_seconds']+=time.monotonic()-stage
                            units=[u for data in prepared for u in data['units']]
                            vectors=numpy.concatenate([data['vectors'] for data in prepared])
                            # Relationship labels depend on immutable meanings, not future support
                            # counts. Only the current episode's labels are consumed during writing.
                            stage=time.monotonic()
                            relations=await spm._batch_relations(units,vectors)
                            profile['preclassify_seconds']+=time.monotonic()-stage
                            stage=time.monotonic()
                            for ep,data in zip(block,prepared,strict=True):
                                data['relations']=relations
                                await spm.ingest(ep,data)
                            profile['commit_seconds']+=time.monotonic()-stage
                            print(json.dumps(dict(progress='memory',method=method,user=user,
                                episodes_done=len(done)+start+len(block),episodes_total=len(public.episodes))),flush=True)
                    atomic_json(path.with_suffix('.snapshot.json'),store.snapshot())
                    profile['build_seconds']=time.monotonic()-started
                    self.memory_profiles[path.name]=profile
                    atomic_json(path.with_suffix('.profile.json'),profile)
                    self.memory_index[identity]=dict(user=user,method=method,write_signature=write_signature,
                                                    episodes=ids,path=str(path.relative_to(self.out)))
                    atomic_json(self.out/'memory'/'index.json',self.memory_index)
                    return str(path),user
                finally:
                    REQUEST_CONTEXT.reset(request_token)
                    store.close()
            task=asyncio.create_task(build());self.prepared[identity]=task
        return await self.prepared[identity]

    async def context(self,public,method):
        cfg=self.configs[method]
        if method in ('b0','b1','b2','b2_matched','raw_m8','raw_m9'):
            return await raw_context('b2_matched' if method in {'raw_m8','raw_m9'} else method,public,self.api,cfg)
        path,user=await self.memory(public,method)
        store=Store(path,user)
        klass=self.memory_class(method)
        try:
            spm=klass(user,store,self.api,cfg)
            context,trace=await spm.query(public.query,force_all=public.given)
            trace['memory_file']=str(Path(path).relative_to(self.out))
            trace['memory_profile_file']=str(Path(path).with_suffix('.profile.json').relative_to(self.out))
            trace['unit_status_counts']={s:sum(u.status==s for u in store.all_units(None)) for s in ('active','candidate','pending','superseded')}
            trace['memory_manager']=cfg.memory_manager
            return context,trace
        finally:store.close()

    async def run_case(self,case,method):
        path=self.out/'samples'/f'{fingerprint([case.id,method])}.json'
        prior=json.loads(path.read_text()) if path.exists() else None
        if prior and prior['case_digest']!=case.digest():
            raise ValueError('Case source changed; use a new output directory')
        if prior and prior['status']=='ok':
            self.rows.append(prior);return
        started=time.monotonic()
        request_token=REQUEST_CONTEXT.set(dict(case_id=case.id,method=method,stage='context'))
        row=dict(case_id=case.id,case_digest=case.digest(),benchmark=case.benchmark,method=method,
                 group=case.group,stratum=case.stratum,provenance=list(case.public.provenance),
                 attempts=(prior or {}).get('attempts',0)+1)
        try:
            stage=time.monotonic()
            context,trace=await self.context(case.public,method)
            trace['context']=context
            trace['context_characters']=len(context)
            row['timing']=dict(context_seconds=round(time.monotonic()-stage,3))
            row['trace']=trace
            stage=time.monotonic()
            REQUEST_CONTEXT.set(dict(case_id=case.id,method=method,stage='evaluation'))
            row['evaluation']=await evaluate(case,method,context,trace,self.api)
            row['timing']['evaluation_seconds']=round(time.monotonic()-stage,3)
            row['status']='ok'
        except Exception as error:
            # Provider bodies/credentials never appear in Gateway exceptions.
            row.update(status='error',error=dict(type=type(error).__name__,message=str(error)[:500]))
            if prior:row['previous_error']=prior.get('error')
        row['seconds']=round(time.monotonic()-started,3)
        REQUEST_CONTEXT.reset(request_token)
        atomic_json(path,row);self.rows.append(row)
        print(json.dumps(dict(case=case.id,method=method,status=row['status'],seconds=row['seconds'],
                             **({'error':row['error']} if row['status']=='error' else {})),ensure_ascii=False),flush=True)

    async def run(self,cases,methods):
        jobs=asyncio.Queue()
        for case in cases:
            for method in methods:jobs.put_nowait((case,method))
        async def worker():
            while not jobs.empty():
                case,method=jobs.get_nowait()
                try:await self.run_case(case,method)
                finally:jobs.task_done()
        await asyncio.gather(*(worker() for _ in range(self.workers)))
        return self.rows


async def run(args):
    cases,inventory=selected_cases(args)
    if not cases:raise ValueError('No cases selected')
    configs={method:replace(VARIANTS.get(method,SPMConfig(use_anticipation=False,use_gate=False,embed_raw='none')),
            gate_apply_threshold=args.apply_threshold,gate_suppress_threshold=args.suppress_threshold,
            render_suppressed=args.render_suppressed,max_context_chars=args.max_context_chars) for method in args.methods}
    eps=endpoints(Path(args.env))
    tpm_limits={}
    for setting in getattr(args,'model_tpm',[]):
        model,value=setting.rsplit('=',1)
        if not model or model in tpm_limits or int(value)<=0:raise ValueError('Expected unique MODEL=positive_TPM')
        tpm_limits[model]=int(value)
    qwen='Qwen/Qwen2.5-7B-Instruct'
    if any(e.model==qwen for e in eps.values()):
        tpm_limits.setdefault(qwen,80000)
        local_tokenizer=Path('datasets/tokenizers/qwen2.5-7b/tokenizer.json')
        if not getattr(args,'tpm_tokenizer',None) and local_tokenizer.exists():
            args.tpm_tokenizer=str(local_tokenizer)
    out=Path(args.output);out.mkdir(parents=True,exist_ok=True)
    scientific=dict(protocol_version='spm-api-v2',seed=args.seed,split=args.split,
        selection=dict(limit=args.limit,perma_limit=getattr(args,'perma_limit',None),
                       max_history_episodes=getattr(args,'max_history_episodes',None),
                       case_file=getattr(args,'case_file',None),
                       case_file_sha256=file_hash(args.case_file) if getattr(args,'case_file',None) else None),
        perma_mode=args.perma_mode,perma_types=list(args.perma_types),methods=args.methods,
        configs={m:c.dump() for m,c in configs.items()},models={r:dict(model=e.model,base_url=e.base_url) for r,e in eps.items()},
        cases=[dict(id=c.id,digest=c.digest()) for c in cases],
        sources={p:file_hash(p) for p in sorted({s for c in cases for s in c.sources})},
        environment=dict(python=sys.version,uv_lock=file_hash('uv.lock'),pydantic=pydantic.__version__,
                         numpy=numpy.__version__,httpx=httpx.__version__,pyarrow=pyarrow.__version__),
        code={p.name:file_hash(p) for p in sorted(Path(__file__).parent.glob('*.py'))})
    manifest_path=out/'manifest.json'
    if manifest_path.exists():
        if json.loads(manifest_path.read_text())['scientific']!=scientific:
            raise ValueError('Existing run has different data/models/config/code; choose a new output directory')
    else:atomic_json(manifest_path,dict(scientific=scientific,inventory=inventory))
    api=Gateway(eps,Path(args.cache),concurrency=args.concurrency,max_calls=args.max_calls,
                max_tokens=args.max_tokens,timeout=args.timeout,tpm_limits=tpm_limits,
                tpm_tokenizer=getattr(args,'tpm_tokenizer',None))
    invocations_path=out/'invocations.json'
    invocations=json.loads(invocations_path.read_text()) if invocations_path.exists() else []
    invocations.append(dict(run_id=api.run,max_calls=args.max_calls,max_tokens=args.max_tokens,
                            workers=args.workers,concurrency=args.concurrency,timeout=args.timeout,
                            tpm_limits=tpm_limits,tpm_tokenizer=getattr(args,'tpm_tokenizer',None)))
    atomic_json(invocations_path,invocations)
    for invocation in invocations[:-1]:
        old=out/'usage'/f"{invocation['run_id']}.json"
        if not old.exists():atomic_json(old,api.usage(invocation['run_id']))
    exp=Experiment(api,out,configs,args.workers)
    usage_path=out/'usage'/f'{api.run}.json'
    try:
        rows=await exp.run(cases,args.methods)
        metrics=aggregate(rows,args.seed)
        atomic_json(out/'metrics.json',metrics)
        atomic_json(out/'comparisons.json',paired_comparisons(rows,seed=args.seed))
        atomic_json(out/'frontier.json',frontier(rows))
        atomic_json(usage_path,api.usage())
        usage=[json.loads(p.read_text()) for p in sorted((out/'usage').glob('*.json'))]
        atomic_json(out/'usage.json',dict(attempts=usage,http_requests=sum(u['http_requests'] for u in usage),
                                        known_tokens=sum(u['known_tokens'] for u in usage)))
        atomic_json(out/'summary.json',dict(metrics=metrics,inventory=inventory,
            usage_file='usage.json',sample_files=len(rows),
            caution='Development pilot. PrefEval/CUPID use API surrogate judges; BenchPreS measures gate decisions, not generated-message adherence.'))
        print(json.dumps(dict(output=str(out),metrics=metrics,usage=api.usage()),ensure_ascii=False,indent=2),flush=True)
        return int(any(r['status']!='ok' for r in rows))
    finally:
        usage=await api.close()
        atomic_json(usage_path,usage)
