"""Four inexpensive API baselines with the same generator/evaluation protocol."""
import json
import numpy as np
from .config import SPMConfig
from .models import Decision, fingerprint
from .pipeline import SPM, Candidate, windows, render_raw_evidence


class FlatMemory(SPM):
    """Append extracted observations, retaining the graph reader and applicability gate."""
    async def _batch_relations(self,units,vectors):return {}

    async def _upsert(self,new,vec,episode,known_relations=None):
        self.store.put(new,vec)
        self.store.event(action='flat_append',unit=new.id,episode=episode.id)


class EntityMemory(SPM):
    async def _batch_relations(self,units,vectors):return {}

    async def _upsert(self,new,vec,episode,known_relations=None):
        # Extraction only. No contradiction detection or learned applicability gate.
        self.store.put(new,vec)
        self.store.event(action='baseline_append',unit=new.id,episode=episode.id)

    async def _query(self,text,force_all=False):
        query=await self.analyze(text)
        candidates,raw=await self.retrieve(query,force_all)
        if not force_all:
            newest={}
            for c in candidates:
                entity=c.unit.condition.entity
                if entity:newest[entity]=max(newest.get(entity,-1),c.unit.last_order)
            candidates=[c for c in candidates if not c.unit.condition.entity or
                        (c.unit.condition.entity in query.entities and c.unit.last_order==newest[c.unit.condition.entity])]
            candidates.sort(key=lambda c:(not c.entity_match,-c.unit.last_order,-c.score))
            candidates=candidates[:self.cfg.max_apply]
        decisions=[Decision(unit_id=c.unit.id,decision='apply',p_apply=1,reason='entity/recent baseline') for c in candidates]
        context=self.render(candidates,decisions)
        return context,dict(parsed=query.model_dump(),candidates=[dict(unit=c.unit.model_dump(),score=c.score) for c in candidates],
                            decisions=[d.model_dump() for d in decisions],context=context)


async def raw_context(method,public,api,cfg):
    if method=='b0':return '',dict(context='',decisions=[])
    if public.given:
        items=[m.content for m in public.episodes[0].messages]
        chosen=list(range(len(items)))
        if method=='b2':
            q=await api.embed([public.query]);v=await api.embed(items)
            chosen=np.argsort(-(q@v.T)[0],kind='stable')[:cfg.top_k_per_probe].tolist()
        return '\n'.join('- '+items[i] for i in chosen),dict(given_indices=chosen,context_kind='given preferences')
    if method=='b1':
        text=json.dumps([e.model_dump(mode='json') for e in public.episodes],ensure_ascii=False)
        if len(text)>cfg.max_context_chars:
            raise ValueError('Full-history baseline exceeds configured context budget; no silent truncation')
        return text,dict(context_kind='complete history',characters=len(text))
    if method=='b2_matched':
        # Exact same raw archive, probe expansion, ranking/tie order and rendering as v6.
        query=await SPM(public.user_id,None,api,cfg).analyze(public.query)
        archive={}
        for ep in public.episodes:
            chunks,_=windows(ep,cfg)
            if cfg.embed_raw=='all':
                chunks,_=windows(ep,SPMConfig(window_messages=max(1,len(ep.messages)),window_chars=cfg.window_chars))
            for i,chunk in enumerate(chunks):
                text=json.dumps(chunk,ensure_ascii=False)
                archive[fingerprint([ep.id,i,text])[:24]]=dict(episode_id=ep.id,text=text)
        items=[archive[key] for key in sorted(archive)]
        raw=[]
        if items:
            vectors=await api.embed([item['text'] for item in items]);probes=await api.embed(query.probes)
            scores=(probes@vectors.T).max(axis=0)
            raw=[dict(**items[i],score=float(scores[i])) for i in np.argsort(-scores,kind='stable')[:cfg.raw_fallback_k]]
        context,evidence=render_raw_evidence(raw,public.episodes,cfg)
        return context,dict(context_kind='matched raw RAG',parsed=query.model_dump(),raw_evidence=evidence,
            raw_ids=[r['episode_id'] for r in raw],decisions=[],context=context)
    chunks=[]
    for ep in public.episodes:
        all_cfg=SPMConfig(window_messages=max(1,len(ep.messages)),window_chars=cfg.window_chars)
        parts,_=windows(ep,all_cfg)
        chunks.extend(json.dumps(dict(time=ep.time,messages=c),ensure_ascii=False) for c in parts)
    if not chunks:return '',dict(context_kind='raw RAG',selected=[])
    v=await api.embed(chunks);q=await api.embed([public.query])
    indices=np.argsort(-(q@v.T)[0],kind='stable')[:cfg.top_k_per_probe].tolist()
    return '\n\n'.join(chunks[i] for i in indices),dict(context_kind='raw RAG',selected=indices)
