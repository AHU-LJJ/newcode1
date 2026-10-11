"""Local dataset readers: public memory inputs are separate from evaluation labels.

No benchmark annotations are available to the memory pipeline. Explicit PrefEval
statements and BenchPreS given preferences are intentionally public inputs.
"""
from collections import Counter, defaultdict
from dataclasses import dataclass, field
import json
from pathlib import Path
import random
import re
import pyarrow.parquet as pq
from .models import Episode, Message, fingerprint


@dataclass(frozen=True)
class MemoryInput:
    user_id: str
    episodes: tuple[Episode, ...]
    query: str
    given: bool = False
    options: str | None = None
    provenance: tuple[str, ...] = ()

    def dump(self):
        return dict(user_id=self.user_id, episodes=[e.model_dump(mode='json') for e in self.episodes],
                    query=self.query, given=self.given, options=self.options, provenance=list(self.provenance))


@dataclass(frozen=True)
class Case:
    id: str
    benchmark: str
    group: str
    stratum: str
    public: MemoryInput
    gold: dict = field(repr=False)
    sources: tuple[str, ...] = ()

    def digest(self):
        return fingerprint([self.id,self.public.dump(),self.gold])


def episode(messages, order=0, time=None):
    msgs=[Message.model_validate(m) for m in messages]
    ident=fingerprint([order,time,[m.model_dump() for m in msgs]])[:24]
    return Episode(id=ident,order=order,time=time,messages=msgs)


def prefeval(root):
    base=root/'PrefEval'
    paths=[('explicit',base/'explicit_preference'),('choice',base/'implicit_preference'/'choice-based'),
           ('persona',base/'implicit_preference'/'persona-driven')]
    cases=[]
    for kind,directory in paths:
        for path in sorted(directory.glob('*.json')):
            for i,r in enumerate(json.loads(path.read_text())):
                group=fingerprint(' '.join(r['question'].casefold().split()))[:24]
                if kind=='explicit':
                    messages=[dict(role='user',content=r['preference'])]
                    source=('preference (public historical explicit user statement)','question')
                elif kind=='choice':
                    conv=r['conversation']
                    messages=[dict(role=role,content=conv[key]) for role,key in
                              [('user','query'),('assistant','assistant_options'),('user','user_selection'),('assistant','assistant_acknowledgment')]]
                    source=('conversation.query','conversation.assistant_options','conversation.user_selection','conversation.assistant_acknowledgment','question')
                else:
                    messages=[dict(role=role,content=turn[role]) for _,turn in
                              sorted(r['conversation'].items(),key=lambda kv:int(kv[0])) for role in ('user','assistant') if role in turn]
                    source=('conversation.*.user','conversation.*.assistant','question')
                ident=f'prefeval/{kind}/{path.stem}/{i}'
                cases.append(Case(ident,'prefeval',group,kind,
                    MemoryInput(ident,(episode(messages),),r['question'],provenance=source),
                    dict(preference=r['preference'],explanation=r.get('explanation','')), (str(path),)))
    return cases


def cupid(root):
    path=root/'cupid'/'test.parquet'
    cases=[]
    for i,r in enumerate(pq.read_table(path).to_pylist()):
        history=tuple(episode(h['dialogue'],j) for j,h in enumerate(r['prior_interactions']))
        cases.append(Case(f'cupid/{i}','cupid',r['persona_id'],r['instance_type'],
            MemoryInput(r['persona_id'],history,r['current_request'],
                        provenance=('prior_interactions.*.dialogue.role/content','current_request')),
            dict(preference=r['current_contextual_preference'],checklist=r['current_checklist']), (str(path),)))
    return cases


def benchpres(root):
    path=root/'benchpres'/'data'/'test-00000-of-00001.parquet'
    cases=[]
    for i,r in enumerate(pq.read_table(path).to_pylist()):
        if len(r['preference_attribute'])!=len(r['preference_label']):
            raise ValueError('BenchPreS preferences/labels length mismatch')
        if '\nWrite a complete message' not in r['prompt']:
            raise ValueError('Unknown BenchPreS prompt boundary; refusing to silently discard context')
        query=r['prompt'].rsplit('\nWrite a complete message',1)[1]
        query='Write a complete message'+query+f"\nDomain: {r['domain']}."
        history=(episode([dict(role='user',content=p) for p in r['preference_attribute']]),)
        cases.append(Case(f'benchpres/{i}','benchpres',r['name'],r['domain'],
            MemoryInput(r['name'],history,query,given=True,
                        provenance=('preference_attribute (public given preferences)','prompt (public final instruction)','domain')),
            dict(labels=r['preference_label']), (str(path),)))
    return cases


def timestamp(value):
    day=str(value)[:10]
    if not re.fullmatch(r'\d{4}-\d{2}-\d{2}',day):
        raise ValueError('PERMA timestamp has an unknown format')
    suffix=str(value)[10:].strip().lower()
    rank={'morning':0,'afternoon':1,'evening':2,'night':3}.get(suffix,4)
    return day,rank,suffix


def perma(root,mode='sd_clean',types=(1,2,3)):
    files={'sd_clean':'input_data_c.json','sd_noise':'input_data_n.json',
           'md_clean':'input_data_multi_c.json','md_noise':'input_data_multi_n.json'}
    if mode not in files:raise ValueError('Unknown PERMA mode')
    cases=[]
    for path in sorted((root/'perma'/'tasks').glob('*/'+files[mode])):
        user=path.parent.name
        for r in json.loads(path.read_text())['overall']:
            if r['type'] not in types:continue
            meta=root/'perma'/'evaluation'/user/'meta'/'overall'/f"{r['task_id']}_{r['type']}.json"
            if not meta.exists():
                raise FileNotFoundError(f'Prepared PERMA question missing: {meta}')
            m=json.loads(meta.read_text())
            question_day=timestamp(m['question_date'])[0]
            history=[]
            if r['type']!=1:  # Official ZeroMemory condition, despite context in input_data.
                ordered=sorted(r['context'],key=lambda h:timestamp(h[1]))
                for messages,time in ordered:
                    if timestamp(time)[0]>question_day:
                        raise ValueError(f'Future conversation in PERMA: {user}/{r["task_id"]}')
                    history.append(episode(messages,len(history),str(time)))
            if not re.fullmatch('[A-H]',m['gold_label']):
                raise ValueError('Expected prepared single-label PERMA MCQ')
            ident=f'perma/{mode}/{user}/{r["task_id"]}/{r["type"]}'
            cases.append(Case(ident,'perma',user,f'type{r["type"]}',
                MemoryInput(user,tuple(history),m['question'],options=m['options'],
                            provenance=('context.*.dialogue.role/content','context.*.timestamp','meta.question','meta.options (answer stage only)')),
                dict(labels=[m['gold_label']],task_id=r['task_id'],type=r['type']), (str(path),str(meta))))
    return cases


def load_cases(root,benchmark,perma_mode='sd_clean',perma_types=(1,2,3)):
    root=Path(root)
    readers={'prefeval':prefeval,'cupid':cupid,'benchpres':benchpres,
             'perma':lambda p:perma(p,perma_mode,perma_types)}
    cases=readers[benchmark](root)
    if not cases:raise ValueError(f'No local {benchmark} cases found under {root}')
    return cases


def partition(cases,split='dev',seed=42):
    groups=sorted({c.group for c in cases})
    random.Random(seed).shuffle(groups)
    dev=set(groups[:max(1,len(groups)//5)])
    if split=='all':return cases
    if split not in ('dev','test'):raise ValueError('Unknown split')
    return [c for c in cases if (c.group in dev)==(split=='dev')]


def select(cases,limit=None,seed=42):
    """Round-robin strata for a useful pilot, random ordering within each stratum."""
    strata=defaultdict(list)
    for c in cases:strata[c.stratum].append(c)
    rng=random.Random(seed)
    for group in strata.values():rng.shuffle(group)
    output=[]
    while any(strata.values()) and (limit is None or len(output)<limit):
        for label in sorted(strata):
            if strata[label] and (limit is None or len(output)<limit):output.append(strata[label].pop())
    return output


def describe(cases):
    return dict(cases=len(cases),groups=len({c.group for c in cases}),
                strata=dict(Counter(c.stratum for c in cases)),
                episodes=sum(len(c.public.episodes) for c in cases),
                public_fields=sorted({p for c in cases for p in c.public.provenance}))
