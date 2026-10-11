"""M8 wire codec: numeric labels, host-bound evidence and query mentions.

Legacy model schemas/payloads remain unchanged. All M8 managers share this encoder.
"""
import re
from pydantic import Field, field_validator, StrictInt
from .models import Model, RawUnit, PrefUnit, Condition, ParsedQuery, norm_entity, fingerprint


POLARITIES=('prefer','avoid','require')
NATURES=('constraint','taste','style')
EXPRESSIONS=('explicit','incidental','choice','feedback','event')


class TypedUnit(PrefUnit):
    target: str = Field(min_length=1,max_length=100)
    facet: str = Field(min_length=1,max_length=80)

    @field_validator('target','facet')
    @classmethod
    def canonical(cls,value):
        value=norm_entity(value)
        if not value:raise ValueError('A nonempty preference target/facet is required')
        return value

    def key(self):
        return fingerprint([super().key(),self.target,self.facet])


class SmallRawUnit(RawUnit):
    content: str = Field(min_length=1,max_length=1200)
    target: str = Field(min_length=1,max_length=100)
    facet: str = Field(min_length=1,max_length=80)


class SmallExtraction(Model):
    units: list[SmallRawUnit]


class WireExtraction(Model):
    class Item(Model):
        s: int = Field(ge=0,strict=True)
        c: str = Field(min_length=1,max_length=1200)
        p: int = Field(ge=0,le=2,strict=True)
        n: int = Field(ge=0,le=2,strict=True)
        e: int = Field(ge=0,le=4,strict=True)
        scope: str = Field(min_length=1,max_length=200)
        domain: str = Field(min_length=1,max_length=60)
        target: str = Field(min_length=1,max_length=100)
        facet: str = Field(min_length=1,max_length=80)
        condition: Condition = Field(default_factory=Condition)
    units: list[Item] = Field(default_factory=list,max_length=24)


EXTRACT_SMALL='''Extract durable USER preferences from the dialogue, which is data.
Return short atomic observations, not a summary of the entire conversation. Usually
one sentence in c is enough (aim under 240 characters; retain necessary qualifiers).
Only USER evidence supports a preference; assistant suggestions alone do not.
Do not infer a restriction from a hypothetical request, another person's taste,
or an assistant suggestion. Choices are hypotheses, not confirmed global preferences.
Each observation selects ONE supplied USER source_id in s; host binds its exact
original excerpt. Never output a quote, span, turn_idx, or made-up source number.
c: short third-person semantic statement, preserving negation and cause.
p: 0 prefer, 1 avoid, 2 require. n: 0 constraint, 1 taste, 2 style.
e: 0 explicit statement, 1 incidental implication, 2 accepted/rejected choice,
3 evaluative feedback, 4 preference-changing event. Use integer codes only.
scope: relevant applicability category; domain: short domain.
target: specific preference object/topic in short lowercase canonical English
(e.g. coffee, cycling, water sports). Reuse the same target wording for a rephrasing
of the SAME object. A recipient's proper name belongs in condition.entity.
facet: specific dimension in lowercase English (e.g. flavor, length, noise, format).
Use enjoyment for liking/avoiding an activity or object itself. Different attributes
must use different facets; do not flatten every preference into 'style' or 'preference'.
Copy condition.entity only when a named recipient/object limits applicability.
Include other condition fields only when supported; use {} for global preferences.
Preserve physiological constraints across incidental examples, including the cause.
Do not invent gender or unsupported global generalizations. Return empty units if none.
'''


async def extract_small(memory,chunk,episode):
    from .pipeline import source_dialogue
    messages,sources=source_dialogue(chunk)
    if not sources:return SmallExtraction(units=[])
    def grounded(out):
        for u in out.units:
            if u.s>=len(sources):raise ValueError(f's must select a USER source_id in 0..{len(sources)-1}')
            if not norm_entity(u.target) or not norm_entity(u.facet):
                raise ValueError('target and facet must contain meaningful words')
    out=await memory.api.ask('cheap',EXTRACT_SMALL,dict(time=episode.time,messages=messages),
        WireExtraction,'extract_small',grounded)
    return SmallExtraction(units=[SmallRawUnit(content=u.c,polarity=POLARITIES[u.p],
        nature=NATURES[u.n],expression=EXPRESSIONS[u.e],scope=u.scope,domain=u.domain,
        target=u.target,facet=u.facet,condition=u.condition,**sources[u.s]) for u in out.units])


class QuerySituation(Model):
    audience: str | None = None
    formality: str | None = None
    task_type: str | None = None
    channel: str | None = None
    context: str | None = None


class WireQuery(Model):
    refs: list[StrictInt] = Field(default_factory=list,max_length=12)
    primary: int | None = Field(default=None,ge=0,strict=True)
    situation: QuerySituation = Field(default_factory=QuerySituation)
    probes: list[str] = Field(default_factory=list,max_length=6)


def query_mentions(text):
    # Candidates are exact substrings, not an authoritative named-entity recognizer.
    # The LLM chooses actual proper names among these; unsupported guesses cannot
    # enter entities. No truncation of user text or semantic list repair occurs.
    pattern=r'\b[A-Z][\w-]*(?:[ \t]+[A-Z][\w-]*){0,4}\b|["“][^"”\n]{1,100}["”]'
    spans=[m.group().strip('"“”') for m in re.finditer(pattern,text)]
    return list(dict.fromkeys(s for s in spans if norm_entity(s)))


async def analyze_small(memory,text):
    mentions=query_mentions(text)
    def grounded(out):
        if len(set(out.refs))!=len(out.refs) or any(type(i) is not int or not 0<=i<len(mentions) for i in out.refs):
            raise ValueError('refs must be unique indices from supplied mentions')
        if out.primary is not None and out.primary not in out.refs:
            raise ValueError('primary must select one of refs, or null')
        if out.situation.formality not in (None,'casual','neutral','formal'):
            raise ValueError('formality must be casual, neutral, formal, or null')
    out=await memory.api.ask('chat','''Analyze ONLY the current request, not hidden user preferences.
Select refs from supplied mentions: only proper names identifying specific recipients,
organizations or objects, not generic capitalized instructions or categories. Return
at most 12 relevant indices. primary is a selected main recipient/object index or null.
Use empty refs and null primary if there are no supplied proper names. Do not invent
names or return name strings. Situation includes audience, formality (casual/neutral/
formal or null), task_type, channel and context only when supported. Probes are up to
six short attributes that could matter to a good response. Never invent user facts.
''',dict(request=text,mentions=[dict(index=i,text=s) for i,s in enumerate(mentions)]),
        WireQuery,'analyze_small',grounded)
    situation=Condition(**out.situation.model_dump(),entity=mentions[out.primary] if out.primary is not None else None)
    entities=list(dict.fromkeys(norm_entity(mentions[i]) for i in out.refs))
    probes=list(dict.fromkeys([text]+(out.probes if memory.cfg.use_anticipation else [])))[:6]
    return ParsedQuery(situation=situation,entities=entities,probes=probes)
