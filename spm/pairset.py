"""P1 self-built paired-situation set: one preference, one request, two situations.

Construction follows the public PairPref design (arXiv 2609.34526) with self-written
content: rule checks, semantic verification by a different model, and context-free
screening. Situation labels and reply roles belong to evaluation only.
"""
from dataclasses import replace
import random
import re
from pathlib import Path
from typing import Literal

import numpy as np
from pydantic import Field

from .config import Endpoint, endpoints
from .models import Model

BUILDER_MODEL = 'zai-org/GLM-5.3-Flash'

CATEGORIES = {
    'companions': 'who the user is with or sharing the outcome with',
    'occasion': 'the kind of event or setting',
    'time_constraint': 'how much time the user has',
    'beneficiary_shift': 'who the result is for, when it is someone other than the user',
    'role_shift': 'the role the user is acting in, such as teacher, manager or host',
    'resource_constraint': 'budget, equipment, ingredients or other available resources',
    'physical_state': 'a temporary bodily state such as an injury, fatigue or a sore throat',
    'emotional_state': 'a temporary emotional state such as grief, stress or nervousness',
}

DOMAINS = {
    'food': ['ordering at restaurants', 'cooking at home', 'snacks and drinks', 'meal planning'],
    'travel': ['choosing destinations', 'accommodation', 'getting around', 'trip itineraries'],
    'writing': ['personal messages', 'work emails', 'social media posts', 'creative writing'],
    'fitness': ['workouts', 'outdoor activities', 'sports', 'daily movement'],
    'entertainment': ['movies and shows', 'music', 'games', 'books'],
    'shopping': ['clothing', 'gadgets', 'gifts', 'home goods'],
    'social': ['gatherings and parties', 'conversation and tone', 'celebrations', 'meeting new people'],
    'work': ['scheduling and planning', 'presentations', 'learning new skills', 'tools and workflow'],
    'home': ['decor', 'cleaning and organizing', 'gardening', 'home routines'],
    'leisure': ['hobbies and crafts', 'weekend plans', 'relaxation', 'photography'],
}

REPLY_ROLES = ('follow', 'set_aside_ack', 'set_aside_silent', 'off_task')


def slots():
    """Deterministic construction grid: domain x subtopic x situation category."""
    return [dict(id=f'{domain}-{i}-{category}', domain=domain, subtopic=subtopic, category=category)
            for domain, subtopics in DOMAINS.items() for i, subtopic in enumerate(subtopics)
            for category in CATEGORIES]


def pilot_slots(limit, seed):
    """Category-balanced deterministic subset for pilots."""
    rng = random.Random(f'{seed}:pilot')
    pools = {c: [s for s in slots() if s['category'] == c] for c in CATEGORIES}
    for pool in pools.values():
        rng.shuffle(pool)
    chosen = []
    while len(chosen) < limit and any(pools.values()):
        for pool in pools.values():
            if pool and len(chosen) < limit:
                chosen.append(pool.pop())
    return chosen


def build_endpoints(path=Path('.env'), builder_env=None):
    """SiliconFlow roles from .env; the builder optionally from a separate provider file.

    The builder file is read without touching os.environ, so its OPENAI_* names
    never leak into the other roles.
    """
    eps = endpoints(path)
    builder = replace(eps['judge'], model=BUILDER_MODEL)
    if builder_env:
        from dotenv import dotenv_values
        values = {k: (v or '').strip() for k, v in dotenv_values(builder_env).items()}
        url = re.sub(r'/chat/completions/?$', '', values.get('OPENAI_BASE_URL', '')).rstrip('/')
        builder = Endpoint(values.get('OPENAI_MODEL') or BUILDER_MODEL, values.get('OPENAI_API_KEY', ''), url)
    eps['builder'] = builder
    eps['verifier'] = eps['judge']
    return eps


class PairDraft(Model):
    preference: str = Field(min_length=5, max_length=300)
    request: str = Field(min_length=5, max_length=400)
    positive_situation: str = Field(min_length=10, max_length=400)
    negative_situation: str = Field(min_length=10, max_length=400)
    reply_follow: str = Field(min_length=20, max_length=2000)
    reply_set_aside_ack: str = Field(min_length=20, max_length=2000)
    reply_set_aside_silent: str = Field(min_length=20, max_length=2000)
    reply_off_task: str = Field(min_length=20, max_length=2000)
    companions_near: list[str] = Field(min_length=4, max_length=4)
    companions_middle: list[str] = Field(min_length=5, max_length=5)
    companions_far: list[str] = Field(min_length=6, max_length=6)
    rationale: str = Field(min_length=5, max_length=500)

    def replies(self):
        return dict(follow=self.reply_follow, set_aside_ack=self.reply_set_aside_ack,
                    set_aside_silent=self.reply_set_aside_silent, off_task=self.reply_off_task)

    def companions(self):
        return self.companions_near + self.companions_middle + self.companions_far


class PairVerdict(Model):
    positive_applies: bool = Field(strict=True)
    negative_should_set_aside: bool = Field(strict=True)
    request_natural_in_both: bool = Field(strict=True)
    follow_enacts_preference: bool = Field(strict=True)
    ack_departs_and_acknowledges: bool = Field(strict=True)
    silent_departs_without_acknowledging: bool = Field(strict=True)
    off_task_answers_other_request: bool = Field(strict=True)
    replies_fit_both_situations: bool = Field(strict=True)
    companions_do_not_interfere: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=2000)

    def passed(self):
        return all(v for k, v in self.model_dump().items() if k != 'reason')


class Choice(Model):
    label: Literal['A', 'B', 'C', 'D']


BUILD = """You construct one test item for a benchmark about WHEN an assistant should use a
remembered user preference. Write in English and return only the requested JSON.

Inputs: a domain, a subtopic and a situation category with its meaning. Write:
- preference: one durable personal preference the user stated earlier, first person
  (for example "I love really spicy food."). It must be a taste or style preference within
  the domain and subtopic, NOT an allergy, medical condition, safety need or moral rule,
  so that setting it aside can be the right call.
- request: one natural request to an assistant, identical in both situations, where the
  preference would normally shape a good answer. Do not mention the preference or either
  situation in the request.
- positive_situation: 1-2 first-person sentences the user says right before the request.
  It concerns the given category, and following the preference is clearly right in it.
- negative_situation: 1-2 first-person sentences of the SAME category and similar length,
  in which a thoughtful assistant should set the preference aside for this request. The
  reason must follow from the situation itself. Do not mention the preference, do not say
  "instead of my usual" or "for once", and do not tell the assistant what to do.
- reply_follow: a good reply (60-140 words) to the request that clearly follows the preference.
- reply_set_aside_ack: a good reply (60-140 words) that does not follow the preference and
  briefly acknowledges that it is setting the preference aside.
- reply_set_aside_silent: a good reply (60-140 words) that does not follow the preference
  and never mentions the preference.
- reply_off_task: a fluent reply (60-140 words) that answers a different, unrelated request.
- companions_near: 4 other first-person preferences in the same domain.
- companions_middle: 5 first-person preferences in neighbouring domains.
- companions_far: 6 first-person preferences in unrelated domains.
- rationale: one sentence on why the negative situation calls for setting the preference aside.

The four replies are shown unchanged in BOTH situations, so they must not refer to
specific details of either situation. The three on-task replies must be similar in
length, quality and specificity; only their use of the preference differs. No reply may
mention memory or what the user said before. Companion preferences must not conflict with
the target preference and must not change the best answer to the request. Use "I" for the
user and never invent the user's name.
"""

VERIFY = """You audit one item of a benchmark about when an assistant should use a remembered
user preference. The same request and the same four replies appear in two situations.
Judge strictly; answer each boolean independently and give a short reason.
- positive_applies: in the positive situation, following the preference is clearly appropriate.
- negative_should_set_aside: in the negative situation, a thoughtful assistant should set
  the preference aside, for a reason that follows from the situation itself.
- request_natural_in_both: the request reads naturally after each situation.
- follow_enacts_preference: the follow reply clearly uses the preference and is a good answer.
- ack_departs_and_acknowledges: the ack reply does not use the preference, briefly says so,
  and is otherwise a good answer.
- silent_departs_without_acknowledging: the silent reply does not use the preference, never
  mentions it, and is otherwise a good answer.
- off_task_answers_other_request: the off-task reply answers a different request.
- replies_fit_both_situations: no reply refers to details that exist in only one situation.
- companions_do_not_interfere: no companion preference conflicts with the target preference
  or changes the best answer to the request.
Keep reason under 80 words.
"""

SCREEN = """Choose the reply that best answers the request. Return only the label."""

ACK_SILENT = re.compile(r"\b(your (usual|preference|taste|love|favou?rite)|you (usually|normally|typically) (like|love|prefer|enjoy|go for)|(i know|since|because|even though) you (like|love|prefer|enjoy)|set(ting)? (that|this|your|the) [\w\s-]{0,30}aside)\b", re.I)
MEMORY = re.compile(r"\b(your memory|from memory|i remember|as you (said|mentioned)|you (mentioned|told me|said)( before| earlier)?|(earlier|previous) (chat|conversation))\b", re.I)
CUE = re.compile(r"\b(prefer\w*|preference|my usual|instead of|for once|skip my|don't (use|give|suggest))\b", re.I)
STOP = set('a an and are as at be but by for from had has have i i\'m in is it its me my of on or our so that the this to was we were with you your'.split())


def words(text):
    return re.findall(r"[a-z']+", text.lower())


def content(text):
    return {w for w in words(text) if w not in STOP and len(w) > 2}


def rule_issues(draft):
    """Programmatic stage-1 screen; returns issue codes (empty means pass)."""
    issues = []
    replies = draft.replies()
    lengths = {k: len(words(v)) for k, v in replies.items()}
    issues += [f'{k}_length' for k, n in lengths.items() if not 40 <= n <= 200]
    on_task = [lengths[k] for k in REPLY_ROLES[:3]]
    if max(on_task) > 2 * min(on_task):
        issues.append('length_imbalance')
    issues += [f'{k}_mentions_memory' for k, v in replies.items() if MEMORY.search(v)]
    if ACK_SILENT.search(draft.reply_set_aside_silent):
        issues.append('silent_acknowledges')
    situations = dict(positive=draft.positive_situation, negative=draft.negative_situation)
    issues += [f'{k}_explicit_cue' for k, v in situations.items() if CUE.search(v)]
    if draft.positive_situation.strip().casefold() == draft.negative_situation.strip().casefold():
        issues.append('identical_situations')
    a, b = len(words(draft.positive_situation)), len(words(draft.negative_situation))
    if max(a, b) > 2.5 * max(1, min(a, b)):
        issues.append('situation_length_imbalance')
    pref = content(draft.preference)
    for k, v in situations.items():
        overlap = len(pref & content(v)) / max(1, len(pref))
        if overlap > .5:
            issues.append(f'{k}_repeats_preference')
    companions = [c.strip().casefold() for c in draft.companions()]
    if len(set(companions)) != len(companions) or draft.preference.strip().casefold() in companions:
        issues.append('duplicate_companions')
    return issues


def dedup(ids, vectors, threshold=.92):
    """Greedy in slot order: keep an item unless it is too similar to a kept one."""
    kept, kept_vectors, dropped = [], [], {}
    for identity, vector in zip(ids, vectors, strict=True):
        if kept_vectors:
            sims = np.stack(kept_vectors) @ vector
            best = int(np.argmax(sims))
            if sims[best] > threshold:
                dropped[identity] = dict(duplicate_of=kept[best], cosine=float(sims[best]))
                continue
        kept.append(identity)
        kept_vectors.append(vector)
    return kept, dropped


def screening_orders(identity, seed, n=3):
    """n reply orders with the follow reply at n distinct positions."""
    rng = random.Random(f'{seed}:screen:{identity}')
    orders = []
    for position in rng.sample(range(4), n):
        others = list(REPLY_ROLES[1:])
        rng.shuffle(others)
        others.insert(position, 'follow')
        orders.append(others)
    return orders


def labelled(order, replies):
    return {label: replies[role] for label, role in zip('ABCD', order, strict=True)}


async def build(api, slot):
    value = dict(domain=slot['domain'], subtopic=slot['subtopic'], category=slot['category'],
                 category_meaning=CATEGORIES[slot['category']])
    # Zhipu GLM-5.3-Flash always reasons; its reasoning tokens count against max_tokens.
    return await api.ask('builder', BUILD, value, PairDraft, 'pair_build', max_tokens=12288)


def audit_view(draft):
    replies = draft.replies()
    return dict(preference=draft.preference, request=draft.request,
                positive_situation=draft.positive_situation, negative_situation=draft.negative_situation,
                replies={k: replies[k] for k in REPLY_ROLES}, companions=draft.companions())


async def verify(api, draft):
    return await api.ask('verifier', VERIFY, audit_view(draft), PairVerdict, 'pair_verify', max_tokens=1024)


async def screen(api, draft, orders):
    """Context-free stage: the screener sees only the request and the replies."""
    import asyncio
    replies = draft.replies()
    async def pick(order):
        out = await api.ask('cheap', SCREEN, dict(request=draft.request, replies=labelled(order, replies)),
                            Choice, 'pair_screen', max_tokens=256)
        return order['ABCD'.index(out.label)]
    return list(await asyncio.gather(*(pick(order) for order in orders)))


def screen_passed(picks, max_follow=1):
    return sum(p == 'follow' for p in picks) <= max_follow
