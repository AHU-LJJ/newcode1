"""P1 evaluation on the self-built paired set: selection track and free generation.

Definitions follow PairPref (arXiv 2609.34526): Use+/Use-, selectivity, Use both and
Pair success. The memory pool (target + 15 companions) is identical on both sides;
only the situation text before the request differs. Reply roles, the target index
and situation polarity are evaluation-only and never reach the system under test.
"""
import asyncio
import random
from typing import Literal

import numpy as np
from pydantic import Field

from .models import Model
from . import pairset

SIDES = ('positive', 'negative')

# Prompt-only baseline P: every memory, plus an explicit applicability instruction.
P_INSTRUCTION = """The notes below are things the user told you in earlier conversations.
Use a remembered preference only when it fits the user's current situation and request.
If the current situation makes a preference inappropriate, set it aside."""

SELECT = """You are a helpful assistant. Read the user's message and the candidate replies,
then choose the single best reply to send. Return only the label."""

ANSWER = """You are a helpful assistant. Answer the user's message directly and helpfully."""

JUDGE = """You grade one assistant reply in a benchmark about when an assistant should use
a remembered user preference. You see the user's preference, the situation the user
described, the request and the reply. Grade two things independently.
uptake: how much the reply relies on the preference.
  0 = the reply does not use the preference;
  0.5 = the reply uses it implicitly or partially, or offers it as one option among others;
  1 = the preference is explicit or central to the reply.
  Uptake measures whether the preference shapes the content of the answer. A reply that
  mentions the preference only to say it is being set aside, and whose content does not
  follow it, has uptake 0.
situation_fit: 1 if the reply is a reasonable, helpful answer to the request given the
situation, else 0. Judge fit by the situation, not by whether the preference is used.
hedged: true when the reply offers both a preference-based option and a generic
alternative without committing to either.
Keep reason under 60 words."""


class Choice(Model):
    label: Literal['A', 'B', 'C', 'D']


class Grade(Model):
    uptake: Literal[0, 0.5, 1]
    situation_fit: Literal[0, 1]
    hedged: bool = Field(strict=True)
    reason: str = Field(min_length=1, max_length=1000)


def memory_pool(item, seed):
    """Target plus companions in a fixed per-item shuffle, same on both sides."""
    draft = item['draft']
    pool = [draft['preference']] + draft['companions_near'] + draft['companions_middle'] + draft['companions_far']
    rng = random.Random(f'{seed}:pool:{item["id"]}')
    order = list(range(len(pool)))
    rng.shuffle(order)
    return [pool[i] for i in order], order.index(0)


def user_message(item, side):
    draft = item['draft']
    return draft[f'{side}_situation'].strip() + ' ' + draft['request'].strip()


def render_notes(notes):
    return '\n'.join('- ' + n for n in notes)


def context_for(method, notes, spm_context=None):
    """Memory text shown to the system under test. L0 sees nothing."""
    if method == 'L0':
        return ''
    if method == 'L1':
        return 'Notes about the user from earlier conversations:\n' + render_notes(notes)
    if method == 'P':
        return P_INSTRUCTION + '\n\n' + render_notes(notes)
    if method == 'L4':
        return spm_context or ''
    raise ValueError(f'Unknown method {method}')


async def spm_contexts(api, item, notes, cfg, directory):
    """L4: install the shared pool as given preferences once, then query each side.

    Both sides read the same store; query() is read-only, so side order cannot leak.
    """
    from .adapters import episode
    from .pipeline import SPM
    from .store import Store
    store = Store(directory/f'{item["id"]}.sqlite', 'the user')
    try:
        memory = SPM('the user', store, api, cfg)
        await memory.install_given(episode([dict(role='user', content=n) for n in notes]))
        out = {}
        for side in SIDES:
            context, trace = await memory.query(user_message(item, side))
            out[side] = dict(context=context, decisions=trace['decisions'],
                             candidates=[c['unit']['evidence'][0]['turn_idx'] for c in trace['candidates']])
        return out
    finally:
        store.close()


ORACLES = ('O1', 'O2')


def oracle_decisions(decisions, candidates, target, side, mode):
    """P3 gate oracle on recorded L4 decisions; labels exist only for the target.

    O1: target decision replaced by its label (apply on positive, suppress on
    negative), companion decisions kept as gated. O2: as O1, and every companion
    suppressed, since by construction companions do not change the best answer.
    A target the retriever missed stays missing: the oracle fixes the gate only.
    """
    from .models import Decision
    out = []
    for decision, index in zip(decisions, candidates, strict=True):
        d = Decision(**decision)
        if index == target:
            label = 'apply' if side == 'positive' else 'suppress'
            d = Decision(unit_id=d.unit_id, decision=label, p_apply=float(label == 'apply'), reason='gate oracle')
        elif mode == 'O2':
            d = Decision(unit_id=d.unit_id, decision='suppress', p_apply=0., reason='gate oracle: companion')
        out.append(d)
    return out


def read_units(path):
    """Units of a frozen P1 memory store, read-only."""
    import sqlite3
    from .store import stored_unit
    db = sqlite3.connect(f'file:{path}?mode=ro', uri=True)
    try:
        return [stored_unit(r[0]) for r in db.execute('SELECT json FROM units ORDER BY id')]
    finally:
        db.close()


def render_oracle(units, decisions, cfg):
    """Render exactly as L4 does; raw evidence is absent for given-preference stores."""
    from .pipeline import SPM, Candidate
    by_id = {u.id: u for u in units}
    candidates = [Candidate(by_id[d.unit_id], 0.) for d in decisions]
    return SPM('the user', None, None, cfg).render(candidates, decisions)


def gate_correct(record, side):
    """Correct by rendering effect: only 'apply' reaches the generator.

    'uncertain', 'suppress' and an unretrieved target are all omitted from the
    context, so on the negative side they are correct and on the positive side wrong.
    """
    applied = (record['spm'][side]['target_decision'] or {}).get('decision') == 'apply'
    return applied if side == 'positive' else not applied


def selection_order(item, side, seed):
    rng = random.Random(f'{seed}:select:{item["id"]}:{side}')
    order = list(pairset.REPLY_ROLES)
    rng.shuffle(order)
    return order


async def select(api, role, item, side, memory, seed):
    order = selection_order(item, side, seed)
    replies = pairset.labelled(order, pairset.PairDraft(**item['draft']).replies())
    value = dict(user_message=user_message(item, side), candidate_replies=replies)
    if memory:
        value = dict(memory=memory, **value)
    # GLM-5.3-Flash always reasons; reasoning tokens count against max_tokens.
    out = await api.ask(role, SELECT, value, Choice, 'p1_select', max_tokens=4096)
    return order['ABCD'.index(out.label)]


async def generate(api, role, item, side, memory):
    system = ANSWER + ('\n\n' + memory if memory else '')
    return await api.ask(role, system, user_message(item, side), tag='p1_generate', max_tokens=6144)


async def grade(api, item, side, answer):
    draft = item['draft']
    value = dict(preference=draft['preference'], situation=draft[f'{side}_situation'],
                 request=draft['request'], reply=answer)
    return await api.ask('judge', JUDGE, value, Grade, 'p1_judge', max_tokens=1024)


def selection_metrics(rows):
    """rows: [{positive: role, negative: role}] for one method/model."""
    n = len(rows)
    if not n:
        return None
    use_pos = sum(r['positive'] == 'follow' for r in rows)
    use_neg = sum(r['negative'] == 'follow' for r in rows)
    success = sum(r['positive'] == 'follow' and r['negative'] in ('set_aside_ack', 'set_aside_silent') for r in rows)
    off = sum(r[s] == 'off_task' for r in rows for s in SIDES)
    return dict(pairs=n, use_pos=use_pos/n, use_neg=use_neg/n, selectivity=(use_pos-use_neg)/n,
                pair_success=success/n, off_task_rate=off/(2*n))


def generation_metrics(rows):
    """rows: [{positive: grade, negative: grade}]; hedged pairs are excluded."""
    kept = [r for r in rows if not (r['positive']['hedged'] or r['negative']['hedged'])]
    n = len(kept)
    if not n:
        return dict(pairs=0, hedged_excluded=len(rows))
    use_pos = sum(r['positive']['uptake'] >= .5 for r in kept)
    use_neg = sum(r['negative']['uptake'] >= .5 for r in kept)
    both = sum(r['positive']['uptake'] >= .5 and r['negative']['uptake'] >= .5 for r in kept)
    success = sum(r['positive']['uptake'] >= .5 and r['negative']['uptake'] == 0 and
                  r['positive']['situation_fit'] == 1 and r['negative']['situation_fit'] == 1 for r in kept)
    fit = sum(r[s]['situation_fit'] for r in kept for s in SIDES)
    return dict(pairs=n, hedged_excluded=len(rows)-n, use_pos=use_pos/n, use_neg=use_neg/n,
                selectivity=(use_pos-use_neg)/n, use_both=both/n, pair_success=success/n,
                situation_fit=fit/(2*n))


def calibrated_generation(rows, p_zero):
    """Expected generation metrics with negative uptake mapped through human labels.

    p_zero[b] is P(human uptake = 0 | judge uptake b) on negative sides; positive
    sides and situation fit are taken as judged. Hedged pairs are excluded as usual.
    """
    kept = [r for r in rows if not (r['positive']['hedged'] or r['negative']['hedged'])]
    if not kept:
        return None
    success = both = neg = 0.
    for r in kept:
        z = p_zero[float(r['negative']['uptake'])]
        neg += 1 - z
        if r['positive']['uptake'] >= .5:
            both += 1 - z
            if r['positive']['situation_fit'] == 1 and r['negative']['situation_fit'] == 1:
                success += z
    return dict(pair_success=success/len(kept), use_both=both/len(kept), use_neg=neg/len(kept))


def bootstrap_pair(values_a, values_b, groups, fn_a, fn_b, seed=42, resamples=2000):
    """Paired cluster bootstrap of fn_a(a)-fn_b(b); a and b are aligned per item."""
    keys = sorted(set(groups))
    clusters = {k: [i for i, g in enumerate(groups) if g == k] for k in keys}
    rng = np.random.default_rng(seed)
    draws = []
    for _ in range(resamples):
        index = [i for k in rng.choice(keys, len(keys)) for i in clusters[k]]
        a, b = fn_a([values_a[i] for i in index]), fn_b([values_b[i] for i in index])
        if a is not None and b is not None:
            draws.append(a-b)
    return np.quantile(draws, [.025, .975]).tolist() if draws else None
