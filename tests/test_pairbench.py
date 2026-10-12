import asyncio

import pytest

from spm import pairbench, pairset


def item(identity='x'):
    reply = ' '.join(['word'] * 60)
    draft = dict(preference='I love really spicy food.', request='What should I order tonight?',
                 positive_situation='I am eating alone tonight.', negative_situation='My toddler shares every plate.',
                 reply_follow='FOLLOW ' + reply, reply_set_aside_ack='ACK ' + reply,
                 reply_set_aside_silent='SILENT ' + reply, reply_off_task='OFF ' + reply,
                 companions_near=[f'near {i}' for i in range(4)], companions_middle=[f'middle {i}' for i in range(5)],
                 companions_far=[f'far {i}' for i in range(6)], rationale='toddler')
    return dict(id=identity, slot=dict(category='companions', domain='food'), draft=draft)


def test_memory_pool_is_shared_and_target_index_correct():
    notes, target = pairbench.memory_pool(item(), 1)
    assert len(notes) == 16 and notes[target] == 'I love really spicy food.'
    assert pairbench.memory_pool(item(), 1) == (notes, target)


@pytest.mark.parametrize('method', ['L0', 'L1', 'P'])
def test_context_never_contains_situation_or_roles(method):
    notes, _ = pairbench.memory_pool(item(), 1)
    text = pairbench.context_for(method, notes)
    assert 'toddler' not in text and 'alone' not in text and 'FOLLOW' not in text
    assert (text == '') == (method == 'L0')
    assert ('Use a remembered preference only when' in text) == (method == 'P')


def test_user_message_differs_only_by_situation():
    i = item()
    pos, neg = (pairbench.user_message(i, s) for s in pairbench.SIDES)
    assert pos.endswith(i['draft']['request']) and neg.endswith(i['draft']['request']) and pos != neg


def test_select_maps_label_through_hidden_order_and_hides_roles():
    seen = []
    class Fake:
        async def ask(self, role, system, value, schema, tag, max_tokens=None):
            seen.append(value)
            return schema(label='C')
    i = item()
    picked = asyncio.run(pairbench.select(Fake(), 'chat', i, 'negative', '', 7))
    assert picked == pairbench.selection_order(i, 'negative', 7)[2]
    assert 'memory' not in seen[0] and set(seen[0]['candidate_replies']) == set('ABCD')
    assert 'set_aside' not in str(seen[0]) and 'follow' not in str(seen[0]).replace('FOLLOW', '')


def test_selection_metrics():
    rows = [dict(positive='follow', negative='set_aside_silent'),
            dict(positive='follow', negative='follow'),
            dict(positive='set_aside_ack', negative='off_task')]
    m = pairbench.selection_metrics(rows)
    assert m['use_pos'] == pytest.approx(2/3) and m['use_neg'] == pytest.approx(1/3)
    assert m['pair_success'] == pytest.approx(1/3) and m['off_task_rate'] == pytest.approx(1/6)


def g(uptake, fit=1, hedged=False):
    return dict(uptake=uptake, situation_fit=fit, hedged=hedged)


def test_generation_metrics_and_hedge_exclusion():
    rows = [dict(positive=g(1), negative=g(0)),          # success
            dict(positive=g(.5), negative=g(.5)),        # use both
            dict(positive=g(1), negative=g(0, fit=0)),   # fails fit
            dict(positive=g(1, hedged=True), negative=g(0))]
    m = pairbench.generation_metrics(rows)
    assert m['pairs'] == 3 and m['hedged_excluded'] == 1
    assert m['pair_success'] == pytest.approx(1/3) and m['use_both'] == pytest.approx(1/3)
    assert m['use_neg'] == pytest.approx(1/3)


def test_bootstrap_pair_returns_interval():
    a = [dict(positive='follow', negative='set_aside_ack')] * 10 + [dict(positive='follow', negative='follow')] * 10
    b = [dict(positive=g(1), negative=g(1))] * 20
    ci = pairbench.bootstrap_pair(a, b, [str(i) for i in range(20)],
                                  lambda r: pairbench.selection_metrics(r)['pair_success'],
                                  lambda r: pairbench.generation_metrics(r)['pair_success'], seed=1, resamples=200)
    assert 0 < ci[0] <= .5 <= ci[1] < 1


def decisions(*labels):
    return [dict(unit_id=f'u{i}', decision=d, p_apply=float(d == 'apply'), reason='gated') for i, d in enumerate(labels)]


@pytest.mark.parametrize('side,expected', [('positive', 'apply'), ('negative', 'suppress')])
def test_oracle_o1_fixes_only_the_target(side, expected):
    out = pairbench.oracle_decisions(decisions('apply', 'suppress', 'uncertain'), [3, 7, 9], 7, side, 'O1')
    assert [d.decision for d in out] == ['apply', expected, 'uncertain']
    assert out[1].reason == 'gate oracle' and out[0].reason == 'gated'


def test_oracle_o2_suppresses_companions_and_keeps_missing_target_missing():
    out = pairbench.oracle_decisions(decisions('apply', 'apply'), [3, 7], 7, 'positive', 'O2')
    assert [d.decision for d in out] == ['suppress', 'apply']
    missing = pairbench.oracle_decisions(decisions('apply', 'apply'), [3, 4], 7, 'positive', 'O2')
    assert [d.decision for d in missing] == ['suppress', 'suppress']


def test_render_oracle_omits_suppressed_target_and_matches_spm_render():
    from spm.config import VARIANTS
    from spm.models import Decision, PrefUnit, Evidence
    from spm.pipeline import SPM, Candidate
    def unit(uid, text):
        return PrefUnit(id=uid, content=text, polarity='prefer', scope='food', domain='food', nature='taste',
                        evidence=[Evidence(episode_id='e', turn_idx=0, span=text, expression='explicit')], confidence=.9)
    units = [unit('u0', 'The user loves spicy food.'), unit('u1', 'The user likes tea.')]
    cfg = VARIANTS['v6_compact']
    ds = [Decision(**d) for d in decisions('suppress', 'apply')]
    text = pairbench.render_oracle(units, ds, cfg)
    assert 'spicy' not in text and 'tea' in text
    assert text == SPM('u', None, None, cfg).render([Candidate(u, 0.) for u in units], ds)


def test_gate_correct_uses_side_label():
    record = dict(spm=dict(positive=dict(target_decision=dict(decision='apply')),
                           negative=dict(target_decision=dict(decision='uncertain'))))
    # uncertain is not rendered, so on the negative side it behaves like suppress
    assert pairbench.gate_correct(record, 'positive') and pairbench.gate_correct(record, 'negative')
    assert not pairbench.gate_correct(dict(spm=dict(negative=dict(target_decision=dict(decision='apply')))), 'negative')
    assert not pairbench.gate_correct(dict(spm=dict(positive=dict(target_decision=None))), 'positive')
    assert pairbench.gate_correct(dict(spm=dict(negative=dict(target_decision=None))), 'negative')


def test_grade_schema_rejects_off_scale_uptake():
    with pytest.raises(Exception):
        pairbench.Grade(uptake=0.7, situation_fit=1, hedged=False, reason='x')
