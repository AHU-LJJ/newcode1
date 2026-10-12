from collections import Counter

import numpy as np
import pytest

from spm import pairset


def draft(**changes):
    reply = ' '.join(['word'] * 70)
    value = dict(preference='I love really spicy food.', request='What should I order tonight?',
                 positive_situation='I am eating alone tonight at a quiet bar.',
                 negative_situation='My toddler is sharing every plate with me tonight.',
                 reply_follow='Order the fiery dishes. ' + reply, reply_set_aside_ack='I will go milder here. ' + reply,
                 reply_set_aside_silent='Try the classic dishes. ' + reply, reply_off_task='Paris is lovely in spring. ' + reply,
                 companions_near=[f'I like near {i}.' for i in range(4)],
                 companions_middle=[f'I like middle {i}.' for i in range(5)],
                 companions_far=[f'I like far {i}.' for i in range(6)], rationale='The toddler shares the food.')
    return pairset.PairDraft(**dict(value, **changes))


def test_slots_cover_every_category_domain_combination():
    slots = pairset.slots()
    assert len(slots) == 320 and len({s['id'] for s in slots}) == 320
    assert Counter(s['category'] for s in slots) == {c: 40 for c in pairset.CATEGORIES}


def test_pilot_slots_are_category_balanced_and_deterministic():
    chosen = pairset.pilot_slots(24, 1)
    assert Counter(s['category'] for s in chosen) == {c: 3 for c in pairset.CATEGORIES}
    assert chosen == pairset.pilot_slots(24, 1) and chosen != pairset.pilot_slots(24, 2)


def test_clean_draft_passes_rules():
    assert pairset.rule_issues(draft()) == []


@pytest.mark.parametrize('changes,issue', [
    (dict(reply_set_aside_silent='Since you love heat, try these. ' + ' '.join(['w'] * 70)), 'silent_acknowledges'),
    (dict(reply_follow='As you mentioned before, go hot. ' + ' '.join(['w'] * 70)), 'follow_mentions_memory'),
    (dict(negative_situation='For once skip my usual spicy picks, my toddler shares.'), 'negative_explicit_cue'),
    (dict(reply_off_task='Too short reply here.'), 'off_task_length'),
    (dict(reply_follow='Hot. ' + ' '.join(['w'] * 190)), 'length_imbalance'),
    (dict(companions_far=['I love really spicy food.'] + [f'I like far {i}.' for i in range(5)]), 'duplicate_companions'),
])
def test_rule_issues_detect_defects(changes, issue):
    assert issue in pairset.rule_issues(draft(**changes))


@pytest.mark.parametrize('text', ['Set aside a regular study slot each week.', 'Refresh accents whenever you like.',
                                  'Funny gifts are the ones people remember.'])
def test_rules_do_not_flag_ordinary_phrases(text):
    reply = text + ' ' + ' '.join(['word'] * 70)
    assert pairset.rule_issues(draft(reply_set_aside_silent=reply, reply_follow=reply)) == []


def test_dedup_keeps_first_and_records_duplicate():
    v = np.array([[1, 0], [0.999, 0.0447], [0, 1]], dtype=np.float32)
    v /= np.linalg.norm(v, axis=1, keepdims=True)
    kept, dropped = pairset.dedup(['a', 'b', 'c'], v, .92)
    assert kept == ['a', 'c'] and dropped['b']['duplicate_of'] == 'a'


def test_screening_orders_place_follow_at_distinct_positions():
    orders = pairset.screening_orders('x', 7)
    assert len(orders) == 3 and len({o.index('follow') for o in orders}) == 3
    assert all(sorted(o) == sorted(pairset.REPLY_ROLES) for o in orders)
    assert orders == pairset.screening_orders('x', 7)


def test_screen_rejects_when_follow_chosen_more_than_once():
    assert pairset.screen_passed(['follow', 'set_aside_ack', 'set_aside_silent'])
    assert not pairset.screen_passed(['follow', 'follow', 'set_aside_ack'])


def test_screen_hides_preference_and_situations():
    import asyncio
    seen = []
    class Fake:
        async def ask(self, role, system, value, schema, tag, max_tokens=None):
            seen.append(value)
            return schema(label='B')
    d = draft()
    orders = pairset.screening_orders('x', 7)
    picks = asyncio.run(pairset.screen(Fake(), d, orders))
    assert picks == [o[1] for o in orders]
    for value in seen:
        text = str(value)
        assert set(value) == {'request', 'replies'}
        assert d.preference not in text and d.positive_situation not in text and d.negative_situation not in text


def test_builder_env_is_isolated_and_url_normalized(tmp_path, monkeypatch):
    for name in ('OPENAI_API_KEY', 'OPENAI_BASE_URL', 'OPENAI_MODEL'):
        monkeypatch.delenv(name, raising=False)
    main = tmp_path/'main.env'
    main.write_text('OPENAI_API_KEY=sf-key\nOPENAI_BASE_URL=https://api.siliconflow.cn/v1\n')
    zhipu = tmp_path/'zhipu.env'
    zhipu.write_text('OPENAI_API_KEY=zp-key\nOPENAI_BASE_URL=https://open.bigmodel.cn/api/paas/v4/chat/completions\nOPENAI_MODEL=glm-5.3-flash\n')
    eps = pairset.build_endpoints(main, zhipu)
    assert eps['builder'].base_url == 'https://open.bigmodel.cn/api/paas/v4' and eps['builder'].model == 'glm-5.3-flash'
    assert eps['builder'].key == 'zp-key' and eps['judge'].key == 'sf-key' and eps['verifier'] == eps['judge']
    assert 'zp-key' not in repr(eps['builder'])


@pytest.mark.parametrize('url,expected', [('https://api.siliconflow.cn/v1', False), ('https://open.bigmodel.cn/api/paas/v4', None)])
def test_enable_thinking_sent_only_to_siliconflow(tmp_path, url, expected):
    import asyncio, httpx
    from spm.api import Gateway
    from spm.config import Endpoint
    seen = []
    def handler(request):
        import json
        seen.append(json.loads(request.content))
        return httpx.Response(200, json=dict(choices=[dict(message=dict(content='ok'), finish_reason='stop')],
                                             usage=dict(total_tokens=3)))
    async def run():
        api = Gateway(dict(chat=Endpoint('m', 'k', url)), tmp_path/'c.sqlite', tpm_limits={})
        await api.http.aclose()
        api.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        assert await api.ask('chat', 'sys', 'hi') == 'ok'
        await api.close()
    asyncio.run(run())
    assert seen[0].get('enable_thinking') is expected


def test_provider_limit_caps_one_host_without_blocking_another(tmp_path):
    import asyncio, httpx, json
    from spm.api import Gateway
    from spm.config import Endpoint
    active = dict(slow=0, fast=0); peak = dict(slow=0, fast=0)
    async def handler(request):
        host = 'slow' if request.url.host == 'slow.example' else 'fast'
        active[host] += 1; peak[host] = max(peak[host], active[host])
        await asyncio.sleep(.02)
        active[host] -= 1
        return httpx.Response(200, json=dict(choices=[dict(message=dict(content='ok'), finish_reason='stop')],
                                             usage=dict(total_tokens=1)))
    async def run():
        eps = dict(slow=Endpoint('m', 'k', 'https://slow.example/v1'), fast=Endpoint('m', 'k', 'https://fast.example/v1'))
        api = Gateway(eps, tmp_path/'c.sqlite', concurrency=10, tpm_limits={}, provider_limits={'slow.example': 2})
        await api.http.aclose()
        api.http = httpx.AsyncClient(transport=httpx.MockTransport(handler))
        await asyncio.gather(*(api.ask(r, 'sys', f'{r}{i}') for r in ('slow', 'fast') for i in range(6)))
        await api.close()
    asyncio.run(run())
    assert peak['slow'] == 2 and peak['fast'] > 2


def test_verdict_passes_only_when_all_checks_true():
    fields = {k: True for k in pairset.PairVerdict.model_fields if k != 'reason'}
    assert pairset.PairVerdict(**fields, reason='ok').passed()
    assert not pairset.PairVerdict(**dict(fields, negative_should_set_aside=False), reason='no').passed()
