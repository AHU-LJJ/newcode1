from pathlib import Path

import pytest

from spm.cupid_alignment import judge_inputs, load_prompt, parse_score


@pytest.mark.parametrize('ending,score', [('1', 1), ('10', 10), ('**9**', 9), ('8.0', 8)])
def test_score_uses_terminal_section(ending, score):
    assert parse_score('### Evaluation of AI Assistant\'s Response\n1. Analysis mentions 10.\n'
                       '### Evaluation Score\n' + ending) == score
    assert parse_score('### **Evaluation Score**\n' + ending) == score


@pytest.mark.parametrize('text', [
    'Analysis: 10', '### Evaluation Score\n0', '### Evaluation Score\n11',
    '### Evaluation Score\n7.5', '### Evaluation Score\nTrue',
    '### Evaluation Score\n8 or 9', '### Evaluation Score\n8\nExtra text',
    '### Evaluation Score\n8\n### Evaluation Score\n9',
])
def test_invalid_grade_stays_missing(text):
    with pytest.raises(ValueError):
        parse_score(text)


def test_official_prompt_uses_only_reference_checklist():
    root = Path(__file__).resolve().parents[1]
    template = load_prompt(root/'paper/cupid_official/prompts/evaluation/response_judger.yaml')
    system, user = judge_inputs(template, 'request {literal}', 'answer {literal}',
                              dict(preference='gold preference', checklist=['gold one', 'gold two']))
    assert '- gold one\n- gold two' in system
    assert 'Focus only on the evaluation checklist' in system
    assert 'request {literal}' in user and 'answer {literal}' in user
    assert 'gold preference' in system
    assert '{checklist}' not in system


def test_invalid_upstream_template_rejected(tmp_path):
    path = tmp_path/'prompt.yaml'
    path.write_text('unexpected: template')
    with pytest.raises(ValueError):
        load_prompt(path)
