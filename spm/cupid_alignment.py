"""CUPID's released response-alignment prompt, without the inference judge.

The upstream YAML is retained verbatim and is data, not executable code.
Only the four official fields reach the judge. Method IDs and prior grades do not.
"""
from pathlib import Path
import re
import textwrap


def load_prompt(path):
    raw = Path(path).read_text()
    prefix = 'system_prompt: |-\n'
    separator = '\nuser_prompt: |-\n'
    if not raw.startswith(prefix) or raw.count(separator) != 1:
        raise ValueError('Unexpected official two-block prompt format')
    system, user = raw[len(prefix):].split(separator)
    return textwrap.dedent(system).rstrip(), textwrap.dedent(user).rstrip()


def judge_inputs(templates, request, response, gold):
    """Use ground-truth checklist, never the model's inferred checklist."""
    if not isinstance(request, str) or not isinstance(response, str) or not response.strip():
        raise ValueError('A saved nonempty response and request are required')
    checklist = gold['checklist']
    if not isinstance(checklist, list) or not all(isinstance(x, str) for x in checklist):
        raise ValueError('Expected the original ground-truth checklist')
    values = dict(user_request=request, ai_response=response,
                  preference=gold['preference'], checklist='\n'.join(f'- {x}' for x in checklist))
    return tuple(template.format(**values) for template in templates)


def parse_score(raw):
    """Require one terminal numeric score, avoiding numbers in the analysis.

    Upstream exposes integer rubric labels. Reject missing/ambiguous/out-of-range
    scores rather than coercing them, clamping them, or recording failures as zero.
    """
    headers = list(re.finditer(r'^###\s+(?:\*\*)?Evaluation Score(?:\*\*)?\s*$', raw, re.M))
    if len(headers) != 1:
        raise ValueError('Exactly one Evaluation Score section is required')
    tail = raw[headers[0].end():].strip()
    match = re.fullmatch(r'(?:\*\*)?(10|[1-9])(?:\.0+)?(?:\*\*)?', tail)
    if not match:
        raise ValueError('The final score must be a single integer from 1 to 10')
    return int(match[1])
