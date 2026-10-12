from dataclasses import asdict, dataclass, replace
import os
from pathlib import Path
from urllib.parse import urlparse
from dotenv import load_dotenv


@dataclass(frozen=True)
class SPMConfig:
    use_condition: bool = True
    use_anticipation: bool = True
    use_gate: bool = True
    gate_policy: str = "answer_filter"
    use_raw_evidence: bool = True
    tool_protocol: str = "compact"
    memory_manager: str = "graph"
    gate_completion: str = "strict"
    update_policy: str = "legacy"
    raw_evidence_chars: int = 6000
    arbitration: str = "arbitrate"
    top_k_per_probe: int = 5
    sim_cand: float = 0.55
    max_relations: int = 6
    max_candidates: int = 20
    raw_fallback_k: int = 3
    embed_raw: str = "flagged"
    gate_apply_threshold: float = 0.5
    gate_suppress_threshold: float = 0.2
    render_suppressed: bool = False
    max_apply: int = 8
    window_messages: int = 12
    window_chars: int = 14000
    max_context_chars: int = 300000

    def __post_init__(self):
        if self.tool_protocol not in {"legacy", "compact", "small", "flags"}:
            raise ValueError("Invalid tool protocol")
        if self.memory_manager not in {'graph', 'flat'}:
            raise ValueError('Invalid memory manager')
        if self.update_policy not in {'legacy','conservative','typed'}:
            raise ValueError('Invalid update policy')
        if self.update_policy=='conservative' and self.memory_manager!='graph':
            raise ValueError('Conservative updates require graph memory')
        if self.update_policy=='typed' and (self.memory_manager!='graph' or self.tool_protocol not in {'small','flags'}):
            raise ValueError('Typed updates require graph memory with small protocol')
        if self.gate_completion not in {'strict', 'targeted'}:
            raise ValueError('Invalid gate completion')
        if self.gate_completion == 'targeted' and self.tool_protocol not in {'compact','small','flags'}:
            raise ValueError('Targeted completion requires compact protocol')
        if self.gate_policy not in {"legacy", "answer_filter", "situational"}:
            raise ValueError("Invalid gate policy")
        if self.arbitration not in {"overwrite", "branch", "arbitrate"}:
            raise ValueError("Invalid arbitration")
        if not 0 <= self.gate_suppress_threshold < self.gate_apply_threshold <= 1:
            raise ValueError("Gate thresholds must satisfy 0 <= suppress < apply <= 1")
        if self.embed_raw not in {"none", "flagged", "all"}:
            raise ValueError("Invalid raw embedding mode")
        if min(self.top_k_per_probe,self.max_relations,self.max_candidates,self.max_apply,self.window_messages,self.max_context_chars,self.raw_evidence_chars)<=0 or self.window_chars<2 or self.raw_fallback_k<0:
            raise ValueError('Memory and context budgets must be positive')
        if not 0<=self.sim_cand<=1:raise ValueError('Invalid candidate similarity threshold')

    def dump(self):
        return asdict(self)

    def write_signature(self, kind='spm'):
        """Only knobs that affect stored units, relations or raw evidence."""
        fields=('use_condition','arbitration','sim_cand','max_relations','embed_raw',
                'window_messages','window_chars','tool_protocol','memory_manager')
        signature=dict(kind=kind,**{name:getattr(self,name) for name in fields})
        # Preserve existing legacy bank identities; the new writer has its own banks.
        if self.update_policy!='legacy':signature['update_policy']=self.update_policy
        return signature


BASE = SPMConfig(use_condition=False, use_anticipation=False, use_gate=False, use_raw_evidence=False, tool_protocol="legacy", arbitration="overwrite")
VARIANTS = {"v0": BASE}
VARIANTS["v1"] = replace(BASE, use_condition=True, arbitration="branch")
VARIANTS["v2"] = replace(VARIANTS["v1"], arbitration="arbitrate")
VARIANTS["v3"] = replace(VARIANTS["v2"], use_anticipation=True)
VARIANTS["v4"] = replace(VARIANTS["v3"], use_gate=True,gate_policy="legacy")
VARIANTS["v5"] = replace(VARIANTS["v4"], gate_policy="answer_filter")
VARIANTS["v6"] = replace(VARIANTS["v5"], use_raw_evidence=True)
VARIANTS["v6_compact"] = replace(VARIANTS["v6"], tool_protocol="compact")
VARIANTS["v6_compact_no_raw"] = replace(VARIANTS["v6_compact"], use_raw_evidence=False)
VARIANTS['v6_m6'] = replace(VARIANTS['v6_compact'], gate_completion='targeted')
VARIANTS['flat_m6'] = replace(VARIANTS['v6_m6'], memory_manager='flat')
VARIANTS['v6_m7'] = replace(VARIANTS['v6_m6'], update_policy='conservative')
VARIANTS['flat_m8'] = replace(VARIANTS['flat_m6'], tool_protocol='small')
VARIANTS['v6_m8'] = replace(VARIANTS['v6_m6'], tool_protocol='small')
VARIANTS['typed_m8'] = replace(VARIANTS['v6_m8'], update_policy='typed')
VARIANTS['raw_m8'] = replace(VARIANTS['v6_m8'], use_gate=False)
VARIANTS['flat_m9'] = replace(VARIANTS['flat_m8'], tool_protocol='flags')
VARIANTS['v6_m9'] = replace(VARIANTS['v6_m8'], tool_protocol='flags')
VARIANTS['typed_m9'] = replace(VARIANTS['typed_m8'], tool_protocol='flags')
VARIANTS['raw_m9'] = replace(VARIANTS['raw_m8'], tool_protocol='flags')
VARIANTS['b2_matched'] = replace(VARIANTS['v6_m6'], use_gate=False)
VARIANTS['v6_situational'] = replace(VARIANTS['v6_compact'], gate_policy='situational')
VARIANTS["no_condition"] = replace(VARIANTS["v5"], use_condition=False)
VARIANTS["no_anticipation"] = replace(VARIANTS["v5"], use_anticipation=False)
METHODS = list(VARIANTS) + ["b0", "b1", "b2", "b4"]


@dataclass(frozen=True)
class Endpoint:
    model: str
    key: str
    base_url: str

    def __repr__(self):
        return f"Endpoint(model={self.model!r}, credentials=<redacted>)"

    def __post_init__(self):
        p = urlparse(self.base_url)
        if not self.key or p.scheme not in {"http", "https"} or not p.netloc or p.username or p.password or p.query or p.fragment:
            raise ValueError("API endpoint requires a key and HTTP(S) URL without embedded credentials")


def endpoints(path=Path(".env")):
    load_dotenv(path, override=False)
    key = os.environ.get("OPENAI_API_KEY", "")
    url = os.environ.get("OPENAI_BASE_URL", "")
    def role(prefix, default, fallback=None):
        return Endpoint(os.environ.get(prefix, "").strip() or default,
            os.environ.get(prefix + "_API_KEY", "").strip() or (fallback.key if fallback else key),
            os.environ.get(prefix + "_BASE_URL", "").strip() or (fallback.base_url if fallback else url))
    chat = role("CHAT_MODEL", "Qwen/Qwen2.5-7B-Instruct")
    cheap = role("CHEAP_MODEL", chat.model, chat)
    embed = Endpoint(os.environ.get("EMBED_MODEL", "").strip() or "BAAI/bge-m3",
        os.environ.get("EMBED_API_KEY", "").strip() or key,
        os.environ.get("EMBED_BASE_URL", "").strip() or url)
    return dict(chat=chat, cheap=cheap, judge=role("JUDGE_MODEL", "Pro/deepseek-ai/DeepSeek-V3.2"),
                embed=embed, bench_judge=role("PRES_JUDGE_MODEL", "Pro/deepseek-ai/DeepSeek-R1"))
