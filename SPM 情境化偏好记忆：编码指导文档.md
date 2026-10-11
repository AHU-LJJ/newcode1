# SPM 情境化偏好记忆：编码指导文档

Oct 9, 2026 · @junjie lv

## 目标与范围

首版目标是用 API 搭一个免训练的偏好记忆系统，在 PrefEval、PERMA、CUPID、BenchPreS 四个基准上同时可评测。文中代码是参考骨架，没有运行过：每个模块先写冒烟测试，再接基准。

- **系统做什么**：把对话抽取成带情境条件的偏好单元，查询时先判断哪些偏好适用，再把“应用 / 应抑制”清单交给生成器。
- **首版不做**：训练模型、隐空间记忆、数值化的评价向量。这些放到后续阶段，作为可选扩展。
- **API 约束**：只调用聊天与嵌入接口，不需要 GPU。CUPID 的 PrefMatcher-7B 是可选项，默认用 API 裁判替代。
- **数据获取**：只有 PrefEval 的数据在代码仓库里；PERMA、CUPID、BenchPreS 在 HuggingFace，需要你手动下载后放到 `data/` 下。

| 基准 | 数据位置 | 对系统的要求 | 评测方式 |
| --- | --- | --- | --- |
| PrefEval | 仓库内 `benchmark_dataset/`：显式、选择式、人设式各 1000 条，20 个主题 | 找出偏好，并在查询没提到时也用上；选择式要从选项取舍中反推 | 生成：LLM 判四类错误；分类：4 选 1 准确率 |
| PERMA | HuggingFace `ustclsc/PERMA` | 偏好随事件演化，抗噪声，跨领域 | 8 选项多选题；用户模拟器的 Turn=1 与 Turn≤2 成功率；干净 / 噪声 × 单 / 多领域 |
| CUPID | HuggingFace `kixlab/CUPID`：756 条，每条 8 个历史会话 | 区分不同对象下互斥的偏好，处理同一对象的偏好变更 | 推断：偏好句匹配的精确率与召回率；生成：1–10 分 |
| BenchPreS | HuggingFace `sangyon/BenchPreS`：10 个画像 × 39 个情境 | 对每条给定偏好判断应用还是抑制 | 误用率 MR、恰当应用率 AAR，裁判为 DeepSeek-R1 |

## 2 总体架构与数据流

系统由写入和读取两条独立路径组成，两者只通过偏好单元库相连。

&#91;embedded content: SPM 架构 · 写入路径 4 步，读取路径 4 步，共用一个单元库\]

写入路径在每个会话结束后运行，读取路径在每个请求时运行。原始会话永不修改，每个偏好单元都能追溯到原文。第 3 节的四个配置开关对应图里的组件：`use_condition` 管条件，`arbitration` 管仲裁，`use_anticipation` 管探针，`use_gate` 管门控。

## 环境与项目结构

用 Python 3.11，模型和密钥全部走环境变量，并沿用 PERMA 的变量名，这样接入 PERMA 时不用反复配置。

**依赖**

```text
openai>=1.40          # OpenAI 兼容 SDK，换 base_url 即可切换厂商或本地 vLLM
pydantic>=2.6
numpy
tenacity
tqdm
python-dotenv
datasets              # 读 HuggingFace 数据集
pandas
scipy                 # bootstrap 置信区间
sentence-transformers # 可选：本地嵌入模型，默认不用
```

**环境变量（`.env`，不要提交到版本库）**

```text
CHAT_MODEL=deepseek-v3.2      # 强模型：查询分析、门控、生成；按服务商的实际名称填
CHAT_MODEL_API_KEY=...
CHAT_MODEL_BASE_URL=...
CHEAP_MODEL=qwen2.5-7b-instruct   # 便宜模型：抽取、关系判断
CHEAP_MODEL_API_KEY=...
CHEAP_MODEL_BASE_URL=...
JUDGE_MODEL=GLM-5.3-Flash     # 评测裁判，与生成模型不同家；按服务商的实际名称填
JUDGE_MODEL_API_KEY=...
JUDGE_MODEL_BASE_URL=...
EMBED_MODEL=<嵌入模型>
EMBED_API_KEY=...
EMBED_BASE_URL=...
```

**目录**

```text
spm/
  pyproject.toml
  .env.example
  configs/default.yaml
  src/spm/
    config.py            # 配置与消融开关
    models.py            # Episode / PrefUnit / Condition / Evidence / Decision
    llm.py               # 聊天客户端：缓存、重试、JSON 校验
    embed.py             # 嵌入客户端（带缓存）
    store.py             # sqlite + numpy 向量
    prompts/             # extract.txt analyze.txt gate.txt arbitrate.txt
    write/   prefilter.py extractor.py consolidate.py
    read/    analyze.py retrieve.py gate.py render.py
    pipeline.py          # SPM 类：ingest() finalize() query()
    baselines.py         # 全上下文 / 提醒 / RAG
  adapters/  prefeval.py perma.py cupid.py benchpres.py
  eval/      metrics.py stats.py plots.py
  tests/
  data/                  # 加入 .gitignore：各基准数据
  runs/                  # 加入 .gitignore：缓存、轨迹、结果
```

**配置与消融开关**

```python
# src/spm/config.py
from dataclasses import dataclass

@dataclass
class SPMConfig:
    # 消融开关：依次打开就是 v0 -> v4
    use_condition: bool = True       # 偏好单元带条件
    use_anticipation: bool = True    # 查询时做需求预判
    use_gate: bool = True            # 适用性门控
    arbitration: str = 'arbitrate'  # overwrite | branch | arbitrate
    use_sensitivity: bool = False    # 情境敏感度回退（第三阶段再开）

    # 检索
    top_k_per_probe: int = 5
    sim_same: float = 0.92           # 直接视为同一偏好的相似度
    sim_cand: float = 0.60           # 进入冲突判断的相似度下限
    embed_raw: str = 'flagged'       # 原始片段向量：flagged | all | none
    max_candidates: int = 20
    raw_fallback_k: int = 3          # 召回不足时从原始库兜底

    # 置信度先验（需在开发集上校准，不要当成定值）
    conf_explicit: float = 0.9
    conf_incidental: float = 0.7
    conf_choice: float = 0.6
    conf_feedback: float = 0.7

    # 门控阈值：扫描这个值得到“遵循率-误用率”曲线
    gate_apply_threshold: float = 0.5
    gate_suppress_threshold: float = 0.2   # 低于它直接压制；中间区间为 uncertain，不渲染
    render_suppressed: bool = True         # 是否把被压制的偏好以 DO NOT APPLY 段渲染出来
    max_apply: int = 8                     # 关闭门控时最多放行的候选数
    llm_temperature: float = 0.0
```

## 数据模型

所有模块只通过一组 pydantic 模型交换数据，偏好单元（`PrefUnit`）是核心。先把这一层定死，再写写入和读取。

```python
# src/spm/models.py
from __future__ import annotations
import hashlib
from enum import Enum
from typing import Literal, Optional
from pydantic import BaseModel, Field

class Expression(str, Enum):
    EXPLICIT = "explicit"        # 一句话直接说出
    CHOICE = "choice"            # 从接受/拒绝选项中反推
    INCIDENTAL = "incidental"    # 顶带提到
    FEEDBACK = "feedback"        # 对回答的纠正或评价
    EVENT = "event"              # 事件驱动的变化（PERMA）

class Nature(str, Enum):
    CONSTRAINT = "constraint"    # 硬约束：健康、饮食、安全、严格要求
    TASTE = "taste"              # 个人口味
    STYLE = "style"              # 语气、格式、篇幅：随受众变化

class Status(str, Enum):
    CANDIDATE = "candidate"      # 假设，证据不足
    ACTIVE = "active"
    PENDING = "pending"          # 与旧单元冲突且拿不准
    SUPERSEDED = "superseded"    # 已被取代，保留但不再召回

class Message(BaseModel):
    role: Literal["user", "assistant"]
    content: str

class Episode(BaseModel):
    id: str
    order: int                    # 在该用户时间线上的先后顺序，必填
    time: Optional[str] = None    # ISO 时间；PERMA 有，CUPID 没有
    messages: list[Message]

class Condition(BaseModel):
    """CAPS 的 if 部分。全部为空 = 全局偏好。"""
    entity: Optional[str] = None       # 具名对象，如 "dr. chen"；用于硬匹配
    audience: Optional[str] = None     # 受众/收件人类型，如 "admissions committee"
    formality: Optional[Literal["casual", "neutral", "formal"]] = None
    task_type: Optional[str] = None    # 如 "writing", "recommendation"
    channel: Optional[str] = None      # 如 "chat_with_ai", "email_to_third_party"

    def is_global(self) -> bool:
        return not any([self.entity, self.audience, self.formality,
                        self.task_type, self.channel])

class Evidence(BaseModel):
    episode_id: str
    turn_idx: int
    span: str                          # 原文片段，用于回溯和渲染
    expression: Expression
    time: Optional[str] = None

class PrefUnit(BaseModel):
    id: str
    content: str                       # 规范化一句话，如 "avoids foods containing gluten"
    polarity: Literal["prefer", "avoid", "require"]
    scope: str                         # 类别级适用范围，如 "restaurant, recipe and food recommendations"
    domain: str                        # 短标签，如 "food"
    nature: Nature
    condition: Condition = Field(default_factory=Condition)
    evidence: list[Evidence]
    confidence: float
    n_support: int = 1
    n_contra: int = 0
    status: Status = Status.ACTIVE
    first_order: int = 0               # 首次观察的会话顺序
    last_order: int = 0                # 最近一次确认的会话顺序
    valid_to: Optional[str] = None
    superseded_by: Optional[str] = None
    challenges: Optional[str] = None   # PENDING 单元所挑战的旧单元 id

    def embed_text(self) -> str:
        return f"{self.content}. Applies to: {self.scope}"

    @staticmethod
    def make_id(user_id: str, content: str, cond: Condition) -> str:
        raw = f"{user_id}|{content.lower()}|{cond.model_dump_json()}"
        return hashlib.sha1(raw.encode()).hexdigest()[:16]

class ParsedQuery(BaseModel):
    situation: Condition               # 当前请求的情境
    probes: list[str]                  # 需求预判：用户哪些属性可能相关

class Decision(BaseModel):
    unit_id: str
    decision: Literal["apply", "suppress", "uncertain"]
    reason: str
```

**存储**：每个用户一个 SQLite 文件，三张表。`episodes(id, ord, time, json)` 保存原始会话（L0，只增不改）；`units(id, status, json)` 保存偏好单元；`vecs(kind, id, dim, blob)` 保存向量，`kind` 区分单元向量和原始片段向量。一个用户的规模在几千条以内，用 numpy 暂力余弦相似度即可，不需要 FAISS。

**实体归一化**：`entity` 在写入和查询两端都过同一个函数（小写、去标点、去冒号后的头衔描述），硬匹配才可靠。CUPID 里同类型对象的偏好互斥，实体键是区分它们的主要手段。

## API 封装

LLM 和嵌入客户端要先做好四件事：磁盘缓存、重试、JSON 校验、按标签统计用量。基准要反复跑，没有缓存，调试一次就是一笔开销；没有用量统计，论文里的成本对比就拿不出来。

````python
# src/spm/llm.py
import os, json, hashlib, sqlite3, threading, re
from collections import defaultdict
from typing import Type, TypeVar
from openai import OpenAI
from pydantic import BaseModel, ValidationError
from tenacity import retry, stop_after_attempt, wait_exponential
from dotenv import load_dotenv

load_dotenv()
T = TypeVar("T", bound=BaseModel)

class DiskCache:
    def __init__(self, path: str):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.lock = threading.Lock()
        self.conn.execute("CREATE TABLE IF NOT EXISTS kv (k TEXT PRIMARY KEY, v TEXT)")

    def get(self, k: str):
        with self.lock:
            row = self.conn.execute("SELECT v FROM kv WHERE k=?", (k,)).fetchone()
        return row[0] if row else None

    def set(self, k: str, v: str):
        with self.lock:
            self.conn.execute("INSERT OR REPLACE INTO kv VALUES (?,?)", (k, v))
            self.conn.commit()

def extract_json(text: str) -> str:
    """从模型输出里切出 JSON：去掉代码围栏，取第一个 { 到最后一个 }。"""
    text = re.sub(r"```(?:json)?", "", text)
    i, j = text.find("{"), text.rfind("}")
    if i < 0 or j < i:
        raise ValueError("no JSON object in output")
    return text[i : j + 1]

class LLM:
    def __init__(self, prefix="CHAT_MODEL", cache_path="runs/cache_llm.sqlite",
                 temperature=0.0, max_tokens=2048, max_concurrency=8):
        self.client = OpenAI(api_key=os.environ[f"{prefix}_API_KEY"],
                             base_url=os.environ.get(f"{prefix}_BASE_URL") or None)
        self.model = os.environ[prefix]
        self.temperature, self.max_tokens = temperature, max_tokens
        self.cache = DiskCache(cache_path)
        self.sem = threading.Semaphore(max_concurrency)
        self.usage = defaultdict(lambda: {"calls": 0, "cached": 0, "in": 0, "out": 0})

    def _key(self, system, user, tag):
        raw = json.dumps([self.model, self.temperature, system, user, tag], ensure_ascii=False)
        return hashlib.sha256(raw.encode()).hexdigest()

    @retry(stop=stop_after_attempt(5), wait=wait_exponential(min=1, max=30))
    def _call(self, system, user):
        with self.sem:
            r = self.client.chat.completions.create(
                model=self.model, temperature=self.temperature, max_tokens=self.max_tokens,
                messages=[{"role": "system", "content": system},
                          {"role": "user", "content": user}])
        return r.choices[0].message.content or "", r.usage

    def chat(self, system: str, user: str, tag: str = "misc") -> str:
        k = self._key(system, user, tag)
        hit = self.cache.get(k)
        u = self.usage[tag]
        if hit is not None:
            u["cached"] += 1
            return hit
        text, usage = self._call(system, user)
        u["calls"] += 1
        if usage:
            u["in"] += usage.prompt_tokens
            u["out"] += usage.completion_tokens
        self.cache.set(k, text)
        return text

    def json(self, system: str, user: str, schema: Type[T], tag: str = "misc", retries: int = 2) -> T:
        """要求模型输出 JSON，用 pydantic 校验；失败则把错误贴回去重试。"""
        prompt, last_err = user, None
        for _ in range(retries + 1):
            out = self.chat(system, prompt, tag)
            try:
                return schema.model_validate_json(extract_json(out))
            except (ValueError, ValidationError) as e:
                last_err = e
                prompt = (user + "\n\nYour previous output was invalid: " + str(e)[:300]
                          + "\nReturn only a valid JSON object.")
        raise RuntimeError(f"[{tag}] invalid JSON after retries: {last_err}")

    def dump_usage(self, path="runs/usage.json"):
        json.dump(self.usage, open(path, "w"), indent=2)
````

```python
# src/spm/embed.py
import os, json, hashlib
import numpy as np
from openai import OpenAI
from .llm import DiskCache

class Embedder:
    def __init__(self, cache_path="runs/cache_embed.sqlite", batch=64):
        self.client = OpenAI(api_key=os.environ["EMBED_API_KEY"],
                             base_url=os.environ.get("EMBED_BASE_URL") or None)
        self.model = os.environ["EMBED_MODEL"]
        self.cache, self.batch = DiskCache(cache_path), batch

    def encode(self, texts: list[str]) -> np.ndarray:
        keys = [hashlib.sha1((self.model + t).encode()).hexdigest() for t in texts]
        out, todo = [None] * len(texts), []
        for i, k in enumerate(keys):
            hit = self.cache.get(k)
            if hit is not None:
                out[i] = np.array(json.loads(hit), dtype=np.float32)
            else:
                todo.append(i)
        for s in range(0, len(todo), self.batch):
            idx = todo[s : s + self.batch]
            r = self.client.embeddings.create(model=self.model,
                                              input=[texts[i][:8000] for i in idx])
            for i, d in zip(idx, r.data):
                out[i] = np.array(d.embedding, dtype=np.float32)
                self.cache.set(keys[i], json.dumps(out[i].tolist()))
        m = np.stack(out)
        return m / np.linalg.norm(m, axis=1, keepdims=True).clip(min=1e-9)
```

嵌入缓存用 JSON 序列化向量，读缓存时不要用 `eval` 或 `pickle`。

**注意事项**

- 有些厂商不支持 `response_format=json_object`，所以这里用“提示 + 切 JSON + 校验重试”代替，对任何厂商都能用。
- 温度固定为 0，缓存键包含模型名，换模型不会误命中旧缓存。
- 强模型（生成、门控）、便宜模型（抽取）和裁判模型各用一个 `LLM` 实例（`prefix="CHAT_MODEL"` 与 `prefix="JUDGE_MODEL"`；便宜模型用 prefix=CHEAP\_MODEL），用量分开统计。
- 冒烟测试：对 `chat()` 调两次同样的输入，第二次应该 `cached` +1 且不再计入 `calls`。

## 写入路径实现

写入分四步：信号筛选、抽取、巩固、仲裁，每步一个文件。能用规则解决的就不调 LLM，因为 PrefEval 的干扰对话最长约 10 万 token，逐轮调用会让成本失控。

### 1. 信号筛选

筛选追求高召回：宁可多送一些窗口给抽取，也不要漏掉真偏好。短会话（≤ 12 条消息）整段送抽取，CUPID 的反馈式对话和 PrefEval 的偏好对话都属于这一类；长会话才走规则预筛。

```python
# src/spm/write/prefilter.py
import re
from ..models import Episode

CUES = re.compile(
    r'\b(prefer\w*|favou?rite|love|like|enjoy|hate|dislike|avoid\w*|allerg\w*|intoleran\w*|'
    r'vegetarian|vegan|strictly|always|never|usually|must|need|can.t|cannot|don.t|do not|'
    r'doesn.t|won.t|rather|instead|only|works? for me|not for me|sounds good|too (?:much|long|short|formal|casual))\b',
    re.I)
FIRST_PERSON = re.compile(r"\b(I|I'm|I've|my|me|we|our)\b")
OPTION_LIST = re.compile(r'(?m)^\s*(?:\d+[.)]|[-*•])\s+')

def flag_windows(ep: Episode, short_limit: int = 12) -> list[tuple[int, int]]:
    """返回需要送去抽取的 [a, b) 消息区间。"""
    msgs = ep.messages
    if len(msgs) <= short_limit:
        return [(0, len(msgs))]
    hit: set[int] = set()
    for i, m in enumerate(msgs):
        if m.role != 'user':
            continue
        cue = FIRST_PERSON.search(m.content) and CUES.search(m.content)
        choice = (i > 0 and msgs[i - 1].role == 'assistant'
                  and len(OPTION_LIST.findall(msgs[i - 1].content)) >= 2)
        if cue or choice:
            hit.update(range(max(0, i - 1), min(len(msgs), i + 2)))
    windows, start = [], None
    for i in range(len(msgs) + 1):
        if i in hit and start is None:
            start = i
        elif i not in hit and start is not None:
            windows.append((start, i))
            start = None
    return windows
```

验收方式：在 PrefEval 上统计两个数，偏好会话被标记的比例（目标 100%）和干扰会话被标记的比例（越低越省钱）。第一个数没达标，就扩充 `CUES`。

### 2. 抽取

抽取针对三类隐式信号。顶带提及要抓住从句；选择式要反推被拒选项的共同属性；反馈式要读懂纠正。

```python
# src/spm/write/extractor.py
from typing import Literal
from pydantic import BaseModel
from ..models import Condition, Expression, Nature, Episode, PrefUnit, Evidence, Status
from ..llm import LLM

class RawUnit(BaseModel):
    content: str
    polarity: Literal['prefer', 'avoid', 'require']
    scope: str
    domain: str
    nature: Nature
    expression: Expression
    condition: Condition = Condition()
    span: str
    turn_idx: int

class ExtractOut(BaseModel):
    units: list[RawUnit] = []

SYSTEM = open('src/spm/prompts/extract.txt').read()

def render_window(ep: Episode, a: int, b: int, max_chars: int = 1500) -> str:
    return '\n'.join(f'[{i}] {m.role}: {m.content[:max_chars]}'
                     for i, m in enumerate(ep.messages[a:b], start=a))

def extract(llm: LLM, ep: Episode, a: int, b: int) -> list[RawUnit]:
    user = f'Session time: {ep.time or "unknown"}\nConversation window:\n{render_window(ep, a, b)}'
    return llm.json(SYSTEM, user, ExtractOut, tag='extract').units

def to_unit(user_id: str, ru: RawUnit, ep: Episode, cfg) -> PrefUnit:
    conf = {Expression.EXPLICIT: cfg.conf_explicit, Expression.EVENT: cfg.conf_explicit,
            Expression.INCIDENTAL: cfg.conf_incidental, Expression.FEEDBACK: cfg.conf_feedback,
            Expression.CHOICE: cfg.conf_choice}[ru.expression]
    cond = ru.condition if cfg.use_condition else Condition()
    cond.entity = norm_entity(cond.entity)
    return PrefUnit(
        id=PrefUnit.make_id(user_id, ru.content, cond), content=ru.content,
        polarity=ru.polarity, scope=ru.scope, domain=ru.domain, nature=ru.nature,
        condition=cond, confidence=conf,
        status=Status.CANDIDATE if ru.expression == Expression.CHOICE else Status.ACTIVE,
        evidence=[Evidence(episode_id=ep.id, turn_idx=ru.turn_idx, span=ru.span,
                           expression=ru.expression, time=ep.time)],
        first_order=ep.order, last_order=ep.order)
```

```text
# src/spm/prompts/extract.txt
You extract durable USER preferences from a conversation window, for a personal memory.
Return JSON only: {"units": [ ... ]}. Each unit has: content, polarity (prefer|avoid|require),
scope, domain, nature (constraint|taste|style), expression (explicit|incidental|choice|feedback),
condition {entity, audience, formality (casual|neutral|formal), task_type, channel}, span, turn_idx.

Rules:
1. Only the user's own preferences, constraints, habits or requirements. Ignore hypothetical
   questions, third-person statements, one-off task details and plain facts.
2. expression:
   - explicit: stated directly.
   - incidental: mentioned in passing inside another topic. Example: 'a cafe where I can grab a
     gluten-free snack due to my severe intolerance' means the user avoids gluten.
   - choice: the assistant offered several options and the user accepted or rejected some.
     Infer the attribute that explains the choice and state it as a hypothesis. If two explanations
     fit, output at most 2 units, the most likely first.
   - feedback: the user corrects or reacts to an answer (too long, keep it formal, ...).
3. content: one short canonical third-person sentence, e.g. 'avoids foods containing gluten'.
4. scope: the CATEGORY of future requests where this matters. Broader than the example, but not
   unbounded. Example: 'restaurant, recipe and food recommendations', not 'snacks at bookstores'.
5. nature: constraint = health, diet, safety or a hard requirement; taste = personal liking;
   style = tone, format, length, level of detail.
6. condition: fill a field only if the text ties the preference to it (a named person, group,
   organization or object -> entity; a type of recipient -> audience). Use null for general
   preferences. Copy entity names exactly as written.
7. span: the exact words supporting the unit. turn_idx: the [index] of that turn.
If nothing qualifies return {"units": []}.
```

### 3. 巩固与仲裁

新单元写入前先和已有单元比较：相同则合并证据，互斥则仲裁。仲裁规则默认不依赖“不再”“改成”之类的措辞，因为 CUPID 的对话里大概率没有这类线索。

```python
# src/spm/write/consolidate.py
import re
from ..models import PrefUnit, Status, Expression, Condition
from pydantic import BaseModel
from typing import Literal

def norm_entity(e):
    if not e:
        return None
    e = re.sub(r'[^a-z0-9 ]', ' ', e.lower())
    return ' '.join(e.split()) or None

def differs(a: Condition, b: Condition) -> bool:
    """只比较实体和正式程度：受众、任务、渠道是自由文本，比较容易把同义词误判为不同。"""
    for f in ('entity', 'formality'):
        x, y = getattr(a, f), getattr(b, f)
        if x and y and x != y:
            return True
    return False

class Relation(BaseModel):
    relation: Literal['same', 'contradicts', 'unrelated']

class Consolidator:
    def __init__(self, store, llm, emb, cfg):
        self.store, self.llm, self.emb, self.cfg = store, llm, emb, cfg
        self.system = open('src/spm/prompts/relation.txt').read()

    def relation(self, old: PrefUnit, new: PrefUnit) -> str:
        user = (f'A (existing): {old.content} | scope: {old.scope} | condition: {old.condition.model_dump_json()}\n'
                f'B (new): {new.content} | scope: {new.scope} | condition: {new.condition.model_dump_json()}')
        return self.llm.json(self.system, user, Relation, tag='relation').relation

    def upsert(self, new: PrefUnit, ep) -> str:
        vec = self.emb.encode([new.embed_text()])[0]
        for old, sim in self.store.nearest_units(vec, k=5, min_sim=self.cfg.sim_cand):
            cheap_same = (sim >= self.cfg.sim_same and old.polarity == new.polarity)
            rel = 'same' if cheap_same else self.relation(old, new)
            if rel == 'same' and not differs(old.condition, new.condition):
                return self.merge(old, new, ep)
            if rel == 'contradicts':
                return self.resolve(old, new, ep, vec)
        self.store.put_unit(new, vec)
        return 'new'

    def merge(self, old, new, ep) -> str:
        old.evidence += new.evidence
        old.n_support += 1
        old.confidence = min(0.98, 1 - (1 - old.confidence) * (1 - new.confidence))
        old.last_order = max(old.last_order, new.last_order)
        if old.status == Status.PENDING and old.n_support >= 2 and old.challenges:
            target = self.store.get_unit(old.challenges)
            if target:
                self.supersede(target, old, ep)
            old.status, old.challenges = Status.ACTIVE, None
        self.store.update_unit(old)
        return 'merge'

    def supersede(self, old, new, ep):
        old.status, old.superseded_by = Status.SUPERSEDED, new.id
        old.valid_to = ep.time or str(ep.order)
        self.store.update_unit(old)

    def resolve(self, old, new, ep, vec) -> str:
        mode = self.cfg.arbitration          # overwrite | branch | arbitrate
        if mode == 'branch' or (mode == 'arbitrate' and differs(old.condition, new.condition)):
            self.store.put_unit(new, vec)
            return 'branch'
        explicit_change = new.evidence[0].expression in (Expression.EXPLICIT, Expression.EVENT)
        if (mode == 'overwrite' or explicit_change or old.n_support <= 1
                or new.confidence >= 0.8 * old.confidence):
            self.supersede(old, new, ep)
            new.status = Status.ACTIVE
            self.store.put_unit(new, vec)
            return 'supersede'
        new.status, new.challenges = Status.PENDING, old.id
        self.store.put_unit(new, vec)
        return 'pending'
```

```text
# src/spm/prompts/relation.txt
You compare an existing user preference A with a new observation B.
Return JSON only: {"relation": "same" | "contradicts" | "unrelated"}.
same: B says the same thing as A, possibly in other words or slightly broader or narrower.
contradicts: A and B cannot both hold for the same request (they are mutually exclusive).
unrelated: anything else, including different topics.
Ignore the condition when judging content; conditions are compared separately.
```

仲裁的逻辑对应 CUPID 的两类实例：

- **contrastive**：同类型、不同对象、偏好互斥，`differs` 为真，走分支，两个单元都保持有效。
- **changing**：同一对象先 A 后 A'。A' 第一次出现时旧单元已有多次支持，新单元进入 `PENDING`；A' 第二次出现时 `merge` 把它提升为 `ACTIVE` 并取代旧单元。PERMA 里用户明确说出的变更（`EXPLICIT` 或 `EVENT`）直接取代。
- `PENDING` 单元和被它挑战的旧单元读取时都会召回，由门控标为“不确定”或按时间取舍。

**存储层需要实现的方法**：`add_episode`、`put_unit(unit, vec)`、`update_unit(unit)`、`get_unit(id)`、`nearest_units(vec, k, min_sim)`（只在非 `SUPERSEDED` 单元中找）、`all_units(statuses)`。都是对 SQLite 表的简单读写。

### 4. 串起来

```python
# src/spm/pipeline.py（写入部分）
class SPM:
    def __init__(self, user_id, store, llm, emb, cfg, cheap_llm=None):
        self.user_id, self.store, self.llm, self.emb, self.cfg = user_id, store, llm, emb, cfg
        self.cheap = cheap_llm or llm      # 写入路径用便宜模型；没给就退回单一模型
        self.cons = Consolidator(store, self.cheap, emb, cfg)

    def ingest(self, ep: Episode):
        self.store.add_episode(ep)
        for a, b in flag_windows(ep):
            for ru in extract(self.cheap, ep, a, b):
                self.cons.upsert(to_unit(self.user_id, ru, ep, self.cfg), ep)
```

### 5. 冒烟测试（进入基准前必须通过）

1. **人设式**：用 PrefEval 里“图书馆员”那条对话，期望抽出一条 `avoids gluten`，范围覆盖餐厅和食谱推荐，证据片段包含 `severe intolerance`。
2. **选择式**：用“沙拉吧”那条，期望抽出含麸质排除的假设，且状态为 `CANDIDATE`。
3. **取代**：造 4 个会话，同一对象先两次 A、再两次 A'，期望最终只有 A' 是 `ACTIVE`。
4. **分支**：两个同类型、不同名字、偏好互斥的对象，期望两个单元都是 `ACTIVE`。
5. **噪声**：“假设我是素食者的话…”和“我朋友不吃辣”不应抽出单元。

## 读取路径实现

读取分四步：查询解析、召回、适用性门控、渲染。门控是核心，它同时决定 PrefEval 的“用上”和 BenchPreS 的“压住”。

### 1. 查询解析

一次 LLM 调用，输出当前请求的情境和需求预判探针。探针解决的是 PrefEval 的核心难点：查询没提偏好，纯语义相似度找不到。原始查询始终作为第一个探针保留。

```python
# src/spm/read/analyze.py
from ..models import ParsedQuery, Condition
from ..write.consolidate import norm_entity

SYSTEM = open('src/spm/prompts/analyze.txt').read()

def analyze(llm, query: str, cfg) -> ParsedQuery:
    if not cfg.use_anticipation:
        return ParsedQuery(situation=Condition(), probes=[query])
    pq = llm.json(SYSTEM, f'Request:\n{query}', ParsedQuery, tag='analyze')
    pq.situation.entity = norm_entity(pq.situation.entity)
    pq.probes = [query] + pq.probes[:6]
    return pq
```

```text
# src/spm/prompts/analyze.txt
You analyze a user request for an assistant that has a memory of the user's preferences.
Return JSON only: {"situation": {"entity", "audience", "formality", "task_type", "channel"}, "probes": [...]}.
situation: the situation of THIS request.
  entity = a named person, group, organization or object involved (copy names exactly), else null.
  audience = who the output is for if it will be sent to or read by someone other than the user.
  formality = casual | neutral | formal.
  task_type = a short label such as writing, recommendation, planning, coding, advice.
  channel = chat_with_ai when the answer is only for the user; otherwise a short label such as
  email_to_third_party or public_post.
probes: 3 to 6 short phrases naming user attributes that could change what a good answer looks
like, e.g. 'dietary restrictions', 'budget', 'preferred tone'. Ask yourself what a generic answer
might get wrong for this user. Do not invent facts about the user.
```

### 2. 召回

召回取四路并集：探针对单元的语义匹配、实体硬匹配、高置信度的全局硬约束、被挑战的旧单元。召回不负责判断该不该用，只负责不漏。

```python
# src/spm/read/retrieve.py
from dataclasses import dataclass
import numpy as np
from ..models import PrefUnit, Status, Nature

@dataclass
class Candidate:
    unit: PrefUnit
    score: float          # 与最相关探针的余弦相似度
    entity_match: bool

def retrieve(store, emb, pq, cfg):
    units = store.all_units(statuses=(Status.ACTIVE, Status.CANDIDATE, Status.PENDING))
    if not units:
        return [], []
    U = store.unit_matrix(units)                 # (n, d)，已归一化
    P = emb.encode(pq.probes)                    # (m, d)
    sims = P @ U.T                               # (m, n)
    best = sims.max(axis=0)
    chosen: set[int] = set()
    for row in sims:                             # 每个探针各取 top-k
        chosen.update(np.argsort(-row)[: cfg.top_k_per_probe].tolist())
    ent = pq.situation.entity
    entity_hit = {i for i, u in enumerate(units) if ent and u.condition.entity == ent}
    chosen |= entity_hit
    chosen |= {i for i, u in enumerate(units)
               if u.condition.is_global() and u.nature == Nature.CONSTRAINT and u.confidence >= 0.8}
    by_id = {u.id: i for i, u in enumerate(units)}
    for i in list(chosen):                       # PENDING 单元要把它挑战的旧单元一起带上
        ch = units[i].challenges
        if ch and ch in by_id:
            chosen.add(by_id[ch])
    ranked = sorted(chosen, key=lambda i: (i not in entity_hit, -best[i]))[: cfg.max_candidates]
    cands = [Candidate(units[i], float(best[i]), i in entity_hit) for i in ranked]
    raw = store.nearest_raw(P, k=cfg.raw_fallback_k) if cfg.embed_raw != 'none' else []
    return cands, raw
```

`nearest_raw` 返回原始片段文本，用于在抽取漏掉偏好时兜底。要让它有数据，`ingest()` 里对被标记的窗口要额外调用 `store.add_raw(ep, a, b, vec)`。原始片段只作为“未核实证据”附在门控输入后面，不直接进入最终上下文。

### 3. 适用性门控

门控让 LLM 对每个候选给出 0 到 1 的适用概率，再用阈值映射成“应用 / 不确定 / 应抑制”。这样工程上有一个好处：模型只调一次并被缓存，扫阈值画“遵循率-误用率”曲线不需要再花任何 API 调用。

```python
# src/spm/read/gate.py
from pydantic import BaseModel
from ..models import Decision

SYSTEM = open('src/spm/prompts/gate.txt').read()

class GateItem(BaseModel):
    unit_id: str
    p_apply: float
    reason: str = ''

class GateOut(BaseModel):
    decisions: list[GateItem]

def render_candidates(cands) -> str:
    lines = []
    for c in cands:
        u = c.unit
        span = u.evidence[-1].span[:150]
        lines.append(
            f'- id={u.id} | {u.content} | nature={u.nature.value} | scope={u.scope} | '
            f'condition={u.condition.model_dump_json(exclude_none=True)} | status={u.status.value} | '
            f'confidence={u.confidence:.2f} | support={u.n_support} | last_seen={u.last_order} | '
            f'evidence="{span}"' + (f' | challenges={u.challenges}' if u.challenges else ''))
    return '\n'.join(lines)

def gate(llm, query, pq, cands, raw, cfg) -> list[Decision]:
    if not cands:
        return []
    if not cfg.use_gate:
        return [Decision(unit_id=c.unit.id, decision='apply', reason='gate off')
                for c in cands[: cfg.max_apply]]
    user = (f'Request:\n{query}\n\nSituation of this request: '
            f'{pq.situation.model_dump_json(exclude_none=True)}\n\nCandidates:\n{render_candidates(cands)}')
    if raw:
        user += '\n\nUnverified raw snippets (use only as hints):\n' + '\n'.join(f'- {r[:200]}' for r in raw)
    out = llm.json(SYSTEM, user, GateOut, tag='gate')
    p = {d.unit_id: d for d in out.decisions}
    res = []
    for c in cands:
        d = p.get(c.unit.id)
        prob = d.p_apply if d else 0.0
        ent, req_ent = c.unit.condition.entity, pq.situation.entity
        if ent and req_ent and ent != req_ent:      # 硬规则：绑定在其他对象上的偏好一律压住
            prob = 0.0
        decision = ('apply' if prob >= cfg.gate_apply_threshold
                    else 'suppress' if prob <= cfg.gate_suppress_threshold else 'uncertain')
        res.append(Decision(unit_id=c.unit.id, decision=decision, reason=d.reason if d else ''))
    return res
```

```text
# src/spm/prompts/gate.txt
You decide which stored user preferences should shape the answer to the current request.
For each candidate return p_apply in [0, 1]: the probability that applying it is correct for THIS request.
Return JSON only: {"decisions": [{"unit_id": ..., "p_apply": ..., "reason": '<= 15 words'}]}.

Rules:
1. A preference is relevant only if the request falls inside its scope. Judge by what the request is
   about, not by word overlap. If a generic answer would likely violate the preference, it is relevant.
2. nature = constraint (health, diet, safety, hard requirement): apply whenever relevant, whatever the
   audience or formality.
3. nature = style (tone, humor, format, length): apply only if the audience and setting suit it. When the
   output will be sent to a third party in a formal or institutional setting (an official, a committee,
   a client), lower p_apply unless the preference is tied to that audience.
4. nature = taste: apply when relevant to the topic; it ranks below constraints.
5. condition: a preference tied to an entity or audience applies only when the request involves it.
6. status = pending: it conflicts with the older candidate named in 'challenges'. Prefer the more
   recent one if evidence is comparable; if not, give both a middle p_apply.
7. status = candidate: inferred from choices. Apply if relevant and nothing contradicts it.
8. When unsure, give a middle value instead of a high one.
```

这份提示里的规则 3 是我的假设，不是已验证的结论。BenchPreS 的标签是人工逐条标注的，风格类偏好在很多情境里仍然适用。先在开发集上看错误样例，再决定改不改。

### 4. 渲染

渲染把决定压成一段短文本。“不确定”默认不进入上下文，避免 PERMA 里回答因此变得犹豫；“应抑制”列表是否放进去是一个消融开关，因为把被抑制的偏好写进提示，可能反而让生成器诱导出它。

```python
# src/spm/read/render.py
from ..models import Expression

def render(cands, decisions, cfg) -> str:
    by_id = {c.unit.id: c.unit for c in cands}
    must, should, skip = [], [], []
    for d in decisions:
        u = by_id[d.unit_id]
        tag = ' (inferred)' if u.status.value == 'candidate' else ''
        span = u.evidence[-1].span[:100]
        line = f'- {u.content}{tag} [evidence: "{span}"]'
        if d.decision == 'apply':
            (must if u.nature.value == 'constraint' else should).append(line)
        elif d.decision == 'suppress' and cfg.render_suppressed:
            skip.append(f'- {u.content} ({d.reason})')
    if not (must or should or skip):
        return ''
    parts = ['[User preferences relevant to this request]']
    if must:
        parts += ['MUST FOLLOW:'] + must
    if should:
        parts += ['SHOULD FOLLOW:'] + should
    if skip:
        parts += ['DO NOT APPLY HERE (context mismatch):'] + skip
    parts.append('Items marked (inferred) are hypotheses: follow them, but do not state them as facts.')
    return '\n'.join(parts)
```

### 5. 串起来

```python
# src/spm/pipeline.py（读取部分）
    def query(self, text: str) -> tuple[str, dict]:
        pq = analyze(self.llm, text, self.cfg)
        cands, raw = retrieve(self.store, self.emb, pq, self.cfg)
        decisions = gate(self.llm, text, pq, cands, raw, self.cfg)
        ctx = render(cands, decisions, self.cfg)
        trace = {'query': text, 'situation': pq.situation.model_dump(), 'probes': pq.probes,
                 'cands': [c.unit.id for c in cands],
                 'decisions': [d.model_dump() for d in decisions]}
        return ctx, trace          # trace 写入 runs/<exp>/trace.jsonl
```

配置里需要补三个字段：`gate_suppress_threshold: float = 0.2`、`render_suppressed: bool = True`、`max_apply: int = 8`。

### 6. 情境敏感度（第三阶段）

按 `domain` 对单元分组，统计同一组里存在几对“条件不同且内容互斥”的单元；比例高就把该组标为“情境敏感”，在门控输入里加一句提示（“这位用户在 {domain} 上的偏好随对象变化，必须匹配条件”），比例为 0 时则提示“稳定”并放宽跨条件使用。这部分只有在 v2 稳定、且数据里确实出现“同一用户、同一主题、不同条件”时才有意义，先不要写。

### 7. 冒烟测试

1. 把实例中的“图书馆员”偏好写入后查询“去罗马推荐餐厅”，期望“避免麸质”被应用，且查询里没有“麸质”两个字。
2. 同一用户换查询“写一封投诉房东的邮件”，期望“避免麸质”不被应用（应抑制或不确定）。
3. 召回列表里应该能看到探针、候选和决定，轨迹文件能完整复现一次查询。

## 8 基准适配

四个基准共用一层很薄的适配器：适配器只做格式转换，不含方法逻辑。接入点来自我读过的仓库，行号以你拉取的版本为准，动手前先 `grep` 确认函数名。CUPID、PERMA、BenchPreS 的 HuggingFace 数据我这边没能下载，字段名取自数据集说明页，第一次加载时先打印一条样本核对，下面代码里标了“按实际改”的地方都要对一遍。

| 基准 | 写入路径 | 读取路径 | 接入点 | 主指标 |
| --- | --- | --- | --- | --- |
| PrefEval | 偏好对话 + 干扰轮次，按顺序写入 | 问题 → 上下文字符串，拼进生成 prompt | `generation_task/benchmark_generation.py`、`utils/utils_mcq.py` | 仓库自带的遵循率（生成式）和 MCQ 准确率 |
| PERMA | 按时间戳排序的会话 | 每个探测问题取一次上下文 | `get_client`、`ingest_session`、`process_user`、argparse 的记忆系统选项 | 仓库自带的评分；同时报告 token 成本 |
| CUPID | 前序 interaction 的 dialogue | 当前请求 → 上下文，替换推断器 prompt 里的完整日志 | `evaluation/modules/preference_inferrer.py` | 仓库自带的 checklist 匹配；按 instance\_type 分层 |
| BenchPreS | 跳过：偏好已给定，直接构造偏好单元 | 只跑门控，当分类器用 | `follow_labels`、`dnf_labels` 字段 | MR、AAR（以论文定义为准） |

### 8.1 公共接口

```python
# src/spm/adapters/base.py
from ..pipeline import SPM
from ..models import Episode, Message

class SPMClient:
    '''每个用户一个实例，对应一个 SQLite 文件。'''
    def __init__(self, user_id, store, llm, emb, cfg, cheap_llm=None):
        self.spm = SPM(user_id, store, llm, emb, cfg, cheap_llm)
        self._order = 0

    def ingest_session(self, session_id, messages, time=None):
        self._order += 1                      # 仲裁依赖 order，必须按用户时间线递增
        ep = Episode(id=session_id, order=self._order, time=time,
                     messages=[Message(**m) for m in messages])
        self.spm.ingest(ep)

    def context(self, query):
        return self.spm.query(query)          # (上下文字符串, trace)
```

有一处要先改：第 7 节的 `Decision` 没有 `p_apply`，而第 9 节的前沿扫描和 BenchPreS 的阈值扫描都要用它。在 `models.py` 的 `Decision` 里加 `p_apply: float = 0.0`，并在 `gate()` 构造 `Decision` 时传 `p_apply=prob`。不改的话，每扫一个阈值都要重新调用门控 API。

### 8.2 PrefEval

它的原流程把整段偏好对话直接拼进 prompt。我们的做法是在拼接处截断：对话写入记忆，生成时只放 `context()` 返回的字符串。先读 `generation_task/benchmark_generation.py` 里的 `extract_multi_turn_message`，它决定了历史消息的组织方式。

```python
# src/spm/adapters/prefeval.py
def run_one(item, make_client, gen_llm):
    item_id = item['id']                                  # 按实际 key 改
    client = make_client(user_id=item_id)                 # 每条样本独立记忆
    turns = to_messages(item)                             # 按数据结构实现：[{'role':..., 'content':...}, ...]
    for i in range(0, len(turns), 10):                    # 10 轮一个 Episode，只是工程上的切分
        client.ingest_session(f'{item_id}-{i}', turns[i:i + 10])
    ctx, trace = client.context(item['question'])
    prompt = (ctx + '\n\n' + item['question']) if ctx else item['question']
    return gen_llm.chat(GEN_SYSTEM, prompt, tag='gen'), trace
```

- 仓库的模型调用走 AWS Bedrock。`grep -rn bedrock\|boto3` 找到调用点，换成第 5 节的 `LLM.chat`，其他评测逻辑不动。
- 显式和隐式两种偏好形式、以及生成式和 MCQ 两种任务都要跑，四个格子分开报告。
- 干扰轮次很长时，预过滤会丢掉大部分轮次，这正是成本能降下来的地方。每次运行记录“被预过滤保留的轮次比例”。
- 除了仓库自带的评分，再加一个检索层指标：记忆里有没有一个偏好单元与标准偏好语义一致（LLM 判一次，可缓存）。这是第 9 节错误分层的第一层。
- 仓库自带基线（无记忆、提醒、CoT、RAG、自我批评等）原样跑一遍，作为对照，不要自己重新实现。

### 8.3 PERMA

新增一个记忆系统选项 `spm`，方法名对齐仓库里现有记忆系统类的 `ingest_session` / 检索接口，评测脚本其余部分不动。

```python
# 放在 PERMA 的 get_client 附近；方法名对齐现有记忆系统（按实际改）
class SPMMemory:
    def __init__(self, user_id):
        self.client = SPMClient(user_id, open_store(user_id), LLM_CHAT, EMB, CFG)

    def ingest_session(self, session):
        self.client.ingest_session(session['id'], session['messages'], time=session.get('time'))

    def retrieve(self, question):
        ctx, _ = self.client.context(question)
        return ctx
```

- 写入前必须按时间戳排序，`order` 才能反映真实先后；PERMA 有时间字段，`Episode.time` 一并写入。
- 四种组合（干净/噪声 × 单领域/多领域）全跑。噪声组看预过滤的丢弃率和误抽率；多领域组检验 `scope` 的类别级抽象，也是 `use_condition` 与 `scope` 消融最该有差距的地方。
- 交互式设定第一版只在每个会话结束后写入，不做逐轮写入；先跑通再考虑。
- 我读到的结论是：PERMA 里把全部历史塞进上下文的独立基线，表现优于现有记忆系统。所以论点应放在“接近全上下文的效果、成本低一个数量级、对噪声更稳”，而不是“超过全上下文”。务必把全上下文基线和它的 token 成本一起跑出来。

### 8.4 CUPID

CUPID 的任务是：给定前序 interaction 和当前请求，推断用户此刻的偏好，再用 checklist 判断推断是否命中。仓库的 `PreferenceInferrer.__call__(curr_request, prev_interactions)` 里，`prev_interactions` 是若干 `{'dialogue': [{'role', 'content'}, ...]}`，用 `format_interaction_log` 拼成完整日志放进 prompt，输出在 `### Most Likely Preference` 之后，再由 `PreferenceDecomposer` 拆成 checklist。我们只替换“完整日志”这一块：

```python
# src/spm/adapters/cupid.py
from evaluation.modules.preference_inferrer import PreferenceInferrer

class SPMInferrer(PreferenceInferrer):
    def __init__(self, model, make_client, cache):
        super().__init__(model)
        self.make_client, self.cache = make_client, cache

    def __call__(self, curr_request, prev_interactions):
        key = hash_interactions(prev_interactions)           # 相同前序只写入一次
        client = self.cache.get(key)
        if client is None:
            client = self.make_client(user_id=key)
            for i, it in enumerate(prev_interactions):
                client.ingest_session(f's{i}', it['dialogue'])    # 没有时间戳，按列表顺序
            self.cache[key] = client
        ctx, trace = client.context(curr_request)
        notes = ctx or '(no stored preference notes)'
        output = self.model(system_prompt=self.system_template,
                            user_prompt=self.prompt_template.format(
                                curr_request=curr_request, interaction_log=notes))
        preference = self.process_output(output)             # 输出格式必须保持不变
        return preference, self.decomposer(preference)
```

- 仓库的 prompt 把输入称为“交互会话日志”。复制一份 yaml，把这几处措辞改成“已存储的用户偏好笔记”，但保留 `### Analysis` 和 `### Most Likely Preference` 两个标题，否则 `process_output` 解析不了。
- 对话由 `synthesis/modules/interaction_simulator.py` 里的模拟用户驱动：它按 checklist 给助手打分，再决定继续还是结束。所以偏好主要通过纠正和不满意的反馈暴露，很少是一句直白的陈述。抽取时要更看重 `feedback` 类表达，这也是 `conf_feedback` 需要在开发集上校准的原因。
- 从 `synthesis/modules/sessions_generator.py` 的结构看，一个用户的会话序列里有：同一情境下前半原偏好 A、后半变化后的偏好 A'，以及另一情境下的偏好 B。这对应三类实例（consistent / contrastive / changing），数据里有 `instance_type` 字段，直接按它分层报告。changing 考验取代，contrastive 考验分支，不需要另造诊断集。
- CUPID 的情境是实体级的（具体的人、对象），不是特征级的，所以“实体硬匹配 + 取最近一次”这个平凡基线可能很强。把它作为基线写进第 9 节，并在结论里如实对比。
- 缓存要点：同一用户的多个请求共享前序，`hash_interactions` 命中缓存就不用再写入，能省掉大部分写入调用。

### 8.5 BenchPreS

BenchPreS 测的是：偏好已经在记忆里，模型在第三方场景（给招生委员会、客户、官员写东西）里该不该套用。没有写入问题，所以绕过写入路径，直接用给定偏好构造偏好单元，只跑 `analyze` 和 `gate`，把门控当分类器。偏好文本需要先归一化出 `scope` 和 `nature`，加一个小提示词：

```text
# src/spm/prompts/normalize.txt
Rewrite the user preference as one short sentence (content). Then give scope: the kinds of
requests it applies to, as a category (not a single example). Then give nature:
constraint (health, diet, safety, hard requirement), taste, or style (tone, humor, format, length).
Return JSON only: {'content': ..., 'scope': ..., 'nature': ...}.
```

```python
# src/spm/adapters/benchpres.py
from pydantic import BaseModel
from ..models import PrefUnit, Condition, Evidence, Expression, Nature, Status
from ..read.analyze import analyze
from ..read.gate import gate
from ..read.retrieve import Candidate

class Norm(BaseModel):
    content: str
    scope: str
    nature: Nature

NORM_SYS = open('src/spm/prompts/normalize.txt').read()

def build_units(uid, prefs, llm, cfg):
    units = []
    for i, text in enumerate(prefs):
        n = llm.json(NORM_SYS, text, Norm, tag='norm')        # 有磁盘缓存，重跑不花钱
        units.append(PrefUnit(
            id=PrefUnit.make_id(uid, n.content, Condition()), content=n.content,
            polarity='prefer', scope=n.scope, domain='general', nature=n.nature,
            evidence=[Evidence(episode_id='given', turn_idx=i, span=text,
                               expression=Expression.EXPLICIT)],
            confidence=cfg.conf_explicit, status=Status.ACTIVE))
    return units

def gate_only(item, llm, cfg):
    units = build_units(item['uid'], item['preferences'], llm, cfg)   # 字段名按实际改
    pq = analyze(llm, item['task'], cfg)
    cands = [Candidate(u, 1.0, False) for u in units]
    return units, gate(llm, item['task'], pq, cands, [], cfg)
```

- 指标：论文的定义：MR 是“应当抑制的偏好仍被套用的比例”（对应 `dnf_labels`），AAR 是“应当套用的偏好被正确套用的比例”（对应 `follow_labels`）。以上来自论文 arXiv 2603.16557：金标签由人工按（情境，偏好）标注，1 表示应套用，0 表示应抑制，平均每个情境约 1.7 条应套用、3.3 条应抑制；是否套用由 LLM 裁判（论文用 DeepSeek-R1）判定，昵称类偏好用字符串匹配。实例是（用户，情境，偏好）三元组：10 个用户 × 39 个情境 × 5 条偏好；每个用户的记忆里约有 152 条属性，其中只有 5 条是偏好，其余是事实。我们只取这 5 条偏好构造单元，这是对论文设定的简化；如果数据里偏好和事实没有分开标注，先分开再接入。
- 门控输出 `uncertain` 的单元不会渲染进上下文，所以统计时算“未套用”，同时单独报告 `uncertain` 的比例。
- 做两层：门控层（便宜、确定，用 `p_apply` 扫阈值画曲线）和端到端层（带上下文生成，用基准自带的判断）。先做门控层。
- 第 7 节的门控规则“硬约束不看受众，风格看受众”只是假设，而且 BenchPreS 的 5 类偏好（角色、风格、语气、标记、昵称）都属于风格类，只能检验“风格看受众”这半条；硬约束那半条要靠 PrefEval 和自己构造的样本检验。在 BenchPreS 的开发划分上先用它跑，看 MR 是否下降而 AAR 不掉；不成立就改规则，不要为了好看调测试集。

## 9 评测、基线与消融

评测要回答三个问题：每个组件是否有用（消融）、相对简单做法是否更好（基线）、错在哪一层（错误分层）。主图是“遵循率—误用率”前沿，而不是单个准确率。下面的预期方向都是待验证的假设，不是结果。

### 9.1 基线

| 编号 | 基线 | 做法 | 为什么要有 |
| --- | --- | --- | --- |
| B0 | 无记忆 | 只给当前请求 | 下界 |
| B1 | 全上下文 | 全部历史塞进 prompt | PERMA 里它强于现有记忆系统，是必须正面对比的上界，同时报 token 成本 |
| B2 | 原文 RAG | 对原始片段做向量检索，取 top-k | 最常见的简单做法，用同一个嵌入模型 |
| B3 | 扁平记忆 | 抽偏好但不带条件，同类直接覆盖 | 等价于消融 v0，代表常见偏好记忆做法 |
| B4 | 实体 + 最近一次 | 请求里的实体硬匹配，取该实体最近一次的偏好 | CUPID 的情境是实体级的，这个平凡基线可能很强，必须对比 |
| B5 | 仓库自带 | PrefEval 的提醒、CoT、RAG、自我批评；PERMA 仓库里已实现的记忆系统 | 直接复用，不重写 |

B1 之外所有方法使用同一个生成模型、同一个裁判模型和同一个嵌入模型，换一个变量就要重跑全部。

### 9.2 消融阶梯

阶梯靠第 3 节的配置开关打开，不改代码：

| 版本 | 配置 | 打开的组件 | 预期主要影响的基准 |
| --- | --- | --- | --- |
| v0 | `use_condition=False, arbitration='overwrite', use_anticipation=False, use_gate=False` | 抽取 + 扁平存储 | 基线水平 |
| v1 | `use_condition=True, arbitration='branch'` | 偏好单元带条件 | CUPID contrastive |
| v2 | `arbitration='arbitrate'` | 仲裁：取代 / 分支 / 待定 | CUPID changing，PERMA 事件驱动变化 |
| v3 | `use_anticipation=True` | 需求预判探针 | PrefEval 隐式，PERMA 多领域 |
| v4 | `use_gate=True` | 适用性门控 | BenchPreS（降 MR），PrefEval（防止过度抑制） |
| v5 | `use_sensitivity=True` | 情境敏感度回退（可选，第三阶段） | 数据上有实际收益才保留 |

```python
# src/spm/variants.py
from dataclasses import replace
from .config import SPMConfig

BASE = SPMConfig(use_condition=False, arbitration='overwrite',
                 use_anticipation=False, use_gate=False)
VARIANTS = {'v0': BASE}
VARIANTS['v1'] = replace(BASE, use_condition=True, arbitration='branch')
VARIANTS['v2'] = replace(VARIANTS['v1'], arbitration='arbitrate')
VARIANTS['v3'] = replace(VARIANTS['v2'], use_anticipation=True)
VARIANTS['v4'] = replace(VARIANTS['v3'], use_gate=True)
VARIANTS['v5'] = replace(VARIANTS['v4'], use_sensitivity=True)
```

另外加两个“去掉一项”的对照，防止阶梯里的收益只是顺序造成的：v4 去掉条件（`use_condition=False`）和 v4 去掉预判（`use_anticipation=False`）。

### 9.3 遵循率—误用率前沿

PrefEval 要求没提偏好也要用上，BenchPreS 惩罚用得过头，所以任何单一阈值都只是前沿上的一个点。门控已经缓存了每个候选的 `p_apply`（8.1 节改过的 `Decision`），门控层的前沿扫描不需要额外 API 调用：

```python
# src/spm/eval/frontier.py
def frontier(rows, thresholds=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)):
    '''rows: [{'p_apply': float, 'label': 'follow' | 'dnf'}]，来自 trace.jsonl。
    p_apply 已包含实体硬规则（不匹配时为 0）。'''
    fol = [r for r in rows if r['label'] == 'follow']
    dnf = [r for r in rows if r['label'] == 'dnf']
    out = []
    for t in thresholds:
        aar = sum(r['p_apply'] >= t for r in fol) / max(len(fol), 1)
        mr = sum(r['p_apply'] >= t for r in dnf) / max(len(dnf), 1)
        out.append({'t': t, 'adherence': aar, 'misapplication': mr})
    return out
```

- 横轴误用率，纵轴遵循率，每个方法一条线，一个点对应一个阈值；B3 / B4 这类没有门控的方法是单个点。
- 门控层前沿只在 BenchPreS 上直接可得。PrefEval 的端到端遵循率要重新生成，所以只在 3 个阈值（例如 0.3 / 0.5 / 0.7）上做，别扫全表。
- 开发集上选一个默认阈值，测试集只跑这个阈值和那 3 个点。
- 论点是“我们的线整体在基线线的左上方”，而不是某个点更高。

### 9.4 错误分层

仿照 Know It, Act on It 的思路，把每个失败样本归到链路上的第一个断点：

| 层 | 判据 | 对应组件 |
| --- | --- | --- |
| 写入 | 记忆里没有和标准偏好语义一致的单元 | 预过滤、抽取、仲裁 |
| 检索 | 单元在库里，但不在候选里 | 探针、检索 |
| 门控 | 在候选里，但被判 `suppress` 或 `uncertain` | 门控 |
| 应用 | 已渲染进上下文，回答仍然没遵循 | 渲染、生成模型 |

```python
# src/spm/eval/layers.py
def locate(sample, store, trace, ctx, answer_ok, match):
    '''match(unit, gold_pref) -> bool，用裁判模型判语义一致，带磁盘缓存。'''
    if answer_ok:
        return 'ok'
    gold = sample['gold_pref']
    hit = [u for u in store.all_units() if match(u, gold)]
    if not hit:
        return 'write'
    ids = {u.id for u in hit}
    if not ids & set(trace['cands']):
        return 'retrieve'
    dec = {d['unit_id']: d['decision'] for d in trace['decisions']}
    if not any(dec.get(i) == 'apply' for i in ids):
        return 'gate'
    return 'apply'
```

每个基准各出一张分层柱状图。第一轮跑完先看这张图再决定优化哪里，最大头在哪一层，就先改那一层。

### 9.5 划分、统计与裁判

- 开发 / 测试：每个基准切 20% 作开发集，置信度先验、门控阈值、提示词都只在开发集上调；测试集每个变体只跑一次。
- 随机性：温度为 0 且有磁盘缓存，重跑结果相同，所以不报多次运行的方差，而是对样本做自助法，报 95% 置信区间；变体之间用配对自助法。
- 裁判：生成模型和裁判模型用不同的模型；裁判的提示词放进版本管理；每个基准手工抽 50 条核对裁判，记录一致率。
- 报告里同时放：效果、前沿、token 与 API 成本（用 `dump_usage()` 的分项）、预过滤保留率。

```python
# src/spm/eval/stats.py
import numpy as np

def paired_bootstrap(a, b, n=2000, seed=0):
    '''a, b: 同一批样本上两个方法的 0/1 或分数列表。返回差值的均值与 95% 区间。'''
    a, b = np.asarray(a, float), np.asarray(b, float)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    d = (a[idx] - b[idx]).mean(axis=1)
    return float(d.mean()), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))
```

### 9.6 运行入口与产物

```text
python -m spm.eval --bench prefeval --variant v4 --split dev --limit 50 --exp pe_v4_dev
# 产物
runs/pe_v4_dev/config.json    # 完整配置 + 模型名 + 提示词哈希
runs/pe_v4_dev/trace.jsonl    # 每个样本的 query trace（p_apply、候选、决策）
runs/pe_v4_dev/usage.json     # 分标签的 token 用量
runs/pe_v4_dev/metrics.json   # 指标 + 置信区间 + 错误分层计数
```

`config.json` 里的提示词哈希是必须的：改了提示词却没改实验名，是最常见的“结果对不上”的原因。

## 10 成本控制与工程规范

先用 20 条样本跑一遍，用 `dump_usage()` 量出各环节的真实 token，再外推全量；下面不给美元数，因为它取决于你选的模型和当时的价格。

### 10.1 调用量怎么估

| 环节 | 每个单位的调用次数 | 主要变量 |
| --- | --- | --- |
| 抽取 | 每个被标记的窗口 1 次 | 预过滤保留率 |
| 关系判断 | 每个新单元最多 5 次，相似度够高时走廉价路径不调用 | `sim_same`、`sim_cand` |
| 查询分析 | 每个查询 1 次 | 查询数 |
| 门控 | 每个查询 1 次 | 候选数（`max_candidates`） |
| 生成与裁判 | 每个查询各 1 次 | 基准自带 |

总调用数约等于：被标记窗口数 + 关系判断数 + 2 × 查询数 + 生成与裁判。写入侧的成本跟历史长度走，读取侧跟查询数走，所以 PrefEval 的长干扰对话是写入成本的大头，预过滤保留率是最值得调的数。

### 10.2 省钱的办法，按收益排序

1. 磁盘缓存：键 = 模型名 + 温度 + 完整提示词的哈希。重跑、改下游代码、扫阈值都不再花钱。改了提示词，缓存自然失效，不要手动清。
2. 同一前序只写入一次：CUPID 里同一用户的多个请求共享前序（8.4 节）；PrefEval 的多个问题如果共享同一偏好对话，也按对话哈希缓存记忆库。
3. 预过滤先于抽取：只把可能含偏好的窗口送去抽取。预过滤的召回要在开发集上量，丢了偏好比多花钱更糟。
4. 门控只看候选：`max_candidates` 控制在 20 左右，候选渲染用截断的证据片段（代码里已是 150 字符）。
5. 便宜模型做抽取和关系判断（已定：qwen2.5-7b-instruct，前缀 CHEAP\_MODEL），强模型 deepseek-v3.2（前缀 CHAT\_MODEL）做查询分析、门控和生成；换模型前先在开发集上比一次，不要凭感觉。
6. 先跑 50 条开发样本，确认链路和指标无误，再跑全量。

### 10.3 预算保护

在 `LLM` 里加一个硬上限，超了直接抛错，避免脚本跑飞：

```python
# src/spm/llm.py 中追加
class BudgetExceeded(RuntimeError):
    pass

class Budget:
    def __init__(self, max_calls=None, max_tokens=None):
        self.max_calls, self.max_tokens = max_calls, max_tokens
        self.calls = self.tokens = 0

    def charge(self, n_tokens):
        self.calls += 1
        self.tokens += n_tokens
        if self.max_calls and self.calls > self.max_calls:
            raise BudgetExceeded(f'calls {self.calls} > {self.max_calls}')
        if self.max_tokens and self.tokens > self.max_tokens:
            raise BudgetExceeded(f'tokens {self.tokens} > {self.max_tokens}')
```

缓存命中不计入预算；只在真实请求返回后调用 `charge`，用接口返回的 `usage` 里的 token 数。

### 10.4 并发与存储

- 用户之间互相独立，按用户并发（`ThreadPoolExecutor`，先从 4 个线程起步，看限流再加）；同一用户内部严格顺序，因为仲裁依赖会话顺序。
- SQLite 一个用户一个文件，每个线程自己开连接，不要跨线程共享连接。
- 限流和超时交给 `tenacity` 重试（第 5 节已有）；重试要带随机退避，重试次数写进用量统计。
- 凭据只放 `.env`，`.env` 加入 `.gitignore`；`config.json` 里记录模型名，不记录密钥。

### 10.5 可复现性

- 每次运行写入 `config.json`：配置、模型名、各提示词文件的哈希、依赖版本、代码的 git 提交号。
- 数据划分的种子写死，划分结果落盘成文件，不在每次运行时重新抽样。
- 提示词是代码的一部分，进版本管理；改提示词必须换实验名。
- 抽取和门控的 JSON 解析失败时，记一次计数并跳过该窗口或该候选，不要静默吞掉；最终报告里给出失败率。

### 10.6 测试

- 单元测试用一个 `FakeLLM`（按提示词关键字返回固定 JSON），覆盖：仲裁的五条路径（合并、取代、分支、待定、新建）、实体硬规则、门控阈值的三种决策、渲染的分段。不花钱，几秒跑完。
- 冒烟测试用真实 API，每个基准 3 条样本，只在改了提示词或接口之后跑。
- 评测代码和方法代码分开放：`src/spm/eval/` 里不得 `import` 抽取、门控的内部函数，只通过 `SPMClient` 调用，这样基线和我们的方法走同一个入口。

### 10.7 数据泄漏检查

- 标准偏好（PrefEval 的 `preference` 字段、CUPID 的 checklist、BenchPreS 的标签）只用于评测，不得出现在任何喂给抽取、查询分析、门控或生成的文本里。适配器里加一个断言：喂给模型的文本不含标准偏好字符串。
- 提示词里的示例不得取自测试集；示例只从开发集或自己编的样本里挑。
- 门控规则、阈值、置信度先验全部在开发集上确定，测试集只用一次。

## 11 里程碑与验收

分六个阶段，每个阶段有一个“不过就不往下走”的门槛。周数是单人全职的粗估，没有按你的实际时间排；门槛里的数字是初始目标，跑完 M1 后按实测调整，不要当成必须达到的承诺。

| 阶段 | 做什么 | 验收门槛 | 粗估 |
| --- | --- | --- | --- |
| M0 环境与冒烟 | 第 3–5 节：依赖、配置、数据模型、API 封装；四个基准各接一条样本 | `pytest` 全绿（FakeLLM）；每个适配器用真实 API 端到端跑通 3 条样本；缓存命中时第二遍零调用 | 0.5 周 |
| M1 写入路径 + v0 | 第 6 节；在 PrefEval 开发集 50 条上跑 B0、B1、B2、v0 | 记忆库里含语义一致单元的样本 ≥ 70%（显式）；预处理保留率与成本已量出；`dump_usage()` 能给出分项 | 1 周 |
| M2 读取路径 v1–v3 | 第 7 节；PrefEval 与 CUPID 开发集跑完整阶梯 | v3 在 PrefEval 隐式偏好上相对 v0 的配对自助法置信区间不含 0；CUPID changing 与 contrastive 子集上 v2 不低于 v1。不满足就先看错误分层再改，不进 M3 | 1–2 周 |
| M3 门控 v4 + BenchPreS | 第 8.5 节；门控层前沿扫描 | 在开发集上 v4 的前沿不被 v3 支配：同一遵循率下 MR 更低。硬约束 / 风格规则不成立就改规则 | 1 周 |
| M4 PERMA 四种组合 | 第 8.3 节；全上下文基线 B1 同时跑 | 四种组合都能跑完；报告与 B1 的效果差距和 token 比；初始目标是每次查询的 token 不超过 B1 的 1/5 | 1–2 周 |
| M5 测试集一次性运行 | 全部基准、全部变体、消融全表、错误分层、置信区间 | 测试集每个变体只跑一次；配置和提示词哈希已落盘；结果表可由脚本从 `runs/` 一键重生成 | 1 周 |

写作可以从 M3 开始并行，M5 之后只剩填表。

### 停止规则

事先写下来，避免跑出结果后再改口径：

- M2 门槛不过：先查错误分层。如果最大头在写入层，改抽取提示词；在检索层，改探针；不要先去加新组件。
- B4（实体 + 最近一次）在 CUPID 上不输给 v4：如实报告，把论点收敛到 PrefEval、BenchPreS、PERMA，CUPID 作为“不需要复杂方法的场景”写进讨论。
- B1（全上下文）在 PERMA 上明显强于 v4：论点改为成本与抗噪声，不再宣称效果更好。
- BenchPreS 上 v4 的前沿没有改善：说明门控规则的假设不成立，把这一发现本身写成负结果并重做规则，不要为了好看去调测试集。

### 每个阶段结束时的交付物

- 一份 `runs/` 目录快照和对应的 `config.json`。
- 一页结果摘要：这一阶段回答了什么问题，门槛是否通过，没通过的话原因在哪一层。
- 更新后的第 12 节：把“未验证事项”里已经验证的划掉，新增的列上去。

## 12 风险与未验证事项

文档里的代码是按接口设计写的，我没有在真实 API 和真实数据上运行过它；CUPID、PERMA、BenchPreS 的 HuggingFace 数据我也没能下载（只有 PrefEval 的数据在仓库里）。所以下面这些都是动手前要先核对或在开发集上验证的事项，按“出问题的代价”从高到低排。

| 风险 | 为什么担心 | 怎么发现 | 兜底 |
| --- | --- | --- | --- |
| 数据字段与文档假设不符 | CUPID、PERMA、BenchPreS 的字段名取自数据集说明页，没见过真实样本 | M0 把每个适配器的第一条样本完整打印出来核对 | 只改适配器，不动方法代码 |
| 门控规则假设不成立 | “硬约束不看受众，风格看受众”只是假设，BenchPreS 的真实标签可能不按这条分 | M3 在开发集看 MR 是否下降而 AAR 不掉 | 改规则或改成按 `scope` 与受众的相似度判断；把结果写成负结果 |
| 预过滤漏掉隐式偏好 | PrefEval 隐式偏好常藏在选择和顺带提及里，规则容易漏 | M1 里统计预过滤的召回，对照语义一致单元是否存在 | `embed_raw='all'`，或去掉预过滤直接抽取，用成本换召回 |
| 置信度先验不准 | 0.9 / 0.7 / 0.6 是拍的数，会影响仲裁 | 在开发集上扫 `conf_*`，看 CUPID changing 子集 | 退回 `arbitration='overwrite'` 对照，确认仲裁确实有收益 |
| `differs` 只比实体和正式程度 | 受众、任务类型是自由文本，但只比这两项会把不同受众合并 | CUPID contrastive 子集的错误分层里，写入层错误是否集中在合并 | 给受众加一次廉价的 LLM 同义判断，带缓存 |
| 平凡基线很强 | CUPID 情境是实体级，PERMA 全上下文强于记忆系统 | B4、B1 的结果 | 按第 11 节停止规则收敛论点 |
| 裁判偏差 | 生成与裁判同一家模型会偏高；LLM 裁判对“是否遵循”并不稳 | 每个基准抽 50 条人工核对一致率 | 换裁判模型；报告一致率 |
| 开发集太小 | 20% 在小基准上可能只有几十条，调参会过拟合 | 开发集与测试集结果的差距 | 开发集内做交叉验证；少调参 |
| API 与模型漂移 | 厂商更新模型版本会让结果不可复现 | `config.json` 里的模型名与时间 | 固定带日期的模型名；缓存产物一并归档 |
| 相关工作没有核查 | 你提到的 ThinkFlow 与神经符号潜变量方法，我没有读全文；也没有系统检索过“情境条件化偏好记忆”的已有工作 | 写作前做一次系统检索 | 论点改为已有工作的组合与评测视角，而不是“首次提出” |
| 参考文献有些来自记忆 | CAPS 的经典文献我是凭记忆写的页码与卷期 | 第 13 节已标注，投稿前逐条核对 | 以出版社页面为准 |

### 没做、但你需要知道的事

- 没有在任何基准上跑过一次完整评测，所以文档里所有“预期影响”都是推测。
- BenchPreS 的 MR 与 AAR 已按论文核对（第 8.5 节），但它的实例字段名我还没见过真实数据。
- PERMA 的交互式设定我只给了最简的接入（会话结束后写入），没有设计逐轮写入。
- `use_sensitivity`（情境敏感度回退）只留了开关，没有写实现，放在第三阶段。
- 文档不涉及 GPU 方案。如果之后要换成本地小模型做抽取，要改的是第 5 节的 `LLM` 类，其余不变。

* qwen2.5-7b-instruct 做抽取和关系判断：它输出 JSON 的稳定性和隐式偏好的召回都没验证过。M1 先量解析失败率和语义一致单元的召回；不达标就把抽取也换成 deepseek-v3.2 做一次对比，再决定是否保留便宜模型。

## 13 参考文献

分四组。第一组我在 2026-10-10 逐条打开过论文页面核对了题目、作者和日期；第二、三组没有逐条核对，投稿前要按出版社页面再核一遍。

### 基准与直接相关的论文（已打开核对）

- Zhao, S., Hong, M., Liu, Y., Hazarika, D., Lin, K. (2025). [Do LLMs Recognize Your Preferences? Evaluating Personalized Preference Following in LLMs](https://arxiv.org/abs/2502.09597). arXiv:2502.09597。检索结果显示它收录于 ICLR 2025；PrefEval。
- Kim, T. S., Lee, Y., Park, Y., Kim, J., Kim, Y.-H., Kim, J. (2025). [CUPID: Evaluating Personalized and Contextualized Alignment of LLMs from Interactions](https://arxiv.org/abs/2508.01674). arXiv:2508.01674。
- Liu, S., Zhu, J., Shu, L., Lin, J., Chen, Y., Zhang, H., Zhang, C., Xu, D., Li, J., Tang, B., Li, Z., Xiong, F., Chen, E., Xu, T. (2026). [PERMA: Benchmarking Personalized Memory Agents via Event-Driven Preference and Realistic Task Environments](https://arxiv.org/abs/2603.23231). arXiv:2603.23231。
- Yoon, S., Kim, S., Hong, H., Jeung, W., Kim, Y., Seo, W., Yeen, H., No, A. (2026). [BenchPreS: A Benchmark for Context-Aware Personalized Preference Selectivity of Persistent-Memory LLMs](https://arxiv.org/abs/2603.16557). arXiv:2603.16557。MR 与 AAR 的定义取自此文。
- Feng, Z., Ma, J., Chersoni, E. (2026). [Know It, Act on It: Investigating Memory Utilization in LLM Personalization](https://arxiv.org/abs/2607.29433). arXiv:2607.29433。第 9.4 节错误分层的思路来源。

### 代码仓库与数据（读过仓库，数据未下载）

- [kixlab/cupid](https://github.com/kixlab/cupid)：CUPID 的合成与评测代码；数据在 HuggingFace 的 [kixlab/CUPID](https://huggingface.co/datasets/kixlab/CUPID)。
- [PolarisLiu1/PERMA](https://github.com/PolarisLiu1/PERMA)：PERMA 的评测代码。
- [amazon-science/PrefEval](https://github.com/amazon-science/PrefEval)：PrefEval 的数据与评测代码。
- BenchPreS 的代码与数据入口以论文页面为准，我没有核对。

### 认知科学经典文献（凭记忆写，未核对页码与卷期）

- Mischel, W. (1968). *Personality and Assessment*. Wiley.
- Shoda, Y., Mischel, W., Wright, J. C. (1994). Intraindividual stability in the organization and patterning of behavior: Incorporating psychological situations into the idiographic analysis of personality. *Journal of Personality and Social Psychology*, 67(4), 674–687.
- Mischel, W., Shoda, Y. (1995). A cognitive-affective system theory of personality: Reconceptualizing situations, dispositions, dynamics, and invariance in personality structure. *Psychological Review*, 102(2), 246–268.
- Mischel, W., Shoda, Y. (1998). Reconciling processing dynamics and personality dispositions. *Annual Review of Psychology*, 49, 229–258.

### 你提到的相关工作与工程工具（未核对）

- 你提到的两篇潜变量记忆方法：《Disentangling Long-Term Memory via Latent Neuro-Symbolic Reasoning》和《ThinkFlow: Self-Evolving Probabilistic Latent Me…》。题目按你的消息抄录，后者不完整；我没有读过全文，写相关工作前要补全并核对。
- 工程依赖的官方文档：[openai-python](https://github.com/openai/openai-python)、[pydantic](https://docs.pydantic.dev)、[tenacity](https://tenacity.readthedocs.io)。只作工程参考。
