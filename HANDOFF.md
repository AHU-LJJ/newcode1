# CAPS / SPM 对话交接

更新：2026-10-11。面向下一个接手的编码助手；本文件记录已核验事实、运行入口与未完成事项。**先读本文件，再读最新用户指示，不必重读 M0–M8 全部报告。**

## 1. 两分钟了解现状

| 项目 | 当前事实 |
| --- | --- |
| 工作目录 | `/home/alhandsome/projects/CAPS`；从此目录执行命令 |
| Git | 已初始化，分支 `main` 跟踪 `origin/main`；仓库 [AHU-LJJ/newcode1](https://github.com/AHU-LJJ/newcode1) |
| 已验证远端基线 | `0616c503d35ff912cab11f5dffcb514126805821`，`first commit`；后续提交以 `git log -3 --oneline` 为准 |
| 本轮文档推送 | 已本地提交；重试遇凭据助手超时和 TLS 中断，普通 push 50 秒超时。最后一次 `ls-remote` 仍为上述基线，**交接文档尚未推送，接手先检查并补推** |
| 当前系统 | `spm/` 情境化偏好记忆；早期 `graphmem/` 保留，暂不继续推进 |
| 默认方法 | `Qwen/Qwen2.5-7B-Instruct + v6_compact`，不要把实验变体自动提升为默认 |
| API | 硅基流动；BGE-M3 嵌入、DeepSeek-V3.2 裁判；密钥在本地 `.env` |
| 数据选择 | **PrefEval 退出后续实验**；默认 `all` 为 PERMA / CUPID / BenchPreS |
| 最近完成 | M8 CUPID 五方法实验，以及原论文 preference alignment score 补测 |
| 尚未执行 | **M9 仅实现和本地测试，未预留题组、未调用 API、没有效果结果**；没有 RL 训练 |
| 测试 | 最近对仅含 Git 待提交文件的干净导出验证：**132 passed, 2 skipped**，两项缺本地基准文件而跳过 |
| 最新新增文件 | `scripts/SPM v2：情境化记忆影响控制——下一阶段项目指导文档.md`，接手时发现未跟踪；已阅读摘要，本轮未实施或提交 |

本轮用户只要求写 handoff，没有要求启动新实验。下一步先结合新 v2 指导文档及新对话用户指示判断研究方向；**不要因为旧 M9 脚本存在就直接启动它**。新文档提出停止图管理线、转向通道化记忆影响控制；它把 M9 写进既往结论，但实际 M9 尚未运行，应纠正这个事实差异。其文献主张和建议阈值尚未在本轮逐项核验。

## 2. 环境与配置

系统为 WSL2 Linux，Python **3.12.3**。用户设备是 RTX 5060 Ti **8 GB**，算力有限，当前实验主要用 API。上轮已验证 PyTorch 的 CPU/CUDA 基础运算；当前进程能否访问 GPU 用 `check_torch.py` 检查。

| 软件 | 本地版本 |
| --- | --- |
| torch / torchvision | `2.13.0+cu130` / `0.28.0+cu130` |
| transformers | `4.57.6` |
| httpx / pydantic | `0.28.1` / `2.14.0` |
| numpy / pyarrow | `2.5.3` / `26.0.0` |
| pytest | `9.1.1` |

环境在 `.venv/`；uv 在 `/home/alhandsome/.local/bin/uv`。`pyproject.toml`、`uv.lock`、`.python-version` 已提交，按锁文件恢复。一次性阅读 PDF 安装的 `pypdf==6.20.0` **不在依赖锁文件中**；系统和测试不依赖它。

```bash
cd /home/alhandsome/projects/CAPS
.venv/bin/python -m spm --help
.venv/bin/python -m pytest -q
# 新克隆需要恢复环境时：
uv sync --locked --group dev
```

如果 uv 默认用户缓存不可写，可用 `UV_CACHE_DIR=/tmp/caps-uv-cache`；不要为普通运行重建现有环境。API 网络在受限执行环境中可能需要工具的网络执行权限，不要用无网络回放去冒充新 API 实验。

### API 角色与变量

当前所有角色的 base URL 都是 `https://api.siliconflow.cn/v1`。`spm/config.py:endpoints()` 使用 `.env`，`load_dotenv(..., override=False)`，**进程已有环境变量优先于文件**。不要打印 `.env` 或凭据；只查看非敏感模型配置。

| 角色 | 模型 | 配置变量 |
| --- | --- | --- |
| `chat` / `cheap` | `Qwen/Qwen2.5-7B-Instruct` | `CHAT_MODEL` / `CHEAP_MODEL` |
| `judge` | `Pro/deepseek-ai/DeepSeek-V3.2` | `JUDGE_MODEL` |
| `embed` | `BAAI/bge-m3` | `EMBED_MODEL` |
| `bench_judge` | `Pro/deepseek-ai/DeepSeek-R1` | `PRES_JUDGE_MODEL`；当前 CUPID 不使用该角色 |

公共认证变量为 `OPENAI_API_KEY` / `OPENAI_BASE_URL`。SPM 可选角色覆盖是 `CHAT_MODEL_API_KEY`、`CHEAP_MODEL_API_KEY`、`JUDGE_MODEL_API_KEY`、`PRES_JUDGE_MODEL_API_KEY` 及对应 `_BASE_URL`；嵌入使用 `EMBED_API_KEY` / `EMBED_BASE_URL`。**graphmem 的变量不同**，见 `.env.example` 和 [GRAPH_MEMORY.md](GRAPH_MEMORY.md)；SPM 示例在 `spm.env.example`。

Qwen 广告 TPM **80,000**，有效预算 **72,000**，chat/cheap 共用模型级限速。DeepSeek-V3.2 用户提供的 TPM 为 **2,000,000**；对齐分补测显式启用了该限速。限速协调以同一缓存数据库为范围，多个不同缓存的进程不能共同控制同一账户总 TPM。推荐并发必须服从预算，不因支持高并发就重复运行多个驱动。

普通 SDK/CLI 有本地 Qwen tokenizer 时使用 `datasets/tokenizers/qwen2.5-7b/tokenizer.json`；缺失则保守按 UTF-8 字节估算。M8/M9 冻结脚本显式引用并校验该 tokenizer，不能在缺失时直接启动。

`SPMConfig()` / CLI 默认：`tool_protocol=compact`、图管理、`update_policy=legacy`、`gate_completion=strict`、原文预算 6000 字符、适用/抑制阈值 0.5/0.2、最多 apply 8 项、最多候选 20 项、每探针 top-k 5。M8/M9 用的是独立实验配置，见 `config.py:VARIANTS` 与冻结计划，不能和默认混同。

## 3. 代码入口：按问题读模块

详细目录表在 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。优先定位：

| 任务 | 读这些文件 |
| --- | --- |
| 运行、恢复、选择方法 | `spm/__main__.py`、`runner.py`、`config.py` |
| 偏好单元、来源、图事件 | `models.py`、`store.py`、`pipeline.py` |
| 查询、门控、渲染、生成 | `pipeline.py`、`prompts.py`、`baselines.py`、`evaluation.py` |
| M7 保守图 / M8 对象维度图 | `conservative.py` / `typed_graph.py` |
| M8 整数协议 / M9 布尔协议 | `small_protocol.py` / `boolean_protocol.py` |
| 缓存、HTTP、402、限速 | `api.py`、`rate_limit.py` |
| CUPID 官方 1–10 分补测 | `cupid_alignment.py`、`scripts/spm_m8_alignment.py`、`paper/cupid_official/` |
| 数据字段与标签隔离 | `adapters.py` |

旧实验的 `scripts/spm_m*.py` 负责抽样、预留、源码冻结、执行和审计，`*_execute.py` 串行调度阶段，`*_replay.py` 离线重建，`*_report.py` 从产物生成 Markdown。**许多脚本依赖本地旧计划、缓存和预留名单，新 Git 克隆并不能直接运行它们复现旧数字。**

## 4. 已完成实验：不要误读指标

M0–M7 的历史在 [SPM_RESULTS.md](SPM_RESULTS.md)、各轮报告；无需全部重读。重要脉络是小模型接口可靠性改善、v6 原文补充、图与平铺对照，最终还未确认图管理带来稳定的生成偏好对齐收益。

### M8 CUPID-only

系统抽取/关系/门控/生成均为 Qwen，裁判 DeepSeek-V3.2、嵌入 BGE-M3 API。开发 6 题，测试 24 题，每 persona 一题，三个原生层各 2/8 题；两阶段成员和源码在首次预测前一起冻结。seed `20261015`。开发 **24/30** 有效，达到预设 80% 运行门槛；测试 **93/120** 有效。两阶段均只执行一次有界失败项续跑，**不要再续跑 M8**。

五方法共同有效 **15/24 题**，changing / consistent / contrastive 为 **6/5/4**。下表 F1 和偏好对齐分使用同一 15 题：

| 方法 ID | 含义 | Checklist F1 | 官方提示偏好对齐分（1–10） |
| --- | --- | ---: | ---: |
| `v6_m8` | 原图，共用 M8 新编码 | 0.525 | 5.47 |
| `typed_m8` | 对象/维度图 | 0.456 | 5.47 |
| `flat_m8` | 平铺追加 | 0.377 | 5.60 |
| `raw_m8` | 同原文预算、窗口、探针、排序的 RAG | 0.249 | 5.33 |
| `b1` | 完整历史，不截断 | 0.280 | 5.07 |

- 预设主比较 **typed−flat F1 +0.078**，95% 区间 **[-0.033, +0.194]**，未确认稳定收益。
- **原图−平铺 F1 +0.148**，区间 **[+0.031, +0.275]**，是未校正多重比较的探索性信号，且缺失可能非随机；不是全量或跨基准优势。
- 原图和新图的对齐分减平铺均 **-0.133**，配对区间分别 **[-1.067, +0.867]**、**[-1.133, +0.867]**；推断 F1 信号尚未体现在生成对齐上。
- 主测试缺失 27 项：抽取 21、查询解析 4、裁判覆盖 1、关系来源 1。失败保持缺失，不计为答错或 0 分。
- 新图的 **3 次 supersede 全在 cupid/53**，日期优先和指令语气两个轴中，认可助手修订被当成偏好变化；来源/时序审计通过不等于真实语义更新正确。
- 开发还发现 **7 条去重的跨 USER turn 复制错绑迹象**；合法来源编号不等于语义支持。
- 主测试 `gate_complete` 157 次 HTTP、922,156 已知 tokens，占阶段已知用量 23.7%。这是 M9 完整概率数组接口的效率动机。

详见 [M8 报告](SPM_M8_RESULTS.md)、[诊断](SPM_M8_DIAGNOSTICS.md)。两阶段合计 117/150 有效，1712 次 HTTP、4,946,888 个已知 tokens，14 次未知用量。归因有共享缓存与写入，不是独立冷启动成本；字符压缩和并发案例时间也不是 token 成本或独立冷启动速度。

### CUPID preference alignment score 补测

已对主测试 **93 份原回答全部完成补评**；没有重新生成、修改 F1 或旧 `response_score`。裁判只看到请求、原回答、**真实偏好和真实 checklist**，看不到方法、记忆、推断 checklist 或旧分数。官方发布提示逐项分析，只关注 checklist，保留 Markdown 输出，再严格解析末尾 1–10 整数。

采用本地 CUPID 原论文 §2.3、A.2、图 16，以及官方仓库固定版本 `a8560cab293ae98be4fe260689d58bddf96b51ef` 的发布提示；发布版比 PDF 增加了分值标签。**本次裁判为 DeepSeek-V3.2，论文为 GPT-4o**，而 M8 生成前还先推断偏好，因此不能直接横比论文绝对分数。旧 `response_score` 是简化代理质量分，不能重命名为本次指标。

补测：90 次 HTTP（3 个相同输入合并）、150,578 已知 tokens、0 未知用量、0 新裁判失败；93 份逐项分析的编号覆盖检查无缺口。93 份新评分精确离线回放 **0 HTTP**。14 项新增评分测试通过。结果在 [SPM_M8_ALIGNMENT_RESULTS.md](SPM_M8_ALIGNMENT_RESULTS.md)，原提示/许可在 [paper/cupid_official/README.md](paper/cupid_official/README.md)。

## 5. 本地数据与产物导航

数据均在本机，**不在 Git 中**：

| 数据/用途 | 路径 |
| --- | --- |
| CUPID | `datasets/cupid/test.parquet` |
| BenchPreS | `datasets/benchpres/data/test-00000-of-00001.parquet` |
| PERMA | `datasets/perma/tasks/` 及 `datasets/perma/evaluation/` |
| 历史 PrefEval | `datasets/PrefEval/`；不再用于新实验 |
| HotpotQA | `datasets/hotpot_qa/distractor/validation-00000-of-00001.parquet` |
| LoCoMo | `datasets/locomo/data/locomo10.json` |
| 用户另下载的 LongMemEval | `datasets/longmemeval/`；有数据不代表 SPM 已完成该基准评测 |
| Qwen tokenizer | `datasets/tokenizers/qwen2.5-7b/tokenizer.json` |
| 本地论文 | `paper/CUPID.pdf` 等；PDF 不提交 |

最近产物：

| 路径 | 用途 |
| --- | --- |
| `runs/spm/m8_{dev,test}_plan.json` / `m8_reservation.json` | 冻结源码摘要、配置、模型、成员、预算及预留 |
| `runs/spm/m8_{dev,test}_frozen/spm/` | 真正用于 M8 的源码，不受当前 M9 源码修改影响 |
| `runs/spm/m8_{dev,test}/samples/` / `memory/` | 原结果、trace、记忆库和快照 |
| 同目录 `verification.json` / `replay_verification.json` | 控制/来源/成本/旧样本核验及精确离线回放 |
| 同目录 `common_success_metrics.json` / `comparisons_vs_*.json` | 同题均值与配对区间；不要只比较各方法不同题集的均值 |
| 同目录 `pending.json` / `typed_operations.json` / `ledger_totals.json` | 缺失原因、真实图事件、HTTP/已知/未知用量 |
| `runs/spm/api_cache_m8.sqlite` | 隔离的 M8 API 缓存，当前约 436 MB |
| `runs/spm/m8_alignment_plan.json` | 独立补测方案及原文件/代码/参考文献摘要保护 |
| `runs/spm/m8_alignment/grades/` / `scores.csv` / `summary.json` | 补测原分析、逐题分数与配对结果 |
| 同目录 `api_cache.sqlite` / `analysis_coverage_audit.json` / `replay_verification.json` | 新裁判缓存、覆盖审计、零 HTTP 回放 |

M8 保护的旧样本 **2478 个**未变。补测额外保护旧样本与 M8 记忆 JSON 共 **2754 个文件**。补测计划校验当时全部 `spm/*.py` 和补测脚本的摘要；以后正常改代码时，旧补测脚本可能拒绝当前源码，应从对应 Git 基线/冻结源另作隔离复现，**不要改旧摘要绕过检查**。

## 6. M9 状态与下一步分岔

现有 [SPM_M9_PROTOCOL.md](SPM_M9_PROTOCOL.md) 是预备方案。`flags` 编码、动态 source/entity 范围、复制错绑保护、完整有序 gate 概率数组、助手修订/偏好变化区别提示已实现；没有 API 效果证据。原文编号绑定不能证明真正改写的内容有语义支持，也不能发现同一 turn 中选错句子。

旧 M9 预案：`flat_m9/v6_m9/typed_m9/raw_m9/b1`，seed `20261016`，开发 6 / 测试 24，每层 2 / 8，每 persona 一题，排除所有历史及预留组，公共历史最多 24 episodes。worker 16、HTTP 并发 8、timeout 180 秒；dev 1200 calls / 250 万 tokens，test 4000 / 800 万；每阶段一次失败续跑 500 calls / 150 万。开发完成率 ≥80% 且控制/回放核验通过才启动主测试。

**仅在用户明确选择继续原 M9 路线后**，确认 API 额度、数据/tokenizer、本地历史缓存齐备，以及不存在新 M9 预留，再考虑这些入口；本次交接不执行：

```bash
.venv/bin/python scripts/spm_m9.py prepare
.venv/bin/python scripts/spm_m9_execute.py
```

`prepare` 会创建成员、源码冻结和独立缓存；已有 reservation 时拒绝覆盖。`execute` 会运行 dev/test、自动决定单次失败续跑、审计与回放。额度 402 时不可直接反复重试；恢复后要检查已存在的阶段及 resume marker，再确定恢复入口，避免自动消耗已用过的续跑机会。

**注意两个待处理点**：

1. 现有 M9 沿用旧 `response_score`，还未将本次官方提示对齐指标集成到新轮冻结计划。若需要同时报告官方提示得分，应在首次新轮预测前定义独立评分协议与预算；不能事后将旧分数当作新指标。
2. 最新 v2 指导文档提出新的通道控制路线。接手先读它，核对用户是否采用；文档中将 M9 当作已跑结论的表述需纠正。未跟踪的新文件属于用户已有修改，本轮只提交 handoff 和 README，**不要使用 `git add .` 顺带纳入它**。

## 7. 实验纪律与常见恢复问题

- 成员、源码、配置、数据摘要与提示在首次 API 预测前冻结；改动需要另开轮次/目录，旧成功结果字节保留。没有模型、标签或 gold 泄漏到记忆和生成器。
- 不重新生成成功答案、不替换原评分、不人工补分。新增指标另存、记录其裁判/提示/预算与观察旧结果后的探索性性质。
- 数据组以 persona/身份隔离；抽样仅使用公共字段，不按成功或质量挑题。不同方法公平比较用共同有效题，单列各方法覆盖和原生层缺失。
- 不能将整数/布尔/来源校验通过等同于语义正确；保留变更事件的实际 USER 证据，避免把助手输出变化视为偏好变化。
- `api.py` 最多三次结构尝试；网络重试也计预算。调用前预留，未知用量保留，不能当免费。API 402 会在该 Gateway 内阻止同一模型/账户的新请求，需恢复额度并重启相应任务。
- 同一目录续跑严格核对源码/数据/科学配置；控制并发等运行参数可调，但不能混入不同模型/提示。不要手工清缓存掩盖失败。
- 冻结缓存回放应 **0 新 HTTP**，上下文/trace/原生成/评分一致；它不是独立重评。不要把共享缓存/TPM 等待的时间或归因成本叫冷启动成本。
- 当前没有正在运行的实验驱动；M8 与补测都已结束。本轮不触发新 API 实验。

## 8. Git 提交与推送：已走通的流程

仓库位于 CAPS 自身的 `.git/`，不要重新初始化父目录 `/home/alhandsome/projects`。提交身份仅在本仓库配置：

```text
user.name = AHU-LJJ
user.email = 229551356+AHU-LJJ@users.noreply.github.com
origin = https://github.com/AHU-LJJ/newcode1.git
http.version = HTTP/1.1
credential.https://github.com.username = AHU-LJJ
```

用户要求代码、测试、依赖声明/锁文件、研究笔记、进度、结果文档入库；**数据集、模型、环境、密钥和原始实验产物不提交**。`.gitignore` 已覆盖 `datasets/models/runs/.venv/.cache`、权重、数据库、日志和下载 PDF；`.env.example` / `spm.env.example` 是空密钥示例。报告里的本地证据链接在 GitHub 新克隆不可用，已在 README/结构文档说明。

常规流程，显式暂存本次修改，避免带入用户未提交文件：

```bash
git status --short --branch
git diff
git add HANDOFF.md README.md     # 替换为本次实际修改的文件
git diff --cached --stat
git diff --cached --check
git commit -m "docs: update project handoff"
GIT_TERMINAL_PROMPT=0 git push -u origin main
git status --short --branch
git rev-parse HEAD origin/main
GIT_TERMINAL_PROMPT=0 git ls-remote origin refs/heads/main
```

推送前检查暂存内容不含 `.env`、真实令牌、数据或模型；不要 `git add -f` 绕过忽略规则，不强推，不改全局 Git 身份。首次提交保留了官方参考文件原始字节，因此它们存在上游空白格式问题；这不是修改第三方提示内容/摘要的理由。

### 已遇到的认证问题及成功方案

普通 push 曾报告 `Invalid username or token`，重试又发生 `gnutls_handshake()` TLS 中断。但现有 `GIT_ASKPASS` 凭据实际通过了 GitHub `/user`，账号为 AHU-LJJ，仓库有 push 权限，HTTPS Git receive-pack 服务也返回 200。最终成功方式为：**明确用户名 AHU-LJJ、HTTP/1.1、临时 askpass 助手转交已验证的现有令牌**。临时文件仅含取环境变量的代码，令牌只在进程内存/子进程环境中，未写入文件或 Git 配置。

**本轮 handoff 的新情况**：上述备用流程在 `git credential fill` 处等待 25 秒后超时，未取得凭据；普通 push 也在 50 秒超时。只读远端查询成功，确认仍为初始提交。因此旧流程曾成功，不代表当前凭据助手仍能正常返回；不要打印凭据或反复无界重试，恢复助手后先核对本地/远端，再补推。

先试普通 push；若出现相同认证传递问题，以下是已验证方案的可复用形式。需要已有 Git 凭据助手可用；它不会生成新令牌，也不会打印令牌。网络执行需要当前会话允许访问 GitHub。

```bash
.venv/bin/python -u - <<'PY'
import os, subprocess, sys, tempfile
from pathlib import Path
r = subprocess.run(
    ['git', 'credential', 'fill'],
    input='protocol=https\nhost=github.com\npath=AHU-LJJ/newcode1.git\n\n',
    text=True, capture_output=True, timeout=25,
    env=dict(os.environ, GIT_TERMINAL_PROMPT='0'),
)
credentials = dict(line.split('=', 1) for line in r.stdout.splitlines() if '=' in line)
if r.returncode or not credentials.get('password'):
    raise SystemExit('Existing Git credential unavailable; refresh local GitHub authentication.')
with tempfile.TemporaryDirectory(prefix='caps-git-askpass-') as folder:
    helper = Path(folder) / 'askpass.py'
    helper.write_text(
        '#!' + sys.executable + '\nimport os,sys\n'
        'print("AHU-LJJ" if "username" in sys.argv[1].lower() '
        'else os.environ["CAPS_GIT_HTTP_TOKEN"])\n'
    )
    helper.chmod(0o700)
    env = dict(os.environ, GIT_TERMINAL_PROMPT='0', GIT_ASKPASS=str(helper),
               CAPS_GIT_HTTP_TOKEN=credentials['password'])
    result = subprocess.run(
        ['git', '-c', 'credential.helper=', '-c', 'credential.username=AHU-LJJ',
         '-c', 'http.version=HTTP/1.1', '-c', 'http.lowSpeedLimit=1',
         '-c', 'http.lowSpeedTime=30', 'push', '-u', 'origin', 'main'],
        env=env, timeout=90,
    )
    raise SystemExit(result.returncode)
PY
```

超时或连接中断后先查询远端是否已收到该提交，不能凭日志静默假定成功。普通 API bearer 验证通过不等于 Git 协议已完成推送。若需要诊断凭据，只输出状态/权限/布尔值，不输出 `git credential fill` 原文；Git 邮箱也不能替代 GitHub 登录。

## 9. 下个对话的最短检查顺序

1. `cd CAPS`，`git status --short --branch`，`git log -3 --oneline`；保留用户新增/未提交文件。
2. 读最新用户指示及未跟踪的 **SPM v2 指导文档**，确定继续 M9 还是转向新路线；不要默认开实验。
3. 只读 [PROJECT_STATUS.md](PROJECT_STATUS.md)、[CODE_STRUCTURE.md](CODE_STRUCTURE.md)、[M8 对齐分报告](SPM_M8_ALIGNMENT_RESULTS.md)；其余报告按需要查。
4. 核对 `.env` 中非敏感角色、数据/tokenizer 和旧产物是否仍存在；不要重装已有环境或下载已有数据。
5. 后续实现或实验先定义控制、指标、样本与预算，再改代码/测试/冻结新轮；旧结果不动。
6. 更新进度与结果文档，检查暂存范围，按已走通的 Git 流程提交和推送。
