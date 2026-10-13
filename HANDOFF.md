# CAPS / SPM 对话交接

更新：2026-10-13。给下一个接手的编码助手。**先读本文件和最新用户指示；不必重读 M0–M8 报告，也不必重读 v2 指导文档全文**（第 3 节已摘出要点）。v1 阶段的交接内容保留在标签 `v1-frozen` 的 `HANDOFF.md` 中，需要时用 `git show v1-frozen:HANDOFF.md` 查看。

## 1. 现状

| 项目 | 当前事实 |
| --- | --- |
| 工作目录 | `/home/alhandsome/projects/CAPS`（WSL2 Ubuntu 24.04）。在 Windows 侧看到的是 `\\wsl.localhost\Ubuntu-24.04\home\alhandsome\projects\CAPS` |
| Git | 当前分支 `v2`，已与 `origin/v2` 同步，HEAD 为 `baf216f`。v1 基线是标签 `v1-frozen`（`404a8e4`），`main` 停在它。仓库 [AHU-LJJ/newcode1](https://github.com/AHU-LJJ/newcode1) |
| 唯一未跟踪文件 | `scripts/SPM v2：情境化记忆影响控制——下一阶段项目指导文档.md`，是用户的 v2 指导文档。**不要提交，也不要 `git add .`** |
| 研究路线 | v2：记忆影响的情境化控制，图管理线已停止，M9 不再执行 |
| 默认方法 | 仍是 `v6_compact`。P4 新增的 `v6_situational` 是可选项，**没有提升为默认** |
| 已完成 | P0、P1、P1 判官人工校验、P3、P4（见第 4 节） |
| 正在运行 | 无。没有后台实验在跑 |
| 测试 | `.venv/bin/python -m pytest -q`，172 passed |

## 2. 环境、API 与操作要点

- Python 3.12，环境在 `.venv/`。常用命令：`.venv/bin/python -m pytest -q`，`.venv/bin/python -m spm --help`。
- **硅基流动**：配置在 `.env`。提供 `Qwen/Qwen2.5-7B-Instruct`（chat/cheap 角色）、`Pro/deepseek-ai/DeepSeek-V3.2`（judge 和 P1 配对核验）、`BAAI/bge-m3`（嵌入）。Qwen 的 TPM 为 80k（有效 72k），比较慢。**余额两次在跑到一半时耗尽，返回 HTTP 402**：遇到 402 立即停下，请用户充值，不要重试。
- **智谱官方 API**：配置在 `.env zhipu`（文件名里有空格），提供 `glm-5.3-flash`，用于构造配对集和强模型设置。读取由 `spm/pairset.py:build_endpoints` 完成，用的是 `dotenv_values`，不会污染 `os.environ`。GLM-5.3-Flash **一定会先思考**，不接受 `enable_thinking`，所以 `spm/api.py` 只在请求硅基流动时带这个字段。思考 token 计入 max_tokens，所以构造配对用 12288，选择用 4096，生成用 6144。
- **智谱限流**：用户说并发上限是 50，但门控请求带 16 个候选，token 很多。实测并发 48 被 429 了 527 次，16 仍有少量 429，4 才完全没有。**门控较重的运行先用 8–16 并发**。Gateway 支持按服务商分别限制并发：`Gateway(..., provider_limits={host: n})`。
- 不要打印 `.env` 里的值，只看变量名。
- **PowerShell 调用 WSL 时引号很容易出错**。`wsl -d Ubuntu-24.04 --cd ... -- bash -c '...'` 里嵌套的引号、`(`、`$` 经常被 PowerShell 解析坏。稳妥做法：把 shell 或 Python 脚本写到 `\\wsl.localhost\Ubuntu-24.04\tmp\xxx.sh`，再执行 `wsl -d Ubuntu-24.04 -- bash /tmp/xxx.sh`，用完删掉。写多步 shell 时加 `set -e -o pipefail`，否则 `pytest | tail` 会吞掉失败。
- WSL 里的 `/tmp` 在 WSL 重启后会被清空。
- Git Bash 工具通过 UNC 路径读文件可以，但在 UNC 路径上跑 `git` 会报 dubious ownership，读写 git 要用 WSL 内的 git。
- **推送**：WSL 里没有凭据助手。用 Windows 侧的 git 推送：`git -c safe.directory='*' -c http.version=HTTP/1.1 -C \\wsl.localhost\Ubuntu-24.04\home\alhandsome\projects\CAPS push origin v2`。用户在 Windows 上登录过一次，之后就能直接推。偶尔会遇到 TLS 握手失败，重试即可。
- **Windows 内存**：P1 跑到一半时曾因系统内存不足被 Claude Code 杀掉。WSL 有约 11 GiB。如果后台任务又被系统杀掉，先汇报，不要自己重启。
- **用户偏好**（已写入记忆）：长时间运行只需检查一两次，确认正常后，只在报错、出现 402/429 或跑完时汇报，不要逐批汇报进度。

## 3. v2 路线要点（来自用户的指导文档）

核心主张：记住不等于该用，判断该用也不等于用对。要把记忆对回答的影响按通道分解（风格、内容选择、立场、硬约束），并做情境化控制。计划的方法梯度：L0 无记忆、L1 全部历史、L2 原文 RAG、L3 平铺、L4 加门控（即 `v6_compact`）、L5 加情境特征、L6 加类型化写入与通道化渲染、L7 加反事实校验与草稿再编辑；另有 P（提示词基线）和 G（MemGate 类门控）。预实验：P0 冻结基线，P1 判断≠行为，P2 通道诊断集（MIV/MS），P3 门控 oracle。训练线：T1 门控、T2 影响检测器、T3 抽取蒸馏、T4 受控生成，**显存只有 8 GB**，只能做 1.5B–3B 的 QLoRA。

文档里有两处与事实不符，已核对：
- M9 从未运行，文档把它写成了已有结论。
- 显存实际是 8 GB。

PairPref（arXiv 2609.34526）的数据集找不到下载链接，所以 P1 按它的公开设计自建了配对集。

## 4. 已完成实验与结论

结果文档：[SPM_P1_RESULTS.md](SPM_P1_RESULTS.md)、[SPM_P3_RESULTS.md](SPM_P3_RESULTS.md)、[SPM_P4_RESULTS.md](SPM_P4_RESULTS.md)、[PROJECT_STATUS.md](PROJECT_STATUS.md)。

**配对集**：320 个槽位（10 领域 × 4 子话题 × 8 类情境），最终通过 **257 对**。每对包含一条偏好、同一个请求、正负两个情境、四个候选回复（遵循 / 放下并说明 / 放下不说明 / 答非所问），以及 15 条干扰偏好。构造流程：GLM 生成 → 规则过滤 → BGE 去重和污染检查 → DeepSeek 核验 → Qwen 无上下文筛查。DeepSeek 核验一对都没拒，可能偏宽松。

**P1**：方法 L0/L1/P/L4 × 两种设置。小模型设置全部用 Qwen；强模型设置的回答、选择、门控、解析用 GLM，偏好整理仍用 Qwen。判官是 DeepSeek-V3.2。
- 强模型下，P 的选择 Pair success 是 0.914，生成只有 0.209，差距 +0.705，H2 通过。
- 小模型差距 +0.139，判为不确定。
- 小模型缺失 103 对，都是 Qwen 的门控或解析输出格式不合规。L4 上下文构建失败时整对都不写样本，所以 L0/L1/P 也被一起带缺失了，这是脚本设计造成的。

**判官人工校验**：60 条盲标加 6 条仲裁。三档 κ 0.53，二值 κ 0.58/0.68。判官偏严：回复里提到了偏好，内容却没有照着做，也会给 0.5。校准映射是：判官 0 → 人工判 0 的概率 1.0，判官 0.5 → 0.4，判官 1 → 0.0。校准后，强模型 P 的差距是 +0.49，H2 仍成立；小模型不显著。

**P3**：把 L4 的门控决定换成真实标签。L4 生成失败中约 85% 是门控先判错了。O1 只修目标偏好的门控，生成 PS 强模型从 0.511 升到 0.769，小模型从 0.535 升到 0.760；O2 再压掉全部干扰偏好，分别到 0.862 和 0.883。门控判对时，生成器基本能放下偏好。

**P4**：只换门控提示，用 `gate_policy='situational'`。
- 强模型：生成 PS 从 0.511 升到 0.661，提升 +0.150 [+0.091, +0.207]；负情境放行从 0.445 降到 0.176；正情境 Use+ 只变了 −0.003。按冻结规则判为"改进"。
- 小模型：+0.053，区间跨 0，判为不确定，有 21 对缺失。
- 提升主要集中在情绪、身体、时间这几类情境。
- **局限**：新提示是照着 P1 的 8 类情境写的，只在同一批配对上测过。

## 5. 代码地图（v2 新增）

| 文件 | 内容 |
| --- | --- |
| `spm/pairset.py` | 配对集构造：槽位、生成、规则过滤、去重、核验、无上下文筛查；`build_endpoints`，负责读取智谱配置 |
| `spm/pairbench.py` | P1 评测：记忆池、各方法上下文、选择、生成、判官、指标、bootstrap；P3 的 oracle 函数（`oracle_decisions`、`render_oracle`、`gate_correct`）；`calibrated_generation` |
| `spm/prompts.py` | 新增 `GATE_SITUATIONAL`（在 `GATE_ANSWER_FILTER` 后追加） |
| `spm/config.py` | 新增 `VARIANTS['v6_situational']` |
| `spm/api.py` | `enable_thinking` 只发给硅基流动；`provider_limits` 按服务商限制并发 |
| `scripts/p1_build.py` / `p1_run.py` / `p1_report.py` | 配对集构造、P1 运行（prepare/run/analyze）、离线严格口径分析 |
| `scripts/p1_annotation.py` | 盲标导出（export）、一致性（score）、人工校准（calibrate） |
| `scripts/p3_run.py` / `p4_run.py` | P3 oracle、P4 情境化门控，结构相同：prepare 冻结方案，run 可续跑，analyze 离线分析 |
| `tests/test_pairset.py` / `test_pairbench.py` | 配对集与评测的测试 |

## 6. 本地产物（`runs/spm/`，不入库）

| 路径 | 内容 |
| --- | --- |
| `p1_pairs/items.json` | 257 对配对集（`accepted=true`）及每个阶段的拒绝原因 |
| `p1/` | P1：`plan.json`、`samples/{small,strong}/`、`memory/`（每对的给定偏好库）、`failures.json`、`summary.json`、`report_analysis.json`、`api_cache.sqlite`（365 MB） |
| `p1_annotation/` | `annotation_sheet_annotated.csv`（用户标注原件）、`annotation_sheet_final.csv`（仲裁后）、`key.json`（盲标答案）、`adjudicate.py`、`agreement.json`、`calibrated_p1.json` |
| `p3/`、`p4/` | 各自的 plan、samples、summary、usage、缓存副本；P4 另有 `memory/`（P1 记忆库的复制件） |
| `p1_pairs_pilot*`、`p1_smoke` | 早期试跑，可忽略 |

**冻结摘要注意**：P1 和 P3 冻结后，`spm/pairbench.py` 又改过，所以 `p1_run.py run` 和 `p3_run.py run` 会因代码摘要不匹配而拒绝续跑，这是预期行为。两轮都已完成，`analyze` 不检查摘要。如果要复现，从对应提交另开目录。**不要改冻结摘要来绕过检查。**

## 7. 实验纪律

- 首次预测前冻结 `plan.json`：配对、代码摘要、提示、模型、阈值，以及**判定规则**。续跑时只能改并发、批量这类运行参数。
- 失败不写样本，也不算 0 分，单独列出。方法比较只用两边都完成的配对。
- 看过结果后才定的口径（比如严格 uptake、人工校准）要标为探索性，不能改冻结的判定。
- 已完成的样本不重跑。缓存命中不计费。提交前先跑测试，只暂存这次实际改动的文件，并检查有没有密钥。

## 8. 下一步候选（等用户选择）

1. **效用侧检查**：`v6_situational` 会不会把该用的偏好也压掉。可以在 CUPID（有对齐分补测协议）或 PERMA 上对比 `v6_compact`。决定改默认之前必须先做这一项。
2. **泛化**：换领域和情境写法，另外构造一批配对（参照 `spm/pairset.py` 的流程），再测 `v6_situational`，排除"提示是照着题写的"这种影响。
3. **小模型门控**：提示收益有限，问题主要是格式失败和两侧误判。对应 T1（训练门控，8 GB 显存，1.5B–3B QLoRA）和 T3（结构蒸馏）。
4. **P2 通道诊断集**（MIV/MS），v2 文档里还没做。
5. 让 `p1_run.py` 里各方法的失败互不影响，避免 L4 失败把 L0/L1/P 一起带缺失。

## 9. 新对话的最短检查顺序

1. `git status --short --branch`、`git log --oneline -4`（在 WSL 内执行），确认在 `v2` 上、HEAD 是 `baf216f` 或更新的提交。
2. 读最新用户指示，再按需读 PROJECT_STATUS 和对应的结果文档。
3. 要调用 API 前，先确认两家都有额度。智谱的门控类调用先用低并发。
4. 新实验沿用 `scripts/p4_run.py` 的结构：prepare 冻结 → run → analyze，把判定规则写进方案。
