# SPM：情境化偏好记忆原型

根据 [编码指导文档](SPM%20情境化偏好记忆：编码指导文档.md) 实现的独立 API 系统，代码在 `spm/`。复用现有 uv / Python 3.12 环境，只新增 Pydantic 依赖。SPM 不加载 torch、本地语言模型或 GPU；已有 `graphmem/` 保持原进度。

当前完成的是指导文档第一阶段的可运行原型及 M0 验证：四个基准各 3 条开发样本、v4/b1 共 24 项评测全部执行成功，41 项测试通过，缓存复现与续跑没有新增 API 请求。[首轮结果与失误分析](SPM_RESULTS.md)保留了指标和协议边界。可以运行四个本地基准、基线和 v0–v6 消融；没有进行强化学习，也没有实现第三阶段的情境敏感度学习或证明论文级效果。

随后进行了 [M1 扩展验证](SPM_M1_RESULTS.md)：恢复轨道使用 DeepSeek-V3.2 统一重评，测试组 117/117 完成，开发组 192/195 成功，3 项裁判标志一致性错误保留为执行失败。原 GLM 轨道成功 293/312 项，因持续 503/超时停止并另行保留。M1 完成时测试数为 47，后续 120 项既有测试通过，本次另通过 14 项 CUPID 对齐评分测试；源码、清单和失败队列可审计。裁判选择会影响评分，当前不能声称系统已稳定取得优势。

用户已移除 PrefEval；后续 `--benchmark all` 仅包含 PERMA、CUPID、BenchPreS。历史文档、显式 PrefEval 选项和已冻结旧实验保留作复现，新增实验不调用 PrefEval。当前 [M8](SPM_M8_PROTOCOL.md)只测 CUPID。

M8 的 `tool_protocol='small'` 在 [small_protocol.py](spm/small_protocol.py) 中实现整数语义标签、原文编号和对象/维度；[typed_graph.py](spm/typed_graph.py) 实现单次分类、跨轴保留、同轴隐含更新与因果校验。三种管理器共用新编码，`raw_m8` 与管理器的原文检索完全匹配；结果见 [M8 报告](SPM_M8_RESULTS.md)，来源问题见 [开发诊断](SPM_M8_DIAGNOSTICS.md)。开发/测试使用独立冻结包，不受后续源码修改影响。

按用户要求，另对 M8 主测试的已有回答补评 CUPID 原论文的 preference alignment score（1–10）：采用官方发布提示、真实偏好及真实 checklist，裁判仍为 DeepSeek-V3.2。[已完成补测](SPM_M8_ALIGNMENT_RESULTS.md)：93/93 评分成功，共同 15 题原图 / 新图 / 平铺 / 原文 RAG / 完整历史为 5.47 / 5.47 / 5.60 / 5.33 / 5.07，尚未确认图方法的对齐收益。逐项分析和分数独立保存，不改原 F1 或旧 response_score，不重新生成回答。来源与边界见 [评分参考](paper/cupid_official/README.md)，执行脚本见 [spm_m8_alignment.py](scripts/spm_m8_alignment.py)，逐题得分见 [CSV](runs/spm/m8_alignment/scores.csv)。

为修复 M8 发现的类型、来源与语义问题，[boolean_protocol.py](spm/boolean_protocol.py) 另实现 `tool_protocol='flags'`：严格布尔语义标志、请求内有界来源/提及编号，以及明确的跨 USER turn 逐字复制错绑校验。查询采用 entity_indices / primary_index；门控一次返回完整概率数组，由宿主按位置绑定。对象/维度关系提示区分助手草稿修订与用户偏好变化。对应 `flat_m9` / `v6_m9` / `typed_m9` / `raw_m9`。这套接口已有本地测试，尚未发起 M9 API 验证；不得把实现通过测试当作基准效果。见 [M9 预备方案](SPM_M9_PROTOCOL.md)。旧 compact payload 与写签名保留，默认仍 v6_compact。

## 系统路径

写入：会话 → 高召回规则预筛 / 分窗 → 带 USER 原文证据的偏好单元 → BGE-M3 → 批量语义关系判定 → 按时间合并、分支、替代或挂起。

读取：请求情境解析 → 多属性探针 → 语义 / 命名实体 / 全局约束 / 挑战关系召回 → 适用概率门控 → 应用、抑制、不确定 → 短记忆上下文 → 生成器。

单元保存条件、作用域、偏好极性、约束/口味/风格类型、置信度、独立会话支持数和证据。SQLite 保存不可变原始会话及更新事件；`challenges` 和 `superseded_by` 保存冲突与替代关系。不同情境可以并存；同一条件下的强旧偏好遇到弱新反馈时，新观测进入 pending，重复独立证据可以促使它生效。明确的变更表述允许直接替代。

| 文件 | 用途 |
| --- | --- |
| `models.py` / `config.py` | 数据模型、配置及消融 |
| `prompts.py` | 抽取、关系、解析、门控、生成提示 |
| `pipeline.py` / `store.py` | 写入、读取、状态转移与存储 |
| `api.py` | 异步调用、限流重试、缓存、预算及调用账本 |
| `adapters.py` | 四个数据集的公共输入和评测标签分离 |
| `baselines.py` | 无记忆、完整历史、原文 RAG、实体近期记忆 |
| `evaluation.py` / `runner.py` | 指标、并发实验、检查点与汇总 |

## 配置与运行

已有 `.env` 中的 `OPENAI_API_KEY` / `OPENAI_BASE_URL` 直接用于 SPM；不需要修改现有 graphmem 模型设置。SPM 默认模型如下，可用 [配置示例](spm.env.example) 中的变量覆盖。

| 角色 | 默认模型 |
| --- | --- |
| 记忆抽取、关系判定 | `Qwen/Qwen2.5-7B-Instruct` |
| 情境解析、门控、生成 | `Qwen/Qwen2.5-7B-Instruct` |
| 独立评测裁判 | `Pro/deepseek-ai/DeepSeek-V3.2` |
| 向量 | `BAAI/bge-m3`，1024 维、L2 归一化 |

按用户选择，抽取与回答均采用 Qwen 小模型，裁判独立采用 DeepSeek。默认 CLI / `SPMConfig()` / `SPM(...)` 使用 `v6_compact`；旧 v4/v5/v6 的接口定义仍保留。Qwen 的 CLI 与 SDK 自动采用 72,000 TPM 有效滚动预算，chat/cheap 共享；显式 `--model-tpm MODEL=TPM` 可以覆盖，离线 tokenizer 缺失时使用保守字节计数。`PRES_JUDGE_MODEL` 配置预留给后续 BenchPreS 生成式评测，当前门控评测不会调用该角色。

在 CAPS 根目录运行：

```bash
uv run python -m spm inspect

# 四个基准各取 3 个开发集样本；v4 对照完整历史，共 24 个方法×样本
uv run python -m spm run --benchmark all --split dev --limit 3 \
  --methods v4 b1 --workers 8 --concurrency 16 \
  --max-calls 2000 --max-tokens 5000000 --output runs/spm/my_m0

# 单基准的小规模消融；避免起步就运行全部数据
uv run python -m spm run --benchmark cupid --split dev --limit 12 \
  --methods v0 v1 v2 v3 v4 b0 b1 b2 b4 \
  --workers 8 --concurrency 16 --output runs/spm/cupid_dev12

# PERMA 四种数据配置：sd_clean / sd_noise / md_clean / md_noise
uv run python -m spm run --benchmark perma --perma-mode md_noise \
  --perma-types 2 3 --limit 3 --output runs/spm/perma_md_noise

# 仅对已保存概率扫阈值，完全离线
uv run python -m spm frontier runs/spm/my_m0 --thresholds 0.3 0.4 0.5 0.6 0.7

uv run pytest -q

# 集成演示：餐厅查询应用麸质限制，投诉邮件抑制该限制
uv run python examples/spm_demo.py
```

重复**同一运行命令、同一目录**自动续跑：成功样本跳过，失败样本重试，已验证的 API 输出和向量缓存复用。模型、数据、提示/代码或科学配置改变时会拒绝混入旧结果，应使用新目录。可以改变 workers、并发、超时及每次调用预算。`--limit 0` 表示所选 split 的全部样本；API 预算仍生效。

默认 `--timeout 180`，全局 HTTP 并发 16、样本 worker 8。抽取以 8 个会话为历史批次；偏好内容间的静态关系每 16 对一批并发预判，状态仲裁仍按会话顺序执行。后续会话的关系不会写入或影响早期会话状态；预判后候选变化导致缺失的比较会补做。相同用户/方法下的已完成历史前缀可以通过 SQLite backup 复用，避免重新仲裁长历史。不同用户及历史分支不会混用数据库。

仅改变读取配置的 `v3`、`v4`、`v5`、`v6`、`no_anticipation` 在同一实验中共用写入任务和数据库。共享签名包含条件开关、仲裁方式、关系候选参数、原文向量模式及分窗参数；改变这些写入配置或换成实体基线时仍单独建库。每个查询按自己的检索和门控配置读取，避免把重复写入耗时误当成消融代价。

接入在线 agent 时，同一用户复用同一个 `SPM` / `SPMClient` 实例；该实例的查询与写入共用锁，避免读到同一 SQLite 连接上尚未提交的半次更新。实验 runner 在历史构建完成后使用独立的只读流程并发查询。当前不提供多进程共享用户数据库的服务端协调。

每次运行都在发请求前预留调用数和 token 上限并写入持久调用账本，重试也计入预算。未知用量保留保守估算。429/5xx 使用退避、抖动、Retry-After 和模型级冷却；结构错误最多修复两次，被输出限额截断时预算依次翻倍、最多 16,384 token；只有通过 schema 和证据/ID 检查的结果会缓存。同一进程的重复请求合并。缓存默认跨实验共享在 `runs/spm/api_cache.sqlite`。

收到 HTTP 402 后，同一 endpoint/model/key 在该 Gateway 实例中停止新网络请求及预算预留，缓存仍可读取，其他模型或账户独立处理。恢复额度后重新启动任务。M1 旧清单对应的源码保存在冻结目录，使用 `scripts/spm_m1_resume.py` 续跑；直接用修改后的包续跑旧目录会被代码 hash 检查拒绝。

完整历史基线默认限额 300,000 字符，可用 `--max-context-chars` 调整；超过限额记为失败，不截断后冒充完整历史。其他上下文或 provider 限额失败也保留为失败。token 预算是每次进程运行的上限，续跑会开启新的调用预算。

更大一些的分组开发/测试验证可以分别运行：

```bash
# 开发集：其他基准各 12 条，PERMA 3 条；去门控和去探针对照
uv run python -m spm run --split dev --limit 12 --perma-limit 3 \
  --methods v4 v3 no_anticipation b1 b2 --workers 12 --concurrency 24 \
  --max-calls 1200 --max-tokens 2000000 --output runs/spm/my_m1_dev

# 保留测试组：PERMA 仅短历史 type2，明确属于资源受限子集
uv run python -m spm run --split test --limit 12 --perma-limit 3 \
  --perma-types 2 --max-history-episodes 24 --methods v4 b1 b2 \
  --workers 12 --concurrency 24 --max-calls 1200 --max-tokens 2000000 \
  --output runs/spm/my_m1_test
```

`--max-history-episodes` 在划分之后、抽样之前，按公共历史长度过滤样本，**不会截断会话**。清单同时记录过滤前后样本量；不能把子集成绩当成全体 PERMA 结果。模型、提示、阈值和样本应在查看测试预测前固定。

## 方法和消融

| 方法 | 含义 |
| --- | --- |
| b0 | 无记忆 |
| b1 | 全部合法历史；BenchPreS 中相当于应用全部给定偏好 |
| b2 | BGE-M3 检索原始会话窗口；不抽取偏好 |
| b4 | 抽取 + 命名实体 / 近期会话检索，不做冲突仲裁或门控 |
| v0 | 平面偏好记忆，最新冲突覆盖，无条件 / 预判 / 门控 |
| v1 | 条件分支，无仲裁 / 预判 / 门控 |
| v2 | 增加冲突仲裁 |
| v3 | 增加需求预判和多探针 |
| v4 | 原概率门控，保留作旧版对照 |
| v5 | 答案过滤适用性门控，旧接口 |
| v6 | v5 + 最多三个、总计 6,000 字符的历史原文片段，旧接口对照 |
| v6_compact | v6 + USER 原文编号绑定 / 候选整数序号，当前默认 |
| v6_compact_no_raw | 同一编号接口，关闭生成 / 推断器的原文补充 |
| no_condition | v5 去掉写入条件与实体硬规则 |
| no_anticipation | v5 去掉扩展探针，仍解析当前情境 |

所有方法使用同一生成器和评测裁判。v0–v3 的门控关闭时，最多应用 `max_apply` 个候选；v4/v5/v6 根据阈值分类。默认应用阈值 0.5，抑制阈值 0.2，区间中间为不确定。不确定偏好不进入生成上下文；抑制文本默认也不进入，可用 `--render-suppressed` 做对照。

## 数据与评测边界

按用户/身份组划分 20% 开发组和 80% 测试组，seed=42；PrefEval 按规范化问题分组，使三种偏好表达方式共享划分。`--limit` 在 split 内按表达方式、CUPID instance_type、PERMA type、BenchPreS domain 轮流抽样。少量开发样本仅用于检查系统和提示，不能代表完整基准性能。

这里的 dev/test 是本项目在已下载样本池内部做的划分，并非官方独立开发集/测试集划分。CUPID 和 BenchPreS 的本地源文件本身名为 test；M1 的提示和阈值在测试预测前冻结。M2 的门控假设参考了 M1 失误（包括旧测试组），因此另选新身份组进行测试；不会把旧测试组继续称为未观察样本。

| 基准 | 当前合法输入 | 评测 |
| --- | --- | --- |
| PrefEval | 显式历史偏好原句，或 choice/persona 的实际对话；当前 question | 生成遵循率与四类失败标签；API 替代裁判 |
| CUPID | `prior_interactions.*.dialogue`；`current_request` | 偏好 checklist precision / recall / F1、回答 1–10 分；API 替代匹配器 |
| PERMA | 历史 context 中的角色/文本/时间；准备好的 question | 提供 MCQ options 后生成单标签，与 native gold_label 精确匹配 |
| BenchPreS | 已给定 preference_attribute；实际 recipient/task 指令及 domain | gate_AAR、gate_MR、不确定率与离线阈值曲线 |

本地 PrefEval 共 3,000 条，CUPID 756 条，BenchPreS 390 条。PERMA 的四种配置都能读取；按模式分别评测，不混合报一个分数。PERMA type1 是官方 ZeroMemory 条件，即使文件提供 context 也不写入；type2/type3 会话按日期和时段排序，拒绝包含问题日期之后历史的样本。MCQ 选项仅在最终选择阶段给生成器，不用于抽取、解析、召回或门控。

两处有意的字段来源例外：PrefEval 显式 `preference` 同时是历史用户原句和目标偏好；BenchPreS 的 `preference_attribute` 本来就是公共给定输入。仅凭“输入出现了 gold 文本”判泄漏会误杀这些合法输入。其余 gold preference、context_factor、checklist、PERMA affinity/profile 注释、BenchPreS preference_label 都留在 `Case.gold` 中，记忆代码只接收 `MemoryInput`。

当前并非原论文评分脚本的完整复现：

- PrefEval 没有下载到长干扰对话和正式 MCQ 文件，当前是 **0 distractor 的生成任务**。历史 choice options 是偏好引导对话，不是当前问题的 MCQ。四类失败允许重叠。
- CUPID 不加载本地 PrefMatcher；API 返回预测 checklist 与 gold 的语义匹配边，使用最大一对一匹配，避免一条宽泛预测反复计分。回答质量由 GLM 独立打分。
- PERMA 仅接入本地 prepared MCQ；原论文交互式用户模拟、turn1/turn≤2 成功率暂未实现。
- BenchPreS 当前是指导文档第一阶段的 **给定偏好、移除个人事实的 gate-only 变体**，不是原论文生成消息后再判断遵循情况的 AAR/MR。输出明确使用 `gate_AAR` / `gate_MR`，不与原论文数字直接比较。两项比率按偏好标签进行 micro 汇总。

依据：[PrefEval 官方代码](https://github.com/amazon-science/PrefEval)、[CUPID 官方仓库](https://github.com/kixlab/CUPID)、[PERMA 官方仓库](https://github.com/PolarisLiu1/PERMA)、[BenchPreS 数据与评测定义](https://huggingface.co/datasets/sangyon/BenchPreS)。适配代码独立编写。

## 输出和复查

`runs/spm/<实验>/` 保存：

- `manifest.json`：模型名、配置、代码/源文件 hash、所选样本和输入来源。没有 API key。
- `samples/*.json`：每个方法×样本的预测、评测理由、耗时、错误，以及解析、探针、候选、原始概率、有效决策、最终上下文。
- `memory/*.sqlite`：不可变会话、偏好、向量、状态和更新事件；`.snapshot.json` 可直接复查写入结果。`index.json` 记录已完成的历史前缀。
- `memory/*.profile.json`：本次建库的准备、关系预判、顺序提交和总耗时，以及已完成/新写入/前缀复用的会话数。样本 `timing` 分开记录上下文获取和评测耗时；共享建库等待和缓存命中也会影响这些数字，不能直接解释成独立冷启动延时。
- `metrics.json` / `summary.json`：整体及分层结果、成功/失败数。存在失败时 `complete_run_metrics` 为 null，成功子集统计单列，不把失败记作答错或隐瞒。
- `comparisons.json`：与 b1 在相同成功样本上的成对差值、排除数量和按身份组重采样的 95% 区间；BenchPreS 继续按标签 micro 计算。有失败或小样本时须同时查看数量与边界。
- `frontier.json`：BenchPreS 的离线门控阈值曲线。
- `invocations.json` / `usage/*.json` / `usage.json`：分调用阶段的已知 provider token、HTTP 请求、缓存命中、合并请求、schema 修复及未知用量。进程被强制中止后，可以从预先写入的调用账本恢复未完成调用及其保守预留；缓存命中计数无法从中恢复。**未知调用用量不会伪装成零；这里不是账单金额。** 开发中更早的旧版本日志不具备全部恢复字段。

新的请求账本还记录 HTTP 尝试耗时，`usage` 按阶段给出中位数和 p95。并发请求的累计耗时不能等同于实验墙钟时间；旧请求没有该字段，不补造历史耗时。

有至少两个独立用户/问题组时提供按组重采样的 bootstrap 区间。小规模试跑中只有一个组的分层不提供区间。检查效果时可沿 `.snapshot.json` 的写入证据 → 候选召回 → 门控概率 → 生成答案定位损失，不能只看最终均值。

## 对指导骨架的调整与剩余限制

向量相似度只用于挑选关系候选，不能直接证明两个偏好相同；相同极性的高相似单元仍可能条件不同或语义相反。支持数按独立会话计算；重复窗口与重放不会增加确认次数。条件不完全一致时保留分支，不自动拓宽作用域。

条件实体先规范化，再匹配请求中明确出现的命名对象；支持多个对象。格式、任务、渠道、受众和 context 也保存在单元中，但尚未实现完整的实体别名消歧和条件语义规范化，因此过细或不同措辞的条件仍可能导致分支过多。

因缺失输出就把概率当 0 会制造漂亮但不可靠的抑制率。现在门控和关系输出必须完整、无重复地覆盖输入 ID；缺失重试，仍失败则记录评测失败。pending 与其挑战对象同时被模型判应用时，确定性保护会把双方改成不确定，原始模型概率也保留。

皮肤刺激、过敏、不耐受等应抽成跨相关类别的约束，避免局部穿衣/用餐话题错误缩窄作用域；显式禁止项渲染为必须遵循。除非用户放宽要求，生成器提示禁止用“绕开活动的一部分”来推荐违反偏好的选项。

原始片段目前只辅助验证已有候选，不会从检索时临时生成新单元；因此**不能保证恢复写入阶段漏抽的偏好**。长对话预筛仍是规则法，可能漏掉不含触发词的隐含偏好。短会话全量抽取，长会话预筛覆盖率保存在更新事件中。

门控概率尚未校准；不要用测试集标签挑阈值。当前 SQLite / 向量扫描适合单用户几百到几千单元；更大规模可更换 ANN 索引，但应保留硬约束和挑战关系的补充召回。


## M2：新门控与新身份组

[实验结果](SPM_M2_RESULTS.md)：开发组 195/195 有效评分，测试组 154/156；两项共享历史的原文证据错误保留为失败。新门控改善部分适用召回，但 CUPID 成功配对子集的 F1 下降，M2 汇总时保留 v4，随后按用户选择切换为 v5。

M2 对照 legacy `v4` 与 `v5`，后者只增加通用的适用性定义：偏好能否改变或过滤当前答案。条件匹配、实体规则、检索、生成器及 0.5/0.2 阈值保持同一设置。裁判 schema 修复现在反馈短校验原因，继续拒绝相互矛盾的标签，不人工改分。

[预先冻结方案](runs/spm/m2_plan.json)固定 39 条开发样本 × 5 方法、39 条新测试样本 × 4 方法，共 351 项。新测试组排除先前 M0/M1 使用的身份或问题组；BenchPreS 仅剩两个新画像，PERMA 为三个新用户的短历史 type2 子集。`--case-file` 可以传入精确样本 ID 列表；清单记录文件摘要，若 ID 被划分、长度过滤或 limit 排除，运行会拒绝继续。M2 使用隔离的 API 缓存副本，开发组复用冻结的旧写入库。

```bash
# 已冻结 M2 的断点续跑入口；源码/样本文件不一致会拒绝执行
uv run python scripts/spm_m2.py run --phase dev
uv run python scripts/spm_m2.py run --phase test
# 两个阶段停止后离线核验及成对比较
uv run python scripts/spm_m2.py analyze
uv run python scripts/spm_m2_report.py
uv run python scripts/spm_m2_replay.py
```

入口统一使用 DeepSeek-V3.2 裁判，进程级覆盖模型配置，不修改 `.env`。每个阶段默认 48 个 worker、HTTP 并发 16、最多 1,500 次调用及 300 万 token；续跑开启新的预算并保留历史账本。生成器和裁判同源的局限仍然存在。


## M3：默认升级与原文证据消融

M3 冻结版本的 CLI 不指定 `--methods` 时使用 `v5`；直接构造 `SPMConfig()` / `SPM(...)` 也默认使用答案过滤门控。旧版仍可明确指定 `--methods v4`。`no_condition/no_anticipation` 现基于 v5；此前实验使用各自冻结源码，定义不追改。

`v6` 在 v5 的结构化上下文后补充检索原文，最多三个片段、累计 6,000 个原文字符，保留角色、会话顺序/时间和截断标志；这些是证据而非已应用偏好。即使抽取单元为空也可检索原文。给定偏好的 BenchPreS 关闭原文回退，因此不能用它证明原文不会引入误用。v5/v6 的门控决策保持一致，区别发生在生成/推断读取的证据上；仍共用写入库，不新增 LLM 读取调用。

原文校验错误反馈现在提供所引用 USER 原句的短片段，或原文确切出现的 USER 索引，帮助模型修复；不会自动改写 span 或接受近似证据。只用于新协议，M2 的失败项仍保留。

[冻结方案](runs/spm/m3_plan.json)预先固定开发组 40 个问题 × 2 方法（含旧失败题的诊断）、测试组 59 个问题 × 5 方法。PrefEval/CUPID/PERMA 分别有 24/12/3 个新身份或问题组；BenchPreS 20 个问题来自十个已见画像，每画像两个新问题，属于问题保留验证。三个 PERMA 用户是剩余的短历史 sd_clean type2 子集，不是全量泛化验证。

```bash
uv run python scripts/spm_m3.py run --phase dev
uv run python scripts/spm_m3.py run --phase test
# 两阶段停止后离线核验
uv run python scripts/spm_m3.py analyze
uv run python scripts/spm_m3_replay.py
uv run python scripts/spm_m3_report.py
```

开发 HTTP 并发 16、测试并发 32，worker 均为 64；两阶段分别设 1,000/2,500 次调用及 150 万/500 万 token 的单次预算。裁判仍统一使用 DeepSeek-V3.2，嵌入 BGE-M3，缓存和新输出独立存放。预先记录旧样本 hash，后续验证原始结果未改动。

[M3 完整结果](SPM_M3_RESULTS.md)：开发 80/80、测试 295/295 有效评分，无未完成项。新测试 v4/v5/v6 的 PrefEval 遵循率为 66.7%/70.8%/79.2%，CUPID F1 为 0.260/0.338/0.374，PERMA 均为 2/3；新门控 BenchPreS 适用召回由 31/34 提高至 32/34，误用均为 0/66。原文证据相对 v5 的主要收益置信区间跨零，且开发组 PrefEval/PERMA 有回退，因此 v6 仍为实验变体。当时默认 v5 保持用户选择。

源码与样本划分核验通过，历史 1,102 项结果及续跑前 288 项有效检查点均保持不变；375 项成功检查点在拒绝网络请求的 mock 下回放通过，新增 HTTP 请求为零。这验证断点跳过行为，不是重新生成或独立裁判评分。

## M4：Qwen2.5-7B 模型对照

[冻结方案](runs/spm/m4_plan.json)使用 M3 相同的 59 个测试问题，运行 v4/v5/v6 共 177 项。抽取、关系、解析、门控、推断和生成全部由 `Qwen/Qwen2.5-7B-Instruct` 执行；裁判保留 DeepSeek-V3.2，嵌入仍为 BGE-M3。Qwen 从原始历史重建独立记忆库，三版本共享写入；不会复用 DeepSeek 的抽取和关系标签。此处为已观察问题上的模型对照，不是新的未见测试集，也没有 Qwen b1/b2 基线。

针对用户提供的 80,000 TPM，增加按模型共享的持久滚动限速：预留分词后的输入、最大输出与余量，已知 provider 用量返回后调整，完成后的用量保留 60 秒。有效上限为 72,000 TPM，HTTP 并发 4，任务 worker 12；重试计入，缓存命中不计入。限速协调只覆盖同一缓存数据库，外部同账户请求仍可能造成 429。

```bash
uv run python scripts/spm_m4.py run
# 停止执行后分析并回放检查点
uv run python scripts/spm_m4.py analyze
uv run python scripts/spm_m4_replay.py
uv run python scripts/spm_m4_report.py
```

通用 CLI 也支持 `--model-tpm 'Qwen/Qwen2.5-7B-Instruct=80000' --tpm-tokenizer datasets/tokenizers/qwen2.5-7b/tokenizer.json`，应配合独立输出目录。未传 tokenizer 时以 UTF-8 字节数保守估计输入；此参数不自动切换模型，模型仍由 `.env` 或进程环境变量决定。不要并行启动指向同一输出目录的实验；本轮 API 缓存与旧轮次隔离。

[M4 结果报告](SPM_M4_RESULTS.md)：首轮与一次有限重试后，177 项计划均已执行，86 项有效、91 项校验失败。首轮并发 4、续跑并发 8，滚动预算保持 72,000 TPM；已知 Qwen 用量的 60 秒窗口峰值 59,942 token，无 HTTP 429。58 项测试通过，历史 1,477 个结果文件与首轮 81 个有效结果保持不变。

PrefEval 三版本共同成功 18/24 题，v4/v5/v6 遵循率为 2/18、6/18、7/18。BenchPreS 共同成功 8/20 题（四个画像），适用召回均为 7/12，误用为 1/28、0/28、0/28；v6 在此禁用原文。CUPID 仅两个共同有效问题、F1 均为零，PERMA 无共同有效问题，不能做完整基准结论。最终 91 项失败包括抽取 45、规范化 36、关系预分类 6、门控 4，数量包含三版本共享的写入失败。缺失项不按答错计入，完整指标保持 null。

## M5：Qwen 原文绑定接口与后续默认

[M5 报告](SPM_M5_RESULTS.md)包含固定 59 题诊断及 36 题未见问题组测试。两个阶段在首次预测前同时冻结；保留所有 190 项检查点，176 项有效，14 项最终校验缺失。一次有界续跑只重试失败项，168 个此前成功结果逐字节不变。

USER 原文分为不改写的句子/换行片段（长片段最多 600 字符），模型选择 source_id，宿主绑定已有原文与 turn_idx。给定偏好逐条规范化并绑定完整原文。关系与门控使用调用内整数序号，仍要求每项恰好一次，并保留实体与冲突保护。此方式保证引用来源对应，不能保证偏好推断语义正确。

新默认是 v6_compact；v6 保留原接口定义。v6_compact_no_raw 与默认共享写入与门控，只关闭回答/推断器的原文补充。写入签名包含 tool_protocol，所以不能混用旧接口的库。Qwen CLI 与直接 Gateway SDK 调用默认均启用限速；已有科学实验继续从各自冻结目录重现。

新 PrefEval 24 题全部有效，无原文 / 原文分别为 8/24 与 9/24，差值区间跨零。CUPID 共同成功 9/12 题的 F1 为 0 与 0.171，差值区间 [0.042, 0.338]；剩余三题属于门控执行缺失，不能当作官方全量成绩。

```bash
# 常规新实验：默认 Qwen + v6_compact，使用新目录
uv run python -m spm run --benchmark prefeval --split dev --limit 3 \
  --workers 8 --concurrency 8 --output runs/spm/qwen_v6_compact_smoke

# 严格重现已冻结的 M5：自动跳过成功项；run 会再次重试失败项
uv run python scripts/spm_m5.py analyze
uv run python scripts/spm_m5_replay.py
uv run python scripts/spm_m5_report.py
```

[默认提升记录](runs/spm/m5_default_promotion.json)记录实验完成后的默认切换及 64 项测试；M5 预测仍使用未改动的冻结源码。没有新增强化学习。

## M6：图管理增量对照

[实验方案](SPM_M6_PROTOCOL.md)、[已核验结果](SPM_M6_RESULTS.md)和[开发诊断](SPM_M6_DIAGNOSTICS.md)单独保留。M6 比较完整历史、与 v6 同窗口/预算的原文 RAG、平铺观察与图管理；系统各角色仍为 Qwen，评审及嵌入模型不变。

| 方法 | 行为 |
| --- | --- |
| `v6_m6` | v6_compact 图管理，保留合法批量门控评分，针对遗漏候选补问 scalar 概率，最终严格覆盖全部候选 |
| `flat_m6` | 相同初始抽取、条件、检索、门控与原文补充，追加观察，不执行关系分类、语义合并、冲突边及时间替换 |
| `b2_matched` | 相同 flagged 原文窗口、查询扩展、BGE 排序、最多 3 窗口和 6,000 片段字符，省去偏好抽取 |
| `b1` | 完整公开历史；超过配置预算时记录执行缺失，不静默截断 |

M6 新测试仅 PrefEval/CUPID。`flat_m6` 的类选择由实验 runner 完成；直接调用 SDK 时应使用 `spm.baselines.FlatMemory`，而不是只给 `SPM` 传入 flat 配置。常规默认仍为 M5 的 `v6_compact`，M6 没有自动改默认策略。

开发核验发现并修复了两个 API 并发问题：TPM 等待移到 HTTP 槽之前，避免阻塞 embedding/裁判；单文本 embedding 向量首次提交后保持不变，避免重叠批次覆盖和方法间数值不一致。71 项测试通过。开发原冻结包及评分不修改，主测试的唯一源码差异为 api.py；同源抽取及原文控制由离线审计验证。

```bash
# 查看冻结阶段的正式指标；无新增 API 请求
uv run python scripts/spm_m6.py analyze --phase test
uv run python scripts/spm_m6_replay.py --phase test
uv run python scripts/spm_m6_report.py

# 常规新实验使用新目录，显式选择四方法
uv run python -m spm run --benchmark cupid --split dev --limit 3 \
  --methods b1 b2_matched flat_m6 v6_m6 --workers 16 --concurrency 8 \
  --output runs/spm/my_m6_comparison
```

图和平铺初始抽取及向量逐字段核验，但管理后候选和最终上下文长度可以不同。完整历史及 RAG 不是与图版总输入 token 完全相同的对照。API 实际调用归属受缓存共享影响，案例时延包含 TPM 等待，不能直接解释为独立冷启动成本或无负载延时。

## M7：保守图更新

`v6_m7` 在 `v6_m6` 之上只改变写入策略。潜在 same/contradicts 关系由同一 Qwen 模型批量核对对象、维度及 USER 原始证据；互补观察并存，明确修订且时间先后合法时才使旧节点失效。弱冲突保留 pending，重复两次不自动淘汰旧节点。来源及核对理由进入 `update_check` 事件。

[预先方案](SPM_M7_PROTOCOL.md)包含新预留开发/测试组、预算与控制；两阶段在首次预测前冻结，旧实验保持不变，常规默认仍为 `v6_compact`。82 项实现测试通过。保守写入签名单列；legacy 签名不变。

[M7 已核验结果](SPM_M7_RESULTS.md)：248/270 项有效，主测试 194/216；主比较未确认稳定质量增益，保守版接近平铺、成功样本无实际时间替换。当前不提升为默认。[诊断](SPM_M7_DIAGNOSTICS.md)记录误替换修复路径及过于保守的可能性；接口错误和 consistent 层缺失会限制结论。

CLI 的 `--methods v6_m7` 自动选择 `ConservativeMemory`。直接 SDK 使用：

```python
from spm.config import VARIANTS
from spm.conservative import ConservativeMemory

memory = ConservativeMemory(user_id, store, api, cfg=VARIANTS['v6_m7'])
```

`SPM` 本体拒绝保守配置，避免只改配置而实际执行旧更新逻辑。M7 运行与离线核验：

```bash
uv run python scripts/spm_m7_execute.py
uv run python scripts/spm_m7.py analyze --phase test
uv run python scripts/spm_m7_replay.py --phase test
uv run python scripts/spm_m7_report.py
```

核对仍可能语义误判，额外成本和执行缺失会单列；这次比较整套保守策略，未拆分每条规则的作用，也不声称额外核对等于独立正确标签。
