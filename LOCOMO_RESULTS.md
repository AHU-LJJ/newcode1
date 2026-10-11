# LoCoMo + BGE-M3：首次对照结果

日期：2026-10-09。Embedding 使用本地 `BAAI/bge-m3`，问答使用 `.env` 中的 `Pro/Qwen/Qwen2.5-7B-Instruct`。未进行 SFT 或 RL。

## 当前默认：用户提供的 Memory-R1 Judge 模板

使用用户提供的 Memory-R1 原文对同一批答案重新评分，judge 仍为
`Pro/deepseek-ai/DeepSeek-V3.2`，temperature 0、非思考模式，每个不同三元组评分一次。
问答模型、预测和 F1/B1 均保持不变。现在默认采用这套宽松同主题评分：

| 模式 | F1 | B1 | J：Memory-R1 模板 | 历史 strict v2 J |
| --- | ---: | ---: | ---: | ---: |
| 普通向量检索 | 48.66% | 39.97% | 58.33% | 33.33% |
| 只读图 | 57.57% | 46.13% | 75.00% | 33.33% |
| 可管理图 | 48.26% | 37.33% | 66.67% | 25.00% |

模板原文填入 question/gold_answer/generated_answer 后直接作为 user 消息发送。
JSON label 为 CORRECT 时记 1，WRONG 时记 0，不附加严格规则。
例如参考包含饮品和巧克力、预测只提饮品时，本轮按同主题口径判 CORRECT，历史严格版因缺项判错；
`last week` 相对时间也获得较宽松的判断。这些分数差异来自评测规则及独立模型判断，
不能解读为问答系统性能在此期间改善。

36/36 条答案完成评分覆盖，复用了 24 个不同三元组。共 28 次 HTTP 尝试，
其中 4 次 429 在重试后恢复；已知用量至少 8,911 tokens，失败请求的 usage 未返回。
模型实际返回了仅含 label 的 JSON，因此本轮没有逐条自然语言理由；原始输出全部保存。
21 项测试通过，缓存续跑验证为 0 次新增 API 请求。

样本仍仅为 12 题。Judge 模型、样本及 B1 实现需一致才能与论文数值对比；这里对齐的是用户提供的 judge 提示词。
历史严格版保留为对照，完整规则和命令见 [评分说明](LOCOMO_EVALUATION.md)。

- [当前默认完整报告](runs/locomo_bge12/evaluations/deepseek_v32_memory_r1/report.md)
- [原文提示词](runs/locomo_bge12/evaluations/deepseek_v32_memory_r1/judge_prompt.txt)
- [逐题评分变化](runs/locomo_bge12/evaluations/deepseek_v32_memory_r1/rubric_comparison.json)
- [验证记录](runs/locomo_bge12/evaluations/deepseek_v32_memory_r1/verification.json)
- [所有 judge 轮次用量](runs/locomo_bge12/evaluations/judge_costs_all.json)

## 设置与验证

从 `datasets/locomo/data/locomo10.json` 的类别 1–4 均衡抽取 12 题，每类 3 题，seed 42、offset 0。涉及 6 个完整会话、3,784 个原始对话 turn。每题从对应会话全部原文建图，不使用随数据附带的答案、观察、事件摘要或会话摘要；提供的图片 caption 作为文字描述。

比较普通向量检索、只读图和可管理图，检索均使用相同 BGE-M3 dense embedding。每题/模式最多 12 次模型调用、32 次工具调用，temperature 0。不跨问题共享模型生成的图事实。

36/36 个流程最终成功提交。所有答案分数逐一与本地 LoCoMo 原始 evaluator 核对一致，所有最终引用均指向已经读取的对话。离线检查共 14 项测试通过，包含官方评分对照、API 响应、缓存、修订/撤回后的检索、标签隔离及续跑统计保护。

## 本地与 API embedding

- 官方模型 revision：`5617a9f61b028005a4858fdac845db406aefb181`，本地 `models/bge-m3`，约 2.3 GB 权重。
- RTX 5060 Ti 上 FP16 推理，CLS pooling，1024 维归一化向量；本轮使用 dense 输出。
- 对实际抽样会话的 3,968 条源节点文本进行独立冷缓存索引测试：包括写缓存的耗时 **12.67 秒**，实际编码累计 9.20 秒，PyTorch 峰值分配约 **1.20 GB** 显存。该数值不包含模型加载、Windows 显示占用或问答 API；详见 [index_benchmark.json](runs/locomo_bge12/index_benchmark.json)。
- 硅基流动 `BAAI/bge-m3` API 也通过两条短文本的真实请求验证，返回 1024 维向量，usage 为 19 tokens；本轮 QA 全部使用本地 embedding。

## 问答结果

| 模式 | LoCoMo 答案分数/F1 | 辅助证据 F1 | Chat 模型调用数 | 有效完成尝试累计 Chat tokens |
| --- | ---: | ---: | ---: | ---: |
| 普通向量检索 | 48.7% | 46.5% | 29 | 104,177 |
| 只读图 | 57.6% | 54.4% | 35 | 178,978 |
| 可管理图 | 48.3% | 56.7% | 57 | 344,407 |

答案分数使用 LoCoMo 的类别相关规则，并非 HotpotQA F1。证据 F1 是额外辅助指标，不能与官方主指标混用。本批没有无法解析的 gold 证据。

## 补充 B1 与 DeepSeek Judge 评测

对同一批 36 条已保存答案补算 BLEU-1，并使用硅基流动
`Pro/deepseek-ai/DeepSeek-V3.2` 作为独立 judge。问答没有重跑，F1 保持不变。
采用严格二元语义评分 v2，每个不同的问答三元组评分一次；24 个不同三元组覆盖三个方法的全部 36 条答案。

| 模式 | F1 | B1 | J：strict v2 |
| --- | ---: | ---: | ---: |
| 普通向量检索 | 48.66% | 39.97% | 33.33% |
| 只读图 | 57.57% | 46.13% | 33.33% |
| 可管理图 | 48.26% | 37.33% | 25.00% |

B1 包含 unigram clipping 和长度惩罚，按题平均；归一化、列表及类别 3 处理见
[评分协议](LOCOMO_EVALUATION.md)。J 是模型判为正确的比例，要求列举项完整、精确日期一致。
本项目的 judge 提示词和模型不同于论文评测设置，不应直接对比榜单。

首次 v1 的 J 分别为 41.67%、58.33%、41.67%。审阅判断理由发现，judge 把参考
January 5 与回答 January 6 判成等价日期。因此保留首轮，增加通用日期精度核对规则后独立复评，
v2 已纠正该例。两轮对 `last week` 等相对时间仍有严格程度差异；v2 时间类 J 均为 0，
需结合无日期锚点时的评分规则理解，不能将其等同于人工确认的时间题准确率。
所有数字均为原始 judge 输出的聚合，没有手工改分。这次规则修订是公开的探索性评测调整。

v2 出现 429/连接超时后，以单并发仅续跑两个失败项，已完成 36/36 条评分覆盖。
两轮共 68 次 HTTP 尝试，已知用量至少 **15,808 tokens**；失败请求缺少的 usage 无法确定。
19 项离线测试通过，真实缓存的 no-op resume 验证为 0 次新增 API 请求。

- [v2 完整报告](runs/locomo_bge12/evaluations/deepseek_v32_v2/report.md)
- [两轮审计记录](runs/locomo_bge12/evaluations/deepseek_v32_v2/manual_audit.json)
- [新增评分验证](runs/locomo_bge12/evaluations/deepseek_v32_v2/verification.json)
- [两轮 judge 用量](runs/locomo_bge12/evaluations/judge_costs.json)

| 类别 | 每组题数 | 普通检索 | 只读图 | 可管理图 |
| --- | ---: | ---: | ---: | ---: |
| 1：多跳 | 3 | 45.6% | 45.6% | 54.2% |
| 2：时间 | 3 | 24.1% | 55.6% | 38.9% |
| 3：常识/开放域 | 3 | 75.0% | 79.2% | 50.0% |
| 4：单跳 | 3 | 50.0% | 50.0% | 50.0% |

## 管理策略的实际行为

可管理图成功写入 **11 条事实**，但后续图搜索/展开结果中返回的派生事实为 **0**。57 次模型调用中，搜索原文 12 次、搜索图 13 次、展开图 8 次、新增事实 11 次、读取文档 1 次、finish 12 次；有些步骤一次调用多个工具，所以工具总数与模型调用数不同。本批没有 revise/retract 调用。

这说明当前策略会写入记忆，却没有在这个样本中通过工具重新取回这些事实。事实仍留在工具对话中，所以也不能说它们完全没有影响。管理图主分数与普通检索基本相同，用量约为其 3.31 倍；只读图在本批更好，时间题尤其明显。当前结果支持继续研究检索与图访问，但尚未显示主动图管理的整体优势。

样本只有 12 题、6 个会话。管理图相对普通检索的分数差值约 -0.4 个百分点，会话配对 bootstrap 95% 区间约 [-17.4, +18.1] 个百分点；相对只读图约 -9.3 个百分点，区间约 [-26.4, +6.9] 个百分点。不能由此推断训练后的结果或长期在线记忆效果。

## 限流、用量与复现记录

运行中出现过一次 HTTP 429 导致的失败 episode。原始失败轨迹保存在 `attempts/`，随后以至少 7 秒模型调用间隔续跑，已完成的 episode 未重跑。恢复后的 36 个完成结果是上表的统计对象。失败尝试另有 **17,140** 个已返回的 Chat tokens，因此已知总用量至少 **644,702** tokens；HTTP 失败没有返回的 usage 不可确定。[用量记录](runs/locomo_bge12/attempt_costs.json) 与 `transport_resume_*.json` 保留了这一区别。

检查中发现旧版本会在有新增工作的 resume 中覆盖初次 embedding 统计，运行后已修复并加入回归测试。该实验目录中的 `embedding_usage*.json` 对应续跑阶段，不能代表全程编码用量；因此单独保存了上述冷缓存索引基准，不把它冒充原实验全程耗时。问答评分与轨迹不受该统计问题影响。

本次实验对应的冻结代码、依赖和恢复脚本位于 `source_snapshot/`，与 manifest 哈希一致。当前代码包含运行后的统计修复，所以严格 resume 会拒绝旧代码哈希；重现实验请使用冻结代码或新输出目录。

- [自动报告](runs/locomo_bge12/report.md)
- [完整汇总与区间](runs/locomo_bge12/summary.json)
- [配置与模型 revision](runs/locomo_bge12/manifest.json)
- [逐项验证](runs/locomo_bge12/verification.json)
- `runs/locomo_bge12/episodes/`：图快照、消息、工具轨迹、原始 dialogue ID、预测和评分。
- [运行说明](GRAPH_MEMORY.md)
