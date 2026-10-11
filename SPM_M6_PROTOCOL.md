# SPM M6 实验方案

M6 主问题：在相同初始抽取、读入模型及原文补充条件下，关系分类、合并、冲突保护与时间更新是否比平铺观察记录带来额外收益？实验任务已由用户授权。旧 M5 冻结包、样本及评分保持不变。

系统各角色使用 `Qwen/Qwen2.5-7B-Instruct`，评审为 `Pro/deepseek-ai/DeepSeek-V3.2`，嵌入为 `BAAI/bge-m3` API。Qwen 80,000 TPM 配额按 72,000 TPM 预留，HTTP 并发 8，worker 16，所有阶段串行，共用隔离的 `api_cache_m6.sqlite` 限速状态。

开发执行中发现 Qwen 等待 TPM 时占据全局 HTTP 槽，导致未受同配额限制的 embedding/裁判也等待。主测试冻结前仅调整 `api.py` 调度顺序：先取得 TPM 准入，再取得 HTTP 槽；真正 HTTP 并发仍最多 8，配额及请求 payload 不变。等待 HTTP 槽期间取消的未发送请求会退还预留。独立测试验证限速等待不阻塞裁判。dev 原冻结包不修改；测试冻结审查记录源码差异，不根据开发分数改提示或筛题。

开发离线核验另外发现，重叠 embedding 批次会为同一文本生成略有差异的向量，并覆盖单文本缓存。dev 中图/平铺的原始抽取与原文分数相同；RAG 选中的原文、顺序和截取相同，但部分相似度分数不同。保留原输出，审计具体差值，不声称 dev 的 RAG 数值输入完全相同。主测试冻结前在 `api.py` 改为单文本向量首次提交后保持不变，后完成的批次读取同一已提交向量；独立重叠批次测试验证。这不是根据成绩改模型或提示。dev 的离线回放验证保存上下文对应的生成与评分缓存，不能从已覆盖的向量重建原相似度；主测试要求完整上下文/trace 重建。

## 阶段

1. reliability：M5 中四个发生门控遗漏的案例，仅使用 `v6_m6`，检验执行修复，不作为新测试效果。
2. dev：12 个 PrefEval、6 个 CUPID 新问题组，四种方法共 72 项。用于执行及成本检查。
3. test：预留 48 个 PrefEval、24 个 CUPID 新问题组，四种方法共 288 项。开发运行结束后冻结实现和模型。测试成员及对照在任何 M6 预测前已预留，开发和测试组不重叠。

采用 seed 20261013，按原生 strata 轮转随机抽样，每个未见组一个问题，不根据答案或成绩排除案例。CUPID 为 contrastive / changing / consistent；PrefEval 为 explicit / choice / persona。每阶段允许一次有预算上限的执行失败续跑，成功答案不重新生成。API 额度中断须按已冻结方案续跑，不能重新抽样。

## 方法

| 方法 | 输入及管理 |
| --- | --- |
| b1 | 完整公开历史；不静默截断，超预算为执行缺失 |
| b2_matched | 与 v6 同样的 flagged 原文窗口、BGE 检索、查询扩展、排序规则、最多 3 个窗口和 6,000 个片段字符；不抽取偏好 |
| flat_m6 | 与图版完全相同的初始抽取及条件字段；追加观察，不做语义合并、关系判断、冲突边、时间替换；采用相同检索、门控与生成器，并补充同样原文 |
| v6_m6 | v6_compact 图管理，加下述门控补齐；原文预算保持不变 |

Flat 与 graph 通过 API 已验证缓存共享初始抽取结果，但构建独立 SQLite 记忆库。`extractions/` 保存管理器无关的原始条目、证据与向量摘要；不同管理器的初始抽取必须逐字段、逐向量一致。flat 仍包含条件字段和适用性门控，本对照只测管理机制的增量，不能声称测量了整个 CAPS 思想或所有条件建模的价值。完全相同 ID 的同一 episode 观察按原 ID 保持唯一，跨 episode 不合并。

原文 RAG 预算与图版一致，但图版另外包含结构化偏好，因此图 vs 原文不是总输入 token 完全匹配的对照。完整历史的信息量和输入成本也不同，须同时报告上下文长度，不将其作为等成本实验。

## 门控补齐

批量响应必须是完整、可解析的 schema；重复/未知/非整数索引、非法概率、截断响应仍触发严格校验和有限修复。仅完整 schema 中的遗漏可以补齐：保留全部合法已返回决策，每个遗漏候选单独请求一个无索引的 `p_apply/reason`，宿主绑定该候选。单项补问保留原候选集合与原文提示以评估冲突。最后要求每候选恰好一次，并保留实体不匹配及 pending 冲突保护。不根据 gold 补分，不自动填零。

批量门控提示及阈值保持 M5 定义。scalar 补问最多 1,024 输出 token，内部最多三次 schema 尝试；失败仍为缺失。trace 的 `gate_coverage` 记录初始返回数和补齐索引，便于报告修复率和额外成本。

## 指标及归因

主比较：v6_m6 vs flat_m6；PrefEval 遵循率、CUPID checklist F1。回答质量和 vs b1/b2 为次比较。按问题对齐，并按独立组 bootstrap 95% 区间；四方法共同有效子集另列。CUPID 三种原生 strata 的比较为探索性，不作多重比较校正。执行缺失单列，完整成绩为空，不能将共同有效子集当作全量。

原生基准之外不添加构造题混入成绩。本轮先用 CUPID changing / contrastive 场景观察图管理的变化和作用域能力；没有未见 BenchPreS 画像或当前资源上限内的未见 PERMA 用户，因此主测试仅 PrefEval / CUPID。

同时报告完成数、门控补齐数、上下文字符、API 调用、已知/未知 provider 用量、HTTP 状态和时延。`experiment_context` 将 HTTP 尝试关联到最先实际触发调用的方法和阶段；缓存及抽取共享影响各方法成本分摊，不能把实际归属成本当作各方法独立冷启动成本。并发案例时延包含 TPM 等待，不当作无负载服务时间。

评审沿用已有 API 代理协议，不是官方 PrefMatcher。证据审计只核实 USER 原文来源，不证明语义抽取正确。小样本、缺失非随机和代理裁判限制结论；不以显著性或开发成绩筛选主测试。

## 复现

```bash
uv run python scripts/spm_m6.py prepare-dev
uv run python scripts/spm_m6.py run --phase reliability
uv run python scripts/spm_m6.py analyze --phase reliability
uv run python scripts/spm_m6.py run --phase dev
uv run python scripts/spm_m6.py analyze --phase dev
uv run python scripts/spm_m6.py prepare-test
uv run python scripts/spm_m6.py run --phase test
uv run python scripts/spm_m6.py analyze --phase test
```

`prepare-*` 拒绝覆盖已有冻结方案。`run --resume` 为一次有上限的失败续跑，存档首轮错误和成功样本摘要。每阶段预测通过自己的 frozen 包运行；离线 analyze 验证源码、模型、数据、同源抽取、同原文控制和旧样本摘要，不产生 API 请求。
