# SPM M2：答案过滤门控的新身份组验证

2026-10-10。旧版 `v4` 保持默认；`v5` 仅添加通用的答案过滤适用性定义。生成器、检索、条件/实体规则及 0.5/0.2 阈值一致。本轮没有强化学习。

执行覆盖：开发 195/195、测试 154/156 项有效评分。未评分项单独保留，不按答错计分。

**新门控减少部分误抑制，但尚未稳定超过基线。** 新测试 PrefEval 为 v4 6/12、v5 7/12，低于完整历史 9/12 和原文 RAG 10/12。BenchPreS 的适用召回提高，CUPID 成功配对子集的 F1 则下降，因此保留 v4 为默认，不把 v5 宣称为总体更优版本。

## 冻结方案与适用范围

开发组沿用 M1 的 39 条样本，比较 v4/v5/v3/b1/b2，共 195 项；新测试组 39 条，比较 v4/v5/b1/b2，共 156 项。假设参考了 M1 失误，包括旧测试组，因此 M2 重新选测试身份。模型、样本、提示、阈值在新测试预测前冻结。

新测试有 29 个身份/问题组：PrefEval 12、CUPID 12、PERMA 3、BenchPreS 2。抽样按公共分层和历史长度进行，离线审核了全部先前样本组（61 个），没有与新测试重叠。BenchPreS 只有两个剩余画像，不具备广泛泛化证据。PERMA 为 sd_clean type2、三个不同用户的短历史子集，实际历史会话数为 2/8/12；没有截断历史。

v4 与 v5 共用写入数据库，使用相同候选记忆，改变的是门控提示。新定义问“偏好是否会改变、约束或过滤一个好答案”，不要求所有可选答案都违反偏好。裁判修复反馈增加短校验原因，仍严格拒绝矛盾标志；该技术改动用于全部方法，不追改 M1/M1R。

模型与裁判均为 DeepSeek-V3.2；嵌入为 BAAI/bge-m3 API。生成器和裁判同源，评分依然有偏差风险。PrefEval 无干扰历史；CUPID 是 API checklist 匹配而非官方 PrefMatcher；PERMA 是单步 MCQ；BenchPreS 是给定偏好门控，不是消息生成遵循率。

[预先固定方案](runs/spm/m2_plan.json)、[核验记录](runs/spm/m2_verification.json)、[实现及完整边界](SPM.md)。

## 新测试组结果

| 指标 | v4 | v5 | b1 | b2 |
| --- | ---: | ---: | ---: | ---: |
| PrefEval 遵循率 | 50.0% | 58.3% | 75.0% | 83.3% |
| CUPID checklist F1 | 未完整评分 | 未完整评分 | 0.368 | 0.410 |
| CUPID 回答质量（1–10） | 未完整评分 | 未完整评分 | 8.500 | 9.333 |
| PERMA MCQ | 66.7% | 100.0% | 100.0% | 100.0% |
| BenchPreS gate_AAR / gate_MR | 77.8% / 0.0% | 96.3% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% |

## 新门控相对旧门控的成对差值

| 基准 / 指标 | 配对问题 / 身份组 | v5−v4 | 分组 bootstrap 95% 区间 |
| --- | --- | ---: | --- |
| prefeval / follows_preference | 12 / 12 | +0.083 | [0.000, 0.250] |
| cupid / checklist_f1 | 11 / 11 | -0.049 | [-0.097, -0.012] |
| cupid / response_score | 11 / 11 | +0.091 | [-0.364, 0.727] |
| perma / mcq_accuracy | 3 / 3 | +0.333 | [0.000, 1.000] |
| benchpres / gate_AAR | 12 / 2 | +0.185 | [0.000, 0.278] |
| benchpres / gate_MR | 12 / 2 | +0.000 | [0.000, 0.000] |

区间按身份/问题组重采样；BenchPreS 比率按偏好标签 micro 汇总。小样本、三道 PERMA 和两个 BenchPreS 画像限制推断。零差值或全对不能证明稳定泛化，跨零区间不能解释为确定收益。

[测试指标](runs/spm/m2_test/metrics.json)、[对照旧门控](runs/spm/m2_test/comparisons_vs_v4.json)、[对照完整历史](runs/spm/m2_test/comparisons_vs_b1.json)、[对照原文 RAG](runs/spm/m2_test/comparisons_vs_b2.json)、[门控变化审计](runs/spm/m2_test/gate_audit.json)。

BenchPreS 共 27 个适用、33 个不适用标签：v4 应用 21/27，v5 应用 26/27；两者均没有应用不适用标签。召回差值 +0.185 的分组区间包含零，且仅两个画像，不能据此推断稳定泛化。开发组也出现相反方向的格式偏好误抑制。

## CUPID：相同成功样本的比较

v4/v5 各有一项执行失败，完整 12 条指标保持 null。以下四种方法均只比较相同的 11 条成功问题、11 个身份组，不与基线全 12 条均值混比。缺失未必随机，子集比较不能补回缺失结果。

| 方法 | 相同成功问题 | checklist F1 | 回答质量 |
| --- | ---: | ---: | ---: |
| v4 | 11 | 0.484 | 9.18 |
| v5 | 11 | 0.436 | 9.27 |
| b1 | 11 | 0.365 | 8.36 |
| b2 | 11 | 0.417 | 9.36 |

v5−v4 的 F1 为 −0.049，95% 区间 [−0.097, −0.012]；这说明更积极的门控并未自动改善偏好推断。v4−完整历史的成功子集 F1 为 +0.120 [0.016, 0.219]，但 v4−原文 RAG 为 +0.067 [−0.099, 0.243]。这些是小规模、同源裁判且有执行缺失的探索性比较，不能替代完整基准或多重比较控制。

## 失败定位

PrefEval 审计中既有候选作用域局限于原工作/交通情境、未应用到当前请求的情形，也有偏好已被应用而生成回答仍违反它的情形。某个 choice 历史还抽出了过强的排除偏好。下一步应优先验证证据抽取、作用域迁移和生成约束，而不是仅继续增大应用概率。裁判关于具体品牌/作品属性的理由未经独立核实，仅作为记录，不当作外部事实。

[失败审计](runs/spm/m2_test/error_audit.json)保留候选、决策、上下文、回答及记录的裁判理由。

## 开发组诊断

| 指标 | v4 | v5 | v3 | b1 | b2 |
| --- | ---: | ---: | ---: | ---: | ---: |
| PrefEval 遵循率 | 83.3% | 83.3% | 83.3% | 66.7% | 75.0% |
| CUPID checklist F1 | 0.312 | 0.308 | 0.331 | 0.343 | 0.416 |
| CUPID 回答质量（1–10） | 6.917 | 6.583 | 6.667 | 8.250 | 9.417 |
| PERMA MCQ | 66.7% | 100.0% | 100.0% | 66.7% | 66.7% |
| BenchPreS gate_AAR / gate_MR | 100.0% / 0.0% | 91.7% / 0.0% | 100.0% / 100.0% | 100.0% / 100.0% | 100.0% / 100.0% |

开发组的 v5 将一条 PERMA type3 从错误改为正确，但漏用了一个原本适用的加粗格式偏好；不能因此认为新提示总体更好。曾经发生裁判矛盾的运动偏好问题，在 M2 的 v4/v3/v5 中通过反馈修复获得有效评分；冻结 M1R 的三个未评分项仍保留原样。

[开发成对比较](runs/spm/m2_dev/comparisons_vs_v4.json)、[开发门控审计](runs/spm/m2_dev/gate_audit.json)。

## 执行失败与用量

以下项在预算内仍未获得有效执行结果；完整指标保持 null，比较仅使用同一成功样本对。

- test / cupid/506 / v4：{'type': 'APIError', 'message': 'extract: invalid model response after three schema attempts (units[0].span must be an exact continuous substring of supplied USER turn 6)'}
- test / cupid/506 / v5：{'type': 'APIError', 'message': 'extract: invalid model response after three schema attempts (units[0].span must be an exact continuous substring of supplied USER turn 6)'}

同一历史首次执行、重试及记录证据的诊断均失败。返回 span 将用户的“cross-referencing … printed urtext editions”改写为“verifying … at least two printed urtext editions”，不是所引用 USER 原文的连续子串。它是实质改写，不是标点格式差异。没有放宽验证或人工修正文句；[初次失败文件](runs/spm/m2_first_pass_errors)、[诊断记录](runs/spm/m2_extract_diagnostic.jsonl)、[失败队列](runs/spm/m2_pending.json)单独保留。

| 阶段 | HTTP 尝试 | 已知 provider token | 未知用量请求 |
| --- | ---: | ---: | ---: |
| dev | 48 | 140,057 | 0 |
| test | 866 | 2,023,929 | 0 |

M2 缓存从已验证的历史缓存复制到独立 SQLite；开发组复用旧记忆，测试组重新建库。表中只统计 M2 invocation 的请求，包含失败及重试，没有把复制来的历史账本重复计费。缓存条件不同，因此开发与测试的耗时不能作为独立冷启动比较。token 不是账单金额，未知用量不能当作免费。

50 项测试通过。旧门控提示逐字保留，原 M1 312 个检查点不变；离线分析没有 API 请求。graphmem 保持原进度。[成功检查点回放](runs/spm/m2_replay_verification.json)验证 349 个有效结果跳过，全部 351 个样本文件保持原样，HTTP 为 0；网络被失败 mock 替代，明确排除两个失败项。这验证检查点复现，不是重新生成或独立重评。

```bash
uv run python scripts/spm_m2.py run --phase dev
uv run python scripts/spm_m2.py run --phase test
uv run python scripts/spm_m2.py analyze
uv run python scripts/spm_m2_report.py
```
