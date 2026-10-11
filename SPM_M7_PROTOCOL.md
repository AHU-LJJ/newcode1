# SPM M7：保守图更新预先方案

用户授权继续验证保守更新。主问题：M6 所见互补偏好误替换能否通过对象/维度核对和明确修订证据要求缓解，同时保留真实时间更新能力？M6 冻结包与样本不改写；默认仍为 `v6_compact`，不引入强化学习。

## 对照与冻结

三种方法：`flat_m6` 追加观察；`v6_m6` 原图；`v6_m7` 保守图。系统各角色均为 `Qwen/Qwen2.5-7B-Instruct`，裁判 `Pro/deepseek-ai/DeepSeek-V3.2`，嵌入 `BAAI/bge-m3` API。抽取、条件字段、窗口、BGE、原文检索、reader、门控、生成器及裁判不变。

开发 12 PrefEval + 6 CUPID，测试 48 PrefEval + 24 CUPID，共 270 方法项。每个历史未见问题组只取一题，排除旧实验及已有预留组，公共历史最多 24 episodes。seed 20261014，原生 strata 轮转抽样，不依据 gold、模型输出或分数筛题。两阶段成员、源码、配置及模型在首次 M7 API 预测前同时冻结；开发只作运行/成本/来源核验，本轮不按开发成绩调参。若发生需要改源码的执行问题，先停止并记录，不能静默更改测试方案。

## 保守写入规则

保留旧关系分类器和候选选择。只对相同条件、初判 same/contradicts 的潜在破坏性关系增加一次同模型结构化核对，按最多 12 对批量、跨 episode 预计算与缓存，重复核对的 payload 合并。核对提供语义字段与最近两条原始 USER 引文，不读取 gold 或当前测试裁判。

- 语义合并需要同一具体对象、同一维度、整个偏好等价且 polarity 一致；仅同类别、部分重叠或额外细节不合并。完全相同 canonical key 的观察仍合并证据，不丢弃其文字细节。
- 冲突需要同一对象/维度且不能同时满足。误判或不确定的潜在冲突按兼容观察保留，互补正负偏好并存。
- 旧观察失效需要明确修订、互斥关系、当前 episode 的原始 USER 引文，以及宿主校验的时间先后。引文必须是模型选中的 NEW source 的连续字面子串，最多 200 字符；同一 turn 中排序产生的先后不能触发替换，同 episode 的更晚 USER turn 可以触发。
- 已确认矛盾但缺乏明确修订证据时，新观察为 pending、保留旧节点并保留挑战边；沿用现有 reader 冲突保护。仅重复两次不再自动令旧节点失效。若只确认部分旧节点可替换，其他冲突保持 pending。

每次用到核对结果记录 `update_check` 事件、初始/有效关系、对象/维度判定、引文及理由。要求完整唯一的整数索引、合法严格布尔值、修订证据来源；非法输出有限 schema 修复后仍失败则记执行缺失，不能静默设为兼容或依赖强模型代写。

这次同时限制合并、替换与 pending 自动晋升，测的是策略整体。核对仍可能语义误判；字面来源正确不代表解释正确。严格修订要求可能漏掉真实但隐含的偏好变化，需报告 CUPID changing 层及 pending 数。

## 预算与运行

HTTP 并发 8，worker 16；Qwen 80,000 TPM 按 72,000 TPM 准入，使用本地 Qwen tokenizer。开发上限 2,000 次 HTTP / 4,000,000 tokens，测试上限 5,000 / 10,000,000。每阶段最多一次失败项续跑（800 / 2,000,000），成功输出逐字节保护。预算上限是停止条件，不代表实际消费目标；实际 provider 已知/未知用量分别报告。额度中断保留冻结方案和断点，不重抽样。

隔离 `api_cache_m7.sqlite` 从已验证 M6 缓存备份，共享相同 payload 的抽取、向量和关系请求；分别建新 bank，保守更新写签名单列，旧 legacy 签名保持一致。方法归属调用含共享/缓存影响，不是独立冷启动成本；时延含 TPM 等待。

## 指标与核验

主比较 `v6_m7 vs v6_m6`：PrefEval 遵循率、CUPID checklist F1，逐题配对、按独立组 bootstrap 95% 区间。`vs flat_m6`、回答质量、CUPID changing/consistent/contrastive、完成率和成本为次要或探索性，无多重比较校正。另列三方法共同有效结果，不把缺失计为答错，不人工代评；不完整基准的全量指标为空。

来源审计核对初始抽取/证据/向量完全相同，原文检索片段/顺序/分数/截取及查询分析一致；审计每个 supersede 的明确修订事件和 USER 字面来源。报告单位保留、候选/apply 数、上下文字符、拒绝的破坏性关系、merge/replace/pending，以及 HTTP 和 tokens。最后用已验证缓存完整重建成功上下文/trace 和原生成/评分，HTTP 必须为零；旧样本及首轮成功样本摘要不变。这不是独立重评，不证明语义核对正确，也不代表官方全量成绩。

## 复现

```bash
uv run python scripts/spm_m7.py prepare
uv run python scripts/spm_m7_execute.py
uv run python scripts/spm_m7.py analyze --phase test
uv run python scripts/spm_m7_replay.py --phase test
uv run python scripts/spm_m7_report.py
```

SDK 使用 `spm.conservative.ConservativeMemory(..., cfg=VARIANTS['v6_m7'])`；CLI 自动选择管理器。`SPM` 本体拒绝保守配置，避免误用成旧更新行为。82 项实现测试在冻结前通过，含误判合并/冲突、真实更新、重复弱冲突、非法引文和时间顺序保护。
