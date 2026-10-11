# LoCoMo 的 F1、B1 和 J

现有问答模型保持 `.env` 中的设置。评测模型单独默认使用
`Pro/deepseek-ai/DeepSeek-V3.2`，复用硅基流动凭据。无需重新运行问答或加载 BGE-M3。
默认 judge 模板现已改为用户提供的 Memory-R1 原文；此前严格版本继续保留作对照。

```bash
uv run python -m graphmem.evaluate_locomo --run runs/locomo_bge12 \
  --judge-rubric memory-r1 --judge-model Pro/deepseek-ai/DeepSeek-V3.2 --workers 1
```

运行中断或 judge 失败后，以相同参数加 `--resume`。已成功的评分不会重新调用 API。
新一轮独立评测应指定新目录，例如 `--output runs/locomo_bge12/evaluations/judge_repeat5`。
`--repeats 5` 对每个不同的问答三元组进行五次独立请求，先平均各题，再平均各模式。
每次重复有独立缓存；temperature 固定为 0，重复评分仍可能相同。
`--no-judge --output runs/locomo_bge12/evaluations/offline` 可离线计算 F1/B1。

## 评分规则

- **F1**：保留原始 LoCoMo evaluator 的类别规则、词干化和归一化。本次增补没有修改旧 F1。
- **B1**：NLTK sentence BLEU-1，权重 `(1.0,)`，包含 clipped unigram precision 和 brevity penalty。
  每题先计算，再做题目平均。使用 LoCoMo 归一化（小写、删除标点、去除 a/an/the/and、空格分词），
  不做词干化。类别 3 的参考答案只取第一个分号段；多跳列表按完整答案计算 BLEU，
  不套用 F1 的逐项最大匹配。空答案或空 token 列表计 0。
- **J（默认 memory-r1）**：原样使用用户提供的 Memory-R1 模板，问题、完整 gold 和生成答案
  填入三个占位符，作为一条 user 消息发送，使用 JSON mode。
  采用宽松同主题评分，允许较长答案、同义表达以及指向相同时段的相对日期。
  不附加列表完整性、精确日期核对、无锚点自动判错等严格规则。
  返回 `{"label":"CORRECT"}` 记 1，`{"label":"WRONG"}` 记 0，然后按题平均。
  模板同时要求短解释和 JSON label：兼容仅 label 的 JSON 和带 explanation/reason 的 JSON，
  有说明则保存；没有说明不强迫模型补写。原始输出也会保留。不合规标签不计成 WRONG。
- **J（可选 strict）**：使用 `--judge-rubric strict` 选择此前严格的二元语义规则。
  允许同义表达、等价日期和简洁答案；完整列举题要求覆盖所有必要项，含实质错误或缺项计 0。
  Judge 接收问题、完整参考答案、生成答案，不接收模式名、图轨迹、检索证据或分数。
  类别 3 的分号后说明可以省略。无日期锚点时不自行发明锚点。

以下日期限制只适用于 `strict`：精确日期需逐一核对年月日，差一天不视为等价格式。参考为绝对日期而回答使用无锚点的相对时间时，
J 计 0，因为无法仅凭评测输入验证等价性；judge 不会收到原始会话来补齐日期锚点。
严格版协议为 `caps-locomo-normalized-bleu1-strict-binary-judge-v2`。
当前默认协议为 `caps-locomo-normalized-bleu1-memory-r1-user-template-v1`。
文献中的 B1/J 不是完全统一的协议：[Mem0 原评测](https://github.com/mem0ai/mem0/blob/393a4fd5a6cfeb754857a2229726f567a9fadf36/evaluation/metrics/llm_judge.py)
使用 GPT-4o-mini 和较宽松的同主题判断；当前默认使用用户提供的模板及指定的 DeepSeek。
因此适合比较本实验内的三个方法，不能把数值直接与论文榜单对比。
BLEU 实现见 [NLTK 文档](https://www.nltk.org/api/nltk.translate.bleu_score.html)。

## 记录、缓存和用量

默认保存到 `<run>/evaluations/deepseek_v32_memory_r1/`；新运行的 `strict` 默认保存到
`<run>/evaluations/deepseek_v32_strict_v2/`，历史严格版目录仍为 `deepseek_v32_v2/`：

- `manifest.json`：数据、原始 gold、原始 episode、评分代码、judge 模型/参数/提示词的哈希。
- `judge_prompt.txt`：实际使用的完整规则。
- `judgments/*.json`：输入、二元判断、理由、模型实际返回名称、原始 JSON、每次 HTTP 尝试和 usage。
- `scores.json`：每个模式每题的 F1/B1/J 与 judge 缓存键。
- `summary.json` / `report.md`：模式与类别汇总、覆盖率、独立评分数量、用量。
- `source_snapshot/graphmem/`：冻结的评分实现。

首轮 v1 保存在 `runs/locomo_bge12/evaluations/deepseek_v32/`。审阅全部 24 条判断理由发现
judge 将 January 5 与 January 6 判成相同日期，因此保留原分数并记录在 `manual_audit.json`，
随后增加通用的日期精度核对规则开展独立 v2 复评。这是探索性评测中一次公开的规则修订；
没有按某个方法的结果修改答案、参考答案或逐条分数。两轮用量分别记录，不能只计 v2。

J 表示“模型判为正确的比例”，并不等于经人工确认的准确率。v2 仍可能对相对时间采用不同的严格程度：
例如参考为 `Last week before 13 October 2022.`、回答为 `last week`，v1 给 1，v2 给 0。
本项目保留两轮原始判断，并在审计中标出该例；扩大实验前应固定协议并抽查时间题，
不要根据希望得到的系统排名手动改分。

相同问题、参考答案和预测答案共享同一次评分，避免模式身份带来的差异并减少 API 调用；
同一题各模式的统计分母仍各计一次。12 题 × 3 模式的历史预测共 36 条，对应 24 个不同三元组。
失败或不合规的 judge JSON 不会被当成错误答案；只要某组有未评分题，该组 J 就保持 pending，
同时显示覆盖率。续跑保留失败尝试及已知用量；没有返回 usage 的请求不能计算确切用量。

默认并发 1，可用 `--workers` 调整。TPM 配额和请求并发是不同约束；遇到 429/5xx 会重试，
最多四次，并处理数字格式的 Retry-After。Judge 使用非思考模式、512 输出 tokens，参数可调。
API 密钥不会进入 manifest 或报告。可在 `.env` 添加 `JUDGE_API_KEY` / `JUDGE_BASE_URL`
单独指定 judge 的服务；空值复用问答服务，问答模型不受影响。

原始 `episodes/`、gold、manifest 和问答报告保持原样。新评测校验完整样本及原始 F1，
严格续跑需要相同评分配置和输入文件。原始 QA 的 `--resume` 校验整个 `graphmem` 代码哈希，
新增评测代码后应使用冻结的 QA 代码续跑旧实验，或为新 QA 实验使用新目录。
各 judge 模板的 protocol、提示词哈希、代码哈希和缓存分别记录，不会复用另一种模板的评分。
历史完整评分无需续跑；重新评估严格规则时使用 `--judge-rubric strict` 的新目录。
