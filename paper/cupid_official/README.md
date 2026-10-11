# CUPID 原始评分参考

这些文件来自 [kixlab/CUPID](https://github.com/kixlab/CUPID) 的固定版本 `a8560cab293ae98be4fe260689d58bddf96b51ef`，仅作为阅读和评分提示参考，不执行下载的 Python 代码。[source.json](source.json) 保存下载 URL 与 SHA-256，[LICENSE](LICENSE) 保留原 MIT 许可。

本地原论文 [CUPID.pdf](../CUPID.pdf) §2.3、附录 A.2（PDF 第 15–16 页）和图 16（第 29 页）定义生成任务的 preference alignment score：提供用户请求、模型回答、真实情境偏好及其 checklist，逐项分析回答满足偏好的程度，给出 1–10 分，只关注 checklist 内的要求。论文裁判是 temperature=0 的 `gpt-4o-2024-11-20`。

官方发布的 [response_judger.yaml](prompts/evaluation/response_judger.yaml) 比论文图 16 增加了各整数分值的文字标签及 Evaluation Label 段。补测使用这一完整发布提示，保留 Markdown 分析与末尾数值，不改写成 JSON，也不输入模型推断的 checklist。[response_judger.py](evaluation/modules/response_judger.py) 和 [evaluate.py](evaluation/pipeline/evaluate.py) 用于核对输入字段及原始评分流程。

本项目用硅基流动 `Pro/deepseek-ai/DeepSeek-V3.2` 替代论文裁判，仅补评 M8 已保存回答；不改变原生成流程、F1 或旧代理 response_score。原实验执行失败保持缺失，不沿用上游把失败置为 0 的做法，以免将执行能力与回答偏好对齐混为一谈。

可复现的本地补测脚本为 [spm_m8_alignment.py](../../scripts/spm_m8_alignment.py)，使用 [已冻结补测方案](../../runs/spm/m8_alignment_plan.json)。
