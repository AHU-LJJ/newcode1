# CAPS

代码仓库：[AHU-LJJ/newcode1](https://github.com/AHU-LJJ/newcode1)。当前进度和下一步见 [PROJECT_STATUS.md](PROJECT_STATUS.md)，模块职责、数据布局、环境恢复与复现边界见 [CODE_STRUCTURE.md](CODE_STRUCTURE.md)。仓库保存代码、测试、依赖锁文件、配置示例、研究笔记与结果文档；本地数据集、模型、环境、密钥及 `runs/` 实验产物不入库。历史报告中的 `runs/` 和本地 PDF 链接需要原实验目录，GitHub 克隆中没有这些文件。

新增独立的 **SPM 情境化偏好记忆系统**：全程使用硅基流动 API 与 BAAI/bge-m3，支持 PrefEval、PERMA、CUPID、BenchPreS、并发缓存及断点续跑。实现、运行命令和评测边界见 [SPM.md](SPM.md)，四基准首轮开发集验证见 [SPM_RESULTS.md](SPM_RESULTS.md)。既有 graphmem 保持当前进度。

扩展开发/测试验证及组件消融见 [SPM_M1_RESULTS.md](SPM_M1_RESULTS.md)：恢复轨道测试组 117/117 完成，开发组 192/195 成功；原 GLM 轨道另行保留。当前尚无稳定优于完整历史或原文 RAG 的证据。

随后完成 [M2 新门控验证](SPM_M2_RESULTS.md)：351 项计划已执行，349 项有效评分，2 项原文证据失败单列。新门控改善部分适用召回，但 CUPID 成功配对子集的 F1 下降，当时保留旧门控。随后按用户选择，当时默认使用新门控 `v5`，旧版 `v4` 保留作对照。

[M3 新门控复验与原文证据消融](SPM_M3_RESULTS.md)已完成 375/375 项评分（开发 80、测试 295）。新测试中 v4/v5/v6 的 PrefEval 遵循率为 66.7%/70.8%/79.2%，CUPID F1 为 0.260/0.338/0.374；v6 补充检索原文，但相对 v5 的收益区间跨零，仍作为实验变体。该轮默认保持 v5。

[M4 Qwen2.5-7B 模型对照](SPM_M4_RESULTS.md)已执行相同的 59 个问题 × v4/v5/v6，并按 80,000 TPM 留余量限速，未出现 429。端到端 Qwen 替换后仅 86/177 项获得有效评分；主要受严格写入校验限制。PrefEval 共同成功的 18 题上，v4/v5/v6 分别为 11.1%/33.3%/38.9%；这些是有执行缺失的子集成绩。

[M5 小模型接口与 v6 消融](SPM_M5_RESULTS.md)已完成 190 项计划的执行与一次失败项续跑，176 项有效评分。相同 59 题上，新 v6 接口的有效覆盖由旧 v6 的 28/59 提升至 55/59。新问题组中，原文补充使 PrefEval 从 33.3% 提升至 37.5%（24 题，区间跨零）；CUPID F1 从 0 提升至 0.171（共同成功 9/12 题，差值区间为正，仍有执行缺失）。

当前默认使用 **Qwen2.5-7B-Instruct + `v6_compact`**，原文证据由短编号绑定，候选关系与门控使用整数序号；旧 `v6` 和 `v6_compact_no_raw` 保留作对照。评审为 DeepSeek-V3.2，嵌入为 BGE-M3 API；CLI 与直接 SDK 调用自动按 Qwen 80,000 TPM 留 10% 余量限速。此前 120 项测试通过，本次另有 14 项 CUPID 偏好对齐评分测试通过；旧 graphmem 实验保持原进度。

按用户最新要求，**PrefEval 已退出后续实验**，历史适配器与结果保留；`--benchmark all` 默认仅包含 PERMA、CUPID、BenchPreS。[M8 方案](SPM_M8_PROTOCOL.md)只测 CUPID，比较平铺、原图、对象/维度图、同预算原文 RAG、完整历史。开发 6 题、测试 24 题的成员与源码在首次 API 预测前同时冻结；冻结前 94 项测试通过。[已完成结果](SPM_M8_RESULTS.md)：开发 24/30、主测试 93/120 有效，五方法共同有效 15/24 题。原图 / 平铺 / 新图 F1 为 0.525 / 0.377 / 0.456；新图减平铺的主比较区间跨零。原图减平铺 +0.148、95% 区间 [+0.031, +0.275] 是存在缺失且未校正多重比较的探索性信号，仍需独立复验。新图 3 次替换全部在同一问题，存在助手修订被误当偏好变化的迹象，见[诊断](SPM_M8_DIAGNOSTICS.md)。旧 2478 个样本不变，93 项主测试成功结果精确离线回放，0 HTTP；默认仍 v6_compact。[M9 接口](SPM_M9_PROTOCOL.md)已实现布尔标志、有界编号、完整概率数组门控及修订语义提示，本地测试增至 120 项并通过，尚未预留或调用 API。

[CUPID 原论文偏好对齐分补测](SPM_M8_ALIGNMENT_RESULTS.md)已完成：采用官方发布的逐项 checklist 评分提示、DeepSeek-V3.2 裁判，共补评 93 份已有回答，0 新增裁判失败。共同 15 题上的原图 / 新图 / 平铺 / 原文 RAG / 完整历史均分为 5.47 / 5.47 / 5.60 / 5.33 / 5.07（1–10）。原图和新图减平铺均 -0.13，配对区间均跨零，尚未确认回答偏好对齐收益；原图的 F1 信号不能替代这一生成指标。原评分不变，93 份新裁判输出精确离线回放，0 HTTP；新增裁判调用实际为 90 次、150,578 已知 tokens。论文原裁判为 GPT-4o，分数不能直接与论文表格横比。

[M6 图管理对照](SPM_M6_RESULTS.md)已完成：353/364 项有效，其中新测试 281/288 项有效。四方法共同有效的 47 个 PrefEval 问题上，完整历史 / 同预算原文 RAG / 平铺 / 图版遵循率分别为 40.4% / 36.2% / 34.0% / 34.0%；22 个 CUPID 问题上 F1 为 0.216 / 0.291 / 0.309 / 0.311。图减平铺的配对区间均跨零，尚未确认图管理稳定增益。门控补齐、TPM 调度及并发向量缓存修复通过验证，281 个主测试成功上下文和评分从缓存离线回放，无新增 HTTP；旧 1844 个样本保持不变。详见[方案](SPM_M6_PROTOCOL.md)和[诊断](SPM_M6_DIAGNOSTICS.md)。

[M7 保守图更新](SPM_M7_RESULTS.md)已完成执行与一次有界续跑：248/270 项有效，主测试 194/216 项有效。共同有效的 48 个 PrefEval / 16 个 CUPID 问题上，原图 / 平铺 / 保守图遵循率为 50.0% / 45.8% / 45.8%，F1 为 0.267 / 0.293 / 0.329；主比较区间包含零。保守版没有实际时间替换，接近平铺保留量，额外核对尚未建立稳定收益。82 项测试通过，旧 2208 个样本不变，主测试 194 个上下文/trace/原生成与评分离线回放，0 新 HTTP。consistent 层共同有效仅 2/7，缺失不可忽略。见[方案](SPM_M7_PROTOCOL.md)与[诊断](SPM_M7_DIAGNOSTICS.md)；默认保持 v6_compact。

现已包含 API 驱动的图记忆管理实验，支持本地 HotpotQA、LoCoMo，默认使用 **BAAI/bge-m3** 检索（本地 GPU 或硅基流动 API）。操作与实验边界见 [GRAPH_MEMORY.md](GRAPH_MEMORY.md)。

LoCoMo 的 12 题 × 3 组真实 API 测试已完成，见 [LoCoMo 结果](LOCOMO_RESULTS.md)。此前使用 BM25 的 HotpotQA 10 题 × 4 组测试见 [历史结果](EXPERIMENT_RESULTS.md)。下方命令使用新实验目录；已有目录请更换名称，或在代码和配置相同的情况下加 `--resume`。

LoCoMo 支持对已保存的答案补算 **F1 / BLEU-1 / LLM Judge**；judge 默认使用硅基流动
`Pro/deepseek-ai/DeepSeek-V3.2`，见 [评分协议与运行说明](LOCOMO_EVALUATION.md)。
默认使用用户提供的 **Memory-R1 judge 模板**，可用 `--judge-rubric strict` 切换到严格规则作对照。

```bash
uv run python -m graphmem run --dataset locomo --limit 12 --seed 42 \
  --embedding-device cuda --output runs/my_locomo12
```

使用 uv 管理的独立 PyTorch 环境。

已验证：Python 3.12.3、PyTorch/torchvision 导入、CPU 矩阵乘法，以及
RTX 5060 Ti（计算能力 12.0）上的 CUDA 矩阵乘法均通过。

- Python 3.12
- PyTorch 2.13.0 + CUDA 13.0
- torchvision 0.28.0
- 虚拟环境：`.venv`
- PyTorch 软件包源：官方 `https://download.pytorch.org/whl/cu130`

## 使用

```bash
cd /home/alhandsome/projects/CAPS
uv run python check_torch.py
uv run python your_script.py
```

也可以激活环境后直接使用 Python：

```bash
source .venv/bin/activate
python check_torch.py
deactivate
```

uv 安装位置为 `~/.local/bin/uv`。若终端找不到 `uv`，可使用完整路径，或运行：

```bash
export PATH="$HOME/.local/bin:$PATH"
```

添加项目依赖：`uv add 包名`。按锁文件恢复环境：`uv sync --locked`。
首次下载较大的 CUDA 软件包时，如遇网络超时，可运行：

```bash
UV_HTTP_TIMEOUT=600 UV_CONCURRENT_DOWNLOADS=2 uv sync --locked
```

PyTorch CUDA 软件包携带运行时依赖；基本 GPU 运算无需额外安装系统 CUDA Toolkit。
`nvidia-smi` 显示的 CUDA 版本表示驱动可支持的 CUDA 版本，与
`torch.version.cuda` 显示的软件包构建版本可以不同。

`check_torch.py` 会检查导入、CPU 运算和真实 GPU 矩阵乘法。
如果当前终端或执行沙箱限制了 GPU 访问，检查会明确报错；请在有 GPU
访问权限的 WSL 终端中重新运行。
