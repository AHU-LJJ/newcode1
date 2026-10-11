# 代码结构与使用说明

本仓库名为 `newcode1`，项目名为 CAPS。当前主要实现是 API 驱动的情境化偏好记忆系统 SPM；早期 `graphmem` 作为独立实验保留。两个系统使用不同 CLI、数据适配器和实验目录，不能直接混用记忆库。

## 目录

| 路径 | 职责 |
| --- | --- |
| `spm/` | 当前情境化偏好记忆系统、图/平铺管理、检索门控与评测 |
| `graphmem/` | 早期 HotpotQA / LoCoMo 图记忆实验，保持当前进度 |
| `scripts/` | 分轮实验的预留、冻结、执行、断点续跑、离线回放和报告生成；多数历史脚本依赖未提交的原运行目录 |
| `tests/` | 状态更新、来源绑定、协议校验、API 缓存与限速、指标计算等测试 |
| `examples/spm_demo.py` | 用少量合成会话运行 SPM 的真实 API 演示，不依赖完整数据集 |
| `paper/` | 理论笔记、引用信息及带来源/许可的 CUPID 官方评分提示参考；下载的 PDF 留在本地 |
| `pyproject.toml` / `uv.lock` / `.python-version` | 可恢复的 uv 依赖与 Python 版本声明，不包含虚拟环境本身 |
| `.env.example` / `spm.env.example` | 空密钥及可选模型配置示例 |
| `PROJECT_STATUS.md` | 当前进度、结论边界与下一轮状态 |
| `SPM*.md` / `GRAPH_MEMORY.md` / `LOCOMO*.md` | 系统说明、预先实验方案、逐轮结果及诊断 |

本地未入库目录包括 `datasets/`、`models/`、`.venv/`、`.cache/` 和 `runs/`。`.gitignore` 同时排除常见权重、数据库、日志及运行缓存文件；`uv.lock` 会提交。

## SPM 模块

| 模块 | 职责 |
| --- | --- |
| `__main__.py` | `python -m spm` 的 `inspect`、`run`、`frontier` 命令 |
| `adapters.py` | CUPID、PERMA、BenchPreS 和历史 PrefEval 读取；将公共历史与裁判标签分开 |
| `config.py` | 记忆策略、消融方法、API 角色与环境变量 |
| `models.py` | 会话、情境、偏好单元、证据、关系与指纹定义 |
| `pipeline.py` | 抽取、记忆写入、关系判断、查询解析、检索与门控的主要流程 |
| `store.py` | 按用户隔离的 SQLite 记忆与事件存储 |
| `conservative.py` | M7 的保守图更新策略 |
| `typed_graph.py` | M8/M9 的对象、维度、条件分轴及因果来源校验 |
| `small_protocol.py` | M8 的整数语义标签及宿主绑定的原文编号 |
| `boolean_protocol.py` | M9 的严格布尔语义、有界来源/实体编号及完整概率数组门控 |
| `prompts.py` | 系统各角色的提示 |
| `api.py` / `rate_limit.py` | API 并发、校验缓存、重复请求合并、预算账本、重试及模型 TPM 调度 |
| `baselines.py` | 完整历史与原文 RAG 等对照 |
| `runner.py` | 方法实例化、身份组选择、同源抽取控制、运行清单及样本续跑 |
| `evaluation.py` | 原有 checklist F1、代理裁判及其他基准指标汇总 |
| `cupid_alignment.py` | 官方 CUPID 评分提示输入构建及 1–10 分严格解析；补测结果独立保存 |

主要数据流：公共历史 → 有来源的偏好观察 → 图或平铺记忆 → 查询检索/适用性门控 → 偏好推断与回答 → 独立评测。真实偏好/checklist 只进入评测，不能进入记忆构建或回答生成。

## 环境恢复

在仓库根目录运行，要求 Python 3.12 和 uv：

```bash
uv sync --locked --group dev
uv run python -m spm --help
uv run python -m graphmem --help
uv run pytest -q
```

锁文件包含 PyTorch CUDA 13.0 依赖，首次安装较大；SPM 使用外部 API，不要求本地加载语言模型或嵌入模型。`check_torch.py` 是本地 CPU/GPU 环境检查，GPU 检查依赖实际驱动和设备。

复制 `.env.example` 为 `.env` 后，在本地填写 API 密钥及 base URL；按 `spm.env.example` 添加 SPM 可选配置。已有 `.env` 时保留现有文件。所有真实密钥仅保存在本地，示例文件不包含密钥。

当前系统默认 `Qwen/Qwen2.5-7B-Instruct + v6_compact`，裁判 `Pro/deepseek-ai/DeepSeek-V3.2`，嵌入 `BAAI/bge-m3`。SPM 各角色通过 `spm/config.py` 中的环境变量配置；早期 `graphmem` 的变量和参数见 [GRAPH_MEMORY.md](GRAPH_MEMORY.md)。Qwen 的 80,000 TPM 采用 72,000 有效预算。

可先用合成会话运行小型 API 演示，执行前需要有效 API 配置：

```bash
uv run python examples/spm_demo.py
```

## 本地数据布局

数据应从对应基准来源单独取得，放到下面的相对路径；仓库不分发原数据、权重或 tokenizer。具体列名和字段由 `spm/adapters.py` 与 `graphmem/data.py` 校验。

| 用途 | 预期本地路径 |
| --- | --- |
| CUPID | `datasets/cupid/test.parquet`；来源 [kixlab/CUPID](https://huggingface.co/datasets/kixlab/CUPID) |
| BenchPreS | `datasets/benchpres/data/test-00000-of-00001.parquet` |
| PERMA | `datasets/perma/tasks/<user>/input_data_*.json` 及 `datasets/perma/evaluation/<user>/meta/overall/<task_id>_<type>.json` 中的 prepared 选择题 |
| 历史 PrefEval | `datasets/PrefEval/`；仅供历史复现，不再纳入后续默认 `all` |
| HotpotQA / LoCoMo | 见 [GRAPH_MEMORY.md](GRAPH_MEMORY.md) 中的数据路径及 CLI 参数 |
| 可选 TPM tokenizer | `datasets/tokenizers/qwen2.5-7b/tokenizer.json`；缺失时通用 CLI/SDK 使用保守字节估算 |

准备 CUPID 后，可检查数据并在新目录试跑少量题：

```bash
uv run python -m spm inspect --benchmark cupid --split all
uv run python -m spm run --benchmark cupid --split dev \
  --methods v6_compact --limit 3 --max-history-episodes 24 \
  --workers 8 --concurrency 4 --max-calls 300 --max-tokens 500000 \
  --output runs/spm/local_cupid_pilot
```

测试使用临时目录、合成样本和模拟 API；不依赖外部 API 密钥。仅导出待提交文件、没有数据集与 `.env` 的目录中已验证 **132 passed, 2 skipped**；两项本地基准兼容性测试在缺少相关数据时跳过。

## 历史结果与复现边界

提交的 Markdown 保存了进度、方法配置、评测定义、均值、配对区间、缺失与诊断。原始回答、逐项裁判输出、向量、记忆库、HTTP 账本、运行 manifest、冻结源码及身份组预留均在本地 `runs/`，不提交。因此报告中的本地证据链接在纯 GitHub 克隆中不可用。

`scripts/spm_m*_replay.py` 和部分 `prepare`/`resume` 脚本会验证这些私有运行文件及其摘要，不能在缺少原文件的新克隆上直接复现历史样本或数字。普通 CLI 可以在自行准备的数据上开展新实验，但不等于复现已冻结的旧轮结果。发布历史原始证据需另行安排，不应绕过 `.gitignore` 上传。

M9 已实现接口并通过本地测试，尚未预留题组或调用 API；不能当作已完成实验。PrefEval 已退出后续实验；PERMA、CUPID、BenchPreS 的适配器保留。默认方法保持 `v6_compact`，没有强化学习训练。
