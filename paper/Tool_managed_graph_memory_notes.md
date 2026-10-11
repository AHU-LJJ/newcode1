# 通过工具管理图结构记忆：文献与实现

检索日期：2026-10-09。以下区分论文机制、工程接口和待验证的研究设想。

## ContextPilot

- 论文：[ContextPilot: Teaching Agents for Proactive Context Management via Fine-grained RL](https://arxiv.org/abs/2608.28476)。[本地全文](ContextPilot_2026.pdf)，已解析验证：19 页，715,598 字节。
- 第 4.1 节与表 1：`memorize` 提取实体、时间和事件并建立记忆关联；`updateMemory` 更新记忆；`readMemory` 读取目标及邻居。另外提供笔记、检索和上下文卸载工具。
- 训练涉及 SFT 与针对关键上下文操作的细粒度 RL。图记忆是整体上下文管理的一部分；高层记忆接口不能直接等同于任意节点/边 CRUD。
- [官方代码](https://github.com/Tencent/ContextPilot)。评测涵盖长上下文问答和深度搜索，不能据此推定长期个性化有效。

## 直接相关系统

| 系统 | 核实的工具或机制 | 定位 |
| --- | --- | --- |
| [GRAM](https://openreview.net/pdf?id=rzGvGnwVC7) | `memory_insert`、`memory_update`、`memory_search`；Entity Tagging + Knowledge Graph；GRPO 训练主动构建和修订图 | 最接近工具 + 图记忆 + RL 的论文候选；主要面向问答 |
| [Graphiti MCP](https://github.com/getzep/graphiti/blob/main/mcp_server/README.md) | `add_memory` 写入 episode 并抽取；`add_triplet` 直接写事实；`search_nodes`、`search_memory_facts` 检索；`delete_entity_edge`、`delete_episode` 删除；支持时间与证据溯源 | 工程接口；接口本身不等于训练好的记忆管理策略，README 标为实验性 MCP 实现 |
| [MCP Knowledge Graph Memory Server](https://github.com/modelcontextprotocol/servers/tree/main/src/memory) | `create_entities`、`create_relations`、`add_observations`、对应删除工具、`search_nodes`、`open_nodes`、`read_graph` | 实体/关系/观察值的直接操作；适合简单工具基线，接口本身没有 RL 训练 |

GRAM 核实范围：搜索引擎索引的原始 PDF 摘要、图 1 与引言；原文标为 Anonymous ACL submission。直接打开全文遇到浏览器验证，未完成全文核读，未核实代码与录用状态。不要引用未经核实的实验数字或详细工具参数。

## 相关但不属于同一图路线

[AgeMem](https://arxiv.org/html/2601.01885v1) 将 ADD、UPDATE、DELETE、RETRIEVE、SUMMARY、FILTER 纳入策略，通过分阶段 RL 训练。该版本附录把记忆定义为内容、嵌入及元数据，使用相似度检索；不能仅凭 agentic memory 名称归为图编辑系统。[官方代码](https://github.com/y1y5/AgeMem)。

## 对 CAPS 方向的研究判断（设想，尚未验证）

单独提出“agent 调工具管理图记忆”不足以构成新意。可考察情境条件是否应成为显式可编辑对象，以及出现表面冲突时，策略能否选择拆分情境、修订长期偏好或仅记录暂时状态。

候选工具：记录带来源的 episode、创建带证据的条件模式、拆分模式的适用条件、撤回证据、按当前情境激活子图。用户心理状态应保留推断标记与不确定性，不把 LLM 推测写成确定事实。

建议基线：普通工具化图记忆；加入条件结构但使用固定管理规则；同样条件结构加学习策略。固定模型、原始信息和 token/工具预算，测试跨情境泛化、情境冲突处理及真实偏好变化。上述切入点是否新颖仍需进一步系统检索。
