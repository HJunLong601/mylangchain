# 可用 Agent 阶段计划

这个目录后续不再按“一个知识点一个小 demo”的方式推进，而是围绕同一个目标迭代：

> 一步步搭建一个真正可以使用的个人学习 Agent。

也就是说，`simple`、`state`、`conditional`、`memory` 这些内容只是学习 LangGraph 机制时的脚手架。后续主线会收敛到一个统一的 Agent 应用：前端有对话界面，后端有 LangGraph 编排，Agent 能记住上下文、调用工具、读取知识库，并把学习内容沉淀下来。

## 最终目标

第一版可用 Agent 应该具备这些能力：

| 能力 | 说明 |
|---|---|
| 对话入口 | 用户在前端输入问题，像普通聊天产品一样使用 |
| 短期记忆 | 同一个会话内能记住前面的上下文 |
| 工具调用 | Agent 可以调用后端工具，例如读取文件、查询天气、保存笔记 |
| 知识库问答 | 用户问到项目知识、学习笔记、文档内容时，Agent 能走 RAG |
| 学习沉淀 | 对话中的结论、问题、待办可以保存为学习笔记 |
| 可视化调试 | 开发阶段能看到 State、节点执行日志和路由结果 |
| 可替换模型 | 通过 OpenAI 兼容接口接入 GLM，后续也能替换其他模型 |

## 推荐演进路线

| 阶段 | 主题 | 目标 | 产出 |
|---|---|---|---|
| A0 | 项目骨架 | 建立 JS/TS 前后端工程 | `package.json`、`server/`、`web/` |
| A1 | 统一 Agent 入口 | 不再让用户选择 demo graph，而是进入一个 Agent 对话页 | `agentGraph.ts`、`POST /api/agent/chat` |
| A2 | 短期记忆 | 同一个 `threadId` 下保留多轮对话 | `MemorySaver` / checkpointer |
| A3 | 基础工具 | 接入保存学习笔记、读取本地资料等工具 | `server/tools/` |
| A4 | Agent 路由 | 根据问题决定普通回答、工具调用、知识库检索 | LangGraph conditional edges |
| A5 | RAG 知识库 | 把学习笔记和文档接入检索流程 | `retrieve -> generate` |
| A6 | 前端聊天体验 | 做成真正的聊天界面，而不是 Graph 选择器 | 消息列表、输入框、会话 ID |
| A7 | 调试面板 | 保留开发者视角，展示 State、节点日志、检索结果 | Debug Panel |
| A8 | 持久化 | 用数据库保存会话、笔记、知识库索引 | SQLite / PostgreSQL |
| A9 | 生产化整理 | 错误处理、配置、部署、模型切换 | README、`.env.example`、部署说明 |

## 当前应该怎么理解已有代码

当前已经写好的 `simpleGraph`、`stateGraph`、`conditionalGraph`、`memoryGraph` 不应该被当成最终功能，而应该当成四块底层能力验证：

| 已有内容 | 在最终 Agent 中对应什么 |
|---|---|
| `simpleGraph` | 验证 LangGraph 最小执行链路 |
| `stateGraph` | 验证 State 如何在节点之间累计信息 |
| `conditionalGraph` | 验证 Agent 如何根据问题选择不同路线 |
| `memoryGraph` | 验证多轮对话的短期记忆 |
| 学习笔记接口 | 未来会变成 Agent 的一个工具 |
| 当前 React 页面 | 未来会改造成聊天界面 + 调试面板 |

## 下一步主线

下一步不继续加新的零散 Graph，而是实现 `agentGraph.ts`：

```text
用户输入
-> loadContext        读取会话上下文
-> classifyIntent     判断用户意图
-> route              决定走普通回答、工具、还是知识库
-> answer             生成最终回复
-> saveTurn           保存本轮对话和必要的学习沉淀
```

对外接口也应该从：

```text
POST /api/graphs/:name/invoke
```

逐步过渡到：

```text
POST /api/agent/chat
```

前端也会从“选择 Graph 并运行”，逐步变成：

```text
左侧：会话列表 / 学习笔记
中间：Agent 聊天窗口
右侧：State、路由、工具调用、RAG 命中结果调试面板
```

## 技术决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 前后端是否分离 | 同目录内前后端分离 | API Key 不暴露到浏览器 |
| 主语言 | TypeScript | State、工具入参、接口返回结构更清楚 |
| Agent 编排 | `@langchain/langgraph` | 适合表达可控的多步骤 Agent |
| 模型接入 | OpenAI 兼容接口 | 方便接入智谱 GLM，也方便后续替换 |
| 前端框架 | Vite + React | 启动快，适合做本地 Agent 控制台 |
| 可视化调试 | React Flow + Debug Panel | 帮助理解 Agent 决策链路 |
| 初期存储 | JSON 文件 | 便于学习，不先引入数据库复杂度 |
| 后期存储 | SQLite / PostgreSQL | 支持真实会话、笔记、知识库持久化 |

## 暂不做的内容

- 暂不做登录权限
- 暂不做复杂部署
- 暂不一开始就接很多工具
- 暂不把 UI 做成复杂后台系统
- 暂不急着上数据库，先把 Agent 主链路跑通

优先级很明确：先让一个 Agent 能被正常使用，再逐步增强它。
