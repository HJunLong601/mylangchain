# LangGraph JS/TS 个人 Agent 工作台

这个目录用于下一阶段学习：使用 **JavaScript / TypeScript** 一步步搭建一个真正可以使用的个人 Agent。

当前定位不是“一个知识点一个小 demo”，而是围绕同一个 Agent 产品雏形持续迭代：

- 前端提供可使用的 Agent 交互界面
- 后端用 LangGraph 编排 Agent 执行流程
- Agent 逐步具备短期记忆、工具调用、RAG 和学习沉淀能力
- 调试面板保留 State、路由、工具调用和执行日志，方便学习底层原理

它和当前 Python RAG Demo 隔离，避免两条学习线互相影响。

## 学习目标

这一阶段的目标不是继续堆零散示例，而是把 LangGraph 的能力逐步用到同一个 Agent 上：

- 用 LangGraph 表达 Agent 的多步骤决策链路
- 用 State 在节点之间传递上下文、意图、工具结果和最终回复
- 用 Edge / Conditional Edge 控制 Agent 走普通回答、工具调用或知识库检索
- 用短期记忆支持多轮对话
- 把之前 RAG 的思路迁移成 Agent 的知识库能力
- 用前端界面持续沉淀学习笔记，而不是只在命令行里跑样例

## 推荐技术栈

| 层级 | 技术 | 用途 |
|---|---|---|
| 语言 | TypeScript | 主开发语言，比纯 JS 更适合学习状态结构 |
| 运行时 | Node.js | 执行 LangGraph 服务端逻辑 |
| 图编排 | `@langchain/langgraph` | LangGraph.js 核心库 |
| LangChain 基础包 | `@langchain/core` | 消息、Runnable、基础类型 |
| 模型接入 | `@langchain/openai` | 通过 OpenAI 兼容接口接入 GLM |
| 前端框架 | Vite + React + TypeScript | 做轻量可视化页面 |
| 图可视化 | React Flow | 展示节点、边、执行状态 |
| 样式 | Tailwind CSS | 快速做界面布局 |
| 本地接口 | Express 或 Hono | 提供 `/invoke`、`/stream` 等接口 |
| 配置 | dotenv | 读取模型 Key、Base URL 等配置 |

> 说明：LangGraph.js 官方文档推荐安装 `@langchain/langgraph` 和 `@langchain/core`。当前计划会以这两个包作为核心依赖。

## 计划文件

- [PLAN.md](./PLAN.md)：可用 Agent 阶段计划
- [PROGRESS.md](./PROGRESS.md)：学习和实现进度表

## 当前已实现

当前已经完成：

- Node/TypeScript 项目初始化
- 最小 LangGraph 链路验证
- State 流转能力验证
- 条件分支能力验证
- 后端 API
- React 页面
- 短期记忆能力验证
- 学习笔记持久化

运行方式：

```powershell
cd langgraph_js_demo
npm install
npm run dev
```

如果想单独运行，也可以使用：

```powershell
npm run typecheck
npm run dev:simple
npm run dev:state
npm run dev:conditional
npm run dev:memory
npm run dev:server
npm run dev:web
```

下面这些 Graph 是为了理解底层能力，不是最终产品形态。

当前最小链路：

```text
START
-> receiveQuestion
-> generateAnswer
-> END
```

当前 State 流转能力：

```text
START
-> normalizeQuestion
-> analyzeQuestion
-> generateAnswer
-> END
```

State 示例重点观察：

- `question` 由 invoke 输入提供
- `normalizedQuestion` 由第一个节点写入
- `questionType` 由第二个节点写入
- `answer` 由第三个节点写入
- `steps` 通过 reducer 追加执行日志

当前条件分支能力：

```text
START
-> classifyQuestion
-> routeQuestion
   -> ragAnswer
   -> toolAnswer
   -> chatAnswer
-> END
```

条件分支重点观察：

- `classifyQuestion` 是普通节点，负责写入 `questionType`
- `routeQuestion` 是路由函数，负责读取 State 并返回分支 key
- `addConditionalEdges` 把分支 key 映射到真实节点
- 不同问题会进入不同 answer 节点

当前后端 API：

```text
GET  /api/health
GET  /api/graphs
POST /api/graphs/:name/invoke
GET  /api/learning/notes
POST /api/learning/notes
```

示例请求：

```powershell
Invoke-RestMethod `
  -Uri 'http://localhost:3001/api/graphs/conditional/invoke' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"question":"RAG 和普通聊天有什么区别？"}'
```

后端 API 的作用是给当前学习工作台使用。后续主线会逐步从 `/api/graphs/:name/invoke` 过渡到统一的 Agent 接口 `/api/agent/chat`。

当前短期记忆能力：

```text
MemorySaver
+ MessagesAnnotation
+ thread_id
= 同一个 thread 内保留 messages 历史
```

运行：

```powershell
npm run dev:memory
```

通过 API 调用时，`threadId` 相同就会复用同一段短期记忆：

```powershell
Invoke-RestMethod `
  -Uri 'http://localhost:3001/api/graphs/memory/invoke' `
  -Method Post `
  -ContentType 'application/json' `
  -Body '{"question":"我叫小龙","threadId":"demo-thread"}'
```

`MemorySaver` 是进程内存储，适合学习和开发。服务重启后记忆会丢，生产环境需要换成数据库型 checkpointer。

当前 React 页面：

```text
http://localhost:5173
```

一键启动后访问：

```text
http://localhost:5173
```

如果需要拆开运行，可以使用：

```powershell
# 终端 1
npm run dev:server

# 终端 2
npm run dev:web
```

页面当前支持：

- 加载后端 Graph 列表
- 选择 `simple` / `state` / `conditional` / `memory` 验证底层能力
- 输入问题
- 调用对应 Graph
- 展示 `steps`
- 展示最终 State JSON
- 把运行观察保存成学习笔记
- 手动记录概念、问题和待办

下一步会收敛到统一的 `agentGraph.ts` 和 `/api/agent/chat`，让前端从“选择 Graph”转向“使用 Agent 对话”。

## 初始目录规划

后续实现时建议演进成下面这样：

```text
langgraph_js_demo/
├─ README.md
├─ PLAN.md
├─ PROGRESS.md
├─ package.json
├─ tsconfig.json
├─ .env.example
├─ server/
│  ├─ index.ts
│  ├─ graphs/
│  │  ├─ simpleGraph.ts
│  │  ├─ stateGraph.ts
│  │  ├─ conditionalGraph.ts
│  │  └─ ragGraph.ts
│  └─ lib/
│     └─ model.ts
└─ web/
   ├─ index.html
   └─ src/
      ├─ App.tsx
      ├─ components/
      │  ├─ GraphCanvas.tsx
      │  ├─ StatePanel.tsx
      │  └─ RunPanel.tsx
      └─ main.tsx
```

## 推荐演进顺序

```text
统一 Agent 入口
-> 短期记忆
-> 工具调用
-> 意图路由
-> RAG 知识库
-> 聊天界面
-> 调试面板
-> 持久化和部署
```

核心原则：每一步都增强同一个 Agent，而不是继续扩散成更多孤立 demo。
