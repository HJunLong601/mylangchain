# LangGraph JS/TS 可视化学习 Demo

这个目录用于下一阶段学习：使用 **JavaScript / TypeScript** 实现 LangGraph 工作流，并配一个可视化界面观察图节点、状态流转和执行结果。

它和当前 Python RAG Demo 隔离，避免两条学习线互相影响。

## 学习目标

这一阶段的目标不是继续堆 RAG 功能，而是学习：

- 如何用 LangGraph 表达多步骤工作流
- 如何用 State 在节点之间传递数据
- 如何用 Edge / Conditional Edge 控制流程
- 如何把执行过程展示到可视化界面
- 如何把之前 RAG 的思路迁移成图工作流

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

- [PLAN.md](./PLAN.md)：阶段计划表
- [PROGRESS.md](./PROGRESS.md)：学习和实现进度表

## 当前已实现

当前已经完成：

- Node/TypeScript 项目初始化
- 最小 LangGraph
- State 流转示例
- 条件分支示例

运行方式：

```powershell
cd langgraph_js_demo
npm install
npm run typecheck
npm run dev:simple
npm run dev:state
npm run dev:conditional
```

当前最小图：

```text
START
-> receiveQuestion
-> generateAnswer
-> END
```

当前 State 示例：

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

当前条件分支示例：

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

## 推荐学习顺序

```text
最小图
-> State
-> 顺序节点
-> 条件分支
-> 工具节点
-> 可视化执行
-> RAG Graph
-> 条件兜底版 RAG Graph
```

先把图工作流和状态流转学清楚，再迁移 RAG。这样后续代码不会变成“把旧逻辑硬塞进 LangGraph”。
