# 个人 Agent 工作台进度表

## 当前状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 新目录创建 | 已完成 | `langgraph_js_demo/` |
| 技术栈确定 | 已完成 | TypeScript + LangGraph.js + React + React Flow |
| 计划表 | 已完成 | 见 `PLAN.md` |
| 代码初始化 | 已完成 | 已创建 `package.json`、`tsconfig.json`、`.env.example` |
| 最小 Graph | 已完成 | 已实现并验证 `server/graphs/simpleGraph.ts` |
| State 示例 | 已完成 | 已实现并验证 `server/graphs/stateGraph.ts` |
| 条件分支 | 已完成 | 已实现并验证 `server/graphs/conditionalGraph.ts` |
| 后端 API | 已完成 | 已实现并验证 `server/index.ts` |
| React 页面 | 已完成 | 已实现并验证 Vite + React 页面 |
| 短期记忆 | 已完成 | 已实现 `MemorySaver` + `MessagesAnnotation` + `threadId` |
| 学习沉淀 | 已完成 | 已实现本地 JSON 学习笔记和前端记录面板 |
| 统一 Agent 入口 | 未开始 | 下一步实现 `agentGraph.ts` 和 `/api/agent/chat` |
| React Flow 可视化 | 未开始 | 后续作为调试面板展示节点和边 |
| RAG 能力接入 | 未开始 | 后续作为 Agent 的知识库分支，而不是独立 demo |

## 详细进度

| 编号 | 任务 | 状态 | 备注 |
|---|---|---|---|
| LG-00 | 建立学习目录 | 已完成 | 与 Python RAG 代码隔离 |
| LG-01 | 编写阶段计划 | 已完成 | `PLAN.md` |
| LG-02 | 创建 Node/TS 项目 | 已完成 | `package.json`、`tsconfig.json`、`.env.example` |
| LG-03 | 安装 LangGraph.js 依赖 | 已完成 | `@langchain/langgraph`、`@langchain/core` |
| LG-04 | 实现最小图 | 已完成 | START -> receiveQuestion -> generateAnswer -> END |
| LG-05 | 实现 State 示例 | 已完成 | normalizeQuestion -> analyzeQuestion -> generateAnswer |
| LG-06 | 实现条件分支 | 已完成 | classifyQuestion -> routeQuestion -> rag/tool/chat |
| LG-07 | 搭建后端 API | 已完成 | `/api/health`、`/api/graphs`、`/api/graphs/:name/invoke` |
| LG-08 | 搭建 React 页面 | 已完成 | Vite + React + TS，支持调用后端 Graph |
| LG-08.5 | 实现短期记忆 Graph | 已完成 | MemorySaver + MessagesAnnotation + threadId |
| LG-08.6 | 实现学习沉淀面板 | 已完成 | 本地 JSON 持久化学习笔记 |
| AG-01 | 实现统一 Agent Graph | 未开始 | 把已有能力收敛到 `agentGraph.ts` |
| AG-02 | 新增 Agent Chat API | 未开始 | `POST /api/agent/chat` |
| AG-03 | 改造前端为聊天入口 | 未开始 | 从 Graph 选择器转向 Agent 对话 |
| AG-04 | 接入工具调用 | 未开始 | 学习笔记保存、读取资料等 |
| AG-05 | 接入 RAG 知识库 | 未开始 | 作为 Agent 的知识库分支 |
| AG-06 | 接入 React Flow 调试面板 | 未开始 | 展示节点、边、State 和路由 |

## 下一步建议

本次已经明确主线：后续不继续扩散成更多小 demo，而是围绕一个可用 Agent 逐步增强。

已验证命令：

```powershell
npm run typecheck
npm run dev
npm run dev:simple
npm run dev:state
npm run dev:conditional
npm run dev:memory
npm run dev:server
npm run dev:web
npm run build:web
```

已验证接口：

```text
GET  /api/health
GET  /api/graphs
POST /api/graphs/conditional/invoke
GET  /api/learning/notes
POST /api/learning/notes
```

已验证前端：

```text
http://localhost:5173
```

下一步建议进入 `AG-01`：实现统一的 `agentGraph.ts`，并新增 `POST /api/agent/chat`。这样前端后续就可以从“选择某个 Graph 运行”改成“直接和 Agent 对话”。
