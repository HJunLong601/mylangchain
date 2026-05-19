# LangGraph JS/TS 进度表

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
| React Flow 可视化 | 未开始 | 待展示节点和边 |
| RAG Graph 迁移 | 未开始 | 等 LangGraph 基础跑通后再做 |

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
| LG-09 | 接入 React Flow | 未开始 | 展示节点和边 |
| LG-10 | 展示执行日志 | 未开始 | 显示 node 输入输出 |
| LG-11 | 迁移简化 RAG Graph | 未开始 | rewrite / retrieve / rerank / generate |
| LG-12 | 条件兜底版 RAG Graph | 未开始 | hasEvidence 分支 |

## 下一步建议

本次已完成 React 页面。

已验证命令：

```powershell
npm run typecheck
npm run dev
npm run dev:simple
npm run dev:state
npm run dev:conditional
npm run dev:server
npm run dev:web
npm run build:web
```

已验证接口：

```text
GET  /api/health
GET  /api/graphs
POST /api/graphs/conditional/invoke
```

已验证前端：

```text
http://localhost:5173
```

下一步建议进入 `LG-09`：接入 React Flow，把 Graph 画成节点和边。
