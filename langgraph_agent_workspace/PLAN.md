# 个人 Agent 迭代计划

这个计划只围绕一个目标：持续增强同一个 Agent。

## 里程碑

| 阶段 | 目标 | 产出 |
|---|---|---|
| A1 | 统一 Agent 入口 | `agentGraph.ts`、`POST /api/agent/chat`、聊天页面 |
| A2 | 工具抽象 | 把保存笔记、查询笔记整理成标准工具 |
| A3 | RAG 知识库 | 学习笔记和文档进入检索增强流程 |
| A4 | 可视化调试 | 用 React Flow 展示 Agent 节点、边和当前执行路径 |
| A5 | 持久化 | 用 SQLite 保存会话、笔记和知识库索引 |
| A6 | 模型增强 | 接入 GLM，支持模型配置和错误兜底 |
| A7 | 产品体验 | 会话列表、笔记管理、知识库管理 |

## 当前架构

```text
前端 React
  -> /api/agent/chat
  -> LangGraph agentGraph
      -> prepareTurn
      -> classifyIntent
      -> routeByIntent
          -> chatAnswer
          -> saveNote
          -> searchNotes
      -> END
```

## 核心原则

- 每一步都增强同一个 Agent。
- Demo 只作为理解概念的历史材料，不再作为主线产物。
- 调试信息服务学习，但用户主入口始终是聊天。
- 先跑通闭环，再逐步替换成更生产化的实现。
