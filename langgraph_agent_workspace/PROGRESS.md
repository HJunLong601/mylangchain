# 个人 Agent 工作台进度

## 当前状态

| 模块 | 状态 | 说明 |
|---|---|---|
| 新目录 | 已完成 | `langgraph_agent_workspace/` |
| 工程配置 | 已完成 | Node + TypeScript + Vite + React |
| 统一 Agent Graph | 已完成 | `server/graphs/agentGraph.ts` |
| Agent Chat API | 已完成 | `POST /api/agent/chat` |
| 短期记忆 | 已完成 | `MemorySaver` + `thread_id` |
| 条件路由 | 已完成 | `chat` / `save_note` / `search_notes` |
| 学习笔记存储 | 已完成 | JSON 文件本地持久化 |
| 前端聊天界面 | 已完成 | `web/src/App.tsx` |
| 调试面板 | 已完成 | 展示 route、steps、debug JSON |
| GLM 接入 | 可选支持 | 配置 OpenAI 兼容环境变量后启用 |
| 标准工具抽象 | 未开始 | 下一步 |
| RAG 知识库 | 未开始 | 后续接入 |
| React Flow 图可视化 | 未开始 | 后续接入 |

## 已完成的链路

```text
用户消息
-> /api/agent/chat
-> agentGraph.invoke
-> prepareTurn
-> classifyIntent
-> 条件路由
-> chatAnswer / saveNote / searchNotes
-> 返回 answer + debugState
-> 前端展示对话和调试信息
```

## 下一步建议

下一步进入 A2：把当前 `saveNote`、`searchNotes` 两个节点背后的逻辑整理成更标准的工具层。

这样后续接天气、文件、RAG 检索时，就不是在 graph 里硬写业务逻辑，而是让 Agent 节点调用统一的 tools。
