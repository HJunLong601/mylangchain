# LangGraph JS/TS 学习计划表

## 阶段计划

| 阶段 | 主题 | 目标 | 产出 |
|---|---|---|---|
| 0 | 项目初始化 | 建立独立 JS/TS 目录，确定技术栈和目录结构 | `package.json`、`tsconfig.json`、`.env.example` |
| 1 | 最小 LangGraph | 跑通 START -> node -> END | `server/graphs/simpleGraph.ts` |
| 2 | State 基础 | 理解状态如何在节点间流转 | `server/graphs/stateGraph.ts` |
| 3 | 顺序编排 | 把一个任务拆成多个节点 | `server/graphs/sequentialGraph.ts` |
| 4 | 条件分支 | 根据 state 决定下一步 | `server/graphs/conditionalGraph.ts` |
| 5 | 工具节点 | 把普通 TS 函数接成工作流节点 | `server/graphs/toolGraph.ts` |
| 6 | 后端 API | 提供图执行接口，供前端调用 | `server/index.ts` |
| 7 | 可视化界面 | 用 React Flow 展示节点和边 | `web/src/components/GraphCanvas.tsx` |
| 8 | 执行状态面板 | 展示输入、输出、当前 state、节点执行日志 | `StatePanel`、`RunPanel` |
| 9 | RAG Graph | 把之前 RAG 思路迁移成图节点 | `server/graphs/ragGraph.ts` |
| 10 | 条件兜底 RAG | 根据检索结果决定回答或兜底 | `server/graphs/ragConditionalGraph.ts` |
| 11 | 总结文档 | 记录 LangGraph 和之前脚本式流程的区别 | `docs/` 或 README 补充 |

## 第一版功能范围

第一版不做复杂 RAG，只做 LangGraph 可视化主线：

- 一个后端服务
- 一个 React 页面
- 页面左侧展示图
- 页面右侧展示输入、输出和 state
- 点击运行后能看到节点执行顺序

第一版目标流程：

```text
START
-> classify_question
-> generate_answer
-> END
```

## 可视化界面规划

| 区域 | 内容 | 作用 |
|---|---|---|
| 顶部 | 当前 Graph 名称、运行按钮 | 控制执行 |
| 左侧 | React Flow 图 | 展示节点和边 |
| 右侧上方 | 用户输入 | 输入问题 |
| 右侧中部 | State JSON | 查看图执行后的状态 |
| 右侧下方 | 节点日志 | 查看每个节点执行情况 |

## 技术决策

| 决策 | 选择 | 原因 |
|---|---|---|
| 前后端是否分离 | 同目录内前后端分离 | API Key 不暴露到浏览器 |
| UI 框架 | Vite + React | 启动快，适合 demo |
| 图展示 | React Flow | 比 Mermaid 更适合交互式节点状态 |
| 语言 | TypeScript | State 类型更清晰 |
| 模型调用 | OpenAI 兼容接口 | 可继续接智谱 GLM |

## 暂不做的内容

- 暂不接数据库
- 暂不做登录权限
- 暂不做复杂部署
- 暂不复用 Python RAG 代码
- 暂不一开始就做完整 RAG Graph

这些内容后面可以加，但第一阶段重点是把 LangGraph 的图、状态和可视化跑通。
