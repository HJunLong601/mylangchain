import { Annotation, END, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { pathToFileURL } from "node:url";
import { generateAssistantReply } from "../lib/assistantModel.js";
import { createLearningNote, searchLearningNotes } from "../lib/learningStore.js";
import type { AgentIntent, AgentMessage, DebugEvent, LearningNote } from "../types.js";

// 这个文件是新目录的核心：统一 Agent Graph。
//
// 它不是一个“演示某个概念的小 demo”，而是把前面学过的几个能力收敛到同一个 Agent：
// 1. simpleGraph 的能力：从 START 到 END 跑通一个完整链路。
// 2. stateGraph 的能力：用 State 在节点之间传递中间结果。
// 3. conditionalGraph 的能力：根据意图选择不同分支。
// 4. memoryGraph 的能力：用 MemorySaver + thread_id 维护短期记忆。
//
// 之后工具调用、RAG、持久化会继续加到这张图上，而不是再新增一堆分散 demo。

const RESET_STEPS = "__RESET_STEPS__";
const RESET_DEBUG = "__RESET_DEBUG__";

const checkpointer = new MemorySaver();

const AgentState = Annotation.Root({
  // 本轮用户输入。每次 POST /api/agent/chat 都会传入新的 question。
  question: Annotation<string>,

  // 规范化后的输入，用于分类和检索。
  normalizedQuestion: Annotation<string>,

  // Agent 判断出的意图。
  // 当前先支持三类：
  // - chat: 普通对话
  // - save_note: 保存学习笔记
  // - search_notes: 查询已有学习笔记
  intent: Annotation<AgentIntent>,

  // 记录为什么走这个分支，方便右侧调试面板展示。
  routeReason: Annotation<string>,

  // 工具执行结果，例如保存了哪条笔记、检索命中了哪些内容。
  toolResult: Annotation<string>,

  // 本轮命中的学习笔记。后续接 RAG 时，这里可以换成 retrievedDocuments。
  retrievedNotes: Annotation<LearningNote[]>({
    reducer: (_current, update) => update,
    default: () => [],
  }),

  // 最终回复给用户的内容。
  answer: Annotation<string>,

  // 会话消息。这里配置 reducer 是为了“追加消息”，而不是覆盖历史。
  //
  // invoke 时传入一条 user message；
  // answer 节点再返回一条 assistant message；
  // 在同一个 thread_id 下，MemorySaver 会把这些 messages 保留下来。
  messages: Annotation<AgentMessage[]>({
    reducer: (currentMessages, newMessages) => currentMessages.concat(newMessages),
    default: () => [],
  }),

  // 当前这一轮的节点日志。
  //
  // 因为 checkpointer 会保存整个 State，如果直接 concat，steps 会跨轮次越积越多。
  // 所以 prepareTurn 节点会先写入 RESET_STEPS 标记，reducer 看到后就清空旧日志。
  steps: Annotation<string[]>({
    reducer: (currentSteps, newSteps) => {
      if (newSteps[0] === RESET_STEPS) {
        return newSteps.slice(1);
      }

      return currentSteps.concat(newSteps);
    },
    default: () => [],
  }),

  // 调试事件，比 steps 更结构化，前端可以展示 node、message 和 data。
  debug: Annotation<DebugEvent[]>({
    reducer: (currentEvents, newEvents) => {
      if (newEvents[0]?.node === RESET_DEBUG) {
        return newEvents.slice(1);
      }

      return currentEvents.concat(newEvents);
    },
    default: () => [],
  }),
});

export type AgentStateType = typeof AgentState.State;

function prepareTurnNode(state: AgentStateType) {
  const normalizedQuestion = state.question.trim().replace(/\s+/g, " ").toLowerCase();

  return {
    normalizedQuestion,
    routeReason: "",
    toolResult: "",
    retrievedNotes: [],
    answer: "",
    steps: [
      RESET_STEPS,
      `prepareTurn: 收到用户输入，并规范化为 "${normalizedQuestion}"`,
    ],
    debug: [
      {
        node: RESET_DEBUG,
        message: "重置上一轮调试信息",
      },
      {
        node: "prepareTurn",
        message: "完成本轮输入预处理",
        data: {
          question: state.question,
          normalizedQuestion,
        },
      },
    ],
  };
}

function classifyIntentNode(state: AgentStateType) {
  const question = state.normalizedQuestion;

  let intent: AgentIntent = "chat";
  let routeReason = "没有命中工具或知识库关键词，走普通对话分支。";

  if (
    question.includes("保存")
    || question.includes("记一下")
    || question.includes("记录")
    || question.includes("沉淀")
  ) {
    intent = "save_note";
    routeReason = "命中保存/记录类关键词，走学习笔记保存分支。";
  } else if (
    question.includes("笔记")
    || question.includes("学过")
    || question.includes("知识库")
    || question.includes("之前")
    || question.includes("总结")
  ) {
    intent = "search_notes";
    routeReason = "命中笔记/知识库/之前等关键词，走学习笔记检索分支。";
  }

  return {
    intent,
    routeReason,
    steps: [`classifyIntent: intent=${intent}`],
    debug: [
      {
        node: "classifyIntent",
        message: "根据用户输入判断 Agent 意图",
        data: {
          intent,
          routeReason,
        },
      },
    ],
  };
}

function routeByIntent(state: AgentStateType): AgentIntent {
  // 条件边只负责返回分支 key，不负责生成回答。
  // 真正的业务逻辑分别放在 saveNote/searchNotes/chatAnswer 节点里。
  return state.intent;
}

async function saveNoteNode(state: AgentStateType) {
  const note = await createLearningNote({
    title: `对话沉淀：${state.question.slice(0, 24)}`,
    content: state.question,
    kind: "observation",
    tags: ["agent", "conversation"],
  });

  const answer = [
    "已经帮你保存为学习笔记。",
    `标题：${note.title}`,
    "后续你可以问我“之前的笔记里有什么”，我会从本地笔记里检索。",
  ].join("\n");

  return {
    toolResult: `已保存学习笔记：${note.id}`,
    answer,
    messages: [
      {
        role: "assistant",
        content: answer,
      },
    ],
    steps: ["saveNote: 调用学习笔记工具并保存成功"],
    debug: [
      {
        node: "saveNote",
        message: "调用本地学习笔记保存工具",
        data: note,
      },
    ],
  };
}

async function searchNotesNode(state: AgentStateType) {
  const notes = await searchLearningNotes(state.normalizedQuestion);

  const answer = notes.length > 0
    ? [
        `我在学习笔记里命中了 ${notes.length} 条内容：`,
        ...notes.map((note, index) => `${index + 1}. ${note.title}\n${note.content}`),
      ].join("\n\n")
    : "我还没有在学习笔记里找到相关内容。你可以先让我保存一些学习结论，后续我就能基于它们回答。";

  return {
    retrievedNotes: notes,
    toolResult: notes.length > 0 ? `命中 ${notes.length} 条学习笔记` : "未命中学习笔记",
    answer,
    messages: [
      {
        role: "assistant",
        content: answer,
      },
    ],
    steps: ["searchNotes: 查询本地学习笔记并生成回答"],
    debug: [
      {
        node: "searchNotes",
        message: "执行学习笔记检索",
        data: {
          query: state.normalizedQuestion,
          matchedCount: notes.length,
          notes,
        },
      },
    ],
  };
}

async function chatAnswerNode(state: AgentStateType) {
  const answer = await generateAssistantReply(state.question, state.messages);

  return {
    answer,
    messages: [
      {
        role: "assistant",
        content: answer,
      },
    ],
    steps: ["chatAnswer: 生成普通对话回复"],
    debug: [
      {
        node: "chatAnswer",
        message: "进入普通对话分支",
        data: {
          historyMessageCount: state.messages.length,
          modelMode: process.env.OPENAI_API_KEY ? "openai-compatible" : "local-rule",
        },
      },
    ],
  };
}

export function buildAgentGraph() {
  return new StateGraph(AgentState)
    .addNode("prepareTurn", prepareTurnNode)
    .addNode("classifyIntent", classifyIntentNode)
    .addNode("saveNote", saveNoteNode)
    .addNode("searchNotes", searchNotesNode)
    .addNode("chatAnswer", chatAnswerNode)
    .addEdge(START, "prepareTurn")
    .addEdge("prepareTurn", "classifyIntent")
    .addConditionalEdges("classifyIntent", routeByIntent, {
      save_note: "saveNote",
      search_notes: "searchNotes",
      chat: "chatAnswer",
    })
    .addEdge("saveNote", END)
    .addEdge("searchNotes", END)
    .addEdge("chatAnswer", END)
    .compile({
      // 这是短期记忆的关键配置。
      // 只要 API 调用时传入相同 configurable.thread_id，
      // LangGraph 就会从 MemorySaver 里恢复上一次的 State。
      checkpointer,
    });
}

export const agentGraph = buildAgentGraph();

async function main() {
  const config = {
    configurable: {
      thread_id: "agent-demo-thread",
    },
  };

  const first = await agentGraph.invoke(
    {
      question: "我正在学习 LangGraph，帮我解释一下 State。",
      messages: [{ role: "user", content: "我正在学习 LangGraph，帮我解释一下 State。" }],
    },
    config,
  );

  const second = await agentGraph.invoke(
    {
      question: "把这句话保存成学习笔记：State 是节点之间共享和流转的数据。",
      messages: [{ role: "user", content: "把这句话保存成学习笔记：State 是节点之间共享和流转的数据。" }],
    },
    config,
  );

  console.log("\n=== Agent First Turn ===");
  console.log(JSON.stringify(first, null, 2));
  console.log("\n=== Agent Second Turn ===");
  console.log(JSON.stringify(second, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
