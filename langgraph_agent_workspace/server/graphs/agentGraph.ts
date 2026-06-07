import { Annotation, END, MemorySaver, START, StateGraph } from "@langchain/langgraph";
import { pathToFileURL } from "node:url";
import {
  extractLearningNoteDraftFromHistory,
  generateAssistantReply,
} from "../lib/assistantModel.js";
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

// 这两个特殊标记只用于“清空本轮调试信息”。
//
// 因为 MemorySaver 会把整个 State 按 thread_id 保存下来，
// 如果 steps/debug 一直使用 concat 追加，第二轮、第三轮会把旧日志也带出来。
// 所以 prepareTurnNode 会先写入 RESET 标记，reducer 看到后就只保留本轮日志。
const RESET_STEPS = "__RESET_STEPS__";
const RESET_DEBUG = "__RESET_DEBUG__";

// MemorySaver 是 LangGraph 提供的内存版 checkpointer。
//
// 它会按照 configurable.thread_id 保存每个会话的 State。
// 注意：这是“进程内短期记忆”，服务重启后数据会丢。
// 后续做真正可用的长期记忆时，可以换成 SQLite / Postgres 等持久化 checkpointer。
const checkpointer = new MemorySaver();

// Annotation.Root 用来定义这张图的 State 结构。
//
// 可以把 State 理解成“Agent 运行过程中的共享上下文”：
// - 每个节点都能读取 State
// - 每个节点只返回自己要更新的字段
// - LangGraph 根据字段规则把局部更新合并回完整 State
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
    // 检索结果只关心“本轮命中内容”，不需要累计历史命中。
    // 所以这里不是 concat，而是直接用本轮 update 覆盖旧值。
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
    // reducer 不是 TypeScript 自带语法，而是 LangGraph 的 State 合并规则。
    //
    // currentMessages 是旧 State 中的消息历史；
    // newMessages 是本轮节点返回的新消息；
    // concat 表示把新消息追加到旧消息后面，从而形成多轮对话上下文。
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
  // prepareTurn 是每轮对话的入口节点。
  //
  // 它不负责回答问题，只做三件事：
  // 1. 清理和规范化用户输入
  // 2. 重置上一轮的临时字段
  // 3. 写入本轮调试日志
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
  // classifyIntent 是当前版本的“意图识别”节点。
  //
  // 这里先用关键词规则实现，方便学习条件分支。
  // 后续可以替换成模型分类、结构化输出，或者更复杂的路由器。
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

type NoteContentSource =
  | "model_history_extraction"
  | "explicit_user_content"
  | "previous_assistant_answer"
  | "current_user_message";

type NoteContentCandidate = {
  title: string;
  content: string;
  source: NoteContentSource;
  reason: string;
  tags: string[];
};

function extractExplicitNoteContent(question: string): string | undefined {
  // 用户明确写出要保存的内容时，通常会使用冒号：
  // - 保存这句话：State 是节点之间共享的数据
  // - 把这段内容保存到笔记: reducer 决定 State 如何合并
  //
  // 这种情况下应该优先保存冒号后面的正文，而不是保存整条命令。
  const separatorIndex = Math.min(
    ...["：", ":"]
      .map((separator) => question.indexOf(separator))
      .filter((index) => index >= 0),
  );

  if (!Number.isFinite(separatorIndex)) {
    return undefined;
  }

  const content = question.slice(separatorIndex + 1).trim();
  return content.length > 0 ? content : undefined;
}

function isAskingToSavePreviousAnswer(question: string) {
  // 这类表达本质上是在引用短期记忆里的上一条 assistant 回复：
  // - 保存上面回答
  // - 将刚才的内容保存到笔记
  // - 把上一条回答记一下
  const normalizedQuestion = question.toLowerCase();
  return [
    "上面",
    "刚才",
    "上一条",
    "前面",
    "回答",
    "内容",
    "这段",
  ].some((keyword) => normalizedQuestion.includes(keyword));
}

function findLatestAssistantMessage(messages: AgentMessage[]) {
  // 当前 saveNoteNode 执行时，messages 里通常已经包含：
  // 历史 user / assistant 消息 + 本轮 user 消息。
  //
  // 因此从后往前找最近一条 assistant，即可拿到“上面回答”的内容。
  return [...messages]
    .reverse()
    .find((message) => {
      if (message.role !== "assistant" || !message.content.trim()) {
        return false;
      }

      // 如果用户连续多次保存，最近一条 assistant 可能是“已经帮你保存为学习笔记”。
      // 这类确认消息不适合作为下一次保存的正文，所以要跳过。
      return !message.content.startsWith("已经帮你保存为学习笔记。");
    });
}

function resolveNoteContentByRule(state: AgentStateType): NoteContentCandidate {
  const explicitContent = extractExplicitNoteContent(state.question);
  if (explicitContent) {
    return {
      title: `对话沉淀：${explicitContent.slice(0, 24)}`,
      content: explicitContent,
      source: "explicit_user_content",
      reason: "用户在冒号后明确提供了要保存的内容",
      tags: ["agent", "conversation", "explicit_user_content"],
    };
  }

  const latestAssistantMessage = findLatestAssistantMessage(state.messages);
  if (isAskingToSavePreviousAnswer(state.question) && latestAssistantMessage) {
    return {
      title: `回答沉淀：${latestAssistantMessage.content.slice(0, 24)}`,
      content: latestAssistantMessage.content,
      source: "previous_assistant_answer",
      reason: "用户引用了上面/刚才/上一条回答，因此保存最近一条 assistant 回复",
      tags: ["agent", "conversation", "previous_assistant_answer"],
    };
  }

  // 兜底策略：如果用户只是说“保存这段知识”，但没有历史 assistant 回复，
  // 就保存本轮用户输入，避免工具什么都不做。
  // 后续可以进一步优化成“追问用户要保存哪段内容”。
  return {
    title: `对话沉淀：${state.question.slice(0, 24)}`,
    content: state.question,
    source: "current_user_message",
    reason: "没有解析到明确内容或上一条 assistant 回复，兜底保存本轮用户输入",
    tags: ["agent", "conversation", "current_user_message"],
  };
}

async function resolveNoteContent(state: AgentStateType): Promise<NoteContentCandidate> {
  // 优先使用模型根据历史对话智能提取笔记。
  //
  // 这样用户说：
  // - “把刚才 reducer 的重点保存一下”
  // - “将上面回答的内容保存到笔记”
  //
  // 模型可以结合最近几轮 messages，选择真正应该保存的内容，
  // 并整理出更适合复习的标题、正文和标签。
  try {
    const modelDraft = await extractLearningNoteDraftFromHistory(
      state.question,
      state.messages,
    );

    if (modelDraft) {
      return {
        title: modelDraft.title,
        content: modelDraft.content,
        source: "model_history_extraction",
        reason: modelDraft.reason,
        tags: ["agent", "conversation", "model_history_extraction", ...modelDraft.tags],
      };
    }
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.warn(`模型提取学习笔记失败，回退到规则解析：${message}`);
  }

  // 没有配置模型，或模型返回结果不可解析时，回退到规则版。
  // 这样保存笔记功能不会因为模型异常而完全不可用。
  return resolveNoteContentByRule(state);
}

async function saveNoteNode(state: AgentStateType) {
  // saveNoteNode 可以理解成当前版本的“工具节点”。
  //
  // 它调用 learningStore，把解析出来的内容保存成本地学习笔记。
  // 注意：这里不能简单保存 state.question。
  // 因为用户可能会说“保存上面回答”，这时真正要保存的是短期记忆里的上一条 assistant 回复。
  // 后续做标准 Tool 抽象时，可以把 createLearningNote 包装成独立工具。
  const noteContent = await resolveNoteContent(state);
  const note = await createLearningNote({
    title: noteContent.title,
    content: noteContent.content,
    kind: "observation",
    tags: noteContent.tags,
  });

  const answer = [
    "已经帮你保存为学习笔记。",
    `标题：${note.title}`,
    `保存来源：${noteContent.reason}`,
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
    steps: [`saveNote: 调用学习笔记工具并保存成功，source=${noteContent.source}`],
    debug: [
      {
        node: "saveNote",
        message: "调用本地学习笔记保存工具",
        data: {
          note,
          resolvedContent: noteContent,
        },
      },
    ],
  };
}

async function searchNotesNode(state: AgentStateType) {
  // searchNotesNode 是当前版本的“检索节点”。
  //
  // 现在只是关键词搜索学习笔记；
  // 后续接 RAG 时，可以在这里替换成 embedding 检索、向量库召回和 rerank。
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
  // chatAnswerNode 是普通对话分支。
  //
  // 如果配置了 OpenAI 兼容模型，这里会调用真实模型；
  // 如果没有配置 API Key，则走本地规则回复，保证项目可以零配置跑通。
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
  // StateGraph 是 LangGraph 的图构建器。
  //
  // 下面这段可以按四层理解：
  // 1. new StateGraph(AgentState): 声明这张图使用哪种 State
  // 2. addNode: 注册业务节点，但还不决定执行顺序
  // 3. addEdge / addConditionalEdges: 定义节点之间怎么流转
  // 4. compile: 编译成可以 invoke 的可执行对象
  return new StateGraph(AgentState)
    // addNode(name, fn): 给图注册一个节点。
    // 节点函数读取完整 State，返回局部 State 更新。
    .addNode("prepareTurn", prepareTurnNode)
    .addNode("classifyIntent", classifyIntentNode)
    .addNode("saveNote", saveNoteNode)
    .addNode("searchNotes", searchNotesNode)
    .addNode("chatAnswer", chatAnswerNode)
    // addEdge(from, to): 定义固定流转路线。
    // START / END 是 LangGraph 提供的虚拟起点和终点，不是业务函数。
    .addEdge(START, "prepareTurn")
    .addEdge("prepareTurn", "classifyIntent")
    // addConditionalEdges 用来定义条件分支。
    //
    // classifyIntent 执行完后，LangGraph 会调用 routeByIntent(state)。
    // routeByIntent 返回 "chat" / "save_note" / "search_notes"，
    // 再通过下面的映射进入对应节点。
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
