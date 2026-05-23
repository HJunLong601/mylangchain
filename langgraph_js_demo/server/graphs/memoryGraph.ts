import {
  END,
  MemorySaver,
  MessagesAnnotation,
  START,
  StateGraph,
} from "@langchain/langgraph";
import { AIMessage } from "@langchain/core/messages";
import { pathToFileURL } from "node:url";

// 这个示例专门演示 LangGraph 的短期记忆。
//
// 关键概念：
// - MemorySaver: 进程内 checkpointer，用来保存每个 thread 的 State
// - MessagesAnnotation: LangGraph 预置的 messages State
// - thread_id: 会话 ID，同一个 thread_id 会复用同一段短期记忆
//
// 注意：这个示例仍然不调用大模型。
// assistant 回复是本地规则生成的，目的是先看懂“记忆如何保存和恢复”。

const checkpointer = new MemorySaver();

type MemoryGraphState = typeof MessagesAnnotation.State;

function createMemoryAnswerNode(state: MemoryGraphState) {
  // MessagesAnnotation 会维护 messages 数组。
  // 每次 invoke 传入的新 user message 会被追加到已有 messages 后面。
  const messages = state.messages;
  const lastMessage = messages[messages.length - 1];
  const userMessageCount = messages.filter((message) => message.getType() === "human").length;

  const answer = [
    `我记得这个 thread 里目前有 ${messages.length} 条消息。`,
    `其中用户消息有 ${userMessageCount} 条。`,
    `你刚才说的是：${String(lastMessage?.content ?? "")}`,
  ].join("\n");

  return {
    // 这里只返回一条新的 AIMessage。
    // MessagesAnnotation 内部的 reducer 会把它追加到 messages 中，
    // 而不是覆盖之前的历史消息。
    messages: [new AIMessage(answer)],
  };
}

export function buildMemoryGraph() {
  return new StateGraph(MessagesAnnotation)
    .addNode("createMemoryAnswer", createMemoryAnswerNode)
    .addEdge(START, "createMemoryAnswer")
    .addEdge("createMemoryAnswer", END)
    .compile({
      // checkpointer 是短期记忆的关键。
      // 没有它，每次 invoke 都是一次独立运行。
      // 有了它，并且传入相同 thread_id，LangGraph 会恢复上一次的 State。
      checkpointer,
    });
}

export const memoryGraph = buildMemoryGraph();

async function main() {
  const config = {
    configurable: {
      thread_id: "demo-thread",
    },
  };

  const firstResult = await memoryGraph.invoke(
    {
      messages: [
        {
          role: "user",
          content: "我叫小龙。",
        },
      ],
    },
    config,
  );

  const secondResult = await memoryGraph.invoke(
    {
      messages: [
        {
          role: "user",
          content: "你还记得我刚才说了什么吗？",
        },
      ],
    },
    config,
  );

  console.log("\n=== Memory Graph First Result ===");
  console.log(JSON.stringify(firstResult, null, 2));
  console.log("\n=== Memory Graph Second Result ===");
  console.log(JSON.stringify(secondResult, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
