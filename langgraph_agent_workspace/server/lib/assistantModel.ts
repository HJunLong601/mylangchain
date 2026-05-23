import "dotenv/config";
import { AIMessage, HumanMessage, SystemMessage } from "@langchain/core/messages";
import { ChatOpenAI } from "@langchain/openai";
import type { AgentMessage } from "../types.js";

// 这个文件是“模型层”。
//
// Agent Graph 不应该直接关心具体模型厂商，否则后面从 GLM 换到别的模型会很痛。
// 所以这里提供一个 generateAssistantReply 方法：
// - 如果配置了 OpenAI 兼容 API，就调用真实大模型。
// - 如果没有配置 API Key，就使用本地规则回复，保证项目一下载就能跑通。

const apiKey = process.env.OPENAI_API_KEY || process.env.GLM_API_KEY;
const baseURL = process.env.OPENAI_BASE_URL || process.env.GLM_BASE_URL;
const modelName = process.env.OPENAI_MODEL || process.env.GLM_MODEL || "glm-5";

function buildSystemPrompt() {
  return [
    "你是一个个人学习 Agent，主要帮助用户学习 LangChain、LangGraph、RAG 和 Agent 工程化。",
    "回答要简洁、清楚，优先给出可执行建议。",
    "如果用户的问题更适合保存为笔记、查询笔记或走知识库，后端 LangGraph 会先完成路由。",
  ].join("\n");
}

function toLangChainMessages(messages: AgentMessage[]) {
  return messages.map((message) => {
    if (message.role === "user") {
      return new HumanMessage(message.content);
    }

    return new AIMessage(message.content);
  });
}

function generateLocalReply(question: string, history: AgentMessage[]) {
  const userMessageCount = history.filter((message) => message.role === "user").length;

  return [
    "我现在运行在本地规则模式，还没有调用真实大模型。",
    `这个会话里我已经看到 ${userMessageCount} 条用户消息。`,
    `你这次的问题是：${question}`,
    "下一步配置 OPENAI_API_KEY / OPENAI_BASE_URL 后，这个节点就可以切换成 GLM 等 OpenAI 兼容模型回答。",
  ].join("\n");
}

export async function generateAssistantReply(
  question: string,
  history: AgentMessage[],
): Promise<string> {
  if (!apiKey) {
    return generateLocalReply(question, history);
  }

  const model = new ChatOpenAI({
    apiKey,
    model: modelName,
    temperature: 0.2,
    configuration: baseURL
      ? {
          baseURL,
        }
      : undefined,
  });

  const response = await model.invoke([
    new SystemMessage(buildSystemPrompt()),
    // 只取最近几轮，避免上下文无限膨胀。
    ...toLangChainMessages(history).slice(-8),
    new HumanMessage(question),
  ]);

  return String(response.content);
}
