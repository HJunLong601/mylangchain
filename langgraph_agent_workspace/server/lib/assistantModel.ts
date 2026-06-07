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

export type LearningNoteDraft = {
  title: string;
  content: string;
  tags: string[];
  reason: string;
};

function buildSystemPrompt() {
  // System Prompt 用来约束模型的角色和回答风格。
  // 它不是用户消息，而是每次模型调用时额外传入的系统规则。
  return [
    "你是一个个人学习 Agent，主要帮助用户学习 LangChain、LangGraph、RAG 和 Agent 工程化。",
    "回答要简洁、清楚，优先给出可执行建议。",
    "如果用户的问题更适合保存为笔记、查询笔记或走知识库，后端 LangGraph 会先完成路由。",
  ].join("\n");
}

function toLangChainMessages(messages: AgentMessage[]) {
  // 项目内部使用简单的 { role, content }。
  // 调用 LangChain 模型前，需要转换成 LangChain 的消息类型：
  // - user -> HumanMessage
  // - assistant -> AIMessage
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

function createChatModel(temperature = 0.2) {
  // 统一创建模型实例，避免普通回答、笔记提取等场景各自散落模型配置。
  return new ChatOpenAI({
    // ChatOpenAI 是 LangChain 提供的 OpenAI 风格聊天模型封装。
    // 如果服务商兼容 OpenAI 协议，只要替换 apiKey/baseURL/model 即可复用这套代码。
    apiKey,
    model: modelName,
    temperature,
    configuration: baseURL
      ? {
          baseURL,
        }
      : undefined,
  });
}

function extractJsonObject(text: string) {
  // 模型有时会返回 ```json ... ```，也可能直接返回 JSON。
  // 这里做一个轻量清洗，尽量把第一个 JSON 对象提取出来。
  const withoutFence = text
    .replace(/^```json\s*/i, "")
    .replace(/^```\s*/i, "")
    .replace(/```$/i, "")
    .trim();
  const start = withoutFence.indexOf("{");
  const end = withoutFence.lastIndexOf("}");

  if (start < 0 || end < start) {
    throw new Error("模型没有返回 JSON 对象");
  }

  return withoutFence.slice(start, end + 1);
}

function parseLearningNoteDraft(rawText: string): LearningNoteDraft {
  const data = JSON.parse(extractJsonObject(rawText)) as Partial<LearningNoteDraft>;
  const title = String(data.title ?? "").trim();
  const content = String(data.content ?? "").trim();
  const tags = Array.isArray(data.tags)
    ? data.tags.map(String).map((tag) => tag.trim()).filter(Boolean)
    : [];
  const reason = String(data.reason ?? "").trim();

  if (!title || !content) {
    throw new Error("模型返回的笔记标题或内容为空");
  }

  return {
    title,
    content,
    tags,
    reason: reason || "模型根据历史对话提取了适合保存的学习笔记",
  };
}

function formatConversationForNoteExtraction(messages: AgentMessage[]) {
  // 只取最近若干条，既能保留“刚才/上面”的上下文，又避免 Prompt 过长。
  return messages
    .slice(-10)
    .map((message, index) => {
      const role = message.role === "user" ? "用户" : "Agent";
      return `${index + 1}. ${role}: ${message.content}`;
    })
    .join("\n\n");
}

export async function extractLearningNoteDraftFromHistory(
  instruction: string,
  history: AgentMessage[],
): Promise<LearningNoteDraft | undefined> {
  // 没有模型配置时，不做智能提取，交给 graph 里的规则兜底。
  if (!apiKey) {
    return undefined;
  }

  const model = createChatModel(0);
  const response = await model.invoke([
    new SystemMessage([
      "你是一个学习笔记整理器。",
      "你的任务是根据用户当前保存指令和最近对话，判断最应该保存哪段内容。",
      "如果用户说“保存上面回答/刚才内容/把 reducer 的重点保存一下”，你要从历史对话中选择相关内容，并整理成适合复习的笔记。",
      "不要把“保存到笔记”这类命令本身当成正文，除非历史里没有任何可保存内容。",
      "只返回 JSON，不要返回 Markdown，不要解释。",
      "JSON 字段固定为：title、content、tags、reason。",
      "tags 是字符串数组，建议包含 langgraph、agent、memory、rag、reducer 等相关关键词。",
    ].join("\n")),
    new HumanMessage([
      "【当前保存指令】",
      instruction,
      "",
      "【最近对话】",
      formatConversationForNoteExtraction(history),
      "",
      "请输出 JSON：",
      '{"title":"...","content":"...","tags":["..."],"reason":"..."}',
    ].join("\n")),
  ]);

  return parseLearningNoteDraft(String(response.content));
}

export async function generateAssistantReply(
  question: string,
  history: AgentMessage[],
): Promise<string> {
  // 没有 API Key 时走本地规则回复。
  // 这样学习 LangGraph 主链路时，不会被模型配置、网络或余额问题卡住。
  if (!apiKey) {
    return generateLocalReply(question, history);
  }

  const model = createChatModel(0.2);

  const response = await model.invoke([
    new SystemMessage(buildSystemPrompt()),
    // 只取最近几轮，避免上下文无限膨胀。
    ...toLangChainMessages(history).slice(-8),
    new HumanMessage(question),
  ]);

  return String(response.content);
}
