import "dotenv/config";
import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import { URL } from "node:url";
import { agentGraph } from "./graphs/agentGraph.js";
import { createLearningNote, listLearningNotes } from "./lib/learningStore.js";
import type { CreateLearningNoteInput } from "./types.js";

// 这是新 Agent 工作台的后端入口。
//
// 和旧 demo 的 /api/graphs/:name/invoke 不同，
// 这里主入口是 /api/agent/chat。
//
// 前端不再关心要运行 simple/state/conditional/memory 哪个 graph，
// 用户只需要发消息；后端 Agent Graph 会在内部完成：
// - State 预处理
// - 意图分类
// - 条件路由
// - 短期记忆
// - 工具调用
// - 回复生成

const DEFAULT_PORT = 3002;

function sendJson(
  response: ServerResponse,
  statusCode: number,
  body: unknown,
) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    "Access-Control-Allow-Origin": "*",
    "Access-Control-Allow-Methods": "GET,POST,OPTIONS",
    "Access-Control-Allow-Headers": "Content-Type",
  });
  response.end(JSON.stringify(body, null, 2));
}

function readRequestBody(request: IncomingMessage): Promise<string> {
  return new Promise((resolve, reject) => {
    const chunks: Buffer[] = [];

    request.on("data", (chunk) => {
      chunks.push(Buffer.from(chunk));
    });
    request.on("end", () => {
      resolve(Buffer.concat(chunks).toString("utf-8"));
    });
    request.on("error", reject);
  });
}

async function parseJsonBody<T>(request: IncomingMessage): Promise<Partial<T>> {
  const rawBody = await readRequestBody(request);
  if (!rawBody.trim()) {
    return {};
  }

  return JSON.parse(rawBody) as Partial<T>;
}

type AgentChatBody = {
  message: string;
  threadId?: string;
};

async function handleAgentChat(
  request: IncomingMessage,
  response: ServerResponse,
) {
  const body = await parseJsonBody<AgentChatBody>(request);
  const message = String(body.message ?? "").trim();

  if (!message) {
    sendJson(response, 400, {
      error: "message is required",
    });
    return;
  }

  // threadId 是短期记忆的会话标识。
  // 同一个 threadId 会复用 MemorySaver 里的历史 State；
  // 不传的话就自动创建一个新会话。
  const threadId = body.threadId?.trim() || `thread_${randomUUID()}`;

  const state = await agentGraph.invoke(
    {
      question: message,
      messages: [
        {
          role: "user",
          content: message,
        },
      ],
    },
    {
      configurable: {
        thread_id: threadId,
      },
    },
  );

  sendJson(response, 200, {
    threadId,
    answer: state.answer,
    // debugState 是给学习和调试看的。
    // 真正做产品时，可以只返回 answer/messages，把调试数据藏到开发模式里。
    debugState: {
      intent: state.intent,
      routeReason: state.routeReason,
      toolResult: state.toolResult,
      retrievedNotes: state.retrievedNotes,
      steps: state.steps,
      debug: state.debug,
      messages: state.messages,
    },
  });
}

async function handleCreateLearningNote(
  request: IncomingMessage,
  response: ServerResponse,
) {
  const body = await parseJsonBody<CreateLearningNoteInput>(request);
  const note = await createLearningNote({
    title: String(body.title ?? ""),
    content: String(body.content ?? ""),
    kind: body.kind,
    tags: Array.isArray(body.tags) ? body.tags.map(String) : [],
  });

  sendJson(response, 201, {
    note,
  });
}

async function handleRequest(
  request: IncomingMessage,
  response: ServerResponse,
) {
  const requestUrl = new URL(
    request.url ?? "/",
    `http://${request.headers.host ?? "localhost"}`,
  );

  if (request.method === "OPTIONS") {
    sendJson(response, 204, {});
    return;
  }

  if (request.method === "GET" && requestUrl.pathname === "/api/health") {
    sendJson(response, 200, {
      ok: true,
      service: "langgraph-agent-workspace",
    });
    return;
  }

  if (request.method === "POST" && requestUrl.pathname === "/api/agent/chat") {
    await handleAgentChat(request, response);
    return;
  }

  if (request.method === "GET" && requestUrl.pathname === "/api/learning/notes") {
    sendJson(response, 200, {
      notes: await listLearningNotes(),
    });
    return;
  }

  if (request.method === "POST" && requestUrl.pathname === "/api/learning/notes") {
    await handleCreateLearningNote(request, response);
    return;
  }

  sendJson(response, 404, {
    error: "Not found",
  });
}

export function createAgentWorkspaceServer() {
  return createServer((request, response) => {
    handleRequest(request, response).catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      sendJson(response, 500, {
        error: message,
      });
    });
  });
}

const server = createAgentWorkspaceServer();
const port = Number(process.env.PORT ?? DEFAULT_PORT);

server.listen(port, () => {
  console.log(`Agent workspace API listening on http://localhost:${port}`);
  console.log("GET  /api/health");
  console.log("POST /api/agent/chat");
  console.log("GET  /api/learning/notes");
  console.log("POST /api/learning/notes");
});
