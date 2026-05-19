import { createServer, type IncomingMessage, type ServerResponse } from "node:http";
import { URL } from "node:url";
import { getGraphDefinition, listGraphs, type GraphInput } from "./graphs/registry.js";

// 这是给后续可视化页面使用的最小后端 API。
//
// 当前阶段先不用 Express/Hono，直接使用 Node 内置 http 模块。
// 好处是依赖少，而且能看清楚接口到底做了什么。
//
// 当前提供两个接口：
// - GET  /api/graphs: 查看当前支持哪些 graph
// - POST /api/graphs/:name/invoke: 执行某个 graph

const DEFAULT_PORT = 3001;

function sendJson(
  response: ServerResponse,
  statusCode: number,
  body: unknown,
) {
  response.writeHead(statusCode, {
    "Content-Type": "application/json; charset=utf-8",
    // 后续 Vite 前端会跑在另一个端口，所以先允许本地跨域。
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

async function parseJsonBody(request: IncomingMessage): Promise<GraphInput> {
  const rawBody = await readRequestBody(request);
  if (!rawBody.trim()) {
    return {
      question: "",
    };
  }

  const body = JSON.parse(rawBody) as Partial<GraphInput>;
  return {
    question: String(body.question ?? ""),
  };
}

async function handleInvoke(
  graphName: string,
  request: IncomingMessage,
  response: ServerResponse,
) {
  const graphDefinition = getGraphDefinition(graphName);
  if (!graphDefinition) {
    sendJson(response, 404, {
      error: `Unknown graph: ${graphName}`,
    });
    return;
  }

  const input = await parseJsonBody(request);
  if (!input.question.trim()) {
    sendJson(response, 400, {
      error: "question is required",
    });
    return;
  }

  const graph = graphDefinition.buildGraph();
  const result = await graph.invoke(input);
  sendJson(response, 200, {
    graph: graphDefinition.name,
    input,
    result,
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
      service: "langgraph-js-demo",
    });
    return;
  }

  if (request.method === "GET" && requestUrl.pathname === "/api/graphs") {
    sendJson(response, 200, {
      graphs: listGraphs(),
    });
    return;
  }

  const invokeMatch = requestUrl.pathname.match(
    /^\/api\/graphs\/([^/]+)\/invoke$/,
  );
  if (request.method === "POST" && invokeMatch) {
    await handleInvoke(invokeMatch[1], request, response);
    return;
  }

  sendJson(response, 404, {
    error: "Not found",
  });
}

export function createLangGraphDemoServer() {
  return createServer((request, response) => {
    handleRequest(request, response).catch((error: unknown) => {
      const message = error instanceof Error ? error.message : String(error);
      sendJson(response, 500, {
        error: message,
      });
    });
  });
}

const server = createLangGraphDemoServer();
const port = Number(process.env.PORT ?? DEFAULT_PORT);

server.listen(port, () => {
  console.log(`LangGraph JS demo API listening on http://localhost:${port}`);
  console.log("GET  /api/health");
  console.log("GET  /api/graphs");
  console.log("POST /api/graphs/:name/invoke");
});
