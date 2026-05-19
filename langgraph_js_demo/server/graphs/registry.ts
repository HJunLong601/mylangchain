import { buildConditionalGraph } from "./conditionalGraph.js";
import { buildSimpleGraph } from "./simpleGraph.js";
import { buildStateDemoGraph } from "./stateGraph.js";

// 这里集中管理当前后端支持的 graph。
//
// 这样后续做可视化页面时，不需要在前端写死有哪些图，
// 只要请求 GET /api/graphs 就能拿到列表。

export type GraphName = "simple" | "state" | "conditional";

export type GraphInput = {
  question: string;
};

export type GraphDefinition = {
  name: GraphName;
  title: string;
  description: string;
  defaultInput: GraphInput;
  buildGraph: () => ReturnType<
    | typeof buildSimpleGraph
    | typeof buildStateDemoGraph
    | typeof buildConditionalGraph
  >;
};

export const graphRegistry: Record<GraphName, GraphDefinition> = {
  simple: {
    name: "simple",
    title: "最小 Graph",
    description: "演示 START -> receiveQuestion -> generateAnswer -> END。",
    defaultInput: {
      question: "LangGraph 的 State 是什么？",
    },
    buildGraph: buildSimpleGraph,
  },
  state: {
    name: "state",
    title: "State 流转 Graph",
    description: "演示多个节点如何读取和更新同一个 State。",
    defaultInput: {
      question: "RAG 和普通聊天有什么区别？",
    },
    buildGraph: buildStateDemoGraph,
  },
  conditional: {
    name: "conditional",
    title: "条件分支 Graph",
    description: "演示 route function 和 addConditionalEdges。",
    defaultInput: {
      question: "上海今天的天气怎么样？",
    },
    buildGraph: buildConditionalGraph,
  },
};

export function listGraphs() {
  return Object.values(graphRegistry).map((graph) => ({
    name: graph.name,
    title: graph.title,
    description: graph.description,
    defaultInput: graph.defaultInput,
  }));
}

export function getGraphDefinition(name: string) {
  return graphRegistry[name as GraphName];
}
