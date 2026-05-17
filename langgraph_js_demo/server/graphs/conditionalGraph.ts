import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { pathToFileURL } from "node:url";

// 这个示例专门学习条件分支。
//
// 前两个示例都是固定顺序：
// START -> A -> B -> END
//
// 真实工作流里经常不是这样。
// 比如用户问题可能有三类：
// - RAG 问题：走知识库回答
// - 工具问题：走工具调用
// - 普通聊天：直接回答
//
// 条件分支要解决的就是：
// 根据当前 State，决定下一步应该进入哪个节点。

type QuestionType = "rag" | "tool" | "chat";

const ConditionalGraphState = Annotation.Root({
  // 用户原始问题。
  question: Annotation<string>,
  // classifyQuestion 节点写入的问题类型。
  questionType: Annotation<QuestionType>,
  // routeQuestion 函数写入的路由说明，方便观察分支选择。
  routeReason: Annotation<string>,
  // 最终回答。
  answer: Annotation<string>,
  // 执行日志。
  steps: Annotation<string[]>({
    reducer: (currentSteps, newSteps) => currentSteps.concat(newSteps),
    default: () => [],
  }),
});

type ConditionalGraphStateType = typeof ConditionalGraphState.State;

function classifyQuestionNode(state: ConditionalGraphStateType) {
  // 这个节点负责“分类”，但不负责决定下一条边。
  // 真正决定走哪条边的是后面的 routeQuestion 函数。
  const question = state.question.trim().toLowerCase();

  let questionType: QuestionType = "chat";
  if (
    question.includes("rag")
    || question.includes("知识库")
    || question.includes("检索")
  ) {
    questionType = "rag";
  } else if (
    question.includes("天气")
    || question.includes("时间")
    || question.includes("工具")
  ) {
    questionType = "tool";
  }

  return {
    questionType,
    steps: [`classifyQuestion: questionType=${questionType}`],
  };
}

function routeQuestion(state: ConditionalGraphStateType): QuestionType {
  // route function 是条件分支的核心。
  //
  // 它不是普通业务节点，不负责生成最终回答。
  // 它只读取当前 State，然后返回一个路由 key。
  //
  // 下面的 addConditionalEdges 会用这个返回值去 pathMap 里找目标节点。
  return state.questionType;
}

function ragAnswerNode(state: ConditionalGraphStateType) {
  return {
    routeReason: "questionType=rag，所以进入 ragAnswer 节点",
    answer: `这是 RAG 分支：问题“${state.question}”后续应该进入知识库检索流程。`,
    steps: ["ragAnswer: 已进入 RAG 分支"],
  };
}

function toolAnswerNode(state: ConditionalGraphStateType) {
  return {
    routeReason: "questionType=tool，所以进入 toolAnswer 节点",
    answer: `这是工具分支：问题“${state.question}”后续可能需要调用时间、天气或其他工具。`,
    steps: ["toolAnswer: 已进入工具分支"],
  };
}

function chatAnswerNode(state: ConditionalGraphStateType) {
  return {
    routeReason: "questionType=chat，所以进入 chatAnswer 节点",
    answer: `这是普通聊天分支：问题“${state.question}”可以直接交给模型回答。`,
    steps: ["chatAnswer: 已进入普通聊天分支"],
  };
}

export function buildConditionalGraph() {
  return new StateGraph(ConditionalGraphState)
    .addNode("classifyQuestion", classifyQuestionNode)
    .addNode("ragAnswer", ragAnswerNode)
    .addNode("toolAnswer", toolAnswerNode)
    .addNode("chatAnswer", chatAnswerNode)
    .addEdge(START, "classifyQuestion")
    // addConditionalEdges 用来注册条件边。
    //
    // 第一个参数：从哪个节点执行完后开始路由。
    // 第二个参数：route function，读取 State 并返回一个路由 key。
    // 第三个参数：pathMap，把路由 key 映射到真实节点名。
    //
    // 这里的含义是：
    // classifyQuestion 执行完后，调用 routeQuestion(state)。
    // 如果返回 "rag"，就进入 ragAnswer。
    // 如果返回 "tool"，就进入 toolAnswer。
    // 如果返回 "chat"，就进入 chatAnswer。
    .addConditionalEdges("classifyQuestion", routeQuestion, {
      rag: "ragAnswer",
      tool: "toolAnswer",
      chat: "chatAnswer",
    })
    .addEdge("ragAnswer", END)
    .addEdge("toolAnswer", END)
    .addEdge("chatAnswer", END)
    .compile();
}

async function main() {
  const graph = buildConditionalGraph();
  const questions = [
    "RAG 和普通聊天有什么区别？",
    "上海今天的天气怎么样？",
    "帮我写一句学习鼓励的话",
  ];

  for (const question of questions) {
    const result = await graph.invoke({ question });
    console.log("\n=== Conditional Graph Result ===");
    console.log(JSON.stringify(result, null, 2));
  }
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
