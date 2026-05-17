import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { pathToFileURL } from "node:url";

// 这个示例专门用来学习 State。
//
// 和 simpleGraph 相比，这里多了几个字段：
// - question: 原始问题
// - normalizedQuestion: 规范化后的问题
// - questionType: 简单分类结果
// - answer: 最终回答
// - steps: 节点执行日志
//
// 重点观察：
// 每个节点都只返回自己负责更新的字段，
// LangGraph 会把这些局部更新合并成最终 State。
//
// 注意：这个示例也不会调用大模型。
// 当前 answer 是本地规则生成的，不会消耗 API Key。
// 后面接入 ChatOpenAI / GLM 时，通常会把模型调用放进某个 Node 里。

type QuestionType = "rag" | "chat" | "tool";

const StateDemoGraphState = Annotation.Root({
  // 原始用户问题。通常由 invoke(...) 的输入提供。
  question: Annotation<string>,
  // 规范化后的问题。由 normalizeQuestion 节点生成。
  normalizedQuestion: Annotation<string>,
  // 问题类型。由 analyzeQuestion 节点生成。
  questionType: Annotation<QuestionType>,
  // 最终回答。由 generateAnswer 节点生成。
  answer: Annotation<string>,
  // 执行日志。多个节点都会写 steps，所以这里必须配置 reducer。
  steps: Annotation<string[]>({
    reducer: (currentSteps, newSteps) => currentSteps.concat(newSteps),
    default: () => [],
  }),
});

type StateDemoGraphStateType = typeof StateDemoGraphState.State;

function normalizeQuestionNode(state: StateDemoGraphStateType) {
  // 这个节点只关心 question。
  // 它不会生成 answer，也不会判断 questionType。
  const normalizedQuestion = state.question
    .trim()
    .replace(/\s+/g, " ")
    .toLowerCase();

  return {
    normalizedQuestion,
    steps: [
      `normalizeQuestion: "${state.question}" -> "${normalizedQuestion}"`,
    ],
  };
}

function analyzeQuestionNode(state: StateDemoGraphStateType) {
  // 这个节点读取上一个节点写入的 normalizedQuestion。
  // 这就是 State 在节点之间流转的最直观体现。
  const question = state.normalizedQuestion;

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
    steps: [`analyzeQuestion: questionType=${questionType}`],
  };
}

function generateAnswerNode(state: StateDemoGraphStateType) {
  // 这个节点读取前面两个节点写入的字段：
  // - normalizedQuestion
  // - questionType
  //
  // 这说明节点之间不需要直接互相调用，
  // 只要通过共享 State 传递中间结果即可。
  const answerByType: Record<QuestionType, string> = {
    rag: "这是一个 RAG 相关问题，后续可以进入检索、重排和基于证据回答的流程。",
    chat: "这是一个普通聊天问题，可以直接交给模型回答。",
    tool: "这是一个可能需要工具的问题，后续可以路由到工具节点。",
  };

  return {
    answer: answerByType[state.questionType],
    steps: [
      `generateAnswer: 使用 questionType=${state.questionType} 生成回答`,
    ],
  };
}

export function buildStateDemoGraph() {
  // 这里的 addNode 和 addEdge 要分开理解：
  //
  // addNode 负责“有哪些处理步骤”：
  // - normalizeQuestion
  // - analyzeQuestion
  // - generateAnswer
  //
  // addEdge 负责“这些步骤按什么顺序执行”：
  // START -> normalizeQuestion -> analyzeQuestion -> generateAnswer -> END
  //
  // 如果把 addNode 类比成注册函数，
  // addEdge 就是定义这些函数之间的调用路线。
  return new StateGraph(StateDemoGraphState)
    .addNode("normalizeQuestion", normalizeQuestionNode)
    .addNode("analyzeQuestion", analyzeQuestionNode)
    .addNode("generateAnswer", generateAnswerNode)
    // START 和 END 是 LangGraph 内置的图边界。
    // START 不是业务节点，只表示从哪里开始。
    // END 也不是业务节点，只表示流程到哪里结束。
    .addEdge(START, "normalizeQuestion")
    .addEdge("normalizeQuestion", "analyzeQuestion")
    .addEdge("analyzeQuestion", "generateAnswer")
    .addEdge("generateAnswer", END)
    // compile 之前只是“图定义”，compile 之后才是可执行对象。
    .compile();
}

async function main() {
  const graph = buildStateDemoGraph();
  // invoke 会执行整张图，并返回最终 State。
  //
  // 执行过程大概是：
  // 1. 初始 State 只有 question
  // 2. normalizeQuestion 写入 normalizedQuestion
  // 3. analyzeQuestion 写入 questionType
  // 4. generateAnswer 写入 answer
  // 5. steps 通过 reducer 持续追加日志
  const result = await graph.invoke({
    question: "  RAG 和普通聊天有什么区别？  ",
  });

  console.log("\n=== State Demo Graph Result ===");
  console.log(JSON.stringify(result, null, 2));
}

if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
