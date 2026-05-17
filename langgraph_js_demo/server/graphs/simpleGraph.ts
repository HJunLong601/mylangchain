import { Annotation, END, START, StateGraph } from "@langchain/langgraph";
import { pathToFileURL } from "node:url";

// 这是第一个最小 LangGraph 示例。
//
// 先不要急着接模型、工具或 RAG。第一步只需要看懂三件事：
// 1. State: 图运行过程中共享的数据
// 2. Node: 读取 State，并返回 State 的局部更新
// 3. Edge: 控制节点之间的执行顺序
//
// 注意：这个文件不会调用大模型。
// 目前没有 ChatOpenAI，也没有 GLM 请求，所有结果都是本地函数生成的。
// 这样做是为了先把 LangGraph 的基本方法看清楚。

// Annotation.Root 用来声明这个图的 State 结构。
// 你可以把它理解成“这张图执行期间允许保存哪些字段”。
// 后续每个 Node 读取的 state、返回的局部更新，都会围绕这个结构来。
const SimpleGraphState = Annotation.Root({
  // 用户输入的问题。
  question: Annotation<string>,
  // 图最终生成的回答。
  answer: Annotation<string>,
  // 用来观察节点执行顺序的日志。
  steps: Annotation<string[]>({
    // reducer 用来告诉 LangGraph：
    // 如果多个 Node 都返回 steps，应该如何合并。
    //
    // 这里我们希望日志不断追加，而不是后一个节点覆盖前一个节点，
    // 所以使用 concat。
    reducer: (currentSteps, newSteps) => currentSteps.concat(newSteps),
    // default 是这个字段的初始值。
    // 如果 invoke 时没有传 steps，就从空数组开始。
    default: () => [],
  }),
});

// typeof SimpleGraphState.State 可以推导出 State 的 TypeScript 类型。
// 后面 Node 函数写 state 参数时使用这个类型，就能获得类型提示。
type SimpleGraphStateType = typeof SimpleGraphState.State;

function receiveQuestionNode(state: SimpleGraphStateType) {
  // Node 不需要修改整个 State，只返回自己负责更新的字段。
  // LangGraph 会把这个局部更新合并回共享 State。
  return {
    steps: [`receiveQuestion: ${state.question}`],
  };
}

function generateAnswerNode(state: SimpleGraphStateType) {
  return {
    answer: `这是一个最小 LangGraph 回答。你刚才的问题是：${state.question}`,
    steps: ["generateAnswer: 已根据 question 生成 answer"],
  };
}

export function buildSimpleGraph() {
  // StateGraph 负责把 State、Node、Edge 组装成一张可执行的图。
  //
  // 可以按这个顺序理解下面这些方法：
  // - new StateGraph(...): 创建一张“使用某个 State 结构”的图
  // - addNode(...): 注册处理函数，但不决定执行顺序
  // - addEdge(...): 定义节点之间的执行路线
  // - compile(): 把图结构编译成可以 invoke 的 runnable
  return new StateGraph(SimpleGraphState)
    // addNode: 注册节点。第一个参数是节点名称，第二个参数是节点函数。
    //
    // 注意：addNode 只是告诉 LangGraph“我有这些节点”。
    // 如果只 addNode，不 addEdge，LangGraph 仍然不知道谁先执行、谁后执行。
    .addNode("receiveQuestion", receiveQuestionNode)
    .addNode("generateAnswer", generateAnswerNode)
    // addEdge: 注册节点之间的执行顺序。
    // START 和 END 是 LangGraph 内置的起点和终点。
    //
    // .addEdge("A", "B") 的意思是：
    // A 执行完以后，接着执行 B。
    //
    // 下面这几行组合起来就是：
    // START -> receiveQuestion -> generateAnswer -> END
    .addEdge(START, "receiveQuestion")
    .addEdge("receiveQuestion", "generateAnswer")
    .addEdge("generateAnswer", END)
    // compile 后才会得到可以 invoke 的 graph。
    .compile();
}

async function main() {
  const graph = buildSimpleGraph();
  // invoke 会真正执行这张图。
  //
  // 这里传入的对象会作为初始 State：
  // {
  //   question: "LangGraph 的 State 是什么？"
  // }
  //
  // 后续每个节点返回局部更新，LangGraph 会自动合并成最终 State。
  const result = await graph.invoke({
    question: "LangGraph 的 State 是什么？",
  });

  console.log("\n=== Simple Graph Result ===");
  console.log(JSON.stringify(result, null, 2));
}

// 只有直接运行这个文件时才执行 main()。
// 后续被 API 服务 import 时，不会自动运行。
if (import.meta.url === pathToFileURL(process.argv[1]).href) {
  await main();
}
