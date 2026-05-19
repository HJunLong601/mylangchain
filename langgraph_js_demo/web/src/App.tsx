import { useEffect, useState } from "react";

type GraphInfo = {
  name: string;
  title: string;
  description: string;
  defaultInput: {
    question: string;
  };
};

type InvokeResponse = {
  graph: string;
  input: {
    question: string;
  };
  result: Record<string, unknown>;
};

const API_BASE_URL = "http://localhost:3001";

export function App() {
  const [graphs, setGraphs] = useState<GraphInfo[]>([]);
  const [selectedGraphName, setSelectedGraphName] = useState("");
  const [question, setQuestion] = useState("");
  const [result, setResult] = useState<InvokeResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const selectedGraph = graphs.find((graph) => graph.name === selectedGraphName);

  useEffect(() => {
    async function loadGraphs() {
      try {
        const response = await fetch(`${API_BASE_URL}/api/graphs`);
        const data = await response.json() as { graphs: GraphInfo[] };
        setGraphs(data.graphs);

        const firstGraph = data.graphs[0];
        if (firstGraph) {
          setSelectedGraphName(firstGraph.name);
          setQuestion(firstGraph.defaultInput.question);
        }
      } catch (loadError) {
        setError(
          loadError instanceof Error
            ? loadError.message
            : "加载 graph 列表失败",
        );
      }
    }

    void loadGraphs();
  }, []);

  function handleGraphChange(nextGraphName: string) {
    const nextGraph = graphs.find((graph) => graph.name === nextGraphName);
    setSelectedGraphName(nextGraphName);
    setQuestion(nextGraph?.defaultInput.question ?? "");
    setResult(null);
    setError("");
  }

  async function runGraph() {
    if (!selectedGraphName || !question.trim()) {
      setError("请选择 Graph 并输入问题。");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const response = await fetch(
        `${API_BASE_URL}/api/graphs/${selectedGraphName}/invoke`,
        {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            question,
          }),
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "执行 Graph 失败");
      }

      setResult(data as InvokeResponse);
    } catch (runError) {
      setError(
        runError instanceof Error
          ? runError.message
          : "执行 Graph 失败",
      );
    } finally {
      setLoading(false);
    }
  }

  const steps = Array.isArray(result?.result.steps)
    ? result.result.steps
    : [];

  return (
    <main className="app-shell">
      <section className="hero">
        <p className="eyebrow">LangGraph JS/TS Demo</p>
        <h1>用可视化界面观察 Graph、State 和执行结果</h1>
        <p>
          当前页面先完成最小可用版本：选择一个 Graph，输入问题，调用后端 API，
          查看最终 State。下一步会接入 React Flow 展示节点和边。
        </p>
      </section>

      <section className="workspace">
        <aside className="panel control-panel">
          <h2>运行 Graph</h2>

          <label>
            Graph
            <select
              value={selectedGraphName}
              onChange={(event) => handleGraphChange(event.target.value)}
            >
              {graphs.map((graph) => (
                <option key={graph.name} value={graph.name}>
                  {graph.title}
                </option>
              ))}
            </select>
          </label>

          {selectedGraph && (
            <p className="description">{selectedGraph.description}</p>
          )}

          <label>
            Question
            <textarea
              value={question}
              onChange={(event) => setQuestion(event.target.value)}
              rows={5}
            />
          </label>

          <button onClick={runGraph} disabled={loading}>
            {loading ? "运行中..." : "运行 Graph"}
          </button>

          {error && <p className="error">{error}</p>}
        </aside>

        <section className="panel result-panel">
          <div className="panel-header">
            <h2>执行结果</h2>
            <span>{result ? result.graph : "等待运行"}</span>
          </div>

          <div className="result-grid">
            <div>
              <h3>Steps</h3>
              {steps.length > 0 ? (
                <ol className="steps">
                  {steps.map((step, index) => (
                    <li key={`${step}-${index}`}>{String(step)}</li>
                  ))}
                </ol>
              ) : (
                <p className="muted">暂无执行日志。</p>
              )}
            </div>

            <div>
              <h3>State JSON</h3>
              <pre>{JSON.stringify(result?.result ?? {}, null, 2)}</pre>
            </div>
          </div>
        </section>
      </section>
    </main>
  );
}
