import { useEffect, useState } from "react";

type GraphInfo = {
  name: string;
  title: string;
  description: string;
  defaultInput: {
    question: string;
    threadId?: string;
  };
};

type InvokeResponse = {
  graph: string;
  input: {
    question: string;
    threadId?: string;
  };
  result: Record<string, unknown>;
};

type LearningNoteKind = "concept" | "observation" | "question" | "todo";

type LearningNote = {
  id: string;
  title: string;
  content: string;
  kind: LearningNoteKind;
  tags: string[];
  createdAt: string;
  updatedAt: string;
};

const API_BASE_URL = "http://localhost:3001";

export function App() {
  const [graphs, setGraphs] = useState<GraphInfo[]>([]);
  const [selectedGraphName, setSelectedGraphName] = useState("");
  const [question, setQuestion] = useState("");
  const [threadId, setThreadId] = useState("");
  const [result, setResult] = useState<InvokeResponse | null>(null);
  const [notes, setNotes] = useState<LearningNote[]>([]);
  const [noteTitle, setNoteTitle] = useState("");
  const [noteContent, setNoteContent] = useState("");
  const [noteKind, setNoteKind] = useState<LearningNoteKind>("observation");
  const [noteTags, setNoteTags] = useState("langgraph");
  const [loading, setLoading] = useState(false);
  const [savingNote, setSavingNote] = useState(false);
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
          setThreadId(firstGraph.defaultInput.threadId ?? "");
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
    void loadNotes();
  }, []);

  async function loadNotes() {
    try {
      const response = await fetch(`${API_BASE_URL}/api/learning/notes`);
      const data = await response.json() as { notes: LearningNote[] };
      setNotes(data.notes);
    } catch (loadError) {
      setError(
        loadError instanceof Error
          ? loadError.message
          : "加载学习笔记失败",
      );
    }
  }

  function handleGraphChange(nextGraphName: string) {
    const nextGraph = graphs.find((graph) => graph.name === nextGraphName);
    setSelectedGraphName(nextGraphName);
    setQuestion(nextGraph?.defaultInput.question ?? "");
    setThreadId(nextGraph?.defaultInput.threadId ?? "");
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
            threadId: threadId || undefined,
          }),
        },
      );
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "执行 Graph 失败");
      }

      setResult(data as InvokeResponse);
      setNoteTitle(`观察：运行 ${selectedGraphName} Graph`);
      setNoteContent(
        [
          `问题：${question}`,
          "",
          "执行结果：",
          JSON.stringify((data as InvokeResponse).result, null, 2),
        ].join("\n"),
      );
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

  async function saveNote() {
    if (!noteTitle.trim() || !noteContent.trim()) {
      setError("请填写笔记标题和内容。");
      return;
    }

    setSavingNote(true);
    setError("");

    try {
      const response = await fetch(`${API_BASE_URL}/api/learning/notes`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          title: noteTitle,
          content: noteContent,
          kind: noteKind,
          tags: noteTags.split(","),
        }),
      });
      const data = await response.json();
      if (!response.ok) {
        throw new Error(data.error ?? "保存学习笔记失败");
      }

      setNoteTitle("");
      setNoteContent("");
      setNoteKind("observation");
      setNoteTags("langgraph");
      await loadNotes();
    } catch (saveError) {
      setError(
        saveError instanceof Error
          ? saveError.message
          : "保存学习笔记失败",
      );
    } finally {
      setSavingNote(false);
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
          这个页面会逐步变成 LangGraph 学习工作台：左侧运行 Graph，
          右侧观察 State，并把每次理解到的内容沉淀成学习笔记。
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

          {selectedGraphName === "memory" && (
            <label>
              Thread ID
              <input
                value={threadId}
                onChange={(event) => setThreadId(event.target.value)}
                placeholder="demo-thread"
              />
            </label>
          )}

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

      <section className="learning-grid">
        <section className="panel note-editor">
          <h2>学习沉淀</h2>
          <p className="description">
            运行 Graph 后，可以把本次观察保存成笔记。也可以手动记录概念、问题和待办。
          </p>

          <label>
            标题
            <input
              value={noteTitle}
              onChange={(event) => setNoteTitle(event.target.value)}
              placeholder="例如：State 是如何被局部更新的"
            />
          </label>

          <label>
            类型
            <select
              value={noteKind}
              onChange={(event) => setNoteKind(event.target.value as LearningNoteKind)}
            >
              <option value="observation">观察记录</option>
              <option value="concept">概念理解</option>
              <option value="question">待澄清问题</option>
              <option value="todo">后续任务</option>
            </select>
          </label>

          <label>
            标签，逗号分隔
            <input
              value={noteTags}
              onChange={(event) => setNoteTags(event.target.value)}
              placeholder="langgraph,state"
            />
          </label>

          <label>
            内容
            <textarea
              value={noteContent}
              onChange={(event) => setNoteContent(event.target.value)}
              rows={8}
              placeholder="写下这次运行看到的现象、理解到的概念，或者还没搞懂的问题。"
            />
          </label>

          <button onClick={saveNote} disabled={savingNote}>
            {savingNote ? "保存中..." : "保存到学习笔记"}
          </button>
        </section>

        <section className="panel notes-panel">
          <div className="panel-header">
            <h2>学习笔记</h2>
            <span>{notes.length} 条</span>
          </div>

          {notes.length === 0 ? (
            <p className="muted">还没有笔记。先运行一个 Graph，然后保存观察结果。</p>
          ) : (
            <div className="note-list">
              {notes.map((note) => (
                <article className="note-card" key={note.id}>
                  <div className="note-meta">
                    <strong>{note.kind}</strong>
                    <span>{new Date(note.createdAt).toLocaleString()}</span>
                  </div>
                  <h3>{note.title}</h3>
                  <p>{note.content}</p>
                  {note.tags.length > 0 && (
                    <div className="tags">
                      {note.tags.map((tag) => (
                        <span key={tag}>{tag}</span>
                      ))}
                    </div>
                  )}
                </article>
              ))}
            </div>
          )}
        </section>
      </section>
    </main>
  );
}
