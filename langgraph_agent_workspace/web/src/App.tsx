import { useEffect, useState } from "react";

type ChatMessage = {
  role: "user" | "assistant";
  content: string;
};

type DebugEvent = {
  node: string;
  message: string;
  data?: unknown;
};

type LearningNote = {
  id: string;
  title: string;
  content: string;
  kind: string;
  tags: string[];
  createdAt: string;
};

type AgentDebugState = {
  // debugState 是后端专门返回给学习/调试面板看的数据。
  // 真实产品里可以隐藏这些字段，只保留 answer 和必要的消息历史。
  intent: string;
  routeReason: string;
  toolResult: string;
  retrievedNotes: LearningNote[];
  steps: string[];
  debug: DebugEvent[];
  messages: ChatMessage[];
};

type AgentChatResponse = {
  threadId: string;
  answer: string;
  debugState: AgentDebugState;
};

const API_BASE_URL = "http://localhost:3002";

function createInitialThreadId() {
  // threadId 是短期记忆的关键。
  // 前端保持同一个 threadId，后端 MemorySaver 才知道这些消息属于同一个会话。
  return `web-thread-${Date.now()}`;
}

export function App() {
  const [threadId, setThreadId] = useState(createInitialThreadId);
  const [input, setInput] = useState("我正在学习 LangGraph，帮我解释一下 State。");
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [debugState, setDebugState] = useState<AgentDebugState | null>(null);
  const [notes, setNotes] = useState<LearningNote[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
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

async function sendMessage() {
    const message = input.trim();
    if (!message) {
      setError("请输入要发送给 Agent 的内容。");
      return;
    }

    setLoading(true);
    setError("");
    setInput("");

    // 先把用户消息显示出来，让界面响应更自然。
    setMessages((current) => current.concat({ role: "user", content: message }));

    try {
      const response = await fetch(`${API_BASE_URL}/api/agent/chat`, {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify({
          message,
          // 保持同一个 threadId，后端 LangGraph 才能恢复同一个会话的短期记忆。
          threadId,
        }),
      });
      const data = await response.json() as AgentChatResponse & { error?: string };

      if (!response.ok) {
        throw new Error(data.error ?? "Agent 调用失败");
      }

      setThreadId(data.threadId);
      setDebugState(data.debugState);
      setMessages((current) => current.concat({
        role: "assistant",
        content: data.answer,
      }));

      // 如果本轮触发了保存笔记工具，重新加载右侧学习笔记。
      await loadNotes();
    } catch (sendError) {
      setError(
        sendError instanceof Error
          ? sendError.message
          : "Agent 调用失败",
      );
    } finally {
      setLoading(false);
    }
  }

  function startNewThread() {
    // 新会话使用新的 threadId，因此后端不会复用上一段短期记忆。
    setThreadId(createInitialThreadId());
    setMessages([]);
    setDebugState(null);
    setError("");
  }

  return (
    <main className="page-shell">
      <section className="hero">
        <p className="eyebrow">LangGraph Agent Workspace</p>
        <h1>把零散 Demo 收敛成一个可用 Agent</h1>
        <p>
          这里的主入口是聊天。State、条件分支、短期记忆和工具调用都在后端
          Agent Graph 内部发生，右侧调试面板用来观察它们。
        </p>
      </section>

      <section className="workspace">
        <aside className="side-panel">
          <div className="panel-header">
            <h2>会话</h2>
            <button className="ghost-button" onClick={startNewThread}>
              新会话
            </button>
          </div>

          <label>
            Thread ID
            <input
              value={threadId}
              onChange={(event) => setThreadId(event.target.value)}
            />
          </label>

          <div className="tips">
            <strong>可以试试：</strong>
            <button onClick={() => setInput("把这句话保存成学习笔记：State 是节点之间共享和流转的数据。")}>
              保存笔记
            </button>
            <button onClick={() => setInput("之前的笔记里有关于 State 的内容吗？")}>
              查询笔记
            </button>
            <button onClick={() => setInput("你还记得我刚才问了什么吗？")}>
              测试记忆
            </button>
          </div>

          <section className="notes-card">
            <h2>学习笔记</h2>
            {notes.length === 0 ? (
              <p className="muted">还没有笔记。可以让 Agent “保存这句话”。</p>
            ) : (
              <div className="note-list">
                {notes.map((note) => (
                  <article key={note.id} className="note-item">
                    <span>{note.kind}</span>
                    <h3>{note.title}</h3>
                    <p>{note.content}</p>
                  </article>
                ))}
              </div>
            )}
          </section>
        </aside>

        <section className="chat-panel">
          <div className="chat-header">
            <div>
              <h2>Agent 对话</h2>
              <p>统一入口：POST /api/agent/chat</p>
            </div>
            <span>{loading ? "运行中" : "就绪"}</span>
          </div>

          <div className="message-list">
            {messages.length === 0 ? (
              <div className="empty-state">
                <h3>开始一轮 Agent 对话</h3>
                <p>
                  发送普通问题、保存笔记，或者查询之前的学习内容。每一次能力增强都会加到这个 Agent 上。
                </p>
              </div>
            ) : (
              messages.map((message, index) => (
                <article
                  key={`${message.role}-${index}`}
                  className={`message ${message.role}`}
                >
                  <strong>{message.role === "user" ? "你" : "Agent"}</strong>
                  <p>{message.content}</p>
                </article>
              ))
            )}
          </div>

          {error && <p className="error">{error}</p>}

          <div className="composer">
            <textarea
              value={input}
              onChange={(event) => setInput(event.target.value)}
              onKeyDown={(event) => {
                if (event.key === "Enter" && !event.shiftKey) {
                  event.preventDefault();
                  void sendMessage();
                }
              }}
              placeholder="输入给 Agent 的问题，Enter 发送，Shift+Enter 换行"
              rows={3}
            />
            <button onClick={sendMessage} disabled={loading}>
              {loading ? "发送中..." : "发送"}
            </button>
          </div>
        </section>

        <aside className="debug-panel">
          <h2>调试面板</h2>
          <p className="muted">
            这里不是用户必须看的功能，而是学习 LangGraph 时用来观察 State 和路由的窗口。
          </p>

          {!debugState ? (
            <p className="muted">发送一条消息后，这里会显示本轮 Agent 执行链路。</p>
          ) : (
            <>
              <section className="debug-block">
                <h3>路由结果</h3>
                <dl>
                  <dt>intent</dt>
                  <dd>{debugState.intent}</dd>
                  <dt>reason</dt>
                  <dd>{debugState.routeReason}</dd>
                  <dt>tool</dt>
                  <dd>{debugState.toolResult || "未调用工具"}</dd>
                </dl>
              </section>

              <section className="debug-block">
                <h3>Steps</h3>
                <ol>
                  {debugState.steps.map((step, index) => (
                    <li key={`${step}-${index}`}>{step}</li>
                  ))}
                </ol>
              </section>

              <section className="debug-block">
                <h3>Debug JSON</h3>
                <pre>{JSON.stringify(debugState.debug, null, 2)}</pre>
              </section>
            </>
          )}
        </aside>
      </section>
    </main>
  );
}
