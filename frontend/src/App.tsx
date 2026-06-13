import { useState } from "react";
import "./App.css";

// 与后端 medrag/api.py 的 pydantic 模型一一对应
interface Source {
  source: string;
  score: number;
}
interface AskResponse {
  answer: string;
  sources: Source[];
}

function App() {
  const [question, setQuestion] = useState("");
  const [loading, setLoading] = useState(false);
  const [result, setResult] = useState<AskResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function ask() {
    if (!question.trim() || loading) return;
    setLoading(true);
    setError(null);
    setResult(null);
    try {
      // 走 Rust 网关的 /api/ask(开发模式由 vite proxy 转发,见 vite.config.ts)
      const res = await fetch("/api/ask", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ question, k: 4 }),
      });
      if (!res.ok) throw new Error(`HTTP ${res.status}`);
      setResult((await res.json()) as AskResponse);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="container">
      <h1>med-rag-assistant</h1>
      <p className="subtitle">医疗文档 RAG 问答(仅合成数据演示)</p>

      <div className="ask-row">
        <input
          value={question}
          onChange={(e) => setQuestion(e.target.value)}
          onKeyDown={(e) => e.key === "Enter" && ask()}
          placeholder="例如:主要诊断是什么?"
          disabled={loading}
        />
        <button onClick={ask} disabled={loading || !question.trim()}>
          {loading ? "检索中…" : "提问"}
        </button>
      </div>

      {error && <div className="error">请求失败:{error}</div>}

      {result && (
        <section className="result">
          <h2>答案</h2>
          <pre className="answer">{result.answer}</pre>
          <h2>命中来源</h2>
          <ul className="sources">
            {result.sources.map((s, i) => (
              <li key={i}>
                <span className="source-name">{s.source}</span>
                <span className="source-score">相似度 {s.score.toFixed(3)}</span>
              </li>
            ))}
          </ul>
        </section>
      )}
    </main>
  );
}

export default App;
