import { FormEvent, useState } from "react";
import { Bot, ChevronDown, MessageCircle, Send, Sparkles, X } from "lucide-react";
import { apiPost } from "../api/client";

type Evidence = { label: string; value: string };
type ChatResponse = {
  answer: string;
  intent: string;
  confidence: number;
  evidence: Evidence[];
  suggested_questions: string[];
  data_mode: string;
  synthetic_data: boolean;
  disclaimer: string;
};

type ChatItem = { role: "user" | "assistant"; content: string; evidence?: Evidence[]; confidence?: number };

const starters = [
  "Why is production risk high?",
  "Which equipment has the highest downtime?",
  "What is the 7-day production forecast?",
  "Which reserve area should we investigate first?",
];

export function MangaiAssistant() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [messages, setMessages] = useState<ChatItem[]>([
    { role: "assistant", content: "Hi. I’m MANGAI AI. Ask me about production risk, reserves, equipment, weather, blasting, recommendations or model health." },
  ]);

  async function ask(question: string) {
    const clean = question.trim();
    if (!clean || busy) return;
    setInput("");
    setMessages((items) => [...items, { role: "user", content: clean }]);
    setBusy(true);
    try {
      const response = await apiPost<ChatResponse>("/api/v1/chat", {
        message: clean,
        history: messages.slice(-12).map(({ role, content }) => ({ role, content })),
      });
      setMessages((items) => [
        ...items,
        { role: "assistant", content: response.answer, evidence: response.evidence, confidence: response.confidence },
      ]);
    } catch (error) {
      setMessages((items) => [...items, { role: "assistant", content: `I couldn’t reach the MANGAI intelligence service. ${error instanceof Error ? error.message : "Please check the backend."}` }]);
    } finally {
      setBusy(false);
    }
  }

  function submit(event: FormEvent) {
    event.preventDefault();
    void ask(input);
  }

  return (
    <>
      {open && (
        <section className="mangai-assistant" aria-label="MANGAI AI Assistant">
          <header className="mangai-assistant-head">
            <div className="assistant-identity">
              <div className="assistant-orb"><Bot size={18} /></div>
              <div><strong>MANGAI AI</strong><span>Evidence-first mining assistant</span></div>
            </div>
            <button className="assistant-icon-btn" onClick={() => setOpen(false)} aria-label="Close assistant"><X size={17} /></button>
          </header>

          <div className="assistant-status"><span className="assistant-live-dot" /> Intelligence services connected <small>DEMO MODE</small></div>

          <div className="assistant-messages">
            {messages.map((message, index) => (
              <article key={`${message.role}-${index}`} className={`assistant-message ${message.role}`}>
                <div className="assistant-bubble">{message.content}</div>
                {message.evidence && message.evidence.length > 0 && (
                  <div className="assistant-evidence">
                    {message.evidence.slice(0, 5).map((item) => <div key={`${item.label}-${item.value}`}><span>{item.label}</span><strong>{item.value}</strong></div>)}
                  </div>
                )}
                {message.confidence !== undefined && <span className="assistant-confidence">Evidence confidence {(message.confidence * 100).toFixed(0)}%</span>}
              </article>
            ))}
            {busy && <article className="assistant-message assistant"><div className="assistant-bubble assistant-thinking"><Sparkles size={14} /> Analysing MANGAI signals<span className="thinking-dots">•••</span></div></article>}
          </div>

          {messages.length < 3 && (
            <div className="assistant-starters">
              {starters.map((question) => <button key={question} onClick={() => void ask(question)}>{question}</button>)}
            </div>
          )}

          <form className="assistant-input" onSubmit={submit}>
            <input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask MANGAI..." maxLength={2000} />
            <button type="submit" disabled={busy || !input.trim()} aria-label="Send question"><Send size={16} /></button>
          </form>
          <footer className="assistant-disclaimer">Decision support only · Human approval required for operational action.</footer>
        </section>
      )}
      <button className={`mangai-assistant-launcher ${open ? "open" : ""}`} onClick={() => setOpen((value) => !value)} aria-label={open ? "Close MANGAI AI" : "Open MANGAI AI"}>
        {open ? <ChevronDown size={20} /> : <MessageCircle size={21} />}
        {!open && <span>MANGAI AI</span>}
      </button>
    </>
  );
}
