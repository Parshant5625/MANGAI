import { FormEvent, useEffect, useMemo, useState } from "react";
import { Bot, ChevronDown, Copy, ExternalLink, MessageCircle, RotateCcw, Send, Sparkles, X } from "lucide-react";
import { apiPost } from "../api/client";

type Evidence = { label: string; value: string };
type AssistantAction = { label: string; action: string };
type ChatResponse = { answer: string; intent: string; confidence: number; evidence: Evidence[]; suggested_questions: string[]; actions: AssistantAction[]; data_mode: string; synthetic_data: boolean; disclaimer: string };
type ChatItem = { role: "user" | "assistant"; content: string; evidence?: Evidence[]; confidence?: number; actions?: AssistantAction[] };

const starters = ["Why is production risk high?", "Which equipment has the highest downtime?", "What is the 7-day production forecast?", "Which reserve area should we investigate first?"];
const routeForAction: Record<string, string> = { overview: "Overview", production: "Production", reserve: "Reserve", "reserve-thermal": "Reserve", equipment: "Equipment", weather: "Weather", recommendations: "Actions", operations: "Operations", health: "Health" };

export function MangaiAssistant() {
  const [open, setOpen] = useState(false);
  const [input, setInput] = useState("");
  const [busy, setBusy] = useState(false);
  const [copied, setCopied] = useState<number | null>(null);
  const [pageContext, setPageContext] = useState(() => window.location.hash || "overview");
  const [messages, setMessages] = useState<ChatItem[]>([{ role: "assistant", content: "MANGAI AI online. I can investigate the mine intelligence graph, explain risk drivers, surface evidence and take you to the relevant intelligence view." }]);

  useEffect(() => {
    const update = () => setPageContext(window.location.hash || "overview");
    window.addEventListener("popstate", update);
    window.addEventListener("hashchange", update);
    return () => { window.removeEventListener("popstate", update); window.removeEventListener("hashchange", update); };
  }, []);

  const visibleSuggestions = useMemo(() => {
    const last = messages[messages.length - 1];
    return last?.role === "assistant" && last.actions?.length ? last.actions.slice(0, 4) : starters;
  }, [messages]);

  async function ask(question: string) {
    const clean = question.trim();
    if (!clean || busy) return;
    const history = messages.slice(-12).map(({ role, content }) => ({ role, content }));
    setInput("");
    setMessages((items) => [...items, { role: "user", content: clean }]);
    setBusy(true);
    try {
      const response = await apiPost<ChatResponse>("/api/v1/chat", { message: clean, history, page_context: pageContext });
      setMessages((items) => [...items, { role: "assistant", content: response.answer, evidence: response.evidence, confidence: response.confidence, actions: response.actions }]);
    } catch (error) {
      setMessages((items) => [...items, { role: "assistant", content: `I couldn't reach the MANGAI intelligence service. ${error instanceof Error ? error.message : "Please check the backend."}` }]);
    } finally { setBusy(false); }
  }

  function executeAction(action: string) {
    if (action === "reserve-thermal") { void ask("Show me the thermal reserve view."); return; }
    const label = routeForAction[action];
    if (!label) return;
    const navButton = Array.from(document.querySelectorAll<HTMLButtonElement>(".nav-item")).find((button) => button.textContent?.trim().toLowerCase() === label.toLowerCase());
    if (navButton) navButton.click();
  }

  function submit(event: FormEvent) { event.preventDefault(); void ask(input); }
  async function copyAnswer(content: string, index: number) { try { await navigator.clipboard.writeText(content); setCopied(index); window.setTimeout(() => setCopied(null), 1200); } catch { /* optional browser permission */ } }
  function reset() { setMessages([{ role: "assistant", content: "Conversation reset. MANGAI AI is ready for a fresh investigation." }]); }

  return <>
    {open && <section className="mangai-assistant" aria-label="MANGAI AI Assistant">
      <header className="mangai-assistant-head">
        <div className="assistant-identity"><div className="assistant-orb"><Bot size={18}/></div><div><strong>MANGAI AI</strong><span>Mining Intelligence Copilot · context: {pageContext.replace("#", "") || "overview"}</span></div></div>
        <div className="assistant-head-actions"><button className="assistant-icon-btn" onClick={reset} aria-label="Reset conversation"><RotateCcw size={15}/></button><button className="assistant-icon-btn" onClick={() => setOpen(false)} aria-label="Close assistant"><X size={17}/></button></div>
      </header>
      <div className="assistant-status"><span className="assistant-live-dot"/> Intelligence services connected <small>DATA-GROUNDED</small></div>
      <div className="assistant-messages">
        {messages.map((message, index) => <article key={`${message.role}-${index}`} className={`assistant-message ${message.role}`}>
          <div className="assistant-bubble-wrap"><div className="assistant-bubble">{message.content}</div>{message.role === "assistant" && <button className="assistant-copy" onClick={() => void copyAnswer(message.content, index)} aria-label="Copy answer">{copied === index ? "✓" : <Copy size={12}/>}</button>}</div>
          {message.evidence && message.evidence.length > 0 && <div className="assistant-evidence">{message.evidence.slice(0, 6).map((item) => <div key={`${item.label}-${item.value}`}><span>{item.label}</span><strong>{item.value}</strong></div>)}</div>}
          {message.confidence !== undefined && <div className="assistant-confidence"><span>Evidence confidence</span><strong>{(message.confidence * 100).toFixed(0)}%</strong></div>}
          {message.actions && message.actions.length > 0 && <div className="assistant-actions">{message.actions.slice(0, 4).map((item) => <button key={`${item.label}-${item.action}`} onClick={() => executeAction(item.action)}>{item.label}<ExternalLink size={10}/></button>)}</div>}
        </article>)}
        {busy && <article className="assistant-message assistant"><div className="assistant-bubble assistant-thinking"><Sparkles size={14}/> Correlating MANGAI signals<span className="thinking-dots">•••</span></div></article>}
      </div>
      {!busy && <div className="assistant-starters">{visibleSuggestions.map((item) => typeof item === "string" ? <button key={item} onClick={() => void ask(item)}>{item}</button> : <button key={item.action} onClick={() => executeAction(item.action)}>{item.label}<ExternalLink size={10}/></button>)}</div>}
      <form className="assistant-input" onSubmit={submit}><input value={input} onChange={(event) => setInput(event.target.value)} placeholder="Ask MANGAI about the mine..." maxLength={2000}/><button type="submit" disabled={busy || !input.trim()} aria-label="Send question"><Send size={16}/></button></form>
      <footer className="assistant-disclaimer">Decision support only · Human approval required · Demo/live mode is always disclosed.</footer>
    </section>}
    <button className={`mangai-assistant-launcher ${open ? "open" : ""}`} onClick={() => setOpen((value) => !value)} aria-label={open ? "Close MANGAI AI" : "Open MANGAI AI"}>{open ? <ChevronDown size={20}/> : <MessageCircle size={21}/>} {!open && <span>MANGAI AI</span>}</button>
  </>;
}
