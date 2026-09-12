import { Bell, Bot, Clock3, Database, MapPinned, ShieldCheck, Wifi } from "lucide-react";

export interface CommandBarProps {
  siteName?: string;
  dataMode?: "DEMO" | "LIVE";
  modelVersion?: string;
  systemStatus?: string;
  onOpenCopilot?: () => void;
}

export function CommandBar({
  siteName = "MANGAI Demo Mine",
  dataMode = "DEMO",
  modelVersion = "AI READY",
  systemStatus = "OPERATIONAL",
  onOpenCopilot,
}: CommandBarProps) {
  const now = new Date();
  const time = now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });

  return (
    <header className="mangai-command-bar">
      <div className="mangai-command-brand">
        <div className="mangai-command-mark">M</div>
        <div>
          <div className="mangai-command-title">MANGAI</div>
          <div className="mangai-command-subtitle">MINING INTELLIGENCE</div>
        </div>
      </div>

      <div className="mangai-command-context">
        <div className="mangai-command-context-item"><MapPinned size={14} /><span className="mangai-command-label">MINE</span><strong>{siteName}</strong></div>
        <div className="mangai-command-context-item"><Database size={14} /><span className="mangai-command-label">DATA</span><strong className={dataMode === "LIVE" ? "is-live" : "is-demo"}>{dataMode}</strong></div>
      </div>

      <div className="mangai-command-statuses">
        <div className="mangai-status-pill"><span className="mangai-status-dot healthy" /><span>{systemStatus}</span></div>
        <div className="mangai-status-pill compact"><Wifi size={13} /><span>{modelVersion}</span></div>
        <div className="mangai-status-pill compact"><Clock3 size={13} /><span>{time}</span></div>
      </div>

      <div className="mangai-command-actions">
        <button className="mangai-icon-button" type="button" aria-label="Notifications"><Bell size={17} /></button>
        <button className="mangai-copilot-button" type="button" onClick={onOpenCopilot}><Bot size={17} /><span>AI COPILOT</span></button>
        <div className="mangai-command-safety"><ShieldCheck size={15} /><span>DECISION SUPPORT</span></div>
      </div>
    </header>
  );
}
