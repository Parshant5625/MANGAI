import { useEffect, useState } from "react";
import { Bell, Bot, Database, Gauge, MapPinned, ShieldCheck, Wifi } from "lucide-react";
import { IntelligencePage, IntelligenceSidebar } from "./IntelligenceSidebar";

const pageLabels: Record<IntelligencePage, string> = {
  overview: "Overview",
  reserve: "Reserve",
  production: "Production",
  operations: "Operations",
  equipment: "Equipment",
  weather: "Weather",
  actions: "Actions",
  models: "Health",
  data: "Health",
  settings: "Settings",
};

function clickLegacyNavigation(label: string) {
  const button = Array.from(document.querySelectorAll<HTMLButtonElement>(".nav-item"))
    .find((item) => item.textContent?.trim().toLowerCase() === label.toLowerCase());
  button?.click();
}

export function MangaiChrome() {
  const [activePage, setActivePage] = useState<IntelligencePage>("overview");
  const [collapsed, setCollapsed] = useState(false);
  const [time, setTime] = useState(() => new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));

  useEffect(() => {
    const timer = window.setInterval(() => {
      setTime(new Date().toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" }));
    }, 30_000);
    return () => window.clearInterval(timer);
  }, []);

  const navigate = (page: IntelligencePage) => {
    setActivePage(page);
    if (page === "models") clickLegacyNavigation("Health");
    else if (page === "data") clickLegacyNavigation("Health");
    else clickLegacyNavigation(pageLabels[page]);
  };

  const openCopilot = () => {
    const launcher = document.querySelector<HTMLButtonElement>(".mangai-assistant-launcher");
    launcher?.click();
  };

  return (
    <>
      <header className="mangai-command-bar">
        <div className="mangai-command-brand">
          <div className="mangai-command-mark">M</div>
          <div>
            <div className="mangai-command-title">MANGAI</div>
            <div className="mangai-command-subtitle">MINING INTELLIGENCE</div>
          </div>
        </div>
        <div className="mangai-command-context">
          <div className="mangai-command-context-item"><MapPinned size={14}/><span className="mangai-command-label">MINE</span><strong>MANGAI DEMO MINE</strong></div>
          <div className="mangai-command-context-item"><Database size={14}/><span className="mangai-command-label">DATA</span><strong className="is-demo">DEMO</strong></div>
        </div>
        <div className="mangai-command-statuses">
          <div className="mangai-status-pill"><span className="mangai-status-dot healthy"/>OPERATIONAL</div>
          <div className="mangai-status-pill compact"><Wifi size={13}/>AI READY</div>
          <div className="mangai-status-pill compact">{time}</div>
        </div>
        <div className="mangai-command-actions">
          <button className="mangai-icon-button" type="button" aria-label="Notifications"><Bell size={17}/></button>
          <button className="mangai-copilot-button" type="button" onClick={openCopilot}><Bot size={17}/>AI COPILOT</button>
          <div className="mangai-command-safety"><ShieldCheck size={15}/>DECISION SUPPORT</div>
        </div>
      </header>
      <div className={`mangai-overlay-sidebar ${collapsed ? "is-collapsed" : ""}`}>
        <IntelligenceSidebar activePage={activePage} onNavigate={navigate} collapsed={collapsed}/>
        <button className="mangai-overlay-collapse" type="button" onClick={() => setCollapsed((v) => !v)} aria-label="Toggle navigation">
          {collapsed ? "→" : "←"}
        </button>
        <div className="mangai-overlay-identity"><Gauge size={13}/><span>SIH 26009</span></div>
      </div>
    </>
  );
}
