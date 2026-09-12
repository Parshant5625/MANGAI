import { useState } from "react";
import { CommandBar } from "../command/CommandBar";
import { IntelligenceSidebar, type IntelligencePage } from "./IntelligenceSidebar";

const pageMap: Record<IntelligencePage, string> = {
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

export function MangaiChrome() {
  const [activePage, setActivePage] = useState<IntelligencePage>("overview");

  const navigate = (page: IntelligencePage) => {
    setActivePage(page);
    const label = pageMap[page];
    const target = Array.from(document.querySelectorAll<HTMLButtonElement>(".app .nav-item"))
      .find((button) => button.textContent?.trim().toLowerCase() === label.toLowerCase());
    target?.click();
    window.dispatchEvent(new CustomEvent("mangai:navigate", { detail: { page } }));
  };

  return (
    <div className="mangai-zero-chrome">
      <CommandBar />
      <div className="mangai-zero-layout">
        <IntelligenceSidebar activePage={activePage} onNavigate={navigate} />
        <div className="mangai-zero-content-spacer" aria-hidden="true" />
      </div>
    </div>
  );
}
