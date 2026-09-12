import { ReactNode, useState } from "react";
import { CommandBar } from "../command/CommandBar";
import { IntelligencePage, IntelligenceSidebar } from "./IntelligenceSidebar";

interface MangaiShellProps {
  activePage: IntelligencePage;
  onNavigate: (page: IntelligencePage) => void;
  onOpenCopilot?: () => void;
  children: ReactNode;
}

export function MangaiShell({ activePage, onNavigate, onOpenCopilot, children }: MangaiShellProps) {
  const [sidebarCollapsed, setSidebarCollapsed] = useState(false);

  return (
    <div className="mangai-shell">
      <CommandBar onOpenCopilot={onOpenCopilot} />
      <div className="mangai-app-body">
        <IntelligenceSidebar
          activePage={activePage}
          onNavigate={onNavigate}
          collapsed={sidebarCollapsed}
        />
        <div className="mangai-main-wrap">
          <div className="mangai-shell-toolbar">
            <button
              type="button"
              className="mangai-sidebar-toggle"
              onClick={() => setSidebarCollapsed((value) => !value)}
              aria-label={sidebarCollapsed ? "Expand navigation" : "Collapse navigation"}
            >
              {sidebarCollapsed ? "→" : "←"}
            </button>
            <span>INTELLIGENCE OPERATING SYSTEM</span>
            <span className="mangai-shell-live"><i /> SYNCED</span>
          </div>
          <main className="mangai-main">{children}</main>
        </div>
      </div>
    </div>
  );
}
