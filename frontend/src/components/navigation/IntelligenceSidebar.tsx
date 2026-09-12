import {
  Activity,
  BarChart3,
  CloudRain,
  Database,
  Gauge,
  Map,
  Settings,
  ShieldCheck,
  Target,
  Wrench,
  Zap,
} from "lucide-react";

export type IntelligencePage =
  | "overview"
  | "reserve"
  | "production"
  | "operations"
  | "equipment"
  | "weather"
  | "actions"
  | "models"
  | "data"
  | "settings";

interface IntelligenceSidebarProps {
  activePage: IntelligencePage;
  onNavigate: (page: IntelligencePage) => void;
  collapsed?: boolean;
}

const commandItems = [
  { id: "overview", label: "Command Center", icon: Gauge },
  { id: "reserve", label: "Reserve Intelligence", icon: Target },
  { id: "production", label: "Production Intelligence", icon: BarChart3 },
  { id: "operations", label: "Operations", icon: Activity },
  { id: "equipment", label: "Equipment", icon: Wrench },
  { id: "weather", label: "Weather & Blasting", icon: CloudRain },
  { id: "actions", label: "AI Action Center", icon: Zap },
] as const;

const systemItems = [
  { id: "models", label: "Model Health", icon: ShieldCheck },
  { id: "data", label: "Data Health", icon: Database },
  { id: "settings", label: "Settings", icon: Settings },
] as const;

export function IntelligenceSidebar({ activePage, onNavigate, collapsed = false }: IntelligenceSidebarProps) {
  const renderItems = (items: readonly { id: IntelligencePage; label: string; icon: typeof Gauge }[]) =>
    items.map(({ id, label, icon: Icon }) => (
      <button
        key={id}
        type="button"
        className={`mangai-nav-item ${activePage === id ? "is-active" : ""}`}
        onClick={() => onNavigate(id)}
        aria-current={activePage === id ? "page" : undefined}
        title={collapsed ? label : undefined}
      >
        <span className="mangai-nav-icon"><Icon size={17} /></span>
        {!collapsed && <span className="mangai-nav-text">{label}</span>}
        {activePage === id && <span className="mangai-nav-state" aria-hidden="true" />}
      </button>
    ));

  return (
    <aside className={`mangai-sidebar ${collapsed ? "is-collapsed" : ""}`}>
      <div className="mangai-sidebar-section">
        {!collapsed && <div className="mangai-sidebar-kicker">COMMAND</div>}
        <nav aria-label="Intelligence navigation">{renderItems(commandItems)}</nav>
      </div>

      <div className="mangai-sidebar-section system">
        {!collapsed && <div className="mangai-sidebar-kicker">SYSTEM</div>}
        <nav aria-label="System navigation">{renderItems(systemItems)}</nav>
      </div>

      <div className="mangai-sidebar-footer">
        <div className="mangai-sidebar-footer-line" />
        {!collapsed && (
          <>
            <span>SIH 26009</span>
            <span>MOIL / MINISTRY OF STEEL</span>
          </>
        )}
      </div>
    </aside>
  );
}
