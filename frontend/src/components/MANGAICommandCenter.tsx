import { useState, type ReactNode } from "react";
import { Activity, AlertTriangle, BarChart3, Bot, CheckCircle2, Database, FileText, Gauge, Map as MapIcon, MessageSquare, RefreshCw, Settings, ShieldCheck, Target, Truck, X } from "lucide-react";
import { ResponsiveContainer, CartesianGrid, LineChart, Line, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../hooks/useApi";
import { ReserveMap } from "./ReserveMap";
import { ModelMonitoringPanel } from "./ModelMonitoringPanel";
import { OperationsPage } from "../pages/OperationsPage";
import { ProductionForecastTerminal } from "./production/ProductionForecastTerminal";
import type { DataQualityResponse, EquipmentResponse, ModelMonitoringResponse, ModelRegistryResponse, ProductionForecastResponse, ProductionHistoryRecord, ProspectivityCell, RecommendationResponse, ReserveProspectivityResponse, ReserveSummaryResponse, OverviewResponse } from "../types/api";
import { number, percent } from "../utils/format";

type Page = "overview" | "reserve" | "production" | "models" | "operations" | "assistant" | "reports";
type Layer = "probability" | "grade" | "thickness" | "confidence";
const nav: Array<{ id: Page; label: string; icon: typeof Activity }> = [
  { id: "overview", label: "Overview", icon: Gauge },
  { id: "reserve", label: "Reserve Intelligence", icon: Target },
  { id: "production", label: "Production Intelligence", icon: BarChart3 },
  { id: "models", label: "Model Monitoring", icon: Database },
  { id: "operations", label: "Operations Intelligence", icon: ShieldCheck },
  { id: "assistant", label: "AI Assistant", icon: Bot },
  { id: "reports", label: "Reports", icon: FileText },
];

export function MANGAICommandCenter() {
  const [page, setPage] = useState<Page>("overview");
  const [assistantOpen, setAssistantOpen] = useState(true);
  const [refresh, setRefresh] = useState(0);
  const [selectedCell, setSelectedCell] = useState<ProspectivityCell | null>(null);
  const [layer, setLayer] = useState<Layer>("probability");
  const [threshold, setThreshold] = useState(0.45);
  const [reportStatus, setReportStatus] = useState<Record<string, string>>({});

  const overview = useApi<OverviewResponse>(`/api/v1/overview?refresh=${refresh}`);
  const production = useApi<ProductionForecastResponse>(`/api/v1/production/forecast?horizon=7&refresh=${refresh}`);
  const history = useApi<{ records: ProductionHistoryRecord[] }>(`/api/v1/production/history?days=90&refresh=${refresh}`);
  const reserveSummary = useApi<ReserveSummaryResponse>(`/api/v1/reserves/summary?refresh=${refresh}`);
  const reserveCells = useApi<ReserveProspectivityResponse>(`/api/v1/reserves/prospectivity?limit=450&min_probability=${threshold.toFixed(2)}&refresh=${refresh}`);
  const boreholes = useApi<{ boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }> }>(`/api/v1/reserves/boreholes?limit=250&refresh=${refresh}`);
  const equipment = useApi<EquipmentResponse>(`/api/v1/equipment?refresh=${refresh}`);
  const models = useApi<ModelRegistryResponse>(`/api/v1/models?refresh=${refresh}`);
  const monitoring = useApi<ModelMonitoringResponse>(`/api/v1/models/monitoring?refresh=${refresh}`);
  const quality = useApi<DataQualityResponse>(`/api/v1/data-quality?refresh=${refresh}`);
  const recommendations = useApi<RecommendationResponse>(`/api/v1/recommendations?refresh=${refresh}`);

  const cells = reserveCells.data?.cells ?? [];
  const confidence = quality.data?.overall_score ?? 0.87;
  const resource = overview.data?.resource_potential_tonnage ?? reserveSummary.data?.resource_potential_tonnage ?? 1240000000;
  const fleet = equipment.data?.fleet_utilization ?? 0.78;
  const go = (next: Page) => { setPage(next); if (next === "assistant") setAssistantOpen(true); };
  const doRefresh = () => setRefresh((v) => v + 1);

  return <div className="mc-app">
    <header className="mc-header">
      <div className="mc-brand"><div className="mc-logo">▲</div><div><strong>MANGAI</strong><span>INDUSTRIAL INTELLIGENCE PLATFORM</span></div></div>
      <div className="mc-system"><i/><b>System Online</b><small>All systems operational</small></div>
      <div className="mc-head-info">Apr 26, 2025<br/><b>14:32 <small>IST</small></b></div>
      <div className="mc-head-info">☀️ <b>18°C</b><small>Clear</small></div>
      <div className="mc-user">● <span>Operations Team<br/><small>Administrator</small></span></div>
    </header>

    <aside className="mc-sidebar">
      {nav.map(({ id, label, icon: Icon }) => <button key={id} className={page === id ? "mc-nav active" : "mc-nav"} onClick={() => go(id)}><Icon size={15}/><span>{label}</span></button>)}
      <div className="mc-divider"/><small className="mc-label">TOOLS</small>
      <button className="mc-nav" onClick={() => go("reserve")}><MapIcon size={15}/><span>Map</span></button>
      <button className="mc-nav" onClick={() => go("reports")}><FileText size={15}/><span>Reports</span></button>
      <div className="mc-divider"/><small className="mc-label">SYSTEM</small>
      <button className="mc-nav" onClick={() => go("models")}><Settings size={15}/><span>Settings</span></button>
      <button className="mc-nav" onClick={() => go("assistant")}><MessageSquare size={15}/><span>Help</span></button>
      <div className="mc-sidebar-brand"><b>MANGAI<span>▲</span></b><small>Smarter Decisions.<br/>Greater Value.</small><em>╱╲╱╲╱╲╱╲</em></div>
    </aside>

    <main className="mc-main">
      {page === "overview" && <Overview data={overview.data} production={production.data} confidence={confidence} resource={resource} fleet={fleet} cells={cells} recommendations={recommendations.data} onNavigate={go}/>} 
      {page === "reserve" && <ReservePage cells={cells} summary={reserveSummary.data} boreholes={boreholes.data?.boreholes ?? []} threshold={threshold} setThreshold={setThreshold} layer={layer} setLayer={setLayer} selected={selectedCell} setSelected={setSelectedCell} onRefresh={doRefresh}/>} 
      {page === "production" && <ProductionPage forecast={production.data} history={history.data?.records ?? []} onRefresh={doRefresh}/>} 
      {page === "models" && <ModelsPage models={models.data} monitoring={monitoring.data} quality={quality.data} onRefresh={doRefresh}/>} 
      {page === "operations" && <OperationsPage/>}
      {page === "assistant" && <AssistantPage recommendations={recommendations.data}/>} 
      {page === "reports" && <ReportsPage statuses={reportStatus} setStatus={setReportStatus}/>} 
    </main>

    {page === "overview" && assistantOpen && <aside className="mc-assistant-rail"><AssistantRail onClose={() => setAssistantOpen(false)} onOpen={() => go("assistant")} recommendations={recommendations.data}/></aside>}
    {page === "overview" && !assistantOpen && <button className="mc-assistant-fab" onClick={() => setAssistantOpen(true)}><Bot/></button>}
    <footer className="mc-footer"><span>MANGAI <b>|</b> Industrial Intelligence for a Smarter Tomorrow</span><span><i/> Backend Online &nbsp; | &nbsp; Frontend Connected &nbsp; | &nbsp; v1.0.0</span></footer>
  </div>;
}

function Overview({ data, production, confidence, resource, fleet, cells, recommendations, onNavigate }: { data?: OverviewResponse | null; production?: ProductionForecastResponse | null; confidence: number; resource: number; fleet: number; cells: ProspectivityCell[]; recommendations?: RecommendationResponse | null; onNavigate: (p: Page) => void }) {
  const high = cells.filter((c) => c.probability >= 0.7).length;
  return <div className="mc-content"><PageTitle title="Executive Overview" subtitle="Real-time mining intelligence across reserves, production and operations"/><div className="mc-kpis">
    <Kpi icon={<Database/>} title="Total Resource (JORC)" value={`${(resource / 1e9).toFixed(2)} Bt`} change="▲ 2.4%"/>
    <Kpi icon={<Truck/>} title="Est. Annual Production" value={`${number(production?.forecast_mt ?? data?.next_7_day_production_mt ?? 12.8, 1)} Mt`} change="▲ 5.2%"/>
    <Kpi icon={<Activity/>} title="Reserve Life" value="42.6 years" change="▲ 3.1%"/>
    <Kpi icon={<Bot/>} title="AI Confidence" value={percent(confidence)} change="▲ 6.3%"/>
  </div><div className="mc-overview-grid">
    <section className="mc-panel mc-overview-map"><PanelHead title="Reserve Map" icon={<MapIcon/>}><button onClick={() => onNavigate("reserve")}>View details</button></PanelHead><div className="mc-map-preview">
      <div className="mc-map-shape a"/><div className="mc-map-shape b"/><div className="mc-map-shape c"/>
      {["North Ridge","Central","East Wing","South Basin"].map((x,i) => <span key={x} className={`mc-map-tag ${["one","two","three","four"][i]}`}>{x}</span>)}
      {cells.slice(0,45).map((cell, i) => <button key={cell.id} className="mc-map-dot" style={{ left: `${8 + (i * 37) % 86}%`, top: `${12 + (i * 61) % 75}%` }} onClick={() => onNavigate("reserve")} title={`${cell.id} ${percent(cell.probability)}`}/>)}
      <div className="mc-map-summary"><small>Active Targets</small><b>{Math.max(7, high)}</b><em>3 high priority</em></div><div className="mc-map-legend">Resource Area<br/>Reserve Area<br/>High Priority<br/>◆ Drill Target<br/>━ Haul Road</div>
    </div></section>
    <section className="mc-panel mc-insights"><PanelHead title="✦ Key Insights"><button onClick={() => onNavigate("assistant")}>View all</button></PanelHead>
      <Insight title="High Value Target Identified" text={recommendations?.recommendations?.[0]?.title ?? "East Wing shows elevated grade potential"} meta="Confidence: 87%"/>
      <Insight title="Production Forecast" text={`${number(production?.forecast_mt ?? 12.8, 1)} Mt forecast from latest model`} meta="Based on latest model run"/>
      <Insight title="Risk Alert" text={(data?.shortfall_probability ?? 0) > 0.65 ? "Production shortfall risk elevated" : "South Basin: monitor geotechnical risk"} meta="Monitor closely" risk/>
      <Insight title="Operational Efficiency" text={`Fleet utilization ${percent(fleet)}`} meta="Live equipment snapshot"/>
    </section>
  </div><div className="mc-bottom"><MiniChart title="Production Performance" value={`${number(production?.forecast_mt ?? 12.8, 1)} Mt`} onClick={() => onNavigate("production")}/><MiniReserve resource={resource} onClick={() => onNavigate("reserve")}/><MiniHealth confidence={confidence} onClick={() => onNavigate("models")}/><MiniOperations fleet={fleet} onClick={() => onNavigate("operations")}/></div></div>;
}

function ReservePage({ cells, summary, boreholes, threshold, setThreshold, layer, setLayer, selected, setSelected, onRefresh }: { cells: ProspectivityCell[]; summary?: ReserveSummaryResponse | null; boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>; threshold: number; setThreshold: (v: number) => void; layer: Layer; setLayer: (v: Layer) => void; selected: ProspectivityCell | null; setSelected: (v: ProspectivityCell | null) => void; onRefresh: () => void }) {
  const tabs: Layer[] = ["probability", "probability", "grade", "thickness"];
  const [tab, setTab] = useState(0);
  return <PageFrame title="Reserve Intelligence" subtitle="AI-powered geological analysis and reserve estimation" refresh={onRefresh}>
    <Tabs labels={["Map View","3D Analysis","Reserve Estimates","Geological Layers"]} active={tab} onChange={(i) => { setTab(i); setLayer(tabs[i]); }}/>
    <div className="mc-page-grid reserve-page"><section className="mc-panel mc-map-panel"><PanelHead title="Prospectivity Map" icon={<MapIcon/>}><span>{cells.length} cells</span></PanelHead><div className="mc-map-wrap"><ReserveMap cells={cells} selectedCell={selected} onSelect={setSelected} layer={layer} boreholes={boreholes}/></div><div className="mc-map-controls-inline"><label>Minimum probability <input type="range" min="0" max=".9" step=".05" value={threshold} onChange={(e) => setThreshold(Number(e.target.value))}/><b>{percent(threshold)}</b></label>{(["probability","grade","thickness","confidence"] as const).map(v => <button key={v} className={layer === v ? "active" : ""} onClick={() => setLayer(v)}>{v}</button>)}</div></section>
      <section className="mc-side-stack"><StatPanel title="Executive Summary"><Metric label="Total Resource" value={`${((summary?.resource_potential_tonnage ?? 1240000000) / 1e9).toFixed(2)} Bt`}/><Metric label="Est. Annual Production" value="12.8 Mt"/><Metric label="Reserve Life" value="42.6 years"/><Metric label="AI Confidence" value="87%"/></StatPanel><StatPanel title="Reserve by Category"><div className="mc-donut">1.24 Bt<small>Total Resource</small></div><Metric label="Measured" value="42%"/><Metric label="Indicated" value="34%"/><Metric label="Inferred" value="24%"/></StatPanel>{selected && <StatPanel title="Selected Target"><Metric label="ID" value={selected.id}/><Metric label="Probability" value={percent(selected.probability)}/><Metric label="Grade" value={`${number(selected.predicted_grade_pct, 1)}% Mn`}/><Metric label="Thickness" value={`${number(selected.predicted_thickness_m, 1)} m`}/><button className="mc-primary" onClick={() => setSelected(null)}>Clear target</button></StatPanel>}</section>
    </div>
  </PageFrame>;
}

function ProductionPage({ forecast, history, onRefresh }: { forecast?: ProductionForecastResponse | null; history: ProductionHistoryRecord[]; onRefresh: () => void }) {
  const [tab, setTab] = useState(0); const rows = history.slice(-12); const titles = ["Production Forecast","Production Analytics","Downtime Analysis","Blasting Optimization"];
  return <PageFrame title="Production Intelligence" subtitle="Forecasting, optimization and production performance" refresh={onRefresh}><Tabs labels={titles} active={tab} onChange={setTab}/>{tab === 0 && forecast && <ProductionForecastTerminal forecast={forecast}/>}<section className="mc-panel mc-chart-panel"><PanelHead title={tab === 0 ? "Production Forecast (Next 6 Months)" : titles[tab]} icon={<BarChart3/>}/><ResponsiveContainer width="100%" height={270}><LineChart data={rows}><CartesianGrid stroke="#15374c" strokeDasharray="3 3"/><XAxis dataKey="date" stroke="#6f91a5"/><YAxis stroke="#6f91a5"/><Tooltip/><Line type="monotone" dataKey="production_mt" stroke="#0bc8ff" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="target_mt" stroke="#00d9a5" strokeDasharray="5 4" dot={false}/></LineChart></ResponsiveContainer></section><div className="mc-two"><StatPanel title="Key Metrics"><Metric label="Current" value={`${number(forecast?.forecast_mt ?? 12.8, 1)} Mt`}/><Metric label="Forecast next month" value="14.1 Mt"/><Metric label="Production variance" value={`${percent(forecast?.shortfall_probability ?? .13)} risk`}/><Metric label="OEE" value="87%"/></StatPanel><StatPanel title="Recent Alerts"><AlertRow title="Blasting delay at South Basin" level="HIGH"/><AlertRow title="Equipment downtime - Excavator EX-13" level="WATCH"/><AlertRow title="Production below target - East Wing" level="INFO"/></StatPanel></div></PageFrame>;
}

function ModelsPage({ models, monitoring, quality, onRefresh }: { models?: ModelRegistryResponse | null; monitoring?: ModelMonitoringResponse | null; quality?: DataQualityResponse | null; onRefresh: () => void }) {
  const [tab, setTab] = useState(0); const labels = ["Model Performance","Drift Detection","Model Registry","Explainability"];
  return <PageFrame title="Model Monitoring" subtitle="Track model performance, drift and health across all AI models" refresh={onRefresh}><Tabs labels={labels} active={tab} onChange={setTab}/>{tab === 0 && monitoring && <ModelMonitoringPanel monitoring={monitoring}/>}<div className="mc-two"><StatPanel title="Model Health"><div className="mc-health-ring">{percent(quality?.overall_score ?? .87)}</div><Metric label="Healthy" value="4"/><Metric label="Warning" value="1"/><Metric label="Critical" value="0"/></StatPanel><StatPanel title="Model Registry"><div className="mc-model-list">{models?.models?.map(model => <div key={`${model.model_name}-${model.version}`}><b>{model.model_name}</b><span>{model.version}</span><em>{model.status}</em></div>)}</div></StatPanel></div></PageFrame>;
}

function AssistantPage({ recommendations }: { recommendations?: RecommendationResponse | null }) {
  const [messages, setMessages] = useState<string[]>([]); const suggestions = ["What's the best expansion option?","Show me production forecast risks","Any anomalies in the models?","Summarise today's operations"]; const send = (text: string) => setMessages(v => [...v, text]);
  return <PageFrame title="MANGAI AI Assistant" subtitle="Your intelligent mining operations companion"><div className="mc-assistant-page"><section className="mc-panel mc-chat"><div className="mc-chat-bubble"><Bot/><div><b>Hello! I'm your MANGAI AI Assistant.</b><p>I can help you with reserves, production forecasts, equipment status, and more.</p></div></div>{messages.map((m, i) => <div className="mc-user-message" key={`${m}-${i}`}>{m}</div>)}<div className="mc-suggestions">{suggestions.map(s => <button key={s} onClick={() => send(s)}>{s}</button>)}</div><div className="mc-chat-input"><input placeholder="Type your message..." onKeyDown={(e) => { if (e.key === "Enter" && e.currentTarget.value.trim()) { send(e.currentTarget.value.trim()); e.currentTarget.value = ""; } }}/><button onClick={() => send("Show latest intelligence")}><MessageSquare/></button></div></section><section className="mc-side-stack"><StatPanel title="Quick Insights"><Insight title="High value target" text={recommendations?.recommendations?.[0]?.title ?? "East Wing target identified"}/><Insight title="Equipment" text="Fleet utilization is being monitored live"/><Insight title="Weather" text="Latest weather inputs are available"/></StatPanel><StatPanel title="Suggested Actions"><button className="mc-action" onClick={() => send("Open reserve map")}>Open reserve map</button><button className="mc-action" onClick={() => send("Check equipment status")}>Check equipment status</button><button className="mc-action" onClick={() => send("Review production risk")}>Review production risk</button></StatPanel></section></div></PageFrame>;
}

function ReportsPage({ statuses, setStatus }: { statuses: Record<string, string>; setStatus: (v: Record<string, string>) => void }) {
  const reports = ["Executive Overview Report","Reserve Intelligence Report","Production Forecast Report","Operations Risk Report"];
  return <PageFrame title="Reports" subtitle="Generate and manage mining intelligence reports"><section className="mc-panel"><PanelHead title="Report Center" icon={<FileText/>}/><div className="mc-report-table"><div className="mc-report-row mc-report-head"><span>Report</span><span>Type</span><span>Generated</span><span>Status</span><span>Action</span></div>{reports.map((name, i) => { const status = statuses[name] ?? "Ready"; return <div className="mc-report-row" key={name}><span>{name}</span><span>{i === 0 ? "Executive" : "Intelligence"}</span><span>Today</span><span><b>{status}</b></span><button onClick={() => setStatus({ ...statuses, [name]: status === "Ready" ? "Generated" : "Ready" })}>{status === "Ready" ? "Generate" : "Reset"}</button></div>; })}</div></section></PageFrame>;
}

function AssistantRail({ onClose, onOpen, recommendations }: { onClose: () => void; onOpen: () => void; recommendations?: RecommendationResponse | null }) { return <div><div className="mc-assistant-head"><div className="mc-ai"><Bot/></div><div><h2>MANGAI AI Assistant</h2><span>● Online</span></div><button onClick={onClose}><X size={15}/></button></div><div className="mc-rail-question"><Bot/><span>What would you like to know?</span></div>{["What's the best expansion option?","Show me production forecast risks","Any anomalies in the models?","Summarise today's operations"].map(q => <button key={q} className="mc-rail-suggestion" onClick={onOpen}>{q}</button>)}<div className="mc-rail-input" onClick={onOpen}>Ask about reserves, production, operations... <MessageSquare size={14}/></div><h3>Recent Activity</h3><Insight title="Model run completed" text="Production Forecast v2.4" meta="12m ago"/><Insight title="New satellite imagery available" text="Area: Central Reserve" meta="34m ago"/><Insight title="High value target" text={recommendations?.recommendations?.[0]?.title ?? "Target identified"} meta="1h ago"/></div>; }

function PageFrame({ title, subtitle, refresh, children }: { title: string; subtitle: string; refresh?: () => void; children: ReactNode }) { return <div className="mc-content"><PageTitle title={title} subtitle={subtitle}/>{refresh && <button className="mc-refresh" onClick={refresh}><RefreshCw size={14}/> Refresh data</button>}{children}</div>; }
function PageTitle({ title, subtitle }: { title: string; subtitle: string }) { return <div className="mc-page-title"><div><h1>{title}</h1><p>{subtitle}</p></div></div>; }
function PanelHead({ title, icon, children }: { title: string; icon?: ReactNode; children?: ReactNode }) { return <div className="mc-panel-head"><h2>{icon}{title}</h2>{children}</div>; }
function Kpi({ icon, title, value, change }: { icon: ReactNode; title: string; value: string; change: string }) { return <section className="mc-kpi"><span>{icon}</span><small>{title}</small><b>{value}</b><em>{change}</em></section>; }
function Metric({ label, value }: { label: string; value: string }) { return <div className="mc-metric"><span>{label}</span><b>{value}</b></div>; }
function StatPanel({ title, children }: { title: string; children: ReactNode }) { return <section className="mc-panel mc-stat-panel"><h3>{title}</h3>{children}</section>; }
function Insight({ title, text, meta, risk }: { title: string; text: string; meta?: string; risk?: boolean }) { return <div className={risk ? "mc-insight risk" : "mc-insight"}><b>{title}</b><span>{text}</span>{meta && <small>{meta}</small>}</div>; }
function Tabs({ labels, active, onChange }: { labels: string[]; active: number; onChange: (index: number) => void }) { return <div className="mc-tabs">{labels.map((label, i) => <button key={label} className={active === i ? "active" : ""} onClick={() => onChange(i)}>{label}</button>)}</div>; }
function AlertRow({ title, level }: { title: string; level: string }) { return <div className="mc-alert-row"><AlertTriangle size={14}/><span>{title}</span><b>{level}</b></div>; }
function MiniChart({ title, value, onClick }: { title: string; value: string; onClick: () => void }) { return <button className="mc-mini-card" onClick={onClick}><small>{title}</small><b>{value}</b><span>View intelligence →</span></button>; }
function MiniReserve({ resource, onClick }: { resource: number; onClick: () => void }) { return <button className="mc-mini-card" onClick={onClick}><small>Reserve Intelligence</small><b>{(resource / 1e9).toFixed(2)} Bt</b><span>Explore targets →</span></button>; }
function MiniHealth({ confidence, onClick }: { confidence: number; onClick: () => void }) { return <button className="mc-mini-card" onClick={onClick}><small>Model Health</small><b>{percent(confidence)}</b><span>Open monitoring →</span></button>; }
function MiniOperations({ fleet, onClick }: { fleet: number; onClick: () => void }) { return <button className="mc-mini-card" onClick={onClick}><small>Operations</small><b>{percent(fleet)} utilization</b><span>View operations →</span></button>; }
