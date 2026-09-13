import { useState, type ReactNode } from "react";
import { Activity, AlertTriangle, BarChart3, Bot, CheckCircle2, CloudRain, Database, FileText, Gauge, Map as MapIcon, MessageSquare, RefreshCw, Settings, ShieldCheck, Target, Truck, Wrench, X } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, CartesianGrid, LineChart, Line, Tooltip, XAxis, YAxis } from "recharts";
import { useApi } from "../hooks/useApi";
import { apiPost } from "../api/client";
import { ReserveMap } from "./ReserveMap";
import { ModelMonitoringPanel } from "./ModelMonitoringPanel";
import { OperationsPage } from "../pages/OperationsPage";
import { ProductionForecastTerminal } from "./production/ProductionForecastTerminal";
import type { DataQualityResponse, EquipmentResponse, ModelMonitoringResponse, ModelRegistryResponse, ProductionForecastResponse, ProductionHistoryRecord, ProspectivityCell, RecommendationResponse, ReserveProspectivityResponse, ReserveSummaryResponse, OverviewResponse } from "../types/api";
import { compactNumber, number, percent } from "../utils/format";

type Page = "overview" | "reserve" | "production" | "models" | "operations" | "assistant" | "reports";
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
  const [layer, setLayer] = useState<"probability" | "grade" | "thickness" | "confidence">("probability");
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

  const data = overview.data;
  const prod = production.data;
  const reserve = reserveSummary.data;
  const cells = reserveCells.data?.cells ?? [];
  const confidence = quality.data?.overall_score ?? 0.87;
  const resource = data?.resource_potential_tonnage ?? reserve?.resource_potential_tonnage ?? 1240000000;
  const fleet = equipment.data?.fleet_utilization ?? 0.78;

  const go = (next: Page) => { setPage(next); if (next === "assistant") setAssistantOpen(true); };
  const doRefresh = () => setRefresh((value) => value + 1);

  return (
    <div className="mc-app">
      <header className="mc-header">
        <div className="mc-brand"><div className="mc-logo">▲</div><div><strong>MANGAI</strong><span>INDUSTRIAL INTELLIGENCE PLATFORM</span></div></div>
        <div className="mc-system"><i/> <b>System Online</b><small>All systems operational</small></div>
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
        {page === "overview" && <Overview data={data} production={prod} confidence={confidence} resource={resource} fleet={fleet} cells={cells} recommendations={recommendations.data} onNavigate={go} />}
        {page === "reserve" && <ReservePage cells={cells} summary={reserve} boreholes={boreholes.data?.boreholes ?? []} threshold={threshold} setThreshold={setThreshold} layer={layer} setLayer={setLayer} selected={selectedCell} setSelected={setSelectedCell} onRefresh={doRefresh} />}
        {page === "production" && <ProductionPage forecast={prod} history={history.data?.records ?? []} onRefresh={doRefresh} />}
        {page === "models" && <ModelsPage models={models.data} monitoring={monitoring.data} quality={quality.data} onRefresh={doRefresh} />}
        {page === "operations" && <OperationsPage />}
        {page === "assistant" && <AssistantPage recommendations={recommendations.data} />}
        {page === "reports" && <ReportsPage statuses={reportStatus} setStatus={setReportStatus} />}
      </main>

      {page === "overview" && assistantOpen && <aside className="mc-assistant-rail"><AssistantRail onClose={() => setAssistantOpen(false)} onOpen={() => go("assistant")} recommendations={recommendations.data}/></aside>}
      {page === "overview" && !assistantOpen && <button className="mc-assistant-fab" onClick={() => setAssistantOpen(true)}><Bot/></button>}
      <footer className="mc-footer"><span>MANGAI <b>|</b> Industrial Intelligence for a Smarter Tomorrow</span><span><i/> Backend Online &nbsp; | &nbsp; Frontend Connected &nbsp; | &nbsp; v1.0.0</span></footer>
    </div>
  );
}

function Overview({data,production,confidence,resource,fleet,cells,recommendations,onNavigate}:{data:OverviewResponse|undefined;production:ProductionForecastResponse|undefined;confidence:number;resource:number;fleet:number;cells:ProspectivityCell[];recommendations:RecommendationResponse|undefined;onNavigate:(page:Page)=>void}) {
  const high = cells.filter((cell) => cell.probability >= .7).length;
  return <div className="mc-content"><div className="mc-title"><div><h1>Executive Overview</h1><p>Real-time mining intelligence across reserves, production and operations</p></div><div className="mc-live">● Live &nbsp; 24h &nbsp; 7d &nbsp; 30d</div></div>
    <div className="mc-kpis"><Kpi icon={<Database/>} title="Total Resource (JORC)" value={`${(resource/1e9).toFixed(2)} Bt`} change="▲ 2.4%"/><Kpi icon={<Truck/>} title="Est. Annual Production" value={`${number(production?.forecast_mt ?? data?.next_7_day_production_mt ?? 12.8,1)} Mt`} change="▲ 5.2%"/><Kpi icon={<Activity/>} title="Reserve Life" value="42.6 years" change="▲ 3.1%"/><Kpi icon={<Bot/>} title="AI Confidence" value={percent(confidence)} change="▲ 6.3%"/></div>
    <div className="mc-overview-grid"><section className="mc-panel mc-overview-map"><div className="mc-panel-head"><h2><MapIcon/> Reserve Map <span>● Live Data</span></h2><button onClick={() => onNavigate("reserve")}>View details</button></div><div className="mc-map-preview"><div className="mc-map-shape a"/><div className="mc-map-shape b"/><div className="mc-map-shape c"/><span className="mc-map-tag one">North Ridge</span><span className="mc-map-tag two">Central</span><span className="mc-map-tag three">East Wing</span><span className="mc-map-tag four">South Basin</span>{cells.slice(0,45).map((cell,index)=><button key={cell.id} className="mc-map-dot" style={{left:`${8+(index*37)%86}%`,top:`${12+(index*61)%75}%`}} onClick={() => onNavigate("reserve")} title={`${cell.id} ${percent(cell.probability)}`}/>)}<div className="mc-map-summary"><small>Active Targets</small><b>{Math.max(7,high)}</b><em>3 high priority</em></div><div className="mc-map-legend">Resource Area<br/>Reserve Area<br/>High Priority<br/>◆ Drill Target<br/>━ Haul Road</div></div></section><section className="mc-panel mc-insights"><div className="mc-panel-head"><h2>✦ Key Insights</h2><button onClick={() => onNavigate("assistant")}>View all</button></div><Insight title="High Value Target Identified" text={recommendations?.recommendations?.[0]?.title ?? "East Wing shows elevated grade potential"} meta="Confidence: 87%"/><Insight title="Production Forecast" text={`${number(production?.forecast_mt ?? 12.8,1)} Mt forecast from latest model`} meta="Based on latest model run"/><Insight title="Risk Alert" text={(data?.shortfall_probability ?? 0) > .65 ? "Production shortfall risk elevated" : "South Basin: monitor geotechnical risk"} meta="Monitor closely" risk/><Insight title="Operational Efficiency" text={`Fleet utilization ${percent(fleet)}`} meta="Live equipment snapshot"/></section></div>
    <div className="mc-bottom"><MiniChart title="Production Performance" value={`${number(production?.forecast_mt ?? 12.8,1)} Mt`} onClick={() => onNavigate("production")}/><MiniReserve resource={resource} onClick={() => onNavigate("reserve")}/><MiniHealth confidence={confidence} onClick={() => onNavigate("models")}/><MiniOperations fleet={fleet} onClick={() => onNavigate("operations")}/></div></div>;
}

function ReservePage({cells,summary,boreholes,threshold,setThreshold,layer,setLayer,selected,setSelected,onRefresh}:{cells:ProspectivityCell[];summary:ReserveSummaryResponse|undefined;boreholes:Array<{borehole_id:string;latitude:number;longitude:number;lithology?:string}>;threshold:number;setThreshold:(v:number)=>void;layer:"probability"|"grade"|"thickness"|"confidence";setLayer:(v:"probability"|"grade"|"thickness"|"confidence")=>void;selected:ProspectivityCell|null;setSelected:(v:ProspectivityCell|null)=>void;onRefresh:()=>void}) { return <PageFrame title="Reserve Intelligence" subtitle="AI-powered geological analysis and reserve estimation" refresh={onRefresh}><div className="mc-tabs"><button className="active">Map View</button><button onClick={()=>setLayer("probability")}>3D Analysis</button><button onClick={()=>setLayer("grade")}>Reserve Estimates</button><button onClick={()=>setLayer("thickness")}>Geological Layers</button></div><div className="mc-page-grid reserve-page"><section className="mc-panel mc-map-panel"><div className="mc-panel-head"><h2><MapIcon/> Prospectivity Map</h2><span>{cells.length} cells</span></div><div className="mc-map-wrap"><ReserveMap cells={cells} selectedCell={selected} onSelect={setSelected} layer={layer} boreholes={boreholes}/></div><div className="mc-map-controls-inline"><label>Minimum probability <input type="range" min="0" max=".9" step=".05" value={threshold} onChange={e=>setThreshold(Number(e.target.value))}/><b>{percent(threshold)}</b></label>{(["probability","grade","thickness","confidence"] as const).map(value=><button key={value} className={layer===value?"active":""} onClick={()=>setLayer(value)}>{value}</button>)}</div></section><section className="mc-side-stack"><StatPanel title="Executive Summary"><Metric label="Total Resource" value={`${((summary?.resource_potential_tonnage ?? 1240000000)/1e9).toFixed(2)} Bt`}/><Metric label="Est. Annual Production" value="12.8 Mt"/><Metric label="Reserve Life" value="42.6 years"/><Metric label="AI Confidence" value="87%"/></StatPanel><StatPanel title="Reserve by Category"><div className="mc-donut">1.24 Bt<small>Total Resource</small></div><Metric label="Measured" value="42%"/><Metric label="Indicated" value="34%"/><Metric label="Inferred" value="24%"/></StatPanel>{selected&&<StatPanel title="Selected Target"><Metric label="ID" value={selected.id}/><Metric label="Probability" value={percent(selected.probability)}/><Metric label="Grade" value={`${number(selected.predicted_grade_pct,1)}% Mn`}/><Metric label="Thickness" value={`${number(selected.predicted_thickness_m,1)} m`}/><button className="mc-primary" onClick={()=>setSelected(null)}>Clear target</button></StatPanel>}</div></div></PageFrame>; }

function ProductionPage({forecast,history,onRefresh}:{forecast:ProductionForecastResponse|undefined;history:ProductionHistoryRecord[];onRefresh:()=>void}) { const rows=history.slice(-12); return <PageFrame title="Production Intelligence" subtitle="Forecasting, optimization and production performance" refresh={onRefresh}><div className="mc-tabs"><button className="active">Production Forecast</button><button>Production Analytics</button><button>Downtime Analysis</button><button>Blasting Optimization</button></div>{forecast&&<ProductionForecastTerminal forecast={forecast}/>}<section className="mc-panel mc-chart-panel"><div className="mc-panel-head"><h2><BarChart3/> Production Forecast (Next 6 Months)</h2><span>{forecast?.model_version ?? "model"}</span></div><ResponsiveContainer width="100%" height={270}><LineChart data={rows}><CartesianGrid stroke="#15374c" strokeDasharray="3 3"/><XAxis dataKey="date" stroke="#6f91a5"/><YAxis stroke="#6f91a5"/><Tooltip contentStyle={{background:"#061a2a",border:"1px solid #0b6388",color:"#fff"}}/><Line type="monotone" dataKey="production_mt" stroke="#0bc8ff" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="target_mt" stroke="#00d9a5" strokeDasharray="5 4" dot={false}/></LineChart></ResponsiveContainer></section><div className="mc-two"><StatPanel title="Key Metrics"><Metric label="Current" value={`${number(forecast?.forecast_mt ?? 12.8,1)} Mt`}/><Metric label="Forecast next month" value="14.1 Mt"/><Metric label="Production variance" value={`${percent(forecast?.shortfall_probability ?? .13)} risk`}/><Metric label="OEE" value="87%"/></StatPanel><StatPanel title="Recent Alerts"><AlertRow title="Blasting delay at South Basin" level="HIGH"/><AlertRow title="Equipment downtime - Excavator EX-13" level="WATCH"/><AlertRow title="Production below target - East Wing" level="INFO"/></StatPanel></div></PageFrame>; }

function ModelsPage({models,monitoring,quality,onRefresh}:{models:ModelRegistryResponse|undefined;monitoring:ModelMonitoringResponse|undefined;quality:DataQualityResponse|undefined;onRefresh:()=>void}) { return <PageFrame title="Model Monitoring" subtitle="Track model performance, drift and health across all AI models" refresh={onRefresh}><div className="mc-tabs"><button className="active">Model Performance</button><button>Drift Detection</button><button>Model Registry</button><button>Explainability</button></div>{monitoring&&<ModelMonitoringPanel monitoring={monitoring}/>}<div className="mc-two"><StatPanel title="Model Health"><div className="mc-health-ring">{percent(quality?.overall_score ?? .87)}</div><Metric label="Healthy" value="4"/><Metric label="Warning" value="1"/><Metric label="Critical" value="0"/></StatPanel><StatPanel title="Model Registry"><div className="mc-model-list">{models?.models?.map(model=><div key={`${model.model_name}-${model.version}`}><b>{model.model_name}</b><span>{model.version}</span><em>{model.status}</em></div>)}</div></StatPanel></div></PageFrame>; }

function AssistantPage({recommendations}:{recommendations:RecommendationResponse|undefined}) { const [messages,setMessages]=useState<string[]>([]); const suggestions=["What's the best expansion option?","Show me production forecast risks","Any anomalies in the models?","Summarise today's operations"]; return <PageFrame title="MANGAI AI Assistant" subtitle="Your intelligent mining operations companion"><div className="mc-assistant-page"><section className="mc-panel mc-chat"><div className="mc-chat-bubble"><Bot/><div><b>Hello! I'm your MANGAI AI Assistant.</b><p>I can help you with reserves, production forecasts, equipment status, and more.</p></div></div>{messages.map((message,index)=><div className="mc-user-message" key={`${message}-${index}`}>{message}</div>)}<div className="mc-suggestions">{suggestions.map(item=><button key={item} onClick={()=>setMessages(current=>[...current,item])}>{item}</button>)}</div><div className="mc-chat-input"><input placeholder="Type your message..." onKeyDown={event=>{if(event.key==="Enter"&&event.currentTarget.value.trim()){setMessages(current=>[...current,event.currentTarget.value.trim()]);event.currentTarget.value="";}}}/><button><MessageSquare/></button></div></section><section className="mc-side-stack"><StatPanel title="Quick Insights"><Insight title="High value target" text={recommendations?.recommendations?.[0]?.title ?? "East Wing target identified"}/><Insight title="Equipment" text="Fleet utilization is being monitored live"/><Insight title="Weather" text="Latest weather inputs are available"/></StatPanel><StatPanel title="Suggested Actions"><button className="mc-action" onClick={()=>setMessages(current=>[...current,"Open reserve map"])}>Open reserve map</button><button className="mc-action" onClick={()=>setMessages(current=>[...current,"Check equipment status"])}>Check equipment status</button><button className="mc-action" onClick={()=>setMessages(current=>[...current,"Review production risk"])}>Review production risk</button></StatPanel></div></div></PageFrame>; }

function ReportsPage({statuses,setStatus}:{statuses:Record<string,string>;setStatus:React.Dispatch<React.SetStateAction<Record<string,string>>>}) { const reports=["Production Summary Report","Reserve Estimation Report","Equipment Utilization Report","Model Performance Report","Environmental Impact Report","Monthly Operations Report"]; return <PageFrame title="Reports" subtitle="Generate and manage system reports and analytics"><div className="mc-tabs"><button className="active">All Reports</button><button>Production</button><button>Reserves</button><button>Operations</button><button>Models</button></div><section className="mc-panel mc-report-table"><div className="mc-report-actions"><span>Last 30 days</span><button className="mc-primary" onClick={()=>setStatus(current=>({...current,"new":"Generated"}))}>Generate Report</button></div>{reports.map((report,index)=>{const status=statuses[report]??"Ready";return <div className="mc-report-row" key={report}><b>{report}</b><span>{index%2?"Reserves":"Production"}</span><span>Apr {22+index}, 2025 12:{10+index}</span><em>{status}</em><button onClick={()=>setStatus(current=>({...current,[report]:"Downloaded"}))}>Download</button><button onClick={()=>setStatus(current=>({...current,[report]:"Viewed"}))}>View</button></div>})}</section></PageFrame>; }

function PageFrame({title,subtitle,refresh,onRefresh,children}:{title:string;subtitle:string;refresh?:boolean;onRefresh?:()=>void;children:ReactNode}) { return <div className="mc-content"><div className="mc-title"><div><h1>{title}</h1><p>{subtitle}</p></div>{onRefresh&&<button className="mc-refresh" onClick={onRefresh}><RefreshCw size={14}/> Refresh</button>}</div>{children}</div>; }
function Kpi({icon,title,value,change}:{icon:ReactNode;title:string;value:string;change:string}){return <article className="mc-kpi"><div className="mc-kpi-icon">{icon}</div><span>{title}</span><strong>{value}</strong><em>{change}</em></article>}
function Metric({label,value}:{label:string;value:string}){return <div className="mc-metric"><span>{label}</span><b>{value}</b></div>}
function StatPanel({title,children}:{title:string;children:ReactNode}){return <section className="mc-panel mc-stat"><div className="mc-panel-head"><h2>{title}</h2></div>{children}</section>}
function Insight({title,text,meta,risk=false}:{title:string;text:string;meta?:string;risk?:boolean}){return <div className={`mc-insight ${risk?"risk":""}`}><span>{risk?<AlertTriangle size={14}/>:<CheckCircle2 size={14}/>}</span><div><b>{title}</b><p>{text}</p>{meta&&<small>{meta}</small>}</div></div>}
function AlertRow({title,level}:{title:string;level:string}){return <div className="mc-alert-row"><span className={level.toLowerCase()}>{level}</span><b>{title}</b><small>2h ago</small></div>}
function MiniChart({title,value,onClick}:{title:string;value:string;onClick:()=>void}){return <section className="mc-panel mc-mini"><div className="mc-panel-head"><h2><BarChart3/> {title}</h2><button onClick={onClick}>View details</button></div><strong>{value}</strong><em>▲ 5.2%</em><div className="mc-spark"><svg viewBox="0 0 300 80" preserveAspectRatio="none"><polyline points="0,62 50,68 95,52 140,56 185,35 225,29 265,18 300,12"/></svg></div></section>}
function MiniReserve({resource,onClick}:{resource:number;onClick:()=>void}){return <section className="mc-panel mc-mini"><div className="mc-panel-head"><h2><Target/> Reserve Intelligence</h2><button onClick={onClick}>View details</button></div><strong>{(resource/1e9).toFixed(2)} Bt</strong><em>▲ 2.4%</em><div className="mc-donut">1.24 Bt<small>Total Resource</small></div></section>}
function MiniHealth({confidence,onClick}:{confidence:number;onClick:()=>void}){return <section className="mc-panel mc-mini"><div className="mc-panel-head"><h2><Database/> Model Health</h2><button onClick={onClick}>View details</button></div><strong>{percent(confidence)}</strong><em>▲ 6.3%</em><div className="mc-health-bars"><i/><i/><i/><i/></div></section>}
function MiniOperations({fleet,onClick}:{fleet:number;onClick:()=>void}){return <section className="mc-panel mc-mini"><div className="mc-panel-head"><h2><Truck/> Operations</h2><button onClick={onClick}>View details</button></div><strong>{percent(fleet)}</strong><em>▲ 12%</em><div className="mc-ops-bars"><i/><i/><i/><i/></div></section>}
function AssistantRail({onClose,onOpen,recommendations}:{onClose:()=>void;onOpen:()=>void;recommendations:RecommendationResponse|undefined}){return <div><div className="mc-assistant-head"><div className="mc-ai"><Bot/></div><div><h2>MANGAI AI Assistant</h2><span>● Online</span></div><button onClick={onClose}><X size={15}/></button></div><div className="mc-rail-question"><Bot/><span>What would you like to know?</span></div>{["What's the best expansion option?","Show me production forecast risks","Any anomalies in the models?","Summarise today's operations"].map(q=><button key={q} className="mc-rail-suggestion" onClick={onOpen}>{q}</button>)}<div className="mc-rail-input" onClick={onOpen}>Ask about reserves, production, operations... <MessageSquare size={14}/></div><h3>Recent Activity</h3><Insight title="Model run completed" text="Production Forecast v2.4" meta="12m ago"/><Insight title="New satellite imagery available" text="Area: Central Reserve" meta="34m ago"/><Insight title="High value target" text={recommendations?.recommendations?.[0]?.title ?? "Target identified"} meta="1h ago"/></div>}
