import { useEffect, useMemo, useState } from "react";
import { Activity, BarChart3, CloudRain, Database, Gauge, Layers3, Map, Menu, Radio, Settings, ShieldCheck, Target, Wrench, X } from "lucide-react";
import { apiPost } from "./api/client";
import { useApi } from "./hooks/useApi";
import { BlastingResponse, DataQualityResponse, EquipmentResponse, ModelRegistryResponse, OverviewResponse, ProductionForecastResponse, ProductionHistoryRecord, ProspectivityCell, RecommendationResponse, ReserveProspectivityResponse, ReserveSummaryResponse, WeatherResponse } from "./types/api";
import { ReserveMap } from "./components/ReserveMap";
import { OverviewPage } from "./pages/OverviewPage";
import { SatellitePage } from "./pages/SatellitePage";
import { OperationsPage } from "./pages/OperationsPage";
import { compactNumber, number, percent, signedNumber } from "./utils/format";
import { Bar, BarChart, CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";

type PageKey = "overview" | "reserve" | "production" | "operations" | "satellite" | "equipment" | "weather" | "recommendations" | "health" | "settings";
const pages: Array<{ key: PageKey; label: string; icon: typeof Activity }> = [
  { key: "overview", label: "Overview", icon: Gauge }, { key: "reserve", label: "Reserve Intelligence", icon: Map }, { key: "production", label: "Production Intelligence", icon: BarChart3 }, { key: "operations", label: "Operations", icon: Activity }, { key: "equipment", label: "Equipment", icon: Wrench }, { key: "weather", label: "Weather + Blasting", icon: CloudRain }, { key: "satellite", label: "Satellite Intelligence", icon: Radio }, { key: "recommendations", label: "Recommendation Center", icon: Target }, { key: "health", label: "Model + Data Health", icon: Database }, { key: "settings", label: "Settings", icon: Settings }
];

type Scene = { id?: string; source?: string; date?: string; cloud_cover?: number; quality?: number; bands?: string[]; thermal_coverage?: number };

export default function App() {
  const [activePage, setActivePage] = useState<PageKey>("overview");
  const [threshold, setThreshold] = useState(0.55);
  const [layer, setLayer] = useState<"probability" | "grade" | "thickness" | "confidence">("probability");
  const [selectedCell, setSelectedCell] = useState<ProspectivityCell | null>(null);
  const [mobileNav, setMobileNav] = useState(false);
  const [now, setNow] = useState(new Date());

  const overview = useApi<OverviewResponse>("/api/v1/overview");
  const reserveSummary = useApi<ReserveSummaryResponse>("/api/v1/reserves/summary");
  const reserveCells = useApi<ReserveProspectivityResponse>(`/api/v1/reserves/prospectivity?limit=450&min_probability=${threshold.toFixed(2)}`);
  const boreholes = useApi<{ boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }> }>("/api/v1/reserves/boreholes?limit=250");
  const production = useApi<ProductionForecastResponse>("/api/v1/production/forecast?horizon=7");
  const productionHistory = useApi<{ records: ProductionHistoryRecord[] }>("/api/v1/production/history?days=90");
  const equipment = useApi<EquipmentResponse>("/api/v1/equipment");
  const weather = useApi<WeatherResponse>("/api/v1/weather");
  const blasting = useApi<BlastingResponse>("/api/v1/blasting");
  const recommendations = useApi<RecommendationResponse>("/api/v1/recommendations");
  const models = useApi<ModelRegistryResponse>("/api/v1/models");
  const dataQuality = useApi<DataQualityResponse>("/api/v1/data-quality");
  const safety = useApi<{ mode: string; boundary: string }>("/api/v1/settings/safety");
  const satellite = useApi<{ scenes: Scene[]; count: number; data_mode: string }>("/api/v1/satellite/scenes?limit=500");

  useEffect(() => { const timer = window.setInterval(() => setNow(new Date()), 1000); return () => window.clearInterval(timer); }, []);
  useEffect(() => { setMobileNav(false); }, [activePage]);

  const loading = overview.loading || production.loading;
  const error = overview.error || production.error;
  const ready = Boolean(overview.data && production.data);
  const status = overview.data?.model_health ?? "INITIALIZING";
  const sceneCount = satellite.data?.count ?? 0;
  const page = useMemo(() => pages.find(item => item.key === activePage)!, [activePage]);

  return <div className="app">
    <aside className={`sidebar ${mobileNav ? "open" : ""}`}>
      <div className="brand"><div className="brand-mark">M</div><div><h1>MANGAI</h1><span>Industrial Geospatial Intelligence</span></div><button className="mobile-close" onClick={() => setMobileNav(false)} aria-label="Close navigation"><X size={18}/></button></div>
      <nav aria-label="Intelligence navigation">{pages.map(item => { const Icon = item.icon; return <button key={item.key} aria-current={activePage === item.key ? "page" : undefined} className={activePage === item.key ? "nav-item active" : "nav-item"} onClick={() => setActivePage(item.key)}><Icon size={17}/><span>{item.label}</span></button>; })}</nav>
      <div className="sidebar-foot"><ShieldCheck size={17}/><span>{status}</span><small>decision support</small></div>
    </aside>
    {mobileNav && <button className="nav-backdrop" onClick={() => setMobileNav(false)} aria-label="Close navigation"/>}
    <main className="main">
      <header className="topbar">
        <button className="mobile-menu" onClick={() => setMobileNav(true)} aria-label="Open navigation"><Menu size={20}/></button>
        <div className="topbar-title"><p className="eyebrow">SIH 26009 · MOIL · MINISTRY OF STEEL</p><h2>{page.label}</h2><span>{overview.data?.site_name ?? "MANGAI Demo Mine"} · {now.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit", second: "2-digit" })}</span></div>
        <div className="command-status"><span className="status-dot"/><span>SYSTEM {status}</span><span className="badge warn">DEMO / SYNTHETIC</span><span className="badge">{sceneCount} satellite feature records</span></div>
      </header>
      <div className="global-boundary"><span>DECISION SUPPORT ONLY</span><span>Prototype outputs are not official reserves or safety clearance.</span></div>
      {loading && <StatePanel label="Loading MANGAI intelligence services…"/>}
      {error && <StatePanel label={`API unavailable: ${error}`} tone="danger"/>}
      {ready && overview.data && production.data && <>
        {activePage === "overview" && recommendations.data && equipment.data && dataQuality.data && reserveCells.data && <OverviewPage overview={overview.data} production={production.data} recommendations={recommendations.data} equipment={equipment.data} quality={dataQuality.data} reserveCells={reserveCells.data.cells}/>} 
        {activePage === "reserve" && reserveSummary.data && reserveCells.data && <ReservePage summary={reserveSummary.data} cells={reserveCells.data.cells} boreholes={boreholes.data?.boreholes ?? []} threshold={threshold} setThreshold={setThreshold} layer={layer} setLayer={setLayer} selectedCell={selectedCell} setSelectedCell={setSelectedCell}/>} 
        {activePage === "production" && <ProductionPage forecast={production.data} history={productionHistory.data?.records ?? []}/>} 
        {activePage === "operations" && equipment.data && weather.data && blasting.data && <OperationsPage equipment={equipment.data} weather={weather.data} blasting={blasting.data} production={production.data}/>} 
        {activePage === "equipment" && equipment.data && <EquipmentPage equipment={equipment.data}/>} 
        {activePage === "weather" && weather.data && blasting.data && <WeatherBlastingPage weather={weather.data} blasting={blasting.data}/>} 
        {activePage === "satellite" && <SatellitePage scenes={satellite.data?.scenes ?? []}/>} 
        {activePage === "recommendations" && recommendations.data && <RecommendationsPage initial={recommendations.data}/>} 
        {activePage === "health" && models.data && dataQuality.data && <HealthPage models={models.data} dataQuality={dataQuality.data} boundary={overview.data.boundary_notice}/>} 
        {activePage === "settings" && <SettingsPage safety={safety.data}/>} 
      </>}
    </main>
  </div>;
}

function StatePanel({ label, tone = "default" }: { label: string; tone?: "default" | "danger" }) { return <section className={tone === "danger" ? "state danger" : "state"} role={tone === "danger" ? "alert" : "status"}>{label}</section>; }
function PanelHeader({ icon: Icon, title, meta }: { icon: typeof Activity; title: string; meta: string }) { return <div className="panel-header"><div className="panel-title"><Icon size={16}/><span>{title}</span></div><span className="panel-meta">{meta}</span></div>; }
function MetricLine({ label, value }: { label: string; value: string }) { return <div className="metric-line"><span>{label}</span><strong>{value}</strong></div>; }
function MetricCard({ icon: Icon, label, value, tone = "ok" }: { icon: typeof Activity; label: string; value: string; tone?: "ok" | "risk" }) { return <div className={`metric-card ${tone}`}><Icon size={17}/><span>{label}</span><strong>{value}</strong></div>; }
function DriverList({ drivers }: { drivers: Array<{ feature: string; direction: string; importance: number; value?: number | string | null }> }) { return <div className="driver-list">{drivers.slice(0, 5).map(driver => <div className="driver" key={`${driver.feature}-${driver.direction}`}><span>{driver.feature}</span><i style={{ width: `${Math.min(100, Math.max(6, driver.importance * 100))}%` }}/><b>{driver.direction}</b></div>)}</div>; }

function ReservePage({ summary, cells, boreholes, threshold, setThreshold, layer, setLayer, selectedCell, setSelectedCell }: { summary: ReserveSummaryResponse; cells: ProspectivityCell[]; boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>; threshold: number; setThreshold: (v: number) => void; layer: "probability" | "grade" | "thickness" | "confidence"; setLayer: (v: "probability" | "grade" | "thickness" | "confidence") => void; selectedCell: ProspectivityCell | null; setSelectedCell: (c: ProspectivityCell) => void }) {
  return <div className="page-grid reserve-layout"><section className="panel wide"><PanelHeader icon={Layers3} title="Reserve Intelligence" meta={`${cells.length} cells · spatial model`}/><div className="toolbar"><label>Prospectivity threshold <input aria-label="Prospectivity threshold" type="range" min="0" max="0.95" step="0.05" value={threshold} onChange={e => setThreshold(Number(e.target.value))}/><strong>{percent(threshold)}</strong></label><div className="layer-toggles">{(["probability", "grade", "thickness", "confidence"] as const).map(item => <button key={item} className={layer === item ? "chip active" : "chip"} aria-pressed={layer === item} onClick={() => setLayer(item)}>{item}</button>)}</div></div><ReserveMap cells={cells} selectedCell={selectedCell} onSelect={setSelectedCell} layer={layer} boreholes={boreholes}/></section><section className="panel detail-panel"><PanelHeader icon={Target} title="Target Intelligence" meta={selectedCell?.prospectivity_class ?? "select target"}/>{selectedCell ? <div className="detail-stack"><h3>{selectedCell.id}</h3><MetricLine label="Coordinates" value={`${number(selectedCell.latitude, 4)}, ${number(selectedCell.longitude, 4)}`}/><MetricLine label="Prospectivity" value={percent(selectedCell.probability)}/><MetricLine label="Predicted grade" value={`${number(selectedCell.predicted_grade_pct, 1)}% Mn`}/><MetricLine label="Predicted thickness" value={`${number(selectedCell.predicted_thickness_m, 1)} m`}/><MetricLine label="Confidence" value={percent(selectedCell.confidence)}/><MetricLine label="Resource P50" value={`${compactNumber(selectedCell.resource_potential.p50)} t`}/><DriverList drivers={selectedCell.top_contributors}/><p className="muted">Prototype resource potential only. Field verification is required before any reserve classification.</p></div> : <p className="muted">Select a target to inspect probability, grade, thickness, confidence and model evidence.</p>}</section><section className="panel"><PanelHeader icon={Gauge} title="Reserve Summary" meta="prototype"/><MetricLine label="High cells" value={number(summary.high_prospectivity_cells)}/><MetricLine label="Very high cells" value={number(summary.very_high_prospectivity_cells)}/><MetricLine label="Average grade" value={`${number(summary.average_predicted_grade_pct, 1)}% Mn`}/><MetricLine label="Average thickness" value={`${number(summary.average_predicted_thickness_m, 1)} m`}/><MetricLine label="Area" value={`${number(summary.high_prospectivity_area_ha, 1)} ha`}/></section></div>;
}

function ProductionPage({ forecast, history }: { forecast: ProductionForecastResponse; history: ProductionHistoryRecord[] }) { return <div className="page-grid"><section className="kpi-grid"><MetricCard icon={Activity} label="7-day forecast" value={`${compactNumber(forecast.forecast_mt)} t`}/><MetricCard icon={Target} label="Target" value={`${compactNumber(forecast.target_mt)} t`}/><MetricCard icon={BarChart3} label="Gap" value={`${signedNumber(forecast.gap_mt)} t`} tone={forecast.gap_mt < 0 ? "risk" : "ok"}/><MetricCard icon={Gauge} label="Shortfall risk" value={percent(forecast.shortfall_probability)} tone={forecast.shortfall_probability > .65 ? "risk" : "ok"}/></section><section className="panel wide chart-panel"><PanelHeader icon={BarChart3} title="Production Outlook" meta="actual · target · 90-day history"/><ResponsiveContainer width="100%" height={340}><LineChart data={history}><CartesianGrid stroke="rgba(145,174,162,.14)" strokeDasharray="3 3"/><XAxis dataKey="date" minTickGap={28} stroke="#6f847b"/><YAxis width={68} stroke="#6f847b"/><Tooltip/><Line type="monotone" dataKey="production_mt" stroke="#35d39f" strokeWidth={2} dot={false}/><Line type="monotone" dataKey="target_mt" stroke="#e7b75b" strokeWidth={2} dot={false}/></LineChart></ResponsiveContainer></section><section className="panel"><PanelHeader icon={Activity} title="Why the forecast moves" meta={forecast.severity}/><DriverList drivers={forecast.top_drivers}/></section><section className="panel"><PanelHeader icon={Gauge} title="Uncertainty" meta={forecast.forecast_date}/><MetricLine label="P10" value={`${compactNumber(forecast.prediction_interval.p10)} t`}/><MetricLine label="P50" value={`${compactNumber(forecast.prediction_interval.p50)} t`}/><MetricLine label="P90" value={`${compactNumber(forecast.prediction_interval.p90)} t`}/><MetricLine label="Baseline" value={`${compactNumber(forecast.baseline_forecast_mt)} t`}/></section></div>; }

function EquipmentPage({ equipment }: { equipment: EquipmentResponse }) { const ranked = equipment.items.slice().sort((a, b) => b.downtime_7d_hours - a.downtime_7d_hours).slice(0, 12); return <div className="page-grid"><section className="kpi-grid"><MetricCard icon={ShieldCheck} label="Fleet availability" value={percent(equipment.fleet_availability)}/><MetricCard icon={Gauge} label="Fleet utilization" value={percent(equipment.fleet_utilization)}/><MetricCard icon={Wrench} label="Critical equipment" value={number(equipment.critical_equipment_count)} tone={equipment.critical_equipment_count > 0 ? "risk" : "ok"}/><MetricCard icon={Activity} label="Downtime" value={`${number(equipment.total_downtime_7d_hours, 0)} h`} tone={equipment.total_downtime_7d_hours > 80 ? "risk" : "ok"}/></section><section className="panel wide"><PanelHeader icon={Wrench} title="Fleet Digital Grid" meta="ranked by downtime"/><div className="data-table-wrap"><table><thead><tr><th>Equipment</th><th>Status</th><th>Availability</th><th>Utilization</th><th>Downtime 7d</th><th>Health</th></tr></thead><tbody>{ranked.map(item => <tr key={item.equipment_id}><td className="mono">{item.equipment_id}</td><td>{item.status}</td><td>{percent(item.availability)}</td><td>{percent(item.utilization)}</td><td>{number(item.downtime_7d_hours, 1)} h</td><td>{percent(item.health_score)}</td></tr>)}</tbody></table></div></section></div>; }

function WeatherBlastingPage({ weather, blasting }: { weather: WeatherResponse; blasting: BlastingResponse }) { const rain = weather.records.slice(-14); return <div className="page-grid"><section className="panel wide chart-panel"><PanelHeader icon={CloudRain} title="Environmental Signal" meta="recent weather"/><ResponsiveContainer width="100%" height={330}><BarChart data={rain}><CartesianGrid stroke="rgba(145,174,162,.14)" strokeDasharray="3 3"/><XAxis dataKey="date" minTickGap={24} stroke="#6f847b"/><YAxis stroke="#6f847b"/><Tooltip/><Bar dataKey="rainfall_mm" fill="#28d7a0" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></section><section className="panel"><PanelHeader icon={Activity} title="Weather risk" meta={weather.risk_level}/><MetricLine label="Rainfall 7d" value={`${number(weather.rainfall_7d_mm, 1)} mm`}/><MetricLine label="Soil moisture" value={percent(weather.soil_moisture)}/><MetricLine label="Vegetation" value={percent(weather.ndvi)}/><MetricLine label="Temperature" value={`${number(weather.temperature_c, 1)} °C`}/></section><section className="panel wide"><PanelHeader icon={Target} title="Blast windows" meta={`${blasting.records.length} records`}/><div className="data-table-wrap"><table><thead><tr><th>Date</th><th>Blast</th><th>Status</th><th>Delay</th><th>Rain overlap</th></tr></thead><tbody>{blasting.records.slice(-14).map((item, i) => <tr key={`${item.date}-${i}`}><td>{item.date}</td><td className="mono">{item.blast_id}</td><td>{item.status}</td><td>{number(item.delay_minutes, 0)} min</td><td>{item.weather_overlap ? "Yes" : "No"}</td></tr>)}</tbody></table></div></section></div>; }

function RecommendationsPage({ initial }: { initial: RecommendationResponse }) { const [data, setData] = useState(initial); const [busy, setBusy] = useState(false); const simulate = async () => { setBusy(true); try { setData(await apiPost<RecommendationResponse>("/api/v1/recommendations/simulate", { scenario: "stabilize-production", production_gap_mt: 0.05, equipment_redeployment: true })); } finally { setBusy(false); } }; return <div className="page-grid"><section className="panel wide"><PanelHeader icon={Target} title="AI Action Center" meta="human approval required"/><div className="recommendation-list">{data.recommendations.map(rec => <article className="recommendation" key={rec.id}><div className="rec-head"><span className={`priority ${rec.priority.toLowerCase()}`}>{rec.priority}</span><span className="rec-status">{rec.status}</span></div><h3>{rec.title}</h3><p>{rec.rationale}</p><button className="action-button" onClick={simulate} disabled={busy}>{busy ? "Simulating…" : "Run what-if simulation"}</button></article>)}</div></section></div>; }

function HealthPage({ models, dataQuality, boundary }: { models: ModelRegistryResponse; dataQuality: DataQualityResponse; boundary: string }) { return <div className="page-grid"><section className="panel wide"><PanelHeader icon={Database} title="Model Registry" meta={`${models.models.length} registered`}/><div className="model-grid">{models.models.map(model => <div className="model-row" key={model.name}><span>{model.name}</span><strong>{model.version}</strong><span>{model.status}</span></div>)}</div></section><section className="panel"><PanelHeader icon={ShieldCheck} title="Data Health" meta="pipeline"/><MetricLine label="Overall quality" value={percent(dataQuality.overall_score)}/><MetricLine label="Freshness" value={dataQuality.freshness_status}/><MetricLine label="Schema checks" value={number(dataQuality.schema_checks_passed)}/></section><section className="panel wide"><PanelHeader icon={Activity} title="Quality by dataset" meta="observability"/><div className="quality-table">{dataQuality.datasets.map(item => <div className="quality-row" key={item.name}><span>{item.name}</span><strong>{percent(item.score)}</strong><small>{item.status}</small></div>)}</div></section><section className="panel wide"><PanelHeader icon={ShieldCheck} title="Scientific boundary" meta="always active"/><p className="muted">{boundary}</p></section></div>; }

function SettingsPage({ safety }: { safety?: { mode: string; boundary: string } }) { return <div className="page-grid"><section className="panel wide"><PanelHeader icon={Settings} title="System Settings" meta="configuration"/><MetricLine label="Data mode" value={safety?.mode ?? "DEMO"}/><MetricLine label="Safety boundary" value="Active"/><p className="muted">{safety?.boundary ?? "Decision-support only. Validate model outputs before operational use."}</p></section></div>; }
