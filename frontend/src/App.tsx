import { useState } from "react";
import { apiPost } from "../api/client";
import { useApi } from "../hooks/useApi";
import {
  BlastingResponse,
  DataQualityResponse,
  EquipmentResponse,
  ModelRegistryResponse,
  OverviewResponse,
  ProductionForecastResponse,
  ProductionHistoryRecord,
  ProspectivityCell,
  RecommendationResponse,
  ReserveProspectivityResponse,
  ReserveSummaryResponse,
  WeatherResponse
} from "../types/api";
import { compactNumber, number, percent, signedNumber } from "../utils/format";
import { ReserveMap } from "./components/ReserveMap";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  ClipboardList,
  CloudRain,
  Database,
  Gauge,
  Layers,
  Map,
  Settings,
  ShieldCheck,
  Target,
  Wrench
} from "lucide-react";
import {
  Area,
  AreaChart,
  Bar,
  BarChart,
  CartesianGrid,
  Line,
  LineChart,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis
} from "recharts";

type PageKey = "overview" | "reserve" | "production" | "equipment" | "weather" | "recommendations" | "health" | "settings";

const pages: Array<{ key: PageKey; label: string; icon: typeof Activity }> = [
  { key: "overview", label: "Overview", icon: Gauge },
  { key: "reserve", label: "Reserve", icon: Map },
  { key: "production", label: "Production", icon: BarChart3 },
  { key: "equipment", label: "Equipment", icon: Wrench },
  { key: "weather", label: "Weather", icon: CloudRain },
  { key: "recommendations", label: "Actions", icon: ClipboardList },
  { key: "health", label: "Health", icon: Database },
  { key: "settings", label: "Settings", icon: Settings }
];

function App() {
  const [activePage, setActivePage] = useState<PageKey>("overview");
  const [threshold, setThreshold] = useState(0.55);
  const [layer, setLayer] = useState<"probability" | "grade" | "thickness" | "confidence">("probability");
  const [selectedCell, setSelectedCell] = useState<ProspectivityCell | null>(null);

  const overview = useApi<OverviewResponse>("/api/v1/overview");
  const reserveSummary = useApi<ReserveSummaryResponse>("/api/v1/reserves/summary");
  const reserveCells = useApi<ReserveProspectivityResponse>(
    `/api/v1/reserves/prospectivity?limit=450&min_probability=${threshold.toFixed(2)}`
  );
  const boreholes = useApi<{ boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }> }>(
    "/api/v1/reserves/boreholes?limit=250"
  );
  const production = useApi<ProductionForecastResponse>("/api/v1/production/forecast?horizon=7");
  const productionHistory = useApi<{ records: ProductionHistoryRecord[] }>("/api/v1/production/history?days=90");
  const equipment = useApi<EquipmentResponse>("/api/v1/equipment");
  const weather = useApi<WeatherResponse>("/api/v1/weather");
  const blasting = useApi<BlastingResponse>("/api/v1/blasting");
  const recommendations = useApi<RecommendationResponse>("/api/v1/recommendations");
  const models = useApi<ModelRegistryResponse>("/api/v1/models");
  const dataQuality = useApi<DataQualityResponse>("/api/v1/data-quality");
  const safety = useApi<{ mode: string; boundary: string }>("/api/v1/settings/safety");

  const loading = overview.loading || production.loading;
  const error = overview.error || production.error;
  const ready = Boolean(overview.data && production.data);

  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <div className="brand-mark">M</div>
          <div>
            <h1>MANGAI</h1>
            <span>Mining Intelligence</span>
          </div>
        </div>
        <nav>
          {pages.map((page) => {
            const Icon = page.icon;
            return (
              <button
                key={page.key}
                className={activePage === page.key ? "nav-item active" : "nav-item"}
                onClick={() => setActivePage(page.key)}
              >
                <Icon size={18} />
                <span>{page.label}</span>
              </button>
            );
          })}
        </nav>
        <div className="sidebar-foot">
          <ShieldCheck size={18} />
          <span>{overview.data?.model_health ?? "READY"}</span>
        </div>
      </aside>
      <main className="main">
        <header className="topbar">
          <div>
            <p className="eyebrow">SIH 26009 · Ministry of Steel / MOIL</p>
            <h2>{overview.data?.site_name ?? "MANGAI Demo Mine"}</h2>
          </div>
          <div className="badges">
            <span className="badge warn">DEMO / SYNTHETIC DATA</span>
            <span className="badge">{production.data?.model_version ?? "loading"}</span>
          </div>
        </header>
        {loading && <StatePanel label="Loading MANGAI intelligence services" />}
        {error && <StatePanel label={`API unavailable: ${error}`} tone="danger" />}
        {ready && overview.data && production.data && (
          <>
            {activePage === "overview" && recommendations.data && equipment.data && dataQuality.data && (
              <OverviewPage
                overview={overview.data}
                production={production.data}
                recommendations={recommendations.data}
                equipment={equipment.data}
                quality={dataQuality.data}
              />
            )}
            {activePage === "reserve" && reserveSummary.data && reserveCells.data && (
              <ReservePage
                summary={reserveSummary.data}
                cells={reserveCells.data.cells}
                boreholes={boreholes.data?.boreholes ?? []}
                threshold={threshold}
                setThreshold={setThreshold}
                layer={layer}
                setLayer={setLayer}
                selectedCell={selectedCell}
                setSelectedCell={setSelectedCell}
              />
            )}
            {activePage === "production" && (
              <ProductionPage forecast={production.data} history={productionHistory.data?.records ?? []} />
            )}
            {activePage === "equipment" && equipment.data && <EquipmentPage equipment={equipment.data} />}
            {activePage === "weather" && weather.data && blasting.data && (
              <WeatherBlastingPage weather={weather.data} blasting={blasting.data} />
            )}
            {activePage === "recommendations" && recommendations.data && (
              <RecommendationsPage initial={recommendations.data} />
            )}
            {activePage === "health" && models.data && dataQuality.data && (
              <HealthPage models={models.data} dataQuality={dataQuality.data} boundary={overview.data.boundary_notice} />
            )}
            {activePage === "settings" && <SettingsPage safety={safety.data} />}
          </>
        )}
      </main>
    </div>
  );
}

function StatePanel({ label, tone = "default" }: { label: string; tone?: "default" | "danger" }) {
  return <section className={tone === "danger" ? "state danger" : "state"}>{label}</section>;
}

function OverviewPage({
  overview,
  production,
  recommendations,
  equipment,
  quality
}: {
  overview: OverviewResponse;
  production: ProductionForecastResponse;
  recommendations: RecommendationResponse;
  equipment: EquipmentResponse;
  quality: DataQualityResponse;
}) {
  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={Target} label="Prototype resource potential" value={`${compactNumber(overview.resource_potential_tonnage)} t`} />
        <MetricCard icon={Map} label="High prospectivity area" value={`${number(overview.high_prospectivity_area_ha, 1)} ha`} />
        <MetricCard icon={BarChart3} label="Next 7-day production" value={`${compactNumber(overview.next_7_day_production_mt)} t`} />
        <MetricCard icon={AlertTriangle} label="Shortfall probability" value={percent(overview.shortfall_probability)} tone={overview.shortfall_probability > 0.65 ? "risk" : "ok"} />
      </section>
      <section className="panel wide">
        <PanelHeader icon={Activity} title="Production Forecast" meta={production.severity} />
        <div className="split">
          <div>
            <p className="massive">{signedNumber(production.gap_mt)} t</p>
            <p className="muted">Forecast gap against 7-day target</p>
          </div>
          <DriverList drivers={production.top_drivers} />
        </div>
      </section>
      <section className="panel">
        <PanelHeader icon={Wrench} title="Equipment" meta={`${equipment.critical_equipment_count} critical`} />
        <div className="rank-list">
          {equipment.items.slice(0, 5).map((item) => (
            <div className="rank-row" key={item.equipment_id}>
              <span>{item.equipment_id}</span>
              <strong>{number(item.downtime_7d_hours, 1)} h</strong>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeader icon={ClipboardList} title="Recommendations" meta={`${recommendations.recommendations.length} proposed`} />
        <div className="stack">
          {recommendations.recommendations.slice(0, 3).map((item) => (
            <div className="action-row" key={item.id}>
              <span className={`priority ${item.priority.toLowerCase()}`}>{item.priority}</span>
              <p>{item.title}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeader icon={Database} title="Data Quality" meta={percent(quality.overall_score)} />
        <div className="quality-bar">
          <span style={{ width: `${quality.overall_score * 100}%` }} />
        </div>
        <p className="muted">{overview.model_health}</p>
      </section>
    </div>
  );
}

function ReservePage({
  summary,
  cells,
  boreholes,
  threshold,
  setThreshold,
  layer,
  setLayer,
  selectedCell,
  setSelectedCell
}: {
  summary: ReserveSummaryResponse;
  cells: ProspectivityCell[];
  boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>;
  threshold: number;
  setThreshold: (value: number) => void;
  layer: "probability" | "grade" | "thickness" | "confidence";
  setLayer: (value: "probability" | "grade" | "thickness" | "confidence") => void;
  selectedCell: ProspectivityCell | null;
  setSelectedCell: (cell: ProspectivityCell) => void;
}) {
  return (
    <div className="page-grid reserve-layout">
      <section className="panel wide">
        <PanelHeader icon={Layers} title="Reserve Intelligence" meta={`${cells.length} cells`} />
        <div className="toolbar">
          <label>
            Threshold
            <input type="range" min="0" max="0.95" step="0.05" value={threshold} onChange={(event) => setThreshold(Number(event.target.value))} />
            <strong>{percent(threshold)}</strong>
          </label>
          <div className="layer-toggles">
            {(["probability", "grade", "thickness", "confidence"] as const).map((item) => (
              <button key={item} className={layer === item ? "chip active" : "chip"} onClick={() => setLayer(item)}>
                {item}
              </button>
            ))}
          </div>
        </div>
        <ReserveMap cells={cells} selectedCell={selectedCell} onSelect={setSelectedCell} layer={layer} boreholes={boreholes} />
      </section>
      <section className="panel detail-panel">
        <PanelHeader icon={Target} title="Cell Detail" meta={selectedCell?.prospectivity_class ?? "select a cell"} />
        {selectedCell ? (
          <div className="detail-stack">
            <h3>{selectedCell.id}</h3>
            <MetricLine label="Coordinates" value={`${number(selectedCell.latitude, 4)}, ${number(selectedCell.longitude, 4)}`} />
            <MetricLine label="Probability" value={percent(selectedCell.probability)} />
            <MetricLine label="Grade" value={`${number(selectedCell.predicted_grade_pct, 1)}% Mn`} />
            <MetricLine label="Thickness" value={`${number(selectedCell.predicted_thickness_m, 1)} m`} />
            <MetricLine label="Confidence" value={percent(selectedCell.confidence)} />
            <MetricLine label="Resource P50" value={`${compactNumber(selectedCell.resource_potential.p50)} t`} />
            <p className="muted">{String(selectedCell.resource_potential.assumptions.classification_boundary ?? "")}</p>
            <DriverList drivers={selectedCell.top_contributors} />
          </div>
        ) : (
          <p className="muted">Click a high-prospectivity cell to inspect grade, thickness, confidence and prototype resource potential.</p>
        )}
      </section>
      <section className="panel">
        <PanelHeader icon={Gauge} title="Summary" meta="prototype" />
        <MetricLine label="High cells" value={number(summary.high_prospectivity_cells)} />
        <MetricLine label="Very high cells" value={number(summary.very_high_prospectivity_cells)} />
        <MetricLine label="Average grade" value={`${number(summary.average_predicted_grade_pct, 1)}% Mn`} />
        <MetricLine label="Average thickness" value={`${number(summary.average_predicted_thickness_m, 1)} m`} />
        <p className="muted">{summary.validation_note}</p>
      </section>
    </div>
  );
}

function ProductionPage({ forecast, history }: { forecast: ProductionForecastResponse; history: ProductionHistoryRecord[] }) {
  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={Activity} label="Forecast" value={`${compactNumber(forecast.forecast_mt)} t`} />
        <MetricCard icon={Target} label="Target" value={`${compactNumber(forecast.target_mt)} t`} />
        <MetricCard icon={AlertTriangle} label="Gap" value={`${signedNumber(forecast.gap_mt)} t`} tone={forecast.gap_mt < 0 ? "risk" : "ok"} />
        <MetricCard icon={Gauge} label="Risk" value={percent(forecast.shortfall_probability)} tone={forecast.shortfall_probability > 0.65 ? "risk" : "ok"} />
      </section>
      <section className="panel wide chart-panel">
        <PanelHeader icon={BarChart3} title="Actual vs Target" meta={forecast.model_version} />
        <ResponsiveContainer width="100%" height={330}>
          <LineChart data={history}>
            <CartesianGrid stroke="#d8ded7" strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={28} />
            <YAxis width={68} />
            <Tooltip />
            <Line type="monotone" dataKey="production_mt" stroke="#1f7a5f" strokeWidth={2} dot={false} />
            <Line type="monotone" dataKey="target_mt" stroke="#9a5323" strokeWidth={2} dot={false} />
          </LineChart>
        </ResponsiveContainer>
      </section>
      <section className="panel">
        <PanelHeader icon={AlertTriangle} title="Top Drivers" meta={forecast.severity} />
        <DriverList drivers={forecast.top_drivers} />
      </section>
      <section className="panel">
        <PanelHeader icon={Gauge} title="Prediction Interval" meta={forecast.forecast_date} />
        <MetricLine label="P10" value={`${compactNumber(forecast.prediction_interval.p10)} t`} />
        <MetricLine label="P50" value={`${compactNumber(forecast.prediction_interval.p50)} t`} />
        <MetricLine label="P90" value={`${compactNumber(forecast.prediction_interval.p90)} t`} />
        <MetricLine label="Baseline" value={`${compactNumber(forecast.baseline_forecast_mt)} t`} />
      </section>
    </div>
  );
}

function EquipmentPage({ equipment }: { equipment: EquipmentResponse }) {
  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={ShieldCheck} label="Fleet availability" value={percent(equipment.fleet_availability)} />
        <MetricCard icon={Gauge} label="Fleet utilization" value={percent(equipment.fleet_utilization)} />
        <MetricCard icon={AlertTriangle} label="Critical equipment" value={number(equipment.critical_equipment_count)} tone={equipment.critical_equipment_count > 0 ? "risk" : "ok"} />
      </section>
      <section className="panel wide chart-panel">
        <PanelHeader icon={Wrench} title="Downtime Ranking" meta={`${equipment.items.length} assets · DEMO telemetry`} />
        <ResponsiveContainer width="100%" height={330}>
          <BarChart data={equipment.items}>
            <CartesianGrid stroke="#d8ded7" strokeDasharray="3 3" />
            <XAxis dataKey="equipment_id" />
            <YAxis />
            <Tooltip />
            <Bar dataKey="downtime_7d_hours" fill="#9a5323" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>
      <section className="table-panel">
        <table>
          <thead>
            <tr>
              <th>Asset</th>
              <th>Type</th>
              <th>Availability</th>
              <th>Utilization</th>
              <th>Status</th>
            </tr>
          </thead>
          <tbody>
            {equipment.items.map((item) => (
              <tr key={item.equipment_id}>
                <td>{item.equipment_id}</td>
                <td>{item.equipment_type}</td>
                <td>{percent(item.availability)}</td>
                <td>{percent(item.utilization)}</td>
                <td>
                  <span className={`priority ${item.status.toLowerCase()}`}>{item.status}</span>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </section>
    </div>
  );
}

function WeatherBlastingPage({ weather, blasting }: { weather: WeatherResponse; blasting: BlastingResponse }) {
  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={CloudRain} label="7-day rainfall" value={`${number(weather.rainfall_7d_mm, 1)} mm`} tone={weather.weather_risk === "HIGH" ? "risk" : "default"} />
        <MetricCard icon={Gauge} label="Soil moisture" value={percent(weather.soil_moisture)} />
        <MetricCard icon={Target} label="Planned blasts" value={number(blasting.planned_blasts_7d)} />
        <MetricCard icon={AlertTriangle} label="Overlap risk" value={blasting.overlap_risk} tone={blasting.overlap_risk === "HIGH" ? "risk" : "default"} />
      </section>
      <section className="panel wide chart-panel">
        <PanelHeader icon={CloudRain} title="Weather Observations" meta={weather.latest_date} />
        <ResponsiveContainer width="100%" height={330}>
          <AreaChart data={weather.observations}>
            <CartesianGrid stroke="#d8ded7" strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={18} />
            <YAxis />
            <Tooltip />
            <Area type="monotone" dataKey="rainfall_mm" stroke="#1f7a5f" fill="#b7d8c8" />
          </AreaChart>
        </ResponsiveContainer>
      </section>
      <section className="panel wide chart-panel">
        <PanelHeader icon={Target} title="Blasting Delay" meta={blasting.delay_trend} />
        <ResponsiveContainer width="100%" height={260}>
          <BarChart data={blasting.events}>
            <CartesianGrid stroke="#d8ded7" strokeDasharray="3 3" />
            <XAxis dataKey="date" minTickGap={18} />
            <YAxis />
            <Tooltip />
            <Bar dataKey="blasting_delay_hours" fill="#344b6f" radius={[4, 4, 0, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>
    </div>
  );
}

function RecommendationsPage({ initial }: { initial: RecommendationResponse }) {
  const [items, setItems] = useState(initial.recommendations);
  const [downtime, setDowntime] = useState(15);
  async function simulate() {
    const response = await apiPost<RecommendationResponse>("/api/v1/recommendations/simulate", {
      reduce_downtime_pct: downtime,
      defer_weather_sensitive_blasts: true
    });
    setItems(response.recommendations);
  }
  return (
    <div className="page-grid">
      <section className="panel wide">
        <PanelHeader icon={ClipboardList} title="Recommendation Queue" meta={`${items.length} items`} />
        <div className="toolbar">
          <label>
            Simulated downtime reduction
            <input type="range" min="0" max="40" value={downtime} onChange={(event) => setDowntime(Number(event.target.value))} />
            <strong>{downtime}%</strong>
          </label>
          <button className="chip active" onClick={() => void simulate()}>
            Simulate impact
          </button>
        </div>
        <div className="recommendation-list">
          {items.map((item) => (
            <article className="recommendation" key={item.id}>
              <div className="rec-head">
                <span className={`priority ${item.priority.toLowerCase()}`}>{item.priority}</span>
                <span>{item.category}</span>
                <span>{item.status}</span>
                <strong>{percent(item.confidence)}</strong>
              </div>
              <h3>{item.title}</h3>
              <p>{item.rationale}</p>
              <div className="evidence-grid">
                {Object.entries(item.evidence).map(([key, value]) => (
                  <MetricLine key={key} label={key.replaceAll("_", " ")} value={String(value)} />
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>
    </div>
  );
}

function HealthPage({
  models,
  dataQuality,
  boundary
}: {
  models: ModelRegistryResponse;
  dataQuality: DataQualityResponse;
  boundary: string;
}) {
  return (
    <div className="page-grid">
      <section className="panel wide">
        <PanelHeader icon={Database} title="Model Registry" meta={`${models.models.length} versions`} />
        <div className="model-grid">
          {models.models.map((model) => (
            <article className="model-row" key={`${model.model_name}-${model.version}`}>
              <strong>{model.model_name}</strong>
              <span>{model.version}</span>
              <span>{model.algorithm}</span>
              <span className="badge">{model.status}</span>
            </article>
          ))}
        </div>
      </section>
      <section className="panel wide">
        <PanelHeader icon={ShieldCheck} title="Data Quality" meta={percent(dataQuality.overall_score)} />
        <div className="quality-table">
          {dataQuality.runs.map((run) => (
            <div className="quality-row" key={run.dataset_name}>
              <span>{run.dataset_name}</span>
              <strong>{percent(run.quality_score)}</strong>
              <span>{number(run.row_count)} rows</span>
              <span>{run.schema_valid ? "schema ok" : "schema watch"}</span>
            </div>
          ))}
        </div>
      </section>
      <section className="panel wide">
        <PanelHeader icon={Settings} title="Boundary" meta="human approval" />
        <p className="boundary">{boundary}</p>
      </section>
    </div>
  );
}

function SettingsPage({ safety }: { safety: { mode: string; boundary: string } | null }) {
  return (
    <div className="page-grid">
      <section className="panel wide">
        <PanelHeader icon={ShieldCheck} title="Safe settings" meta={safety?.mode ?? "demo"} />
        <p className="boundary">{safety?.boundary}</p>
        <p className="muted">No operational dispatch, blasting design, or official reserve classification is enabled in this prototype.</p>
      </section>
    </div>
  );
}

function MetricCard({
  icon: Icon,
  label,
  value,
  tone = "default"
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  tone?: "default" | "risk" | "ok";
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <Icon size={19} />
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

function PanelHeader({ icon: Icon, title, meta }: { icon: typeof Activity; title: string; meta?: string }) {
  return (
    <div className="panel-header">
      <div>
        <Icon size={18} />
        <h3>{title}</h3>
      </div>
      {meta && <span>{meta}</span>}
    </div>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function DriverList({ drivers }: { drivers: Array<{ feature: string; direction: string; importance: number; value?: number | string | null }> }) {
  return (
    <div className="driver-list">
      {drivers.map((driver) => (
        <div className="driver" key={driver.feature}>
          <div>
            <span>{driver.feature.replaceAll("_", " ")}</span>
            <strong>{driver.direction}</strong>
          </div>
          <div className="mini-bar">
            <span style={{ width: `${Math.max(6, driver.importance * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}

export default App;
