import type { CSSProperties } from "react";
import {
  Activity,
  AlertTriangle,
  BarChart3,
  CheckCircle2,
  ClipboardList,
  Database,
  Map,
  Satellite,
  Target,
  Wrench
} from "lucide-react";
import { DriverList, MetricCard, PanelHeader } from "../components/ui";
import { OverviewReserveIntelligence } from "../components/OverviewReserveIntelligence";
import { OverviewSignalRail } from "../components/OverviewSignalRail";
import {
  DataQualityResponse,
  EquipmentResponse,
  OverviewResponse,
  ProductionForecastResponse,
  RecommendationResponse
} from "../types/api";
import { compactNumber, number, percent, signedNumber } from "../utils/format";

export function OverviewPage({ overview, production, recommendations, equipment, quality }: { overview: OverviewResponse; production: ProductionForecastResponse; recommendations: RecommendationResponse; equipment: EquipmentResponse; quality: DataQualityResponse }) {
  const healthScore = quality.overall_score;
  const topRecommendation = recommendations.recommendations[0];
  const riskLevel = overview.shortfall_probability > 0.75 ? "CRITICAL" : overview.shortfall_probability > 0.65 ? "HIGH" : overview.shortfall_probability > 0.45 ? "MEDIUM" : "LOW";

  return (
    <div className="page-grid overview-page">
      <section className="overview-hero">
        <div className="hero-copy">
          <div className="hero-kicker"><Satellite size={15} /> CROSS-DOMAIN MINING INTELLIGENCE</div>
          <h1>See the mine before you move it.</h1>
          <p>One operational view connecting geological prospectivity, satellite context, production risk, fleet health and AI-assisted actions.</p>
          <div className="hero-status-row">
            <span className="hero-status"><CheckCircle2 size={15} /> Intelligence services online</span>
            <span>Site · {overview.site_id}</span>
            <span>Model · {production.model_version}</span>
          </div>
        </div>
        <div className="hero-orbit" aria-hidden="true">
          <div className="orbit orbit-one" />
          <div className="orbit orbit-two" />
          <div className="orbit orbit-three" />
          <div className="hero-core"><span>M</span><small>AI</small></div>
          <span className="orbit-dot dot-one" /><span className="orbit-dot dot-two" /><span className="orbit-dot dot-three" />
        </div>
      </section>

      <section className="kpi-grid">
        <MetricCard icon={Target} label="Prototype resource potential" value={`${compactNumber(overview.resource_potential_tonnage)} t`} />
        <MetricCard icon={Map} label="High prospectivity area" value={`${number(overview.high_prospectivity_area_ha, 1)} ha`} />
        <MetricCard icon={BarChart3} label="Next 7-day production" value={`${compactNumber(overview.next_7_day_production_mt)} t`} />
        <MetricCard icon={AlertTriangle} label="Shortfall probability" value={percent(overview.shortfall_probability)} tone={overview.shortfall_probability > 0.65 ? "risk" : "ok"} />
      </section>

      <section className="kpi-grid">
        <MetricCard icon={Activity} label="Production gap" value={`${signedNumber(overview.production_gap_mt)} t`} tone={overview.production_gap_mt < 0 ? "risk" : "ok"} />
        <MetricCard icon={Wrench} label="Critical equipment" value={number(overview.critical_equipment_count)} tone={overview.critical_equipment_count > 0 ? "risk" : "ok"} />
        <MetricCard icon={ClipboardList} label="Recommendations" value={number(overview.recommendation_count)} />
        <MetricCard icon={Database} label="Data quality" value={percent(healthScore)} tone={healthScore >= 0.85 ? "ok" : "risk"} />
      </section>

      <OverviewReserveIntelligence />
      <OverviewSignalRail />

      <section className="panel wide chart-panel overview-forecast-panel">
        <PanelHeader icon={Activity} title="Production Forecast" meta={`${riskLevel} RISK`} />
        <div className="split">
          <div>
            <p className="massive">{signedNumber(production.gap_mt)} t</p>
            <p className="muted">Forecast gap against the 7-day target.</p>
            <div className="forecast-strip">
              <div><span>Forecast</span><strong>{compactNumber(production.forecast_mt)} t</strong></div>
              <div><span>Target</span><strong>{compactNumber(production.target_mt)} t</strong></div>
              <div><span>Risk</span><strong>{percent(production.shortfall_probability)}</strong></div>
            </div>
          </div>
          <DriverList drivers={production.top_drivers} />
        </div>
      </section>

      <section className="panel">
        <PanelHeader icon={Wrench} title="Fleet Pulse" meta={`${equipment.critical_equipment_count} critical`} />
        <div className="pulse-meter">
          <div className="pulse-ring"><strong>{percent(equipment.fleet_availability)}</strong><span>availability</span></div>
          <div className="pulse-details">
            <MetricLine label="Utilization" value={percent(equipment.fleet_utilization)} />
            {equipment.items.slice(0, 3).map((item) => <MetricLine key={item.equipment_id} label={item.equipment_id} value={`${number(item.downtime_7d_hours, 1)} h`} />)}
          </div>
        </div>
      </section>

      <section className="panel">
        <PanelHeader icon={ClipboardList} title="AI Action Queue" meta={`${recommendations.recommendations.length} proposed`} />
        <div className="stack">
          {recommendations.recommendations.slice(0, 3).map((item, index) => (
            <div className="action-row action-row-animated" key={item.id} style={{ animationDelay: `${index * 90}ms` }}>
              <span className={`priority ${item.priority.toLowerCase()}`}>{item.priority}</span><p>{item.title}</p>
            </div>
          ))}
        </div>
        {topRecommendation && <p className="muted queue-note">Highest-priority signal: {topRecommendation.category.replaceAll("_", " ")} · {percent(topRecommendation.confidence)} confidence.</p>}
      </section>

      <section className="panel">
        <PanelHeader icon={Database} title="Data & Model Health" meta={overview.model_health} />
        <div className="health-grid">
          <div className="health-ring" style={{ "--score": `${healthScore * 360}deg` } as CSSProperties}><strong>{percent(healthScore)}</strong><span>data quality</span></div>
          <div className="health-copy"><MetricLine label="Model status" value={overview.model_health} /><MetricLine label="Data mode" value={overview.synthetic_data ? "DEMO" : "LIVE"} /><MetricLine label="Recommendations" value={number(overview.recommendation_count)} /></div>
        </div>
      </section>

      <section className="panel wide insight-panel">
        <PanelHeader icon={Satellite} title="MANGAI Intelligence Loop" meta="geology → operations → action" />
        <div className="intelligence-loop">
          <LoopStep number="01" title="Discover" text="Map geological and satellite indicators into prospectivity signals." icon={Map} />
          <div className="loop-arrow">→</div>
          <LoopStep number="02" title="Predict" text="Forecast production and quantify shortfall drivers with uncertainty." icon={BarChart3} />
          <div className="loop-arrow">→</div>
          <LoopStep number="03" title="Act" text="Prioritize evidence-backed corrective actions for human approval." icon={ClipboardList} />
        </div>
      </section>

      <section className="panel wide boundary-panel">
        <PanelHeader icon={Database} title="Prototype boundary" meta="important" />
        <p className="boundary">{overview.boundary_notice}</p>
      </section>
    </div>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) { return <div className="metric-line"><span>{label}</span><strong>{value}</strong></div>; }
function LoopStep({ number: stepNumber, title, text, icon: Icon }: { number: string; title: string; text: string; icon: typeof Map }) { return <div className="loop-step"><div className="loop-icon"><Icon size={19} /><span>{stepNumber}</span></div><div><strong>{title}</strong><p>{text}</p></div></div>; }
