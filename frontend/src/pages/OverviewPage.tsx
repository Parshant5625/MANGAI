import { Activity, AlertTriangle, BarChart3, CheckCircle2, ClipboardList, Database, Satellite, Target, Wrench } from "lucide-react";
import { DriverList, PanelHeader } from "../components/ui";
import { OverviewReserveIntelligence } from "../components/OverviewReserveIntelligence";
import { OverviewSignalRail } from "../components/OverviewSignalRail";
import { DataQualityResponse, EquipmentResponse, OverviewResponse, ProductionForecastResponse, RecommendationResponse } from "../types/api";
import { compactNumber, number, percent, signedNumber } from "../utils/format";

export function OverviewPage({ overview, production, recommendations, equipment, quality }: { overview: OverviewResponse; production: ProductionForecastResponse; recommendations: RecommendationResponse; equipment: EquipmentResponse; quality: DataQualityResponse }) {
  const risk = overview.shortfall_probability;
  const riskLabel = risk >= 0.75 ? "CRITICAL" : risk >= 0.65 ? "HIGH" : risk >= 0.45 ? "WATCH" : "LOW";
  const topRecommendation = recommendations.recommendations[0];

  return (
    <div className="mangai-page mangai-command-center-page">
      <section className="command-center-header">
        <div>
          <div className="mangai-page-kicker"><Satellite size={12}/> MINE INTELLIGENCE COMMAND CENTER</div>
          <h1 className="mangai-page-title">See the mine. Understand the risk. Decide what to investigate next.</h1>
          <p className="mangai-page-description">A cross-domain intelligence surface connecting geology, satellite context, production, fleet health, environment and evidence-backed actions.</p>
        </div>
        <div className="command-center-status">
          <div><span className="status-pulse"/> OPERATIONAL</div>
          <small>Site {overview.site_id} · Model {production.model_version}</small>
        </div>
      </section>

      <section className="command-metric-strip">
        <CommandMetric label="SHORTFALL RISK" value={percent(risk)} detail={riskLabel} tone={riskLabel === "LOW" ? "good" : "risk"}/>
        <CommandMetric label="7D PRODUCTION" value={`${compactNumber(production.forecast_mt)} t`} detail={`${signedNumber(production.gap_mt)} vs target`} tone={production.gap_mt < 0 ? "risk" : "good"}/>
        <CommandMetric label="PROTOTYPE RESOURCE" value={`${compactNumber(overview.resource_potential_tonnage)} t`} detail={`${number(overview.high_prospectivity_area_ha, 1)} ha high prospectivity`}/>
        <CommandMetric label="FLEET PULSE" value={percent(equipment.fleet_utilization)} detail={`${equipment.critical_equipment_count} critical assets`} tone={equipment.critical_equipment_count ? "warn" : "good"}/>
        <CommandMetric label="DATA QUALITY" value={percent(quality.overall_score)} detail={overview.model_health} tone={quality.overall_score >= .85 ? "good" : "warn"}/>
      </section>

      <section className="command-main-grid">
        <div className="command-map-stage">
          <OverviewReserveIntelligence />
        </div>
        <aside className="command-intelligence-rail">
          <OverviewSignalRail />
          <section className="command-driver-panel">
            <PanelHeader icon={AlertTriangle} title="Risk drivers" meta={production.severity}/>
            <DriverList drivers={production.top_drivers} />
          </section>
        </aside>
      </section>

      <section className="command-bottom-grid">
        <section className="panel command-forecast-panel">
          <PanelHeader icon={BarChart3} title="Production outlook" meta="7 DAY / DECISION WINDOW"/>
          <div className="command-forecast-values">
            <div><span>FORECAST</span><strong>{compactNumber(production.forecast_mt)} t</strong></div>
            <div><span>TARGET</span><strong>{compactNumber(production.target_mt)} t</strong></div>
            <div><span>GAP</span><strong className={production.gap_mt < 0 ? "negative" : "positive"}>{signedNumber(production.gap_mt)} t</strong></div>
          </div>
          <div className="command-risk-track"><span style={{ width: `${Math.max(3, Math.min(100, risk * 100))}%` }}/></div>
          <div className="command-footnote"><span>Shortfall probability</span><b>{percent(risk)}</b></div>
        </section>

        <section className="panel command-action-panel">
          <PanelHeader icon={ClipboardList} title="Next investigation" meta="AI ACTION CENTER"/>
          {topRecommendation ? <>
            <span className={`command-priority ${topRecommendation.priority.toLowerCase()}`}>{topRecommendation.priority}</span>
            <h3>{topRecommendation.title}</h3>
            <p>{topRecommendation.rationale}</p>
            <div className="command-evidence"><span>CONFIDENCE <b>{percent(topRecommendation.confidence)}</b></span><span>STATUS <b>HUMAN REVIEW</b></span></div>
          </> : <p>No active recommendations.</p>}
        </section>

        <section className="panel command-system-panel">
          <PanelHeader icon={Database} title="System pulse" meta="LIVE SERVICES"/>
          <SystemRow icon={Target} label="Reserve intelligence" value="READY"/>
          <SystemRow icon={Activity} label="Production intelligence" value="READY"/>
          <SystemRow icon={Wrench} label="Fleet intelligence" value="READY"/>
          <SystemRow icon={Satellite} label="Satellite fusion" value={overview.synthetic_data ? "DEMO" : "LIVE"}/>
          <SystemRow icon={CheckCircle2} label="Decision boundary" value="ENFORCED"/>
        </section>
      </section>

      <section className="command-boundary">
        <span>DECISION SUPPORT SYSTEM</span>
        <p>{overview.boundary_notice}</p>
      </section>
    </div>
  );
}

function CommandMetric({ label, value, detail, tone = "neutral" }: { label: string; value: string; detail: string; tone?: "neutral" | "good" | "warn" | "risk" }) {
  return <div className={`command-metric ${tone}`}><span>{label}</span><strong>{value}</strong><small>{detail}</small></div>;
}
function SystemRow({ icon: Icon, label, value }: { icon: typeof Target; label: string; value: string }) {
  return <div className="system-row"><span className="system-icon"><Icon size={13}/></span><span>{label}</span><b>{value}</b></div>;
}
