import { Activity, AlertTriangle, Cpu, Crosshair, Database, Gauge, Layers, Satellite, ShieldCheck, Target, TrendingDown, Wrench } from "lucide-react";
import type { OverviewResponse, ProductionForecastResponse, RecommendationResponse, EquipmentResponse, DataQualityResponse } from "../../types/api";

type Props = {
  overview: OverviewResponse;
  production: ProductionForecastResponse;
  recommendations: RecommendationResponse;
  equipment: EquipmentResponse;
  quality: DataQualityResponse;
  onOpenReserve?: () => void;
  onOpenProduction?: () => void;
};

function pct(value: number | null | undefined) {
  if (value == null || Number.isNaN(value)) return "—";
  return `${(value * 100).toFixed(0)}%`;
}

function num(value: number | null | undefined, digits = 0) {
  if (value == null || Number.isNaN(value)) return "—";
  return value.toLocaleString("en-IN", { maximumFractionDigits: digits });
}

export function CommandCenterOverview({ overview, production, recommendations, equipment, quality, onOpenReserve, onOpenProduction }: Props) {
  const risk = production.shortfall_probability ?? 0;
  const fleet = equipment.fleet_utilization ?? 0;
  const health = quality.overall_quality_score ?? 0;
  const lead = production.top_drivers?.[0];
  const topRecommendation = recommendations.recommendations?.[0];

  return (
    <section className="cc-page" aria-label="MANGAI mining intelligence command center">
      <header className="cc-hero">
        <div>
          <span className="cc-kicker"><Crosshair size={13} /> MINE INTELLIGENCE COMMAND CENTER</span>
          <h1>{overview.site_name ?? "MANGAI Demo Mine"}</h1>
          <p>Decision-support view across geology, production, fleet and environmental signals.</p>
        </div>
        <div className="cc-state"><span className="cc-pulse" /> OPERATIONAL <small>DECISION SUPPORT</small></div>
      </header>

      <div className="cc-grid">
        <article className="cc-map-card cc-panel cc-span-8">
          <div className="cc-panel-head">
            <div><span className="cc-label">GEOSPATIAL INTELLIGENCE</span><strong>Prospectivity field</strong></div>
            <button type="button" onClick={onOpenReserve}><Layers size={14} /> OPEN RESERVE</button>
          </div>
          <div className="cc-map-stage">
            <div className="cc-map-grid" />
            <div className="cc-map-contours" />
            <div className="cc-map-glow glow-a" /><div className="cc-map-glow glow-b" /><div className="cc-map-glow glow-c" />
            <div className="cc-map-target t1">A-042<span>92%</span></div>
            <div className="cc-map-target t2">B-017<span>84%</span></div>
            <div className="cc-map-target t3">C-031<span>76%</span></div>
            <div className="cc-map-scan" />
            <div className="cc-map-legend"><span>LOW</span><i /><span>PROSPECTIVITY</span><b>HIGH</b></div>
            <div className="cc-map-meta"><Satellite size={13} /> SATELLITE CONTEXT <em>DEMO / SYNTHETIC GEOLOGY</em></div>
          </div>
        </article>

        <aside className="cc-panel cc-span-4 cc-rail">
          <div className="cc-panel-head"><div><span className="cc-label">LIVE SIGNALS</span><strong>Intelligence rail</strong></div><Activity size={15} /></div>
          <div className="cc-signal cc-risk"><div><span>PRODUCTION RISK</span><strong>{pct(risk)}</strong></div><small>{lead ? `Primary: ${lead.feature ?? lead.name ?? "operational driver"}` : "Monitoring operational drivers"}</small><div className="cc-bar"><i style={{ width: `${Math.min(100, risk * 100)}%` }} /></div></div>
          <div className="cc-signal"><div><span>FLEET UTILIZATION</span><strong>{pct(fleet)}</strong></div><small>{equipment.critical_equipment_count ?? 0} critical assets detected</small><div className="cc-bar"><i style={{ width: `${Math.min(100, fleet * 100)}%` }} /></div></div>
          <div className="cc-signal"><div><span>DATA HEALTH</span><strong>{pct(health)}</strong></div><small>Pipeline quality across intelligence sources</small><div className="cc-bar"><i style={{ width: `${Math.min(100, health * 100)}%` }} /></div></div>
          <div className="cc-action"><span>AI PRIORITY</span><strong>{topRecommendation?.title ?? "Review current operational risk signals"}</strong><small>{topRecommendation?.rationale ?? "No priority recommendation is currently available."}</small></div>
        </aside>

        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><TrendingDown size={13} /> PRODUCTION OUTLOOK</span><strong>{num(production.forecast_mt)} t</strong><small>7-day forecast</small><button type="button" onClick={onOpenProduction}>VIEW FORECAST <span>→</span></button></article>
        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><Target size={13} /> SHORTFALL GAP</span><strong className={production.gap_mt < 0 ? "risk-text" : "ok-text"}>{num(production.gap_mt)} t</strong><small>against target {num(production.target_mt)} t</small></article>
        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><Wrench size={13} /> FLEET PULSE</span><strong>{pct(fleet)}</strong><small>{num(equipment.items?.length)} tracked assets</small></article>

        <article className="cc-panel cc-span-7 cc-relationship">
          <div className="cc-panel-head"><div><span className="cc-label">SYSTEM RELATIONSHIPS</span><strong>Mine digital thread</strong></div><Database size={15} /></div>
          <div className="cc-thread"><span><Satellite size={14} /> ENVIRONMENT</span><b>→</b><span><Layers size={14} /> GEOLOGY</span><b>→</b><span><Target size={14} /> RESERVE</span><b>→</b><span><Wrench size={14} /> FLEET</span><b>→</b><span><Gauge size={14} /> PRODUCTION</span></div>
          <p>Signals are presented as decision support; operational actions require human validation.</p>
        </article>
        <article className="cc-panel cc-span-5 cc-integrity">
          <div className="cc-panel-head"><div><span className="cc-label">SYSTEM INTEGRITY</span><strong>Platform state</strong></div><ShieldCheck size={15} /></div>
          <div className="cc-integrity-row"><span><Cpu size={13} /> AI services</span><b>READY</b></div>
          <div className="cc-integrity-row"><span><Database size={13} /> Data pipeline</span><b>{pct(health)}</b></div>
          <div className="cc-integrity-row"><span><AlertTriangle size={13} /> Risk engine</span><b className={risk > .65 ? "warn" : ""}>{risk > .65 ? "ATTENTION" : "MONITORING"}</b></div>
        </article>
      </div>
    </section>
  );
}
