import { Activity, AlertTriangle, Cpu, Crosshair, Database, Gauge, Layers, Satellite, ShieldCheck, Target, TrendingDown, Wrench } from "lucide-react";
import { DataQualityResponse, EquipmentResponse, OverviewResponse, ProductionForecastResponse, RecommendationResponse, ProspectivityCell } from "../types/api";
import { ReserveMap } from "../components/ReserveMap";
import { number } from "../utils/format";

export function OverviewPage({ overview, production, recommendations, equipment, quality, reserveCells, selectedCell, onSelectCell }: {
  overview: OverviewResponse;
  production: ProductionForecastResponse;
  recommendations: RecommendationResponse;
  equipment: EquipmentResponse;
  quality: DataQualityResponse;
  reserveCells: ProspectivityCell[];
  selectedCell: ProspectivityCell | null;
  onSelectCell: (cell: ProspectivityCell) => void;
}) {
  const risk = overview.shortfall_probability ?? production.shortfall_probability ?? 0;
  const fleet = equipment.fleet_utilization ?? 0;
  const health = quality.overall_score ?? overview.data_quality_score ?? 0;
  const lead = production.top_drivers?.[0];
  const topRecommendation = recommendations.recommendations?.[0];
  const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
  const tonnes = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString("en-IN", { maximumFractionDigits: 0 });
  const selected = selectedCell;

  return (
    <section className="cc-page" aria-label="MANGAI mining intelligence command center">
      <header className="cc-hero">
        <div><span className="cc-kicker"><Crosshair size={13} /> MINE INTELLIGENCE COMMAND CENTER</span><h1>{overview.site_name || "MANGAI Demo Mine"}</h1><p>Environment → geology → reserve → fleet → production, unified for operational investigation.</p></div>
        <div className="cc-state"><span className="cc-pulse" /> OPERATIONAL <small>{overview.data_mode || "DECISION SUPPORT"}</small></div>
      </header>
      <div className="cc-grid">
        <article className="cc-map-card cc-panel cc-span-8">
          <div className="cc-panel-head"><div><span className="cc-label">GEOSPATIAL INTELLIGENCE</span><strong>Prospectivity field</strong></div><span className="cc-map-mode"><Satellite size={13}/> {reserveCells.length} LIVE CELLS</span></div>
          <div className="cc-real-map"><ReserveMap cells={reserveCells} selectedCell={selected} onSelect={onSelectCell} layer="probability" boreholes={[]} /></div>
          <div className="cc-map-footer"><span><Layers size={13}/> PROBABILITY OVERLAY</span><span>Thresholded reserve intelligence</span><em>PROTOTYPE / DECISION SUPPORT</em></div>
        </article>
        <aside className="cc-panel cc-span-4 cc-rail">
          <div className="cc-panel-head"><div><span className="cc-label">LIVE SIGNALS</span><strong>Intelligence rail</strong></div><Activity size={15}/></div>
          <div className="cc-signal cc-risk"><div><span>PRODUCTION RISK</span><strong>{pct(risk)}</strong></div><small>{lead ? `Primary: ${lead.feature}` : "Monitoring operational drivers"}</small><div className="cc-bar"><i style={{width:`${Math.min(100,risk*100)}%`}}/></div></div>
          <div className="cc-signal"><div><span>FLEET UTILIZATION</span><strong>{pct(fleet)}</strong></div><small>{equipment.critical_equipment_count} critical assets detected</small><div className="cc-bar"><i style={{width:`${Math.min(100,fleet*100)}%`}}/></div></div>
          <div className="cc-signal"><div><span>DATA HEALTH</span><strong>{pct(health)}</strong></div><small>Pipeline quality across intelligence sources</small><div className="cc-bar"><i style={{width:`${Math.min(100,health*100)}%`}}/></div></div>
          <div className="cc-action"><span>AI PRIORITY</span><strong>{topRecommendation?.title ?? "Review current operational risk signals"}</strong><small>{topRecommendation?.rationale ?? "No priority recommendation is currently available."}</small></div>
        </aside>
        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><TrendingDown size={13}/> PRODUCTION OUTLOOK</span><strong>{tonnes(production.forecast_mt)} t</strong><small>{production.horizon_days}-day forecast · {production.model_version}</small><div className="cc-mini-track"><i style={{width:`${Math.min(100,(production.forecast_mt/Math.max(1,production.target_mt))*100)}%`}}/></div></article>
        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><Target size={13}/> SHORTFALL GAP</span><strong className={production.gap_mt < 0 ? "risk-text" : "ok-text"}>{tonnes(production.gap_mt)} t</strong><small>against target {tonnes(production.target_mt)} t</small></article>
        <article className="cc-panel cc-span-4 cc-metric-feature"><span className="cc-label"><Wrench size={13}/> FLEET PULSE</span><strong>{pct(fleet)}</strong><small>{number(equipment.items?.length ?? 0)} tracked assets</small></article>
        <article className="cc-panel cc-span-7 cc-relationship"><div className="cc-panel-head"><div><span className="cc-label">SELECTED TARGET</span><strong>{selected?.id ?? "Select a prospectivity cell"}</strong></div><Target size={15}/></div>{selected ? <div className="cc-target-grid"><div><span>PROSPECTIVITY</span><strong>{pct(selected.probability)}</strong></div><div><span>GRADE</span><strong>{number(selected.predicted_grade_pct,1)}% Mn</strong></div><div><span>THICKNESS</span><strong>{number(selected.predicted_thickness_m,1)} m</strong></div><div><span>CONFIDENCE</span><strong>{pct(selected.confidence)}</strong></div></div> : <p>Selecting a target exposes model outputs and evidence for the next investigation step.</p>}<p className="cc-note">Prototype resource potential is not an official reserve estimate; human validation required.</p></article>
        <article className="cc-panel cc-span-5 cc-integrity"><div className="cc-panel-head"><div><span className="cc-label">SYSTEM INTEGRITY</span><strong>Platform state</strong></div><ShieldCheck size={15}/></div><div className="cc-integrity-row"><span><Cpu size={13}/> AI services</span><b>READY</b></div><div className="cc-integrity-row"><span><Database size={13}/> Data pipeline</span><b>{pct(health)}</b></div><div className="cc-integrity-row"><span><AlertTriangle size={13}/> Risk engine</span><b className={risk > .65 ? "warn" : ""}>{risk > .65 ? "ATTENTION" : "MONITORING"}</b></div></article>
      </div>
    </section>
  );
}
