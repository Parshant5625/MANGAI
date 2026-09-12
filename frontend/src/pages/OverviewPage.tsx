import { useState } from "react";
import { Activity, AlertTriangle, Cpu, Crosshair, Database, Layers, MessageSquare, Send, Satellite, ShieldCheck, Target, TrendingDown, Wrench } from "lucide-react";
import { apiPost } from "../api/client";
import { DataQualityResponse, EquipmentResponse, OverviewResponse, ProductionForecastResponse, RecommendationResponse, ProspectivityCell } from "../types/api";
import { ReserveMap } from "../components/ReserveMap";
import { number } from "../utils/format";

export function OverviewPage({ overview, production, recommendations, equipment, quality, reserveCells }: {
  overview: OverviewResponse;
  production: ProductionForecastResponse;
  recommendations: RecommendationResponse;
  equipment: EquipmentResponse;
  quality: DataQualityResponse;
  reserveCells: ProspectivityCell[];
}) {
  const [selectedCell, setSelectedCell] = useState<ProspectivityCell | null>(null);
  const [question, setQuestion] = useState("");
  const [copilot, setCopilot] = useState<{answer:string;confidence?:number}|null>(null);
  const [copilotLoading, setCopilotLoading] = useState(false);
  const risk = overview.shortfall_probability ?? production.shortfall_probability ?? 0;
  const fleet = equipment.fleet_utilization ?? 0;
  const health = quality.overall_score ?? overview.data_quality_score ?? 0;
  const lead = production.top_drivers?.[0];
  const topRecommendation = recommendations.recommendations?.[0];
  const pct = (value: number) => `${(value * 100).toFixed(0)}%`;
  const tonnes = (value: number | null | undefined) => value == null ? "—" : value.toLocaleString("en-IN", { maximumFractionDigits: 0 });

  const askCopilot = async () => {
    const prompt = question.trim();
    if (!prompt || copilotLoading) return;
    setCopilotLoading(true);
    try {
      const result = await apiPost<Record<string, unknown>>("/api/v1/chat", {
        message: prompt,
        context: {
          site_id: overview.site_id,
          data_mode: overview.data_mode,
          production_risk: risk,
          production_gap_mt: production.gap_mt,
          fleet_utilization: fleet,
          data_quality: health,
          selected_target: selectedCell?.id ?? null,
        },
      });
      const answer = String(result.answer ?? result.response ?? result.message ?? "No evidence-backed response was returned.");
      const rawConfidence = result.confidence;
      setCopilot({ answer, confidence: typeof rawConfidence === "number" ? rawConfidence : undefined });
    } catch (error) {
      setCopilot({ answer: `Copilot unavailable: ${String(error)}` });
    } finally {
      setCopilotLoading(false);
    }
  };

  return (
    <section className="cc-page" aria-label="MANGAI mining intelligence command center">
      <header className="cc-hero">
        <div><span className="cc-kicker"><Crosshair size={13} /> MINE INTELLIGENCE COMMAND CENTER</span><h1>{overview.site_name || "MANGAI Demo Mine"}</h1><p>Environment → geology → reserve → fleet → production, unified for operational investigation.</p></div>
        <div className="cc-state"><span className="cc-pulse" /> OPERATIONAL <small>{overview.data_mode || "DECISION SUPPORT"}</small></div>
      </header>
      <div className="cc-grid">
        <article className="cc-map-card cc-panel cc-span-8">
          <div className="cc-panel-head"><div><span className="cc-label">GEOSPATIAL INTELLIGENCE</span><strong>Prospectivity field</strong></div><span className="cc-map-mode"><Satellite size={13}/> {reserveCells.length} TARGET CELLS</span></div>
          <div className="cc-real-map"><ReserveMap cells={reserveCells} selectedCell={selectedCell} onSelect={setSelectedCell} layer="probability" boreholes={[]} /></div>
          <div className="cc-map-footer"><span><Layers size={13}/> PROBABILITY OVERLAY</span><span>Threshold ≥ 55% · click a target for intelligence</span><em>PROTOTYPE / DECISION SUPPORT</em></div>
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
        <article className="cc-panel cc-span-7 cc-relationship"><div className="cc-panel-head"><div><span className="cc-label">SELECTED TARGET</span><strong>{selectedCell?.id ?? "Select a prospectivity cell"}</strong></div><Target size={15}/></div>{selectedCell ? <div className="cc-target-grid"><div><span>PROSPECTIVITY</span><strong>{pct(selectedCell.probability)}</strong></div><div><span>GRADE</span><strong>{number(selectedCell.predicted_grade_pct,1)}% Mn</strong></div><div><span>THICKNESS</span><strong>{number(selectedCell.predicted_thickness_m,1)} m</strong></div><div><span>CONFIDENCE</span><strong>{pct(selectedCell.confidence)}</strong></div></div> : <p>Selecting a target exposes model outputs and evidence for the next investigation step.</p>}<p className="cc-note">Prototype resource potential is not an official reserve estimate; human validation required.</p></article>
        <article className="cc-panel cc-span-5 cc-integrity"><div className="cc-panel-head"><div><span className="cc-label">SYSTEM INTEGRITY</span><strong>Platform state</strong></div><ShieldCheck size={15}/></div><div className="cc-integrity-row"><span><Cpu size={13}/> AI services</span><b>READY</b></div><div className="cc-integrity-row"><span><Database size={13}/> Data pipeline</span><b>{pct(health)}</b></div><div className="cc-integrity-row"><span><AlertTriangle size={13}/> Risk engine</span><b className={risk > .65 ? "warn" : ""}>{risk > .65 ? "ATTENTION" : "MONITORING"}</b></div></article>
        <article className="cc-panel cc-span-12 cc-copilot">
          <div className="cc-panel-head"><div><span className="cc-label"><MessageSquare size={13}/> EVIDENCE COPILOT</span><strong>Ask about the current mine state</strong></div><span className="cc-map-mode">API /api/v1/chat</span></div>
          <div className="cc-copilot-body">
            <div className="cc-copilot-prompts">
              <button onClick={() => setQuestion("Why is production at risk right now?")}>Why is production at risk?</button>
              <button onClick={() => setQuestion("What should the operator investigate next?")}>What should we investigate next?</button>
              <button onClick={() => setQuestion("Which signals currently support the highest-priority action?")}>Show action evidence</button>
            </div>
            <div className="cc-copilot-row"><input value={question} onChange={e=>setQuestion(e.target.value)} onKeyDown={e=>{if(e.key==="Enter")void askCopilot()}} placeholder="Ask a decision-support question…" aria-label="Copilot question"/><button className="cc-copilot-send" onClick={()=>void askCopilot()} disabled={copilotLoading || !question.trim()}><Send size={14}/>{copilotLoading?"Analyzing":"Ask"}</button></div>
            {copilot && <div className="cc-copilot-answer"><div><span>RESPONSE</span>{copilot.confidence != null && <b>Confidence {pct(copilot.confidence)}</b>}</div><p>{copilot.answer}</p><small>Decision support only · verify against site procedures and accountable human operators.</small></div>}
          </div>
        </article>
      </div>
    </section>
  );
}
