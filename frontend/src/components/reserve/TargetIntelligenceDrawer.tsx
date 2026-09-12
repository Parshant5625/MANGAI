import { X, Crosshair, Satellite, Mountain, Drill, ShieldCheck, ArrowUpRight } from "lucide-react";
import type { ProspectivityCell } from "../../types/api";
import { compactNumber, number, percent } from "../../utils/format";

interface TargetIntelligenceDrawerProps {
  cell: ProspectivityCell | null;
  onClose: () => void;
}

export function TargetIntelligenceDrawer({ cell, onClose }: TargetIntelligenceDrawerProps) {
  if (!cell) return null;
  const potential = cell.resource_potential?.p50 ?? 0;
  const support = cell.confidence >= 0.8 ? "HIGH" : cell.confidence >= 0.6 ? "MEDIUM" : "LIMITED";
  const signal = cell.probability >= 0.8 ? "STRONG" : cell.probability >= 0.6 ? "MODERATE" : "WEAK";

  return (
    <aside className="mangai-target-drawer" aria-label="Target intelligence">
      <div className="mangai-drawer-head">
        <div>
          <span className="mangai-micro-label"><Crosshair size={12} /> TARGET INTELLIGENCE</span>
          <h2>{cell.id}</h2>
        </div>
        <button type="button" className="mangai-drawer-close" onClick={onClose} aria-label="Close target drawer"><X size={18} /></button>
      </div>

      <div className="mangai-target-score">
        <div><span>PROSPECTIVITY</span><strong>{percent(cell.probability)}</strong></div>
        <div className="mangai-score-track"><span style={{ width: `${Math.max(2, cell.probability * 100)}%` }} /></div>
      </div>

      <div className="mangai-target-metrics">
        <Metric label="Predicted Mn grade" value={`${number(cell.predicted_grade_pct, 1)}%`} />
        <Metric label="Predicted thickness" value={`${number(cell.predicted_thickness_m, 1)} m`} />
        <Metric label="Confidence" value={percent(cell.confidence)} />
        <Metric label="Prototype resource P50" value={`${compactNumber(potential)} t`} />
      </div>

      <section className="mangai-evidence-block">
        <span className="mangai-micro-label">WHY THIS TARGET</span>
        <Evidence icon={Mountain} label="Geological support" value={support} />
        <Evidence icon={Satellite} label="Satellite signal" value={signal} />
        <Evidence icon={Drill} label="Subsurface context" value={cell.top_contributors?.length ? "AVAILABLE" : "LIMITED"} />
        <Evidence icon={ShieldCheck} label="Decision confidence" value={support} />
      </section>

      <div className="mangai-target-location">
        <span>LOCATION</span>
        <strong>{number(cell.latitude, 5)}, {number(cell.longitude, 5)}</strong>
      </div>

      <button type="button" className="mangai-investigate-button">
        <ArrowUpRight size={16} />
        PRIORITIZE INVESTIGATION
      </button>
      <p className="mangai-drawer-disclaimer">Decision-support prototype. This estimate is not an official mineral-reserve classification.</p>
    </aside>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div className="mangai-target-metric"><span>{label}</span><strong>{value}</strong></div>;
}

function Evidence({ icon: Icon, label, value }: { icon: typeof Mountain; label: string; value: string }) {
  return <div className="mangai-evidence-row"><span><Icon size={14} />{label}</span><strong>{value}</strong></div>;
}
