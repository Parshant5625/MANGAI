import { useState } from "react";
import { Layers, Satellite, Target } from "lucide-react";
import { useApi } from "../hooks/useApi";
import { ProspectivityCell, ReserveProspectivityResponse } from "../types/api";
import { compactNumber, number, percent } from "../utils/format";
import { ReserveMap } from "./ReserveMap";

type OverviewLayer = "probability" | "grade" | "thickness" | "confidence";

export function OverviewReserveIntelligence() {
  const [layer, setLayer] = useState<OverviewLayer>("probability");
  const [selectedCell, setSelectedCell] = useState<ProspectivityCell | null>(null);
  const reserve = useApi<ReserveProspectivityResponse>("/api/v1/reserves/prospectivity?limit=450&min_probability=0.35");
  const boreholes = useApi<{ boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }> }>("/api/v1/reserves/boreholes?limit=180");
  const cells = reserve.data?.cells ?? [];

  return (
    <section className="panel wide overview-reserve-command">
      <div className="overview-reserve-header">
        <div>
          <span className="section-kicker"><Satellite size={14} /> RESERVE INTELLIGENCE</span>
          <h2>See where the ore signal is strongest.</h2>
          <p>Satellite context + geology + ML prospectivity, with continuous heatmaps for probability, grade, thickness, confidence and thermal proxy.</p>
        </div>
        <div className="overview-layer-switch">
          {(["probability", "grade", "thickness", "confidence"] as const).map((item) => (
            <button key={item} className={layer === item ? "active" : ""} onClick={() => setLayer(item)}>{item}</button>
          ))}
          <button className="thermal-tab" onClick={() => document.querySelector<HTMLElement>(".reserve-map-layerbar button:last-child")?.click()}>Thermal</button>
        </div>
      </div>
      <div className="overview-reserve-grid">
        <div className="overview-map-wrap">
          <ReserveMap cells={cells} selectedCell={selectedCell} onSelect={setSelectedCell} layer={layer} boreholes={boreholes.data?.boreholes ?? []} />
        </div>
        <aside className="overview-reserve-detail">
          <div className="detail-kicker"><Target size={14} /> SELECTED TARGET</div>
          {selectedCell ? (
            <>
              <h3>{selectedCell.id}</h3>
              <div className="overview-target-badge">{selectedCell.prospectivity_class}</div>
              <div className="overview-detail-metrics">
                <Metric label="Probability" value={percent(selectedCell.probability)} />
                <Metric label="Mn grade" value={`${number(selectedCell.predicted_grade_pct, 1)}%`} />
                <Metric label="Thickness" value={`${number(selectedCell.predicted_thickness_m, 1)} m`} />
                <Metric label="Confidence" value={percent(selectedCell.confidence)} />
                <Metric label="Resource P50" value={`${compactNumber(selectedCell.resource_potential.p50)} t`} />
                <Metric label="Coordinates" value={`${number(selectedCell.latitude, 3)}, ${number(selectedCell.longitude, 3)}`} />
              </div>
              <p className="muted">Prototype resource potential only; not an official reserve classification.</p>
            </>
          ) : (
            <div className="overview-target-empty">
              <Layers size={28} />
              <strong>Click a prospectivity cell</strong>
              <p>Inspect probability, Mn grade, thickness, confidence and prototype resource potential.</p>
              <span className="thermal-note">Thermal is displayed as a demo proxy until spatial Landsat LST is connected.</span>
            </div>
          )}
        </aside>
      </div>
    </section>
  );
}

function Metric({ label, value }: { label: string; value: string }) {
  return <div><span>{label}</span><strong>{value}</strong></div>;
}
