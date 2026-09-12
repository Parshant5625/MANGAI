import { Satellite, Thermometer, Cloud, Activity, Radio } from "lucide-react";
import { PanelHeader, MetricLine } from "../components/command-center/IntelligencePrimitives";

type Scene = { id?: string; source?: string; date?: string; cloud_cover?: number; quality?: number; latitude?: number; longitude?: number; bands?: string[]; thermal_coverage?: number };

export function SatellitePage({ scenes = [] }: { scenes?: Scene[] }) {
  const thermal = scenes.filter(s => (s.source ?? "").toLowerCase().includes("landsat") || s.thermal_coverage !== undefined);
  return <div className="page-grid satellite-layout">
    <section className="panel wide satellite-stage">
      <PanelHeader icon={Satellite} title="Satellite Intelligence" meta={`${scenes.length} scenes · temporal evidence`} />
      <div className="satellite-visual" role="img" aria-label="Satellite intelligence evidence field">
        <div className="sat-grid" />
        <div className="sat-orbit orbit-a" /><div className="sat-orbit orbit-b" />
        <div className="sat-scan"><span>LIVE EVIDENCE FIELD</span><span>REMOTE SENSING / MULTISPECTRAL</span></div>
        <div className="thermal-hotspot hotspot-a" /><div className="thermal-hotspot hotspot-b" />
        <div className="sat-coordinates">AOI 21.60°N · 80.19°E<br/>DEMO PROXY · NOT OFFICIAL SURVEY</div>
      </div>
    </section>
    <section className="panel"><PanelHeader icon={Thermometer} title="Thermal Field" meta="Landsat" />
      <MetricLine label="Thermal scenes" value={String(thermal.length)} />
      <MetricLine label="Coverage" value={thermal.length ? `${Math.round((thermal[0].thermal_coverage ?? 1) * 100)}%` : "—"} />
      <MetricLine label="Interpretation" value="Surface-temperature proxy" />
      <p className="muted">Thermal imagery is contextual evidence. Validate anomalous signatures against geology and field observations.</p>
    </section>
    <section className="panel"><PanelHeader icon={Radio} title="Scene Quality" meta="ingestion" />
      <MetricLine label="Sentinel-2" value={String(scenes.filter(s => (s.source ?? "").toLowerCase().includes("sentinel")).length)} />
      <MetricLine label="Landsat" value={String(thermal.length)} />
      <MetricLine label="Best quality" value={scenes.length ? `${Math.round(Math.max(...scenes.map(s => s.quality ?? 0)) * 100)}%` : "—"} />
      <MetricLine label="Cloud-free signal" value={scenes.length ? `${Math.round(Math.max(...scenes.map(s => 1 - (s.cloud_cover ?? 0))) * 100)}%` : "—"} />
    </section>
    <section className="panel wide"><PanelHeader icon={Cloud} title="Evidence Timeline" meta="multisource" />
      <div className="scene-timeline">{scenes.slice(0, 12).map((scene, i) => <article className="scene-row" key={scene.id ?? i}><div><strong>{scene.source ?? "Satellite"}</strong><span>{scene.date ?? "Unknown acquisition"}</span></div><span>{scene.cloud_cover !== undefined ? `${Math.round(scene.cloud_cover * 100)}% cloud` : "cloud n/a"}</span><span>{scene.bands?.join(", ") ?? "multispectral"}</span></article>)}</div>
      {!scenes.length && <p className="muted">No live scene metadata is currently exposed by the API. The intelligence surface remains ready for Sentinel-2/Landsat ingestion.</p>}
    </section>
    <section className="panel wide"><PanelHeader icon={Activity} title="Scientific Boundary" meta="required" /><p className="muted">Satellite signals support prospectivity screening and operational context; they do not independently establish mineral reserves. DEMO PROXY labels remain active until validated field data are connected.</p></section>
  </div>;
}
