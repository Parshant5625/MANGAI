import { Activity, Cloud, Radio, Satellite, Thermometer } from "lucide-react";
import { useMemo } from "react";

type Scene = { id?: string; source?: string; date?: string; cloud_cover?: number; quality?: number; bands?: string[]; thermal_coverage?: number };
const Header = ({ icon: Icon, title, meta }: { icon: typeof Satellite; title: string; meta: string }) => <div className="panel-header"><div className="panel-title"><Icon size={17}/><span>{title}</span></div><span className="panel-meta">{meta}</span></div>;
const Metric = ({ label, value }: { label: string; value: string }) => <div className="metric-line"><span>{label}</span><strong>{value}</strong></div>;

export function SatellitePage({ scenes = [] }: { scenes?: Scene[] }) {
  const stats = useMemo(() => {
    const qualities = scenes.map(s => s.quality ?? 0).filter(Number.isFinite);
    const clouds = scenes.map(s => s.cloud_cover ?? 0).filter(Number.isFinite);
    const thermal = scenes.filter(s => s.thermal_coverage != null);
    const sourceCounts = scenes.reduce<Record<string, number>>((acc, scene) => { const source = scene.source || "Satellite"; acc[source] = (acc[source] || 0) + 1; return acc; }, {});
    return { best: qualities.length ? Math.max(...qualities) : 0, avgCloud: clouds.length ? clouds.reduce((a,b)=>a+b,0)/clouds.length : 0, thermal, sourceCounts };
  }, [scenes]);
  return <div className="page-grid satellite-layout">
    <section className="panel wide satellite-stage"><Header icon={Satellite} title="Satellite Intelligence" meta={`${scenes.length} records · remote-sensing evidence`} /><div className="satellite-visual"><div className="sat-grid"/><div className="sat-scan"><span>REMOTE-SENSING EVIDENCE FIELD</span><span>DEMO / LOCAL-FILE DATA</span></div><div className="sat-orbit orbit-a"/><div className="sat-orbit orbit-b"/><div className="sat-signal-core"><div className="sat-core-ring"/><strong>{scenes.length}</strong><span>satellite records</span></div><div className="satellite-callout"><b>Evidence status</b><span>Backend satellite records are connected. Imagery tiles are not exposed by the current API, so this view intentionally shows metadata/feature evidence instead of fabricated imagery.</span></div><div className="sat-coordinates">AOI / DEMO DATASET<br/>Use field validation before geological conclusions.</div></div></section>
    <section className="panel"><Header icon={Radio} title="Acquisition" meta="connected"/><Metric label="Records" value={String(scenes.length)}/><Metric label="Distinct sources" value={String(Object.keys(stats.sourceCounts).length)}/><Metric label="Best quality" value={scenes.length ? `${Math.round(stats.best*100)}%` : "—"}/><Metric label="Mean cloud" value={scenes.length ? `${Math.round(stats.avgCloud*100)}%` : "—"}/></section>
    <section className="panel"><Header icon={Thermometer} title="Thermal Evidence" meta="context only"/><Metric label="Thermal records" value={String(stats.thermal.length)}/><Metric label="Coverage" value={stats.thermal.length ? `${Math.round((stats.thermal[0].thermal_coverage ?? 0)*100)}%` : "Not exposed"}/><Metric label="Use" value="Contextual signal"/><p className="muted">Thermal measurements must be checked against acquisition metadata, calibration and field observations.</p></section>
    <section className="panel wide"><Header icon={Activity} title="Feature Evidence" meta="available to MANGAI"/><div className="satellite-feature-grid">{scenes.slice(0, 12).map((scene, i) => <article className="sat-feature" key={scene.id ?? i}><div><b>{scene.source || "Satellite"}</b><span>{scene.date || "record date unavailable"}</span></div><strong>{scene.quality != null ? `${Math.round(scene.quality*100)}% quality` : "quality n/a"}</strong><small>{scene.bands?.length ? scene.bands.join(" · ") : "spectral feature record"}</small><div className="feature-bar"><i style={{width:`${Math.max(4,Math.min(100,(scene.quality ?? .5)*100))}%`}}/></div></article>)}</div>{!scenes.length && <p className="muted">No satellite records are currently returned by the backend.</p>}</section>
    <section className="panel wide"><Header icon={Cloud} title="Scientific Boundary" meta="required"/><p className="muted">The current backend endpoint exposes local satellite feature records and metadata. It does not expose Sentinel-2/Landsat image URLs or map tiles. MANGAI therefore does not fabricate an image layer. Connect the live imagery adapter when imagery rendering is required; until then, satellite evidence remains contextual and decision-support only.</p></section>
  </div>;
}
