import { Activity, Cloud, Layers3, Radio, Satellite, Thermometer } from "lucide-react";
import maplibregl from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from "recharts";
import "maplibre-gl/dist/maplibre-gl.css";

type Scene = {
  id?: string;
  source?: string;
  date?: string;
  cloud_cover?: number | null;
  quality?: number | null;
  bands?: string[];
  thermal_coverage?: number | null;
  latitude?: number | null;
  longitude?: number | null;
  sample_id?: string;
  ndvi?: number | null;
  ndwi?: number | null;
  swir_ratio?: number | null;
  bare_soil_index?: number | null;
  land_surface_temperature?: number | null;
};

type Mode = "spectral" | "thermal" | "ndvi" | "swir" | "bare";

const Header = ({ icon: Icon, title, meta }: { icon: typeof Satellite; title: string; meta: string }) => (
  <div className="panel-header"><div className="panel-title"><Icon size={17}/><span>{title}</span></div><span className="panel-meta">{meta}</span></div>
);
const Metric = ({ label, value }: { label: string; value: string }) => <div className="metric-line"><span>{label}</span><strong>{value}</strong></div>;

const MODES: Array<{ id: Mode; label: string; description: string }> = [
  { id: "spectral", label: "Satellite / Spectral", description: "Multiband feature field" },
  { id: "thermal", label: "Thermal Map", description: "Synthetic LST proxy" },
  { id: "ndvi", label: "NDVI", description: "Vegetation signal" },
  { id: "swir", label: "SWIR Ratio", description: "Surface/mineral signal" },
  { id: "bare", label: "Bare Soil", description: "Exposure signal" },
];

function mean(values: Array<number | null | undefined>): number | null {
  const valid = values.filter((value): value is number => Number.isFinite(value));
  return valid.length ? valid.reduce((sum, value) => sum + value, 0) / valid.length : null;
}

function SatelliteEvidenceMap({ scenes, mode, onSelect }: { scenes: Scene[]; mode: Mode; onSelect: (scene: Scene) => void }) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<maplibregl.Map | null>(null);
  const [mapReady, setMapReady] = useState(false);

  const points = useMemo(() => scenes.filter(scene => Number.isFinite(scene.latitude) && Number.isFinite(scene.longitude)), [scenes]);

  useEffect(() => {
    if (!containerRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [80.1855, 21.5998],
      zoom: 9,
      minZoom: 5,
      maxZoom: 17,
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors",
            maxzoom: 19,
          },
        },
        layers: [{ id: "osm", type: "raster", source: "osm" }],
      },
      attributionControl: { compact: true },
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: true }), "top-right");
    map.on("load", () => setMapReady(true));
    map.on("error", event => console.warn("MANGAI satellite map error", event.error));
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !mapReady) return;
    const data = {
      type: "FeatureCollection" as const,
      features: points.map((scene, index) => ({
        type: "Feature" as const,
        properties: {
          index,
          id: scene.id ?? `feature-${index}`,
          sample_id: scene.sample_id ?? scene.id ?? `feature-${index}`,
          ndvi: scene.ndvi ?? 0,
          swir: scene.swir_ratio ?? 1,
          bare: scene.bare_soil_index ?? 0,
          thermal: scene.land_surface_temperature ?? 31,
        },
        geometry: { type: "Point" as const, coordinates: [scene.longitude as number, scene.latitude as number] },
      })),
    };

    const metric: "ndvi" | "swir" | "bare" | "thermal" = mode === "ndvi" ? "ndvi" : mode === "swir" ? "swir" : mode === "bare" ? "bare" : "thermal";
    const source = map.getSource("satellite-features") as maplibregl.GeoJSONSource | undefined;
    if (source) source.setData(data);
    else {
      map.addSource("satellite-features", { type: "geojson", data });
      map.addLayer({
        id: "satellite-feature-glow",
        type: "circle",
        source: "satellite-features",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["zoom"], 5, 4, 12, 8, 17, 13],
          "circle-color": [
            "interpolate", ["linear"], ["get", metric],
            metric === "thermal" ? 24 : metric === "swir" ? 0.7 : metric === "ndvi" ? -0.2 : -0.4, "#123f31",
            metric === "thermal" ? 31 : metric === "swir" ? 1.1 : metric === "ndvi" ? 0.3 : 0, "#28d7a0",
            metric === "thermal" ? 38 : metric === "swir" ? 1.6 : metric === "ndvi" ? 0.8 : 0.4, metric === "thermal" ? "#e7b75b" : "#f1f7f4",
          ],
          "circle-opacity": 0.76,
          "circle-stroke-color": "#06110c",
          "circle-stroke-width": 1,
        },
      });
      map.on("click", "satellite-feature-glow", event => {
        const feature = event.features?.[0];
        const index = feature?.properties?.index;
        if (typeof index === "number" && points[index]) onSelect(points[index]);
      });
      map.on("mouseenter", "satellite-feature-glow", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "satellite-feature-glow", () => { map.getCanvas().style.cursor = ""; });
    }
    if (map.getLayer("satellite-feature-glow")) {
      map.setPaintProperty("satellite-feature-glow", "circle-color", [
        "interpolate", ["linear"], ["get", metric],
        metric === "thermal" ? 24 : metric === "swir" ? 0.7 : metric === "ndvi" ? -0.2 : -0.4, "#123f31",
        metric === "thermal" ? 31 : metric === "swir" ? 1.1 : metric === "ndvi" ? 0.3 : 0, "#28d7a0",
        metric === "thermal" ? 38 : metric === "swir" ? 1.6 : metric === "ndvi" ? 0.8 : 0.4, metric === "thermal" ? "#e7b75b" : "#f1f7f4",
      ]);
    }
    if (points.length > 1) {
      const bounds = new maplibregl.LngLatBounds();
      points.forEach(point => bounds.extend([point.longitude as number, point.latitude as number]));
      map.fitBounds(bounds, { padding: 55, maxZoom: 12, duration: 500 });
    }
  }, [mapReady, mode, onSelect, points]);

  return <div className="sat-map-shell">
    <div ref={containerRef} className="sat-map" />
    <div className="sat-map-overlay"><span className="map-live-dot"/><b>{MODES.find(item => item.id === mode)?.label}</b><small>DEMO FEATURE FIELD</small></div>
    <div className="sat-map-legend"><span>low</span><i/><span>high</span><em>{MODES.find(item => item.id === mode)?.description}</em></div>
    {!points.length && <div className="sat-map-empty">No georeferenced satellite observations are exposed by the current API.</div>}
  </div>;
}

export function SatellitePage({ scenes = [] }: { scenes?: Scene[] }) {
  const [mode, setMode] = useState<Mode>("spectral");
  const [selected, setSelected] = useState<Scene | null>(null);
  const stats = useMemo(() => ({
    ndvi: mean(scenes.map(scene => scene.ndvi)),
    ndwi: mean(scenes.map(scene => scene.ndwi)),
    swir: mean(scenes.map(scene => scene.swir_ratio)),
    bare: mean(scenes.map(scene => scene.bare_soil_index)),
    lst: mean(scenes.map(scene => scene.land_surface_temperature)),
    quality: mean(scenes.map(scene => scene.quality)),
  }), [scenes]);
  const chartData = [
    { name: "NDVI", value: stats.ndvi ?? 0 },
    { name: "NDWI", value: stats.ndwi ?? 0 },
    { name: "SWIR", value: stats.swir ?? 0 },
    { name: "Bare soil", value: stats.bare ?? 0 },
  ];

  return <div className="page-grid satellite-layout">
    <section className="panel wide satellite-stage"><Header icon={Satellite} title="Satellite Intelligence" meta={`${scenes.length} feature observations · DEMO / LOCAL FILE`} />
      <div className="satellite-modebar"><div><b>Evidence layers</b><span>These are feature fields, not fabricated satellite imagery.</span></div><div className="satellite-mode-buttons">{MODES.map(item => <button key={item.id} className={mode === item.id ? "chip active" : "chip"} onClick={() => setMode(item.id)}>{item.label}</button>)}</div></div>
      <SatelliteEvidenceMap scenes={scenes} mode={mode} onSelect={setSelected}/>
    </section>

    <section className="panel"><Header icon={Radio} title="Acquisition" meta="feature dataset"/><Metric label="Observations" value={String(scenes.length)}/><Metric label="Georeferenced" value={String(scenes.filter(scene => Number.isFinite(scene.latitude) && Number.isFinite(scene.longitude)).length)}/><Metric label="Mean quality" value={stats.quality != null ? `${Math.round(stats.quality * 100)}%` : "n/a"}/><Metric label="Source" value="Synthetic feature set"/></section>

    <section className="panel"><Header icon={Thermometer} title="Thermal Evidence" meta={stats.lst != null ? "LST proxy available" : "not exposed"}/><Metric label="Mean LST" value={stats.lst != null ? `${stats.lst.toFixed(1)} °C` : "Not exposed"}/><Metric label="Thermal map" value={stats.lst != null ? "Available as proxy" : "Unavailable"}/><Metric label="Interpretation" value="Contextual only"/><p className="muted">Thermal values come from the demo feature dataset. They are not calibrated thermal imagery and must not be treated as field temperature measurements.</p></section>

    <section className="panel wide chart-panel"><Header icon={Activity} title="Spectral Feature Profile" meta="dataset-level signal"/><div className="sat-chart"><ResponsiveContainer width="100%" height={250}><BarChart data={chartData}><CartesianGrid stroke="rgba(145,174,162,.14)" strokeDasharray="3 3"/><XAxis dataKey="name" stroke="#9ab0a6"/><YAxis stroke="#9ab0a6"/><Tooltip/><Bar dataKey="value" fill="#28d7a0" radius={[4,4,0,0]}/></BarChart></ResponsiveContainer></div></section>

    <section className="panel"><Header icon={Layers3} title="Selected Evidence" meta={selected ? selected.sample_id ?? selected.id ?? "point" : "select a map point"}/>{selected ? <div className="detail-stack"><Metric label="Coordinates" value={`${Number(selected.latitude).toFixed(5)}, ${Number(selected.longitude).toFixed(5)}`}/><Metric label="NDVI" value={selected.ndvi != null ? selected.ndvi.toFixed(3) : "n/a"}/><Metric label="SWIR ratio" value={selected.swir_ratio != null ? selected.swir_ratio.toFixed(3) : "n/a"}/><Metric label="Bare soil" value={selected.bare_soil_index != null ? selected.bare_soil_index.toFixed(3) : "n/a"}/><Metric label="LST proxy" value={selected.land_surface_temperature != null ? `${selected.land_surface_temperature.toFixed(1)} °C` : "n/a"}/></div> : <p className="muted">Click a georeferenced observation on the map to inspect its spectral and thermal-proxy values.</p>}</section>

    <section className="panel wide"><Header icon={Cloud} title="Feature Evidence Records" meta="sampled observations · no fake scene cards"/><div className="data-table-wrap"><table className="sat-evidence-table"><thead><tr><th>Sample</th><th>Lat</th><th>Lon</th><th>NDVI</th><th>SWIR</th><th>Bare soil</th><th>LST</th><th>Quality</th></tr></thead><tbody>{scenes.slice(0, 10).map((scene, index) => <tr key={scene.id ?? index}><td className="mono">{scene.sample_id ?? scene.id ?? `feature-${index + 1}`}</td><td>{scene.latitude != null ? scene.latitude.toFixed(4) : "—"}</td><td>{scene.longitude != null ? scene.longitude.toFixed(4) : "—"}</td><td>{scene.ndvi != null ? scene.ndvi.toFixed(3) : "—"}</td><td>{scene.swir_ratio != null ? scene.swir_ratio.toFixed(3) : "—"}</td><td>{scene.bare_soil_index != null ? scene.bare_soil_index.toFixed(3) : "—"}</td><td>{scene.land_surface_temperature != null ? `${scene.land_surface_temperature.toFixed(1)}°C` : "—"}</td><td>{scene.quality != null ? `${Math.round(scene.quality * 100)}%` : "—"}</td></tr>)}</tbody></table></div></section>

    <section className="panel wide"><Header icon={Cloud} title="Scientific Boundary" meta="required"/><p className="muted">The current branch exposes local satellite feature records, not Sentinel-2/Landsat image tiles. The new map therefore visualizes georeferenced spectral and LST-proxy observations on a geographic basemap. Real satellite imagery and calibrated thermal rasters should be connected through the live satellite adapter before being presented as imagery evidence.</p></section>
  </div>;
}
