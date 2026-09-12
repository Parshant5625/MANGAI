import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";
import "../reserve-intelligence.css";
import "../reserve-intelligence-details.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence" | "thermal";
type BaseMapKey = "satellite" | "terrain";
type ViewMode = "cells" | "thermal";

const SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRAIN_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const HEATMAP_COLORS: [number, string][] = [[0, "#15345f"], [0.18, "#1676a0"], [0.38, "#15a69e"], [0.58, "#6bc95b"], [0.76, "#f3d84d"], [0.9, "#f48b35"], [1, "#ef4034"]];

interface ReserveMapProps {
  cells: ProspectivityCell[];
  selectedCell: ProspectivityCell | null;
  onSelect: (cell: ProspectivityCell) => void;
  layer: "probability" | "grade" | "thickness" | "confidence";
  boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>;
}

export function ReserveMap({ cells, selectedCell, onSelect, layer, boreholes }: ReserveMapProps) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellsRef = useRef(cells);
  const onSelectRef = useRef(onSelect);
  const [baseMap, setBaseMap] = useState<BaseMapKey>("satellite");
  const [visualLayer, setVisualLayer] = useState<LayerKey>(layer);
  const [viewMode, setViewMode] = useState<ViewMode>("thermal");

  useEffect(() => { cellsRef.current = cells; onSelectRef.current = onSelect; }, [cells, onSelect]);
  useEffect(() => { setVisualLayer(layer); }, [layer]);

  const mapStats = useMemo(() => {
    if (!cells.length) return { high: 0, veryHigh: 0, avgProbability: 0, avgConfidence: 0 };
    const high = cells.filter((cell) => cell.probability >= 0.7).length;
    const veryHigh = cells.filter((cell) => cell.probability >= 0.85).length;
    const avgProbability = cells.reduce((sum, cell) => sum + cell.probability, 0) / cells.length;
    const avgConfidence = cells.reduce((sum, cell) => sum + cell.confidence, 0) / cells.length;
    return { high, veryHigh, avgProbability, avgConfidence };
  }, [cells]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {
          satellite: { type: "raster", tiles: [SATELLITE_TILES], tileSize: 256, attribution: "© Esri" },
          terrain: { type: "raster", tiles: [TERRAIN_TILES], tileSize: 256, attribution: "© OpenStreetMap contributors" },
        },
        layers: [
          { id: "terrain-base", type: "raster", source: "terrain", paint: { "raster-opacity": 0.04, "raster-saturation": -0.05, "raster-contrast": 0.04 } },
          { id: "satellite-base", type: "raster", source: "satellite", paint: { "raster-opacity": 1, "raster-saturation": 0.08, "raster-contrast": 0.02, "raster-brightness-min": 0.04, "raster-brightness-max": 1 } },
        ],
      },
      center: [80.3, 21.4], zoom: 9, minZoom: 6, maxZoom: 16, attributionControl: false, dragRotate: false, pitchWithRotate: false,
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false, visualizePitch: false }), "top-right");
    map.on("load", () => {
      map.addSource("cells", { type: "geojson", data: emptyCollection() });
      map.addSource("boreholes", { type: "geojson", data: emptyCollection() });
      map.addLayer({ id: "cells-heatmap", type: "heatmap", source: "cells", maxzoom: 16, paint: {
        "heatmap-weight": ["interpolate", ["linear"], ["get", "intensity"], 0, 0, 0.22, 0.08, 0.55, 0.42, 0.8, 0.75, 1, 1],
        "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 6, 0.3, 9, 0.48, 12, 0.68, 16, 0.82],
        "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 6, 14, 9, 21, 12, 28, 16, 35],
        "heatmap-opacity": 0.44,
        "heatmap-color": ["interpolate", ["linear"], ["heatmap-density"], ...HEATMAP_COLORS.flat()],
      } });
      map.addLayer({ id: "cells-circles", type: "circle", source: "cells", paint: {
        "circle-radius": ["interpolate", ["linear"], ["zoom"], 6, 5, 9, 7, 12, 10, 16, 15],
        "circle-color": ["interpolate", ["linear"], ["get", "intensity"], ...HEATMAP_COLORS.flat()],
        "circle-opacity": 0.72, "circle-stroke-width": 1, "circle-stroke-color": "rgba(238,255,248,0.62)", "circle-stroke-opacity": 0.7,
      } });
      map.addLayer({ id: "selected-cell", type: "circle", source: "cells", filter: ["==", ["get", "id"], "__none__"], paint: { "circle-radius": 9, "circle-color": "rgba(255,255,255,0)", "circle-opacity": 0, "circle-stroke-width": 2.5, "circle-stroke-color": "#ffe58c", "circle-stroke-opacity": 1 } });
      map.addLayer({ id: "boreholes-layer", type: "circle", source: "boreholes", paint: { "circle-radius": 2.2, "circle-color": "#eafbf4", "circle-stroke-width": 0.9, "circle-stroke-color": "#06150f", "circle-opacity": 0.9 } });
      const selectFromEvent = (event: maplibregl.MapMouseEvent & { features?: maplibregl.MapGeoJSONFeature[] }) => {
        const id = event.features?.[0]?.properties?.id as string | undefined;
        const match = cellsRef.current.find((cell) => cell.id === id);
        if (match) onSelectRef.current(match);
      };
      map.on("click", "cells-heatmap", selectFromEvent);
      map.on("click", "cells-circles", selectFromEvent);
      map.on("mouseenter", "cells-heatmap", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-heatmap", () => { map.getCanvas().style.cursor = ""; });
      map.on("mouseenter", "cells-circles", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-circles", () => { map.getCanvas().style.cursor = ""; });
    });
    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("satellite-base") || !map.getLayer("terrain-base")) return;
    map.setPaintProperty("satellite-base", "raster-opacity", baseMap === "satellite" ? 1 : 0);
    map.setPaintProperty("terrain-base", "raster-opacity", baseMap === "terrain" ? 0.98 : 0);
  }, [baseMap]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("cells-heatmap") || !map.getLayer("cells-circles")) return;
    map.setLayoutProperty("cells-heatmap", "visibility", viewMode === "thermal" ? "visible" : "none");
    map.setLayoutProperty("cells-circles", "visibility", viewMode === "cells" ? "visible" : "none");
  }, [viewMode]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getSource("cells")) return;
    const rawValues = cells.map((cell) => layerValue(cell, visualLayer));
    const min = Math.min(...rawValues, 0);
    const max = Math.max(...rawValues, 1);
    const range = Math.max(max - min, 0.0001);
    (map.getSource("cells") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features: cells.map((cell) => ({ type: "Feature", properties: { id: cell.id, probability: cell.probability, intensity: Math.max(0, Math.min(1, (layerValue(cell, visualLayer) - min) / range)) }, geometry: { type: "Point", coordinates: [cell.longitude, cell.latitude] } })) });
    if (map.getSource("boreholes")) (map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features: boreholes.map((hole) => ({ type: "Feature", properties: { id: hole.borehole_id, lithology: hole.lithology ?? "" }, geometry: { type: "Point", coordinates: [hole.longitude, hole.latitude] } })) });
    if (map.getLayer("selected-cell")) map.setFilter("selected-cell", ["==", ["get", "id"], selectedCell?.id ?? "__none__"]);
    if (cells.length) { const bounds = new maplibregl.LngLatBounds(); cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude])); map.fitBounds(bounds, { padding: 70, maxZoom: 11, duration: 500 }); }
  }, [cells, visualLayer, boreholes, selectedCell]);

  const layerButtons: Array<{ key: LayerKey; label: string }> = [
    { key: "probability", label: "Prospectivity" }, { key: "grade", label: "Grade" }, { key: "thickness", label: "Thickness" }, { key: "confidence", label: "Confidence" }, { key: "thermal", label: "Thermal" },
  ];

  return (
    <div className={`map-surface maplibre-wrap reserve-map-command ${visualLayer === "thermal" ? "thermal-active" : ""} ${viewMode === "cells" ? "cells-view" : "thermal-view"}`}>
      <div ref={containerRef} className="maplibre-canvas" />
      <div className="reserve-map-vignette" />
      <div className="reserve-map-scanline" />
      <div className="reserve-map-hud"><div className="reserve-map-title"><span className="hud-live-dot" /><div><strong>{visualLayer === "thermal" ? "THERMAL SURFACE" : "PROSPECTIVITY FIELD"}</strong><small>{visualLayer === "thermal" ? "Landsat LST-ready thermal visualization · demo proxy" : `satellite + geology · ${visualLayer.toUpperCase()}`}</small></div></div><div className="reserve-map-stats"><span><b>{cells.length}</b> cells</span><span><b>{mapStats.high}</b> high</span><span><b>{mapStats.veryHigh}</b> very high</span></div></div>
      <div className="reserve-map-layerbar">{layerButtons.map((item) => <button type="button" key={item.key} className={visualLayer === item.key ? "active" : ""} onClick={() => { setVisualLayer(item.key); if (item.key === "thermal") setViewMode("thermal"); }}>{item.label}</button>)}</div>
      <div className="reserve-map-viewbar" role="group" aria-label="Cell visualization"><span>VIEW</span><button type="button" className={viewMode === "cells" ? "active" : ""} onClick={() => setViewMode("cells")}>Circular Cells</button><button type="button" className={viewMode === "thermal" ? "active" : ""} onClick={() => setViewMode("thermal")}>Thermal Heat</button></div>
      <div className="reserve-map-controls" role="group" aria-label="Base map"><button type="button" className={baseMap === "satellite" ? "active" : ""} onClick={() => setBaseMap("satellite")}>Satellite</button><button type="button" className={baseMap === "terrain" ? "active" : ""} onClick={() => setBaseMap("terrain")}>Map</button></div>
      <div className="reserve-map-corner"><span>AVG P</span><strong>{percent(mapStats.avgProbability)}</strong><span>CONF.</span><strong>{percent(mapStats.avgConfidence)}</strong></div>
      <div className="map-legend reserve-map-legend"><div className="legend-caption">{viewMode === "thermal" ? "thermal intensity · proxy" : `${visualLayer} intensity · cell view`}</div><div className="legend-scale thermal-scale"><i /><i /><i /><i /><i /><i /></div><div className="legend-labels"><span>cool / low</span><span>hot / high</span></div>{selectedCell && <em><span />{selectedCell.id} · {percent(selectedCell.probability)}</em>}</div>
      <div className="reserve-map-footnote"><span>● borehole context</span><span>◎ selected target</span><span>Heatmap interpolation · © Esri</span></div>
    </div>
  );
}

function layerValue(cell: ProspectivityCell, layer: LayerKey): number { if (layer === "grade") return cell.predicted_grade_pct; if (layer === "thickness") return cell.predicted_thickness_m; if (layer === "confidence") return cell.confidence; if (layer === "thermal") return thermalValue(cell); return cell.probability; }
function thermalValue(cell: ProspectivityCell): number { const geology = cell.geology ?? {}; const candidate = [geology.land_temperature_c, geology.lst_c, geology.thermal_c, geology.temperature_c].find((value) => typeof value === "number"); if (typeof candidate === "number") return candidate; return 22 + cell.probability * 14 + (1 - cell.confidence) * 4; }
function emptyCollection() { return { type: "FeatureCollection" as const, features: [] }; }
