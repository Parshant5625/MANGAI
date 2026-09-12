import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";
import "../reserve-intelligence.css";
import "../reserve-intelligence-details.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence" | "thermal";
type BaseMapKey = "satellite" | "terrain";

const SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRAIN_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

const HEATMAP_COLORS: [number, string][] = [
  [0, "#102c5c"],
  [0.2, "#155c8f"],
  [0.4, "#16a5a0"],
  [0.58, "#63c95c"],
  [0.72, "#f2dc55"],
  [0.86, "#f58d36"],
  [1, "#ef3d2f"]
];

export function ReserveMap({
  cells,
  selectedCell,
  onSelect,
  layer,
  boreholes
}: {
  cells: ProspectivityCell[];
  selectedCell: ProspectivityCell | null;
  onSelect: (cell: ProspectivityCell) => void;
  layer: "probability" | "grade" | "thickness" | "confidence";
  boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellsRef = useRef(cells);
  const onSelectRef = useRef(onSelect);
  const [baseMap, setBaseMap] = useState<BaseMapKey>("satellite");
  const [visualLayer, setVisualLayer] = useState<LayerKey>(layer);

  useEffect(() => {
    cellsRef.current = cells;
    onSelectRef.current = onSelect;
  }, [cells, onSelect]);

  useEffect(() => {
    if (visualLayer !== "thermal") setVisualLayer(layer);
  }, [layer]);

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
          terrain: { type: "raster", tiles: [TERRAIN_TILES], tileSize: 256, attribution: "© OpenStreetMap contributors" }
        },
        layers: [
          { id: "terrain-base", type: "raster", source: "terrain", paint: { "raster-opacity": 0 } },
          { id: "satellite-base", type: "raster", source: "satellite", paint: { "raster-opacity": 1, "raster-saturation": -0.15, "raster-contrast": 0.08, "raster-brightness-min": 0.02, "raster-brightness-max": 0.94 } }
        ]
      },
      center: [80.3, 21.4],
      zoom: 9,
      minZoom: 6,
      maxZoom: 16,
      attributionControl: false,
      dragRotate: false,
      pitchWithRotate: false
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("cells", { type: "geojson", data: emptyCollection() });
      map.addSource("boreholes", { type: "geojson", data: emptyCollection() });

      map.addLayer({
        id: "cells-heatmap",
        type: "heatmap",
        source: "cells",
        maxzoom: 16,
        paint: {
          "heatmap-weight": ["interpolate", ["linear"], ["get", "intensity"], 0, 0, 1, 1],
          "heatmap-intensity": ["interpolate", ["linear"], ["zoom"], 6, 0.75, 10, 1.15, 14, 1.8],
          "heatmap-radius": ["interpolate", ["linear"], ["zoom"], 6, 20, 9, 30, 12, 44, 16, 58],
          "heatmap-opacity": 0.88,
          "heatmap-color": [
            "interpolate",
            ["linear"],
            ["heatmap-density"],
            ...HEATMAP_COLORS.flat()
          ]
        }
      });

      map.addLayer({
        id: "selected-cell",
        type: "circle",
        source: "cells",
        filter: ["==", ["get", "id"], "__none__"],
        paint: {
          "circle-radius": 8,
          "circle-color": "rgba(255,255,255,0)",
          "circle-opacity": 0,
          "circle-stroke-width": 2.5,
          "circle-stroke-color": "#ffe58c",
          "circle-stroke-opacity": 1
        }
      });

      map.addLayer({
        id: "boreholes-layer",
        type: "circle",
        source: "boreholes",
        paint: {
          "circle-radius": 2.5,
          "circle-color": "#eafbf4",
          "circle-stroke-width": 1,
          "circle-stroke-color": "#071d16",
          "circle-opacity": 0.9
        }
      });

      map.on("click", "cells-heatmap", (event) => {
        const id = event.features?.[0]?.properties?.id as string | undefined;
        const match = cellsRef.current.find((cell) => cell.id === id);
        if (match) onSelectRef.current(match);
      });
      map.on("mouseenter", "cells-heatmap", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-heatmap", () => { map.getCanvas().style.cursor = ""; });
    });

    mapRef.current = map;
    return () => { map.remove(); mapRef.current = null; };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getLayer("satellite-base") || !map.getLayer("terrain-base")) return;
    map.setPaintProperty("satellite-base", "raster-opacity", baseMap === "satellite" ? 1 : 0);
    map.setPaintProperty("terrain-base", "raster-opacity", baseMap === "terrain" ? 0.96 : 0);
  }, [baseMap]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getSource("cells")) return;

    const rawValues = cells.map((cell) => layerValue(cell, visualLayer));
    const min = Math.min(...rawValues, 0);
    const max = Math.max(...rawValues, 1);
    const range = Math.max(max - min, 0.0001);

    const geojson = {
      type: "FeatureCollection" as const,
      features: cells.map((cell) => ({
        type: "Feature" as const,
        properties: {
          id: cell.id,
          probability: cell.probability,
          intensity: (layerValue(cell, visualLayer) - min) / range
        },
        geometry: { type: "Point" as const, coordinates: [cell.longitude, cell.latitude] }
      }))
    };

    (map.getSource("cells") as maplibregl.GeoJSONSource).setData(geojson);

    if (map.getSource("boreholes")) {
      (map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({
        type: "FeatureCollection",
        features: boreholes.map((hole) => ({
          type: "Feature",
          properties: { id: hole.borehole_id, lithology: hole.lithology ?? "" },
          geometry: { type: "Point", coordinates: [hole.longitude, hole.latitude] }
        }))
      });
    }

    if (map.getLayer("selected-cell")) {
      map.setFilter("selected-cell", ["==", ["get", "id"], selectedCell?.id ?? "__none__"]);
    }

    if (cells.length) {
      const bounds = new maplibregl.LngLatBounds();
      cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude]));
      map.fitBounds(bounds, { padding: 54, maxZoom: 12, duration: 650 });
    }
    if (selectedCell) map.easeTo({ center: [selectedCell.longitude, selectedCell.latitude], duration: 450, zoom: Math.max(map.getZoom(), 10) });
  }, [cells, visualLayer, boreholes, selectedCell]);

  const layerButtons: Array<{ key: LayerKey; label: string }> = [
    { key: "probability", label: "Prospectivity" },
    { key: "grade", label: "Grade" },
    { key: "thickness", label: "Thickness" },
    { key: "confidence", label: "Confidence" },
    { key: "thermal", label: "Thermal" }
  ];

  return (
    <div className={`map-surface maplibre-wrap reserve-map-command ${visualLayer === "thermal" ? "thermal-active" : ""}`}>
      <div ref={containerRef} className="maplibre-canvas" />
      <div className="reserve-map-vignette" />
      <div className="reserve-map-scanline" />

      <div className="reserve-map-hud">
        <div className="reserve-map-title">
          <span className="hud-live-dot" />
          <div>
            <strong>{visualLayer === "thermal" ? "THERMAL SURFACE" : "PROSPECTIVITY FIELD"}</strong>
            <small>{visualLayer === "thermal" ? "Landsat LST-ready thermal visualization · demo proxy" : `satellite + geology · ${visualLayer.toUpperCase()}`}</small>
          </div>
        </div>
        <div className="reserve-map-stats"><span><b>{cells.length}</b> cells</span><span><b>{mapStats.high}</b> high</span><span><b>{mapStats.veryHigh}</b> very high</span></div>
      </div>

      <div className="reserve-map-layerbar">
        {layerButtons.map((item) => (
          <button key={item.key} className={visualLayer === item.key ? "active" : ""} onClick={() => setVisualLayer(item.key)}>{item.label}</button>
        ))}
      </div>

      <div className="reserve-map-controls">
        <button className={baseMap === "satellite" ? "active" : ""} onClick={() => setBaseMap("satellite")}>Satellite</button>
        <button className={baseMap === "terrain" ? "active" : ""} onClick={() => setBaseMap("terrain")}>Map</button>
      </div>

      <div className="reserve-map-corner"><span>AVG P</span><strong>{percent(mapStats.avgProbability)}</strong><span>CONF.</span><strong>{percent(mapStats.avgConfidence)}</strong></div>

      <div className="map-legend reserve-map-legend">
        <div className="legend-caption">{visualLayer === "thermal" ? "thermal intensity · proxy" : `${visualLayer} intensity`}</div>
        <div className="legend-scale thermal-scale"><i /><i /><i /><i /><i /><i /></div>
        <div className="legend-labels"><span>cool / low</span><span>hot / high</span></div>
        {selectedCell && <em><span />{selectedCell.id} · {percent(selectedCell.probability)}</em>}
      </div>
      <div className="reserve-map-footnote"><span>● borehole context</span><span>◎ selected target</span><span>Heatmap interpolation · © Esri</span></div>
    </div>
  );
}

function layerValue(cell: ProspectivityCell, layer: LayerKey): number {
  if (layer === "grade") return cell.predicted_grade_pct;
  if (layer === "thickness") return cell.predicted_thickness_m;
  if (layer === "confidence") return cell.confidence;
  if (layer === "thermal") return thermalValue(cell);
  return cell.probability;
}

function thermalValue(cell: ProspectivityCell): number {
  const geology = cell.geology ?? {};
  const candidate = [geology.land_temperature_c, geology.lst_c, geology.thermal_c, geology.temperature_c].find((value) => typeof value === "number");
  if (typeof candidate === "number") return candidate;
  // Demo-only thermal proxy. It is deliberately labelled as a proxy until spatial Landsat LST is supplied.
  return 22 + cell.probability * 14 + (1 - cell.confidence) * 4;
}

function emptyCollection() {
  return { type: "FeatureCollection" as const, features: [] };
}
