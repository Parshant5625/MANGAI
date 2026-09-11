import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";
import "../reserve-intelligence.css";
import "../reserve-intelligence-details.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence";
type BaseMapKey = "satellite" | "terrain";

const SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRAIN_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";

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
  layer: LayerKey;
  boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>;
}) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellsRef = useRef(cells);
  const onSelectRef = useRef(onSelect);
  const [baseMap, setBaseMap] = useState<BaseMapKey>("satellite");

  useEffect(() => {
    cellsRef.current = cells;
    onSelectRef.current = onSelect;
  }, [cells, onSelect]);

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
          { id: "satellite-base", type: "raster", source: "satellite", paint: { "raster-opacity": 1, "raster-saturation": -0.05, "raster-contrast": 0.04, "raster-brightness-min": 0.08, "raster-brightness-max": 1 } }
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

      map.addLayer({ id: "cells-glow", type: "circle", source: "cells", paint: {
        "circle-radius": ["interpolate", ["linear"], ["get", "probability"], 0, 7, 1, 18],
        "circle-color": "#6be0b2",
        "circle-opacity": ["interpolate", ["linear"], ["get", "probability"], 0, 0, 0.65, 0.10, 1, 0.28],
        "circle-blur": 1
      }});

      map.addLayer({ id: "cells-heat", type: "circle", source: "cells", paint: {
        "circle-radius": ["interpolate", ["linear"], ["get", "value"], 0, 4, 0.5, 7, 1, 11],
        "circle-color": ["interpolate", ["linear"], ["get", "value"], 0, "#16443a", 0.28, "#268264", 0.55, "#d6a24b", 0.78, "#ef9d37", 1, "#ff503f"],
        "circle-opacity": 0.92,
        "circle-stroke-width": ["interpolate", ["linear"], ["get", "probability"], 0, 0.3, 0.7, 1, 1, 1.5],
        "circle-stroke-color": "#092119"
      }});

      map.addLayer({ id: "selected-cell", type: "circle", source: "cells", filter: ["==", ["get", "id"], "__none__"], paint: {
        "circle-radius": 17,
        "circle-color": "rgba(0,0,0,0)",
        "circle-opacity": 0,
        "circle-stroke-width": 2.5,
        "circle-stroke-color": "#ffe08b",
        "circle-stroke-opacity": 1
      }});

      map.addLayer({ id: "boreholes-layer", type: "circle", source: "boreholes", paint: {
        "circle-radius": 3.5,
        "circle-color": "#f4fff9",
        "circle-stroke-width": 1,
        "circle-stroke-color": "#12382b",
        "circle-opacity": 0.95
      }});

      map.on("click", "cells-heat", (event) => {
        const id = event.features?.[0]?.properties?.id as string | undefined;
        const match = cellsRef.current.find((cell) => cell.id === id);
        if (match) onSelectRef.current(match);
      });
      map.on("mouseenter", "cells-heat", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-heat", () => { map.getCanvas().style.cursor = ""; });
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
    const values = cells.map((cell) => layerValue(cell, layer));
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 1);
    const geojson = {
      type: "FeatureCollection" as const,
      features: cells.map((cell) => ({
        type: "Feature" as const,
        properties: { id: cell.id, probability: cell.probability, value: (layerValue(cell, layer) - min) / Math.max(max - min, 0.0001) },
        geometry: { type: "Point" as const, coordinates: [cell.longitude, cell.latitude] }
      }))
    };
    (map.getSource("cells") as maplibregl.GeoJSONSource).setData(geojson);
    if (map.getSource("boreholes")) {
      (map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({
        type: "FeatureCollection",
        features: boreholes.map((hole) => ({ type: "Feature", properties: { id: hole.borehole_id, lithology: hole.lithology ?? "" }, geometry: { type: "Point", coordinates: [hole.longitude, hole.latitude] } }))
      });
    }
    if (map.getLayer("selected-cell")) map.setFilter("selected-cell", ["==", ["get", "id"], selectedCell?.id ?? "__none__"]);
    if (cells.length) {
      const bounds = new maplibregl.LngLatBounds();
      cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude]));
      map.fitBounds(bounds, { padding: 54, maxZoom: 12, duration: 650 });
    }
    if (selectedCell) map.easeTo({ center: [selectedCell.longitude, selectedCell.latitude], duration: 450, zoom: Math.max(map.getZoom(), 10) });
  }, [cells, layer, boreholes, selectedCell]);

  return (
    <div className="map-surface maplibre-wrap reserve-map-command">
      <div ref={containerRef} className="maplibre-canvas" />
      <div className="reserve-map-scanline" />
      <div className="reserve-map-hud">
        <div className="reserve-map-title"><span className="hud-live-dot" /><div><strong>PROSPECTIVITY FIELD</strong><small>satellite + geology · {layer.toUpperCase()}</small></div></div>
        <div className="reserve-map-stats"><span><b>{cells.length}</b> cells</span><span><b>{mapStats.high}</b> high</span><span><b>{mapStats.veryHigh}</b> very high</span></div>
      </div>
      <div className="reserve-map-corner"><span>AVG P</span><strong>{percent(mapStats.avgProbability)}</strong><span>CONF.</span><strong>{percent(mapStats.avgConfidence)}</strong></div>
      <div className="reserve-map-controls"><button className={baseMap === "satellite" ? "active" : ""} onClick={() => setBaseMap("satellite")}>Satellite</button><button className={baseMap === "terrain" ? "active" : ""} onClick={() => setBaseMap("terrain")}>Map</button></div>
      <div className="map-legend reserve-map-legend"><div className="legend-caption">{layer} intensity</div><div className="legend-scale"><i /><i /><i /><i /><i /></div><div className="legend-labels"><span>low</span><span>high</span></div>{selectedCell && <em><span />{selectedCell.id} · {percent(selectedCell.probability)}</em>}</div>
      <div className="reserve-map-footnote"><span>● borehole context</span><span>◎ selected target</span><span>© Esri · © OpenStreetMap</span></div>
    </div>
  );
}

function layerValue(cell: ProspectivityCell, layer: LayerKey): number {
  if (layer === "grade") return cell.predicted_grade_pct;
  if (layer === "thickness") return cell.predicted_thickness_m;
  if (layer === "confidence") return cell.confidence;
  return cell.probability;
}

function emptyCollection() {
  return { type: "FeatureCollection" as const, features: [] };
}
