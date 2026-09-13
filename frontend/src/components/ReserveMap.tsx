import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence";
type Props = { cells: ProspectivityCell[]; selectedCell: ProspectivityCell | null; onSelect: (cell: ProspectivityCell) => void; layer: LayerKey; boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }> };

export function ReserveMap({ cells, selectedCell, onSelect, layer, boreholes }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellsRef = useRef(cells);
  const onSelectRef = useRef(onSelect);
  const [mapReady, setMapReady] = useState(false);
  const [mapError, setMapError] = useState<string | null>(null);

  useEffect(() => { cellsRef.current = cells; onSelectRef.current = onSelect; }, [cells, onSelect]);

  useEffect(() => {
    const container = containerRef.current;
    if (!container || mapRef.current) return;

    const map = new maplibregl.Map({
      container,
      center: [80.1855, 21.5998],
      zoom: 12,
      minZoom: 5,
      maxZoom: 18,
      attributionControl: { compact: true },
      style: "https://tiles.openfreemap.org/styles/liberty",
      cooperativeGestures: true,
    });
    mapRef.current = map;

    const handleLoad = () => {
      if (map.getSource("cells")) return;
      map.addSource("cells", { type: "geojson", data: emptyCollection() });
      map.addSource("boreholes", { type: "geojson", data: emptyCollection() });
      map.addLayer({ id: "cells-halo", type: "circle", source: "cells", paint: { "circle-radius": ["interpolate", ["linear"], ["get", "value"], 0, 6, 1, 18], "circle-color": "#28e0a5", "circle-opacity": 0.13 } });
      map.addLayer({ id: "cells-heat", type: "circle", source: "cells", paint: { "circle-radius": ["interpolate", ["linear"], ["get", "value"], 0, 4, 1, 12], "circle-color": ["interpolate", ["linear"], ["get", "value"], 0, "#164b3b", 0.3, "#1f9d76", 0.65, "#48d7a2", 1, "#e6b84f"], "circle-opacity": 0.88, "circle-stroke-width": 1, "circle-stroke-color": "#062018" } });
      map.addLayer({ id: "boreholes-layer", type: "circle", source: "boreholes", paint: { "circle-radius": 4, "circle-color": "#f4faf7", "circle-stroke-width": 1, "circle-stroke-color": "#0b2118" } });
      map.on("click", "cells-heat", (event) => { const id = event.features?.[0]?.properties?.id as string | undefined; const match = cellsRef.current.find((cell) => cell.id === id); if (match) onSelectRef.current(match); });
      map.on("mouseenter", "cells-heat", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-heat", () => { map.getCanvas().style.cursor = ""; });
      map.resize();
      setMapReady(true);
    };
    const handleError = (event: maplibregl.ErrorEvent) => { console.warn("MANGAI map error", event.error); setMapError(event.error?.message ?? "Basemap could not be loaded"); };
    map.once("load", handleLoad);
    map.on("error", handleError);

    const resize = () => map.resize();
    window.addEventListener("resize", resize);
    const observer = new ResizeObserver(resize);
    observer.observe(container);

    return () => { observer.disconnect(); window.removeEventListener("resize", resize); map.remove(); mapRef.current = null; setMapReady(false); };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapReady || !map?.getSource("cells")) return;
    const values = cells.map((cell) => layerValue(cell, layer));
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
    const source = map.getSource("cells") as maplibregl.GeoJSONSource;
    source.setData({ type: "FeatureCollection", features: cells.map((cell) => ({ type: "Feature", properties: { id: cell.id, value: (layerValue(cell, layer) - min) / Math.max(max - min, 0.0001) }, geometry: { type: "Point", coordinates: [cell.longitude, cell.latitude] } })) });
    (map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({ type: "FeatureCollection", features: boreholes.map((hole) => ({ type: "Feature", properties: { id: hole.borehole_id }, geometry: { type: "Point", coordinates: [hole.longitude, hole.latitude] } })) });
    if (cells.length) { const bounds = new maplibregl.LngLatBounds(); cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude])); map.fitBounds(bounds, { padding: 80, maxZoom: 14, duration: 350 }); }
    if (selectedCell) map.easeTo({ center: [selectedCell.longitude, selectedCell.latitude], duration: 300 });
  }, [cells, layer, boreholes, selectedCell, mapReady]);

  return <div className="map-surface maplibre-wrap">
    <div ref={containerRef} className="maplibre-canvas" />
    {!mapReady && !mapError && <div className="map-loading"><span className="map-loading-dot" /> Connecting to geospatial basemap…</div>}
    {mapError && <div className="map-error"><strong>Basemap unavailable</strong><span>{mapError}</span><small>Prospectivity data is still available; check internet access and reload.</small></div>}
    <div className="map-overlay-top"><span className="map-live-dot" /> GEOSPATIAL FIELD <span>{cells.length} targets</span></div>
    <div className="map-legend"><span>Low {layer}</span><i /><span>High</span>{selectedCell && <em>{selectedCell.id} · {percent(selectedCell.probability)}</em>}</div>
  </div>;
}

function layerValue(cell: ProspectivityCell, layer: LayerKey): number { if (layer === "grade") return cell.predicted_grade_pct; if (layer === "thickness") return cell.predicted_thickness_m; if (layer === "confidence") return cell.confidence; return cell.probability; }
function emptyCollection() { return { type: "FeatureCollection" as const, features: [] }; }
