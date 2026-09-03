import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useRef } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence";

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
  cellsRef.current = cells;
  onSelectRef.current = onSelect;

  useEffect(() => {
    if (!containerRef.current || mapRef.current) {
      return;
    }
    const map = new maplibregl.Map({
      container: containerRef.current,
      style: {
        version: 8,
        sources: {},
        layers: [{ id: "background", type: "background", paint: { "background-color": "#18231f" } }]
      },
      center: [80.3, 21.4],
      zoom: 9,
      attributionControl: false
    });
    map.addControl(new maplibregl.NavigationControl({ showCompass: false }), "top-right");
    map.on("load", () => {
      map.addSource("cells", { type: "geojson", data: emptyCollection() });
      map.addSource("boreholes", { type: "geojson", data: emptyCollection() });
      map.addLayer({
        id: "cells-heat",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "probability"], 0, 4, 1, 11],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "value"],
            0,
            "#7a5a2b",
            0.5,
            "#d6a24b",
            1,
            "#3dd6a0"
          ],
          "circle-opacity": 0.86,
          "circle-stroke-width": 0.6,
          "circle-stroke-color": "#0f1714"
        }
      });
      map.addLayer({
        id: "boreholes-layer",
        type: "circle",
        source: "boreholes",
        paint: {
          "circle-radius": 3.2,
          "circle-color": "#dbe7e1",
          "circle-stroke-width": 1,
          "circle-stroke-color": "#1c2924"
        }
      });
      map.on("click", "cells-heat", (event) => {
        const id = event.features?.[0]?.properties?.id as string | undefined;
        const match = cellsRef.current.find((cell) => cell.id === id);
        if (match) {
          onSelectRef.current(match);
        }
      });
      map.on("mouseenter", "cells-heat", () => {
        map.getCanvas().style.cursor = "pointer";
      });
      map.on("mouseleave", "cells-heat", () => {
        map.getCanvas().style.cursor = "";
      });
    });
    mapRef.current = map;
    return () => {
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!map?.getSource("cells")) {
      return;
    }
    const values = cells.map((cell) => layerValue(cell, layer));
    const min = Math.min(...values, 0);
    const max = Math.max(...values, 1);
    const geojson = {
      type: "FeatureCollection" as const,
      features: cells.map((cell) => ({
        type: "Feature" as const,
        properties: {
          id: cell.id,
          probability: cell.probability,
          value: (layerValue(cell, layer) - min) / Math.max(max - min, 0.0001)
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
    if (cells.length) {
      const bounds = new maplibregl.LngLatBounds();
      cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude]));
      map.fitBounds(bounds, { padding: 48, maxZoom: 11, duration: 600 });
    }
    if (selectedCell) {
      map.easeTo({ center: [selectedCell.longitude, selectedCell.latitude], duration: 400 });
    }
  }, [cells, layer, boreholes, selectedCell]);

  return (
    <div className="map-surface maplibre-wrap">
      <div ref={containerRef} className="maplibre-canvas" />
      <div className="map-legend">
        <span>Low {layer}</span>
        <i />
        <span>High</span>
        {selectedCell && <em>{selectedCell.id} {percent(selectedCell.probability)}</em>}
      </div>
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
