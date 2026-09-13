import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useRef, useState } from "react";
import { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence";

type Props = {
  cells: ProspectivityCell[];
  selectedCell: ProspectivityCell | null;
  onSelect: (cell: ProspectivityCell) => void;
  layer: LayerKey;
  boreholes: Array<{ borehole_id: string; latitude: number; longitude: number; lithology?: string }>;
};

export function ReserveMap({ cells, selectedCell, onSelect, layer, boreholes }: Props) {
  const containerRef = useRef<HTMLDivElement | null>(null);
  const mapRef = useRef<MapLibreMap | null>(null);
  const cellsRef = useRef(cells);
  const onSelectRef = useRef(onSelect);
  const [mapLoaded, setMapLoaded] = useState(false);

  useEffect(() => {
    cellsRef.current = cells;
    onSelectRef.current = onSelect;
  }, [cells, onSelect]);

  useEffect(() => {
    if (!containerRef.current || mapRef.current) return;

    const map = new maplibregl.Map({
      container: containerRef.current,
      center: [80.1855, 21.5998],
      zoom: 11,
      minZoom: 5,
      maxZoom: 18,
      attributionControl: { compact: true },
      style: {
        version: 8,
        sources: {
          osm: {
            type: "raster",
            tiles: ["https://tile.openstreetmap.org/{z}/{x}/{y}.png"],
            tileSize: 256,
            attribution: "© OpenStreetMap contributors"
          }
        },
        layers: [
          {
            id: "osm",
            type: "raster",
            source: "osm",
            paint: {
              "raster-opacity": 0.78,
              "raster-saturation": -0.75,
              "raster-contrast": 0.12,
              "raster-brightness-min": 0.05,
              "raster-brightness-max": 0.82
            }
          }
        ]
      }
    });

    map.addControl(new maplibregl.NavigationControl({ showCompass: false, visualizePitch: false }), "top-right");

    const handleLoad = () => {
      map.addSource("cells", { type: "geojson", data: emptyCollection() });
      map.addSource("boreholes", { type: "geojson", data: emptyCollection() });

      map.addLayer({
        id: "cells-halo",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "value"], 0, 5, 1, 14],
          "circle-color": "#24d6a0",
          "circle-opacity": 0.10,
          "circle-stroke-width": 0
        }
      });

      map.addLayer({
        id: "cells-heat",
        type: "circle",
        source: "cells",
        paint: {
          "circle-radius": ["interpolate", ["linear"], ["get", "value"], 0, 4, 1, 10],
          "circle-color": [
            "interpolate",
            ["linear"],
            ["get", "value"],
            0, "#31564a",
            0.35, "#2aa97f",
            0.68, "#65d5a6",
            1, "#f0bd59"
          ],
          "circle-opacity": 0.86,
          "circle-stroke-width": 1,
          "circle-stroke-color": "#082018"
        }
      });

      map.addLayer({
        id: "boreholes-layer",
        type: "circle",
        source: "boreholes",
        paint: {
          "circle-radius": 3.5,
          "circle-color": "#e9f4ef",
          "circle-opacity": 0.95,
          "circle-stroke-width": 1,
          "circle-stroke-color": "#10241c"
        }
      });

      map.on("click", "cells-heat", (event) => {
        const id = event.features?.[0]?.properties?.id as string | undefined;
        const match = cellsRef.current.find((cell) => cell.id === id);
        if (match) onSelectRef.current(match);
      });
      map.on("mouseenter", "cells-heat", () => { map.getCanvas().style.cursor = "pointer"; });
      map.on("mouseleave", "cells-heat", () => { map.getCanvas().style.cursor = ""; });
      setMapLoaded(true);
    };

    map.once("load", handleLoad);
    mapRef.current = map;

    return () => {
      map.remove();
      mapRef.current = null;
      setMapLoaded(false);
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    if (!mapLoaded || !map?.getSource("cells")) return;

    const values = cells.map((cell) => layerValue(cell, layer));
    const min = values.length ? Math.min(...values) : 0;
    const max = values.length ? Math.max(...values) : 1;
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
    (map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({
      type: "FeatureCollection",
      features: boreholes.map((hole) => ({
        type: "Feature",
        properties: { id: hole.borehole_id, lithology: hole.lithology ?? "" },
        geometry: { type: "Point", coordinates: [hole.longitude, hole.latitude] }
      }))
    });

    if (cells.length) {
      const bounds = new maplibregl.LngLatBounds();
      cells.forEach((cell) => bounds.extend([cell.longitude, cell.latitude]));
      map.fitBounds(bounds, { padding: { top: 70, right: 70, bottom: 70, left: 70 }, maxZoom: 14, duration: 450 });
    }
    if (selectedCell) map.easeTo({ center: [selectedCell.longitude, selectedCell.latitude], duration: 350 });
  }, [cells, layer, boreholes, selectedCell, mapLoaded]);

  return (
    <div className="map-surface maplibre-wrap">
      <div ref={containerRef} className="maplibre-canvas" />
      {!mapLoaded && <div className="map-loading"><span className="map-loading-dot" /> Initializing geospatial field…</div>}
      <div className="map-overlay-top">
        <span className="map-live-dot" /> LIVE SPATIAL VIEW
        <span>{cells.length} model targets</span>
      </div>
      <div className="map-legend">
        <span>Low {layer}</span><i /><span>High</span>
        {selectedCell && <em>{selectedCell.id} · {percent(selectedCell.probability)}</em>}
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
