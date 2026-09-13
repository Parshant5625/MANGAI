import maplibregl, { Map as MapLibreMap } from "maplibre-gl";
import { useEffect, useMemo, useRef, useState } from "react";
import type { ProspectivityCell } from "../types/api";
import { percent } from "../utils/format";
import "maplibre-gl/dist/maplibre-gl.css";

type LayerKey = "probability" | "grade" | "thickness" | "confidence" | "thermal";
type BaseMapKey = "satellite" | "terrain";
type ViewMode = "cells" | "thermal";
const SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}";
const TERRAIN_TILES = "https://tile.openstreetmap.org/{z}/{x}/{y}.png";
const HEATMAP_COLORS: [number,string][] = [[0,"#15345f"],[0.18,"#1676a0"],[0.38,"#15a69e"],[0.58,"#6bc95b"],[0.76,"#f3d84d"],[0.9,"#f48b35"],[1,"#ef4034"]];

export function ReserveMapConnected({cells,selectedCell,onSelect,layer,boreholes}:{cells:ProspectivityCell[];selectedCell:ProspectivityCell|null;onSelect:(cell:ProspectivityCell)=>void;layer:"probability"|"grade"|"thickness"|"confidence";boreholes:Array<{borehole_id:string;latitude:number;longitude:number;lithology?:string}>}) {
  const containerRef=useRef<HTMLDivElement|null>(null); const mapRef=useRef<MapLibreMap|null>(null); const cellsRef=useRef(cells); const selectRef=useRef(onSelect);
  const [ready,setReady]=useState(false); const [baseMap,setBaseMap]=useState<BaseMapKey>("satellite"); const [visualLayer,setVisualLayer]=useState<LayerKey>(layer); const [viewMode,setViewMode]=useState<ViewMode>("thermal");
  useEffect(()=>{cellsRef.current=cells;selectRef.current=onSelect},[cells,onSelect]); useEffect(()=>setVisualLayer(layer),[layer]);
  const stats=useMemo(()=>{if(!cells.length)return{high:0,veryHigh:0,p:0,c:0};return{high:cells.filter(c=>c.probability>=.7).length,veryHigh:cells.filter(c=>c.probability>=.85).length,p:cells.reduce((s,c)=>s+c.probability,0)/cells.length,c:cells.reduce((s,c)=>s+c.confidence,0)/cells.length}},[cells]);
  useEffect(()=>{
    if(!containerRef.current||mapRef.current)return;
    const map=new maplibregl.Map({container:containerRef.current,style:{version:8,sources:{satellite:{type:"raster",tiles:[SATELLITE_TILES],tileSize:256},terrain:{type:"raster",tiles:[TERRAIN_TILES],tileSize:256}},layers:[{id:"terrain-base",type:"raster",source:"terrain",paint:{"raster-opacity":0}},{id:"satellite-base",type:"raster",source:"satellite",paint:{"raster-opacity":1}}]},center:[80.3,21.4],zoom:9,minZoom:6,maxZoom:16,attributionControl:false,dragRotate:false,pitchWithRotate:false});
    map.addControl(new maplibregl.NavigationControl({showCompass:false,visualizePitch:false}),"top-right");
    map.on("load",()=>{
      map.addSource("cells",{type:"geojson",data:emptyCollection()}); map.addSource("boreholes",{type:"geojson",data:emptyCollection()});
      map.addLayer({id:"cells-heatmap",type:"heatmap",source:"cells",paint:{"heatmap-weight":["interpolate",["linear"],["get","intensity"],0,0,.22,.08,.55,.42,.8,.75,1,1],"heatmap-intensity":["interpolate",["linear"],["zoom"],6,.3,9,.48,12,.68,16,.82],"heatmap-radius":["interpolate",["linear"],["zoom"],6,14,9,21,12,28,16,35],"heatmap-opacity":.44,"heatmap-color":["interpolate",["linear"],["heatmap-density"],...HEATMAP_COLORS.flat()]}});
      map.addLayer({id:"cells-circles",type:"circle",source:"cells",paint:{"circle-radius":["interpolate",["linear"],["zoom"],6,5,9,7,12,10,16,15],"circle-color":["interpolate",["linear"],["get","intensity"],...HEATMAP_COLORS.flat()],"circle-opacity":.72,"circle-stroke-width":1,"circle-stroke-color":"#dffcff"}});
      map.addLayer({id:"selected-cell",type:"circle",source:"cells",filter:["==",["get","id"],"__none__"],paint:{"circle-radius":9,"circle-color":"rgba(255,255,255,0)","circle-opacity":0,"circle-stroke-width":2.5,"circle-stroke-color":"#ffe58c"}});
      map.addLayer({id:"boreholes-layer",type:"circle",source:"boreholes",paint:{"circle-radius":2.2,"circle-color":"#eafbf4","circle-stroke-width":.9,"circle-stroke-color":"#06150f"}});
      const click=(event:maplibregl.MapMouseEvent&{features?:maplibregl.MapGeoJSONFeature[]})=>{const id=event.features?.[0]?.properties?.id as string|undefined;const cell=cellsRef.current.find(item=>item.id===id);if(cell)selectRef.current(cell)};
      map.on("click","cells-circles",click); map.on("click","cells-heatmap",click); map.on("mouseenter","cells-circles",()=>map.getCanvas().style.cursor="pointer"); map.on("mouseleave","cells-circles",()=>map.getCanvas().style.cursor="");
      mapRef.current=map; setReady(true); requestAnimationFrame(()=>map.resize());
    });
    return()=>{map.remove();mapRef.current=null;setReady(false)};
  },[]);
  useEffect(()=>{const map=mapRef.current;if(!ready||!map)return;map.setPaintProperty("satellite-base","raster-opacity",baseMap==="satellite"?1:0);map.setPaintProperty("terrain-base","raster-opacity",baseMap==="terrain"?.98:0)},[baseMap,ready]);
  useEffect(()=>{const map=mapRef.current;if(!ready||!map)return;map.setLayoutProperty("cells-heatmap","visibility",viewMode==="thermal"?"visible":"none");map.setLayoutProperty("cells-circles","visibility",viewMode==="cells"?"visible":"none")},[viewMode,ready]);
  useEffect(()=>{const map=mapRef.current;if(!ready||!map||!map.getSource("cells"))return;const values=cells.map(c=>layerValue(c,visualLayer));const min=Math.min(...values,0);const max=Math.max(...values,1);const range=Math.max(max-min,.0001);(map.getSource("cells") as maplibregl.GeoJSONSource).setData({type:"FeatureCollection",features:cells.map(cell=>({type:"Feature",properties:{id:cell.id,intensity:Math.max(0,Math.min(1,(layerValue(cell,visualLayer)-min)/range))},geometry:{type:"Point",coordinates:[cell.longitude,cell.latitude]}}))});(map.getSource("boreholes") as maplibregl.GeoJSONSource).setData({type:"FeatureCollection",features:boreholes.map(h=>({type:"Feature",properties:{id:h.borehole_id},geometry:{type:"Point",coordinates:[h.longitude,h.latitude]}}))});if(map.getLayer("selected-cell"))map.setFilter("selected-cell",["==",["get","id"],selectedCell?.id??"__none__"]);if(cells.length){const bounds=new maplibregl.LngLatBounds();cells.forEach(c=>bounds.extend([c.longitude,c.latitude]));map.fitBounds(bounds,{padding:60,maxZoom:11,duration:350})}},[cells,boreholes,visualLayer,selectedCell,ready]);
  return <div className="map-surface mc-connected-map"><div ref={containerRef} className="maplibre-canvas"/><div className="mc-map-overlay"><b>{visualLayer.toUpperCase()} · {viewMode=== "thermal"?"HEATMAP":"CELLS"}</b><span>{cells.length} cells · {stats.high} high · {percent(stats.c)} confidence</span></div><div className="mc-map-switch"><button className={baseMap==="satellite"?"active":""} onClick={()=>setBaseMap("satellite")}>Satellite</button><button className={baseMap==="terrain"?"active":""} onClick={()=>setBaseMap("terrain")}>Map</button><button className={viewMode==="thermal"?"active":""} onClick={()=>setViewMode("thermal")}>Thermal</button><button className={viewMode==="cells"?"active":""} onClick={()=>setViewMode("cells")}>Cells</button></div><div className="mc-map-legend"><span>Low</span><i/><span>High</span></div></div>;
}
function layerValue(cell:ProspectivityCell,layer:LayerKey){if(layer==="grade")return cell.predicted_grade_pct;if(layer==="thickness")return cell.predicted_thickness_m;if(layer==="confidence")return cell.confidence;if(layer==="thermal"){const g=cell.geology??{};const t=[g.land_temperature_c,g.lst_c,g.thermal_c,g.temperature_c].find(v=>typeof v==="number");return typeof t==="number"?t:22+cell.probability*14+(1-cell.confidence)*4}return cell.probability}
function emptyCollection(){return{type:"FeatureCollection" as const,features:[]}}
