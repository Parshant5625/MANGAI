import { useMemo, useState } from "react";
import {
  Activity, AlertTriangle, BarChart3, Bot, CheckCircle2, CircleHelp, Database,
  Gauge, Layers3, Map as MapIcon, Menu, MessageSquare, Settings, ShieldCheck,
  Satellite, Target, Truck, Wrench, X
} from "lucide-react";
import { useApi } from "../hooks/useApi";
import type {
  DataQualityResponse, EquipmentResponse, OverviewResponse,
  ProductionForecastResponse, RecommendationResponse, ProspectivityCell
} from "../types/api";
import { compactNumber, number, percent } from "../utils/format";

type Nav = "overview" | "reserve" | "production" | "models" | "operations" | "assistant";

export function ReferenceDashboard() {
  const [nav, setNav] = useState<Nav>("overview");
  const [range, setRange] = useState("Live");
  const [assistantOpen, setAssistantOpen] = useState(true);
  const overview = useApi<OverviewResponse>("/api/v1/overview");
  const production = useApi<ProductionForecastResponse>("/api/v1/production/forecast?horizon=7");
  const equipment = useApi<EquipmentResponse>("/api/v1/equipment");
  const recommendations = useApi<RecommendationResponse>("/api/v1/recommendations");
  const quality = useApi<DataQualityResponse>("/api/v1/data-quality");
  const cells = useApi<{ cells: ProspectivityCell[] }>("/api/v1/reserves/prospectivity?limit=180&min_probability=0.45");

  const data = overview.data;
  const prod = production.data;
  const eq = equipment.data;
  const recs = recommendations.data;
  const health = quality.data;
  const cellData = cells.data?.cells ?? [];
  const highCells = cellData.filter(c => c.probability >= .7);
  const avgArea = data?.high_prospectivity_area_ha ?? 2847;
  const resource = data?.resource_potential_tonnage ?? 1240000000;
  const annual = prod?.forecast_mt ?? data?.next_7_day_production_mt ?? 12.8;
  const risk = data?.shortfall_probability ?? prod?.shortfall_probability ?? .13;
  const fleet = eq?.fleet_utilization ?? .78;
  const confidence = health?.overall_score ?? .87;

  const navClick = (next: Nav) => {
    setNav(next);
    if (next === "assistant") setAssistantOpen(true);
  };

  if (nav !== "overview") {
    return <div className="rd-page-placeholder"><button className="rd-back" onClick={() => setNav("overview")}>← Executive Overview</button><h1>{nav === "reserve" ? "Reserve Intelligence" : nav === "production" ? "Production Intelligence" : nav === "models" ? "Model Monitoring" : nav === "operations" ? "Operations Intelligence" : "MANGAI AI Assistant"}</h1><p>The full backend-connected module remains available in the original application. This reference shell keeps the executive dashboard as the primary command view.</p></div>;
  }

  return (
    <div className="reference-app">
      <header className="rd-header">
        <div className="rd-brand"><div className="rd-mountain">▲</div><div><strong>MANGAI</strong><span>MINING INTELLIGENCE PLATFORM</span></div></div>
        <div className="rd-header-status"><span className="rd-online-dot"/> <b>System Online</b><small>All systems operational</small></div>
        <div className="rd-header-block"><span>Apr 26, 2025</span><b>14:32 <small>AEST</small></b></div>
        <div className="rd-header-block rd-weather">☀️ <span>18°C<br/><small>Clear</small></span></div>
        <div className="rd-user"><div className="rd-avatar">●</div><span>Operations Team<br/><small>Administrator</small></span><span>⌄</span></div>
      </header>

      <aside className="rd-sidebar">
        <button className={`rd-nav ${nav === "overview" ? "active" : ""}`} onClick={() => navClick("overview")}><Gauge/>Overview</button>
        <button className="rd-nav" onClick={() => navClick("reserve")}><Target/>Reserve Intelligence</button>
        <button className="rd-nav" onClick={() => navClick("production")}><BarChart3/>Production Intelligence</button>
        <button className="rd-nav" onClick={() => navClick("models")}><Database/>Model Monitoring</button>
        <button className="rd-nav" onClick={() => navClick("operations")}><ShieldCheck/>Operations Intelligence</button>
        <button className="rd-nav" onClick={() => navClick("assistant")}><Bot/>AI Assistant</button>
        <div className="rd-nav-divider"/><span className="rd-section-label">TOOLS</span>
        <button className="rd-nav"><MapIcon/>Map</button><button className="rd-nav"><MessageSquare/>Reports</button>
        <div className="rd-nav-divider"/><span className="rd-section-label">SYSTEM</span>
        <button className="rd-nav"><Settings/>Settings</button><button className="rd-nav"><CircleHelp/>Help</button>
        <div className="rd-side-brand"><strong>MANGAI<span>▲</span></strong><p>Smarter Decisions.<br/>Greater Value.</p><div className="rd-side-mountains">╱╲╱╲╱╲</div></div>
      </aside>

      <main className="rd-content">
        <section className="rd-title-row"><div><h1>Executive Overview</h1><p>Real-time mining intelligence across reserves, production and operations</p></div><div className="rd-range">{["Live","24h","7d","30d","Custom"].map(r => <button key={r} className={range === r ? "active" : ""} onClick={() => setRange(r)}>{r}</button>)}</div></section>

        <section className="rd-kpis">
          <Kpi icon={<Layers3/>} title="Total Resource (JORC)" value={`${(resource/1e9).toFixed(2)} Bt`} change="▲ 2.4%" sub="vs. last update"/>
          <Kpi icon={<Truck/>} title="Est. Annual Production" value={`${number(annual,1)} Mt`} change="▲ 5.2%" sub="vs. last quarter"/>
          <Kpi icon={<Activity/>} title="Reserve Life" value="42.6 years" change="▲ 3.1%" sub="vs. last update"/>
          <Kpi icon={<Bot/>} title="AI Confidence" value={percent(confidence)} change="▲ 6.3%" sub="vs. last update"/>
        </section>

        <section className="rd-main-grid">
          <ReserveVisual cells={cellData} highCount={highCells.length} area={avgArea}/>
          <Insights recommendations={recs} risk={risk}/>
        </section>

        <section className="rd-bottom-grid">
          <ProductionCard value={annual}/>
          <ReserveCard resource={resource}/>
          <ModelCard confidence={confidence}/>
          <OperationsCard utilization={fleet}/>
        </section>
      </main>

      {assistantOpen && <aside className="rd-assistant-panel">
        <div className="rd-assistant-head"><div className="rd-ai-logo"><Bot/></div><div><h2>MANGAI AI Assistant</h2><span className="rd-online-pill">● Online</span></div><button onClick={() => setAssistantOpen(false)}><X/></button></div>
        <div className="rd-ai-conversation"><div className="rd-ai-question"><Bot/> <span>What would you like to know?</span></div>
          {[
            "What's the best expansion option?","Show me production forecast risks","Any anomalies in the models?","Summarise today's operations"
          ].map(q => <button className="rd-suggestion" key={q}>{q}</button>)}
        </div>
        <div className="rd-ai-input"><span>Ask about reserves, production,<br/>operations...</span><button><MessageSquare/></button></div>
        <div className="rd-activity"><h3>Recent Activity</h3>
          <ActivityRow icon={<Database/>} title="Model run completed" detail="Production Forecast v2.4" time="12m ago"/>
          <ActivityRow icon={<Satellite/>} title="New satellite imagery available" detail="Area: Central Reserve" time="34m ago"/>
          <ActivityRow icon={<Target/>} title="Drilling program updated" detail="7 new targets added" time="1h ago"/>
          <ActivityRow icon={<Truck/>} title="Fleet utilization improved" detail="+12% vs. last week" time="2h ago"/>
        </div>
      </aside>}
      {!assistantOpen && <button className="rd-assistant-fab" onClick={() => setAssistantOpen(true)}><Bot/></button>}

      <footer className="rd-footer"><span>MANGAI <b>|</b> Industrial Intelligence for a Smarter Tomorrow</span><span><i/> Backend Online &nbsp; | &nbsp; Frontend Connected &nbsp; | &nbsp; v1.0.0</span></footer>
    </div>
  );
}

function Kpi({icon,title,value,change,sub}:{icon:React.ReactNode;title:string;value:string;change:string;sub:string}) { return <div className="rd-kpi"><div className="rd-kpi-icon">{icon}</div><div className="rd-kpi-title">{title}</div><strong>{value}</strong><span className="rd-kpi-change">{change}</span><small>{sub}</small></div>; }

function ReserveVisual({cells,highCount,area}:{cells:ProspectivityCell[];highCount:number;area:number}) {
  const points = useMemo(() => cells.slice(0,80).map((c,i) => ({...c,x:8+(i*37)%84,y:12+(i*61)%74})), [cells]);
  return <section className="rd-card rd-reserve"><div className="rd-panel-title"><h2><MapIcon/> Reserve Map <span>● Live Data</span></h2><a>View details</a></div><div className="rd-map-visual">
    <div className="rd-terrain t1"/><div className="rd-terrain t2"/><div className="rd-terrain t3"/><div className="rd-terrain t4"/>
    <div className="rd-route r1"/><div className="rd-route r2"/><div className="rd-route r3"/>
    {points.map((p,i)=><button key={p.id ?? i} className={`rd-map-point ${p.probability>.75?'hot':''}`} style={{left:`${p.x}%`,top:`${p.y}%`}} title={`${p.id}: ${percent(p.probability)}`} />)}
    <span className="rd-map-label l1">North Ridge</span><span className="rd-map-label l2">Central</span><span className="rd-map-label l3">East Wing</span><span className="rd-map-label l4">South Basin</span>
    <div className="rd-map-stat"><div><small>Active Targets</small><b>{Math.max(7,Math.min(99,highCount))}</b><em>3 high priority</em></div><div><small>Total Area</small><b>{number(area,0)} ha</b><em>▲ 4.2%</em></div></div>
    <div className="rd-map-legend"><b>Resource Area</b><b>Reserve Area</b><b>High Priority</b><b>◆ Drill Target</b><b>━ Haul Road</b><b>⌁ Pit Boundary</b></div>
    <div className="rd-scale">0 &nbsp; 1 &nbsp; 2 &nbsp;&nbsp;&nbsp; 5 km</div><div className="rd-north">N</div>
  </div></section>;
}

function Insights({recommendations,risk}:{recommendations:RecommendationResponse|undefined;risk:number}) { const items = recommendations?.recommendations?.slice(0,3) ?? []; return <section className="rd-card rd-insights"><div className="rd-panel-title"><h2>✦ Key Insights</h2><a>View all</a></div><Insight title="High Value Target Identified" text={items[0]?.title ?? "East Wing shows 18% higher grade potential"} meta="Confidence: 87%" time="2h ago"/><Insight title="Production Forecast" text={`Q2 forecast increased to ${(12.8).toFixed(1)} Mt (+5.2%)`} meta="Based on latest model run" time="4h ago"/><Insight title="Risk Alert" text={risk>.65?"South Basin: elevated production risk":"South Basin: elevated geotechnical risk"} meta="Monitor closely" time="6h ago" risk/><Insight title="Operational Efficiency" text="Fleet utilization up 12% this week" meta="On track for target" time="8h ago"/></section>; }
function Insight({title,text,meta,time,risk=false}:{title:string;text:string;meta:string;time:string;risk?:boolean}) { return <div className={`rd-insight ${risk?'risk':''}`}><div className="rd-insight-icon">{risk?<AlertTriangle/>:<CheckCircle2/>}</div><div><h3>{title}</h3><p>{text}</p><small>{meta}</small></div><time>{time}</time></div>; }

function ProductionCard({value}:{value:number}) { return <section className="rd-card rd-mini"><div className="rd-panel-title"><h2><BarChart3/> Production Performance</h2><a>View details</a></div><div className="rd-mini-value">{number(value,1)} Mt <span>▲ 5.2%</span></div><small>vs. last quarter</small><div className="rd-svg-chart"><svg viewBox="0 0 300 100" preserveAspectRatio="none"><polyline points="0,70 45,77 90,62 135,57 180,43 225,35 270,22 300,18"/><polyline className="forecast" points="180,45 225,35 270,22 300,18"/></svg></div><div className="rd-months">Jan &nbsp;&nbsp; Feb &nbsp;&nbsp; Mar &nbsp;&nbsp; Apr &nbsp;&nbsp; May &nbsp;&nbsp; Jun</div></section>; }
function ReserveCard({resource}:{resource:number}) { return <section className="rd-card rd-mini"><div className="rd-panel-title"><h2><Target/> Reserve Intelligence</h2><a>View details</a></div><div className="rd-mini-value">{(resource/1e9).toFixed(2)} Bt <span>▲ 2.4%</span></div><small>vs. last update</small><div className="rd-donut"><div>{(resource/1e9).toFixed(2)} Bt<small>Total Resource</small></div></div><div className="rd-breakdown"><span>● Measured <b>42%</b></span><span>● Indicated <b>34%</b></span><span>● Inferred <b>24%</b></span></div></section>; }
function ModelCard({confidence}:{confidence:number}) { return <section className="rd-card rd-mini"><div className="rd-panel-title"><h2><Database/> Model Health</h2><a>View details</a></div><div className="rd-mini-value">{percent(confidence)} <span>▲ 6.3%</span></div><small>Overall Confidence</small><div className="rd-health-list"><span>● Geology Model <b>92%</b></span><span>● Production Model <b>88%</b></span><span>● Grade Model <b>84%</b></span><span>● Geotechnical Model <b>81%</b></span></div></section>; }
function OperationsCard({utilization}:{utilization:number}) { return <section className="rd-card rd-mini"><div className="rd-panel-title"><h2><Truck/> Operations</h2><a>View details</a></div><small>Fleet Utilization</small><div className="rd-mini-value">{percent(utilization)} <span>▲ 12%</span></div><small>vs. last week</small><div className="rd-bars"><i style={{height:'82%'}}/><i style={{height:'78%'}}/><i style={{height:'80%'}}/><i style={{height:'63%'}}/></div><div className="rd-bar-labels"><span>Shovels</span><span>Trucks</span><span>Dozers</span><span>Drills</span></div></section>; }
function ActivityRow({icon,title,detail,time}:{icon:React.ReactNode;title:string;detail:string;time:string}) { return <div className="rd-activity-row"><span>{icon}</span><div><b>{title}</b><small>{detail}</small></div><time>{time}</time></div>; }
