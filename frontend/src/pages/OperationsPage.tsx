import { Activity, AlertTriangle, CloudRain, Gauge, Radio, Target, Wrench } from "lucide-react";
import { BlastingResponse, EquipmentResponse, ProductionForecastResponse, WeatherResponse } from "../types/api";
import { compactNumber, number, percent } from "../utils/format";

type Props = { equipment: EquipmentResponse; weather: WeatherResponse; blasting: BlastingResponse; production: ProductionForecastResponse };
const Header = ({ icon: Icon, title, meta }: { icon: typeof Activity; title: string; meta: string }) => <div className="panel-header"><div className="panel-title"><Icon size={16}/><span>{title}</span></div><span className="panel-meta">{meta}</span></div>;
const Metric = ({ label, value, tone = "" }: { label: string; value: string; tone?: string }) => <div className={`metric-line ${tone}`}><span>{label}</span><strong>{value}</strong></div>;

export function OperationsPage({ equipment, weather, blasting, production }: Props) {
  const critical = equipment.items.filter(item => item.status === "CRITICAL");
  const watch = equipment.items.filter(item => item.status === "WATCH");
  return <div className="page-grid operations-layout">
    <section className="kpi-grid">
      <div className="metric-card"><Gauge size={18}/><span>Fleet utilization</span><strong>{percent(equipment.fleet_utilization)}</strong><small>productive fleet loading</small></div>
      <div className="metric-card"><Wrench size={18}/><span>Critical assets</span><strong>{number(equipment.critical_equipment_count)}</strong><small>require investigation</small></div>
      <div className="metric-card"><CloudRain size={18}/><span>Weather risk</span><strong>{weather.weather_risk}</strong><small>{number(weather.rainfall_7d_mm, 1)} mm rain / 7d</small></div>
      <div className="metric-card"><Target size={18}/><span>Production risk</span><strong>{percent(production.shortfall_probability)}</strong><small>{production.severity} shortfall signal</small></div>
    </section>
    <section className="panel wide"><Header icon={Radio} title="Operational Control Thread" meta="environment → fleet → production"/><div className="thread-grid"><article><span>01 · ENVIRONMENT</span><strong>{weather.weather_risk}</strong><p>Rainfall {number(weather.rainfall_7d_mm,1)} mm · soil moisture {number(weather.soil_moisture,2)}</p></article><article><span>02 · BLASTING</span><strong>{blasting.overlap_risk}</strong><p>{number(blasting.planned_blasts_7d)} planned blasts · {number(blasting.delay_hours_7d,1)} h delay</p></article><article><span>03 · FLEET</span><strong>{critical.length ? `${critical.length} critical` : "stable"}</strong><p>{watch.length} assets on watch · {percent(equipment.fleet_availability)} availability</p></article><article><span>04 · PRODUCTION</span><strong>{compactNumber(production.forecast_mt)} t</strong><p>7-day forecast · gap {number(production.gap_mt,0)} t</p></article></div></section>
    <section className="panel"><Header icon={AlertTriangle} title="Critical Fleet" meta="next investigation"/>{critical.length ? critical.slice(0, 8).map(item => <Metric key={item.equipment_id} label={item.equipment_id} value={`${number(item.downtime_7d_hours,1)} h downtime`} tone="risk"/>) : <p className="muted">No critical assets in the current operating snapshot.</p>}</section>
    <section className="panel"><Header icon={Activity} title="Decision Signal" meta="bounded support"/><Metric label="Production forecast" value={`${compactNumber(production.forecast_mt)} t`}/><Metric label="Target" value={`${compactNumber(production.target_mt)} t`}/><Metric label="Blast delay" value={`${number(blasting.delay_hours_7d,1)} h`}/><p className="muted">Investigate the highest-impact driver first, then validate against site procedures and supervisor judgement.</p></section>
    <section className="panel wide"><Header icon={Target} title="Operator Checklist" meta="human approval required"/><div className="checklist"><div><b>1</b><span>Inspect critical equipment and confirm actual availability.</span></div><div><b>2</b><span>Check weather/blasting overlap before changing the operating window.</span></div><div><b>3</b><span>Compare production risk drivers with current shift conditions.</span></div><div><b>4</b><span>Approve or reject recommendations only after operational validation.</span></div></div></section>
  </div>;
}
