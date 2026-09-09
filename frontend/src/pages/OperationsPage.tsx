import { AlertTriangle, BarChart3, CloudRain, Gauge, ShieldCheck, Target, Wrench } from "lucide-react";
import { ResponsiveContainer, BarChart, Bar, CartesianGrid, XAxis, YAxis, Tooltip } from "recharts";
import { useApi } from "../hooks/useApi";
import { MetricCard, PanelHeader } from "../components/ui";
import { number, percent } from "../utils/format";
import type { OperationsSummaryResponse } from "../types/api";

function levelClass(level: string) {
  return `priority ${level.toLowerCase()}`;
}

export function OperationsPage() {
  const { data, loading, error } = useApi<OperationsSummaryResponse>("/api/v1/operations/summary?days=30");

  if (loading) return <section className="state">Loading cross-domain operational intelligence…</section>;
  if (error) return <section className="state danger">Operations intelligence unavailable: {error}</section>;
  if (!data) return <section className="state">No operations snapshot available.</section>;

  const associationData = data.production_associations.map((item) => ({
    driver: item.driver.replaceAll("_", " "),
    correlation: item.correlation
  }));

  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={ShieldCheck} label="Operational risk" value={data.overall_operational_risk} tone={data.overall_operational_risk === "HIGH" ? "risk" : "default"} />
        <MetricCard icon={Wrench} label="Fleet availability" value={percent(data.fleet_availability)} />
        <MetricCard icon={Gauge} label="Fleet utilization" value={percent(data.fleet_utilization)} />
        <MetricCard icon={AlertTriangle} label="7-day blast delay" value={`${number(data.blasting_delay_7d_hours, 1)} h`} tone={data.blasting_delay_7d_hours > 12 ? "risk" : "default"} />
      </section>

      <section className="panel wide">
        <PanelHeader icon={ShieldCheck} title="Operational Risk Signals" meta={`score ${number(data.overall_risk_score, 2)}`} />
        <div className="signal-grid">
          {data.risk_signals.map((signal) => (
            <article className="signal-card" key={signal.source}>
              <div className="signal-head">
                <span className={levelClass(signal.level)}>{signal.level}</span>
                <strong>{percent(signal.score)}</strong>
              </div>
              <h3>{signal.title}</h3>
              <div className="evidence-grid">
                {Object.entries(signal.evidence).map(([key, value]) => (
                  <div className="metric-line" key={key}>
                    <span>{key.replaceAll("_", " ")}</span>
                    <strong>{typeof value === "object" ? JSON.stringify(value) : String(value)}</strong>
                  </div>
                ))}
              </div>
            </article>
          ))}
        </div>
      </section>

      <section className="panel">
        <PanelHeader icon={CloudRain} title="Weather Exposure" meta={data.latest_date} />
        <MetricLine label="Rainfall, 7d" value={`${number(data.rainfall_7d_mm, 1)} mm`} />
        <MetricLine label="Soil moisture" value={percent(data.soil_moisture)} />
        <MetricLine label="Aligned weather days" value={number(data.data_coverage.weather)} />
      </section>

      <section className="panel">
        <PanelHeader icon={Target} title="Blasting Exposure" meta={`${data.planned_blasts_7d} planned`} />
        <MetricLine label="Delay, 7d" value={`${number(data.blasting_delay_7d_hours, 1)} h`} />
        <MetricLine label="Aligned blasting days" value={number(data.data_coverage.blasting)} />
        <MetricLine label="Production records" value={number(data.production_records)} />
      </section>

      <section className="panel wide chart-panel">
        <PanelHeader icon={BarChart3} title="Observed Production Associations" meta="descriptive · not causal" />
        <ResponsiveContainer width="100%" height={300}>
          <BarChart data={associationData} layout="vertical" margin={{ left: 32, right: 18 }}>
            <CartesianGrid stroke="#d8ded7" strokeDasharray="3 3" />
            <XAxis type="number" domain={[-1, 1]} />
            <YAxis type="category" dataKey="driver" width={130} />
            <Tooltip />
            <Bar dataKey="correlation" fill="#1f7a5f" radius={[0, 4, 4, 0]} />
          </BarChart>
        </ResponsiveContainer>
      </section>

      <section className="panel">
        <PanelHeader icon={Gauge} title="Data Coverage" meta={`${data.analysis_window_days}-day window`} />
        <MetricLine label="Production" value={number(data.data_coverage.production)} />
        <MetricLine label="Weather" value={number(data.data_coverage.weather)} />
        <MetricLine label="Equipment" value={number(data.data_coverage.equipment)} />
        <MetricLine label="Blasting" value={number(data.data_coverage.blasting)} />
        <MetricLine label="Aligned days" value={number(data.data_coverage.aligned_days)} />
      </section>

      <section className="panel wide">
        <PanelHeader icon={Wrench} title="Interpretation Boundary" meta="analytics only" />
        <p className="boundary">{data.methodology_note}</p>
      </section>
    </div>
  );
}

function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}
