import { Activity, AlertTriangle, ShieldCheck } from "lucide-react";
import { ModelMonitoringResponse } from "../types/api";
import { number, percent } from "../utils/format";
import { PanelHeader } from "./ui";

export function ModelMonitoringPanel({ monitoring }: { monitoring: ModelMonitoringResponse }) {
  const critical = monitoring.drift.filter((item) => item.status === "critical").length;
  const warning = monitoring.drift.filter((item) => item.status === "warning").length;

  return (
    <section className="panel wide">
      <PanelHeader
        icon={monitoring.status === "healthy" && critical === 0 ? ShieldCheck : AlertTriangle}
        title="Model Monitoring"
        meta={`${monitoring.status} · checked ${new Date(monitoring.checked_at).toLocaleString()}`}
      />
      <div className="kpi-grid monitoring-kpis">
        <article className="metric-card ok">
          <Activity size={19} />
          <span>Registry</span>
          <strong>{monitoring.registry.status}</strong>
        </article>
        <article className={critical ? "metric-card risk" : "metric-card ok"}>
          <AlertTriangle size={19} />
          <span>Critical drift</span>
          <strong>{number(critical)}</strong>
        </article>
        <article className="metric-card">
          <Activity size={19} />
          <span>Warning drift</span>
          <strong>{number(warning)}</strong>
        </article>
        <article className="metric-card">
          <ShieldCheck size={19} />
          <span>Models tracked</span>
          <strong>{number(monitoring.registry.record_count)}</strong>
        </article>
      </div>
      {monitoring.drift.length ? (
        <div className="monitoring-table">
          {monitoring.drift.map((item) => (
            <div className="monitoring-row" key={item.feature}>
              <span>{item.feature.replaceAll("_", " ")}</span>
              <strong>{item.psi.toFixed(3)} PSI</strong>
              <span className={`priority ${item.status}`}>{item.status}</span>
              <span>{percent(Math.min(1, item.current_count / Math.max(1, item.baseline_count)))}</span>
            </div>
          ))}
        </div>
      ) : (
        <p className="muted">No feature drift baseline is configured for this demo snapshot.</p>
      )}
      {monitoring.alerts.length > 0 && (
        <div className="monitoring-alerts">
          {monitoring.alerts.map((alert, index) => (
            <p key={index}>{String(alert.message ?? alert.detail ?? "Monitoring alert")}</p>
          ))}
        </div>
      )}
    </section>
  );
}
