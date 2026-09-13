import { Activity, AlertTriangle, ShieldCheck } from "lucide-react";
import type { ModelMonitoringResponse } from "../types/api";
import { number, percent } from "../utils/format";
import { PanelHeader } from "./ui";

export function ModelMonitoringPanel({ monitoring }: { monitoring: ModelMonitoringResponse }) {
  const critical = monitoring.drift.filter((item) => item.status === "critical").length;
  const warning = monitoring.drift.filter((item) => item.status === "warning").length;
  const healthyRatio = monitoring.registry.record_count
    ? Math.max(0, Math.min(1, (monitoring.registry.record_count - critical) / monitoring.registry.record_count))
    : 1;

  return (
    <section className="panel wide model-monitoring-shell">
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
          <small>{number(monitoring.registry.record_count)} models registered</small>
        </article>
        <article className={critical ? "metric-card risk" : "metric-card ok"}>
          <AlertTriangle size={19} />
          <span>Critical drift</span>
          <strong>{number(critical)}</strong>
          <small>PSI threshold breaches</small>
        </article>
        <article className="metric-card">
          <Activity size={19} />
          <span>Warning drift</span>
          <strong>{number(warning)}</strong>
          <small>Features requiring watch</small>
        </article>
        <article className="metric-card">
          <ShieldCheck size={19} />
          <span>Health coverage</span>
          <strong>{percent(healthyRatio)}</strong>
          <small>Registry health ratio</small>
        </article>
      </div>

      <div className="monitoring-detail-grid">
        <div>
          <div className="model-health-visual">
            <div className="mc-health-ring" style={{ background: `conic-gradient(#00dca0 0 ${healthyRatio * 100}%, #102c3c ${healthyRatio * 100}% 100%)` }}>
              {percent(healthyRatio)}
            </div>
            <div>
              <b>Model Health Index</b>
              <span>Registry + drift + monitoring status</span>
              <strong>{monitoring.status.toUpperCase()}</strong>
            </div>
          </div>
          <div className="targeted-monitoring-bars">
            {monitoring.drift.slice(0, 6).map((item) => (
              <div key={item.feature}>
                <span>{item.feature.replaceAll("_", " ")}</span>
                <i style={{ width: `${Math.min(100, item.psi * 100)}%` }} />
                <b>{item.psi.toFixed(3)}</b>
              </div>
            ))}
          </div>
        </div>

        <div className="monitoring-table">
          <div className="monitoring-row monitoring-row-head"><span>Feature</span><strong>PSI</strong><span>Status</span><span>Coverage</span></div>
          {monitoring.drift.length ? monitoring.drift.map((item) => (
            <div className="monitoring-row" key={item.feature}>
              <span>{item.feature.replaceAll("_", " ")}</span>
              <strong>{item.psi.toFixed(3)}</strong>
              <span className={`priority ${item.status}`}>{item.status}</span>
              <span>{percent(Math.min(1, item.current_count / Math.max(1, item.baseline_count)))}</span>
            </div>
          )) : <p className="muted">No feature drift baseline is configured for this demo snapshot.</p>}
        </div>
      </div>

      {monitoring.alerts.length > 0 && (
        <div className="monitoring-alerts">
          {monitoring.alerts.map((alert, index) => <p key={index}>{String(alert.message ?? alert.detail ?? "Monitoring alert")}</p>)}
        </div>
      )}
    </section>
  );
}
