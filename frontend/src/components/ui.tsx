import { Activity } from "lucide-react";
import { TopDriver } from "../types/api";

export function StatePanel({ label, tone = "default" }: { label: string; tone?: "default" | "danger" }) {
  return <section className={tone === "danger" ? "state danger" : "state"}>{label}</section>;
}

export function MetricCard({
  icon: Icon,
  label,
  value,
  tone = "default"
}: {
  icon: typeof Activity;
  label: string;
  value: string;
  tone?: "default" | "risk" | "ok";
}) {
  return (
    <article className={`metric-card ${tone}`}>
      <Icon size={19} />
      <span>{label}</span>
      <strong>{value}</strong>
    </article>
  );
}

export function PanelHeader({ icon: Icon, title, meta }: { icon: typeof Activity; title: string; meta?: string }) {
  return (
    <div className="panel-header">
      <div>
        <Icon size={18} />
        <h3>{title}</h3>
      </div>
      {meta && <span>{meta}</span>}
    </div>
  );
}

export function MetricLine({ label, value }: { label: string; value: string }) {
  return (
    <div className="metric-line">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

export function DriverList({ drivers }: { drivers: TopDriver[] }) {
  if (!drivers.length) {
    return <p className="muted">No driver attributions available yet.</p>;
  }
  return (
    <div className="driver-list">
      {drivers.map((driver) => (
        <div className="driver" key={driver.feature}>
          <div>
            <span>{driver.feature.replaceAll("_", " ")}</span>
            <strong>{driver.direction}</strong>
          </div>
          <div className="mini-bar">
            <span style={{ width: `${Math.max(6, driver.importance * 100)}%` }} />
          </div>
        </div>
      ))}
    </div>
  );
}
