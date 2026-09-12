import type { LucideIcon } from "lucide-react";

interface IntelligenceMetricProps {
  label: string;
  value: string;
  detail?: string;
  icon: LucideIcon;
  tone?: "neutral" | "healthy" | "warning" | "critical" | "intelligence";
}

export function IntelligenceMetric({ label, value, detail, icon: Icon, tone = "neutral" }: IntelligenceMetricProps) {
  return (
    <article className={`rebuild-metric rebuild-metric-${tone}`}>
      <div className="rebuild-metric-top">
        <span>{label}</span>
        <Icon size={15} aria-hidden="true" />
      </div>
      <strong>{value}</strong>
      {detail && <small>{detail}</small>}
    </article>
  );
}
