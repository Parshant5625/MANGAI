import { ArrowUpRight, CircleAlert, CircleCheck, Info } from "lucide-react";

interface Signal {
  title: string;
  value: string;
  detail: string;
  tone: "healthy" | "warning" | "critical" | "intelligence";
}

const icons = { healthy: CircleCheck, warning: Info, critical: CircleAlert, intelligence: ArrowUpRight };

export function SignalStack({ signals }: { signals: Signal[] }) {
  return (
    <div className="rebuild-signal-stack">
      {signals.map((signal) => {
        const Icon = icons[signal.tone];
        return (
          <article className={`rebuild-signal rebuild-signal-${signal.tone}`} key={signal.title}>
            <div className="rebuild-signal-icon"><Icon size={14} /></div>
            <div className="rebuild-signal-copy">
              <span>{signal.title}</span>
              <strong>{signal.value}</strong>
              <small>{signal.detail}</small>
            </div>
          </article>
        );
      })}
    </div>
  );
}
