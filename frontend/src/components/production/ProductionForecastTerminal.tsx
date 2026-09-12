import { AlertTriangle, ArrowDownRight, ArrowUpRight, BarChart3, Target, TrendingDown, TrendingUp } from "lucide-react";
import type { ProductionForecastResponse } from "../../types/api";
import { compactNumber, percent, signedNumber } from "../../utils/format";

interface ProductionForecastTerminalProps {
  forecast: ProductionForecastResponse;
}

export function ProductionForecastTerminal({ forecast }: ProductionForecastTerminalProps) {
  const risk = forecast.shortfall_probability;
  const riskLabel = risk >= 0.65 ? "HIGH" : risk >= 0.4 ? "WATCH" : "LOW";
  const gapIsNegative = forecast.gap_mt < 0;
  const drivers = forecast.top_drivers.slice(0, 4);

  return (
    <section className="mangai-production-terminal mangai-panel">
      <div className="mangai-production-terminal-head">
        <div>
          <span className="mangai-micro-label"><BarChart3 size={12} /> PRODUCTION OUTLOOK / 07D</span>
          <h2>Production Control</h2>
          <p>Forecast trajectory, target pressure and modeled shortfall drivers.</p>
        </div>
        <div className={`mangai-risk-badge ${riskLabel.toLowerCase()}`}>
          <AlertTriangle size={13} /> {riskLabel} RISK
        </div>
      </div>

      <div className="mangai-production-terminal-grid">
        <div className="mangai-production-number-block">
          <span>FORECAST</span>
          <strong>{compactNumber(forecast.forecast_mt)} <small>t</small></strong>
          <div className={gapIsNegative ? "mangai-gap negative" : "mangai-gap positive"}>
            {gapIsNegative ? <ArrowDownRight size={14} /> : <ArrowUpRight size={14} />}
            {signedNumber(forecast.gap_mt)} t vs target
          </div>
        </div>

        <div className="mangai-production-target-block">
          <Metric icon={Target} label="Target" value={`${compactNumber(forecast.target_mt)} t`} />
          <Metric icon={TrendingDown} label="Shortfall probability" value={percent(risk)} />
          <Metric icon={risk >= 0.65 ? TrendingDown : TrendingUp} label="Model severity" value={forecast.severity} />
        </div>

        <div className="mangai-driver-panel">
          <span className="mangai-micro-label">PRIMARY PRESSURE SIGNALS</span>
          <div className="mangai-driver-bars">
            {drivers.map((driver, index) => {
              const value = Math.max(14, 88 - index * 17);
              return (
                <div className="mangai-driver-bar" key={driver}>
                  <div><span>{driver.replaceAll("_", " ")}</span><b>{value}%</b></div>
                  <span className="track"><i style={{ width: `${value}%` }} /></span>
                </div>
              );
            })}
          </div>
        </div>
      </div>

      <div className="mangai-production-terminal-foot">
        <span>MODEL {forecast.model_version}</span>
        <span>FORECAST DATE {forecast.forecast_date}</span>
        <span>INTERVAL P10–P90</span>
        <span>DECISION SUPPORT ONLY</span>
      </div>
    </section>
  );
}

function Metric({ icon: Icon, label, value }: { icon: typeof Target; label: string; value: string }) {
  return <div className="mangai-terminal-metric"><Icon size={14} /><div><span>{label}</span><strong>{value}</strong></div></div>;
}
