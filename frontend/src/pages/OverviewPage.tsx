import {
  Activity,
  AlertTriangle,
  BarChart3,
  ClipboardList,
  Database,
  Map,
  Target,
  Wrench
} from "lucide-react";
import { DriverList, MetricCard, PanelHeader } from "../components/ui";
import {
  DataQualityResponse,
  EquipmentResponse,
  OverviewResponse,
  ProductionForecastResponse,
  RecommendationResponse
} from "../types/api";
import { compactNumber, number, percent, signedNumber } from "../utils/format";

export function OverviewPage({
  overview,
  production,
  recommendations,
  equipment,
  quality
}: {
  overview: OverviewResponse;
  production: ProductionForecastResponse;
  recommendations: RecommendationResponse;
  equipment: EquipmentResponse;
  quality: DataQualityResponse;
}) {
  return (
    <div className="page-grid">
      <section className="kpi-grid">
        <MetricCard icon={Target} label="Prototype resource potential" value={`${compactNumber(overview.resource_potential_tonnage)} t`} />
        <MetricCard icon={Map} label="High prospectivity area" value={`${number(overview.high_prospectivity_area_ha, 1)} ha`} />
        <MetricCard icon={BarChart3} label="Next 7-day production" value={`${compactNumber(overview.next_7_day_production_mt)} t`} />
        <MetricCard
          icon={AlertTriangle}
          label="Shortfall probability"
          value={percent(overview.shortfall_probability)}
          tone={overview.shortfall_probability > 0.65 ? "risk" : "ok"}
        />
      </section>
      <section className="kpi-grid">
        <MetricCard icon={Activity} label="Production gap" value={`${signedNumber(overview.production_gap_mt)} t`} tone={overview.production_gap_mt < 0 ? "risk" : "ok"} />
        <MetricCard icon={Wrench} label="Critical equipment" value={number(overview.critical_equipment_count)} tone={overview.critical_equipment_count > 0 ? "risk" : "ok"} />
        <MetricCard icon={ClipboardList} label="Recommendations" value={number(overview.recommendation_count)} />
        <MetricCard icon={Database} label="Data quality" value={percent(overview.data_quality_score)} />
      </section>
      <section className="panel wide">
        <PanelHeader icon={Activity} title="Production Forecast" meta={production.severity} />
        <div className="split">
          <div>
            <p className="massive">{signedNumber(production.gap_mt)} t</p>
            <p className="muted">Forecast gap against 7-day target · {production.model_version}</p>
          </div>
          <DriverList drivers={production.top_drivers} />
        </div>
      </section>
      <section className="panel">
        <PanelHeader icon={Wrench} title="Equipment" meta={`${equipment.critical_equipment_count} critical`} />
        {equipment.items.length ? (
          <div className="rank-list">
            {equipment.items.slice(0, 5).map((item) => (
              <div className="rank-row" key={item.equipment_id}>
                <span>{item.equipment_id}</span>
                <strong>{number(item.downtime_7d_hours, 1)} h</strong>
              </div>
            ))}
          </div>
        ) : (
          <p className="muted">No equipment telemetry in this snapshot.</p>
        )}
      </section>
      <section className="panel">
        <PanelHeader icon={ClipboardList} title="Recommendations" meta={`${recommendations.recommendations.length} proposed`} />
        <div className="stack">
          {recommendations.recommendations.slice(0, 3).map((item) => (
            <div className="action-row" key={item.id}>
              <span className={`priority ${item.priority.toLowerCase()}`}>{item.priority}</span>
              <p>{item.title}</p>
            </div>
          ))}
        </div>
      </section>
      <section className="panel">
        <PanelHeader icon={Database} title="Data Quality" meta={percent(quality.overall_score)} />
        <div className="quality-bar">
          <span style={{ width: `${quality.overall_score * 100}%` }} />
        </div>
        <p className="muted">{overview.model_health} · synthetic prototype</p>
      </section>
    </div>
  );
}
