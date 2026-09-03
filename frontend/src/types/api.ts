export type Direction = "positive" | "negative" | "neutral";

export interface DemoEnvelope {
  data_mode: string;
  synthetic_data: boolean;
  boundary_notice: string;
}

export interface TopDriver {
  feature: string;
  direction: Direction;
  importance: number;
  value?: number | string | null;
}

export interface ResourcePotential {
  label: string;
  expected_tonnage: number;
  p10: number;
  p50: number;
  p90: number;
  assumptions: Record<string, unknown>;
}

export interface ProspectivityCell {
  id: string;
  latitude: number;
  longitude: number;
  probability: number;
  prospectivity_class: string;
  predicted_grade_pct: number;
  predicted_thickness_m: number;
  confidence: number;
  resource_potential: ResourcePotential;
  top_contributors: TopDriver[];
  data_support?: Record<string, unknown>;
  geology?: Record<string, number | string>;
}

export interface OverviewResponse extends DemoEnvelope {
  site_id: string;
  site_name: string;
  resource_potential_tonnage: number;
  high_prospectivity_area_ha: number;
  next_7_day_production_mt: number;
  shortfall_probability: number;
  production_gap_mt: number;
  critical_equipment_count: number;
  recommendation_count: number;
  model_health: string;
  data_quality_score: number;
  kpis: Record<string, number | string>;
}

export interface ReserveProspectivityResponse extends DemoEnvelope {
  site_id: string;
  count: number;
  cells: ProspectivityCell[];
}

export interface ReserveSummaryResponse extends DemoEnvelope {
  high_prospectivity_cells: number;
  very_high_prospectivity_cells: number;
  high_prospectivity_area_ha: number;
  average_probability: number;
  average_predicted_grade_pct: number;
  average_predicted_thickness_m: number;
  prototype_resource_potential: ResourcePotential;
  validation_note: string;
}

export interface HorizonPoint {
  date: string;
  horizon_day: number;
  forecast_mt: number;
  target_mt: number;
  p10?: number;
  p90?: number;
}

export interface ProductionForecastResponse extends DemoEnvelope {
  forecast_date: string;
  forecast_origin: string;
  horizon_days: number;
  forecast_mt: number;
  target_mt: number;
  gap_mt: number;
  shortfall_probability: number;
  severity: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  prediction_interval: { p10: number; p50: number; p90: number };
  top_drivers: TopDriver[];
  model_version: string;
  baseline_forecast_mt: number;
  data_freshness: Record<string, unknown>;
  horizon_series: HorizonPoint[];
}

export interface ProductionHistoryRecord {
  date: string;
  production_mt: number;
  target_mt: number;
  gap_signed_mt: number;
  rainfall_mm: number;
  downtime_hours: number;
  blasting_delay_hours: number;
}

export interface EquipmentItem {
  equipment_id: string;
  equipment_type: string;
  availability: number;
  utilization: number;
  downtime_7d_hours: number;
  downtime_30d_hours: number;
  maintenance_events_30d: number;
  status: "NORMAL" | "WATCH" | "CRITICAL";
}

export interface EquipmentResponse extends DemoEnvelope {
  fleet_availability: number;
  fleet_utilization: number;
  critical_equipment_count: number;
  items: EquipmentItem[];
  maintenance_trend: Array<{ date: string; maintenance_events: number }>;
}

export interface WeatherResponse extends DemoEnvelope {
  latest_date: string;
  rainfall_7d_mm: number;
  rainfall_30d_mm: number;
  soil_moisture: number;
  temperature_c: number;
  weather_risk: "LOW" | "MEDIUM" | "HIGH";
  observations: Array<Record<string, number | string>>;
}

export interface BlastingResponse extends DemoEnvelope {
  latest_date: string;
  planned_blasts_7d: number;
  delay_hours_7d: number;
  delay_trend: "IMPROVING" | "STABLE" | "WORSENING";
  overlap_risk: "LOW" | "MEDIUM" | "HIGH";
  events: Array<Record<string, number | string>>;
}

export interface Recommendation {
  id: string;
  category: string;
  priority: "LOW" | "MEDIUM" | "HIGH" | "CRITICAL";
  title: string;
  rationale: string;
  evidence: Record<string, number | string | boolean>;
  estimated_impact: Record<string, number | number[] | string>;
  confidence: number;
  affected_equipment: string[];
  affected_area?: string | null;
  suggested_window: Record<string, number | string | boolean>;
  status: "PROPOSED" | "SIMULATED";
}

export interface RecommendationResponse extends DemoEnvelope {
  recommendations: Recommendation[];
}

export interface ModelRegistryResponse extends DemoEnvelope {
  models: Array<{
    model_name: string;
    version: string;
    task: string;
    algorithm: string;
    metrics: Record<string, unknown>;
    status: string;
    notes: string;
    drift?: Record<string, unknown>;
  }>;
}

export interface DataQualityResponse extends DemoEnvelope {
  overall_score: number;
  runs: Array<{
    dataset_name: string;
    row_count: number;
    missing_rate: number;
    duplicate_rate: number;
    schema_valid: boolean;
    quality_score: number;
    details: Record<string, unknown>;
  }>;
}
