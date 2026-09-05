"""Tests for backend services."""

from backend.app.services.reserve import ReserveService
from backend.app.services.production import ProductionService
from backend.app.services.recommendations import RecommendationService
from backend.app.services.operations import OperationsService
from backend.app.services.data_quality import DataQualityService


def test_reserve_service_prospectivity(demo_store):
    """Test reserve prospectivity service returns valid data."""
    service = ReserveService(store=demo_store)
    result = service.get_prospectivity(limit=50)
    
    assert result["count"] > 0
    assert len(result["cells"]) > 0
    assert result["synthetic_data"] is True
    
    cell = result["cells"][0]
    assert 0 <= cell["probability"] <= 1
    assert cell["prospectivity_class"] in ["LOW", "MODERATE", "HIGH", "VERY_HIGH"]
    assert cell["confidence"] > 0


def test_reserve_service_summary(demo_store):
    """Test reserve summary service returns valid data."""
    service = ReserveService(store=demo_store)
    result = service.get_summary()
    
    assert result["cells_evaluated"] > 0
    assert result["high_prospectivity_cells"] >= 0
    assert result["prototype_resource_potential"]["expected_tonnage"] > 0
    assert result["prototype_resource_potential"]["p10"] > 0


def test_production_service_forecast(demo_store):
    """Test production forecast service returns valid data."""
    service = ProductionService(store=demo_store)
    result = service.forecast(horizon=7)
    
    assert result["horizon_days"] == 7
    assert result["forecast_mt"] > 0
    assert 0 <= result["shortfall_probability"] <= 1
    assert result["severity"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
    assert result["prediction_interval"]["p10"] <= result["prediction_interval"]["p50"]
    assert result["prediction_interval"]["p50"] <= result["prediction_interval"]["p90"]


def test_production_service_history(demo_store):
    """Test production history service returns valid data."""
    service = ProductionService(store=demo_store)
    result = service.history(days=30)
    
    assert len(result) > 0
    assert "production_mt" in result[0]
    assert "target_mt" in result[0]


def test_operations_service_equipment(demo_store):
    """Test equipment service returns valid data."""
    service = OperationsService(store=demo_store)
    result = service.equipment()
    
    assert result["fleet_availability"] >= 0
    assert result["fleet_utilization"] >= 0
    assert len(result["items"]) > 0
    
    item = result["items"][0]
    assert item["status"] in ["NORMAL", "WATCH", "CRITICAL"]


def test_operations_service_weather(demo_store):
    """Test weather service returns valid data."""
    service = OperationsService(store=demo_store)
    result = service.weather()
    
    assert result["weather_risk"] in ["LOW", "MEDIUM", "HIGH"]
    assert len(result["observations"]) > 0


def test_operations_service_blasting(demo_store):
    """Test blasting service returns valid data."""
    service = OperationsService(store=demo_store)
    result = service.blasting()
    
    assert result["delay_trend"] in ["IMPROVING", "STABLE", "WORSENING"]
    assert result["overlap_risk"] in ["LOW", "MEDIUM", "HIGH"]


def test_recommendation_service_list(demo_store):
    """Test recommendation service returns valid data."""
    service = RecommendationService()
    result = service.list_recommendations()
    
    assert "recommendations" in result
    if result["recommendations"]:
        rec = result["recommendations"][0]
        assert rec["priority"] in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]
        assert rec["category"] in ["EQUIPMENT", "PRODUCTION", "WEATHER", "RESERVE", "BLASTING"]
        assert "evidence" in rec
        assert "estimated_impact" in rec


def test_recommendation_service_simulate(demo_store):
    """Test recommendation simulation returns valid data."""
    service = RecommendationService()
    result = service.simulate({"reduce_downtime_pct": 20})
    
    assert "recommendations" in result
    if result["recommendations"]:
        assert result["recommendations"][0]["status"] == "SIMULATED"


def test_data_quality_service(demo_store):
    """Test data quality service returns valid data."""
    service = DataQualityService(store=demo_store)
    result = service.run()
    
    assert 0 <= result["overall_score"] <= 1
    assert len(result["runs"]) > 0
    
    run = result["runs"][0]
    assert run["quality_score"] >= 0
    assert isinstance(run["schema_valid"], bool)