from backend.app.services.recommendations import RecommendationService


def test_recommendations_are_structured(demo_store):
    response = RecommendationService().list_recommendations()
    assert response["synthetic_data"] is True
    for item in response["recommendations"]:
        assert item["rationale"]
        assert item["evidence"]
        assert "requires_human_approval" in item["suggested_window"]
        assert item["priority"] in {"LOW", "MEDIUM", "HIGH", "CRITICAL"}
