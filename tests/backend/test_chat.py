from fastapi.testclient import TestClient

from backend.app.main import app


client = TestClient(app)


def test_chat_help_returns_structured_response() -> None:
    response = client.post("/api/v1/chat", json={"message": "What can you do?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "help"
    assert payload["answer"]
    assert payload["suggested_questions"]
    assert payload["data_mode"] in {"demo", "live"}


def test_chat_production_uses_mangai_forecast() -> None:
    response = client.post("/api/v1/chat", json={"message": "What is the 7-day production forecast?"})
    assert response.status_code == 200
    payload = response.json()
    assert payload["intent"] == "production"
    labels = {item["label"] for item in payload["evidence"]}
    assert "7-day forecast" in labels
    assert "Shortfall probability" in labels


def test_chat_unknown_is_safe() -> None:
    response = client.post("/api/v1/chat", json={"message": "Tell me something unrelated."})
    assert response.status_code == 200
    assert response.json()["intent"] == "unknown"
