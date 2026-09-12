from backend.app.api.v1.router import chat


def test_chat_returns_bounded_evidence_response() -> None:
    result = chat(
        {
            "message": "Why is production at risk right now?",
            "context": {
                "production_risk": 0.72,
                "production_gap_mt": -850,
                "fleet_utilization": 0.61,
                "data_quality": 0.91,
                "selected_target": "cell-17",
            },
        }
    )

    assert result["mode"] == "bounded-decision-support"
    assert result["confidence"] >= 0.45
    assert "shortfall risk is elevated" in result["answer"]
    assert result["evidence"]


def test_chat_handles_empty_question() -> None:
    result = chat({"message": "", "context": {"data_quality": 0.8}})

    assert result["answer"]
    assert result["mode"] == "bounded-decision-support"
