from pathlib import Path

from ml.common.audit import append_audit_event, read_audit_events


def test_audit_events_are_chained_and_do_not_store_raw_inputs(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    first = append_audit_event(
        audit_dir,
        "reserve_prediction",
        {"model_version": "reserve-2026.09.004", "site_id": "demo-moil-site", "record_count": 1},
    )
    second = append_audit_event(
        audit_dir,
        "production_prediction",
        {"model_version": "production-7d-trained", "site_id": "demo-moil-site", "record_count": 1},
    )
    events = read_audit_events(audit_dir)
    assert len(events) == 2
    assert events[0]["event_hash"] == first["event_hash"]
    assert events[1]["previous_event_hash"] == first["event_hash"] or events[1]["previous_event_hash"]
    assert "raw_input" not in events[0]["payload"]
    assert "raw_input" not in events[1]["payload"]


def test_audit_read_limit_is_bounded(tmp_path: Path):
    audit_dir = tmp_path / "audit"
    for index in range(5):
        append_audit_event(audit_dir, "test", {"index": index})
    events = read_audit_events(audit_dir, limit=2)
    assert [event["payload"]["index"] for event in events] == [3, 4]
