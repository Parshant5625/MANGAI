from __future__ import annotations

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from backend.app.main import app


def main() -> None:
    client = TestClient(app)
    checks = [
        ("GET", "/health"),
        ("GET", "/ready"),
        ("GET", "/api/v1/overview"),
        ("GET", "/api/v1/reserves/summary"),
        ("GET", "/api/v1/reserves/prospectivity?limit=10"),
        ("GET", "/api/v1/production/forecast?horizon=7"),
        ("GET", "/api/v1/production/risk"),
        ("GET", "/api/v1/equipment"),
        ("GET", "/api/v1/weather"),
        ("GET", "/api/v1/blasting"),
        ("GET", "/api/v1/recommendations"),
        ("GET", "/api/v1/models"),
        ("GET", "/api/v1/data-quality"),
    ]
    passed = 0
    failed = 0
    for method, path in checks:
        response = client.get(path)
        status = "PASS" if response.status_code == 200 else "FAIL"
        print(f"[{status}] {method} {path} -> {response.status_code}")
        if response.status_code == 200:
            passed += 1
        else:
            failed += 1
    print(f"\nResults: {passed} passed, {failed} failed")
    if failed > 0:
        raise SystemExit(f"Smoke tests failed: {failed} endpoints returned non-200")
    print("All smoke tests passed!")


if __name__ == "__main__":
    main()