from __future__ import annotations

from fastapi.testclient import TestClient

from backend.app.main import app


def main() -> None:
    client = TestClient(app)
    checks = [
        ("GET", "/health"),
        ("GET", "/api/v1/overview"),
        ("GET", "/api/v1/reserves/summary"),
        ("GET", "/api/v1/production/forecast?horizon=7"),
        ("GET", "/api/v1/recommendations"),
        ("GET", "/api/v1/models"),
        ("GET", "/api/v1/data-quality"),
    ]
    for method, path in checks:
        response = client.get(path)
        print(f"{method} {path} -> {response.status_code}")
        if response.status_code != 200:
            raise SystemExit(f"Smoke test failed: {path}")
    print("smoke tests passed")


if __name__ == "__main__":
    main()
