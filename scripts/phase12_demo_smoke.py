from __future__ import annotations

import sys
from urllib.error import URLError
from urllib.request import Request, urlopen

BASE = "http://127.0.0.1:8000"
ENDPOINTS = [
    "/api/v1/overview",
    "/api/v1/reserves/summary",
    "/api/v1/production/forecast?horizon=7",
    "/api/v1/operations/summary?days=30",
    "/api/v1/recommendations",
    "/api/v1/models",
    "/api/v1/models/monitoring",
    "/api/v1/data-quality",
]


def main() -> int:
    failures: list[str] = []
    for endpoint in ENDPOINTS:
        request = Request(BASE + endpoint, headers={"Accept": "application/json"})
        try:
            with urlopen(request, timeout=15) as response:
                status = response.status
                payload = response.read(1024).decode("utf-8", errors="replace")
        except (URLError, TimeoutError, OSError) as exc:
            failures.append(f"{endpoint}: {exc}")
            continue
        if status != 200:
            failures.append(f"{endpoint}: HTTP {status}")
            continue
        if not payload.lstrip().startswith("{"):
            failures.append(f"{endpoint}: non-JSON response")
            continue
        print(f"PASS {endpoint}")

    if failures:
        print("\nFAILURES")
        for failure in failures:
            print(f"- {failure}")
        return 1
    print("\nMANGAI Phase 12 demo smoke: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
