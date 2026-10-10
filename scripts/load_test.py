#!/usr/bin/env python
"""Generate realistic traffic against the API (standard library only).

    python scripts/load_test.py --url http://localhost:8000 --requests 300 --invalid-rate 0.05

Patients are sampled from plausible clinical ranges; a small share of
requests is deliberately invalid so validation-error metrics show up in
Prometheus/Grafana. Exits non-zero if any valid request fails.
"""
from __future__ import annotations

import argparse
import json
import random
import statistics
import time
import urllib.error
import urllib.request


def random_patient(rng: random.Random) -> dict:
    sick = rng.random() < 0.5
    p = {
        "age": rng.randint(32, 77),
        "sex": rng.choice([0, 1, 1]),
        "cp": rng.choice([4, 4, 4, 3, 2]) if sick else rng.choice([1, 2, 3, 3, 4]),
        "trestbps": rng.randint(100, 180),
        "chol": rng.randint(150, 350),
        "fbs": rng.choice([0, 0, 0, 1]),
        "restecg": rng.choice([0, 1, 2]),
        "thalach": rng.randint(95, 150) if sick else rng.randint(130, 195),
        "exang": int(rng.random() < (0.6 if sick else 0.15)),
        "oldpeak": round(rng.uniform(0.5, 4.0) if sick else rng.uniform(0, 1.2), 1),
        "slope": rng.choice([2, 2, 3]) if sick else rng.choice([1, 1, 2]),
        "ca": rng.choice([0, 1, 2, 3]) if sick else rng.choice([0, 0, 0, 1]),
        "thal": rng.choice([7, 7, 6, 3]) if sick else rng.choice([3, 3, 3, 7]),
    }
    for f in ("ca", "thal", "slope", "chol"):       # mimic real-world missing inputs
        if rng.random() < 0.15:
            p.pop(f)
    return p


def post(url: str, body: dict, timeout: float = 5.0) -> tuple[int, float]:
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"content-type": "application/json"})
    t0 = time.perf_counter()
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            r.read()
            return r.status, time.perf_counter() - t0
    except urllib.error.HTTPError as e:
        return e.code, time.perf_counter() - t0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://localhost:8000")
    ap.add_argument("--requests", type=int, default=200)
    ap.add_argument("--invalid-rate", type=float, default=0.05)
    ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    codes, lat, failures = {}, [], 0
    for _ in range(args.requests):
        invalid = rng.random() < args.invalid_rate
        body = {"age": 5, "sex": 1, "cp": 9} if invalid else random_patient(rng)
        code, t = post(f"{args.url.rstrip('/')}/predict", body)
        codes[code] = codes.get(code, 0) + 1
        lat.append(t * 1000)
        if not invalid and code != 200:
            failures += 1
        if args.delay:
            time.sleep(args.delay)

    lat.sort()
    print(json.dumps({"requests": args.requests, "status_codes": codes,
                      "latency_ms": {"p50": round(statistics.median(lat), 2),
                                     "p95": round(lat[int(len(lat) * 0.95) - 1], 2),
                                     "max": round(lat[-1], 2)},
                      "unexpected_failures": failures}, indent=2))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
