"""Smoke test for a running service (for example the Docker container): standard library only, so it runs anywhere."""
from __future__ import annotations

import argparse
import json
import logging
import time
import urllib.error
import urllib.request
from dataclasses import replace
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from fraud import benchmark, preprocess
from fraud.benchmark import percentiles
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging

LOGGER = logging.getLogger(__name__)


def call(url: str, payload: Optional[Dict[str, Any]] = None, timeout: float = 10.0) -> Dict[str, Any]:
    """GET (no payload) or POST JSON; returns the decoded JSON body.

    Raises:
        RuntimeError: On an HTTP error (with the server's message) or when the server cannot be reached.
    """
    data = json.dumps(payload).encode() if payload is not None else None
    request = urllib.request.Request(url, data=data, headers={"Content-Type": "application/json"} if data else {})
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read())
    except urllib.error.HTTPError as error:
        raise RuntimeError(f"{url} returned {error.code}: {error.read().decode(errors='replace')[:300]}") from error
    except urllib.error.URLError as error:
        raise RuntimeError(f"cannot reach {url}: {error.reason}") from error


def run(base_url: str, records: Sequence[Dict[str, float]], budget_ms: float = 100.0) -> Dict[str, Any]:
    """Check /health, score every record one by one (timing the round trip), score them as a batch, and read /drift.

    Raises:
        RuntimeError: If any call fails or the batch result disagrees with the single results.
    """
    base_url = base_url.rstrip("/")
    health = call(f"{base_url}/health")
    if health.get("status") != "ok":
        raise RuntimeError(f"unhealthy: {health}")
    singles, trips = [], []
    for record in records:
        start = time.perf_counter()
        singles.append(call(f"{base_url}/score", record))
        trips.append((time.perf_counter() - start) * 1000)
    batch = call(f"{base_url}/score/batch", {"transactions": list(records)})["results"]
    if [r["tier"] for r in batch] != [r["tier"] for r in singles]:
        raise RuntimeError("batch tiers differ from single-transaction tiers")
    explained = call(f"{base_url}/score?explain=true", records[0])
    drift = call(f"{base_url}/drift") if health.get("drift_monitoring") else {"overall": "off"}
    summary = {"model": health["model"], "tiers": [r["tier"] for r in singles], "scores": [r["score"] for r in singles],
               "round_trip_ms": percentiles(trips), "reasons_for_first": explained.get("reasons"), "drift": drift.get("overall"),
               "within_budget": bool(percentiles(trips)["p95"] <= budget_ms)}
    LOGGER.info("model %s | tiers %s | round trip p50 %.1f ms, p95 %.1f ms (budget %.0f) | drift: %s", summary["model"], summary["tiers"],
                summary["round_trip_ms"]["p50"], summary["round_trip_ms"]["p95"], budget_ms, summary["drift"])
    return summary


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point: takes a few real test transactions (frauds and genuine) from ``data/processed`` and sends them."""
    parser = argparse.ArgumentParser(description="Smoke-test a running scoring service (Step 9).")
    parser.add_argument("--url", default="http://localhost:8000")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root holding data/processed (default: current folder)")
    parser.add_argument("--count", type=int, default=6, help="how many transactions to send (half frauds)")
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=args.root))
    test = preprocess.load_split(cfg)["test"]
    label = cfg.data.label_column
    half = max(args.count // 2, 1)
    picked = list(benchmark.raw_records(test[test[label] == 1].head(half), cfg, half)) + list(benchmark.raw_records(test[test[label] == 0], cfg, args.count - half))
    try:
        return 0 if run(args.url, picked, cfg.service.latency_budget_ms)["within_budget"] else 1
    except RuntimeError as error:
        LOGGER.error("%s\nIs the service running? Start it first in another window (leave that window open): "
                     "`docker run --rm -p 8000:8000 fraud-api`, or locally `uvicorn fraud.api:create_default_app --factory --port 8000`.", error)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
