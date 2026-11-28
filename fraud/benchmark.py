"""Latency benchmark: how long does one scoring request take, against the 100 ms budget?"""
from __future__ import annotations

import argparse
import json
import logging
import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from fraud import preprocess
from fraud.config import ProjectConfig, ServiceConfig
from fraud.logging_utils import configure_logging
from fraud.scoring import Scorer

LOGGER = logging.getLogger(__name__)


def percentiles(samples_ms: Sequence[float]) -> Dict[str, float]:
    """Mean, p50, p95, p99 and max of latency samples in milliseconds.

    Raises:
        ValueError: If there are no samples.
    """
    s = np.asarray(samples_ms, dtype=float)
    if s.size == 0:
        raise ValueError("no samples")
    return {"mean": round(float(s.mean()), 3), "p50": round(float(np.percentile(s, 50)), 3), "p95": round(float(np.percentile(s, 95)), 3),
            "p99": round(float(np.percentile(s, 99)), 3), "max": round(float(s.max()), 3)}


def time_calls(call: Callable[[Dict[str, float]], Any], records: Sequence[Dict[str, float]], warmup: int) -> List[float]:
    """Latency in ms of ``call(record)`` for each record, after ``warmup`` untimed calls (the first calls are slower: lazy loading, caches)."""
    if warmup < 0 or len(records) <= warmup:
        raise ValueError("need more records than warm-up calls")
    for record in records[:warmup]:
        call(record)
    out = []
    for record in records[warmup:]:
        start = time.perf_counter()
        call(record)
        out.append((time.perf_counter() - start) * 1000)
    return out


def raw_records(test: pd.DataFrame, cfg: ProjectConfig, n: int) -> List[Dict[str, float]]:
    """``n`` raw transactions (Time, V1-V28, Amount) taken from the processed test rows."""
    d, p = cfg.data, cfg.prep
    chosen = test.sample(n=min(n, len(test)), random_state=42)
    frame = chosen[list(d.pca_columns)].copy()
    frame[d.time_column], frame[d.amount_column] = chosen[p.raw_time_name].to_numpy(), chosen[p.raw_amount_name].to_numpy()
    return frame.to_dict("records")


def run(cfg: ProjectConfig, split: Optional[str] = None, scorer: Optional[Scorer] = None) -> Dict[str, Any]:
    """Time single-transaction scoring (with and without reasons) and, if FastAPI + httpx are installed, a full HTTP round trip in-process."""
    svc: ServiceConfig = cfg.service
    logging.getLogger("httpx").setLevel(logging.WARNING)       # one INFO line per request would bury the results
    scorer = scorer or Scorer.load(cfg)
    test = preprocess.load_split(cfg, split)["test"]
    records = raw_records(test, cfg, svc.benchmark_requests + svc.benchmark_warmup)
    results: Dict[str, Any] = {"requests": len(records) - svc.benchmark_warmup, "budget_ms": svc.latency_budget_ms}
    results["scoring"] = percentiles(time_calls(lambda r: scorer.score_records([r]), records, svc.benchmark_warmup))
    try:
        results["scoring_with_reasons"] = percentiles(time_calls(lambda r: scorer.score_records([r], explain=True), records, svc.benchmark_warmup))
    except ImportError as error:
        results["scoring_with_reasons"] = {"skipped": str(error)}
    try:
        from fastapi.testclient import TestClient
        from fraud.api import create_app
        client = TestClient(create_app(scorer))
        results["http_round_trip"] = percentiles(time_calls(lambda r: client.post("/score", json=r).raise_for_status(), records, svc.benchmark_warmup))
    except (ImportError, RuntimeError):                        # starlette's TestClient raises RuntimeError when httpx is missing
        results["http_round_trip"] = {"skipped": "fastapi / httpx not installed"}
    measured = [v["p99"] for v in results.values() if isinstance(v, dict) and "p99" in v]
    results["worst_p99_ms"] = max(measured)
    results["within_budget"] = bool(max(measured) <= svc.latency_budget_ms)
    for name in ("scoring", "scoring_with_reasons", "http_round_trip"):
        LOGGER.info("%-22s %s", name, results[name])
    LOGGER.info("worst p99 %.1f ms against a budget of %.0f ms: %s", results["worst_p99_ms"], svc.latency_budget_ms, "WITHIN BUDGET" if results["within_budget"] else "OVER BUDGET")
    cfg.paths.reports_path.mkdir(parents=True, exist_ok=True)
    (cfg.paths.reports_path / "latency_report.json").write_text(json.dumps(results, indent=2), encoding="utf-8")
    return results


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Command-line entry point; exits 1 if the 99th percentile is over budget."""
    parser = argparse.ArgumentParser(description="Scoring latency benchmark (Step 9).")
    parser.add_argument("--root", type=Path, default=Path("."), help="project root (default: current folder)")
    parser.add_argument("--split", choices=preprocess.METHODS, default=None)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args(argv)
    configure_logging(args.log_level)
    base = ProjectConfig()
    return 0 if run(replace(base, paths=replace(base.paths, root=args.root)), args.split)["within_budget"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
