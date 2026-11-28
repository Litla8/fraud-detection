"""Tests for Step 9: schemas, feature building, scoring, explanations, drift, benchmark, smoke test, requirement pinning and the API."""
from __future__ import annotations

import json
import sys
import threading
import types
from dataclasses import replace
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from fraud import benchmark, drift, explain, optimize_threshold, pin_requirements, preprocess, report, schemas, scoring, smoke_test, train_supervised
from fraud.config import DataConfig, ImbalanceConfig, ProjectConfig, ServiceConfig, SupervisedConfig, ThresholdConfig
from tests.conftest import make_transactions

SERVICE = ServiceConfig(min_window=50, monitor_buffer=800, benchmark_requests=30, benchmark_warmup=5, shap_rows=300)


@pytest.fixture(scope="module")
def project(tmp_path_factory: pytest.TempPathFactory) -> ProjectConfig:
    """A tiny finished project: split, logistic-regression model (native explanations), decision policy and drift reference."""
    root = tmp_path_factory.mktemp("service")
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=root), service=SERVICE,
                  thresholds=ThresholdConfig(bootstrap_resamples=15, savings_resamples=15, tier_candidates=30),
                  supervised=SupervisedConfig(models=("logistic_regression",), variants=("baseline",), bootstrap_resamples=20),
                  imbalance=ImbalanceConfig(sampling_ratio=0.2))
    cfg.paths.raw_path.parent.mkdir(parents=True)
    make_transactions(4000, 120).to_csv(cfg.paths.raw_path, index=False)
    preprocess.run(cfg, ("time",))
    train_supervised.run(cfg, method="time")
    optimize_threshold.run(cfg, method="time")
    drift.run(cfg, "time")
    return cfg


def _raw(n: int = 8, seed: int = 3):
    rows = make_transactions(2000, 40, seed).drop(columns="Class").head(n)
    return rows.to_dict("records")


# ---- schemas ---------------------------------------------------------------------------------------------------------------------
def test_transaction_schema_accepts_and_rejects() -> None:
    Transaction = schemas.transaction_model()
    good = _raw(1)[0]
    assert Transaction.model_validate(good).Amount == good["Amount"]
    for bad in ({**good, "Amount": -1.0}, {**good, "V3": float("nan")}, {**good, "extra": 1.0}, {k: v for k, v in good.items() if k != "V9"}, {**good, "Time": "soon"}):
        with pytest.raises(ValidationError):
            Transaction.model_validate(bad)


def test_batch_schema_limits_and_records() -> None:
    Transaction = schemas.transaction_model()
    Batch = schemas.batch_model(Transaction, ServiceConfig(max_batch=3))
    rows = _raw(4)
    assert len(schemas.to_records(Batch.model_validate({"transactions": rows[:3]}))) == 3
    for bad in ([], rows):
        with pytest.raises(ValidationError):
            Batch.model_validate({"transactions": bad})
    assert schemas.to_records(Transaction.model_validate(rows[0]))[0]["V1"] == rows[0]["V1"]


# ---- feature building and scoring --------------------------------------------------------------------------------------------------
def test_build_features_matches_preprocess(project: ProjectConfig) -> None:
    frame = make_transactions(600, 15, seed=5)
    scaler = joblib.load(project.paths.scaler_path("time"))
    expected = preprocess.transform(frame, scaler, project)[preprocess.feature_names()]
    got = scoring.build_features(frame.drop(columns="Class"), scaler, project)
    pd.testing.assert_frame_equal(got, expected)
    with pytest.raises(ValueError, match="missing fields"):
        scoring.build_features(frame.drop(columns=["V5", "Class"]), scaler, project)
    broken = frame.drop(columns="Class").copy()
    broken.loc[broken.index[0], "Amount"] = np.nan
    with pytest.raises(ValueError, match="NaN"):
        scoring.build_features(broken, scaler, project)


def test_scorer_scores_single_and_batch_consistently(project: ProjectConfig) -> None:
    scorer = scoring.Scorer.load(project)
    rows = _raw(10)
    batch = scorer.score_records(rows)
    singles = [scorer.score_records([r])[0] for r in rows]
    assert [b["tier"] for b in batch] == [s["tier"] for s in singles]
    assert np.allclose([b["score"] for b in batch], [s["score"] for s in singles], atol=1e-6)
    assert all(0 <= r["score"] <= 1 and r["tier"] in ("low", "medium", "high") and r["latency_ms"] > 0 for r in batch)
    assert {r["action"] for r in batch} <= set(scorer.policy["tiers"].values())
    assert "reasons" not in batch[0]


def test_scorer_tiers_follow_the_policy_thresholds(project: ProjectConfig) -> None:
    scorer = scoring.Scorer.load(project)
    results = scorer.score_records(_raw(40))
    lo, hi = scorer.policy["otp_threshold"], scorer.policy["high_threshold"]
    for r in results:
        assert r["tier"] == ("high" if r["score"] >= hi else "medium" if r["score"] >= lo else "low")


def test_scorer_reasons_and_drift_monitor(project: ProjectConfig) -> None:
    scorer = scoring.Scorer.load(project)
    reasons = scorer.score_records(_raw(3), explain=True)[1]["reasons"]
    assert len(reasons) == SERVICE.reasons and {"feature", "contribution", "value", "effect"} <= set(reasons[0])
    assert scorer.monitor is not None and scorer.monitor.report()["overall"] == "not_enough_data"
    scorer.score_records(_raw(60, seed=9))
    assert scorer.monitor.report()["overall"] in ("ok", "warn", "alert")
    assert scorer.info()["drift_monitoring"] is True


def test_scorer_errors(project: ProjectConfig, tmp_path: Path) -> None:
    scorer = scoring.Scorer.load(project)
    with pytest.raises(ValueError, match="no transactions"):
        scorer.score_records([])
    with pytest.raises(ValueError, match="missing fields"):
        scorer.score_records([{"Time": 1.0}])
    good = dict(scorer.policy)
    for key in ("tiers", "otp_threshold"):
        with pytest.raises(ValueError, match="missing"):
            scoring.Scorer(scorer.model, scorer.scaler, {k: v for k, v in good.items() if k != key}, project)
    with pytest.raises(ValueError, match="different features"):
        scoring.Scorer(scorer.model, scorer.scaler, {**good, "features": good["features"][:-1]}, project)
    with pytest.raises(ValueError, match="must not exceed"):
        scoring.Scorer(scorer.model, scorer.scaler, {**good, "otp_threshold": 0.9, "high_threshold": 0.1}, project)
    with pytest.raises(FileNotFoundError, match="Step 5"):
        scoring.Scorer.load(replace(project, paths=replace(project.paths, root=tmp_path)))


def test_scorer_switches_forests_to_single_thread(project: ProjectConfig) -> None:
    from sklearn.ensemble import RandomForestClassifier
    scorer = scoring.Scorer.load(project)
    forest = RandomForestClassifier(5, n_jobs=-1, random_state=0).fit(np.random.default_rng(0).normal(size=(50, 30)), np.arange(50) % 2)
    scoring.Scorer(forest, scorer.scaler, scorer.policy, project)
    assert forest.get_params()["n_jobs"] == 1


def test_scorer_without_reference_runs_without_monitoring(project: ProjectConfig, tmp_path: Path) -> None:
    models = project.paths.models_path
    other = replace(project, paths=replace(project.paths, root=tmp_path))
    (tmp_path / "models").mkdir()
    for name in ("supervised_best.joblib", "decision_policy.json", "scaler_time.joblib"):
        (tmp_path / "models" / name).write_bytes((models / name).read_bytes())
    scorer = scoring.Scorer.load(other)
    assert scorer.monitor is None and scorer.score_records(_raw(2))[0]["score"] >= 0 and scorer.info()["drift_monitoring"] is False


# ---- explanations -----------------------------------------------------------------------------------------------------------------
def test_linear_contributions_add_up_to_the_log_odds(project: ProjectConfig) -> None:
    model = joblib.load(project.paths.models_path / "supervised_best.joblib")
    width = len(preprocess.feature_names())                                  # 29 or 30, depending on use_time
    X = np.random.default_rng(0).normal(size=(20, width))
    contrib, base, method = explain.contributions(model, X)
    assert method == "linear_coefficients" and contrib.shape == (20, width)
    assert np.allclose(base + contrib.sum(axis=1), model.decision_function(X))


def test_lightgbm_native_branch_uses_pred_contrib() -> None:
    class LGBMClassifier:
        def predict(self, X, pred_contrib=False):
            assert pred_contrib
            return np.hstack([np.ones((len(X), 3)), np.full((len(X), 1), -2.0)])
    LGBMClassifier.__module__ = "lightgbm.sklearn"
    contrib, base, method = explain.contributions(LGBMClassifier(), np.zeros((4, 3)))
    assert contrib.shape == (4, 3) and base == -2.0 and method == "lightgbm_native_treeshap"


def test_xgboost_native_branch_uses_booster_contribs(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = types.ModuleType("xgboost")
    fake.DMatrix = lambda X: X
    monkeypatch.setitem(sys.modules, "xgboost", fake)

    class Booster:
        def predict(self, data, pred_contribs=False):
            assert pred_contribs
            return np.hstack([np.full((len(data), 2), 0.5), np.full((len(data), 1), 0.1)])

    class XGBClassifier:
        def get_booster(self):
            return Booster()
    XGBClassifier.__module__ = "xgboost.sklearn"
    contrib, base, method = explain.contributions(XGBClassifier(), np.zeros((3, 2)))
    assert contrib.shape == (3, 2) and base == pytest.approx(0.1) and method == "xgboost_native_treeshap"


def test_forest_contributions_need_shap_and_add_up() -> None:
    from sklearn.ensemble import RandomForestClassifier
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 4))
    y = (X[:, 0] + rng.normal(scale=0.3, size=300) > 1).astype(int)
    forest = RandomForestClassifier(20, max_depth=4, random_state=0).fit(X, y)
    try:
        import shap  # noqa: F401
    except ImportError:
        with pytest.raises(ImportError, match="shap"):
            explain.contributions(forest, X[:5])
        return
    contrib, base, method = explain.contributions(forest, X[:20])
    assert method == "shap_tree_explainer" and contrib.shape == (20, 4)
    assert np.allclose(base + contrib.sum(axis=1), forest.predict_proba(X[:20])[:, 1], atol=1e-6)


def test_contribution_input_validation() -> None:
    with pytest.raises(ValueError):
        explain.contributions(object(), np.zeros((2, 2)), method="lime")
    with pytest.raises(ValueError):
        explain.contributions(object(), np.zeros(3))


def test_top_reasons_and_global_importance() -> None:
    got = explain.top_reasons([0.1, -0.9, 0.3], ["a", "b", "c"], [1.0, 2.0, 3.0], 2)
    assert [g["feature"] for g in got] == ["b", "c"] and got[0]["effect"] == "lowers fraud risk" and got[1]["effect"] == "raises fraud risk"
    with pytest.raises(ValueError):
        explain.top_reasons([1, 2], ["a"], [1, 2])
    contrib = np.array([[1.0, -3.0], [1.0, 3.0], [5.0, 0.0]])
    table = explain.global_importance(contrib, ["f1", "f2"], np.array([0, 0, 1]))
    assert table.iloc[0]["feature"] == "f1" and table["share"].sum() == pytest.approx(1.0) and table.set_index("feature").loc["f1", "mean_abs_fraud"] == 5.0
    assert "mean_abs_fraud" not in explain.global_importance(contrib, ["f1", "f2"]).columns
    with pytest.raises(ValueError):
        explain.global_importance(contrib, ["only_one"])


def test_sample_rows_keeps_every_fraud() -> None:
    frame = make_transactions(1000, 20)
    sample = explain.sample_rows(frame, "Class", 100)
    assert len(sample) == 100 and sample["Class"].sum() == 20 and explain.sample_rows(frame, "Class", 10)["Class"].sum() == 20


def test_explain_run_writes_reports(project: ProjectConfig) -> None:
    summary = explain.run(project)
    assert summary["method"] == "linear_coefficients" and set(summary["examples"]) == {"caught_fraud", "missed_fraud", "false_alarm"}
    reports = project.paths.reports_path
    assert (reports / "shap_importance.csv").is_file() and (reports / "explanations.json").is_file()
    assert (reports / "figures" / "shap_importance.png").is_file() and (reports / "figures" / "example_explanations.png").is_file()
    assert summary["examples"]["caught_fraud"]["score"] >= summary["examples"]["missed_fraud"]["score"]


# ---- drift --------------------------------------------------------------------------------------------------------------------------
def test_psi_basic_properties() -> None:
    same = [0.25, 0.25, 0.25, 0.25]
    assert drift.psi(same, same) == pytest.approx(0.0)
    assert drift.psi(same, [0.7, 0.1, 0.1, 0.1]) > 0.25
    assert drift.psi([0.5, 0.5], [0.0, 1.0]) > 0                         # empty bins do not blow up
    with pytest.raises(ValueError):
        drift.psi([0.5, 0.5], [1.0])


def test_compare_detects_a_shift_and_accepts_the_same_distribution() -> None:
    rng = np.random.default_rng(1)
    ref_frame = pd.DataFrame(rng.normal(size=(5000, 3)), columns=["a", "b", "c"])
    ref = drift.build_reference(ref_frame, rng.beta(1, 20, 5000), 10)
    same = pd.DataFrame(rng.normal(size=(2000, 3)), columns=["a", "b", "c"])
    out = drift.compare(ref, same, rng.beta(1, 20, 2000), SERVICE)
    assert out["overall"] == "ok" and out["features_alert"] == 0
    shifted = same.assign(a=same["a"] + 2.0)
    out = drift.compare(ref, shifted, rng.beta(1, 20, 2000), SERVICE)
    table = out["table"].set_index("feature")
    assert table.loc["a", "status"] == "alert" and table.loc["b", "status"] == "ok" and out["table"].iloc[0]["feature"] == "a"
    assert table.loc["a", "mean_shift_in_std"] > 1.5
    with_clock = same.assign(Time_scaled=np.linspace(50, 60, len(same)))                  # a clock outside the reference range
    ref_clock = drift.build_reference(ref_frame.assign(Time_scaled=np.linspace(0, 1, len(ref_frame))), None, 10)
    assert "Time_scaled" not in set(drift.compare(ref_clock, with_clock, None, SERVICE)["table"]["feature"])
    score_drift = drift.compare(ref, same, rng.beta(5, 5, 2000), SERVICE)
    assert score_drift["overall"] == "alert"                              # a changed score distribution alone raises the alert


def test_compare_and_reference_errors() -> None:
    rng = np.random.default_rng(2)
    frame = pd.DataFrame(rng.normal(size=(100, 2)), columns=["a", "b"])
    ref = drift.build_reference(frame, None, 5)
    with pytest.raises(ValueError, match="at least"):
        drift.compare(ref, frame.head(10), None, SERVICE)
    with pytest.raises(ValueError, match="missing features"):
        drift.compare(ref, frame[["a"]].sample(60, replace=True, random_state=0), None, SERVICE)
    with pytest.raises(ValueError):
        drift.build_reference(frame.head(5), None, 5)
    with pytest.raises(ValueError):
        drift.build_reference(frame.assign(a=np.nan), None, 5)
    constant = drift.build_reference(pd.DataFrame({"a": np.ones(100)}), None, 5)
    assert len(constant["features"]["a"]["shares"]) >= 1


def test_monitor_buffer_is_bounded_and_validates_shape() -> None:
    rng = np.random.default_rng(4)
    frame = pd.DataFrame(rng.normal(size=(500, 2)), columns=["a", "b"])
    monitor = drift.DriftMonitor(drift.build_reference(frame, rng.random(500), 5), SERVICE)
    monitor.add(rng.normal(size=(900, 2)), rng.random(900))
    assert monitor.report()["rows"] == SERVICE.monitor_buffer          # 900 added, only the newest 800 kept
    with pytest.raises(ValueError):
        monitor.add(np.zeros((3, 5)), [0.1, 0.2, 0.3])
    with pytest.raises(ValueError):
        monitor.add(np.zeros((3, 2)), [0.1])


def test_drift_run_writes_reference_and_report(project: ProjectConfig) -> None:
    reference = json.loads((project.paths.models_path / SERVICE.reference_name).read_text())
    assert set(reference["features"]) == set(preprocess.feature_names()) | {drift.SCORE_KEY}
    summary = json.loads((project.paths.reports_path / "drift_report.json").read_text())
    assert {"validation", "test (all)", "test slice 1 of 4", "test slice 4 of 4"} <= set(summary["windows"])
    assert all(w["overall"] in ("ok", "warn", "alert") for w in summary["windows"].values())
    assert (project.paths.reports_path / "drift_report.csv").is_file() and (project.paths.reports_path / "figures" / "drift_psi.png").is_file()


def test_service_config_validation() -> None:
    for bad in (dict(latency_budget_ms=0), dict(max_batch=0), dict(benchmark_requests=5), dict(psi_bins=1), dict(psi_warn=0.3, psi_alert=0.2),
                dict(alert_feature_share=0), dict(min_window=5), dict(min_window=1000, monitor_buffer=500)):
        with pytest.raises(ValueError):
            ServiceConfig(**bad)


# ---- benchmark ----------------------------------------------------------------------------------------------------------------------
def test_percentiles_and_time_calls() -> None:
    got = benchmark.percentiles(list(range(1, 101)))
    assert got["p50"] == pytest.approx(50.5) and got["max"] == 100 and got["p50"] <= got["p95"] <= got["p99"] <= got["max"]
    with pytest.raises(ValueError):
        benchmark.percentiles([])
    calls = []
    out = benchmark.time_calls(calls.append, [{"i": i} for i in range(10)], 3)
    assert len(out) == 7 and len(calls) == 10 and min(out) >= 0
    with pytest.raises(ValueError):
        benchmark.time_calls(calls.append, [{}] * 3, 3)


def test_benchmark_run_reports_budget(project: ProjectConfig) -> None:
    result = benchmark.run(project)
    assert result["requests"] == SERVICE.benchmark_requests and result["scoring"]["p99"] > 0
    assert result["within_budget"] is True and result["worst_p99_ms"] <= SERVICE.latency_budget_ms
    assert "http_round_trip" in result and (project.paths.reports_path / "latency_report.json").is_file()
    tight = replace(project, service=replace(SERVICE, latency_budget_ms=0.0001))
    assert benchmark.run(tight)["within_budget"] is False


# ---- smoke test against a stub server ------------------------------------------------------------------------------------------------
class _Stub(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:
        pass

    def _send(self, body, code: int = 200) -> None:
        data = json.dumps(body).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self) -> None:
        if self.path == "/health":
            self._send({"status": "ok", "model": "stub/model", "drift_monitoring": True})
        elif self.path == "/drift":
            self._send({"overall": "ok"})
        else:
            self._send({"detail": "nope"}, 404)

    def do_POST(self) -> None:
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        result = {"score": 0.5, "tier": "medium", "action": "OTP", "reasons": [{"feature": "V1"}]}
        if self.path == "/score/batch":
            self._send({"results": [result] * len(body["transactions"])})
        elif self.path.startswith("/score"):
            self._send(result)
        else:
            self._send({"detail": "nope"}, 404)


@pytest.fixture()
def stub_url():
    server = HTTPServer(("127.0.0.1", 0), _Stub)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_port}"
    server.shutdown()
    server.server_close()


def test_smoke_test_against_a_stub_service(stub_url: str) -> None:
    summary = smoke_test.run(stub_url, _raw(3), budget_ms=5000)
    assert summary["model"] == "stub/model" and summary["tiers"] == ["medium"] * 3 and summary["drift"] == "ok" and summary["within_budget"] is True
    assert summary["reasons_for_first"] == [{"feature": "V1"}]
    with pytest.raises(RuntimeError, match="404"):
        smoke_test.call(stub_url + "/missing")
    with pytest.raises(RuntimeError, match="cannot reach"):
        smoke_test.call("http://127.0.0.1:9/health", timeout=1)


def test_smoke_test_main_explains_an_unreachable_service(tmp_path: Path, capsys: pytest.CaptureFixture) -> None:
    cfg = replace(ProjectConfig(), paths=replace(ProjectConfig().paths, root=tmp_path))
    parts, scaler = preprocess.build_split(make_transactions(3000, 90), "time", cfg)
    preprocess.write_split(parts, scaler, "time", cfg)
    code = smoke_test.main(["--url", "http://127.0.0.1:9", "--root", str(tmp_path), "--count", "2"])
    assert code == 2 and "docker run" in capsys.readouterr().err


# ---- the results document -------------------------------------------------------------------------------------------------------------
def test_report_with_available_reports_and_gaps(project: ProjectConfig) -> None:
    explain.run(project)
    benchmark.run(project)
    path = report.run(project)
    text = path.read_text(encoding="utf-8")
    assert path.name == "RESULTS.md" and "## Scorecard" in text and "## 5. Thresholds, risk tiers and cost-benefit" in text
    assert "logistic_regression/baseline" in text and "three tiers" in text and "Latency against the 100 ms budget" in text
    assert "python -m fraud.run_imbalance" in text and "python -m fraud.define" in text            # not run in this fixture: says how to create them
    assert "%" in text and "0.6939" not in text


def test_report_on_an_empty_project(tmp_path: Path) -> None:
    base = ProjectConfig()
    text = report.build(replace(base, paths=replace(base.paths, root=tmp_path)))
    assert text.count("Not available yet") >= 5 and "## Limitations" in text and "not available yet" in text
    assert report.run(replace(base, paths=replace(base.paths, root=tmp_path))).is_file()


# ---- requirement pinning ------------------------------------------------------------------------------------------------------------
def test_pin_requirements() -> None:
    assert "lightgbm" in pin_requirements.packages_for("lightgbm/smote") and "xgboost" in pin_requirements.packages_for("xgboost/baseline")
    assert "shap" in pin_requirements.packages_for("random_forest/smote") and "lightgbm" not in pin_requirements.packages_for("logistic_regression/baseline")
    pins = pin_requirements.pinned(["numpy", "pandas"])
    assert all("==" in p for p in pins) and pins[0].startswith("numpy==")
    with pytest.raises(RuntimeError, match="not-a-real-package"):
        pin_requirements.pinned(["numpy", "not-a-real-package"])


# ---- the HTTP API (runs where fastapi and httpx are installed) -------------------------------------------------------------------------
@pytest.fixture()
def client(project: ProjectConfig):
    pytest.importorskip("fastapi")
    pytest.importorskip("httpx")
    from fastapi.testclient import TestClient
    from fraud import api
    return TestClient(api.create_app(scoring.Scorer.load(project)))


def test_api_health_policy_and_score(client) -> None:
    health = client.get("/health").json()
    assert health["status"] == "ok" and client.get("/policy").json()["tiers"]["high"].startswith("block")
    response = client.post("/score", json=_raw(1)[0])
    body = response.json()
    assert response.status_code == 200 and body["tier"] in ("low", "medium", "high") and "X-Process-Time-ms" in response.headers
    assert "reasons" in client.post("/score?explain=true", json=_raw(1)[0]).json()


def test_api_rejects_bad_input_and_scores_batches(client) -> None:
    good = _raw(1)[0]
    assert client.post("/score", json={**good, "Amount": -5}).status_code == 422
    assert client.post("/score", json={k: v for k, v in good.items() if k != "V1"}).status_code == 422
    assert client.post("/score/batch", json={"transactions": []}).status_code == 422
    batch = client.post("/score/batch", json={"transactions": _raw(5)})
    assert batch.status_code == 200 and len(batch.json()["results"]) == 5
    assert client.get("/drift").json()["overall"] in ("not_enough_data", "ok", "warn", "alert")
