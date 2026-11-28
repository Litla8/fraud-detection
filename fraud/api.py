"""FastAPI scoring service.

Note: this module intentionally does NOT use ``from __future__ import annotations``. FastAPI reads the endpoint annotations at runtime,
and the request models are created dynamically, so the annotations must be real classes, not strings.
"""
import logging
import os
import time
from dataclasses import replace
from pathlib import Path
from typing import Optional

from fastapi import FastAPI, HTTPException, Request

from fraud import schemas
from fraud.config import ProjectConfig
from fraud.logging_utils import configure_logging
from fraud.scoring import Scorer

LOGGER = logging.getLogger(__name__)


def create_app(scorer: Scorer) -> FastAPI:
    """The service around an already-loaded scorer (tests pass a scorer built from synthetic data)."""
    cfg = scorer.cfg
    Transaction = schemas.transaction_model(cfg.data)
    Batch = schemas.batch_model(Transaction, cfg.service)
    app = FastAPI(title="Credit card fraud scoring", version="1.0",
                  description="Scores a card transaction and returns a risk tier: low (approve), medium (OTP check), high (block and review).")

    @app.middleware("http")
    async def add_timing_header(request: Request, call_next):
        start = time.perf_counter()
        response = await call_next(request)
        response.headers["X-Process-Time-ms"] = f"{(time.perf_counter() - start) * 1000:.2f}"
        return response

    @app.get("/health")
    def health():
        return {"status": "ok", **scorer.info()}

    @app.get("/policy")
    def get_policy():
        return scorer.info()

    @app.post("/score")
    def score(transaction: Transaction, explain: bool = False):
        try:
            return scorer.score_records(schemas.to_records(transaction), explain)[0]
        except ImportError as error:
            raise HTTPException(status_code=501, detail=str(error))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

    @app.post("/score/batch")
    def score_batch(batch: Batch, explain: bool = False):
        try:
            return {"results": scorer.score_records(schemas.to_records(batch), explain)}
        except ImportError as error:
            raise HTTPException(status_code=501, detail=str(error))
        except ValueError as error:
            raise HTTPException(status_code=422, detail=str(error))

    @app.get("/drift")
    def drift():
        if scorer.monitor is None:
            raise HTTPException(status_code=404, detail="drift monitoring is off: build models/drift_reference.json with `python -m fraud.drift`")
        return scorer.monitor.report()

    return app


def create_default_app() -> FastAPI:
    """Entry point for uvicorn (``uvicorn fraud.api:create_default_app --factory``): loads everything from ``$FRAUD_ROOT`` (default: current folder)."""
    configure_logging(os.environ.get("LOG_LEVEL", "INFO"))
    base = ProjectConfig()
    cfg = replace(base, paths=replace(base.paths, root=Path(os.environ.get("FRAUD_ROOT", "."))))
    scorer = Scorer.load(cfg)
    LOGGER.info("loaded %s; OTP from %.4f, block from %.4f", scorer.policy["model_variant"], scorer.policy["otp_threshold"], scorer.policy["high_threshold"])
    return create_app(scorer)
