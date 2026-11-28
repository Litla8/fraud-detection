"""Request validation for the scoring service (pydantic only, so it is testable without FastAPI)."""
from __future__ import annotations

from typing import Any, List, Type

from pydantic import BaseModel, ConfigDict, Field, create_model

from fraud.config import DataConfig, ServiceConfig


def transaction_model(cfg: DataConfig = DataConfig()) -> Type[BaseModel]:
    """A model with one required, finite float per input column (``Time``, ``V1``-``V28``, ``Amount``); unknown fields are rejected.

    ``Amount`` must be non-negative. Time is seconds since the first transaction, exactly as in the Kaggle file.
    """
    fields: dict = {cfg.time_column: (float, Field(allow_inf_nan=False)),
                    cfg.amount_column: (float, Field(ge=0, allow_inf_nan=False))}
    for column in cfg.pca_columns:
        fields[column] = (float, Field(allow_inf_nan=False))
    return create_model("Transaction", __config__=ConfigDict(extra="forbid"), **fields)


def batch_model(transaction: Type[BaseModel], service: ServiceConfig = ServiceConfig()) -> Type[BaseModel]:
    """A batch of 1 to ``service.max_batch`` transactions."""
    return create_model("TransactionBatch", __config__=ConfigDict(extra="forbid"),
                        transactions=(List[transaction], Field(min_length=1, max_length=service.max_batch)))  # type: ignore[valid-type]


def to_records(model_instance: Any) -> List[dict]:
    """Plain dicts from a validated single transaction or batch."""
    if hasattr(model_instance, "transactions"):
        return [t.model_dump() for t in model_instance.transactions]
    return [model_instance.model_dump()]
