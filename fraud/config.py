"""Every setting in one place: paths, column names, the cost matrix, targets and split rules.

Frozen dataclasses, so nothing changes during a run; ``ProjectConfig`` groups them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

SEED = 42


@dataclass(frozen=True)
class PathsConfig:
    """Folders and the files derived from them (all relative to ``root``)."""

    root: Path = Path(".")
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    models_dir: str = "models"
    reports_dir: str = "reports"
    raw_name: str = "creditcard.csv"

    @property
    def raw_path(self) -> Path:
        """The Kaggle file."""
        return self.root / self.raw_dir / self.raw_name

    @property
    def processed_path(self) -> Path:
        return self.root / self.processed_dir

    @property
    def models_path(self) -> Path:
        return self.root / self.models_dir

    @property
    def reports_path(self) -> Path:
        return self.root / self.reports_dir

    def split_path(self, method: str) -> Path:
        """Folder holding train / validation / test for one split method (``time`` or ``stratified``)."""
        return self.processed_path / method

    def scaler_path(self, method: str) -> Path:
        """The fitted scaler for one split method."""
        return self.models_path / f"scaler_{method}.joblib"

    @property
    def problem_report_path(self) -> Path:
        """Where Step 1 writes the problem definition."""
        return self.reports_path / "problem_definition.json"


@dataclass(frozen=True)
class DataConfig:
    """Column names of the ULB credit-card file (V1-V28 are anonymised PCA components)."""

    label_column: str = "Class"             # 1 = fraud, 0 = genuine
    amount_column: str = "Amount"
    time_column: str = "Time"               # seconds since the first transaction (two days in total)
    n_components: int = 28
    expected_rows: int = 284_807            # used only to warn when the file looks truncated
    expected_frauds: int = 492

    @property
    def pca_columns(self) -> Tuple[str, ...]:
        return tuple(f"V{i}" for i in range(1, self.n_components + 1))

    @property
    def required_columns(self) -> Tuple[str, ...]:
        return (self.time_column, *self.pca_columns, self.amount_column, self.label_column)


@dataclass(frozen=True)
class CostConfig:
    """The cost matrix that drives threshold selection (Step 8). Currency units are the dataset's (euros).

    These are assumptions, not facts from the data: change them and every cost number in the project follows.

    * A **missed fraud** (false negative) loses the transaction amount (``fn_uses_amount``) plus a fixed chargeback and handling fee.
    * A **false alarm** (false positive) costs a fixed amount: a declined card, an OTP prompt, a support call, and some
      customers who stop using the card.
    * A **caught fraud** (true positive) still costs an analyst's review time, but its amount is saved.
    * Medium-risk transactions get an OTP check (cheap, small friction); high-risk ones are blocked and reviewed.
    """

    fn_fixed: float = 15.0
    fn_uses_amount: bool = True
    fp_cost: float = 3.0
    tp_review_cost: float = 1.0
    otp_cost: float = 0.25                  # one OTP verification of a genuine customer
    otp_fraud_catch_rate: float = 0.90      # share of fraud that fails or abandons the OTP check

    def __post_init__(self) -> None:
        """Fail loudly on impossible settings."""
        if min(self.fn_fixed, self.fp_cost, self.tp_review_cost, self.otp_cost) < 0:
            raise ValueError("costs must not be negative")
        if not 0 <= self.otp_fraud_catch_rate <= 1:
            raise ValueError("otp_fraud_catch_rate must be between 0 and 1")
        if self.fp_cost == 0 and self.fn_fixed == 0 and not self.fn_uses_amount:
            raise ValueError("at least one cost must be positive")


@dataclass(frozen=True)
class TargetConfig:
    """Success metrics from the project brief."""

    pr_auc: float = 0.80                    # area under the precision-recall curve (test set)
    recall: float = 0.80                    # recall ...
    precision_floor: float = 0.85           # ... at precision of at least this


@dataclass(frozen=True)
class SplitConfig:
    """Train / validation / test rules (used from Step 3)."""

    method: str = "time"                    # "time": oldest rows train, newest test (mimics production); "stratified": random
    test_fraction: float = 0.20
    val_fraction: float = 0.15              # taken from the end of the training period
    cv_folds: int = 5
    seed: int = SEED

    def __post_init__(self) -> None:
        """Fail loudly on impossible settings."""
        if self.method not in ("time", "stratified"):
            raise ValueError("method must be 'time' or 'stratified'")
        if not 0 < self.test_fraction < 0.5 or not 0 < self.val_fraction < 0.5:
            raise ValueError("fractions must be between 0 and 0.5")
        if self.cv_folds < 2:
            raise ValueError("cv_folds must be at least 2")


@dataclass(frozen=True)
class PreprocessConfig:
    """Step 3 settings: duplicates, scaling and the columns the models will see."""

    drop_duplicates: bool = True            # identical rows (same Time, V1-V28, Amount, Class) would leak between train and test
    scale_columns: Tuple[str, ...] = ("Amount", "Time")
    use_time: bool = True                   # keep scaled Time as a feature (Step 5 compares with and without it)
    hour_features: bool = False             # optional hour_sin / hour_cos from Time (two days only, so off by default)
    scaled_suffix: str = "_scaled"
    raw_amount_name: str = "Amount_raw"     # kept next to the features: the cost matrix needs the real amount
    raw_time_name: str = "Time_raw"
    id_column: str = "row_id"               # position in the original file, for tracing a row back
    file_suffix: str = ".pkl"               # pickle: no extra dependency (pyarrow) needed
    report_name: str = "preprocessing_report.json"


@dataclass(frozen=True)
class ProjectConfig:
    """Everything a pipeline step needs."""

    paths: PathsConfig = field(default_factory=PathsConfig)
    data: DataConfig = field(default_factory=DataConfig)
    costs: CostConfig = field(default_factory=CostConfig)
    targets: TargetConfig = field(default_factory=TargetConfig)
    split: SplitConfig = field(default_factory=SplitConfig)
    prep: PreprocessConfig = field(default_factory=PreprocessConfig)
