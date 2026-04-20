from dataclasses import dataclass
from typing import Callable, List, Optional


@dataclass
class DatasetConfig:
    """Everything the pipeline needs from a dataset.

    The ``load`` callable must return:
        (X, Xi, y, Phi, feature_names, col_means, col_stds)
    where
        X   : (n, d) float   — standardised feature matrix (0 for missing entries)
        Xi  : (n, d) int     — missingness indicator (1 = missing)
        y   : (n,)   int     — labels: +1 = favorable, -1 = unfavorable
        Phi : (n, 2d) float  — build_phi(X, Xi)
        feature_names : list[str]
        col_means, col_stds : (d,) float — for unstandardising
    """
    name: str                           # slug: "diabetes", "german", …
    display_name: str                   # for figure titles
    feature_names: List[str]
    mutable_cols: List[int]
    immutable_cols: List[int]
    load: Callable                      # () -> 7-tuple above
    download: Optional[Callable] = None # () -> None, downloads raw data

    # per-dataset hyperparameter defaults
    default_C: float = 1.0          # logistic regression inverse-regularization
    default_kappa: float = 0.5      # per-feature reveal cost
    default_delta_max: float = 2.0  # max feature change (in standardized units)

    # label semantics for printing
    positive_label: str = "approved"
    negative_label: str = "denied"

    def __post_init__(self):
        d = len(self.feature_names)
        mut = set(self.mutable_cols)
        imm = set(self.immutable_cols)

        out_of_range = [i for i in mut | imm if not (0 <= i < d)]
        if out_of_range:
            raise ValueError(
                f"[{self.name}] indices out of range [0, {d}): {out_of_range}"
            )

        overlap = mut & imm
        if overlap:
            names = [self.feature_names[i] for i in sorted(overlap)]
            raise ValueError(
                f"[{self.name}] features in BOTH mutable_cols and "
                f"immutable_cols: {names}"
            )

        missing = set(range(d)) - (mut | imm)
        if missing:
            names = [self.feature_names[i] for i in sorted(missing)]
            raise ValueError(
                f"[{self.name}] features in NEITHER mutable_cols nor "
                f"immutable_cols: {names}"
            )
