import os
import urllib.request
import numpy as np
import pandas as pd
from .data_utils import build_phi, train_test_split
from .dataset_config import DatasetConfig


# Standard FICO HELOC feature names (mapped from HuggingFace CSV columns)
FEATURE_NAMES = [
    "ExternalRiskEstimate",
    "MSinceOldestTradeOpen",
    "MSinceMostRecentTradeOpen",
    "AverageMInFile",
    "NumSatisfactoryTrades",
    "NumTrades60Ever2DerogPubRec",
    "NumTrades90Ever2DerogPubRec",
    "PercentTradesNeverDelq",
    "MSinceMostRecentDelq",
    "MaxDelq2PublicRecLast12M",
    "MaxDelqEver",
    "NumTotalTrades",
    "NumTradesOpeninLast12M",
    "PercentInstallTrades",
    "MSinceMostRecentInqexcl7days",
    "NumInqLast6M",
    "NumInqLast6Mexcl7days",
    "NetFractionRevolvingBurden",
    "NetFractionInstallBurden",
    "NumRevolvingTradesWBalance",
    "NumInstallTradesWBalance",
    "NumBank2NatlTradesWHighUtilization",
    "PercentTradesWBalance",
]

# Column names as they appear in the HuggingFace CSV (same order as FEATURE_NAMES)
_HF_COLS = [
    "estimate_of_risk",
    "months_since_first_trade",
    "months_since_last_trade",
    "average_duration_of_resolution",
    "number_of_satisfactory_trades",
    "nr_trades_insolvent_for_over_60_days",
    "nr_trades_insolvent_for_over_90_days",
    "percentage_of_legal_trades",
    "months_since_last_illegal_trade",
    "maximum_illegal_trades_over_last_year",
    "maximum_illegal_trades",
    "nr_total_trades",
    "nr_trades_initiated_in_last_year",
    "percentage_of_installment_trades",
    "months_since_last_inquiry_not_recent",
    "nr_inquiries_in_last_6_months",
    "nr_inquiries_in_last_6_months_not_recent",
    "net_fraction_of_revolving_burden",
    "net_fraction_of_installment_burden",
    "nr_revolving_trades_with_balance",
    "nr_installment_trades_with_balance",
    "nr_banks_with_high_ratio",
    "percentage_trades_with_balance",
]
_TARGET_COL = "is_at_risk"

# Special values that encode missingness in FICO HELOC:
#   -9  No Bureau Record or No Investigation
#   -8  No Usable/Valid Trades or Inquiries
#   -7  Condition not Met (e.g. no delinquency ever occurred)
_MISSING_SPECIAL = {-9, -8, -7}

# Features that cannot be changed by the individual (historical / external)
_IMMUTABLE_NAMES = {
    "ExternalRiskEstimate",           # external composite risk score
    "MSinceOldestTradeOpen",          # when first trade opened
    "AverageMInFile",                 # average history length
    "NumTrades60Ever2DerogPubRec",    # past delinquency count (ever)
    "NumTrades90Ever2DerogPubRec",    # past delinquency count (ever)
    "MSinceMostRecentDelq",           # timing of past delinquency
    "MaxDelq2PublicRecLast12M",       # max delinquency last 12 months
    "MaxDelqEver",                    # max delinquency ever
    "MSinceMostRecentInqexcl7days",   # timing of past inquiry
}
IMMUTABLE_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n in _IMMUTABLE_NAMES]
MUTABLE_COLS   = [i for i, n in enumerate(FEATURE_NAMES) if n not in _IMMUTABLE_NAMES]

_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), os.pardir, "data")
CSV_PATH  = os.path.join(_DATA_DIR, "fico.csv")

_URL = (
    "https://huggingface.co/datasets/mstz/heloc/"
    "resolve/main/risk/train.csv"
)


def load_fico():
    if not os.path.exists(CSV_PATH):
        raise FileNotFoundError("run download_data.py first")

    df = pd.read_csv(CSV_PATH)

    # Drop 588 "no bureau record" rows: these have -9 across all trade
    # features and represent individuals with no credit history at all.
    # Use NumTotalTrades (nr_total_trades) as sentinel — it is -9 only for
    # these rows and never -8 or -7.
    no_bureau = df["nr_total_trades"] == -9
    df = df[~no_bureau].reset_index(drop=True)

    X_raw = df[_HF_COLS].values.astype(float)
    n, d = X_raw.shape

    # Remaining -7, -8, -9 values are genuine missingness:
    #   -7  Condition not Met (e.g. never delinquent → months-since undefined)
    #   -8  No Usable/Valid Trades or Inquiries
    #   -9  (rare, e.g. 10 rows in ExternalRiskEstimate)
    Xi = np.zeros((n, d), dtype=int)
    for j in range(d):
        missing_mask = np.isin(X_raw[:, j], list(_MISSING_SPECIAL))
        Xi[missing_mask, j] = 1
        X_raw[missing_mask, j] = np.nan

    # standardize on observed values only so special codes don't skew stats
    col_means = np.nanmean(X_raw, axis=0)
    col_stds  = np.nanstd(X_raw,  axis=0)
    col_stds[col_stds == 0] = 1.0

    X_scaled = np.zeros((n, d))
    for j in range(d):
        obs_mask = Xi[:, j] == 0
        X_scaled[obs_mask, j] = (X_raw[obs_mask, j] - col_means[j]) / col_stds[j]

    X   = X_scaled
    # is_at_risk: 1 = bad (denied), 0 = good (approved)
    y   = np.where(df[_TARGET_COL].values == 1, -1, 1)
    Phi = build_phi(X, Xi)

    print(f"n={n}  d={d}  good={(y == 1).sum()}  bad={(y == -1).sum()}")
    for j, name in enumerate(FEATURE_NAMES):
        pct = 100 * Xi[:, j].mean()
        if pct > 0:
            print(f"  {name:<40} {Xi[:, j].sum():5d} missing ({pct:.1f}%)")

    return X, Xi, y, Phi, FEATURE_NAMES, col_means, col_stds


def download():
    if os.path.exists(CSV_PATH):
        print(f"  {CSV_PATH} already exists — skipping.")
        return
    os.makedirs(os.path.dirname(CSV_PATH), exist_ok=True)
    print(f"  downloading FICO HELOC dataset from HuggingFace...")
    urllib.request.urlretrieve(_URL, CSV_PATH)
    print(f"  saved {CSV_PATH}")


def get_config():
    return DatasetConfig(
        name="fico",
        display_name="FICO HELOC",
        feature_names=FEATURE_NAMES,
        mutable_cols=MUTABLE_COLS,
        immutable_cols=IMMUTABLE_COLS,
        load=load_fico,
        download=download,
        default_C=0.01,
        positive_label="good",
        negative_label="bad (at risk)",
    )


if __name__ == "__main__":
    X, Xi, y, Phi, names, means, stds = load_fico()
    train_idx, test_idx = train_test_split(len(y))
    print(f"train={len(train_idx)}  test={len(test_idx)}  phi shape={Phi.shape}")
