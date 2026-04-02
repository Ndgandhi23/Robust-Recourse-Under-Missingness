import os
import numpy as np
import pandas as pd


FEATURE_NAMES = [
    "Pregnancies",
    "Glucose",
    "BloodPressure",
    "SkinThickness",
    "Insulin",
    "BMI",
    "DiabetesPedigreeFunction",
    "Age",
]

# zeros in these columns are physiologically impossible — treat as missing
_MISSING_ZERO_NAMES = {"Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI"}
MISSING_ZERO_COLS = [i for i, n in enumerate(FEATURE_NAMES) if n in _MISSING_ZERO_NAMES]


def load_diabetes():
    local_path = "diabetes.csv"
    if not os.path.exists(local_path):
        raise FileNotFoundError("run download_data.py first")

    df = pd.read_csv(local_path)
    X_raw = df[FEATURE_NAMES].values.astype(float)
    n, d  = X_raw.shape

    # detect missingness
    Xi = np.zeros((n, d), dtype=int)
    for j in MISSING_ZERO_COLS:
        missing_mask        = X_raw[:, j] == 0.0
        Xi[missing_mask, j] = 1
        X_raw[missing_mask, j] = np.nan

    # standardize on observed values only so missing zeros don't skew stats
    col_means = np.nanmean(X_raw, axis=0)
    col_stds  = np.nanstd(X_raw,  axis=0)
    col_stds[col_stds == 0] = 1.0

    X_scaled = np.zeros((n, d))
    for j in range(d):
        obs_mask = Xi[:, j] == 0
        X_scaled[obs_mask, j] = (X_raw[obs_mask, j] - col_means[j]) / col_stds[j]

    X   = X_scaled
    y   = np.where(df["Outcome"].values == 1, 1, -1)
    Phi = build_phi(X, Xi)

    print(f"n={n}  d={d}  diabetic={( y==1).sum()}  not={( y==-1).sum()}")
    for j, name in enumerate(FEATURE_NAMES):
        pct = 100 * Xi[:, j].mean()
        if pct > 0:
            print(f"  {name:<30} {Xi[:,j].sum():3d} missing ({pct:.1f}%)")

    return X, Xi, y, Phi, FEATURE_NAMES, col_means, col_stds


def build_phi(X, Xi):
    # phi(x, xi) = [x, xi]  shape (n, 2d)
    return np.hstack([X, Xi])


def train_test_split(n, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    return idx[n_test:], idx[:n_test]


if __name__ == "__main__":
    X, Xi, y, Phi, names, means, stds = load_diabetes()
    train_idx, test_idx = train_test_split(len(y))
    print(f"train={len(train_idx)}  test={len(test_idx)}  phi shape={Phi.shape}")