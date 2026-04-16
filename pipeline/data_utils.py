import numpy as np


def build_phi(X, Xi):
    """phi(x, xi) = [x, xi]  shape (n, 2d)"""
    return np.hstack([X, Xi])


def train_test_split(n, test_frac=0.2, seed=42):
    rng = np.random.RandomState(seed)
    idx = rng.permutation(n)
    n_test = int(test_frac * n)
    return idx[n_test:], idx[:n_test]


def stratified_kfold(y, n_splits=4, seed=42):
    """Yield (train_idx, test_idx) for each fold, preserving class balance."""
    from sklearn.model_selection import StratifiedKFold
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))


def stratified_train_val_split(y, val_frac=0.2, seed=42):
    """Stratified split into (train_idx, val_idx) preserving class balance."""
    rng = np.random.RandomState(seed)
    classes = np.unique(y)
    train_parts, val_parts = [], []
    for c in classes:
        idx = np.where(y == c)[0]
        rng.shuffle(idx)
        n_val = max(1, int(val_frac * len(idx)))
        val_parts.append(idx[:n_val])
        train_parts.append(idx[n_val:])
    return np.concatenate(train_parts), np.concatenate(val_parts)
