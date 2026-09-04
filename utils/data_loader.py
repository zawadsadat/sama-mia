"""
Data loader router.
Loads the correct dataset based on config.DATASET.
"""

import numpy as np
import torch
from sklearn.datasets import load_breast_cancer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from utils.config import DATASET, TEST_SIZE


# Datasets whose imputation and standardization are fit inside each
# train/eval split rather than once over the whole file. Fitting a scaler on
# all rows lets held-out data influence the released model, which both
# inflates reported utility and puts a non-private, data-dependent step
# outside the DP-SGD mechanism. Add dataset names here as each loader is
# converted; anything not listed keeps the original whole-file behaviour.
PER_SPLIT_SCALING = {"breast_cancer", "heart_disease",
                     "cardiotocography", "diabetes_hospital"}

_RAW_CACHE = {}


def scales_per_split(dataset=None):
    """True if the named dataset defers scaling to the per-model split."""
    return (dataset or DATASET) in PER_SPLIT_SCALING


def fit_apply_scaler(X_train, X_eval):
    """
    Impute and standardize using training-split statistics only.

    Column medians and the StandardScaler are fit on X_train and applied to
    both halves, so no held-out row influences either.

    Returns:
        X_train_t, X_eval_t (np.float32), the fitted scaler, and the median
        vector, so the same transform can be replayed later (for example
        during white-box gradient extraction).
    """
    X_train = np.asarray(X_train, dtype=np.float64)
    X_eval = np.asarray(X_eval, dtype=np.float64)

    with np.errstate(all="ignore"):
        med = np.nanmedian(X_train, axis=0)
    med = np.where(np.isnan(med), 0.0, med)          # all-NaN column
    X_train = np.where(np.isnan(X_train), med, X_train)
    X_eval = np.where(np.isnan(X_eval), med, X_eval)

    scaler = StandardScaler().fit(X_train)
    return (scaler.transform(X_train).astype(np.float32),
            scaler.transform(X_eval).astype(np.float32),
            scaler, med)


def get_raw_data(scale=True):
    """
    Load and return (X, y) for the configured dataset.

    scale=True  (default) returns features already imputed and standardized
                over the whole file, preserving the original behaviour for
                callers that only need shapes, labels or metadata.
    scale=False returns raw features with missing values left as NaN, for
                callers that fit the transform inside their own split via
                fit_apply_scaler(). Only loaders listed in PER_SPLIT_SCALING
                honour scale=False; the others ignore it.

    Results are cached per (dataset, scale) for the life of the process.
    Several scripts call this three times per run -- via get_dataset_info(),
    via get_input_dim(), and for the data itself -- which would otherwise
    mean three downloads or three full preprocessing passes. A copy is
    returned so callers cannot mutate the cache.
    """
    key = (DATASET, bool(scale))
    if key not in _RAW_CACHE:
        _RAW_CACHE[key] = _load_raw(scale)
    X, y = _RAW_CACHE[key]
    return X.copy(), y.copy()


def _load_raw(scale):
    """Uncached dispatch to the per-dataset loader."""
    if DATASET == "breast_cancer":
        return _load_breast_cancer(scale=scale)
    elif DATASET == "heart_disease":
        from utils.data_loader_heart_disease import load_heart_disease
        return load_heart_disease(scale=scale)
    elif DATASET == "cardiotocography":
        from utils.data_loader_cardiotocography import load_cardiotocography
        return load_cardiotocography(scale=scale)
    elif DATASET == "diabetes_hospital":
        # NOTE: load_diabetes_hospital() returns FOUR torch tensors
        # (X_train, X_test, y_train, y_test), unlike every other loader here,
        # which returns two numpy arrays (X, y). Routing to it broke the
        # get_raw_data() contract and crashed get_dataset_info()/get_input_dim()
        # on this dataset. get_raw_diabetes_hospital() returns (X, y) numpy,
        # matching the other three loaders.
        from utils.data_loader_diabetes_hospital import get_raw_diabetes_hospital
        return get_raw_diabetes_hospital(scale=scale)
    else:
        raise ValueError(f"Unknown dataset: {DATASET}. "
                         f"Options: breast_cancer, heart_disease, cardiotocography, diabetes_hospital")


def _load_breast_cancer(scale=True):
    """Load Breast Cancer Wisconsin dataset."""
    data = load_breast_cancer()
    X = data.data.astype(np.float32)
    y = data.target.astype(np.float32)

    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)

    print(f"Breast Cancer Wisconsin: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Positive rate: {y.mean():.1%}")

    return X.astype(np.float32), y.astype(np.float32)


def load_data(test_size=None, random_state=42):
    """
    Load the active dataset as torch tensors.

    For datasets in PER_SPLIT_SCALING, imputation and standardization are fit
    on the training half only and applied to both.

    Returns:
        X_train, X_test, y_train, y_test as float32 tensors
    """
    if test_size is None:
        test_size = TEST_SIZE

    per_split = scales_per_split()
    X, y = get_raw_data(scale=not per_split)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if per_split:
        X_train, X_test, _, _ = fit_apply_scaler(X_train, X_test)

    return (
        torch.tensor(X_train, dtype=torch.float32),
        torch.tensor(X_test, dtype=torch.float32),
        torch.tensor(y_train, dtype=torch.float32),
        torch.tensor(y_test, dtype=torch.float32),
    )


def get_input_dim():
    """Return the feature dimensionality of the active dataset."""
    X, _ = get_raw_data(scale=True)
    return X.shape[1]


def get_dataset_info():
    """Return a dict with dataset metadata for logging."""
    X, y = get_raw_data(scale=True)
    return {
        "name": DATASET,
        "samples": len(y),
        "features": X.shape[1],
        "positive_rate": y.mean(),
    }


if __name__ == "__main__":
    X, y = get_raw_data()
    print(f"\nDataset: {DATASET}")
    print(f"X shape: {X.shape}, y shape: {y.shape}")
    print(f"TEST_SIZE: {TEST_SIZE}")
    print(f"Train samples: ~{int(len(X) * (1 - TEST_SIZE))}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
