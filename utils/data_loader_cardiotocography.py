"""
Data loader for the Cardiotocography (CTG) dataset (UCI).

"""

import os

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

# Ground truth for the UCI CTG dataset, used to verify we loaded the right
# column. NSP: 1=Normal, 2=Suspect, 3=Pathologic.
EXPECTED_N = 2126
EXPECTED_NSP_COUNTS = {1: 1655, 2: 295, 3: 176}
EXPECTED_POS_RATE = (295 + 176) / 2126          # 0.2216

# The 21 diagnostic features. Anything else in the sheet (CLASS, the ten
# one-hot pattern columns A..SUSP, DATE/FileName/SegFile bookkeeping) is not a
# feature and must not be fed to the model -- CLASS in particular leaks the
# morphologic pattern, and the A..SUSP dummies ARE that pattern one-hot.
FEATURE_COLS = [
    "LB", "AC", "FM", "UC", "DL", "DS", "DP",
    "ASTV", "MSTV", "ALTV", "MLTV",
    "Width", "Min", "Max", "Nmax", "Nzeros",
    "Mode", "Mean", "Median", "Variance", "Tendency",
]

# Columns that must never become features even if present in the source.
LEAKY_COLS = ["CLASS", "Class", "class", "NSP", "nsp",
              "A", "B", "C", "D", "SH", "AD", "DE", "LD", "FS", "SUSP",
              "DATE", "FileName", "SegFile", "b", "e"]

LOCAL_PATHS = ["data/CTG.xls", "data/CTG.xlsx", "data/cardiotocography.csv"]


def _from_local():
    """UCI's CTG.xls, sheet 'Raw Data'. Most reliable source: it has NSP."""
    for path in LOCAL_PATHS:
        if not os.path.exists(path):
            continue
        print(f"  Reading local file: {path}")
        if path.endswith(".csv"):
            return pd.read_csv(path)
        # 'Raw Data' holds one row per record plus a trailing summary row.
        try:
            return pd.read_excel(path, sheet_name="Raw Data", header=0)
        except Exception:
            return pd.read_excel(path, sheet_name=0, header=0)
    return None


def _from_ucimlrepo():
    """UCI's own API. Returns features and a targets frame containing NSP."""
    try:
        from ucimlrepo import fetch_ucirepo
    except ImportError:
        print("  (ucimlrepo not installed; pip install ucimlrepo)")
        return None
    try:
        repo = fetch_ucirepo(id=193)
        df = pd.concat([repo.data.features, repo.data.targets], axis=1)
        print("  Loaded via ucimlrepo (UCI id=193)")
        return df
    except Exception as e:
        print(f"  (ucimlrepo fetch failed: {e})")
        return None


def _from_openml():
    """
    OpenML fallback. Several CTG copies exist and they are NOT interchangeable:
    some expose NSP, some expose only the 10-class CLASS label under the name
    'Class'. Every candidate is checked for a real NSP column and rejected
    otherwise -- this is exactly where the previous loader went wrong.
    """
    from sklearn.datasets import fetch_openml

    for kwargs in ({"data_id": 1466}, {"data_id": 1560},
                   {"name": "cardiotocography", "version": 1}):
        try:
            data = fetch_openml(as_frame=True, parser="auto", **kwargs)
            df = data.frame
            if _find_nsp(df) is not None:
                print(f"  Loaded via OpenML {kwargs} (NSP present)")
                return df
            print(f"  Skipping OpenML {kwargs}: no NSP column "
                  f"(columns look like {list(df.columns)[:4]}...)")
        except Exception as e:
            print(f"  (OpenML {kwargs} failed: {str(e)[:80]})")
    return None


def _find_nsp(df):
    """Locate the NSP column by name only. No positional fallback."""
    for name in ("NSP", "nsp", "Nsp", "NSP_class"):
        if name in df.columns:
            return name
    return None


def load_cardiotocography(strict=True, scale=True):
    """
    Load the CTG dataset with the NSP (fetal status) label.

    Args
        scale : if True (default) standardize over the whole file; if False
            return raw features so the caller can fit the scaler inside its
            own train/eval split (see utils.data_loader.fit_apply_scaler).
            Fitting on all rows lets held-out data influence the released
            model and places a data-dependent step outside the DP-SGD
            mechanism, so training paths should pass scale=False.

    Returns
        X : features, standardized iff scale=True, (2126, 21)
        y : 0 = Normal, 1 = Suspect or Pathologic  (~22.2% positive)

    Raises RuntimeError if NSP cannot be found or the label distribution does
    not match UCI's published counts. strict=False downgrades the distribution
    check to a warning; the NSP requirement always applies.
    """
    print("Loading Cardiotocography (CTG)...")

    df = _from_local()
    if df is None:
        df = _from_ucimlrepo()
    if df is None:
        df = _from_openml()
    if df is None:
        raise RuntimeError(
            "Could not load CTG from any source.\n"
            "Fix: download CTG.xls from "
            "https://archive.ics.uci.edu/dataset/193/cardiotocography "
            "and place it at data/CTG.xls, or `pip install ucimlrepo`."
        )

    nsp_col = _find_nsp(df)
    if nsp_col is None:
        raise RuntimeError(
            f"No NSP column found. Columns: {list(df.columns)}\n"
            "This source does not contain the fetal-status label. The previous "
            "loader fell back to the last column, which is CLASS -- the 10-value "
            "morphologic pattern -- and silently trained on the wrong task.\n"
            "Fix: download CTG.xls from UCI to data/CTG.xls."
        )

    # Drop trailing summary/blank rows present in CTG.xls
    df = df[df[nsp_col].notna()].copy()

    y_raw = pd.to_numeric(df[nsp_col], errors="coerce")
    df = df[y_raw.notna()]
    y_raw = y_raw[y_raw.notna()].astype(int)

    counts = y_raw.value_counts().sort_index().to_dict()
    print(f"  NSP distribution: {counts}")

    if set(counts) != {1, 2, 3}:
        raise RuntimeError(
            f"NSP should take values {{1,2,3}} (Normal/Suspect/Pathologic) but "
            f"found {sorted(counts)}. Wrong column or wrong encoding."
        )

    if counts != EXPECTED_NSP_COUNTS:
        msg = (f"NSP counts {counts} do not match UCI's published "
               f"{EXPECTED_NSP_COUNTS}. The source may be filtered or reordered.")
        if strict:
            raise RuntimeError(msg + " Pass strict=False to continue anyway.")
        print(f"  WARNING: {msg}")

    # Binarize: Normal(1) -> 0, Suspect(2) + Pathologic(3) -> 1.
    # Fixed mapping from the documented encoding. No frequency inference and
    # no auto-flip: if the rate looks wrong, the input is wrong.
    y = (y_raw > 1).astype(np.float32).values

    available = [c for c in FEATURE_COLS if c in df.columns]
    if len(available) == len(FEATURE_COLS):
        X_df = df[FEATURE_COLS]
    else:
        missing = sorted(set(FEATURE_COLS) - set(available))
        print(f"  WARNING: {len(missing)} named features missing ({missing}); "
              f"falling back to all non-target columns.")
        X_df = df.drop(columns=[c for c in LEAKY_COLS if c in df.columns],
                       errors="ignore")

    X_df = X_df.apply(pd.to_numeric, errors="coerce")

    keep = X_df.notna().all(axis=1).values
    if not keep.all():
        print(f"  Dropping {int((~keep).sum())} rows with missing features")
    X_df, y = X_df[keep], y[keep]

    n, pos_rate = len(y), float(y.mean())
    print(f"  Samples: {n}, Features: {X_df.shape[1]}")
    print(f"  Positive rate (suspect/pathologic): {pos_rate:.1%}")

    if strict:
        if n != EXPECTED_N:
            raise RuntimeError(
                f"Expected {EXPECTED_N} samples, got {n}. Wrong CTG variant."
            )
        if abs(pos_rate - EXPECTED_POS_RATE) > 0.01:
            raise RuntimeError(
                f"Positive rate {pos_rate:.1%} differs from the expected "
                f"{EXPECTED_POS_RATE:.1%}. A rate near 27.2% means the CLASS "
                f"(morphologic pattern) column was loaded instead of NSP."
            )

    X = X_df.values.astype(np.float32)
    if scale:
        X = StandardScaler().fit_transform(X)

    print(f"\nCardiotocography: {X.shape[0]} samples, {X.shape[1]} features")
    print(f"  Scaling: {'whole-file' if scale else 'deferred to per-split'}")
    print(f"  Task: Normal (0) vs Suspect/Pathologic (1), from NSP")
    return X.astype(np.float32), y.astype(np.float32)


if __name__ == "__main__":
    X, y = load_cardiotocography(scale=True)
    print(f"\nX shape: {X.shape}, y shape: {y.shape}")
    print(f"Class distribution: {np.bincount(y.astype(int))}")
