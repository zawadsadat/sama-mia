"""
SAMA v2 

Usage:
    python attacks/compute_fmpas_v2.py --dp   --seed 42 --n-bootstrap 2000
    python attacks/compute_fmpas_v2.py --no-dp --seed 42 --n-bootstrap 2000
    python attacks/compute_fmpas_v2.py --no-dp --row-split   # old behaviour
"""

import sys, os, argparse
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GroupShuffleSplit, train_test_split
from sklearn.utils import resample

from utils.config import (DATASET, NUM_SHADOW_MODELS, TEST_SIZE,
                          OBJECTIVE, VAL_FRACTION)
from utils.artifact_paths import load_artifact
from utils.data_loader import get_raw_data
from attacks.per_example_attacks import reconstruct_shadow_indices, out_statistics, score_rows

BB_PROB1, BB_LOSS, BB_ENTROPY = 1, 2, 3


def build_matrix(dp, min_out=2, fl=False):
    # Federated artifacts live under a separate kind prefix. Row ordering and
    # the per-shadow split are identical to the centralized pipeline, which is
    # what lets the same replay and the same estimator serve both.
    pfx = "fl_" if fl else ""
    X, _ = load_artifact(f"{pfx}attack_features", DATASET, dp)
    y, _ = load_artifact(f"{pfx}attack_labels", DATASET, dp)
    _, y_all = get_raw_data()

    # val_frac MUST match what shadow_models.py used for these artifacts.
    # The realistic ("weighted") configuration holds out a validation split per
    # shadow and excludes it from the attack set, so replaying a two-way split
    # against three-way features yields a different row count and a different
    # ordering. The assertion below is the guard; it fires on the row count
    # before any silently-misaligned scoring can happen.
    vf = VAL_FRACTION if OBJECTIVE == "weighted" else 0.0
    sample_idx, _, member = reconstruct_shadow_indices(
        y_all, NUM_SHADOW_MODELS, TEST_SIZE, val_frac=vf)
    assert len(sample_idx) == len(y), (
        f"row-count mismatch: replayed {len(sample_idx)} rows against "
        f"{len(y)} saved rows. OBJECTIVE={OBJECTIVE!r} implies val_frac={vf}; "
        f"if the artifacts were generated under a different configuration, "
        f"regenerate them with attacks/shadow_models.py before scoring.")
    assert (member == y).all(), "membership vector mismatch after replay"

    loss = X[:, BB_LOSS].astype(float)
    mu, sd, _ = out_statistics(loss, sample_idx, member, len(y_all), min_out=min_out)
    lira, calib = score_rows(loss, sample_idx, mu, sd)
    keep = ~np.isnan(lira) & ~np.isnan(calib)

    p_y = np.exp(-loss)                    # A3: true-label probability
    F = np.column_stack([X[:, BB_PROB1], loss, X[:, BB_ENTROPY], p_y, lira, calib])
    return F[keep], y[keep].astype(int), sample_idx[keep]


COLS = dict(prob1=0, loss=1, entropy=2, p_y=3, lira=4, calib=5)


def build_attacks(X_fit, y_fit, seed):
    a = {}
    # Portfolio matches Table I: loss, entropy, two per-example, RF, NN.
    # The confidence threshold is excluded -- on p_1 it scores the positive
    # class rather than the true label and falls below chance; corrected to
    # p_y = exp(-loss) it is a monotone transform of the loss threshold and
    # therefore has an identical ROC.
    a["Loss threshold"] = lambda X: -X[:, COLS["loss"]]
    a["Entropy threshold"] = lambda X: -X[:, COLS["entropy"]]
    a["Difficulty-calibrated loss"] = lambda X: X[:, COLS["calib"]]
    a["Offline LiRA"] = lambda X: X[:, COLS["lira"]]
    f = [COLS["prob1"], COLS["loss"], COLS["entropy"]]
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed,
                                class_weight="balanced").fit(X_fit[:, f], y_fit)
    a["Random Forest"] = lambda X, m=rf, c=f: m.predict_proba(X[:, c])[:, 1]
    nn = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200,
                       random_state=seed).fit(X_fit[:, f], y_fit)
    a["Neural Network"] = lambda X, m=nn, c=f: m.predict_proba(X[:, c])[:, 1]
    return a


def balance(X, y, g, seed):
    m, nm = np.where(y == 1)[0], np.where(y == 0)[0]
    k = min(len(m), len(nm))
    idx = np.concatenate([resample(m, n_samples=k, replace=False, random_state=seed),
                          resample(nm, n_samples=k, replace=False, random_state=seed)])
    idx = idx[np.random.RandomState(seed).permutation(len(idx))]
    return X[idx], y[idx], g[idx]


def split3(X, y, g, seed, grouped):
    if grouped:
        s1 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        i_fit, i_rest = next(s1.split(X, y, groups=g))
        s2 = GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=seed)
        j_sel, j_aud = next(s2.split(X[i_rest], y[i_rest], groups=g[i_rest]))
        i_sel, i_aud = i_rest[j_sel], i_rest[j_aud]
    else:
        i_all = np.arange(len(y))
        i_fit, i_rest = train_test_split(i_all, test_size=0.5, random_state=seed, stratify=y)
        i_sel, i_aud = train_test_split(i_rest, test_size=0.5, random_state=seed, stratify=y[i_rest])
    return i_fit, i_sel, i_aud


def max_adv(y, s):
    fpr, tpr, thr = roc_curve(y, s)
    j = int(np.argmax(tpr - fpr))
    return float(tpr[j] - fpr[j]), float(thr[j])


def adv_at(y, s, tau):
    p = (s >= tau)
    if (y == 1).sum() == 0 or (y == 0).sum() == 0:
        return float("nan")
    return float(p[y == 1].mean() - p[y == 0].mean())


def boot_ci(y, s, g, tau, n, seed, cluster):
    rng = np.random.RandomState(seed)
    vals = []
    if cluster:
        recs = np.unique(g)
        by = {r: np.where(g == r)[0] for r in recs}
        for _ in range(n):
            pick = rng.choice(recs, len(recs), replace=True)
            i = np.concatenate([by[r] for r in pick])
            v = adv_at(y[i], s[i], tau)
            if not np.isnan(v):
                vals.append(v)
    else:
        pos, neg = np.where(y == 1)[0], np.where(y == 0)[0]
        for _ in range(n):
            i = np.concatenate([rng.choice(pos, len(pos), True), rng.choice(neg, len(neg), True)])
            v = adv_at(y[i], s[i], tau)
            if not np.isnan(v):
                vals.append(v)
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def main():
    p = argparse.ArgumentParser()
    g_ = p.add_mutually_exclusive_group(required=True)
    g_.add_argument("--dp", action="store_true")
    g_.add_argument("--no-dp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--row-split", action="store_true", help="old row-level split")
    p.add_argument("--fl", action="store_true",
                   help="score the federated (FedAvg) artifacts")
    args = p.parse_args()

    grouped = not args.row_split
    X, y, g = build_matrix(args.dp, fl=args.fl)
    Xb, yb, gb = balance(X, y, g, args.seed)
    i_fit, i_sel, i_aud = split3(Xb, yb, gb, args.seed, grouped)
    print(f"{DATASET} | {OBJECTIVE} | {'FL' if args.fl else 'centralized'} | "
          f"{'DP' if args.dp else 'No-DP'} | "
          f"{'GROUP' if grouped else 'ROW'} split | rows={len(yb)} "
          f"fit={len(i_fit)} sel={len(i_sel)} aud={len(i_aud)} "
          f"| records in audit={len(np.unique(gb[i_aud]))} "
          f"| record overlap fit/aud={len(np.intersect1d(gb[i_fit], gb[i_aud]))}")

    atk = build_attacks(Xb[i_fit], yb[i_fit], args.seed)
    rows = []
    for name, sc in atk.items():
        s_sel, s_aud = sc(Xb[i_sel]), sc(Xb[i_aud])
        a_sel, tau = max_adv(yb[i_sel], s_sel)
        a_aud = adv_at(yb[i_aud], s_aud, tau)
        lo_r, hi_r = boot_ci(yb[i_aud], s_aud, gb[i_aud], tau, args.n_bootstrap, args.seed, False)
        lo_c, hi_c = boot_ci(yb[i_aud], s_aud, gb[i_aud], tau, min(args.n_bootstrap, 400),
                             args.seed, True)
        rows.append(dict(Attack=name, AUC=round(roc_auc_score(yb[i_aud], s_aud), 4),
                         Adv_sel=round(a_sel, 4), Adv_heldout=round(a_aud, 4),
                         CI_row=f"[{lo_r:.4f}, {hi_r:.4f}]",
                         CI_record=f"[{lo_c:.4f}, {hi_c:.4f}]"))
    df = pd.DataFrame(rows)
    print(df.to_string(index=False))
    best = df.loc[df["Adv_sel"].idxmax()]
    print(f"\nFMPAS = {best['Adv_heldout']}  attack={best['Attack']}  "
          f"row CI {best['CI_row']}  record CI {best['CI_record']}")
    df.to_csv(f"experiments/fmpas_v2_{DATASET}_{'dp' if args.dp else 'nodp'}.csv", index=False)


if __name__ == "__main__":
    main()
