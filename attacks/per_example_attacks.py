"""
Per-example offline LiRA and difficulty-calibrated loss.

USAGE
-----
    python attacks/per_example_attacks.py --no-dp
    python attacks/per_example_attacks.py --dp
    python attacks/per_example_attacks.py --dp --verify-only

"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from utils.splits import three_way_split
from sklearn.metrics import roc_curve, roc_auc_score

from utils.config import (DATASET, TEST_SIZE, NUM_SHADOW_MODELS,
                          OBJECTIVE, VAL_FRACTION)
from utils.data_loader import get_raw_data
from utils.artifact_paths import load_artifact, save_artifact

LOSS_COL = 2  # attack feature layout: [prob0, prob1, loss, entropy]


# --------------------------------------------------------------------------
# split reconstruction
# --------------------------------------------------------------------------

def reconstruct_shadow_indices(y_all, n_shadow, test_size, val_frac=0.0):
    """
    Rebuild the exact per-shadow train/test splits used by shadow_models.py.

    That loop does, for model_id in range(NUM_SHADOW_MODELS), either a plain
    train_test_split (stress test) or three_way_split with val_frac
    (realistic configuration, validation rows excluded from the attack set),
    then appends member features (train, label 1) followed by non-member
    features (test, label 0). Reproducing the same calls on an index array
    recovers the row ordering exactly.

    Returns
        sample_idx  original-sample index for each attack row
        shadow_idx  which shadow produced each attack row
        member      1 if the row was a member row, else 0
    """
    idx = np.arange(len(y_all))
    sample_idx, shadow_idx, member = [], [], []

    for model_id in range(n_shadow):
        if val_frac > 0:
            # Realistic configuration: shadow_models.py holds out a validation
            # split per shadow and EXCLUDES it from the attack set, so the
            # replay must use the same three-way split or the row ordering
            # will not line up with the saved features.
            (tr, _), (_, _), (te, _) = three_way_split(
                idx, y_all, test_size, val_frac=val_frac, random_state=model_id
            )
        else:
            tr, te = train_test_split(
                idx, test_size=test_size, random_state=model_id, stratify=y_all
            )
        # order matters: members first, then non-members
        sample_idx.append(tr)
        shadow_idx.append(np.full(len(tr), model_id))
        member.append(np.ones(len(tr), dtype=int))

        sample_idx.append(te)
        shadow_idx.append(np.full(len(te), model_id))
        member.append(np.zeros(len(te), dtype=int))

    return (np.concatenate(sample_idx),
            np.concatenate(shadow_idx),
            np.concatenate(member))


def verify(sample_idx, member, y_attack):
    """
    Check the reconstruction against the saved labels.

    Row count and the full membership-label vector must both match. If they
    do, the mapping is exact -- there is no ambiguity left to worry about.
    """
    ok_len = len(sample_idx) == len(y_attack)
    ok_labels = ok_len and bool(np.array_equal(member, y_attack.astype(int)))
    return ok_len, ok_labels


# --------------------------------------------------------------------------
# per-example statistics
# --------------------------------------------------------------------------

def out_statistics(losses, sample_idx, member, n_samples, min_out=2):
    """
    Per-sample mean and sd of loss over the shadows where the sample was OUT.

    Returns mu_out, sigma_out, n_out. Samples with fewer than `min_out`
    observations get NaN -- their sigma is not estimable and they must be
    excluded from scoring rather than silently given a garbage value.

    sigma is floored at a small positive value: a sample that every out-shadow
    scores identically would otherwise divide by zero. This is not rare --
    on CTG the member-loss sd is exactly 0.
    """
    out_mask = member == 0
    s = sample_idx[out_mask]
    l = losses[out_mask]

    n_out = np.bincount(s, minlength=n_samples).astype(float)
    sum_l = np.bincount(s, weights=l, minlength=n_samples)
    sum_l2 = np.bincount(s, weights=l ** 2, minlength=n_samples)

    with np.errstate(invalid="ignore", divide="ignore"):
        mu = sum_l / n_out
        var = sum_l2 / n_out - mu ** 2
    var = np.maximum(var, 0.0)          # kill negative values from rounding
    sigma = np.sqrt(var)

    too_few = n_out < min_out
    mu[too_few] = np.nan
    sigma[too_few] = np.nan

    return mu, sigma, n_out


def score_rows(losses, sample_idx, mu_out, sigma_out, sigma_floor=1e-6):
    """
    Offline-LiRA and difficulty-calibrated scores for every attack row.

    Both are oriented so that higher = more likely member (a member's loss
    should sit BELOW the out-distribution of its own example).

    IMPORTANT: for a member row we use the out-statistics of that same original
    sample, gathered from the shadows where it was held out -- never from the
    shadow that produced the row. That is what makes this per-example rather
    than global, and it is why no membership label of the scored row is used.
    """
    mu = mu_out[sample_idx]
    sigma = np.maximum(sigma_out[sample_idx], sigma_floor)

    lira = (mu - losses) / sigma
    calib = mu - losses
    return lira, calib


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------

def evaluate(y, scores, name):
    """AUC plus max-advantage. Reported for comparison with the old numbers;
    the unbiased held-out value comes from compute_fmpas.py."""
    auc = float(roc_auc_score(y, scores))
    fpr, tpr, _ = roc_curve(y, scores)
    j = int(np.argmax(tpr - fpr))
    return {
        "Attack": name,
        "AUC": round(auc, 4),
        "Adv_naive": round(float(tpr[j] - fpr[j]), 4),
        "TPR@1%": round(float(np.interp(0.01, fpr, tpr)), 4),
        "TPR@0.1%": round(float(np.interp(0.001, fpr, tpr)), 4),
        "n": int(len(y)),
    }


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dp", action="store_true")
    g.add_argument("--no-dp", action="store_true")
    p.add_argument("--min-out", type=int, default=2,
                   help="minimum out-shadow observations to score a sample")
    p.add_argument("--verify-only", action="store_true",
                   help="check the reconstruction and exit")
    p.add_argument("--fl", action="store_true",
                   help="use the federated attack set from "
                        "attacks/fl_shadow_models.py instead of the centralized one")
    args = p.parse_args()

    use_dp = args.dp
    print(f"=== Per-example attacks: {DATASET} | "
          f"{'DP' if use_dp else 'No-DP'} ===")
    print(f"  NUM_SHADOW_MODELS={NUM_SHADOW_MODELS}  TEST_SIZE={TEST_SIZE}")

    X_all, y_all = get_raw_data()
    y_all = np.asarray(y_all).ravel()
    n_samples = len(y_all)

    kind = "fl_attack_features" if args.fl else "attack_features"
    kind_lbl = "fl_attack_labels" if args.fl else "attack_labels"
    X_att, bb_path = load_artifact(kind, DATASET, use_dp)
    y_att, _ = load_artifact(kind_lbl, DATASET, use_dp)
    print(f"  attack set: {bb_path}  rows={len(y_att)}")

    # val_frac must match what shadow_models.py used, or the replayed row
    # ordering will not line up with the saved features. verify() below is the
    # guard: it checks both the row count and the full membership vector.
    vf = VAL_FRACTION if OBJECTIVE == "weighted" else 0.0
    print(f"  replaying splits with val_frac={vf} (OBJECTIVE={OBJECTIVE})")
    sample_idx, shadow_idx, member = reconstruct_shadow_indices(
        y_all, NUM_SHADOW_MODELS, TEST_SIZE, val_frac=vf
    )

    ok_len, ok_labels = verify(sample_idx, member, y_att)
    print(f"\n  reconstructed rows : {len(sample_idx)}")
    print(f"  saved rows         : {len(y_att)}")
    print(f"  row count match    : {'YES' if ok_len else 'NO'}")
    print(f"  label vector match : {'YES' if ok_labels else 'NO'}")

    if not (ok_len and ok_labels):
        print("\n  RECONSTRUCTION FAILED. The saved artifact was not produced")
        print("  with the current NUM_SHADOW_MODELS / TEST_SIZE, so rows cannot")
        print("  be mapped back to samples. Regenerate the features, or set the")
        print("  config to whatever produced this file, before using per-example")
        print("  attacks on this dataset.")
        sys.exit(1)

    if args.verify_only:
        print("\n  Verification passed.")
        return

    losses = X_att[:, LOSS_COL].astype(float)
    mu_out, sigma_out, n_out = out_statistics(
        losses, sample_idx, member, n_samples, min_out=args.min_out
    )

    print(f"\n  OUT observations per sample: mean {n_out.mean():.2f}  "
          f"min {int(n_out.min())}  max {int(n_out.max())}")
    n_in = NUM_SHADOW_MODELS - n_out
    print(f"  IN  observations per sample: mean {n_in.mean():.2f}  "
          f"never-IN {int((n_in == 0).sum())} "
          f"({100 * (n_in == 0).mean():.1f}%)")
    print(f"  -> online (two-sided) LiRA is not estimable at this shadow count;")
    print(f"     using the offline variant, which needs only the OUT side.")

    n_drop = int(np.isnan(mu_out).sum())
    if n_drop:
        print(f"  {n_drop} samples have < {args.min_out} out-observations "
              f"and are excluded.")

    lira, calib = score_rows(losses, sample_idx, mu_out, sigma_out)

    keep = ~np.isnan(lira) & ~np.isnan(calib)
    y_keep = y_att[keep].astype(int)

    rows = [
        evaluate(y_keep, -losses[keep], "Loss Threshold (reference)"),
        evaluate(y_keep, calib[keep], "Difficulty-Calibrated Loss (per-example)"),
        evaluate(y_keep, lira[keep], "Offline LiRA (per-example)"),
    ]
    df = pd.DataFrame(rows)

    print("\nRESULTS")
    print(df.to_string(index=False))

    identical = np.isclose(df.loc[0, "AUC"], df.loc[1, "AUC"])
    if identical:
        print("\n  NOTE: calibrated loss still matches the loss threshold. That")
        print("  means per-example difficulty carries no information on this")
        print("  dataset -- a finding, not a bug.")
    else:
        print("\n  The per-example attacks are now distinct from the loss")
        print("  threshold, unlike the global-scalar versions they replace.")

    # Save the per-row scores so compute_fmpas.py can audit them on the same
    # three-way split as everything else. Column order: [offline_lira, calib].
    scores = np.stack([lira[keep], calib[keep]], axis=1)
    pe_kind = "fl_perexample_scores" if args.fl else "perexample_scores"
    pe_lbl = "fl_perexample_labels" if args.fl else "perexample_labels"
    save_artifact(pe_kind, DATASET, use_dp, scores, {
        "columns": ["offline_lira", "difficulty_calibrated"],
        "n_shadow_models": NUM_SHADOW_MODELS,
        "test_size": TEST_SIZE,
        "min_out": args.min_out,
        "n_excluded": n_drop,
        "mean_out_obs": float(n_out.mean()),
        "mean_in_obs": float(n_in.mean()),
        "note": "offline LiRA only; online variant not estimable at this shadow count",
    })
    save_artifact(pe_lbl, DATASET, use_dp, y_keep)

    out_csv = f"experiments/perexample_attacks_{'fl_' if args.fl else ''}{DATASET}_" \
              f"{'dp' if use_dp else 'nodp'}.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")


if __name__ == "__main__":
    main()
