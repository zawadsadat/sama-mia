"""
Compute ALL requested evaluation metrics for MIA attacks.

Outputs:
  - experiments/metrics_comprehensive.csv  (per-attack metrics)
  - experiments/metrics_summary.csv        (headline comparison)

Metrics computed:
  - AUC
  - Balanced Accuracy
  - TPR @ 0.1%, 1%, 5%, 10% FPR
  - Attack Advantage (TPR - FPR) at optimal threshold
  - PPV / Precision under realistic membership priors (1%, 5%, 10%, 50%)
  - Loss gap
  - Member/non-member mean loss

Usage:
    python attacks/compute_all_metrics.py --no-dp
    python attacks/compute_all_metrics.py --dp
"""

import sys
import os
import argparse
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_curve, roc_auc_score, balanced_accuracy_score,
    precision_score, confusion_matrix
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import norm

from utils.config import DATASET
from utils.artifact_paths import load_artifact

DATASET_NAMES = {
    "breast_cancer": "Breast Cancer",
    "diabetes_hospital": "Diabetes Hospital",
}

ds_name = DATASET_NAMES.get(DATASET, DATASET)

os.makedirs("experiments", exist_ok=True)


from utils.metrics import tpr_at_fpr as get_tpr_at_fpr  # noqa: E402


def compute_ppv(tpr, fpr, prior):
    """
    Compute Positive Predictive Value (precision) under a given membership prior.
    PPV = (TPR * prior) / (TPR * prior + FPR * (1 - prior))
    """
    numerator = tpr * prior
    denominator = tpr * prior + fpr * (1 - prior)
    if denominator == 0:
        return 0.0
    return numerator / denominator


def compute_attack_advantage(fpr_arr, tpr_arr):
    """Compute max attack advantage = max(TPR - FPR) across all thresholds."""
    adv = tpr_arr - fpr_arr
    best_idx = np.argmax(adv)
    return adv[best_idx], fpr_arr[best_idx], tpr_arr[best_idx]


def run_threshold_attack(X, y, feature_idx, name, lower_is_member=True):
    """Run a threshold-based attack on a single feature."""
    scores = -X[:, feature_idx] if lower_is_member else X[:, feature_idx]
    fpr, tpr, thresholds = roc_curve(y, scores)
    auc = roc_auc_score(y, scores)
    preds = (scores > np.median(scores)).astype(int)
    bal_acc = balanced_accuracy_score(y, preds)
    return name, scores, fpr, tpr, auc, bal_acc


def run_lira_attack(X_all, y_all, feature_idx=2, n_shadow=15):
    """
    Simplified LiRA: fit Gaussian to in/out loss distributions across shadow models.
    Uses the loss feature (index 2) and splits by membership label as proxy.
    """
    member_losses = X_all[y_all == 1, feature_idx]
    nonmember_losses = X_all[y_all == 0, feature_idx]

    mu_in, std_in = member_losses.mean(), max(member_losses.std(), 1e-8)
    mu_out, std_out = nonmember_losses.mean(), max(nonmember_losses.std(), 1e-8)

    all_losses = X_all[:, feature_idx]
    log_ratio = norm.logpdf(all_losses, mu_in, std_in) - norm.logpdf(all_losses, mu_out, std_out)

    fpr, tpr, _ = roc_curve(y_all, log_ratio)
    auc = roc_auc_score(y_all, log_ratio)
    preds = (log_ratio > 0).astype(int)
    bal_acc = balanced_accuracy_score(y_all, preds)
    return "LiRA", log_ratio, fpr, tpr, auc, bal_acc


def run_calibrated_loss_attack(X_all, y_all, feature_idx=2):
    """
    Calibrated loss: normalize each sample's loss by the average non-member loss.
    """
    nonmember_mean = X_all[y_all == 0, feature_idx].mean()
    calibrated = X_all[:, feature_idx] / max(nonmember_mean, 1e-8)
    scores = -calibrated  # lower calibrated loss = more likely member

    fpr, tpr, _ = roc_curve(y_all, scores)
    auc = roc_auc_score(y_all, scores)
    preds = (scores > np.median(scores)).astype(int)
    bal_acc = balanced_accuracy_score(y_all, preds)
    return "Calibrated Loss", scores, fpr, tpr, auc, bal_acc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--no-dp", dest="dp", action="store_false")
    args = parser.parse_args()

    suffix = "_dp" if args.dp else "_nodp"
    condition = "DP" if args.dp else "No-DP"

    print(f"=== Computing ALL metrics: {ds_name} | {condition} ===")

    # Load data.
    # This used to build its own filenames as attack_features{suffix}.npy,
    # omitting the dataset tag, so it read the legacy untagged artifacts --
    # which meant it silently scored whichever dataset happened to have
    # written them last, or crashed when they were absent. load_artifact()
    # builds the tagged path and verifies the sidecar names the dataset we
    # asked for, exactly as every other script in the repo does.
    X, _ = load_artifact("attack_features", DATASET, args.dp)
    y, _ = load_artifact("attack_labels", DATASET, args.dp)

    print(f"Loaded {len(X)} samples ({y.sum():.0f} members, {(1-y).sum():.0f} non-members)")

    # Balance classes
    member_idx = np.where(y == 1)[0]
    nonmember_idx = np.where(y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)
    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X[balanced_idx], y[balanced_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # ============ Run all attacks ============
    results = []
    fpr_thresholds = [0.001, 0.01, 0.05, 0.10]
    membership_priors = [0.01, 0.05, 0.10, 0.50]

    # --- 1. Loss threshold ---
    name, scores, fpr, tpr, auc, bal_acc = run_threshold_attack(
        X_test, y_test, 2, "Loss Threshold", lower_is_member=True
    )
    results.append((name, scores, fpr, tpr, auc, bal_acc))

    # --- 2. Confidence threshold ---
    name, scores, fpr, tpr, auc, bal_acc = run_threshold_attack(
        X_test, y_test, 1, "Confidence Threshold", lower_is_member=False
    )
    results.append((name, scores, fpr, tpr, auc, bal_acc))

    # --- 3. Entropy threshold ---
    name, scores, fpr, tpr, auc, bal_acc = run_threshold_attack(
        X_test, y_test, 3, "Entropy Threshold", lower_is_member=True
    )
    results.append((name, scores, fpr, tpr, auc, bal_acc))

    # --- 4. Calibrated Loss ---
    name, scores, fpr, tpr, auc, bal_acc = run_calibrated_loss_attack(X_test, y_test)
    results.append((name, scores, fpr, tpr, auc, bal_acc))

    # --- 5. LiRA ---
    name, scores, fpr, tpr, auc, bal_acc = run_lira_attack(X_test, y_test)
    results.append((name, scores, fpr, tpr, auc, bal_acc))

    # --- 6. Random Forest ---
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=42, class_weight="balanced"
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    rf_fpr, rf_tpr, _ = roc_curve(y_test, rf_probs)
    rf_auc = roc_auc_score(y_test, rf_probs)
    rf_preds = (rf_probs > 0.5).astype(int)
    rf_bal_acc = balanced_accuracy_score(y_test, rf_preds)
    results.append(("Random Forest", rf_probs, rf_fpr, rf_tpr, rf_auc, rf_bal_acc))

    # --- 7. Neural Network (simple) ---
    from sklearn.neural_network import MLPClassifier
    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=42
    )
    nn.fit(X_train, y_train)
    nn_probs = nn.predict_proba(X_test)[:, 1]
    nn_fpr, nn_tpr, _ = roc_curve(y_test, nn_probs)
    nn_auc = roc_auc_score(y_test, nn_probs)
    nn_preds = (nn_probs > 0.5).astype(int)
    nn_bal_acc = balanced_accuracy_score(y_test, nn_preds)
    results.append(("Neural Network", nn_probs, nn_fpr, nn_tpr, nn_auc, nn_bal_acc))

    # ============ Compute comprehensive metrics ============
    rows = []

    for name, scores, fpr_arr, tpr_arr, auc, bal_acc in results:
        row = {
            "Dataset": ds_name,
            "Condition": condition,
            "Attack": name,
            "AUC": round(auc, 4),
            "Balanced_Accuracy": round(bal_acc, 4),
        }

        # TPR at various FPR thresholds
        for t in fpr_thresholds:
            tpr_val = get_tpr_at_fpr(fpr_arr, tpr_arr, t)
            pct = f"{t*100:.1f}".replace(".0", "")
            row[f"TPR@{pct}%FPR"] = round(tpr_val, 4)

        # Attack advantage
        max_adv, adv_fpr, adv_tpr = compute_attack_advantage(fpr_arr, tpr_arr)
        row["Attack_Advantage"] = round(max_adv, 4)
        row["Adv_at_FPR"] = round(adv_fpr, 4)
        row["Adv_at_TPR"] = round(adv_tpr, 4)

        # PPV under different priors, reported at TWO operating points.
        # The advantage-maximising threshold sits mid-ROC where false
        # positives are abundant, so PPV there is close to the prior; a
        # low-FPR point inverts the picture. Reporting only one of them
        # hides that, and the paper's text quotes the low-FPR figures, so
        # both are recorded here with the operating point in the column name.
        for op_fpr, tag in ((0.001, "0.1"), (0.05, "5")):
            tpr_op = get_tpr_at_fpr(fpr_arr, tpr_arr, op_fpr)
            for prior in membership_priors:
                ppv = compute_ppv(tpr_op, op_fpr, prior)
                pct = f"{prior*100:.0f}"
                row[f"PPV@{tag}%FPR_prior{pct}%"] = round(ppv, 4)

        rows.append(row)

    # ============ Loss gap ============
    member_mask = y_test == 1
    member_loss = X_test[member_mask, 2].mean()
    nonmember_loss = X_test[~member_mask, 2].mean()
    loss_gap = abs(member_loss - nonmember_loss)

    # ============ Save to CSV ============
    df = pd.DataFrame(rows)
    out_file = f"experiments/metrics_{DATASET}{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nSaved: {out_file}")

    # ============ Print summary ============
    print(f"\n{'='*80}")
    print(f"COMPREHENSIVE METRICS: {ds_name} | {condition}")
    print(f"{'='*80}")
    print(f"Loss Gap: {loss_gap:.4f} (member mean: {member_loss:.4f}, non-member mean: {nonmember_loss:.4f})")
    print(f"\n{'Attack':<22} {'AUC':>6} {'BalAcc':>7} {'T@0.1%':>7} {'T@1%':>6} {'T@5%':>6} {'T@10%':>6} {'Adv':>6}")
    print("-" * 80)
    for r in rows:
        print(f"{r['Attack']:<22} {r['AUC']:>6.4f} {r['Balanced_Accuracy']:>7.4f} "
              f"{r['TPR@0.1%FPR']:>7.4f} {r['TPR@1%FPR']:>6.4f} {r['TPR@5%FPR']:>6.4f} "
              f"{r['TPR@10%FPR']:>6.4f} {r['Attack_Advantage']:>6.4f}")

    print(f"\n  PPV at the 0.1% FPR operating point, by membership prior")
    print(f"\n{'Attack':<22} {'PPV@1%':>7} {'PPV@5%':>7} {'PPV@10%':>8} {'PPV@50%':>8}")
    print("-" * 55)
    for r in rows:
        print(f"{r['Attack']:<22} {r['PPV@0.1%FPR_prior1%']:>7.4f} {r['PPV@0.1%FPR_prior5%']:>7.4f} "
              f"{r['PPV@0.1%FPR_prior10%']:>8.4f} {r['PPV@0.1%FPR_prior50%']:>8.4f}")


if __name__ == "__main__":
    main()
