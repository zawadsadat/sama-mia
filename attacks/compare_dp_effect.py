"""
Compare DP vs No-DP MIA results side by side.

Usage:
    python attacks/compare_dp_effect.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier


def load_and_evaluate(suffix, label):
    feat_path = f"experiments/attack_features{suffix}.npy"
    label_path = f"experiments/attack_labels{suffix}.npy"

    if not os.path.exists(feat_path):
        print(f"  {label}: NOT FOUND — run shadow_models.py {suffix.replace('_', '--')}")
        return None

    X = np.load(feat_path)
    y = np.load(label_path)

    member_idx = np.where(y == 1)[0]
    nonmember_idx = np.where(y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    member_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nonmember_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)

    balanced_idx = np.concatenate([member_sample, nonmember_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X[balanced_idx], y[balanced_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, (rf_probs > 0.5).astype(float))
    auc = roc_auc_score(y_test, rf_probs)

    fpr, tpr, _ = roc_curve(y_test, rf_probs)

    tpr_at = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at[target_fpr] = tpr[idx]

    member_mask = y == 1
    loss_gap = abs(X[member_mask][:, 2].mean() - X[~member_mask][:, 2].mean())

    return {
        "accuracy": acc, "auc": auc,
        "tpr_1": tpr_at[0.01], "tpr_5": tpr_at[0.05], "tpr_10": tpr_at[0.10],
        "loss_gap": loss_gap,
    }


print("=" * 60)
print("  MIA COMPARISON: Effect of Differential Privacy")
print("=" * 60)

nodp = load_and_evaluate("_nodp", "No-DP")
dp = load_and_evaluate("_dp", "DP")

if nodp and dp:
    print(f"\n{'Metric':<25} {'No DP':>10} {'With DP':>10} {'Δ':>10}")
    print("-" * 55)
    print(f"{'Attack Accuracy':<25} {nodp['accuracy']:>10.4f} {dp['accuracy']:>10.4f} {dp['accuracy']-nodp['accuracy']:>+10.4f}")
    print(f"{'Attack AUC':<25} {nodp['auc']:>10.4f} {dp['auc']:>10.4f} {dp['auc']-nodp['auc']:>+10.4f}")
    print(f"{'TPR @ 1% FPR':<25} {nodp['tpr_1']:>10.4f} {dp['tpr_1']:>10.4f} {dp['tpr_1']-nodp['tpr_1']:>+10.4f}")
    print(f"{'TPR @ 5% FPR':<25} {nodp['tpr_5']:>10.4f} {dp['tpr_5']:>10.4f} {dp['tpr_5']-nodp['tpr_5']:>+10.4f}")
    print(f"{'TPR @ 10% FPR':<25} {nodp['tpr_10']:>10.4f} {dp['tpr_10']:>10.4f} {dp['tpr_10']-nodp['tpr_10']:>+10.4f}")
    print(f"{'Member/Non-member':<25} {nodp['loss_gap']:>10.4f} {dp['loss_gap']:>10.4f} {dp['loss_gap']-nodp['loss_gap']:>+10.4f}")
    print(f"{'  loss gap':<25}")
    print("-" * 55)
