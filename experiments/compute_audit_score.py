"""

Usage:
    python experiments/compute_audit_score.py
"""

import sys
import os
import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from utils.config import DATASET

os.makedirs("experiments/plots", exist_ok=True)

DATASET_NAMES = {
    "breast_cancer": {"title": "Breast Cancer Wisconsin", "prefix": "breast_cancer"},
    "diabetes_hospital": {"title": "Diabetes 130-US Hospitals", "prefix": "diabetes_hospital"},
}

ds_title = DATASET_NAMES.get(DATASET, {"title": DATASET})["title"]
ds_prefix = DATASET_NAMES.get(DATASET, {"prefix": DATASET})["prefix"]

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "serif",
    "font.size": 11,
})


def compute_risk_from_metrics_csv(csv_path):
    """
    Compute Risk = max(TPR - FPR) across all attacks from a metrics CSV.
    """
    df = pd.read_csv(csv_path)
    # Attack advantage is already max(TPR - FPR) per attack
    if "Attack_Advantage" in df.columns:
        risk = df["Attack_Advantage"].max()
        worst_attack = df.loc[df["Attack_Advantage"].idxmax(), "Attack"]
        return risk, worst_attack
    return None, None


def compute_risk_from_features(X, y, fpr_thresholds=[0.001, 0.01, 0.05, 0.10]):
    """
    Compute Risk = max over attacks and FPR thresholds of (TPR - FPR).
    Quick version using loss threshold and RF only.
    """
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import roc_curve, roc_auc_score
    from sklearn.utils import resample
    from sklearn.ensemble import RandomForestClassifier

    # Balance
    member_idx = np.where(y == 1)[0]
    nonmember_idx = np.where(y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))
    m_s = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_s = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)
    bal_idx = np.concatenate([m_s, nm_s])
    perm = np.random.RandomState(42).permutation(len(bal_idx))
    bal_idx = bal_idx[perm]
    X_b, y_b = X[bal_idx], y[bal_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_b, y_b, test_size=0.3, random_state=42, stratify=y_b
    )

    max_adv = 0.0
    worst_attack = "none"

    # Loss threshold
    scores = -X_te[:, 2]
    fpr, tpr, _ = roc_curve(y_te, scores)
    adv = (tpr - fpr).max()
    if adv > max_adv:
        max_adv = adv
        worst_attack = "Loss Threshold"

    # RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42, class_weight="balanced")
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_te, rf_probs)
    adv = (tpr - fpr).max()
    if adv > max_adv:
        max_adv = adv
        worst_attack = "Random Forest"

    auc = roc_auc_score(y_te, rf_probs)
    return max_adv, worst_attack, auc


def main():
    print(f"=== Federated Membership Privacy Audit Score ===")
    print(f"Dataset: {ds_title}\n")

    # ============ Check for epsilon sweep results ============
    sweep_file = "experiments/epsilon_sweep_results.csv"

    if os.path.exists(sweep_file):
        print(f"Found epsilon sweep results: {sweep_file}")
        sweep_df = pd.read_csv(sweep_file)
        print(sweep_df.to_string(index=False))
        print()
    else:
        print(f"No epsilon sweep CSV found. Computing from .npy files only.\n")
        sweep_df = None

    # ============ Compute risk for no-DP and default DP ============
    audit_rows = []

    for suffix, condition, sigma, epsilon in [("_nodp", "No-DP", float("inf"), float("inf")),
                                                ("_dp", "DP (default)", 1.3, None)]:
        feat_file = f"experiments/attack_features{suffix}.npy"
        label_file = f"experiments/attack_labels{suffix}.npy"

        if not os.path.exists(feat_file):
            print(f"Skipping {condition}: {feat_file} not found")
            continue

        X = np.load(feat_file)
        y = np.load(label_file)

        risk, worst, auc = compute_risk_from_features(X, y)

        # Also check if comprehensive metrics CSV exists
        csv_path = f"experiments/metrics_{DATASET}{suffix}.csv"
        if os.path.exists(csv_path):
            csv_risk, csv_worst = compute_risk_from_metrics_csv(csv_path)
            if csv_risk is not None:
                risk = csv_risk
                worst = csv_worst

        row = {
            "Dataset": ds_title,
            "Condition": condition,
            "Sigma": sigma,
            "Epsilon": epsilon,
            "Risk_Score": round(risk, 4),
            "Worst_Attack": worst,
            "AUC": round(auc, 4),
        }
        audit_rows.append(row)
        print(f"{condition}: Risk = {risk:.4f} (worst: {worst}, AUC: {auc:.4f})")

    # ============ If epsilon sweep data exists, compute risk per level ============
    if sweep_df is not None and "NoiseMul" in sweep_df.columns:
        print(f"\nComputing risk from epsilon sweep data...")

        # The sweep results have AUC per noise level
        # Risk approximation: use AUC to estimate advantage
        # Risk ≈ 2*(AUC - 0.5) for well-calibrated binary classifiers
        for _, row in sweep_df.iterrows():
            nm = row.get("NoiseMul", row.get("noise_multiplier", None))
            eps = row.get("ε", row.get("epsilon", None))
            auc_val = row.get("AUC", row.get("auc", None))
            test_acc = row.get("TestAcc", row.get("test_acc", None))

            if nm is None or auc_val is None:
                continue

            # Approximate risk from AUC
            approx_risk = max(0, 2 * (auc_val - 0.5))

            loss_gap = row.get("LossGap", row.get("loss_gap", 0))

            audit_rows.append({
                "Dataset": ds_title,
                "Condition": f"DP (σ={nm})",
                "Sigma": nm,
                "Epsilon": eps,
                "Risk_Score": round(approx_risk, 4),
                "Worst_Attack": "Estimated",
                "AUC": round(auc_val, 4),
                "Test_Accuracy": round(test_acc, 4) if test_acc else None,
                "Loss_Gap": round(loss_gap, 4) if loss_gap else None,
            })

    # ============ Build audit DataFrame ============
    audit_df = pd.DataFrame(audit_rows)
    out_csv = f"experiments/privacy_audit_scores_{ds_prefix}.csv"
    audit_df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv}")

    # ============ Recommendation ============
    print(f"\n{'='*70}")
    print("PRIVACY AUDIT RECOMMENDATION")
    print(f"{'='*70}")

    risk_thresholds = [0.01, 0.05, 0.10]

    # Filter to DP rows with numeric sigma
    dp_rows = audit_df[
        (audit_df["Sigma"] != float("inf")) &
        (audit_df["Sigma"].notna())
    ].copy()

    if len(dp_rows) > 0:
        dp_rows = dp_rows.sort_values("Sigma")

        for target in risk_thresholds:
            safe = dp_rows[dp_rows["Risk_Score"] <= target]
            if len(safe) > 0:
                rec = safe.iloc[0]  # smallest sigma that is safe
                acc_str = f", accuracy={rec['Test_Accuracy']:.2%}" if pd.notna(rec.get("Test_Accuracy")) else ""
                print(f"  Target Risk ≤ {target:.2f}: Use σ ≥ {rec['Sigma']:.1f} "
                      f"(ε={rec['Epsilon']:.2f}, risk={rec['Risk_Score']:.4f}{acc_str})")
            else:
                print(f"  Target Risk ≤ {target:.2f}: No tested σ achieves this threshold")

    # ============ Plot ============
    if len(dp_rows) > 1:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        sigmas = dp_rows["Sigma"].values
        risks = dp_rows["Risk_Score"].values
        accs = dp_rows["Test_Accuracy"].values if "Test_Accuracy" in dp_rows else None

        # Risk vs sigma
        ax1.plot(sigmas, risks, "o-", color="#d62728", linewidth=2, markersize=8, label="Risk Score")
        for t in risk_thresholds:
            ax1.axhline(y=t, color="#888888", ls=":", lw=1, alpha=0.7)
            ax1.text(sigmas[-1] * 0.95, t + 0.003, f"target={t}", fontsize=8, ha="right", color="#888888")
        ax1.axhline(y=0, color="gray", ls="-", lw=0.5)
        ax1.set_xlabel("Noise Multiplier (σ)")
        ax1.set_ylabel("Risk Score = max(TPR − FPR)")
        ax1.set_title(f"Privacy Audit Risk — {ds_title}")
        ax1.legend()

        # Risk vs accuracy (Pareto)
        if accs is not None and not np.all(np.isnan(accs)):
            ax2.scatter(accs, risks, c=sigmas, cmap="coolwarm", s=100, edgecolors="black", linewidth=0.5, zorder=5)
            for i, s in enumerate(sigmas):
                ax2.annotate(f"σ={s}", (accs[i], risks[i]), fontsize=8, ha="left",
                             xytext=(5, 5), textcoords="offset points")
            for t in risk_thresholds:
                ax2.axhline(y=t, color="#888888", ls=":", lw=1, alpha=0.7)
            ax2.set_xlabel("Test Accuracy")
            ax2.set_ylabel("Risk Score")
            ax2.set_title("Privacy-Utility Pareto Frontier")
            cbar = plt.colorbar(ax2.collections[0], ax=ax2)
            cbar.set_label("σ")
        else:
            ax2.text(0.5, 0.5, "Test accuracy data\nnot available", ha="center", va="center",
                     transform=ax2.transAxes, fontsize=14, color="gray")
            ax2.set_title("Privacy-Utility Pareto (no accuracy data)")

        plt.tight_layout()
        plt.savefig(f"experiments/plots/audit_score_{ds_prefix}.png", dpi=200, bbox_inches="tight")
        plt.close()
        print(f"\nPlot saved: experiments/plots/audit_score_{ds_prefix}.png")

    # ============ Print full table ============
    print(f"\n{'='*70}")
    print("FULL AUDIT TABLE")
    print(f"{'='*70}")
    print(audit_df.to_string(index=False))


if __name__ == "__main__":
    main()
