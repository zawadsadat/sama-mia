"""
Generate plots for the Privacy Budget (ε) sweep.

Dual-dataset comparison: Breast Cancer and Diabetes 130-US Hospitals.

Usage:
    python experiments/plot_epsilon_sweep.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("experiments/plots", exist_ok=True)

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "legend.fontsize": 9,
})

# === Breast Cancer results ===
bc = {
    "name": "Breast Cancer Wisconsin",
    "short": "Breast Cancer",
    "color_auc": "#d62728",
    "color_acc": "#ff7f0e",
    "nodp": {"epsilon": None, "test_acc": 0.9538, "auc": 0.5240, "loss_gap": 0.2930},
    "sweep": [
        {"nm": 0.3, "epsilon": 380.31, "test_acc": 0.7579, "auc": 0.4870, "loss_gap": 0.0177,
         "tpr_1": 0.0119, "tpr_5": 0.0595, "tpr_10": 0.0833},
        {"nm": 0.5, "epsilon": 137.18, "test_acc": 0.6266, "auc": 0.5209, "loss_gap": 0.0085,
         "tpr_1": 0.0238, "tpr_5": 0.1071, "tpr_10": 0.1905},
        {"nm": 0.8, "epsilon": 58.08, "test_acc": 0.6765, "auc": 0.5089, "loss_gap": 0.0063,
         "tpr_1": 0.0119, "tpr_5": 0.0595, "tpr_10": 0.0714},
        {"nm": 1.0, "epsilon": 39.99, "test_acc": 0.6266, "auc": 0.6251, "loss_gap": 0.0029,
         "tpr_1": 0.0000, "tpr_5": 0.0833, "tpr_10": 0.1548},
        {"nm": 1.3, "epsilon": 26.51, "test_acc": 0.6266, "auc": 0.4568, "loss_gap": 0.0040,
         "tpr_1": 0.0119, "tpr_5": 0.0238, "tpr_10": 0.0714},
        {"nm": 2.0, "epsilon": 14.30, "test_acc": 0.6266, "auc": 0.4910, "loss_gap": 0.0016,
         "tpr_1": 0.0238, "tpr_5": 0.0357, "tpr_10": 0.0595},
        {"nm": 3.0, "epsilon": 8.40, "test_acc": 0.6266, "auc": 0.5177, "loss_gap": 0.0011,
         "tpr_1": 0.0119, "tpr_5": 0.0476, "tpr_10": 0.0952},
        {"nm": 5.0, "epsilon": 4.50, "test_acc": 0.6266, "auc": 0.4337, "loss_gap": 0.0011,
         "tpr_1": 0.0000, "tpr_5": 0.0238, "tpr_10": 0.0595},
    ],
}

# === Diabetes 130-US Hospitals results ===
db = {
    "name": "Diabetes 130-US Hospitals",
    "short": "Diabetes Hospital",
    "color_auc": "#1f77b4",
    "color_acc": "#2ca02c",
    "nodp": {"epsilon": None, "test_acc": 0.8732, "auc": 0.6108, "loss_gap": 0.7738},
    "sweep": [
        {"nm": 0.3, "epsilon": 110.60, "test_acc": 0.9103, "auc": 0.5073, "loss_gap": 0.0191,
         "tpr_1": 0.0123, "tpr_5": 0.0531, "tpr_10": 0.1066},
        {"nm": 0.5, "epsilon": 11.56, "test_acc": 0.9103, "auc": 0.5040, "loss_gap": 0.0137,
         "tpr_1": 0.0123, "tpr_5": 0.0528, "tpr_10": 0.1012},
        {"nm": 0.8, "epsilon": 2.27, "test_acc": 0.9103, "auc": 0.5006, "loss_gap": 0.0120,
         "tpr_1": 0.0083, "tpr_5": 0.0487, "tpr_10": 0.0991},
        {"nm": 1.0, "epsilon": 1.43, "test_acc": 0.9103, "auc": 0.5009, "loss_gap": 0.0082,
         "tpr_1": 0.0103, "tpr_5": 0.0484, "tpr_10": 0.0987},
        {"nm": 1.3, "epsilon": 0.94, "test_acc": 0.9103, "auc": 0.5051, "loss_gap": 0.0080,
         "tpr_1": 0.0098, "tpr_5": 0.0511, "tpr_10": 0.1052},
        {"nm": 2.0, "epsilon": 0.53, "test_acc": 0.9103, "auc": 0.4997, "loss_gap": 0.0095,
         "tpr_1": 0.0098, "tpr_5": 0.0509, "tpr_10": 0.1026},
        {"nm": 3.0, "epsilon": 0.33, "test_acc": 0.9103, "auc": 0.4996, "loss_gap": 0.0040,
         "tpr_1": 0.0099, "tpr_5": 0.0483, "tpr_10": 0.0993},
        {"nm": 5.0, "epsilon": 0.19, "test_acc": 0.9103, "auc": 0.4976, "loss_gap": 0.0086,
         "tpr_1": 0.0093, "tpr_5": 0.0482, "tpr_10": 0.0966},
    ],
}


def get_arrays(dataset):
    """Extract arrays for plotting, including the no-DP point."""
    epsilons = [r["epsilon"] for r in dataset["sweep"]]
    aucs = [r["auc"] for r in dataset["sweep"]]
    accs = [r["test_acc"] for r in dataset["sweep"]]
    loss_gaps = [r["loss_gap"] for r in dataset["sweep"]]
    return epsilons, aucs, accs, loss_gaps


# ================================================================
# Figure 1: AUC vs ε (both datasets)
# ================================================================
print("Generating AUC vs epsilon...")

fig, ax = plt.subplots(figsize=(9, 5.5))

for ds in [bc, db]:
    eps, aucs, _, _ = get_arrays(ds)
    # Plot sweep points
    ax.plot(eps, aucs, marker="o", linewidth=2, markersize=7,
            color=ds["color_auc"], label=f'{ds["short"]}')
    # Plot no-DP point as star
    ax.scatter([max(eps) * 2], [ds["nodp"]["auc"]], marker="*", s=200,
               color=ds["color_auc"], zorder=5, edgecolors="black", linewidth=0.5)
    ax.annotate(f'No DP\n(AUC={ds["nodp"]["auc"]:.3f})',
                xy=(max(eps) * 2, ds["nodp"]["auc"]),
                fontsize=8, ha="left", va="bottom",
                xytext=(10, 5), textcoords="offset points")

ax.axhline(y=0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Random (0.5)")
ax.set_xscale("log")
ax.set_xlabel("Privacy Budget ε (log scale) — lower = stronger privacy")
ax.set_ylabel("MIA Attack AUC")
ax.set_title("MIA Vulnerability vs Privacy Budget")
ax.legend(loc="best")
ax.invert_xaxis()

plt.tight_layout()
plt.savefig("experiments/plots/epsilon_01_auc_vs_epsilon.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 2: Dual-axis — AUC + Test Accuracy vs ε (per dataset)
# ================================================================
print("Generating dual-axis plots...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

for ax, ds in [(ax1, bc), (ax2, db)]:
    eps, aucs, accs, _ = get_arrays(ds)

    # AUC (left axis)
    ln1 = ax.plot(eps, aucs, marker="o", linewidth=2, markersize=7,
                  color=ds["color_auc"], label="MIA AUC")
    ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.5)
    ax.set_xlabel("ε (log scale)")
    ax.set_ylabel("MIA AUC", color=ds["color_auc"])
    ax.tick_params(axis="y", labelcolor=ds["color_auc"])
    ax.set_xscale("log")
    ax.invert_xaxis()

    # No-DP star
    ax.scatter([max(eps) * 2], [ds["nodp"]["auc"]], marker="*", s=150,
               color=ds["color_auc"], zorder=5)

    # Accuracy (right axis)
    ax_r = ax.twinx()
    ln2 = ax_r.plot(eps, accs, marker="s", linewidth=2, markersize=6,
                    color=ds["color_acc"], linestyle="--", label="Test Accuracy")
    ax_r.scatter([max(eps) * 2], [ds["nodp"]["test_acc"]], marker="*", s=150,
                 color=ds["color_acc"], zorder=5)
    ax_r.set_ylabel("Test Accuracy", color=ds["color_acc"])
    ax_r.tick_params(axis="y", labelcolor=ds["color_acc"])

    # Combined legend
    lns = ln1 + ln2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc="center left", fontsize=9)

    ax.set_title(f"{ds['short']}")

fig.suptitle("Privacy-Utility-Vulnerability Tradeoff", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/plots/epsilon_02_dual_axis.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 3: Loss gap vs ε (both datasets)
# ================================================================
print("Generating loss gap vs epsilon...")

fig, ax = plt.subplots(figsize=(9, 5.5))

for ds in [bc, db]:
    eps, _, _, loss_gaps = get_arrays(ds)
    ax.plot(eps, loss_gaps, marker="o", linewidth=2, markersize=7,
            color=ds["color_auc"], label=f'{ds["short"]}')
    # No-DP point
    ax.scatter([max(eps) * 2], [ds["nodp"]["loss_gap"]], marker="*", s=200,
               color=ds["color_auc"], zorder=5, edgecolors="black", linewidth=0.5)

ax.axhline(y=0, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("Privacy Budget ε (log scale)")
ax.set_ylabel("Member/Non-member Loss Gap")
ax.set_title("Memorization Signal vs Privacy Budget")
ax.legend(loc="best")
ax.invert_xaxis()

plt.tight_layout()
plt.savefig("experiments/plots/epsilon_03_lossgap_vs_epsilon.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 4: Noise multiplier vs ε mapping + AUC (both datasets)
# ================================================================
print("Generating noise multiplier comparison...")

fig, ax = plt.subplots(figsize=(9, 5.5))

for ds in [bc, db]:
    nms = [r["nm"] for r in ds["sweep"]]
    aucs = [r["auc"] for r in ds["sweep"]]
    ax.plot(nms, aucs, marker="o", linewidth=2, markersize=7,
            color=ds["color_auc"], label=f'{ds["short"]}')

ax.axhline(y=0.5, color="gray", ls=":", lw=1.5, alpha=0.7, label="Random (0.5)")
ax.set_xlabel("Noise Multiplier (σ)")
ax.set_ylabel("MIA Attack AUC")
ax.set_title("MIA Vulnerability vs DP-SGD Noise Level")
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("experiments/plots/epsilon_04_auc_vs_noise.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 5: Combined summary (2x2)
# ================================================================
print("Generating combined summary...")

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) AUC vs ε
ax = axes[0, 0]
for ds in [bc, db]:
    eps, aucs, _, _ = get_arrays(ds)
    ax.plot(eps, aucs, marker="o", linewidth=2, markersize=6,
            color=ds["color_auc"], label=ds["short"])
    ax.scatter([max(eps) * 2], [ds["nodp"]["auc"]], marker="*", s=150,
               color=ds["color_auc"], zorder=5)
ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.5)
ax.set_xscale("log")
ax.set_xlabel("ε")
ax.set_ylabel("MIA AUC")
ax.set_title("(a) MIA AUC vs Privacy Budget")
ax.legend(fontsize=8)
ax.invert_xaxis()

# (0,1) Loss gap vs ε
ax = axes[0, 1]
for ds in [bc, db]:
    eps, _, _, lgs = get_arrays(ds)
    ax.plot(eps, lgs, marker="o", linewidth=2, markersize=6,
            color=ds["color_auc"], label=ds["short"])
    ax.scatter([max(eps) * 2], [ds["nodp"]["loss_gap"]], marker="*", s=150,
               color=ds["color_auc"], zorder=5)
ax.set_xscale("log")
ax.set_xlabel("ε")
ax.set_ylabel("Loss Gap")
ax.set_title("(b) Loss Gap vs Privacy Budget")
ax.legend(fontsize=8)
ax.invert_xaxis()

# (1,0) BC dual axis
ax = axes[1, 0]
eps, aucs, accs, _ = get_arrays(bc)
ln1 = ax.plot(eps, aucs, marker="o", linewidth=2, markersize=6,
              color=bc["color_auc"], label="AUC")
ax.set_xscale("log")
ax.set_xlabel("ε")
ax.set_ylabel("AUC", color=bc["color_auc"])
ax.tick_params(axis="y", labelcolor=bc["color_auc"])
ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.5)
ax.invert_xaxis()
ax_r = ax.twinx()
ln2 = ax_r.plot(eps, accs, marker="s", linewidth=2, markersize=5,
                color=bc["color_acc"], ls="--", label="Accuracy")
ax_r.set_ylabel("Accuracy", color=bc["color_acc"])
ax_r.tick_params(axis="y", labelcolor=bc["color_acc"])
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="center left")
ax.set_title(f"(c) {bc['short']}: Privacy-Utility Tradeoff")

# (1,1) DB dual axis
ax = axes[1, 1]
eps, aucs, accs, _ = get_arrays(db)
ln1 = ax.plot(eps, aucs, marker="o", linewidth=2, markersize=6,
              color=db["color_auc"], label="AUC")
ax.set_xscale("log")
ax.set_xlabel("ε")
ax.set_ylabel("AUC", color=db["color_auc"])
ax.tick_params(axis="y", labelcolor=db["color_auc"])
ax.axhline(y=0.5, color="gray", ls=":", lw=1, alpha=0.5)
ax.invert_xaxis()
ax_r = ax.twinx()
ln2 = ax_r.plot(eps, accs, marker="s", linewidth=2, markersize=5,
                color=db["color_acc"], ls="--", label="Accuracy")
ax_r.set_ylabel("Accuracy", color=db["color_acc"])
ax_r.tick_params(axis="y", labelcolor=db["color_acc"])
lns = ln1 + ln2
ax.legend(lns, [l.get_label() for l in lns], fontsize=8, loc="center left")
ax.set_title(f"(d) {db['short']}: Privacy-Utility Tradeoff")

fig.suptitle("Privacy Budget Sweep: Effect of ε on MIA Vulnerability",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("experiments/plots/epsilon_05_combined.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nAll plots saved to experiments/plots/")
print("  epsilon_01_auc_vs_epsilon.png")
print("  epsilon_02_dual_axis.png")
print("  epsilon_03_lossgap_vs_epsilon.png")
print("  epsilon_04_auc_vs_noise.png")
print("  epsilon_05_combined.png")
