"""
Generate plots for white-box attack results.

Usage:
    python attacks/plot_whitebox.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

os.makedirs("experiments/plots", exist_ok=True)

# --- Results from runs ---
attacks = [
    "Grad Norm\n(L2)",
    "Grad Norm\n(L1)",
    "Loss\nThreshold",
    "RF\n(grad only)",
    "RF\n(all features)",
]

auc_nodp = [0.5885, 0.5894, 0.5862, 0.6029, 0.6280]
auc_dp   = [0.4980, 0.4976, 0.4984, 0.5031, 0.5016]

tpr10_nodp = [0.1240, 0.1260, 0.1287, 0.1533, 0.1927]
tpr10_dp   = [0.0980, 0.0980, 0.0953, 0.0873, 0.0907]

COLOR_NODP = "#d62728"
COLOR_DP = "#1f77b4"
COLOR_RANDOM = "#7f7f7f"

plt.rcParams.update({
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "grid.linestyle": "--",
    "font.family": "serif",
    "font.size": 11,
})


# ================================================================
# Figure 1: AUC comparison + TPR@10% comparison (side by side)
# ================================================================
print("Generating white-box comparison...")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5.5))

x = np.arange(len(attacks))
width = 0.32

# AUC
bars1 = ax1.bar(x - width/2, auc_nodp, width, label="No DP", color=COLOR_NODP, alpha=0.85)
bars2 = ax1.bar(x + width/2, auc_dp, width, label="With DP", color=COLOR_DP, alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
             ha="center", va="bottom", fontsize=8, color=COLOR_NODP)
for bar in bars2:
    h = bar.get_height()
    ax1.text(bar.get_x() + bar.get_width()/2, h + 0.005, f"{h:.3f}",
             ha="center", va="bottom", fontsize=8, color=COLOR_DP)

ax1.axhline(y=0.5, color=COLOR_RANDOM, ls="--", lw=1.5, alpha=0.6, label="Random (0.5)")
ax1.set_xticks(x)
ax1.set_xticklabels(attacks, fontsize=9)
ax1.set_ylabel("AUC")
ax1.set_title("(a) White-box Attack AUC")
ax1.legend(fontsize=9)
ax1.set_ylim([0.45, 0.68])

# Annotate white-box advantage
ax1.annotate("", xy=(4 - width/2, auc_nodp[4]),
             xytext=(4 - width/2, auc_nodp[2]),
             arrowprops=dict(arrowstyle="<->", color="black", lw=1.5))
ax1.text(4.35, (auc_nodp[4] + auc_nodp[2]) / 2, f"+{auc_nodp[4]-auc_nodp[2]:.3f}\nWB advantage",
         fontsize=8, ha="left", va="center", color="black")

# TPR@10%
bars1 = ax2.bar(x - width/2, tpr10_nodp, width, label="No DP", color=COLOR_NODP, alpha=0.85)
bars2 = ax2.bar(x + width/2, tpr10_dp, width, label="With DP", color=COLOR_DP, alpha=0.85)

for bar in bars1:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
             ha="center", va="bottom", fontsize=8, color=COLOR_NODP)
for bar in bars2:
    h = bar.get_height()
    ax2.text(bar.get_x() + bar.get_width()/2, h + 0.003, f"{h:.3f}",
             ha="center", va="bottom", fontsize=8, color=COLOR_DP)

ax2.axhline(y=0.10, color=COLOR_RANDOM, ls="--", lw=1.5, alpha=0.6, label="Random baseline (0.10)")
ax2.set_xticks(x)
ax2.set_xticklabels(attacks, fontsize=9)
ax2.set_ylabel("TPR @ 10% FPR")
ax2.set_title("(b) White-box Attack TPR at 10% FPR")
ax2.legend(fontsize=9)
ax2.set_ylim([0.0, 0.25])

plt.tight_layout()
plt.savefig("experiments/plots/whitebox_01_comparison.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 2: Gradient norm distributions (members vs non-members)
# ================================================================
print("Generating gradient norm distributions...")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

for ax, suffix, label in [
    (axes[0], "_nodp", "Without DP"),
    (axes[1], "_dp", "With DP"),
]:
    feat_path = f"experiments/whitebox_features{suffix}.npy"
    label_path = f"experiments/whitebox_labels{suffix}.npy"

    if not os.path.exists(feat_path):
        ax.text(0.5, 0.5, f"Data not found:\n{feat_path}",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_title(label)
        continue

    X = np.load(feat_path)
    y = np.load(label_path)

    member_mask = y == 1
    member_norms = X[member_mask, 0]     # column 0 = grad_norm_l2
    nonmember_norms = X[~member_mask, 0]

    # Clip extreme outliers for visualization
    clip_val = np.percentile(np.concatenate([member_norms, nonmember_norms]), 99)
    member_norms = np.clip(member_norms, 0, clip_val)
    nonmember_norms = np.clip(nonmember_norms, 0, clip_val)

    bins = np.linspace(0, clip_val, 60)

    ax.hist(member_norms, bins=bins, alpha=0.6, color=COLOR_NODP, density=True,
            label=f"Members (mean={member_norms.mean():.4f})")
    ax.hist(nonmember_norms, bins=bins, alpha=0.6, color=COLOR_DP, density=True,
            label=f"Non-members (mean={nonmember_norms.mean():.4f})")
    ax.axvline(member_norms.mean(), color=COLOR_NODP, ls="--", lw=1.5)
    ax.axvline(nonmember_norms.mean(), color=COLOR_DP, ls="--", lw=1.5)

    gap = abs(member_norms.mean() - nonmember_norms.mean())
    ax.set_title(f"{label}\nGrad norm gap = {gap:.4f}")
    ax.set_xlabel("Gradient L2 Norm")
    ax.set_ylabel("Density")
    ax.legend(fontsize=9)

fig.suptitle("Gradient Norm Distribution: Members vs Non-Members",
             fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("experiments/plots/whitebox_02_grad_distributions.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nPlots saved to experiments/plots/")
print("  whitebox_01_comparison.png")
print("  whitebox_02_grad_distributions.png")
