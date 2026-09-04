"""
Generate plots for the Non-IID FL MIA sweep results.

Produces:
  1. Heatmap: AUC vs (alpha, K)
  2. Line plot: AUC vs K for each alpha
  3. Line plot: AUC vs alpha for each K
  4. Heatmap: Loss gap vs (alpha, K)
  5. Scatter: AUC vs heterogeneity (HetStd)
  6. Combined summary figure

Usage:
    python experiments/plot_noniid_results.py
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

os.makedirs("experiments/plots", exist_ok=True)

# --- Data from the sweep (hardcoded from results) ---
alphas = [0.1, 0.3, 0.5, 1.0, 10.0]
clients = [3, 5, 10, 20]

# AUC[alpha_idx][K_idx]
auc_data = {
    0.1:  {3: 0.4988, 5: 0.6275, 10: 0.5210, 20: 0.5118},
    0.3:  {3: 0.4903, 5: 0.5096, 10: 0.5147, 20: 0.4997},
    0.5:  {3: 0.5103, 5: 0.5085, 10: 0.5183, 20: 0.5058},
    1.0:  {3: 0.5203, 5: 0.5103, 10: 0.5166, 20: 0.5076},
    10.0: {3: 0.8421, 5: 0.8570, 10: 0.8615, 20: 0.9010},
}

loss_gap_data = {
    0.1:  {3: 0.0286, 5: 1.1192, 10: 0.0222, 20: 0.1660},
    0.3:  {3: 0.0013, 5: 0.1450, 10: 0.0212, 20: 0.0037},
    0.5:  {3: 0.2197, 5: 0.1135, 10: 0.1605, 20: 0.0423},
    1.0:  {3: 0.1809, 5: 0.2084, 10: 0.2283, 20: 0.1427},
    10.0: {3: 0.5863, 5: 0.3730, 10: 0.2645, 20: 0.1486},
}

test_acc_data = {
    0.1:  {3: 0.9086, 5: 0.8701, 10: 0.9092, 20: 0.9053},
    0.3:  {3: 0.9096, 5: 0.9100, 10: 0.9092, 20: 0.9103},
    0.5:  {3: 0.9054, 5: 0.9098, 10: 0.9103, 20: 0.9099},
    1.0:  {3: 0.9019, 5: 0.8911, 10: 0.9034, 20: 0.9076},
    10.0: {3: 0.8421, 5: 0.8570, 10: 0.8615, 20: 0.9010},
}

tpr10_data = {
    0.1:  {3: 0.0997, 5: 0.2214, 10: 0.1125, 20: 0.1128},
    0.3:  {3: 0.0934, 5: 0.1115, 10: 0.0956, 20: 0.0899},
    0.5:  {3: 0.1099, 5: 0.1166, 10: 0.1175, 20: 0.1036},
    1.0:  {3: 0.1096, 5: 0.1088, 10: 0.1201, 20: 0.1137},
    10.0: {3: 0.1528, 5: 0.1271, 10: 0.1369, 20: 0.1236},
}

het_std_data = {
    0.1:  {3: 0.471, 5: 0.468, 10: 0.424, 20: 0.309},
    0.3:  {3: 0.407, 5: 0.445, 10: 0.459, 20: 0.382},
    0.5:  {3: 0.404, 5: 0.446, 10: 0.382, 20: 0.345},
    1.0:  {3: 0.359, 5: 0.330, 10: 0.304, 20: 0.253},
    10.0: {3: 0.007, 5: 0.050, 10: 0.030, 20: 0.026},
}

# IID baseline
iid_auc = 0.5879
iid_loss_gap = 0.5604

# Style
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

CMAP_AUC = "RdYlBu_r"
CMAP_LOSS = "YlOrRd"
COLORS_K = {3: "#d62728", 5: "#ff7f0e", 10: "#1f77b4", 20: "#2ca02c"}
COLORS_A = {0.1: "#d62728", 0.3: "#ff7f0e", 0.5: "#9467bd", 1.0: "#1f77b4", 10.0: "#2ca02c"}


def to_matrix(data_dict):
    """Convert nested dict to 2D array [alpha x K]."""
    mat = np.zeros((len(alphas), len(clients)))
    for i, a in enumerate(alphas):
        for j, k in enumerate(clients):
            mat[i, j] = data_dict[a][k]
    return mat


# ================================================================
# Figure 1: AUC Heatmap
# ================================================================
print("Generating AUC heatmap...")
fig, ax = plt.subplots(figsize=(7, 5))

mat = to_matrix(auc_data)
norm = TwoSlopeNorm(vmin=mat.min(), vcenter=0.5, vmax=mat.max())
im = ax.imshow(mat, cmap=CMAP_AUC, norm=norm, aspect="auto")

ax.set_xticks(range(len(clients)))
ax.set_xticklabels([str(k) for k in clients])
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("Number of Clients (K)")
ax.set_ylabel("Dirichlet α")
ax.set_title("MIA Attack AUC Across Non-IID Configurations")

for i in range(len(alphas)):
    for j in range(len(clients)):
        val = mat[i, j]
        color = "white" if val > 0.7 or val < 0.45 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=11, fontweight="bold", color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("AUC (0.5 = random)")
cbar.ax.axhline(y=0.5, color="black", linewidth=1.5, linestyle="--")

plt.tight_layout()
plt.savefig("experiments/plots/noniid_01_auc_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 2: AUC vs K for each alpha
# ================================================================
print("Generating AUC vs K...")
fig, ax = plt.subplots(figsize=(8, 5))

for a in alphas:
    vals = [auc_data[a][k] for k in clients]
    ax.plot(clients, vals, marker="o", linewidth=2, markersize=7,
            label=f"α={a}", color=COLORS_A[a])

ax.axhline(y=iid_auc, color="gray", linestyle="--", linewidth=1.5,
           label=f"IID baseline ({iid_auc:.3f})")
ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5,
           label="Random (0.5)")

ax.set_xlabel("Number of Clients (K)")
ax.set_ylabel("MIA AUC")
ax.set_title("MIA Vulnerability vs Number of Clients")
ax.set_xticks(clients)
ax.legend(loc="best")
ax.set_ylim([0.45, 0.95])

plt.tight_layout()
plt.savefig("experiments/plots/noniid_02_auc_vs_K.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 3: AUC vs alpha for each K
# ================================================================
print("Generating AUC vs alpha...")
fig, ax = plt.subplots(figsize=(8, 5))

for k in clients:
    vals = [auc_data[a][k] for a in alphas]
    ax.plot(alphas, vals, marker="s", linewidth=2, markersize=7,
            label=f"K={k}", color=COLORS_K[k])

ax.axhline(y=iid_auc, color="gray", linestyle="--", linewidth=1.5,
           label=f"IID baseline ({iid_auc:.3f})")
ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5,
           label="Random (0.5)")

ax.set_xlabel("Dirichlet α (lower = more heterogeneous)")
ax.set_ylabel("MIA AUC")
ax.set_title("MIA Vulnerability vs Data Heterogeneity")
ax.set_xscale("log")
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas])
ax.legend(loc="best")
ax.set_ylim([0.45, 0.95])

plt.tight_layout()
plt.savefig("experiments/plots/noniid_03_auc_vs_alpha.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 4: Loss Gap Heatmap
# ================================================================
print("Generating loss gap heatmap...")
fig, ax = plt.subplots(figsize=(7, 5))

mat_loss = to_matrix(loss_gap_data)
im = ax.imshow(mat_loss, cmap=CMAP_LOSS, aspect="auto")

ax.set_xticks(range(len(clients)))
ax.set_xticklabels([str(k) for k in clients])
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("Number of Clients (K)")
ax.set_ylabel("Dirichlet α")
ax.set_title("Member/Non-Member Loss Gap")

for i in range(len(alphas)):
    for j in range(len(clients)):
        val = mat_loss[i, j]
        color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.3f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)

cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Loss Gap")

plt.tight_layout()
plt.savefig("experiments/plots/noniid_04_lossgap_heatmap.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 5: AUC vs Heterogeneity (scatter)
# ================================================================
print("Generating AUC vs heterogeneity scatter...")
fig, ax = plt.subplots(figsize=(8, 5))

for a in alphas:
    for k in clients:
        het = het_std_data[a][k]
        auc = auc_data[a][k]
        ax.scatter(het, auc, color=COLORS_K[k], s=80, zorder=5,
                   edgecolors="white", linewidth=0.5)
        ax.annotate(f"α={a}", (het, auc), fontsize=7,
                    textcoords="offset points", xytext=(5, 5), color="gray")

# Legend for K
for k in clients:
    ax.scatter([], [], color=COLORS_K[k], s=80, label=f"K={k}",
               edgecolors="white", linewidth=0.5)

ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax.axhline(y=iid_auc, color="gray", linestyle="--", linewidth=1.5, alpha=0.5,
           label=f"IID baseline")

ax.set_xlabel("Heterogeneity (Std of Positive Rate Across Clients)")
ax.set_ylabel("MIA AUC")
ax.set_title("MIA Vulnerability vs Data Heterogeneity")
ax.legend(loc="best")

plt.tight_layout()
plt.savefig("experiments/plots/noniid_05_auc_vs_heterogeneity.png", dpi=200, bbox_inches="tight")
plt.close()


# ================================================================
# Figure 6: Combined Summary (2x2)
# ================================================================
print("Generating combined summary...")
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# (0,0) AUC heatmap
ax = axes[0, 0]
mat = to_matrix(auc_data)
norm = TwoSlopeNorm(vmin=mat.min(), vcenter=0.5, vmax=mat.max())
im = ax.imshow(mat, cmap=CMAP_AUC, norm=norm, aspect="auto")
ax.set_xticks(range(len(clients)))
ax.set_xticklabels([str(k) for k in clients])
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("K")
ax.set_ylabel("α")
ax.set_title("(a) MIA AUC")
for i in range(len(alphas)):
    for j in range(len(clients)):
        val = mat[i, j]
        color = "white" if val > 0.7 or val < 0.45 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, shrink=0.8)

# (0,1) AUC vs K
ax = axes[0, 1]
for a in alphas:
    vals = [auc_data[a][k] for k in clients]
    ax.plot(clients, vals, marker="o", linewidth=2, markersize=6,
            label=f"α={a}", color=COLORS_A[a])
ax.axhline(y=iid_auc, color="gray", linestyle="--", linewidth=1.5, label="IID")
ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax.set_xlabel("K")
ax.set_ylabel("AUC")
ax.set_title("(b) AUC vs Number of Clients")
ax.set_xticks(clients)
ax.legend(fontsize=8, loc="best")
ax.set_ylim([0.45, 0.95])

# (1,0) Loss gap heatmap
ax = axes[1, 0]
mat_loss = to_matrix(loss_gap_data)
im = ax.imshow(mat_loss, cmap=CMAP_LOSS, aspect="auto")
ax.set_xticks(range(len(clients)))
ax.set_xticklabels([str(k) for k in clients])
ax.set_yticks(range(len(alphas)))
ax.set_yticklabels([str(a) for a in alphas])
ax.set_xlabel("K")
ax.set_ylabel("α")
ax.set_title("(c) Loss Gap")
for i in range(len(alphas)):
    for j in range(len(clients)):
        val = mat_loss[i, j]
        color = "white" if val > 0.6 else "black"
        ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                fontsize=10, fontweight="bold", color=color)
plt.colorbar(im, ax=ax, shrink=0.8)

# (1,1) AUC vs alpha
ax = axes[1, 1]
for k in clients:
    vals = [auc_data[a][k] for a in alphas]
    ax.plot(alphas, vals, marker="s", linewidth=2, markersize=6,
            label=f"K={k}", color=COLORS_K[k])
ax.axhline(y=iid_auc, color="gray", linestyle="--", linewidth=1.5, label="IID")
ax.axhline(y=0.5, color="black", linestyle=":", linewidth=1, alpha=0.5)
ax.set_xlabel("α (log scale)")
ax.set_ylabel("AUC")
ax.set_title("(d) AUC vs Heterogeneity")
ax.set_xscale("log")
ax.set_xticks(alphas)
ax.set_xticklabels([str(a) for a in alphas])
ax.legend(fontsize=8, loc="best")
ax.set_ylim([0.45, 0.95])

fig.suptitle("Non-IID Federated Learning: Effect on Membership Inference Risk",
             fontsize=15, fontweight="bold", y=1.01)
plt.tight_layout()
plt.savefig("experiments/plots/noniid_06_combined.png", dpi=200, bbox_inches="tight")
plt.close()

print("\nAll plots saved to experiments/plots/")
print("  noniid_01_auc_heatmap.png")
print("  noniid_02_auc_vs_K.png")
print("  noniid_03_auc_vs_alpha.png")
print("  noniid_04_lossgap_heatmap.png")
print("  noniid_05_auc_vs_heterogeneity.png")
print("  noniid_06_combined.png")
