"""
Privacy Budget (ε) Sweep.

Usage:
    python experiments/run_epsilon_sweep.py
    python experiments/run_epsilon_sweep.py --noise-multipliers 0.5 1.0 2.0
"""

import sys
import os
import argparse
import time

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, roc_curve, balanced_accuracy_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from opacus import PrivacyEngine

from models.model import TargetModel
from utils.data_loader import (
    get_raw_data, get_input_dim, get_dataset_info,
    scales_per_split, fit_apply_scaler,
)
from utils.objective import make_criterion
from utils.config import (
    OBJECTIVE,
    LR, BATCH_SIZE, TEST_SIZE, MAX_GRAD_NORM,
    SHADOW_EPOCHS_DP, NUM_SHADOW_MODELS,
)
from utils.utility_metrics import evaluate_utility, format_utility, UTILITY_COLUMNS

import json

# Per-sigma checkpoints. Each noise level writes its own file the moment it
# finishes, so a crash costs one sigma (~80 min) instead of the whole sweep
# (~11 h). Re-running the same command skips any sigma already on disk.
CKPT_DIR = "experiments/eps_ckpt"


def _ckpt_path(dataset, nm):
    return f"{CKPT_DIR}/{dataset}_sigma{nm}.json"


def _save_ckpt(dataset, nm, result):
    os.makedirs(CKPT_DIR, exist_ok=True)
    with open(_ckpt_path(dataset, nm), "w") as f:
        json.dump({k: (None if isinstance(v, float) and np.isinf(v) else v)
                   for k, v in result.items()}, f, indent=2, default=float)


def _load_ckpt(dataset, nm):
    path = _ckpt_path(dataset, nm)
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train_model_dp(X_train, y_train, input_dim, noise_multiplier, epochs):
    """Train a model with DP-SGD at a given noise level."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TargetModel(input_dim=input_dim).to(device)
    criterion, _ = make_criterion(y_train, OBJECTIVE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    privacy_engine = PrivacyEngine()
    model, optimizer, loader = privacy_engine.make_private(
        module=model,
        optimizer=optimizer,
        data_loader=loader,
        noise_multiplier=noise_multiplier,
        max_grad_norm=MAX_GRAD_NORM,
    )

    for epoch in range(epochs):
        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    epsilon = privacy_engine.get_epsilon(delta=1e-5)
    return model, epsilon


def train_no_dp(X_train, y_train, input_dim, epochs):
    """Train without DP (ε = ∞ baseline)."""
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TargetModel(input_dim=input_dim).to(device)
    criterion, _ = make_criterion(y_train, OBJECTIVE)
    optimizer = optim.Adam(model.parameters(), lr=LR)

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    return model


def extract_attack_features(model, X, y):
    """Extract [prob0, prob1, loss, entropy] per sample."""
    features = []
    X_t = torch.tensor(X, dtype=torch.float32).to(device)
    y_t = torch.tensor(y, dtype=torch.float32).to(device)

    with torch.no_grad():
        preds = model(X_t).squeeze(-1)
        for i, p in enumerate(preds):
            c = p.item()
            loss = F.binary_cross_entropy(p, y_t[i], reduction='none').item()
            entropy = -(c * math.log(c + 1e-10) + (1-c) * math.log(1-c + 1e-10))
            features.append([1-c, c, loss, entropy])

    return np.array(features)


def run_mia(shadow_features_list, shadow_labels_list):
    """Train RF attack and return metrics."""
    X_attack = np.vstack(shadow_features_list)
    y_attack = np.concatenate(shadow_labels_list)

    # Balance
    member_idx = np.where(y_attack == 1)[0]
    nonmember_idx = np.where(y_attack == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)
    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X_attack[balanced_idx], y_attack[balanced_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)
    probs = rf.predict_proba(X_te)[:, 1]

    auc = roc_auc_score(y_te, probs)
    bal_acc = balanced_accuracy_score(y_te, (probs > 0.5).astype(float))

    fpr, tpr, _ = roc_curve(y_te, probs)
    tpr_at = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at[target_fpr] = tpr[idx]

    # Loss gap
    member_mask = y_attack == 1
    loss_gap = abs(X_attack[member_mask, 2].mean() - X_attack[~member_mask, 2].mean())

    return {
        "auc": auc, "bal_acc": bal_acc,
        "tpr_1": tpr_at[0.01], "tpr_5": tpr_at[0.05], "tpr_10": tpr_at[0.10],
        "loss_gap": loss_gap,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--noise-multipliers", nargs="+", type=float,
                        default=[0.3, 0.5, 0.8, 1.0, 1.3, 2.0, 3.0, 5.0])
    parser.add_argument("--num-shadow", type=int, default=10,
                        help="Shadow models per noise level (fewer = faster)")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--no-resume", action="store_true",
                        help="recompute every sigma even if a checkpoint exists")
    args = parser.parse_args()

    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"{'=' * 70}")
    print(f"  Privacy Budget (ε) Sweep")
    print(f"  Dataset: {info['name']} ({info['samples']} samples)")
    print(f"  Noise multipliers: {args.noise_multipliers}")
    print(f"  Shadow models per config: {args.num_shadow}")
    print(f"  Epochs: {args.epochs}")
    print(f"{'=' * 70}\n")

    per_split = scales_per_split()
    if per_split:
        print("Scaling: per-split (fit inside each target/shadow split)\n")
    X_all, y_all = get_raw_data(scale=not per_split)

    # --- No-DP baseline first ---
    print("Training no-DP baseline...")
    X_train_base, X_test_base, y_train_base, y_test_base = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=42, stratify=y_all
    )
    if per_split:
        X_train_base, X_test_base, _, _ = fit_apply_scaler(
            X_train_base, X_test_base)

    nodp_model = train_no_dp(X_train_base, y_train_base, input_dim, args.epochs)

    with torch.no_grad():
        X_te_t = torch.tensor(X_test_base, dtype=torch.float32).to(device)
        y_te_t = torch.tensor(y_test_base, dtype=torch.float32).to(device)
        probs = nodp_model(X_te_t).squeeze(-1).cpu().numpy()

    # Reviewer comment 4: report prevalence-robust utility, not bare accuracy.
    nodp_util = evaluate_utility(probs, y_test_base)
    nodp_acc = nodp_util["acc"]   # keep the old name; downstream code uses it
    print("\n  No-DP baseline utility:")
    print(format_utility(nodp_util, prefix="    "))

    # Shadow models for no-DP
    print("Training no-DP shadow models...")
    nodp_features = []
    nodp_labels = []

    for s in range(args.num_shadow):
        X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
            X_all, y_all, test_size=TEST_SIZE, random_state=s, stratify=y_all
        )
        if per_split:
            X_s_tr, X_s_te, _, _ = fit_apply_scaler(X_s_tr, X_s_te)
        shadow = train_no_dp(X_s_tr, y_s_tr, input_dim, 30)

        mem_feat = extract_attack_features(shadow, X_s_tr, y_s_tr)
        nonmem_feat = extract_attack_features(shadow, X_s_te, y_s_te)

        nodp_features.append(mem_feat)
        nodp_labels.append(np.ones(len(mem_feat)))
        nodp_features.append(nonmem_feat)
        nodp_labels.append(np.zeros(len(nonmem_feat)))

        print(f"  Shadow {s+1}/{args.num_shadow} done")

    nodp_mia = run_mia(nodp_features, nodp_labels)

    _save_ckpt(info["name"], "nodp", {
        "noise_multiplier": float("inf"), "epsilon": float("inf"),
        "test_acc": nodp_acc,
        **{f"util_{k}": nodp_util[k] for k in UTILITY_COLUMNS},
        **nodp_mia,
    })
    print(f"  [checkpoint] no-DP baseline saved to "
          f"{_ckpt_path(info['name'], 'nodp')}")

    print(f"\n  No-DP: TestAcc={nodp_acc:.4f}, MIA AUC={nodp_mia['auc']:.4f}, "
          f"LossGap={nodp_mia['loss_gap']:.4f}")

    all_results = []

    # --- Sweep noise multipliers ---
    for nm in args.noise_multipliers:
        if not args.no_resume:
            cached = _load_ckpt(info["name"], nm)
            if cached is not None:
                print(f"\n  [resume] sigma={nm} already computed "
                      f"({_ckpt_path(info['name'], nm)}) -- skipping. "
                      f"Use --no-resume to force recomputation.")
                all_results.append(cached)
                continue

        print(f"\n{'─' * 70}")
        print(f"  noise_multiplier = {nm}")
        print(f"{'─' * 70}")

        start = time.time()

        # Train target model
        target, epsilon = train_model_dp(
            X_train_base, y_train_base, input_dim, nm, args.epochs
        )

        with torch.no_grad():
            probs = target(X_te_t).squeeze(-1).cpu().numpy()

        util = evaluate_utility(probs, y_test_base)
        test_acc = util["acc"]
        print(f"\n  Utility at sigma={nm}:")
        print(format_utility(util, prefix="    "))
        if util["degenerate"]:
            print("    !! constant predictor: MIA/FMPAS at this sigma measures"
                  " the absence of a model")

        # Train shadow models with same noise
        shadow_features = []
        shadow_labels = []

        for s in range(args.num_shadow):
            X_s_tr, X_s_te, y_s_tr, y_s_te = train_test_split(
                X_all, y_all, test_size=TEST_SIZE, random_state=s, stratify=y_all
            )
            if per_split:
                X_s_tr, X_s_te, _, _ = fit_apply_scaler(X_s_tr, X_s_te)

            shadow, _ = train_model_dp(X_s_tr, y_s_tr, input_dim, nm, args.epochs)

            mem_feat = extract_attack_features(shadow, X_s_tr, y_s_tr)
            nonmem_feat = extract_attack_features(shadow, X_s_te, y_s_te)

            shadow_features.append(mem_feat)
            shadow_labels.append(np.ones(len(mem_feat)))
            shadow_features.append(nonmem_feat)
            shadow_labels.append(np.zeros(len(nonmem_feat)))

            print(f"  Shadow {s+1}/{args.num_shadow} done")

        mia = run_mia(shadow_features, shadow_labels)
        elapsed = time.time() - start

        result = {
            "noise_multiplier": nm,
            "epsilon": epsilon,
            "test_acc": test_acc,
            **{f"util_{k}": util[k] for k in UTILITY_COLUMNS},
            **mia,
            "time_sec": elapsed,
        }
        all_results.append(result)
        _save_ckpt(info["name"], nm, result)
        print(f"  [checkpoint] saved {_ckpt_path(info['name'], nm)}")

        print(f"\n  ε={epsilon:.2f}, TestAcc={test_acc:.4f}, "
              f"MIA AUC={mia['auc']:.4f}, LossGap={mia['loss_gap']:.4f} "
              f"({elapsed:.0f}s)")

    # --- Summary ---
    print(f"\n\n{'=' * 80}")
    print(f"  SUMMARY: Privacy Budget Sweep")
    print(f"{'=' * 80}")
    print(f"  {'NoiseMul':>8} {'ε':>8} {'TestAcc':>8} {'MajAcc':>8} {'TaskBal':>8} "
          f"{'TaskAUC':>8} {'F1':>7} {'MIA_AUC':>8} {'LossGap':>8}  Deg")
    print(f"  {'-' * 90}")

    # No-DP row
    print(f"  {'∞ (none)':>8} {'∞':>8} {nodp_acc:>8.4f} "
          f"{nodp_util['majority_acc']:>8.4f} {nodp_util['bal_acc']:>8.4f} "
          f"{nodp_util['auroc']:>8.4f} {nodp_util['f1']:>7.4f} "
          f"{nodp_mia['auc']:>8.4f} {nodp_mia['loss_gap']:>8.4f}  "
          f"{'YES' if nodp_util['degenerate'] else '-'}")

    for r in sorted(all_results, key=lambda x: -x["epsilon"]):
        print(f"  {r['noise_multiplier']:>8.1f} {r['epsilon']:>8.2f} "
              f"{r['test_acc']:>8.4f} {r['util_majority_acc']:>8.4f} "
              f"{r['util_bal_acc']:>8.4f} {r['util_auroc']:>8.4f} "
              f"{r['util_f1']:>7.4f} {r['auc']:>8.4f} {r['loss_gap']:>8.4f}  "
              f"{'YES' if r['util_degenerate'] else '-'}")

    print(f"  {'-' * 90}")
    print("  TaskBal/TaskAUC/F1 = TARGET-TASK utility (not attack metrics).")
    print("  Deg=YES: model predicts one class for every sample; its TestAcc")
    print("  equals MajAcc and no privacy conclusion can be drawn from it.")

    # --- Save ---
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/epsilon_sweep_results.npy", {
        "nodp": {"test_acc": nodp_acc,
                 **{f"util_{k}": nodp_util[k] for k in UTILITY_COLUMNS},
                 **nodp_mia},
        "sweep": all_results,
    }, allow_pickle=True)

    print(f"\nResults saved to experiments/epsilon_sweep_results.npy")

    # --- Save as CSV for audit score ---
    import pandas as pd
    # NOTE: "BalAcc" is the ATTACK model's balanced accuracy (from run_mia).
    # "util_bal_acc" is the TARGET TASK's balanced accuracy. They are different
    # quantities and were easy to confuse in the original CSV.
    csv_rows = [{"NoiseMul": float("inf"), "epsilon": float("inf"),
                 "TestAcc": nodp_acc, "AUC": nodp_mia["auc"],
                 "BalAcc": nodp_mia["bal_acc"], "T@1%": nodp_mia["tpr_1"],
                 "T@5%": nodp_mia["tpr_5"], "T@10%": nodp_mia["tpr_10"],
                 "LossGap": nodp_mia["loss_gap"],
                 **{f"util_{k}": nodp_util[k] for k in UTILITY_COLUMNS}}]
    for r in sorted(all_results, key=lambda x: -x["epsilon"]):
        csv_rows.append({"NoiseMul": r["noise_multiplier"], "epsilon": r["epsilon"],
                         "TestAcc": r["test_acc"], "AUC": r["auc"],
                         "BalAcc": r["bal_acc"], "T@1%": r["tpr_1"],
                         "T@5%": r["tpr_5"], "T@10%": r["tpr_10"],
                         "LossGap": r["loss_gap"],
                         **{f"util_{k}": r.get(f"util_{k}") for k in UTILITY_COLUMNS}})
    pd.DataFrame(csv_rows).to_csv("experiments/epsilon_sweep_results.csv", index=False)
    print(f"Results saved to experiments/epsilon_sweep_results.csv")


if __name__ == "__main__":
    main()
