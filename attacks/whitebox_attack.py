"""
White-box Gradient-based Membership Inference Attack.

Usage:
    python attacks/whitebox_attack.py --no-dp
    python attacks/whitebox_attack.py --dp
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    balanced_accuracy_score
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from models.model import TargetModel
from utils.data_loader import (
    get_raw_data, get_input_dim, get_dataset_info,
    scales_per_split, fit_apply_scaler,
)
from utils.objective import make_criterion
from utils.splits import (three_way_split, val_score, val_split_usable,
                          EarlyStopper)
from utils.artifact_paths import save_artifact
from utils.utility_metrics import evaluate_utility, format_utility
from utils.config import (
    SHADOW_EPOCHS_NO_DP, SHADOW_EPOCHS_DP,
    LR, BATCH_SIZE, TEST_SIZE,
    NOISE_MULTIPLIER, MAX_GRAD_NORM,
    OBJECTIVE, VAL_FRACTION,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_gradient_features(model, x, y_true, criterion):
    """
    Compute per-sample gradient features.

    The model must be in eval mode but with gradients enabled.
    We do a forward + backward pass for a single sample and
    extract features from the resulting gradients.

    Returns a feature vector:
      [grad_norm_l2, grad_norm_l1, layer0_norm, layer1_norm, ...,
       layer4_norm, grad_var, loss, confidence, entropy]
    """
    model.zero_grad()

    x_input = x.unsqueeze(0).to(device).requires_grad_(False)
    y_target = y_true.unsqueeze(0).to(device)

    # Forward pass
    output = model(x_input).squeeze(-1)
    loss = criterion(output, y_target)

    # Backward pass
    loss.backward()

    # Collect all gradients
    all_grads = []
    layer_norms = []

    for name, param in model.named_parameters():
        if param.grad is not None:
            grad = param.grad.detach().flatten()
            all_grads.append(grad)

            # Per-layer L2 norm
            layer_norms.append(grad.norm(2).item())

    if len(all_grads) == 0:
        # No gradients (shouldn't happen)
        return None

    full_grad = torch.cat(all_grads)

    # Features
    grad_norm_l2 = full_grad.norm(2).item()
    grad_norm_l1 = full_grad.norm(1).item()
    grad_var = full_grad.var().item()
    grad_max = full_grad.abs().max().item()
    grad_mean = full_grad.abs().mean().item()

    # Pad or truncate layer norms to fixed size (10 layers: 5 weight + 5 bias)
    while len(layer_norms) < 10:
        layer_norms.append(0.0)
    layer_norms = layer_norms[:10]

    # Model output features
    conf = output.item()
    sample_loss = loss.item()
    entropy = -(
        conf * math.log(conf + 1e-10) +
        (1 - conf) * math.log(1 - conf + 1e-10)
    )

    features = [
        grad_norm_l2,
        grad_norm_l1,
        grad_var,
        grad_max,
        grad_mean,
    ] + layer_norms + [
        sample_loss,
        conf,
        entropy,
    ]

    model.zero_grad()

    return features


def train_model(X_train, y_train, input_dim, use_dp, epochs,
                X_val=None, y_val=None):
    """
    Train a target model, optionally with DP-SGD.

    Mirrors the target trained by experiments/run_multiseed.py and the shadows
    trained by attacks/shadow_models.py: same objective, and in the realistic
    configuration the same three-way split with early stopping on validation
    AUROC. The white-box audit is only interpretable if the model it probes was
    produced the same way as the models the black-box audit scores.
    """
    X_t = torch.tensor(X_train, dtype=torch.float32).to(device)
    y_t = torch.tensor(y_train, dtype=torch.float32).to(device)

    dataset = TensorDataset(X_t, y_t)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = TargetModel(input_dim=input_dim).to(device)
    criterion, pos_w = make_criterion(y_train, OBJECTIVE)
    optimizer = optim.Adam(model.parameters(), lr=LR)
    print(f"  Objective: {OBJECTIVE} (pos_weight={pos_w:.3f})")

    if use_dp:
        from opacus import PrivacyEngine
        privacy_engine = PrivacyEngine()
        model, optimizer, loader = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=loader,
            noise_multiplier=NOISE_MULTIPLIER,
            max_grad_norm=MAX_GRAD_NORM,
        )

    stopper = (EarlyStopper()
               if (X_val is not None and val_split_usable(y_val)) else None)

    for epoch in range(epochs):
        for xb, yb in loader:
            optimizer.zero_grad()
            out = model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

        if stopper is not None:
            stopper.update(epoch, val_score(model, X_val, y_val, device), model)

    if stopper is not None:
        model = stopper.restore(model)
        print(f"  Early stopping selected epoch {stopper.best_epoch} "
              f"(validation AUROC {stopper.best:.4f})")

    # Get epsilon if DP
    eps = None
    if use_dp:
        eps = privacy_engine.get_epsilon(delta=1e-5)

    return model, eps


def report_metrics(y_true, scores, name):
    """Print metrics."""
    predicted = (scores > 0.5).astype(float)
    acc = accuracy_score(y_true, predicted)
    bal_acc = balanced_accuracy_score(y_true, predicted)

    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5

    fpr, tpr, _ = roc_curve(y_true, scores)
    tpr_at = {}
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        tpr_at[target_fpr] = tpr[idx]

    print(f"  {name:<40} AUC={auc:.4f}  Acc={acc:.4f}  BalAcc={bal_acc:.4f}  "
          f"T@1%={tpr_at[0.01]:.4f}  T@5%={tpr_at[0.05]:.4f}  T@10%={tpr_at[0.10]:.4f}")

    return {
        "name": name, "auc": auc, "acc": acc, "bal_acc": bal_acc,
        "tpr_1": tpr_at[0.01], "tpr_5": tpr_at[0.05], "tpr_10": tpr_at[0.10],
        "fpr": fpr, "tpr": tpr,
    }


def normalize_scores(scores):
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.full_like(scores, 0.5)
    return (scores - s_min) / (s_max - s_min)


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dp", action="store_true")
    group.add_argument("--no-dp", action="store_true")
    parser.add_argument("--max-samples", type=int, default=5000,
                        help="Max samples to extract gradients from (for speed)")
    args = parser.parse_args()

    use_dp = args.dp
    suffix = "_dp" if use_dp else "_nodp"
    mode_str = "WITH DP-SGD" if use_dp else "WITHOUT DP (baseline)"
    epochs = SHADOW_EPOCHS_DP if use_dp else SHADOW_EPOCHS_NO_DP

    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"{'=' * 70}")
    print(f"  White-box Gradient-based MIA — {mode_str}")
    print(f"  Dataset: {info['name']} ({info['samples']} samples, {info['features']} features)")
    print(f"{'=' * 70}\n")

    # Load data
    per_split = scales_per_split()
    if per_split:
        print("Scaling: per-split (fit on the target model's training half)")

    # Realistic configuration: three-way split with early stopping, and the
    # validation rows are EXCLUDED from the gradient audit -- they influenced
    # the released model through the stopping rule, so they are neither members
    # nor untouched non-members, exactly as in the black-box pipeline.
    use_val = (OBJECTIVE == "weighted")
    X_all, y_all = get_raw_data(scale=not per_split)
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = three_way_split(
        X_all, y_all, TEST_SIZE,
        val_frac=(VAL_FRACTION if use_val else 0.0), random_state=42
    )
    if per_split:
        X_train, X_test, sc, med = fit_apply_scaler(X_train, X_test)
        if X_val is not None:
            X_val = sc.transform(
                np.where(np.isnan(X_val), med, X_val)).astype(np.float32)

    print(f"Training set:   {len(X_train)} samples")
    if X_val is not None:
        print(f"Validation set: {len(X_val)} samples (excluded from the audit)")
    print(f"Test set:       {len(X_test)} samples")

    # --- Train target model ---
    print(f"\nTraining target model {mode_str}...")
    model, eps = train_model(X_train, y_train, input_dim, use_dp, epochs,
                             X_val=X_val, y_val=y_val)

    if eps is not None:
        print(f"  Privacy budget: ε = {eps:.2f}")

    # Evaluate model utility
    with torch.no_grad():
        X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
        y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
        probs = model(X_test_t).squeeze(-1).cpu().numpy()

    # Bare accuracy is achievable by a constant majority-class predictor on
    # imbalanced data; report prevalence-robust metrics alongside it.
    utility = evaluate_utility(probs, y_test)
    test_acc = utility["acc"]
    print("\n  Target task utility:")
    print(format_utility(utility, prefix="    "))
    if utility["degenerate"]:
        print("    !! constant predictor: white-box gradient signal below")
        print("       reflects the absence of a model, not privacy.")

    # --- Extract gradient features ---
    # Subsample for speed (gradient extraction is slow — one backward pass per sample)
    max_member = min(args.max_samples, len(X_train))
    max_nonmember = min(args.max_samples, len(X_test))

    member_idx = np.random.RandomState(42).choice(len(X_train), max_member, replace=False)
    nonmember_idx = np.random.RandomState(42).choice(len(X_test), max_nonmember, replace=False)

    print(f"\nExtracting gradient features...")
    print(f"  Members: {max_member}, Non-members: {max_nonmember}")

    criterion = nn.BCELoss()
    all_features = []
    all_labels = []

    # Members
    print(f"  Processing members...", end="", flush=True)
    for i, idx in enumerate(member_idx):
        x = torch.tensor(X_train[idx], dtype=torch.float32)
        y = torch.tensor(y_train[idx], dtype=torch.float32)

        feat = extract_gradient_features(model, x, y, criterion)
        if feat is not None:
            all_features.append(feat)
            all_labels.append(1)

        if (i + 1) % 1000 == 0:
            print(f" {i+1}", end="", flush=True)
    print(f" done ({len([l for l in all_labels if l == 1])} samples)")

    # Non-members
    print(f"  Processing non-members...", end="", flush=True)
    for i, idx in enumerate(nonmember_idx):
        x = torch.tensor(X_test[idx], dtype=torch.float32)
        y = torch.tensor(y_test[idx], dtype=torch.float32)

        feat = extract_gradient_features(model, x, y, criterion)
        if feat is not None:
            all_features.append(feat)
            all_labels.append(0)

        if (i + 1) % 1000 == 0:
            print(f" {i+1}", end="", flush=True)
    print(f" done ({len([l for l in all_labels if l == 0])} samples)")

    X_attack = np.array(all_features)
    y_attack = np.array(all_labels)

    print(f"\n  Feature vector size: {X_attack.shape[1]}")
    print(f"  Total samples: {len(X_attack)} "
          f"({y_attack.sum():.0f} members, {len(y_attack) - y_attack.sum():.0f} non-members)")

    # --- Feature analysis ---
    member_mask = y_attack == 1
    feature_names = [
        "grad_norm_l2", "grad_norm_l1", "grad_var", "grad_max", "grad_mean",
        "layer0", "layer1", "layer2", "layer3", "layer4",
        "layer5", "layer6", "layer7", "layer8", "layer9",
        "loss", "confidence", "entropy",
    ]

    print(f"\n  Feature means:")
    print(f"  {'Feature':<18} {'Members':>12} {'Non-members':>12} {'Ratio':>8}")
    print(f"  {'-' * 52}")
    for i, name in enumerate(feature_names[:len(X_attack[0])]):
        m_mean = X_attack[member_mask, i].mean()
        nm_mean = X_attack[~member_mask, i].mean()
        ratio = m_mean / nm_mean if nm_mean > 1e-10 else float('inf')
        print(f"  {name:<18} {m_mean:>12.6f} {nm_mean:>12.6f} {ratio:>8.3f}")

    # --- Balance classes ---
    member_idx_bal = np.where(y_attack == 1)[0]
    nonmember_idx_bal = np.where(y_attack == 0)[0]
    min_size = min(len(member_idx_bal), len(nonmember_idx_bal))

    m_sample = resample(member_idx_bal, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx_bal, n_samples=min_size, replace=False, random_state=42)
    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X_attack[balanced_idx], y_attack[balanced_idx]

    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # --- Run attacks ---
    print(f"\n{'=' * 70}")
    print(f"  White-box Attack Results — {mode_str}")
    print(f"{'=' * 70}")

    results = []

    # Attack 1: Gradient norm threshold (L2)
    grad_scores = normalize_scores(-X_te[:, 0])  # lower norm → more likely member
    results.append(report_metrics(y_te, grad_scores, "WB-1. Gradient Norm (L2) Threshold"))

    # Attack 2: Gradient norm threshold (L1)
    grad_l1_scores = normalize_scores(-X_te[:, 1])
    results.append(report_metrics(y_te, grad_l1_scores, "WB-2. Gradient Norm (L1) Threshold"))

    # Attack 3: Loss threshold (white-box has same access)
    loss_scores = normalize_scores(-X_te[:, -3])
    results.append(report_metrics(y_te, loss_scores, "WB-3. Loss Threshold"))

    # Attack 4: Random Forest on gradient features only (no loss/conf/entropy)
    n_grad_features = len(feature_names) - 3  # exclude loss, conf, entropy
    rf_grad = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf_grad.fit(X_tr[:, :n_grad_features], y_tr)
    rf_grad_probs = rf_grad.predict_proba(X_te[:, :n_grad_features])[:, 1]
    r4 = report_metrics(y_te, rf_grad_probs, "WB-4. RF (gradient features only)")
    results.append(r4)

    # Print gradient feature importances
    imp = rf_grad.feature_importances_
    print(f"    Top gradient features: ", end="")
    top_idx = np.argsort(imp)[::-1][:5]
    for idx in top_idx:
        print(f"{feature_names[idx]}={imp[idx]:.3f} ", end="")
    print()

    # Attack 5: Random Forest on ALL features (gradient + black-box)
    rf_all = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf_all.fit(X_tr, y_tr)
    rf_all_probs = rf_all.predict_proba(X_te)[:, 1]
    r5 = report_metrics(y_te, rf_all_probs, "WB-5. RF (all features)")
    results.append(r5)

    # Print all feature importances
    imp_all = rf_all.feature_importances_
    print(f"    Top features: ", end="")
    top_idx = np.argsort(imp_all)[::-1][:5]
    for idx in top_idx:
        name = feature_names[idx] if idx < len(feature_names) else f"feat_{idx}"
        print(f"{name}={imp_all[idx]:.3f} ", end="")
    print()

    # --- Comparison: black-box vs white-box ---
    print(f"\n  {'─' * 50}")
    print(f"  Black-box (loss only) AUC:        {results[2]['auc']:.4f}")
    print(f"  White-box (gradients only) AUC:    {results[3]['auc']:.4f}")
    print(f"  White-box (all features) AUC:      {results[4]['auc']:.4f}")
    bb_auc = results[2]['auc']
    wb_auc = results[4]['auc']
    improvement = wb_auc - bb_auc
    print(f"  White-box advantage over BB:       {improvement:+.4f}")

    # --- Save ---
    os.makedirs("experiments", exist_ok=True)
    # Dataset-tagged with a .meta.json sidecar. NOTE the sidecar records
    # n_target_models=1: unlike the black-box pipeline (NUM_SHADOW_MODELS
    # shadows), these gradient features come from a SINGLE target model, so
    # the white-box attack set is far smaller than the black-box one and its
    # advantage estimates carry much wider confidence intervals. Do not take a
    # max over black-box and white-box attacks without accounting for that.
    wb_meta = {
        "n_target_models": 1,
        "n_rows": int(len(y_attack)),
        "n_members": int((y_attack == 1).sum()),
        "n_nonmembers": int((y_attack == 0).sum()),
        "max_samples_per_class": args.max_samples,
        "objective": OBJECTIVE,
        "val_frac": (VAL_FRACTION if OBJECTIVE == "weighted" else 0.0),
        "epochs": epochs,
        "test_size": TEST_SIZE,
        "target_test_acc": float(test_acc),
        "target_bal_acc": float(utility["bal_acc"]),
        "target_degenerate": bool(utility["degenerate"]),
    }
    fp = save_artifact("whitebox_features", info["name"], use_dp, X_attack, wb_meta)
    save_artifact("whitebox_labels", info["name"], use_dp, y_attack, wb_meta)
    print(f"\nWhite-box attack set saved to {fp}")
    print(f"  dataset={info['name']}  condition={'dp' if use_dp else 'nodp'}  "
          f"rows={len(y_attack)}  members={int((y_attack == 1).sum())}  "
          f"(single target model)")
    np.save(f"experiments/whitebox_results{suffix}.npy",
            {r["name"]: {k: v for k, v in r.items() if k not in ["fpr", "tpr"]}
             for r in results}, allow_pickle=True)

    print(f"\nResults saved to experiments/whitebox_results{suffix}.npy")


if __name__ == "__main__":
    main()
