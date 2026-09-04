"""
Run the trained MIA attack model against a target model.

Reports both raw accuracy and balanced accuracy to avoid
misleading numbers when member/non-member counts are imbalanced.

Usage:
    python attacks/evaluate_mia.py --model experiments/target_model.pt
    python attacks/evaluate_mia.py --model experiments/fl_global_model.pt
    python attacks/evaluate_mia.py --model experiments/fl_global_model.pt --attack-model experiments/attack_model_dp.pt
"""

import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import argparse
import torch
import torch.nn.functional as F
import numpy as np
import math
from sklearn.metrics import (
    accuracy_score, roc_auc_score, roc_curve,
    balanced_accuracy_score, confusion_matrix
)
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from models.model import TargetModel, AttackModel
from utils.data_loader import load_data, get_input_dim


def extract_features_for_sample(model, x, y_true):
    with torch.no_grad():
        p = model(x.unsqueeze(0)).squeeze()
        confidence = p.item()
        prob0 = 1 - confidence
        prob1 = confidence
        loss = F.binary_cross_entropy(p, y_true, reduction='none').item()
        entropy = -(
            confidence * math.log(confidence + 1e-10) +
            (1 - confidence) * math.log(1 - confidence + 1e-10)
        )
    return [prob0, prob1, loss, entropy]


def report_metrics(y_true, scores, label):
    """Report all metrics including balanced accuracy."""
    predicted = (scores > 0.5).astype(float)

    acc = accuracy_score(y_true, predicted)
    bal_acc = balanced_accuracy_score(y_true, predicted)
    auc = roc_auc_score(y_true, scores)

    n_members = int(y_true.sum())
    n_nonmembers = len(y_true) - n_members

    print(f"\n--- {label} ---")
    print(f"  Members: {n_members}, Non-members: {n_nonmembers}")
    print(f"  Raw Accuracy:      {acc:.4f}")
    print(f"  Balanced Accuracy: {bal_acc:.4f}")
    print(f"  AUC:               {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, scores)
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f"  TPR @ {target_fpr*100:.0f}% FPR:    {tpr[idx]:.4f}")

    # Confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, predicted).ravel()
    print(f"  TP={tp}, FP={fp}, FN={fn}, TN={tn}")
    if tp + fn > 0:
        print(f"  Member recall:     {tp/(tp+fn):.4f}")
    if tn + fp > 0:
        print(f"  Non-member recall: {tn/(tn+fp):.4f}")

    return {"acc": acc, "bal_acc": bal_acc, "auc": auc}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        help="Path to target model state dict")
    parser.add_argument("--attack-model", type=str, default=None,
                        help="Path to attack model (default: tries both nodp and dp)")
    args = parser.parse_args()

    input_dim = get_input_dim()

    # Load target model
    target = TargetModel(input_dim=input_dim)
    target.load_state_dict(torch.load(args.model, map_location="cpu"))
    target.eval()

    # Load data
    X_train, X_test, y_train, y_test = load_data()

    # Extract features
    print(f"Target model: {args.model}")
    print(f"Extracting attack features...")

    all_features = []
    all_labels = []

    for i in range(len(X_train)):
        features = extract_features_for_sample(target, X_train[i], y_train[i])
        all_features.append(features)
        all_labels.append(1)

    for i in range(len(X_test)):
        features = extract_features_for_sample(target, X_test[i], y_test[i])
        all_features.append(features)
        all_labels.append(0)

    X_attack = np.array(all_features)
    y_attack = np.array(all_labels)

    print(f"{'=' * 50}")
    print(f"MIA Evaluation: {args.model}")
    print(f"{'=' * 50}")
    print(f"Members: {(y_attack == 1).sum()}, Non-members: {(y_attack == 0).sum()}")

    # --- Method 1: Train a fresh RF on the raw features (most reliable) ---
    # Balance the dataset, train RF, evaluate on held-out balanced set
    member_idx = np.where(y_attack == 1)[0]
    nonmember_idx = np.where(y_attack == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)

    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X_attack[balanced_idx], y_attack[balanced_idx]

    from sklearn.model_selection import train_test_split
    X_tr, X_te, y_tr, y_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_tr, y_tr)
    rf_probs = rf.predict_proba(X_te)[:, 1]

    report_metrics(y_te, rf_probs, "Fresh RF (balanced evaluation)")

    # --- Method 2: Use pre-trained attack models if available ---
    attack_models = []
    if args.attack_model:
        attack_models.append(("Custom", args.attack_model))
    else:
        for suffix, label in [("_nodp", "No-DP attack model"), ("_dp", "DP attack model")]:
            path = f"experiments/attack_model{suffix}.pt"
            if os.path.exists(path):
                attack_models.append((label, path))

    X_attack_t = torch.tensor(X_attack, dtype=torch.float32)

    for label, path in attack_models:
        attacker = AttackModel(input_dim=4)
        attacker.load_state_dict(torch.load(path, map_location="cpu"))
        attacker.eval()

        with torch.no_grad():
            preds = attacker(X_attack_t).squeeze().numpy()

        # Report on full (imbalanced) set
        report_metrics(y_attack, preds, f"{label} — full set")

        # Also report on balanced subset
        preds_bal = preds[balanced_idx]
        preds_te = preds_bal[len(X_tr):]  # approximate — use same split indices
        # Better: re-evaluate on the balanced test set
        X_bal_t = torch.tensor(X_bal, dtype=torch.float32)
        with torch.no_grad():
            preds_bal_all = attacker(X_bal_t).squeeze().numpy()

        _, preds_bal_test, _, y_bal_test = train_test_split(
            preds_bal_all, y_bal, test_size=0.3, random_state=42, stratify=y_bal
        )
        report_metrics(y_bal_test, preds_bal_test, f"{label} — balanced evaluation")

    # --- Loss gap ---
    member_mask = y_attack == 1
    loss_gap = abs(X_attack[member_mask, 2].mean() - X_attack[~member_mask, 2].mean())
    print(f"\nMember avg loss:     {X_attack[member_mask, 2].mean():.4f}")
    print(f"Non-member avg loss: {X_attack[~member_mask, 2].mean():.4f}")
    print(f"Loss gap:            {loss_gap:.4f}")
    print(f"Random baseline:     {(y_attack == 1).mean():.4f}")


if __name__ == "__main__":
    main()
