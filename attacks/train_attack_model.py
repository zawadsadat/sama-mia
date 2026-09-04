"""
Train and evaluate MIA attack classifiers.

Usage:
    python attacks/train_attack_model.py --no-dp
    python attacks/train_attack_model.py --dp
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier

from models.model import AttackModel


def report_metrics(y_true, scores, name):
    predicted = (scores > 0.5).astype(float)
    acc = accuracy_score(y_true, predicted)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = 0.5

    print(f"\n--- {name} ---")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  AUC:      {auc:.4f}")

    fpr, tpr, _ = roc_curve(y_true, scores)
    for target_fpr in [0.01, 0.05, 0.10]:
        idx = np.argmin(np.abs(fpr - target_fpr))
        print(f"  TPR @ {target_fpr*100:.0f}% FPR: {tpr[idx]:.4f}")

    return auc


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dp", action="store_true")
    group.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()

    suffix = "_dp" if args.dp else "_nodp"
    mode_str = "WITH DP" if args.dp else "WITHOUT DP (baseline)"

    print(f"=== MIA Attack Evaluation — {mode_str} ===\n")

    feat_path = f"experiments/attack_features{suffix}.npy"
    label_path = f"experiments/attack_labels{suffix}.npy"

    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        sys.exit(1)

    X = np.load(feat_path)
    y = np.load(label_path)

    print(f"Raw dataset: {len(X)} samples "
          f"({y.sum():.0f} members, {len(y) - y.sum():.0f} non-members)")

    # Balance
    member_idx = np.where(y == 1)[0]
    nonmember_idx = np.where(y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    member_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nonmember_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)

    balanced_idx = np.concatenate([member_sample, nonmember_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = X[balanced_idx], y[balanced_idx]
    print(f"Balanced dataset: {len(X_bal)} samples")

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # Attack 1: Loss threshold
    loss_scores_test = -X_test[:, 2]
    loss_min, loss_max = loss_scores_test.min(), loss_scores_test.max()
    loss_scores_norm = (loss_scores_test - loss_min) / (loss_max - loss_min + 1e-10)
    report_metrics(y_test, loss_scores_norm, "Attack 1: Loss Threshold")

    # Attack 2: Random Forest
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    rf_probs = rf.predict_proba(X_test)[:, 1]
    report_metrics(y_test, rf_probs, "Attack 2: Random Forest")
    print(f"  Feature importances: {dict(zip(['prob0','prob1','loss','entropy'], rf.feature_importances_.round(3)))}")

    # Attack 3: Neural Network
    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)

    model = AttackModel(input_dim=4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_train_t).squeeze()
        loss = criterion(outputs, y_train_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"  NN Epoch {epoch} Loss: {loss.item():.4f}")

    with torch.no_grad():
        nn_preds = model(X_test_t).squeeze().numpy()
    report_metrics(y_test, nn_preds, "Attack 3: Neural Network")

    torch.save(model.state_dict(), f"experiments/attack_model{suffix}.pt")
    print(f"\nAttack model saved to experiments/attack_model{suffix}.pt")


if __name__ == "__main__":
    main()
