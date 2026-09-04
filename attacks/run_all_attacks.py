"""
Comprehensive MIA attack suite.

Usage:
    python attacks/run_all_attacks.py --no-dp
    python attacks/run_all_attacks.py --dp
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
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve, balanced_accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from torch.utils.data import TensorDataset, DataLoader

from models.model import TargetModel, AttackModel
from utils.data_loader import get_raw_data, get_input_dim, get_dataset_info
from utils.config import (
    NUM_SHADOW_MODELS, SHADOW_EPOCHS_NO_DP, SHADOW_EPOCHS_DP,
    LR, BATCH_SIZE, TEST_SIZE,
    NOISE_MULTIPLIER, MAX_GRAD_NORM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# Helper functions
# ================================================================

def report_metrics(y_true, scores, name):
    """Print metrics and return dict."""
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

    print(f"  {name:<35} AUC={auc:.4f}  Acc={acc:.4f}  BalAcc={bal_acc:.4f}  "
          f"T@1%={tpr_at[0.01]:.4f}  T@5%={tpr_at[0.05]:.4f}  T@10%={tpr_at[0.10]:.4f}")

    return {
        "name": name, "auc": auc, "acc": acc, "bal_acc": bal_acc,
        "tpr_1": tpr_at[0.01], "tpr_5": tpr_at[0.05], "tpr_10": tpr_at[0.10],
        "fpr": fpr, "tpr": tpr,
    }


def normalize_scores(scores):
    """Normalize to [0, 1]."""
    s_min, s_max = scores.min(), scores.max()
    if s_max - s_min < 1e-10:
        return np.full_like(scores, 0.5)
    return (scores - s_min) / (s_max - s_min)


def extract_per_sample_stats(model, X, y):
    """Extract loss, confidence, entropy for each sample."""
    model.eval()
    losses = []
    confidences = []
    entropies = []

    with torch.no_grad():
        preds = model(X).squeeze()

        for i, p in enumerate(preds):
            conf = p.item()
            # Confidence in the correct class
            correct_conf = conf if y[i].item() == 1 else (1 - conf)
            confidences.append(correct_conf)

            loss = F.binary_cross_entropy(p, y[i], reduction='none').item()
            losses.append(loss)

            entropy = -(
                conf * math.log(conf + 1e-10) +
                (1 - conf) * math.log(1 - conf + 1e-10)
            )
            entropies.append(entropy)

    return np.array(losses), np.array(confidences), np.array(entropies)


# ================================================================
# Train shadow models and collect per-model statistics for LiRA
# ================================================================

def train_shadow_models(X_all, y_all, input_dim, use_dp, num_models=None):
    """
    Train shadow models and return per-sample loss distributions.

    For LiRA, we need the distribution of losses for each sample
    across shadow models where it was IN vs OUT of training.

    Returns:
        shadow_losses_in[i]  = list of losses for sample i when it was in training
        shadow_losses_out[i] = list of losses for sample i when it was out of training
        attack_features      = (X, y) for the standard shadow model attack
    """
    if num_models is None:
        num_models = NUM_SHADOW_MODELS

    epochs = SHADOW_EPOCHS_DP if use_dp else SHADOW_EPOCHS_NO_DP
    n_total = len(X_all)

    # Per-sample loss tracking for LiRA
    shadow_losses_in = {i: [] for i in range(n_total)}
    shadow_losses_out = {i: [] for i in range(n_total)}

    # Per-sample confidence tracking for calibrated attack
    shadow_confs_in = {i: [] for i in range(n_total)}
    shadow_confs_out = {i: [] for i in range(n_total)}

    # Standard attack features
    attack_X = []
    attack_y = []

    for model_id in range(num_models):
        print(f"  Shadow Model {model_id + 1}/{num_models}", end="", flush=True)

        X_train_np, X_test_np, y_train_np, y_test_np = train_test_split(
            X_all, y_all,
            test_size=TEST_SIZE,
            random_state=model_id,
            stratify=y_all,
        )

        # Track which global indices are in training vs test
        # Use the same split logic to get indices
        np.random.seed(model_id)
        from sklearn.model_selection import StratifiedShuffleSplit
        sss = StratifiedShuffleSplit(n_splits=1, test_size=TEST_SIZE, random_state=model_id)
        train_idx, test_idx = next(sss.split(X_all, y_all))

        X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = TargetModel(input_dim=input_dim).to(device)
        criterion = nn.BCELoss()
        optimizer = optim.Adam(model.parameters(), lr=LR)

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

        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

        # Extract per-sample stats
        X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)
        y_all_t = torch.tensor(y_all, dtype=torch.float32).to(device)

        losses_all, confs_all, entropies_all = extract_per_sample_stats(
            model, X_all_t, y_all_t
        )

        # Record IN/OUT losses for each sample
        train_set = set(train_idx.tolist())
        for i in range(n_total):
            if i in train_set:
                shadow_losses_in[i].append(losses_all[i])
                shadow_confs_in[i].append(confs_all[i])
            else:
                shadow_losses_out[i].append(losses_all[i])
                shadow_confs_out[i].append(confs_all[i])

        # Standard attack features (prob0, prob1, loss, entropy)
        model.eval()
        with torch.no_grad():
            preds_train = model(X_train).squeeze()
            preds_test = model(X_test).squeeze()

        for i, p in enumerate(preds_train):
            c = p.item()
            l = F.binary_cross_entropy(p, y_train[i], reduction='none').item()
            e = -(c * math.log(c + 1e-10) + (1-c) * math.log(1-c + 1e-10))
            attack_X.append([1-c, c, l, e])
            attack_y.append(1)

        for i, p in enumerate(preds_test):
            c = p.item()
            l = F.binary_cross_entropy(p, y_test[i], reduction='none').item()
            e = -(c * math.log(c + 1e-10) + (1-c) * math.log(1-c + 1e-10))
            attack_X.append([1-c, c, l, e])
            attack_y.append(0)

        train_acc = ((model(X_train).squeeze() > 0.5).float() == y_train).float().mean()
        test_acc = ((model(X_test).squeeze() > 0.5).float() == y_test).float().mean()
        print(f"  train={train_acc:.4f} test={test_acc:.4f}")

    return {
        "losses_in": shadow_losses_in,
        "losses_out": shadow_losses_out,
        "confs_in": shadow_confs_in,
        "confs_out": shadow_confs_out,
        "attack_X": np.array(attack_X),
        "attack_y": np.array(attack_y),
    }


# ================================================================
# Attack implementations
# ================================================================

def attack_loss_threshold(X_test, y_test):
    """Attack 1: Lower loss → more likely member."""
    scores = normalize_scores(-X_test[:, 2])
    return report_metrics(y_test, scores, "1. Loss Threshold")


def attack_confidence_threshold(X_test, y_test):
    """Attack 2: Higher confidence in correct class → more likely member."""
    # prob1 is confidence in class 1; for correct-class confidence we need the label
    # But in the attack features, prob1 = sigmoid output regardless of true label
    # Use the max of prob0, prob1 as a proxy for confidence
    confidence = np.maximum(X_test[:, 0], X_test[:, 1])
    scores = normalize_scores(confidence)
    return report_metrics(y_test, scores, "2. Confidence Threshold")


def attack_entropy_threshold(X_test, y_test):
    """Attack 3: Lower entropy → more likely member."""
    scores = normalize_scores(-X_test[:, 3])
    return report_metrics(y_test, scores, "3. Entropy Threshold")


def attack_calibrated_loss(shadow_data, X_all, y_all, target_losses, target_labels):
    """
    Attack 4: Calibrated loss attack.

    Normalizes each sample's loss by the average loss across shadow
    models where that sample was NOT in training. This accounts for
    per-sample difficulty — some samples are inherently harder.

    Score = loss_out_mean - target_loss
    (higher means target loss is unusually low → likely member)
    """
    scores = []
    valid_mask = []

    for i in range(len(target_losses)):
        out_losses = shadow_data["losses_out"].get(i, [])
        if len(out_losses) >= 2:
            ref_mean = np.mean(out_losses)
            # If target loss is much lower than reference, likely member
            calibrated = ref_mean - target_losses[i]
            scores.append(calibrated)
            valid_mask.append(True)
        else:
            scores.append(0.0)
            valid_mask.append(False)

    scores = np.array(scores)
    valid_mask = np.array(valid_mask)

    if valid_mask.sum() < 10:
        print("  4. Calibrated Loss               NOT ENOUGH DATA")
        return None

    scores_valid = normalize_scores(scores[valid_mask])
    labels_valid = target_labels[valid_mask]

    return report_metrics(labels_valid, scores_valid, "4. Calibrated Loss")


def attack_lira(shadow_data, target_losses, target_labels):
    """
    Attack 5: LiRA-style likelihood ratio attack (Carlini et al., 2022).

    For each sample, fits Gaussians to the loss distribution when the
    sample is IN vs OUT of shadow model training sets. The attack score
    is the log-likelihood ratio: log P(loss | IN) - log P(loss | OUT).

    Simplified version: uses mean and variance of IN/OUT loss distributions.
    """
    scores = []
    valid_mask = []

    for i in range(len(target_losses)):
        in_losses = shadow_data["losses_in"].get(i, [])
        out_losses = shadow_data["losses_out"].get(i, [])

        # Need at least 2 samples in each to estimate variance
        if len(in_losses) >= 2 and len(out_losses) >= 2:
            mu_in = np.mean(in_losses)
            sigma_in = max(np.std(in_losses), 1e-6)

            mu_out = np.mean(out_losses)
            sigma_out = max(np.std(out_losses), 1e-6)

            target_loss = target_losses[i]

            # Log-likelihood under IN distribution
            ll_in = -0.5 * ((target_loss - mu_in) / sigma_in) ** 2 - np.log(sigma_in)

            # Log-likelihood under OUT distribution
            ll_out = -0.5 * ((target_loss - mu_out) / sigma_out) ** 2 - np.log(sigma_out)

            # LiRA score: higher means more likely IN (member)
            lira_score = ll_in - ll_out
            scores.append(lira_score)
            valid_mask.append(True)
        else:
            scores.append(0.0)
            valid_mask.append(False)

    scores = np.array(scores)
    valid_mask = np.array(valid_mask)

    if valid_mask.sum() < 10:
        print("  5. LiRA (Likelihood Ratio)        NOT ENOUGH DATA")
        return None

    scores_valid = normalize_scores(scores[valid_mask])
    labels_valid = target_labels[valid_mask]

    return report_metrics(labels_valid, scores_valid, "5. LiRA (Likelihood Ratio)")


def attack_rf(X_train, y_train, X_test, y_test):
    """Attack 6: Random Forest on all 4 features."""
    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5,
        random_state=42, class_weight='balanced'
    )
    rf.fit(X_train, y_train)
    probs = rf.predict_proba(X_test)[:, 1]
    result = report_metrics(y_test, probs, "6. Random Forest")
    result["importances"] = dict(zip(
        ["prob0", "prob1", "loss", "entropy"],
        rf.feature_importances_.round(3)
    ))
    return result


def attack_nn(X_train, y_train, X_test, y_test):
    """Attack 7: Neural network on all 4 features."""
    X_tr_t = torch.tensor(X_train, dtype=torch.float32)
    y_tr_t = torch.tensor(y_train, dtype=torch.float32)
    X_te_t = torch.tensor(X_test, dtype=torch.float32)

    model = AttackModel(input_dim=4)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        optimizer.zero_grad()
        outputs = model(X_tr_t).squeeze()
        loss = criterion(outputs, y_tr_t)
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        preds = model(X_te_t).squeeze().numpy()

    return report_metrics(y_test, preds, "7. Neural Network")


# ================================================================
# Main
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dp", action="store_true")
    group.add_argument("--no-dp", action="store_true")
    args = parser.parse_args()

    use_dp = args.dp
    suffix = "_dp" if use_dp else "_nodp"
    mode_str = "WITH DP-SGD" if use_dp else "WITHOUT DP (baseline)"

    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"{'=' * 70}")
    print(f"  Comprehensive MIA Attack Suite — {mode_str}")
    print(f"  Dataset: {info['name']} ({info['samples']} samples, {info['features']} features)")
    print(f"{'=' * 70}\n")

    X_all, y_all = get_raw_data()

    # --- Train shadow models (with LiRA tracking) ---
    print("Training shadow models...")
    shadow_data = train_shadow_models(X_all, y_all, input_dim, use_dp)

    # --- Prepare attack dataset ---
    attack_X = shadow_data["attack_X"]
    attack_y = shadow_data["attack_y"]

    # Balance
    member_idx = np.where(attack_y == 1)[0]
    nonmember_idx = np.where(attack_y == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)

    balanced_idx = np.concatenate([m_sample, nm_sample])
    perm = np.random.RandomState(42).permutation(len(balanced_idx))
    balanced_idx = balanced_idx[perm]

    X_bal, y_bal = attack_X[balanced_idx], attack_y[balanced_idx]

    X_train, X_test, y_train, y_test = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=42, stratify=y_bal
    )

    # --- Train a reference target model for calibrated/LiRA attacks ---
    print("\nTraining reference target model for calibrated attacks...")
    X_target_train, X_target_test, y_target_train, y_target_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=99, stratify=y_all
    )

    X_tt = torch.tensor(X_target_train, dtype=torch.float32).to(device)
    y_tt = torch.tensor(y_target_train, dtype=torch.float32).to(device)

    target_dataset = TensorDataset(X_tt, y_tt)
    target_loader = DataLoader(target_dataset, batch_size=BATCH_SIZE, shuffle=True)

    target_model = TargetModel(input_dim=input_dim).to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(target_model.parameters(), lr=LR)

    target_epochs = SHADOW_EPOCHS_DP if use_dp else SHADOW_EPOCHS_NO_DP

    if use_dp:
        from opacus import PrivacyEngine
        pe = PrivacyEngine()
        target_model, optimizer, target_loader = pe.make_private(
            module=target_model, optimizer=optimizer, data_loader=target_loader,
            noise_multiplier=NOISE_MULTIPLIER, max_grad_norm=MAX_GRAD_NORM,
        )

    for epoch in range(target_epochs):
        for xb, yb in target_loader:
            optimizer.zero_grad()
            out = target_model(xb).squeeze(-1)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()

    # Get target model losses on all data
    X_all_t = torch.tensor(X_all, dtype=torch.float32).to(device)
    y_all_t = torch.tensor(y_all, dtype=torch.float32).to(device)
    target_losses, target_confs, target_entropies = extract_per_sample_stats(
        target_model, X_all_t, y_all_t
    )

    # Labels: 1 = member of target training set, 0 = non-member
    target_labels = np.zeros(len(X_all))
    # Find which indices were in the target training set
    sss = __import__('sklearn.model_selection', fromlist=['StratifiedShuffleSplit']).StratifiedShuffleSplit(
        n_splits=1, test_size=TEST_SIZE, random_state=99
    )
    target_train_idx, _ = next(sss.split(X_all, y_all))
    target_labels[target_train_idx] = 1

    # --- Run all attacks ---
    print(f"\n{'=' * 70}")
    print(f"  Attack Results — {mode_str}")
    print(f"{'=' * 70}")
    print(f"  {'Attack':<35} {'AUC':>6}  {'Acc':>6}  {'BalAcc':>6}  "
          f"{'T@1%':>6}  {'T@5%':>6}  {'T@10%':>6}")
    print(f"  {'-' * 65}")

    results = []

    # 1-3: Threshold attacks (on balanced shadow-model test set)
    results.append(attack_loss_threshold(X_test, y_test))
    results.append(attack_confidence_threshold(X_test, y_test))
    results.append(attack_entropy_threshold(X_test, y_test))

    # 4: Calibrated loss (on target model, all data)
    r4 = attack_calibrated_loss(shadow_data, X_all, y_all, target_losses, target_labels)
    if r4:
        results.append(r4)

    # 5: LiRA (on target model, all data)
    r5 = attack_lira(shadow_data, target_losses, target_labels)
    if r5:
        results.append(r5)

    # 6-7: Learned attacks (on balanced shadow-model test set)
    r6 = attack_rf(X_train, y_train, X_test, y_test)
    results.append(r6)
    if "importances" in r6:
        print(f"    Feature importances: {r6['importances']}")

    results.append(attack_nn(X_train, y_train, X_test, y_test))

    # --- Save results ---
    os.makedirs("experiments", exist_ok=True)

    # Save attack features for plotting
    np.save(f"experiments/attack_features{suffix}.npy", attack_X)
    np.save(f"experiments/attack_labels{suffix}.npy", attack_y)

    # Save results summary
    summary = {r["name"]: {k: v for k, v in r.items() if k not in ["fpr", "tpr", "name"]}
               for r in results}
    np.save(f"experiments/attack_results{suffix}.npy", summary, allow_pickle=True)

    # Save ROC data for plotting
    roc_data = {r["name"]: {"fpr": r["fpr"], "tpr": r["tpr"], "auc": r["auc"]}
                for r in results if "fpr" in r}
    np.save(f"experiments/roc_data{suffix}.npy", roc_data, allow_pickle=True)

    print(f"\nResults saved to experiments/attack_results{suffix}.npy")
    print(f"ROC data saved to experiments/roc_data{suffix}.npy")


if __name__ == "__main__":
    main()
