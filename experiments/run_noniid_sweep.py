"""
Non-IID Federated Learning MIA Experiment.

Sweeps over:
    - Dirichlet alpha: {0.1, 0.3, 0.5, 1.0, 10.0}
    - Number of clients K: {3, 5, 10, 20}

For each (alpha, K) configuration:
    1. Partition training data using Dirichlet label skew
    2. Simulate FedAvg locally (no Flower — pure simulation for speed)
    3. Evaluate MIA on the resulting global model
    4. Record AUC, accuracy, TPR@FPR, loss gap

Usage:
    python experiments/run_noniid_sweep.py
    python experiments/run_noniid_sweep.py --alphas 0.1 0.5 --clients 3 5
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

from models.model import TargetModel, AttackModel
from utils.data_loader import get_raw_data, get_input_dim, get_dataset_info
from utils.partition import dirichlet_partition, get_partition_stats, print_partition_stats
from utils.config import LR, BATCH_SIZE, TEST_SIZE

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ================================================================
# FedAvg simulation (no Flower needed — much faster)
# ================================================================

def fedavg_simulate(X_train, y_train, client_indices, input_dim,
                    num_rounds=10, local_epochs=5, lr=0.001):
    """
    Simulate FedAvg training without Flower.

    Args:
        X_train:        full training data (numpy)
        y_train:        full training labels (numpy)
        client_indices: list of K arrays of indices into X_train
        input_dim:      feature dimensionality
        num_rounds:     FL communication rounds
        local_epochs:   local training epochs per round
        lr:             learning rate

    Returns:
        global_model: trained global model
    """
    global_model = TargetModel(input_dim=input_dim).to(device)

    for round_num in range(num_rounds):
        client_state_dicts = []
        client_sizes = []

        for k, idx in enumerate(client_indices):
            if len(idx) == 0:
                continue

            # Create local model with current global weights
            local_model = TargetModel(input_dim=input_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())

            X_k = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
            y_k = torch.tensor(y_train[idx], dtype=torch.float32).to(device)

            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            criterion = nn.BCELoss()

            dataset = TensorDataset(X_k, y_k)
            loader = DataLoader(dataset, batch_size=min(BATCH_SIZE, len(idx)), shuffle=True)

            # Local training
            for epoch in range(local_epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    out = local_model(xb).squeeze(-1)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

            client_state_dicts.append(local_model.state_dict())
            client_sizes.append(len(idx))

        # FedAvg aggregation
        if len(client_state_dicts) == 0:
            continue

        total_samples = sum(client_sizes)
        global_state = {}

        for key in global_model.state_dict().keys():
            weighted_sum = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
            for sd, size in zip(client_state_dicts, client_sizes):
                weighted_sum += sd[key].float() * (size / total_samples)
            global_state[key] = weighted_sum

        global_model.load_state_dict(global_state)

    return global_model


# ================================================================
# MIA evaluation on a given model
# ================================================================

def evaluate_mia_on_model(model, X_train, y_train, X_test, y_test):
    """
    Run MIA (Random Forest) on a trained model.

    Returns dict with AUC, balanced accuracy, TPR@FPR, loss gap.
    """
    model.eval()

    def extract_features(model, X, y):
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

    # Extract features
    member_features = extract_features(model, X_train, y_train)
    nonmember_features = extract_features(model, X_test, y_test)

    X_attack = np.vstack([member_features, nonmember_features])
    y_attack = np.concatenate([np.ones(len(member_features)),
                                np.zeros(len(nonmember_features))])

    # Loss gap
    loss_gap = abs(member_features[:, 2].mean() - nonmember_features[:, 2].mean())

    # Balance and split
    member_idx = np.where(y_attack == 1)[0]
    nonmember_idx = np.where(y_attack == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))

    if min_size < 20:
        return {"auc": 0.5, "bal_acc": 0.5, "tpr_1": 0, "tpr_5": 0.05,
                "tpr_10": 0.10, "loss_gap": loss_gap, "n_train": len(X_train),
                "note": "too few samples"}

    m_sample = resample(member_idx, n_samples=min_size, replace=False, random_state=42)
    nm_sample = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=42)
    balanced_idx = np.concatenate([m_sample, nm_sample])

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

    return {
        "auc": auc,
        "bal_acc": bal_acc,
        "tpr_1": tpr_at[0.01],
        "tpr_5": tpr_at[0.05],
        "tpr_10": tpr_at[0.10],
        "loss_gap": loss_gap,
        "n_train": len(X_train),
    }


# ================================================================
# Main sweep
# ================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.1, 0.3, 0.5, 1.0, 10.0])
    parser.add_argument("--clients", nargs="+", type=int,
                        default=[3, 5, 10, 20])
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--local-epochs", type=int, default=5)
    args = parser.parse_args()

    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"{'=' * 70}")
    print(f"  Non-IID Federated Learning MIA Experiment")
    print(f"  Dataset: {info['name']} ({info['samples']} samples, {info['features']} features)")
    print(f"  Alphas: {args.alphas}")
    print(f"  Clients: {args.clients}")
    print(f"  Rounds: {args.rounds}, Local epochs: {args.local_epochs}")
    print(f"{'=' * 70}\n")

    # Load and split data
    X_all, y_all = get_raw_data()

    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X_all, y_all, test_size=TEST_SIZE, random_state=42, stratify=y_all
    )

    print(f"Training set: {len(X_train_full)} samples")
    print(f"Test set:     {len(X_test)} samples")

    # --- Run IID baseline first ---
    print(f"\n{'=' * 70}")
    print(f"  IID Baseline (K=3, uniform random split)")
    print(f"{'=' * 70}")

    n = len(X_train_full)
    iid_indices = np.random.RandomState(42).permutation(n)
    chunk = n // 3
    iid_client_idx = [
        iid_indices[:chunk],
        iid_indices[chunk:2*chunk],
        iid_indices[2*chunk:]
    ]

    iid_model = fedavg_simulate(
        X_train_full, y_train_full, iid_client_idx, input_dim,
        num_rounds=args.rounds, local_epochs=args.local_epochs
    )
    iid_result = evaluate_mia_on_model(
        iid_model, X_train_full, y_train_full, X_test, y_test
    )
    print(f"  IID Baseline: AUC={iid_result['auc']:.4f}  "
          f"BalAcc={iid_result['bal_acc']:.4f}  "
          f"T@10%={iid_result['tpr_10']:.4f}  "
          f"LossGap={iid_result['loss_gap']:.4f}")

    # --- Sweep ---
    all_results = []
    total_configs = len(args.alphas) * len(args.clients)
    config_num = 0

    for alpha in args.alphas:
        for K in args.clients:
            config_num += 1
            print(f"\n{'=' * 70}")
            print(f"  Config {config_num}/{total_configs}: α={alpha}, K={K}")
            print(f"{'=' * 70}")

            start_time = time.time()

            # Partition
            client_idx = dirichlet_partition(
                y_train_full, num_clients=K, alpha=alpha
            )
            stats = get_partition_stats(y_train_full, client_idx)
            print_partition_stats(stats, alpha)

            # Train FL model
            print(f"\n  Training FedAvg ({args.rounds} rounds, {args.local_epochs} local epochs)...")
            fl_model = fedavg_simulate(
                X_train_full, y_train_full, client_idx, input_dim,
                num_rounds=args.rounds, local_epochs=args.local_epochs
            )

            # Evaluate model utility
            with torch.no_grad():
                X_test_t = torch.tensor(X_test, dtype=torch.float32).to(device)
                y_test_t = torch.tensor(y_test, dtype=torch.float32).to(device)
                preds = fl_model(X_test_t).squeeze(-1)
                test_acc = ((preds > 0.5).float() == y_test_t).float().mean().item()

            # Evaluate MIA
            print(f"  Evaluating MIA...")
            mia_result = evaluate_mia_on_model(
                fl_model, X_train_full, y_train_full, X_test, y_test
            )

            elapsed = time.time() - start_time

            result = {
                "alpha": alpha,
                "K": K,
                "test_acc": test_acc,
                "pos_rate_std": stats["pos_rate_std"],
                "min_samples": stats["min_samples"],
                "max_samples": stats["max_samples"],
                **mia_result,
                "time_sec": elapsed,
            }
            all_results.append(result)

            print(f"\n  Test Accuracy: {test_acc:.4f}")
            print(f"  MIA AUC:       {mia_result['auc']:.4f}")
            print(f"  MIA BalAcc:    {mia_result['bal_acc']:.4f}")
            print(f"  TPR@10%FPR:    {mia_result['tpr_10']:.4f}")
            print(f"  Loss Gap:      {mia_result['loss_gap']:.4f}")
            print(f"  Time:          {elapsed:.1f}s")

    # --- Summary table ---
    print(f"\n\n{'=' * 90}")
    print(f"  SUMMARY: Non-IID MIA Results")
    print(f"{'=' * 90}")
    print(f"  {'α':>5} {'K':>4}  {'TestAcc':>8} {'MIA AUC':>8} {'BalAcc':>8} "
          f"{'T@1%':>7} {'T@5%':>7} {'T@10%':>7} {'LossGap':>8} {'HetStd':>8}")
    print(f"  {'-' * 82}")

    # Print IID baseline
    print(f"  {'IID':>5} {'3':>4}  {0:>8.4f} {iid_result['auc']:>8.4f} "
          f"{iid_result['bal_acc']:>8.4f} {iid_result['tpr_1']:>7.4f} "
          f"{iid_result['tpr_5']:>7.4f} {iid_result['tpr_10']:>7.4f} "
          f"{iid_result['loss_gap']:>8.4f} {'0.000':>8}")

    for r in all_results:
        print(f"  {r['alpha']:>5.1f} {r['K']:>4}  {r['test_acc']:>8.4f} {r['auc']:>8.4f} "
              f"{r['bal_acc']:>8.4f} {r['tpr_1']:>7.4f} {r['tpr_5']:>7.4f} "
              f"{r['tpr_10']:>7.4f} {r['loss_gap']:>8.4f} {r['pos_rate_std']:>8.3f}")

    print(f"  {'-' * 82}")
    print(f"  HetStd = std of positive-class rate across clients (higher = more heterogeneous)")

    # --- Save ---
    os.makedirs("experiments", exist_ok=True)
    np.save("experiments/noniid_results.npy", {
        "iid_baseline": iid_result,
        "sweep": all_results,
        "alphas": args.alphas,
        "clients": args.clients,
    }, allow_pickle=True)

    print(f"\nResults saved to experiments/noniid_results.npy")


if __name__ == "__main__":
    main()
