"""
Federated shadow-model pipeline.

USAGE
-----
    python attacks/fl_shadow_models.py                    # K=3, 10 rounds, E=5
    python attacks/fl_shadow_models.py --clients 5
    python attacks/fl_shadow_models.py --num-shadow 15 --rounds 10

Then, exactly as for the centralized pipeline:
    python attacks/per_example_attacks.py --no-dp --fl
    python attacks/compute_fmpas.py --no-dp --fl
"""

import sys
import os
import time
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from models.model import TargetModel
from utils.data_loader import (
    get_raw_data, get_input_dim, get_dataset_info,
    scales_per_split, fit_apply_scaler,
)
from utils.objective import make_criterion
from utils.splits import three_way_split, val_score, val_split_usable, EarlyStopper
from utils.config import OBJECTIVE, VAL_FRACTION, NUM_SHADOW_MODELS, LR, BATCH_SIZE, TEST_SIZE
from utils.artifact_paths import save_artifact
from utils.utility_metrics import evaluate_utility, format_utility

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def iid_partition(n, num_clients, seed):
    """Disjoint IID partition of the training indices."""
    idx = np.arange(n)
    np.random.RandomState(seed).shuffle(idx)
    return np.array_split(idx, num_clients)


def fedavg_simulate(X_train, y_train, client_indices, input_dim,
                    num_rounds=10, local_epochs=5, lr=LR,
                    X_val=None, y_val=None):
    """
    In-process FedAvg. Mirrors experiments/run_noniid_sweep.py so that the
    federated models here and in the non-IID sweep are trained identically.

    If a validation split is supplied, the global model is scored after each
    aggregation round and the best-scoring round's weights are kept. Stopping
    is evaluated on the GLOBAL model between rounds, not inside a client's
    local epochs: a client cannot see the coordinator's validation split, and
    halting locally would change what FedAvg averages rather than when training
    stops. This mirrors the centralized pipeline at the level at which the
    released model actually exists.
    """
    global_model = TargetModel(input_dim=input_dim).to(device)
    stopper = (EarlyStopper()
               if (X_val is not None and val_split_usable(y_val)) else None)

    for round_idx in range(num_rounds):
        client_state_dicts, client_sizes = [], []

        for idx in client_indices:
            if len(idx) == 0:
                continue

            local_model = TargetModel(input_dim=input_dim).to(device)
            local_model.load_state_dict(global_model.state_dict())

            X_k = torch.tensor(X_train[idx], dtype=torch.float32).to(device)
            y_k = torch.tensor(y_train[idx], dtype=torch.float32).to(device)

            optimizer = optim.Adam(local_model.parameters(), lr=lr)
            # pos_weight from this client's own partition: a real client can
            # only see its own label distribution, and under a Dirichlet skew
            # that differs from the global rate.
            criterion, _ = make_criterion(y_k, OBJECTIVE)
            loader = DataLoader(TensorDataset(X_k, y_k),
                                batch_size=min(BATCH_SIZE, len(idx)), shuffle=True)

            for _ in range(local_epochs):
                for xb, yb in loader:
                    optimizer.zero_grad()
                    loss = criterion(local_model(xb).squeeze(-1), yb)
                    loss.backward()
                    optimizer.step()

            client_state_dicts.append(local_model.state_dict())
            client_sizes.append(len(idx))

        if not client_state_dicts:
            continue

        total = sum(client_sizes)
        global_state = {}
        for key in global_model.state_dict().keys():
            acc = torch.zeros_like(global_model.state_dict()[key], dtype=torch.float32)
            for sd, size in zip(client_state_dicts, client_sizes):
                acc += sd[key].float() * (size / total)
            global_state[key] = acc
        global_model.load_state_dict(global_state)

        if stopper is not None:
            stopper.update(round_idx,
                           val_score(global_model, X_val, y_val, device),
                           global_model)

    if stopper is not None:
        global_model = stopper.restore(global_model)

    return global_model


def extract_attack_features(model, X, y):
    """
    Black-box features: [prob0, prob1, loss, entropy].

    Column order matches attacks/shadow_models.py exactly. per_example_attacks.py
    reads the loss from index 2 and will silently produce nonsense if this
    changes.
    """
    model.eval()
    with torch.no_grad():
        Xt = torch.tensor(X, dtype=torch.float32).to(device)
        yt = torch.tensor(y, dtype=torch.float32).to(device)
        p1 = model(Xt).squeeze(-1).clamp(1e-7, 1 - 1e-7)
        p0 = 1.0 - p1
        loss = nn.functional.binary_cross_entropy(p1, yt, reduction="none")
        entropy = -(p1 * torch.log(p1) + p0 * torch.log(p0))
        feats = torch.stack([p0, p1, loss, entropy], dim=1)
    return feats.cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--num-shadow", type=int, default=NUM_SHADOW_MODELS)
    p.add_argument("--clients", type=int, default=3)
    p.add_argument("--rounds", type=int, default=10)
    p.add_argument("--local-epochs", type=int, default=5)
    args = p.parse_args()

    info = get_dataset_info()
    input_dim = get_input_dim()
    per_split = scales_per_split()

    # The realistic configuration holds out a validation split per shadow, as
    # the centralized pipeline does. Table VI compares federated against
    # centralized under an identical estimator, which requires an identical
    # training procedure -- otherwise the comparison confounds federation with
    # the stopping rule. Validation rows are excluded from the attack set.
    #
    # Early stopping is applied to the GLOBAL model between rounds, not inside
    # a client's local epochs: a client cannot see the coordinator's validation
    # split, and stopping locally would change what FedAvg averages rather than
    # when it halts.
    use_val = (OBJECTIVE == "weighted")
    if use_val:
        print(f"  Three-way split per shadow (val_frac={VAL_FRACTION}); "
              f"validation rows excluded from the attack set")
    X_all, y_all = get_raw_data(scale=not per_split)
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)

    print(f"=== Federated shadow models: {info['name']} ===")
    print(f"  K={args.clients} clients, {args.rounds} rounds, "
          f"E={args.local_epochs} local epochs")
    print(f"  {args.num_shadow} shadow models, TEST_SIZE={TEST_SIZE}")
    print(f"  No DP: these models carry no formal privacy guarantee.\n")

    attack_X, attack_y = [], []
    utilities = []
    t0 = time.time()

    for model_id in range(args.num_shadow):
        # Identical split procedure to attacks/shadow_models.py. This is what
        # lets per_example_attacks.py reconstruct which sample each row came
        # from without retraining.
        (X_tr, y_tr), (X_val, y_val), (X_te, y_te) = three_way_split(
            X_all, y_all, TEST_SIZE,
            val_frac=(VAL_FRACTION if use_val else 0.0), random_state=model_id
        )
        if per_split:
            X_tr, X_te, sc, med = fit_apply_scaler(X_tr, X_te)
            if X_val is not None:
                X_val = sc.transform(
                    np.where(np.isnan(X_val), med, X_val)).astype(np.float32)

        parts = iid_partition(len(X_tr), args.clients, seed=model_id)
        model = fedavg_simulate(
            X_tr, y_tr, parts, input_dim,
            num_rounds=args.rounds, local_epochs=args.local_epochs,
            X_val=X_val, y_val=y_val
        )

        # Members first, then non-members -- same row order as the centralized
        # pipeline.
        attack_X.append(extract_attack_features(model, X_tr, y_tr))
        attack_y.append(np.ones(len(y_tr)))
        attack_X.append(extract_attack_features(model, X_te, y_te))
        attack_y.append(np.zeros(len(y_te)))

        with torch.no_grad():
            probs = model(torch.tensor(X_te, dtype=torch.float32).to(device)
                          ).squeeze(-1).cpu().numpy()
        u = evaluate_utility(probs, y_te)
        utilities.append(u)

        print(f"  FL shadow {model_id + 1}/{args.num_shadow} done "
              f"(bal_acc {u['bal_acc']:.4f}"
              f"{', DEGENERATE' if u['degenerate'] else ''})")

    attack_X = np.concatenate(attack_X).astype(np.float32)
    attack_y = np.concatenate(attack_y).astype(np.float32)

    mean_bal = float(np.mean([u["bal_acc"] for u in utilities]))
    n_degen = sum(u["degenerate"] for u in utilities)

    print(f"\n  Mean FL target-task balanced accuracy: {mean_bal:.4f} "
          f"(majority baseline {utilities[0]['majority_acc']:.4f})")
    if n_degen:
        print(f"  WARNING: {n_degen}/{args.num_shadow} FL models are constant "
              f"predictors; leakage measured on them reflects the absence of a "
              f"model.")

    meta = {
        "training": "federated",
        "n_shadow_models": args.num_shadow,
        "n_clients": args.clients,
        "rounds": args.rounds,
        "local_epochs": args.local_epochs,
        "n_rows": int(len(attack_y)),
        "n_members": int((attack_y == 1).sum()),
        "n_nonmembers": int((attack_y == 0).sum()),
        "test_size": TEST_SIZE,
        "dp": False,
        "mean_target_bal_acc": mean_bal,
        "n_degenerate": int(n_degen),
        "feature_columns": ["prob0", "prob1", "loss", "entropy"],
    }
    fp = save_artifact("fl_attack_features", info["name"], False, attack_X, meta)
    save_artifact("fl_attack_labels", info["name"], False, attack_y, meta)

    print(f"\nFederated attack set saved to {fp}")
    print(f"  rows={len(attack_y)}  members={int((attack_y == 1).sum())}  "
          f"elapsed={time.time() - t0:.0f}s")
    print("\nNext:")
    print("  python attacks/per_example_attacks.py --no-dp --fl")
    print("  python attacks/compute_fmpas.py --no-dp --fl")


if __name__ == "__main__":
    main()
