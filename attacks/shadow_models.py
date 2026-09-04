"""
Shadow model training for Membership Inference Attack.

Supports two modes:
  --no-dp   : Train without DP (baseline)
  --dp      : Train with DP-SGD matching target

Usage:
    python attacks/shadow_models.py --no-dp
    python attacks/shadow_models.py --dp
"""

import sys
import os
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import math
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from models.model import TargetModel
from utils.data_loader import (
    get_raw_data, get_input_dim, get_dataset_info,
    scales_per_split, fit_apply_scaler,
)
from utils.artifact_paths import save_artifact
from utils.objective import make_criterion
from utils.splits import (three_way_split, val_score, val_split_usable,
                          EarlyStopper)
from utils.config import (
    OBJECTIVE, VAL_FRACTION,
    NUM_SHADOW_MODELS, SHADOW_EPOCHS_NO_DP, SHADOW_EPOCHS_DP,
    LR, BATCH_SIZE, TEST_SIZE,
    NOISE_MULTIPLIER, MAX_GRAD_NORM,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def extract_attack_features(model, X, y):
    """Extract [prob0, prob1, loss, entropy] for each sample."""
    features = []

    with torch.no_grad():
        preds = model(X).squeeze()

        for i, p in enumerate(preds):
            confidence = p.item()
            prob0 = 1 - confidence
            prob1 = confidence

            sample_loss = F.binary_cross_entropy(
                p, y[i], reduction='none'
            ).item()

            entropy = -(
                confidence * math.log(confidence + 1e-10) +
                (1 - confidence) * math.log(1 - confidence + 1e-10)
            )

            features.append([prob0, prob1, sample_loss, entropy])

    return features


def main():
    parser = argparse.ArgumentParser()
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--dp", action="store_true", help="Train with DP-SGD")
    group.add_argument("--no-dp", action="store_true", help="Train without DP")
    args = parser.parse_args()

    use_dp = args.dp
    epochs = SHADOW_EPOCHS_DP if use_dp else SHADOW_EPOCHS_NO_DP
    mode_str = "WITH DP-SGD" if use_dp else "WITHOUT DP (baseline)"
    suffix = "_dp" if use_dp else "_nodp"

    # Dataset info
    info = get_dataset_info()
    input_dim = get_input_dim()

    print(f"=== Shadow Model Training {mode_str} ===")
    print(f"Dataset: {info['name']} ({info['samples']} samples, {info['features']} features)")
    print(f"Input dim: {input_dim}\n")

    # Datasets in PER_SPLIT_SCALING are loaded raw; each shadow model fits
    # its own imputer and scaler on its own training half, so no held-out
    # row influences the model that scores it.
    per_split = scales_per_split()
    if per_split:
        print("Scaling: per-split (fit on each shadow's training half)\n")

    # The realistic configuration holds out a validation split per shadow for
    # early stopping, exactly as the target does. It MUST match the target's
    # training procedure: the offline-LiRA out-statistics compare the target's
    # loss on a record against the losses shadows assign it, and shadows
    # trained to a different point on the optimisation path are not a valid
    # reference. Validation rows are excluded from the attack set, so the
    # member and non-member pools stay clean.
    use_val = (OBJECTIVE == "weighted")
    if use_val:
        print(f"Three-way split per shadow (val_frac={VAL_FRACTION}); "
              f"validation rows excluded from the attack set\n")
    X_all, y_all = get_raw_data(scale=not per_split)

    attack_X = []
    attack_y = []

    for model_id in range(NUM_SHADOW_MODELS):

        print(f"Training Shadow Model {model_id + 1}/{NUM_SHADOW_MODELS}")

        (X_train_np, y_train_np), (X_val_np, y_val_np), (X_test_np, y_test_np) = \
            three_way_split(X_all, y_all, TEST_SIZE,
                            val_frac=(VAL_FRACTION if use_val else 0.0),
                            random_state=model_id)

        if per_split:
            X_train_np, X_test_np, sc, med = fit_apply_scaler(X_train_np, X_test_np)
            if X_val_np is not None:
                X_val_np = sc.transform(
                    np.where(np.isnan(X_val_np), med, X_val_np)).astype(np.float32)

        X_train = torch.tensor(X_train_np, dtype=torch.float32).to(device)
        y_train = torch.tensor(y_train_np, dtype=torch.float32).to(device)
        X_test = torch.tensor(X_test_np, dtype=torch.float32).to(device)
        y_test = torch.tensor(y_test_np, dtype=torch.float32).to(device)

        dataset = TensorDataset(X_train, y_train)
        loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        model = TargetModel(input_dim=input_dim).to(device)
        # Same objective as the target: the offline-LiRA out-statistics are
        # meaningless if the shadows optimize a different loss.
        criterion, _ = make_criterion(y_train_np, OBJECTIVE)
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

        stopper = (EarlyStopper()
                   if (X_val_np is not None and val_split_usable(y_val_np))
                   else None)

        for epoch in range(epochs):
            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                outputs = model(X_batch).squeeze(-1)
                loss = criterion(outputs, y_batch)
                loss.backward()
                optimizer.step()

            if stopper is not None:
                stopper.update(epoch, val_score(model, X_val_np, y_val_np, device),
                               model)

        if stopper is not None:
            model = stopper.restore(model)

        # Extract features
        member_features = extract_attack_features(model, X_train, y_train)
        for f in member_features:
            attack_X.append(f)
            attack_y.append(1)

        nonmember_features = extract_attack_features(model, X_test, y_test)
        for f in nonmember_features:
            attack_X.append(f)
            attack_y.append(0)

        # Stats
        with torch.no_grad():
            train_acc = ((model(X_train).squeeze() > 0.5).float() == y_train).float().mean()
            test_acc = ((model(X_test).squeeze() > 0.5).float() == y_test).float().mean()
            member_losses = [f[2] for f in member_features]
            nonmember_losses = [f[2] for f in nonmember_features]

        eps_str = ""
        if use_dp:
            eps_str = f", ε: {privacy_engine.get_epsilon(delta=1e-5):.2f}"

        print(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}{eps_str}")
        print(f"  Avg loss — members: {np.mean(member_losses):.4f}, "
              f"non-members: {np.mean(nonmember_losses):.4f}")

    attack_X = np.array(attack_X)
    attack_y = np.array(attack_y)

    print(f"\nAttack dataset: {len(attack_X)} samples "
          f"({attack_y.sum():.0f} members, {len(attack_y) - attack_y.sum():.0f} non-members)")

    member_mask = attack_y == 1
    print(f"\nFeature means — Members:     {attack_X[member_mask].mean(axis=0).round(4)}")
    print(f"Feature means — Non-members: {attack_X[~member_mask].mean(axis=0).round(4)}")
    print(f"Feature stdev — Members:     {attack_X[member_mask].std(axis=0).round(4)}")
    print(f"Feature stdev — Non-members: {attack_X[~member_mask].std(axis=0).round(4)}")
    print(f"  (columns: prob0, prob1, loss, entropy)")

    # Artifacts are tagged with the dataset name and written with a .meta.json
    # sidecar. The previous condition-only names (attack_features_dp.npy) were
    # overwritten whenever DATASET changed in utils/config.py, with no way to
    # tell afterwards which dataset a file held -- which is how Cardiotocography
    # white-box features ended up inside a Diabetes audit.
    os.makedirs("experiments", exist_ok=True)
    meta = {
        "n_shadow_models": NUM_SHADOW_MODELS,
        "n_rows": int(len(attack_y)),
        "n_members": int((attack_y == 1).sum()),
        "n_nonmembers": int((attack_y == 0).sum()),
        "epochs": epochs,
        "feature_columns": ["prob0", "prob1", "loss", "entropy"],
        "test_size": TEST_SIZE,
    }
    fp = save_artifact("attack_features", info["name"], use_dp, attack_X, meta)
    save_artifact("attack_labels", info["name"], use_dp, attack_y, meta)

    print(f"\nAttack dataset saved to {fp}")
    print(f"  dataset={info['name']}  condition={'dp' if use_dp else 'nodp'}  "
          f"shadows={NUM_SHADOW_MODELS}  rows={len(attack_y)}  "
          f"members={int((attack_y == 1).sum())}")


if __name__ == "__main__":
    main()
