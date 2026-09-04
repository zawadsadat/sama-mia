"""
Multi-Seed Statistical Significance Analysis.

Runs the full attack pipeline N times with different random seeds,
then computes mean ± std for all metrics.

Usage:
    python experiments/run_multiseed.py --seeds 5 --no-dp
    python experiments/run_multiseed.py --seeds 5 --dp

Output:
    experiments/multiseed_{dataset}_{condition}.csv
"""

import sys
import os
import argparse
import random

import numpy as np
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score, balanced_accuracy_score
from sklearn.utils import resample
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn as nn
import torch.optim as optim

from utils.objective import make_criterion
from utils.splits import (three_way_split, val_score, tune_threshold,
                          val_split_usable, EarlyStopper)
from utils.config import OBJECTIVE, VAL_FRACTION, DATASET, TEST_SIZE, NOISE_MULTIPLIER, MAX_GRAD_NORM, DP_DELTA, TARGET_EPOCHS, BATCH_SIZE, LR, NUM_SHADOW_MODELS
from utils.data_loader import get_raw_data, scales_per_split, fit_apply_scaler
from utils.metrics import tpr_at_fpr, ppv_at as compute_ppv
from utils.utility_metrics import evaluate_utility, format_utility, UTILITY_COLUMNS

os.makedirs("experiments", exist_ok=True)

DATASET_NAMES = {
    "breast_cancer": "Breast Cancer",
    "heart_disease": "Heart Disease",
    "cardiotocography": "Cardiotocography",
    "diabetes_hospital": "Diabetes Hospital",
}


def build_model(input_dim):
    """Build target/shadow model."""
    model = nn.Sequential(
        nn.Linear(input_dim, 256), nn.ReLU(),
        nn.Linear(256, 128), nn.ReLU(),
        nn.Linear(128, 64), nn.ReLU(),
        nn.Linear(64, 32), nn.ReLU(),
        nn.Linear(32, 1), nn.Sigmoid()
    )
    return model


def train_model(model, X_train, y_train, epochs, use_dp=False, noise_mul=1.3,
                max_norm=1.0, objective=None, X_val=None, y_val=None):
    """
    Train a model with or without DP-SGD.

    If X_val is given, the weights from the best validation epoch are kept.
    Training still runs the full epoch budget: with DP-SGD the privacy budget
    is spent per step whether or not the resulting weights are used, so
    stopping early would not refund it, and epsilon must be accounted over the
    steps actually taken. The returned stop_epoch records which epoch was
    selected, so the paper can report both it and the budget.
    """
    optimizer = optim.Adam(model.parameters(), lr=LR)
    criterion, _ = make_criterion(y_train, objective or OBJECTIVE)
    # Selecting on a validation split that is too small is worse than not
    # selecting at all: on a 19-row split the chosen epoch ranged 0-43 across
    # seeds. val_split_usable() gates both uses of the split.
    stopper = EarlyStopper() if val_split_usable(y_val) else None

    if use_dp:
        from opacus import PrivacyEngine
        from opacus.utils.batch_memory_manager import BatchMemoryManager

        privacy_engine = PrivacyEngine()
        model, optimizer, _ = privacy_engine.make_private(
            module=model,
            optimizer=optimizer,
            data_loader=torch.utils.data.DataLoader(
                torch.utils.data.TensorDataset(X_train, y_train),
                batch_size=BATCH_SIZE, shuffle=True
            ),
            noise_multiplier=noise_mul,
            max_grad_norm=max_norm,
        )

        for epoch in range(epochs):
            model.train()
            with BatchMemoryManager(
                data_loader=torch.utils.data.DataLoader(
                    torch.utils.data.TensorDataset(X_train, y_train),
                    batch_size=BATCH_SIZE, shuffle=True
                ),
                max_physical_batch_size=BATCH_SIZE,
                optimizer=optimizer
            ) as memory_safe_data_loader:
                for batch_X, batch_y in memory_safe_data_loader:
                    optimizer.zero_grad()
                    out = model(batch_X).squeeze(-1)
                    loss = criterion(out, batch_y)
                    loss.backward()
                    optimizer.step()

            if stopper is not None:
                stopper.update(epoch, val_score(model, X_val, y_val,
                                                next(model.parameters()).device),
                               model)

        epsilon = privacy_engine.get_epsilon(delta=DP_DELTA)
        stop_epoch = -1
        if stopper is not None:
            model = stopper.restore(model)
            stop_epoch = stopper.best_epoch
        return model, epsilon, stop_epoch
    else:
        dataset = torch.utils.data.TensorDataset(X_train, y_train)
        loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

        for epoch in range(epochs):
            model.train()
            for batch_X, batch_y in loader:
                optimizer.zero_grad()
                out = model(batch_X).squeeze(-1)
                loss = criterion(out, batch_y)
                loss.backward()
                optimizer.step()

            if stopper is not None:
                stopper.update(epoch, val_score(model, X_val, y_val,
                                                next(model.parameters()).device),
                               model)

        stop_epoch = -1
        if stopper is not None:
            model = stopper.restore(model)
            stop_epoch = stopper.best_epoch
        return model, float("inf"), stop_epoch


def extract_features(model, X, y_true):
    """Extract black-box attack features from a model."""
    model.eval()
    with torch.no_grad():
        X_tensor = torch.tensor(X, dtype=torch.float32) if not isinstance(X, torch.Tensor) else X
        probs = model(X_tensor).squeeze().numpy()

    p1 = probs
    p0 = 1.0 - probs
    y_np = y_true.numpy() if isinstance(y_true, torch.Tensor) else y_true
    loss = -(y_np * np.log(np.clip(p1, 1e-7, 1)) + (1 - y_np) * np.log(np.clip(p0, 1e-7, 1)))
    entropy = -(p1 * np.log(np.clip(p1, 1e-7, 1)) + p0 * np.log(np.clip(p0, 1e-7, 1)))

    features = np.column_stack([p0, p1, loss, entropy])
    return features


def seed_everything(seed):
    """
    Seed every generator the pipeline draws from.

    torch.manual_seed() alone is not enough for the DP path: Opacus draws its
    Gaussian noise from its own generator, so two invocations with identical
    arguments produced different results. On Cardiotocography at sigma=0.8 the
    selected stopping epoch moved from 48.6 +/- 0.9 to 23.4 +/- 23.7 and
    balanced accuracy from 0.7519 to 0.7238 between two runs of the same code.
    Seeding torch's global CUDA/CPU generators covers Opacus when it is
    constructed without an explicit generator, which is our case.

    This makes a run reproducible given the same library versions. It does NOT
    make DP results deterministic in principle -- if Opacus is later given its
    own generator, that must be seeded too. Verify rather than assume: run the
    same DP configuration twice and compare.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_single_seed(seed, X_all, y_all, use_dp, input_dim, num_shadows=10):
    """Run the full pipeline for a single seed."""
    seed_everything(seed)
    print(f"\n  === Seed {seed} ===")
    rng = np.random.RandomState(seed)

    # Split data
    # The realistic configuration holds out a validation split for early
    # stopping; the stress test does not, so its numbers stay comparable to
    # the original pipeline. Validation rows are EXCLUDED from the attack set
    # entirely -- they influenced the model through the stopping rule, so they
    # are neither members nor untouched non-members.
    use_val = (OBJECTIVE == "weighted")
    (X_train, y_train), (X_val, y_val), (X_test, y_test) = three_way_split(
        X_all, y_all, TEST_SIZE,
        val_frac=(VAL_FRACTION if use_val else 0.0), random_state=seed
    )
    if scales_per_split():
        X_train, X_test, sc, med = fit_apply_scaler(X_train, X_test)
        if X_val is not None:
            X_val = sc.transform(
                np.where(np.isnan(X_val), med, X_val)).astype(np.float32)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)
    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    # Train target model
    target = build_model(input_dim)
    target, epsilon, stop_epoch = train_model(target, X_train_t, y_train_t, TARGET_EPOCHS,
                                   use_dp=use_dp, noise_mul=NOISE_MULTIPLIER,
                                   max_norm=MAX_GRAD_NORM,
                                   X_val=X_val, y_val=y_val)

    # Evaluate target-task utility.
    # Reviewer comment 4: bare accuracy on imbalanced data is achievable by a
    # constant majority-class predictor, so report prevalence-robust metrics.
    target.eval()
    with torch.no_grad():
        probs = target(X_test_t).squeeze().cpu().numpy()

    # Decision threshold tuned on the validation split when one exists.
    # A fixed 0.5 measures calibration rather than whether a usable classifier
    # exists: a DP model can reach AUROC 0.8 with every probability below 0.5.
    tau, tau_val_ba, tuned = 0.5, None, False
    if X_val is not None and val_split_usable(y_val):
        tau, tau_val_ba = tune_threshold(target, X_val, y_val,
                                         next(target.parameters()).device)
        tuned = True
    elif X_val is not None:
        yv = np.asarray(y_val).ravel().astype(int)
        print(f"    (validation split too small to select on: "
              f"{int((yv==1).sum())} pos / {int((yv==0).sum())} neg; "
              f"threshold fixed at 0.5, full epoch budget used)")

    util = evaluate_utility(probs, y_test_t.cpu().numpy(), threshold=tau)
    util_at_half = evaluate_utility(probs, y_test_t.cpu().numpy(), threshold=0.5)
    test_acc = util["acc"]
    print(f"    \u03B5={epsilon:.2f}"
          + (f"  stop_epoch={stop_epoch}" if stop_epoch >= 0 else "")
          + (f"  val_bal_acc={tau_val_ba:.4f}" if tau_val_ba is not None else ""))
    print(format_utility(util, prefix="    "))
    if util["degenerate"]:
        print("    !! constant predictor at this seed")
    if X_val is not None and util_at_half["degenerate"] and not util["degenerate"]:
        print(f"    (constant at threshold 0.5; usable at tuned {tau:.4f} -- "
              f"bal_acc {util_at_half['bal_acc']:.4f} -> {util['bal_acc']:.4f}. "
              f"Decalibrated, not destroyed.)")

    # Train shadow models and collect attack features
    all_features = []
    all_labels = []

    for s in range(num_shadows):
        shadow_seed = seed * 1000 + s
        (X_s_tr, y_s_tr), (X_s_val, y_s_val), (X_s_te, y_s_te) = three_way_split(
            X_all, y_all, TEST_SIZE,
            val_frac=(VAL_FRACTION if use_val else 0.0), random_state=shadow_seed
        )
        if scales_per_split():
            X_s_tr, X_s_te, sc_s, med_s = fit_apply_scaler(X_s_tr, X_s_te)
            if X_s_val is not None:
                X_s_val = sc_s.transform(
                    np.where(np.isnan(X_s_val), med_s, X_s_val)).astype(np.float32)

        shadow = build_model(input_dim)
        shadow, _, _ = train_model(
            shadow,
            torch.tensor(X_s_tr, dtype=torch.float32),
            torch.tensor(y_s_tr, dtype=torch.float32),
            TARGET_EPOCHS, use_dp=use_dp,
            noise_mul=NOISE_MULTIPLIER, max_norm=MAX_GRAD_NORM,
            X_val=X_s_val, y_val=y_s_val
        )

        # Extract features for members
        feat_in = extract_features(shadow, X_s_tr, y_s_tr)
        all_features.append(feat_in)
        all_labels.append(np.ones(len(feat_in)))

        # Extract features for non-members
        feat_out = extract_features(shadow, X_s_te, y_s_te)
        all_features.append(feat_out)
        all_labels.append(np.zeros(len(feat_out)))

    # Build attack dataset
    X_atk = np.vstack(all_features)
    y_atk = np.concatenate(all_labels)

    # Balance
    member_idx = np.where(y_atk == 1)[0]
    nonmember_idx = np.where(y_atk == 0)[0]
    min_size = min(len(member_idx), len(nonmember_idx))
    m_s = resample(member_idx, n_samples=min_size, replace=False, random_state=seed)
    nm_s = resample(nonmember_idx, n_samples=min_size, replace=False, random_state=seed)
    bal_idx = np.concatenate([m_s, nm_s])
    perm = rng.permutation(len(bal_idx))
    bal_idx = bal_idx[perm]
    X_bal, y_bal = X_atk[bal_idx], y_atk[bal_idx]

    X_a_tr, X_a_te, y_a_tr, y_a_te = train_test_split(
        X_bal, y_bal, test_size=0.3, random_state=seed, stratify=y_bal
    )

    # Conventional black-box portfolio, matching Table I of the paper.
    #
    # Three strategies were removed. The confidence threshold scored
    # p(y=1) rather than the true-label probability; corrected to p_y it is
    # exp(-loss), a monotone transform of attack 1, so it is not distinct.
    # The globally calibrated loss is likewise monotone in the loss. The
    # two-Gaussian LiRA estimated mu/sigma from the SAME array it scored,
    # which leaks the evaluation labels into the attack, and it is quadratic
    # in the loss when its variances differ, so it could be selected as the
    # headline. The per-example difficulty-calibrated loss and offline LiRA
    # in attacks/per_example_attacks.py replace them.
    results = {}

    # 1. Loss threshold
    scores = -X_a_te[:, 2]
    fpr, tpr, _ = roc_curve(y_a_te, scores)
    results["Loss"] = {"auc": roc_auc_score(y_a_te, scores),
                       "adv": (tpr - fpr).max(), "scores": scores}

    # 2. Entropy
    scores = -X_a_te[:, 3]
    fpr, tpr, _ = roc_curve(y_a_te, scores)
    results["Entropy"] = {"auc": roc_auc_score(y_a_te, scores),
                          "adv": (tpr - fpr).max(), "scores": scores}

    # 3. RF
    rf = RandomForestClassifier(n_estimators=200, max_depth=5, random_state=seed, class_weight="balanced")
    rf.fit(X_a_tr, y_a_tr)
    rf_probs = rf.predict_proba(X_a_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_a_te, rf_probs)
    results["RF"] = {"auc": roc_auc_score(y_a_te, rf_probs),
                     "adv": (tpr - fpr).max(), "scores": rf_probs}

    # 4. NN
    nn_clf = MLPClassifier(hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=seed)
    nn_clf.fit(X_a_tr, y_a_tr)
    nn_probs = nn_clf.predict_proba(X_a_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_a_te, nn_probs)
    results["NN"] = {"auc": roc_auc_score(y_a_te, nn_probs),
                     "adv": (tpr - fpr).max(), "scores": nn_probs}

    # Loss gap
    member_loss = X_a_te[y_a_te == 1, 2].mean()
    nonmember_loss = X_a_te[y_a_te == 0, 2].mean()
    loss_gap = abs(member_loss - nonmember_loss)

    # NAIVE risk: the maximum advantage taken on the SAME split it is
    # measured on. This is the upward-biased estimator that Section V-E of
    # the paper argues against; the held-out FMPAS is produced by
    # attacks/compute_fmpas.py. Reported here only as a per-seed diagnostic
    # and named accordingly so the two are never confused.
    best_attack = max(results, key=lambda k: results[k]["adv"])
    risk_naive = results[best_attack]["adv"]

    # TPR@FPR for the attack that actually won, not a hardcoded one.
    fpr_b, tpr_b, _ = roc_curve(y_a_te, results[best_attack]["scores"])
    tpr_1 = tpr_at_fpr(fpr_b, tpr_b, 0.01)
    tpr_5 = tpr_at_fpr(fpr_b, tpr_b, 0.05)
    tpr_10 = tpr_at_fpr(fpr_b, tpr_b, 0.10)

    # PPV at 5% FPR
    # PPV is reported at the 5% FPR operating point. The operating point is
    # part of the number: PPV at the advantage-maximising threshold sits close
    # to the prior, while a low-FPR point is far above it, so the column names
    # carry the FPR they were computed at.
    ppv_1 = compute_ppv(tpr_5, 0.05, 0.01)
    ppv_5 = compute_ppv(tpr_5, 0.05, 0.05)
    ppv_10 = compute_ppv(tpr_5, 0.05, 0.10)

    row = {
        "seed": seed,
        "test_acc": test_acc,
        **{f"util_{k}": util[k] for k in UTILITY_COLUMNS},
        "epsilon": epsilon,
        "loss_gap": loss_gap,
        "stop_epoch": stop_epoch,
        "threshold": tau,
        "threshold_tuned": tuned,
        "bal_acc_at_half": util_at_half["bal_acc"],
        "degenerate_at_half": util_at_half["degenerate"],
        "fmpas_risk_naive": risk_naive,
        "selected_attack": best_attack,
        "tpr@1%": tpr_1,
        "tpr@5%": tpr_5,
        "tpr@10%": tpr_10,
        "ppv@5%FPR_prior1%": ppv_1,
        "ppv@5%FPR_prior5%": ppv_5,
        "ppv@5%FPR_prior10%": ppv_10,
    }
    for atk_name, atk_res in results.items():
        row[f"auc_{atk_name}"] = atk_res["auc"]
        row[f"adv_{atk_name}"] = atk_res["adv"]

    print(f"    Best AUC: {max(r['auc'] for r in results.values()):.4f}, "
          f"Naive risk: {risk_naive:.4f} ({best_attack}), "
          f"Loss Gap: {loss_gap:.4f}")

    return row


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds")
    parser.add_argument("--dp", action="store_true", default=False)
    parser.add_argument("--no-dp", dest="dp", action="store_false")
    parser.add_argument("--shadows", type=int, default=10, help="Shadow models per seed")
    args = parser.parse_args()

    condition = "DP" if args.dp else "No-DP"
    ds_name = DATASET_NAMES.get(DATASET, DATASET)

    print(f"{'='*70}")
    print(f"  Multi-Seed Analysis: {ds_name} | {condition}")
    print(f"  Seeds: {args.seeds}, Shadows per seed: {args.shadows}")
    print(f"  Objective: {OBJECTIVE}")
    print(f"{'='*70}")

    # get_raw_data() now returns (X, y) numpy for every dataset. The previous
    # line unpacked a 4-tuple and re-concatenated the train/test split, which
    # only worked because the diabetes route returned tensors.
    # Datasets in PER_SPLIT_SCALING are loaded raw; each target and shadow
    # model fits its own imputer and scaler on its own training half, so no
    # held-out row influences the model that scores it.
    per_split = scales_per_split()
    if per_split:
        print("  Scaling: per-split (fit inside each target/shadow split)")
    X_all, y_all = get_raw_data(scale=not per_split)
    X_all = np.asarray(X_all, dtype=np.float32)
    y_all = np.asarray(y_all, dtype=np.float32)
    input_dim = X_all.shape[1]

    all_rows = []
    for i in range(args.seeds):
        seed = 42 + i * 7  # deterministic but spread out
        row = run_single_seed(seed, X_all, y_all, args.dp, input_dim, args.shadows)
        all_rows.append(row)

    df = pd.DataFrame(all_rows)

    n_degen = int(df["util_degenerate"].sum())
    if n_degen:
        print(f"\n  {'!'*66}")
        print(f"  WARNING: {n_degen}/{len(df)} seeds produced a CONSTANT predictor.")
        print(f"  Mean target balanced accuracy: {df['util_bal_acc'].mean():.4f} "
              f"(chance = 0.5000)")
        print("  Low FMPAS risk on these seeds reflects the absence of a usable")
        print("  model, not privacy protection. Do not report these as evidence")
        print("  that DP-SGD preserves utility.")
        print(f"  {'!'*66}")

    # Save raw results
    suffix = "dp" if args.dp else "nodp"
    out_file = f"experiments/multiseed_{DATASET}_{suffix}.csv"
    df.to_csv(out_file, index=False)
    print(f"\nRaw results saved: {out_file}")

    # Compute summary statistics
    print(f"\n{'='*70}")
    print(f"  SUMMARY: {ds_name} | {condition} | {args.seeds} seeds")
    print(f"{'='*70}")

    # util_* columns are TARGET-TASK utility; the rest are attack metrics.
    # util_bal_acc == 0.5 with zero variance is the signature of a constant
    # predictor, and it is the number Table XIII should report.
    summary_cols = ["test_acc", "util_majority_acc", "util_bal_acc",
                    "bal_acc_at_half", "threshold", "stop_epoch",
                    "util_recall", "util_f1", "util_auroc", "util_auprc",
                    "epsilon", "loss_gap", "fmpas_risk_naive",
                    "tpr@1%", "tpr@5%", "tpr@10%"]
    atk_names = ["Loss", "Entropy", "RF", "NN"]

    # epsilon is inf on the no-DP path; aggregating it yields inf/nan and a
    # pandas warning, and would silently corrupt the column if DP and no-DP
    # rows were ever combined.
    if not args.dp:
        summary_cols = [c for c in summary_cols if c != "epsilon"]

    print(f"\n  {'Metric':<20} {'Mean':>10} {'Std':>10} {'Min':>10} {'Max':>10}")
    print(f"  {'-'*60}")
    for col in summary_cols:
        vals = df[col]
        print(f"  {col:<22} {vals.mean():>10.4f} {vals.std():>10.4f} "
              f"{vals.min():>10.4f} {vals.max():>10.4f}")

    picks = df["selected_attack"].value_counts().to_dict()
    print(f"\n  Selected attack (naive, per seed): {picks}")

    print(f"\n  {'Attack':<20} {'AUC Mean':>10} {'AUC Std':>10} {'Adv Mean':>10} {'Adv Std':>10}")
    print(f"  {'-'*60}")
    for atk in atk_names:
        auc_vals = df[f"auc_{atk}"]
        adv_vals = df[f"adv_{atk}"]
        print(f"  {atk:<20} {auc_vals.mean():>10.4f} {auc_vals.std():>10.4f} "
              f"{adv_vals.mean():>10.4f} {adv_vals.std():>10.4f}")

    # Save summary
    summary_file = f"experiments/multiseed_summary_{DATASET}_{suffix}.csv"
    summary_rows = []
    for col in summary_cols:
        vals = df[col]
        summary_rows.append({"metric": col, "mean": vals.mean(), "std": vals.std(),
                             "min": vals.min(), "max": vals.max()})
    for atk in atk_names:
        auc_vals = df[f"auc_{atk}"]
        adv_vals = df[f"adv_{atk}"]
        summary_rows.append({"metric": f"auc_{atk}", "mean": auc_vals.mean(), "std": auc_vals.std(),
                             "min": auc_vals.min(), "max": auc_vals.max()})
        summary_rows.append({"metric": f"adv_{atk}", "mean": adv_vals.mean(), "std": adv_vals.std(),
                             "min": adv_vals.min(), "max": adv_vals.max()})

    pd.DataFrame(summary_rows).to_csv(summary_file, index=False)
    print(f"\nSummary saved: {summary_file}")


if __name__ == "__main__":
    main()
