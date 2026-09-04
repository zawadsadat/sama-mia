import numpy as np
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    recall_score,
    precision_score,
)

# Columns written to CSV, in order. Import this so every experiment
# script emits the same schema.
UTILITY_COLUMNS = [
    "acc", "majority_acc", "acc_lift",
    "bal_acc", "recall", "precision", "f1",
    "auroc", "auprc", "auprc_baseline",
    "pos_rate", "pred_pos_rate", "degenerate", "n", "threshold",
]


def _bal_acc(y_true, pred):
    """
    Balanced accuracy, defined even when y_true or pred contains one class.
    sklearn warns and can mis-shape the confusion matrix in that case, which
    happens on tiny per-client or high-noise splits.
    """
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return float(balanced_accuracy_score(y_true, pred))


def evaluate_utility(probs, y_true, threshold=0.5):
    """
    Compute prevalence-robust utility metrics for a binary classifier.

    Args:
        probs:     array of predicted probabilities for the positive class.
                   If the model outputs logits, pass sigmoid(logits).
        y_true:    array of 0/1 ground-truth task labels.
        threshold: decision threshold for hard predictions.

    Returns dict with UTILITY_COLUMNS keys.

    Key fields:
        majority_acc   accuracy of always predicting the majority class
        acc_lift       acc - majority_acc. <= 0 means the model is no better
                       than a constant predictor on accuracy.
        bal_acc        mean of per-class recall. Exactly 0.5 for a constant
                       predictor regardless of class balance.
        auprc_baseline positive rate = AUPRC of a random classifier.
        degenerate     True if every sample gets the same hard prediction.
    """
    probs = np.asarray(probs, dtype=float).ravel()
    y_true = np.asarray(y_true).ravel().astype(int)

    if probs.shape != y_true.shape:
        raise ValueError(f"shape mismatch: probs {probs.shape} vs y {y_true.shape}")

    pred = (probs > threshold).astype(int)

    pos_rate = float(y_true.mean())
    majority_acc = float(max(pos_rate, 1.0 - pos_rate))
    acc = float((pred == y_true).mean())

    # A single-class prediction vector is the failure mode this module exists
    # to catch. sklearn will happily return 0.0 for f1/recall in that case;
    # we flag it explicitly rather than letting a 0.0 look like a bad model.
    degenerate = bool(len(np.unique(pred)) == 1)

    # Ranking metrics are undefined when y_true has one class (can happen on
    # tiny per-client splits). Fall back to the chance value.
    single_class_labels = len(np.unique(y_true)) == 1
    if single_class_labels:
        auroc = 0.5
        auprc = pos_rate
    else:
        auroc = float(roc_auc_score(y_true, probs))
        auprc = float(average_precision_score(y_true, probs))

    return {
        "acc": acc,
        "majority_acc": majority_acc,
        "acc_lift": acc - majority_acc,
        "bal_acc": _bal_acc(y_true, pred),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "auroc": auroc,
        "auprc": auprc,
        "auprc_baseline": pos_rate,
        "pos_rate": pos_rate,
        "pred_pos_rate": float(pred.mean()),
        "degenerate": degenerate,
        "n": int(len(y_true)),
        # Recorded because it is no longer always 0.5: under DP-SGD the model
        # decalibrates and the operating point is tuned on validation. A
        # balanced accuracy is not interpretable without the threshold that
        # produced it.
        "threshold": float(threshold),
    }


def is_degenerate(u, bal_acc_tol=0.01):
    """
    Broader degeneracy test than the hard-prediction check.

    Returns True if the model is a constant predictor OR its balanced
    accuracy is within `bal_acc_tol` of chance. The second condition
    catches near-degenerate models that flip on a handful of samples.

    Apply this to utility measured at a VALIDATION-TUNED threshold, not at a
    fixed 0.5. At 0.5 a decalibrated DP model can show balanced accuracy
    exactly 0.5000 while its AUROC is near 0.8 -- it ranks well and only needs
    its operating point moved, so calling it degenerate confuses a calibration
    failure with the absence of a model. Judged at 0.5, the flag answers
    "is this model calibrated?"; judged at a tuned threshold, it answers
    "does a usable classifier exist?", which is the question the FMPAS verdict
    depends on.

    This is the condition to attach to an FMPAS verdict: a model that
    trips it scores "Safe" trivially, because there is no usable model
    left for an attacker to extract membership signal from.
    """
    return bool(u["degenerate"] or u["bal_acc"] <= 0.5 + bal_acc_tol)


def utility_floor_satisfied(u_dp, u_nodp, gamma=0.90, metric="bal_acc"):
    """
    Utility-floor test for the revised FMPAS recommendation rule:

        sigma* = min{ sigma : Risk(sigma) <= rho  AND  U(sigma) >= gamma * U(no-DP) }

    Scores are measured above chance so the ratio is meaningful:
    balanced accuracy and AUROC are chance-0.5 metrics, so a model at
    exactly 0.5 scores 0 regardless of gamma. AUPRC uses the positive
    rate as its chance level.
    """
    if metric in ("bal_acc", "auroc"):
        chance = 0.5
    elif metric == "auprc":
        chance = u_nodp["auprc_baseline"]
    else:
        chance = 0.0

    lift_dp = u_dp[metric] - chance
    lift_nodp = u_nodp[metric] - chance

    if lift_nodp <= 0:
        # The no-DP model itself has no signal. Nothing to preserve, and
        # no utility claim should be made from this configuration at all.
        return False

    return bool(lift_dp >= gamma * lift_nodp)


def format_utility(u, prefix="  "):
    """Human-readable block for stdout."""
    flag = "  <<< DEGENERATE (constant predictor)" if u["degenerate"] else ""
    tau = u.get("threshold", 0.5)
    tau_s = "" if abs(tau - 0.5) < 1e-9 else f"  (threshold {tau:.4f})"
    return (
        f"{prefix}n={u['n']}  pos_rate={u['pos_rate']:.4f}\n"
        f"{prefix}acc      {u['acc']:.4f}   (majority baseline {u['majority_acc']:.4f}, "
        f"lift {u['acc_lift']:+.4f})\n"
        f"{prefix}bal_acc  {u['bal_acc']:.4f}   recall {u['recall']:.4f}   "
        f"precision {u['precision']:.4f}   f1 {u['f1']:.4f}\n"
        f"{prefix}auroc    {u['auroc']:.4f}   auprc {u['auprc']:.4f} "
        f"(baseline {u['auprc_baseline']:.4f})\n"
        f"{prefix}pred_pos_rate {u['pred_pos_rate']:.4f}{tau_s}{flag}"
    )


def utility_row(u, **extra):
    """Flat dict for pandas, with any extra identifying columns prepended."""
    row = dict(extra)
    row.update({k: u[k] for k in UTILITY_COLUMNS})
    return row
