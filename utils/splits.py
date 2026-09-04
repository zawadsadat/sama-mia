"""
Train / validation / test partitioning, and early stopping on the validation
split.

Two configurations:

  "unweighted" stress test  -- two-way split, fixed epoch count, no validation
                               set. Unchanged from the original pipeline so the
                               stress-test numbers stay comparable.
  "weighted" realistic      -- three-way split with early stopping.
"""


import contextlib
import warnings

import numpy as np
import torch
from sklearn.metrics import roc_auc_score, balanced_accuracy_score
from sklearn.model_selection import train_test_split


def three_way_split(X, y, test_size, val_frac=0.2, random_state=42):
    """
    Split into train / val / test.

    test_size is the held-out fraction, matching the two-way pipeline, so the
    test split is the same size as before. val_frac is then taken OUT OF THE
    TRAINING PORTION, so the training set shrinks rather than the non-member
    pool: members are the scarce class in every one of our datasets.

    Returns (X_tr, y_tr), (X_val, y_val), (X_te, y_te).
    """
    X_fit, X_te, y_fit, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    if val_frac <= 0:
        return (X_fit, y_fit), (None, None), (X_te, y_te)

    # A stratified validation split needs at least one of each class on both
    # sides; on the smallest training sets (Breast: 28 rows) that can fail.
    n_val = int(round(len(y_fit) * val_frac))
    if n_val < 2 or (len(y_fit) - n_val) < 2 or len(np.unique(y_fit)) < 2:
        return (X_fit, y_fit), (None, None), (X_te, y_te)

    try:
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_fit, y_fit, test_size=val_frac,
            random_state=random_state, stratify=y_fit
        )
    except ValueError:
        # Stratification can still fail on a small training portion for a
        # particular seed (too few of the minority class to place on both
        # sides). Fall back to no validation split for that model rather than
        # crashing a campaign run part-way through; the caller trains for the
        # full epoch budget instead, and the absence is visible because
        # X_val is None.
        return (X_fit, y_fit), (None, None), (X_te, y_te)
    return (X_tr, y_tr), (X_val, y_val), (X_te, y_te)


@contextlib.contextmanager
def _hooks_disabled(model):
    """
    Suspend Opacus per-sample-gradient hooks around an evaluation forward pass.

    opacus.GradSampleModule registers forward hooks that PUSH activations onto
    a per-module stack, which the backward hook then POPS. A forward pass made
    outside a training step -- an early-stopping check or a threshold search --
    pushes activations that are never popped, so the next loss.backward()
    raises "IndexError: pop from empty list" once the stack desynchronises.

    disable_hooks()/enable_hooks() is Opacus's own remedy. The getattr guard
    keeps this a no-op for plain nn.Modules, so the same code path serves the
    private and non-private branches.
    """
    disable = getattr(model, "disable_hooks", None)
    enable = getattr(model, "enable_hooks", None)
    if disable is None or enable is None:
        yield
        return
    disable()
    try:
        yield
    finally:
        enable()


def val_score(model, X_val, y_val, device):
    """
    Validation AUROC. Higher is better; returns 0.5 when the split is
    single-class.

    NOT validation loss. Under DP-SGD the model decalibrates as it trains:
    predicted probabilities shrink toward zero while the RANKING keeps
    improving. On Cardiotocography at sigma=0.8 the weighted validation loss
    bottomed out at epoch 2, where test AUROC was 0.4975, and rose steadily
    thereafter even as test AUROC climbed to 0.7933 by epoch 49 -- so loss-based
    stopping selected the single worst epoch of the run. Weighted BCE penalises
    low probabilities on positives by pos_weight, so it tracks calibration, not
    discrimination. AUROC is threshold-free and measures what actually improves.
    """
    was_training = model.training
    model.eval()
    with _hooks_disabled(model), torch.no_grad():
        Xv = torch.as_tensor(X_val, dtype=torch.float32).to(device)
        p = model(Xv).squeeze(-1).detach().cpu().numpy().ravel()
    if was_training:
        model.train()
    yv = np.asarray(y_val).ravel()
    if len(np.unique(yv)) < 2:
        return 0.5
    return float(roc_auc_score(yv, p))


MIN_VAL_PER_CLASS = 10   # below this, tuning fits noise -- see tune_threshold


def tune_threshold(model, X_val, y_val, device, n_grid=199, min_per_class=None):
    """
    Choose the decision threshold maximising balanced accuracy on the
    validation split, and return (threshold, validation balanced accuracy).

    A fixed 0.5 threshold is wrong for a decalibrated model. Under DP-SGD the
    maximum predicted probability can sit below 0.5 while AUROC is near 0.8:
    at 0.5 the model looks like a constant predictor and trips the degeneracy
    flag, when in fact it ranks well and only needs its operating point moved.
    Reporting utility at 0.5 in that regime measures calibration, not whether
    a usable classifier exists.

    The threshold is chosen on validation and applied to test, so it uses no
    test information. Validation rows are already excluded from the attack set
    (see three_way_split), so this adds no membership leakage.

    Tuning is SKIPPED, and 0.5 returned, when either class has fewer than
    min_per_class validation rows. On Heart Disease the validation split holds
    19 rows, and tuning on it selected thresholds spanning 0.04 to 0.74 across
    five seeds and LOWERED test balanced accuracy from 0.7721 to 0.7438 -- the
    tuned threshold was fitting validation noise. Cardiotocography, with 64
    validation rows, gave 0.065 +/- 0.034 and raised balanced accuracy from
    0.5000 to 0.7519. A threshold estimated from a handful of rows is worse
    than the default it replaces, so the guard prefers 0.5 and says so.
    """
    if min_per_class is None:
        min_per_class = MIN_VAL_PER_CLASS
    was_training = model.training
    model.eval()
    with _hooks_disabled(model), torch.no_grad():
        Xv = torch.as_tensor(X_val, dtype=torch.float32).to(device)
        p = model(Xv).squeeze(-1).detach().cpu().numpy().ravel()
    if was_training:
        model.train()
    yv = np.asarray(y_val).ravel().astype(int)
    if len(np.unique(yv)) < 2 or len(np.unique(p)) < 2:
        return 0.5, 0.5

    n_pos, n_neg = int((yv == 1).sum()), int((yv == 0).sum())
    if min(n_pos, n_neg) < min_per_class:
        return 0.5, float("nan")

    # Candidates spread over the observed score range rather than [0,1]: when
    # every probability sits below 0.4, a uniform [0,1] grid wastes most of
    # its points above the data.
    cand = np.unique(np.quantile(p, np.linspace(0.005, 0.995, n_grid)))
    best_tau, best_ba = 0.5, -1.0
    for tau in cand:
        pred = (p > tau).astype(int)
        if len(np.unique(pred)) < 2:
            continue
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            ba = balanced_accuracy_score(yv, pred)
        if ba > best_ba:
            best_tau, best_ba = float(tau), float(ba)
    return best_tau, best_ba


def val_split_usable(y_val, min_per_class=None):
    """
    True if the validation split is large enough to select on.

    Applies to BOTH uses of the split -- the stopping epoch and the decision
    threshold. On Heart Disease (19 validation rows) the selected epoch was
    10.6 +/- 18.4 across five seeds, ranging from 0 to 43: the criterion was
    not measuring anything. On Cardiotocography (64 rows) it was 48.6 +/- 0.9.
    """
    if min_per_class is None:
        min_per_class = MIN_VAL_PER_CLASS
    if y_val is None:
        return False
    yv = np.asarray(y_val).ravel().astype(int)
    return bool(min(int((yv == 1).sum()), int((yv == 0).sum())) >= min_per_class)


class EarlyStopper:
    """
    Track the best validation SCORE (higher is better) and its epoch.

    Used to select a stopping epoch rather than to halt training mid-run: with
    DP-SGD the privacy budget is spent per step regardless of whether the
    resulting weights are kept, so stopping early does not refund it. We
    therefore train the full epoch budget, keep the weights from the best
    validation epoch, and report both the chosen epoch and the full budget the
    epsilon was accounted over.
    """

    def __init__(self, patience=None):
        self.best = -float("inf")
        self.best_epoch = -1
        self.best_state = None
        self.patience = patience

    def update(self, epoch, score, model):
        if score > self.best:
            self.best = score
            self.best_epoch = epoch
            self.best_state = {k: v.detach().clone()
                               for k, v in model.state_dict().items()}
        return self

    def exhausted(self, epoch):
        return (self.patience is not None
                and self.best_epoch >= 0
                and epoch - self.best_epoch >= self.patience)

    def restore(self, model):
        if self.best_state is not None:
            model.load_state_dict(self.best_state)
        return model
