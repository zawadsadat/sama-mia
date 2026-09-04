import torch
import torch.nn as nn


def pos_weight_from(y_train):
    """
    n_neg / n_pos on the training split. Returns 1.0 when the split is
    single-class or balanced, so the weighted path degrades gracefully.
    """
    y = torch.as_tensor(y_train).reshape(-1).float()
    n_pos = float((y == 1).sum())
    n_neg = float((y == 0).sum())
    if n_pos == 0 or n_neg == 0:
        return 1.0
    return n_neg / n_pos


class WeightedBCE:
    """
    BCELoss with a per-example weight of pos_weight on positives and 1.0 on
    negatives. Call it exactly like nn.BCELoss(): criterion(outputs, targets).

    pos_weight=1.0 reproduces plain nn.BCELoss() bit for bit, so a single code
    path serves both configurations.
    """

    def __init__(self, pos_weight=1.0):
        self.pos_weight = float(pos_weight)

    def __call__(self, outputs, targets):
        if self.pos_weight == 1.0:
            return nn.functional.binary_cross_entropy(outputs, targets)
        w = torch.ones_like(targets) + (self.pos_weight - 1.0) * targets
        return nn.functional.binary_cross_entropy(outputs, targets, weight=w)


def make_criterion(y_train, objective):
    """
    Build the criterion for a training split.

    objective: "unweighted" or "weighted". Anything else raises, so a typo in
    a config or CLI flag fails loudly instead of silently training the wrong
    configuration -- which would be invisible in the results.
    """
    if objective == "unweighted":
        return WeightedBCE(1.0), 1.0
    if objective == "weighted":
        pw = pos_weight_from(y_train)
        return WeightedBCE(pw), pw
    raise ValueError(
        f"Unknown objective {objective!r}; expected 'unweighted' or 'weighted'."
    )
