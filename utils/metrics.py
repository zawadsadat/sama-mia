"""
Shared ROC-derived metrics.

Single definition of TPR-at-FPR for the whole repo. Three implementations
previously coexisted and disagreed; see tpr_at_fpr() for what they did wrong.
"""

import numpy as np


def tpr_at_fpr(fpr_arr, tpr_arr, target_fpr):
    """
    TPR at a target FPR, linearly interpolated between ROC vertices.

    The empirical ROC is a step function, so a target FPR usually falls
    between two vertices. The interpolated value is the TPR achievable by
    randomising between the two adjacent thresholds, which is the convention
    used in the membership-inference literature.

    Replaces three earlier implementations that disagreed:

      - np.searchsorted(fpr_arr, target) then tpr_arr[idx] returned the TPR
        at the first vertex with FPR >= target, which overshoots the false
        positive budget. Worse, it returned the SAME value for every target
        whenever a single ROC step spanned them all, which happens when the
        attack's score distribution has a large mass point. On Diabetes this
        made TPR@1%, TPR@5% and TPR@10% identical to four decimal places.

      - np.argmin(|fpr_arr - target|) took the nearest vertex in either
        direction, which can also overshoot the budget.
    """
    fpr_arr = np.asarray(fpr_arr, dtype=float)
    tpr_arr = np.asarray(tpr_arr, dtype=float)
    return float(np.interp(target_fpr, fpr_arr, tpr_arr))


def ppv_at(tpr, fpr, prior):
    """
    Positive predictive value at an operating point, under a membership prior.

        PPV = TPR * prior / (TPR * prior + FPR * (1 - prior))

    The operating point matters: PPV at the advantage-maximising threshold
    (mid-ROC, abundant false positives) is close to the prior, while PPV at a
    low-FPR point is far above it. Always record which point a reported PPV
    came from.
    """
    num = tpr * prior
    den = tpr * prior + fpr * (1.0 - prior)
    return float(num / den) if den > 0 else 0.0
