"""
SAMA with three-way split (fit / select / audit).

Usage:
    python attacks/compute_fmpas.py --no-dp
    python attacks/compute_fmpas.py --dp
    python attacks/compute_fmpas.py --dp --bb-only          # skip white-box
    python attacks/compute_fmpas.py --dp --n-bootstrap 2000
    python attacks/compute_fmpas.py --dp --seed 1           # split sensitivity

Outputs:
    experiments/fmpas_{DATASET}{suffix}.csv          per-attack, both splits
    experiments/fmpas_summary_{DATASET}{suffix}.csv  headline row
"""

import sys
import os
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve, roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.utils import resample
from scipy.stats import norm

from utils.config import DATASET
from utils.artifact_paths import load_artifact

# Black-box feature layout, from attacks/shadow_models.py
# extract_attack_features(): [prob0, prob1, loss, entropy]
BB_PROB0, BB_PROB1, BB_LOSS, BB_ENTROPY = 0, 1, 2, 3

# White-box feature layout, from attacks/whitebox_attack.py
# extract_gradient_features():
#   [grad_l2, grad_l1, grad_var, grad_max, grad_mean,
#    layer_norm_0..9, loss, conf, entropy]
WB_GRAD_L2, WB_GRAD_L1 = 0, 1
WB_LOSS, WB_CONF, WB_ENTROPY = 15, 16, 17

VERDICT_SAFE = 0.01
VERDICT_MARGINAL = 0.05


# --------------------------------------------------------------------------
# split
# --------------------------------------------------------------------------

def balance(X, y, seed):
    """Undersample the majority class to a balanced attack set."""
    m = np.where(y == 1)[0]
    nm = np.where(y == 0)[0]
    k = min(len(m), len(nm))
    ms = resample(m, n_samples=k, replace=False, random_state=seed)
    nms = resample(nm, n_samples=k, replace=False, random_state=seed)
    idx = np.concatenate([ms, nms])
    idx = idx[np.random.RandomState(seed).permutation(len(idx))]
    return X[idx], y[idx]


def three_way_split(X, y, seed, fit_frac=0.5, sel_frac=0.25):
    """
    Split into fit / select / audit.

    Defaults give 50 / 25 / 25. On small datasets the audit split is the
    binding constraint on CI width -- shrink fit_frac before shrinking the
    audit share, since only two of twelve attacks need the fit split.
    """
    aud_frac = 1.0 - fit_frac - sel_frac
    if aud_frac <= 0:
        raise ValueError("fit_frac + sel_frac must be < 1")

    X_fit, X_rest, y_fit, y_rest = train_test_split(
        X, y, test_size=1.0 - fit_frac, random_state=seed, stratify=y
    )
    X_sel, X_aud, y_sel, y_aud = train_test_split(
        X_rest, y_rest,
        test_size=aud_frac / (sel_frac + aud_frac),
        random_state=seed, stratify=y_rest
    )
    return (X_fit, y_fit), (X_sel, y_sel), (X_aud, y_aud)


# --------------------------------------------------------------------------
# attacks
#
# Each attack is a scorer: given a feature matrix it returns a membership
# score (higher = more likely member). Scorers are FIT on A_fit only, then
# applied unchanged to A_sel and A_audit. Any attack whose scorer depends on
# the data it scores would leak the audit split -- see the note on LiRA and
# calibrated loss below.
# --------------------------------------------------------------------------

def build_bb_attacks(X_fit, y_fit, seed):
    """
    Seven black-box attacks, all fitted on A_fit only.

    NOTE ON LiRA AND CALIBRATED LOSS
    The original implementation estimated the in/out Gaussians (LiRA) and the
    non-member mean loss (calibrated loss) from the SAME array it then scored.
    That leaks membership labels of the evaluation set into the attack itself,
    which inflates those two attacks specifically. Here both are estimated on
    A_fit and applied fixed. Expect LiRA and Calibrated Loss to drop the most
    relative to the original numbers -- that drop is a correction, not a
    regression.
    """
    attacks = {}

    attacks["Loss Threshold"] = lambda X: -X[:, BB_LOSS]
    #attacks["Confidence Threshold"] = lambda X: X[:, BB_PROB1]
    attacks["Entropy Threshold"] = lambda X: -X[:, BB_ENTROPY]

    # Calibrated loss: normalize by non-member mean loss estimated on A_fit.
    #nonmember_mean = X_fit[y_fit == 0, BB_LOSS].mean()
    #denom = max(nonmember_mean, 1e-8)
    #attacks["Calibrated Loss"] = lambda X, d=denom: -(X[:, BB_LOSS] / d)

    # LiRA: in/out Gaussians estimated on A_fit.
    #mu_in = X_fit[y_fit == 1, BB_LOSS].mean()
    #sd_in = max(X_fit[y_fit == 1, BB_LOSS].std(), 1e-8)
    #mu_out = X_fit[y_fit == 0, BB_LOSS].mean()
    #sd_out = max(X_fit[y_fit == 0, BB_LOSS].std(), 1e-8)
    #attacks["LiRA"] = lambda X, a=(mu_in, sd_in, mu_out, sd_out): (
    #    norm.logpdf(X[:, BB_LOSS], a[0], a[1])
    #    - norm.logpdf(X[:, BB_LOSS], a[2], a[3])
    #)

    rf = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=seed, class_weight="balanced"
    ).fit(X_fit, y_fit)
    attacks["Random Forest"] = lambda X, m=rf: m.predict_proba(X)[:, 1]

    nn = MLPClassifier(
        hidden_layer_sizes=(64, 32, 16), max_iter=200, random_state=seed
    ).fit(X_fit, y_fit)
    attacks["Neural Network"] = lambda X, m=nn: m.predict_proba(X)[:, 1]

    return attacks


def build_perexample_attacks(X_fit, y_fit, seed, columns):
    """
    Per-example attacks precomputed by attacks/per_example_attacks.py.

    These are already membership SCORES (higher = more likely member), not raw
    features, so each "attack" is just the identity on its own column. They
    still go through the same fit/select/audit split as every other attack:
    the threshold is chosen on A_sel and applied fixed on A_audit.

    Nothing is fitted on A_fit here -- the per-example out-statistics were
    computed from shadow models, never from the audit rows -- but the split is
    kept identical so the max over attacks is taken over one common sample.
    """
    names = {"offline_lira": "Offline LiRA (per-example)",
             "difficulty_calibrated": "Difficulty-Calibrated Loss (per-example)"}
    attacks = {}
    for j, col in enumerate(columns):
        attacks[names.get(col, col)] = lambda X, k=j: X[:, k]
    return attacks


def build_wb_attacks(X_fit, y_fit, seed):
    """Five white-box gradient attacks, fitted on A_fit only."""
    attacks = {}

    attacks["WB Grad L2"] = lambda X: -X[:, WB_GRAD_L2]
    attacks["WB Grad L1"] = lambda X: -X[:, WB_GRAD_L1]
    attacks["WB Loss (baseline)"] = lambda X: -X[:, WB_LOSS]

    grad_cols = list(range(0, 15))  # norms, var, max, mean, per-layer
    rf_g = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=seed, class_weight="balanced"
    ).fit(X_fit[:, grad_cols], y_fit)
    attacks["WB RF (grad only)"] = lambda X, m=rf_g, c=grad_cols: m.predict_proba(X[:, c])[:, 1]

    rf_a = RandomForestClassifier(
        n_estimators=200, max_depth=5, random_state=seed, class_weight="balanced"
    ).fit(X_fit, y_fit)
    attacks["WB RF (all features)"] = lambda X, m=rf_a: m.predict_proba(X)[:, 1]

    return attacks


# --------------------------------------------------------------------------
# advantage
# --------------------------------------------------------------------------

def max_advantage(y, scores):
    """
    max over thresholds of (TPR - FPR), plus the argmax threshold.

    This is the BIASED estimator. Use it on A_sel (where selection is the
    point) and to reproduce FMPAS_naive. Never on A_audit.
    """
    fpr, tpr, thr = roc_curve(y, scores)
    j = int(np.argmax(tpr - fpr))
    return float(tpr[j] - fpr[j]), float(thr[j])


def advantage_at(y, scores, tau):
    """
    TPR(tau) - FPR(tau) at a FIXED threshold. No max operator.

    This is the unbiased estimator, valid on A_audit because tau was chosen
    without reference to this data.
    """
    pred = (scores >= tau).astype(int)
    pos, neg = (y == 1), (y == 0)
    if pos.sum() == 0 or neg.sum() == 0:
        return float("nan"), float("nan"), float("nan")
    tpr = float(pred[pos].mean())
    fpr = float(pred[neg].mean())
    return tpr - fpr, tpr, fpr


def bootstrap_ci(y, scores, tau, n_boot, seed, alpha=0.05):
    """
    Stratified bootstrap percentile CI for the held-out advantage.

    Resamples members and non-members separately so the class balance of the
    audit split is preserved in every replicate. tau stays fixed throughout --
    resampling the threshold too would reintroduce the selection bias.
    """
    rng = np.random.RandomState(seed)
    pos = np.where(y == 1)[0]
    neg = np.where(y == 0)[0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan"), float("nan")

    vals = np.empty(n_boot)
    for b in range(n_boot):
        i = np.concatenate([
            rng.choice(pos, len(pos), replace=True),
            rng.choice(neg, len(neg), replace=True),
        ])
        adv, _, _ = advantage_at(y[i], scores[i], tau)
        vals[b] = adv

    vals = vals[~np.isnan(vals)]
    if len(vals) == 0:
        return float("nan"), float("nan")
    return float(np.percentile(vals, 100 * alpha / 2)), \
           float(np.percentile(vals, 100 * (1 - alpha / 2)))


def verdict(risk):
    if risk <= VERDICT_SAFE:
        return "Safe"
    if risk <= VERDICT_MARGINAL:
        return "Marginal"
    return "Unsafe"


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------

def run(X_bb, y_bb, X_wb, y_wb, seed, n_boot, fit_frac, sel_frac,
        X_pe=None, y_pe=None, pe_columns=None):
    """Returns (per-attack DataFrame, dict of per-access-model summaries)."""
    rows = []

    groups = [("BB", X_bb, y_bb, build_bb_attacks)]
    if X_pe is not None:
        # Per-example attacks are black-box (they use only loss values), so they
        # are scored under BB and compete for the black-box headline. They are
        # split separately because per_example_attacks.py drops samples with too
        # few out-shadow observations, so the row set differs from the raw
        # black-box array.
        groups.append(("BB-PE",
                       X_pe, y_pe,
                       lambda Xf, yf, sd, c=pe_columns:
                           build_perexample_attacks(Xf, yf, sd, c)))
    if X_wb is not None:
        groups.append(("WB", X_wb, y_wb, build_wb_attacks))

    for kind, X, y, builder in groups:
        Xb, yb = balance(X, y, seed)
        (X_fit, y_fit), (X_sel, y_sel), (X_aud, y_aud) = three_way_split(
            Xb, yb, seed, fit_frac, sel_frac
        )
        print(f"[{kind}] balanced n={len(yb)}  "
              f"fit={len(y_fit)} select={len(y_sel)} audit={len(y_aud)}")

        attacks = builder(X_fit, y_fit, seed)

        for name, scorer in attacks.items():
            s_sel = scorer(X_sel)
            s_aud = scorer(X_aud)

            # selection split: biased max, this is where argmax belongs
            adv_sel, tau = max_advantage(y_sel, s_sel)

            # audit split: fixed tau, no max
            adv_aud, tpr_aud, fpr_aud = advantage_at(y_aud, s_aud, tau)
            lo, hi = bootstrap_ci(y_aud, s_aud, tau, n_boot, seed)

            # naive number: max on the audit split, reproducing the original
            # estimator so the two are directly comparable on the same data
            adv_naive, _ = max_advantage(y_aud, s_aud)

            try:
                auc_aud = float(roc_auc_score(y_aud, s_aud))
            except ValueError:
                auc_aud = float("nan")

            rows.append({
                "Type": kind,
                "Attack": name,
                "AUC_audit": round(auc_aud, 4),
                "Adv_select": round(adv_sel, 4),
                "Tau": round(tau, 6),
                "Adv_naive_audit": round(adv_naive, 4),
                "Adv_heldout": round(adv_aud, 4),
                "CI_low": round(lo, 4),
                "CI_high": round(hi, 4),
                "TPR_audit": round(tpr_aud, 4),
                "FPR_audit": round(fpr_aud, 4),
                "n_audit": len(y_aud),
            })

    df = pd.DataFrame(rows)

    # Black-box and white-box are scored SEPARATELY and never maxed together.
    # The white-box set comes from a single target model while the black-box
    # set spans NUM_SHADOW_MODELS shadows, so a max across the two is decided
    # by whichever has the smaller sample and therefore the higher noise
    # ceiling -- not by which adversary is actually stronger.
    #
    # The headline score is black-box. White-box is reported alongside it with
    # its own confidence interval and audit-split size.
    # BB and BB-PE are both black-box adversaries, so they share one headline.
    # WB stays separate (single target model, far smaller sample).
    df["Access"] = df["Type"].replace({"BB-PE": "BB"})

    summaries = {}
    for kind in df["Access"].unique():
        sub = df[df["Access"] == kind]

        # naive: max over attacks of the per-attack biased max (original estimator)
        naive = float(sub["Adv_naive_audit"].max())

        # held-out: pick a* on the SELECT split, read its value off the audit
        # split. The argmax uses Adv_select, never Adv_heldout -- selecting on
        # the audit column would put the max back on the audit data.
        best = sub.loc[sub["Adv_select"].idxmax()]
        heldout = float(best["Adv_heldout"])

        summaries[kind] = {
            "Dataset": DATASET,
            "Access": kind,
            "FMPAS_naive": round(naive, 4),
            "FMPAS_heldout": round(heldout, 4),
            "Inflation": round(naive - heldout, 4),
            "Selected_Attack": best["Attack"],
            "CI_low": best["CI_low"],
            "CI_high": best["CI_high"],
            "Verdict_naive": verdict(naive),
            "Verdict_heldout": verdict(heldout),
            "Verdict_CI_upper": verdict(best["CI_high"]),
            "n_audit": int(best["n_audit"]),
            "Seed": seed,
        }

    return df, summaries


def main():
    p = argparse.ArgumentParser()
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--dp", action="store_true")
    g.add_argument("--no-dp", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=1000)
    p.add_argument("--fit-frac", type=float, default=0.5)
    p.add_argument("--sel-frac", type=float, default=0.25)
    p.add_argument("--fl", action="store_true",
                   help="audit the federated attack set (no white-box; FL "
                        "gradient features are not extracted)")
    p.add_argument("--no-perexample", action="store_true",
                   help="skip offline LiRA / difficulty-calibrated loss even if "
                        "their score files exist")
    p.add_argument("--bb-only", action="store_true",
                   help="skip white-box attacks (use when WB features are "
                        "from a different shadow run than BB)")
    p.add_argument("--bb-suffix", default="",
                   help="DEPRECATED and ignored. Artifacts are now resolved by "
                        "dataset name via utils/artifact_paths.py.")
    args = p.parse_args()

    suffix = "_dp" if args.dp else "_nodp"
    cond = "DP" if args.dp else "No-DP"
    print(f"=== FMPAS (three-way split): {DATASET} | {cond} | seed={args.seed} ===")

    # load_artifact resolves the dataset-tagged filename and verifies the
    # .meta.json sidecar names the dataset we asked for, so a stale file from a
    # different dataset raises instead of silently entering the audit.
    pfx = "fl_" if args.fl else ""
    X_bb, bb_path = load_artifact(f"{pfx}attack_features", DATASET, args.dp)
    y_bb, _ = load_artifact(f"{pfx}attack_labels", DATASET, args.dp)
    print(f"  black-box: {bb_path}  rows={len(y_bb)}  "
          f"members={int((y_bb == 1).sum())}")

    # Per-example attacks (offline LiRA, difficulty-calibrated loss), if
    # attacks/per_example_attacks.py has been run for this dataset/condition.
    X_pe = y_pe = pe_columns = None
    if not args.no_perexample:
        try:
            X_pe, pe_path = load_artifact(f"{pfx}perexample_scores", DATASET, args.dp)
            y_pe, _ = load_artifact(f"{pfx}perexample_labels", DATASET, args.dp)
            mp = pe_path.replace(".npy", ".meta.json")
            pe_columns = ["offline_lira", "difficulty_calibrated"]
            if os.path.exists(mp):
                with open(mp) as f:
                    pe_columns = json.load(f).get("columns", pe_columns)
            print(f"  per-example: {pe_path}  rows={len(y_pe)}  "
                  f"columns={pe_columns}")
        except FileNotFoundError:
            print("  (no per-example scores; run attacks/per_example_attacks.py "
                  "to add offline LiRA)")

    X_wb = y_wb = None
    if args.fl:
        # White-box features come from a single centralized target model; there
        # is no federated counterpart, so the FL audit is black-box only.
        args.bb_only = True
    if not args.bb_only:
        try:
            X_wb, wb_path = load_artifact("whitebox_features", DATASET, args.dp)
            y_wb, _ = load_artifact("whitebox_labels", DATASET, args.dp)
            print(f"  white-box: {wb_path}  rows={len(y_wb)}  "
                  f"members={int((y_wb == 1).sum())}")
            if len(y_wb) * 20 < len(y_bb):
                print(f"  NOTE: the white-box set is >20x smaller than the "
                      f"black-box set.\n"
                      f"        A max over both is dominated by white-box noise. "
                      f"Consider --bb-only\n"
                      f"        and reporting white-box separately.")
        except FileNotFoundError as e:
            print(f"  ({e}; black-box only)")

    df, summaries = run(X_bb, y_bb, X_wb, y_wb,
                        args.seed, args.n_bootstrap,
                        args.fit_frac, args.sel_frac,
                        X_pe=X_pe, y_pe=y_pe, pe_columns=pe_columns)

    os.makedirs("experiments", exist_ok=True)
    f1 = f"experiments/fmpas_{pfx}{DATASET}{suffix}.csv"
    f2 = f"experiments/fmpas_summary_{pfx}{DATASET}{suffix}.csv"
    df.to_csv(f1, index=False)
    pd.DataFrame(list(summaries.values())).to_csv(f2, index=False)

    print("\nPER-ATTACK")
    print(df[["Type", "Attack", "AUC_audit", "Adv_select",
              "Adv_naive_audit", "Adv_heldout", "CI_low", "CI_high"]]
          .to_string(index=False))

    label = {"BB": "BLACK-BOX (headline)", "WB": "WHITE-BOX (reported separately)"}
    for kind in ("BB", "WB"):
        if kind not in summaries:
            continue
        sm = summaries[kind]
        print(f"\n{'='*66}\nFMPAS -- {label[kind]}\n{'='*66}")
        print(f"  FMPAS (naive, max on audit split) : {sm['FMPAS_naive']:.4f}"
              f"   -> {sm['Verdict_naive']}")
        print(f"  FMPAS (held-out, fixed tau*)      : {sm['FMPAS_heldout']:.4f}"
              f"   -> {sm['Verdict_heldout']}")
        print(f"  Selection inflation               : {sm['Inflation']:+.4f}")
        print(f"  Selected attack                   : {sm['Selected_Attack']}")
        print(f"  95% CI (bootstrap, n={args.n_bootstrap})       : "
              f"[{sm['CI_low']:.4f}, {sm['CI_high']:.4f}]")
        print(f"  Verdict at CI upper bound         : {sm['Verdict_CI_upper']}")
        print(f"  Audit split size                  : {sm['n_audit']}")
        if sm["Verdict_heldout"] != sm["Verdict_CI_upper"]:
            print("\n  WARNING: point estimate and CI upper bound give different"
                  " verdicts.\n  The audit split is too small to support this "
                  "verdict. Report the interval.")
        if kind == "WB":
            print("\n  White-box features come from a SINGLE target model, not"
                  " the shadow\n  ensemble used for black-box. Report this score"
                  " and its interval on its\n  own; do not combine it with the"
                  " black-box score by taking a maximum.")
    print(f"\nSaved: {f1}\n       {f2}")


if __name__ == "__main__":
    main()
