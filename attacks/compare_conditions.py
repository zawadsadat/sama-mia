"""
Usage:
    python attacks/compare_conditions.py --n-bootstrap 2000 --seed 42
    python attacks/compare_conditions.py --a-dp --b-dp     # compare two runs
"""

import sys, os, argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import numpy as np
from sklearn.metrics import roc_curve

from utils.config import DATASET, OBJECTIVE, NUM_SHADOW_MODELS, TEST_SIZE, VAL_FRACTION
from utils.artifact_paths import load_artifact
from utils.data_loader import get_raw_data
from attacks.per_example_attacks import (reconstruct_shadow_indices,
                                         out_statistics, score_rows)
from attacks.compute_fmpas_v2 import (build_attacks, balance, split3,
                                      max_adv, adv_at, COLS)
from attacks.compute_fmpas_v2 import build_matrix


def fit_and_select(dp, seed, grouped=True, fl=False):
    """
    Score one condition and return everything needed to re-evaluate it on an
    arbitrary subset of audit records:

        scores  : the selected attack's score for every audit row
        y       : membership label for every audit row
        g       : record id for every audit row
        tau     : the threshold selected on that condition's selection split
        name    : which attack was selected
        adv     : the point estimate on the full audit split
    """
    X, y, g = build_matrix(dp, fl=fl)
    Xb, yb, gb = balance(X, y, g, seed)
    i_fit, i_sel, i_aud = split3(Xb, yb, gb, seed, grouped)

    atk = build_attacks(Xb[i_fit], yb[i_fit], seed)
    best_name, best_sel, best_tau = None, -np.inf, 0.5
    for name, sc in atk.items():
        a_sel, tau = max_adv(yb[i_sel], sc(Xb[i_sel]))
        if a_sel > best_sel:
            best_name, best_sel, best_tau = name, a_sel, tau

    s_aud = atk[best_name](Xb[i_aud])
    return dict(scores=s_aud, y=yb[i_aud], g=gb[i_aud], tau=best_tau,
                name=best_name, adv=adv_at(yb[i_aud], s_aud, best_tau))


def paired_delta(a, b, n_boot, seed):
    """
    Cluster bootstrap over the records the two conditions share.

    Records present in only one condition's audit split are dropped: a paired
    difference is only defined on records both conditions actually scored. The
    number dropped is reported, because a large overlap loss would mean the two
    audit splits are not comparable and the pairing is doing little.
    """
    shared = np.intersect1d(np.unique(a["g"]), np.unique(b["g"]))
    if len(shared) == 0:
        raise SystemExit("no shared audit records between the two conditions")

    idx_a = {r: np.where(a["g"] == r)[0] for r in shared}
    idx_b = {r: np.where(b["g"] == r)[0] for r in shared}

    rng = np.random.RandomState(seed)
    deltas, a_vals, b_vals = [], [], []
    for _ in range(n_boot):
        pick = rng.choice(shared, len(shared), replace=True)
        ia = np.concatenate([idx_a[r] for r in pick])
        ib = np.concatenate([idx_b[r] for r in pick])
        va = adv_at(a["y"][ia], a["scores"][ia], a["tau"])
        vb = adv_at(b["y"][ib], b["scores"][ib], b["tau"])
        if np.isnan(va) or np.isnan(vb):
            continue
        a_vals.append(va); b_vals.append(vb); deltas.append(va - vb)

    d = np.asarray(deltas)
    return dict(
        n_shared=len(shared),
        n_only_a=len(np.setdiff1d(np.unique(a["g"]), shared)),
        n_only_b=len(np.setdiff1d(np.unique(b["g"]), shared)),
        delta=float(d.mean()),
        lo=float(np.percentile(d, 2.5)),
        hi=float(np.percentile(d, 97.5)),
        p_two_sided=float(2 * min((d <= 0).mean(), (d >= 0).mean())),
        corr=float(np.corrcoef(a_vals, b_vals)[0, 1]),
        sd_paired=float(d.std()),
        sd_unpaired=float(np.sqrt(np.var(a_vals) + np.var(b_vals))),
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--a-dp", action="store_true",
                   help="condition A uses DP (default: A = no DP)")
    p.add_argument("--b-dp", dest="b_dp", action="store_true", default=None,
                   help="condition B uses DP")
    p.add_argument("--b-no-dp", dest="b_dp", action="store_false",
                   help="force condition B to no-DP. Needed for federated "
                        "comparisons: FedAvg runs here are non-private by "
                        "construction, so no fl_*_dp artifact exists and the "
                        "DP default would look for a file that cannot be "
                        "produced.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n-bootstrap", type=int, default=2000)
    p.add_argument("--row-split", action="store_true")
    p.add_argument("--a-fl", action="store_true",
                   help="condition A is federated (default: centralized)")
    p.add_argument("--b-fl", action="store_true",
                   help="condition B is federated")
    args = p.parse_args()

    # Default for condition B: DP when comparing training mechanisms, but
    # no-DP when B is federated, since the federated pipeline is non-private.
    # An explicit --b-dp / --b-no-dp always wins.
    if args.b_dp is None:
        b_dp = not args.b_fl
    else:
        b_dp = args.b_dp

    grouped = not args.row_split
    lab = lambda dp: "DP" if dp else "No-DP"

    print(f"\n{DATASET} | {OBJECTIVE} | "
          f"Delta = FMPAS({'FL ' if args.a_fl else 'cent. '}{lab(args.a_dp)}) - "
          f"FMPAS({'FL ' if args.b_fl else 'cent. '}{lab(b_dp)}) | "
          f"{'record' if grouped else 'row'} bootstrap\n")

    a = fit_and_select(args.a_dp, args.seed, grouped, fl=args.a_fl)
    b = fit_and_select(b_dp, args.seed, grouped, fl=args.b_fl)
    tag_a = ("FL " if args.a_fl else "cent. ") + lab(args.a_dp)
    tag_b = ("FL " if args.b_fl else "cent. ") + lab(b_dp)
    for tag, c in ((tag_a, a), (tag_b, b)):
        print(f"  {tag:<12} FMPAS {c['adv']:+.4f}   attack={c['name']}")

    r = paired_delta(a, b, args.n_bootstrap, args.seed)

    print(f"\n  shared audit records : {r['n_shared']} "
          f"(only in A: {r['n_only_a']}, only in B: {r['n_only_b']})")
    print(f"  Delta                : {r['delta']:+.4f}  "
          f"95% CI [{r['lo']:+.4f}, {r['hi']:+.4f}]")
    print(f"  two-sided p          : {r['p_two_sided']:.4f}")
    print(f"  correlation of the two estimates across replicates: {r['corr']:+.3f}")
    print(f"  SD of Delta: paired {r['sd_paired']:.4f} vs "
          f"unpaired {r['sd_unpaired']:.4f}")

    excl = (r["lo"] > 0) or (r["hi"] < 0)
    print(f"\n  {'Interval excludes zero' if excl else 'Interval contains zero'}"
          f" -- the difference is {'' if excl else 'not '}resolved at this audit size.")
    print("  Pairing removes shared audit-sample noise only. The two conditions\n"
          "  come from different trained models, so training-seed variance is\n"
          "  NOT removed and this interval understates total uncertainty.\n")


if __name__ == "__main__":
    main()
