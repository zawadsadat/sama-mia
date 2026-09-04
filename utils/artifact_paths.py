"""
Canonical, dataset-tagged paths for saved experiment artifacts.

The objective segment is omitted for the repo default so existing
"unweighted"-era files keep loading; see obj_str().
"""

import os
import json

import numpy as np

EXP_DIR = "experiments"

# Kept out of the filename for this value so that artifacts generated before
# the realistic configuration existed continue to resolve unchanged.
DEFAULT_OBJECTIVE = "unweighted"


def cond_str(use_dp):
    return "dp" if use_dp else "nodp"


def obj_str(objective=None):
    """
    Filename segment for the training objective.

    Returns "" for the default so that experiments/attack_features_X_nodp.npy
    keeps its meaning, and "_weighted" (etc.) otherwise. Reading the objective
    from config when not given means every caller is tagged without having to
    thread the argument through.
    """
    if objective is None:
        from utils.config import OBJECTIVE as objective
    return "" if objective == DEFAULT_OBJECTIVE else f"_{objective}"


def artifact_path(kind, dataset, use_dp, ext="npy", objective=None):
    """kind: attack_features | attack_labels | whitebox_features | whitebox_labels | ..."""
    return os.path.join(
        EXP_DIR,
        f"{kind}_{dataset}{obj_str(objective)}_{cond_str(use_dp)}.{ext}")


def meta_path(kind, dataset, use_dp, objective=None):
    return artifact_path(kind, dataset, use_dp, ext="meta.json",
                         objective=objective)


def save_artifact(kind, dataset, use_dp, array, meta=None):
    """Save an array plus a sidecar describing exactly what it is."""
    os.makedirs(EXP_DIR, exist_ok=True)
    path = artifact_path(kind, dataset, use_dp)
    np.save(path, array)

    from utils.config import OBJECTIVE
    sidecar = {
        "kind": kind,
        "dataset": dataset,
        "objective": OBJECTIVE,
        "condition": cond_str(use_dp),
        "shape": list(np.asarray(array).shape),
    }
    if meta:
        sidecar.update(meta)
    with open(meta_path(kind, dataset, use_dp), "w") as f:
        json.dump(sidecar, f, indent=2, default=float)

    return path


def load_artifact(kind, dataset, use_dp, allow_legacy=True, strict=True):
    """
    Load an artifact, verifying its sidecar names the dataset requested.

    Falls back to the old condition-only names when the tagged file is absent
    (so previously generated Diabetes files still load), but warns loudly --
    those files have no provenance record and cannot be verified.
    """
    path = artifact_path(kind, dataset, use_dp)

    if os.path.exists(path):
        arr = np.load(path)
        mp = meta_path(kind, dataset, use_dp)
        if os.path.exists(mp):
            with open(mp) as f:
                meta = json.load(f)
            from utils.config import OBJECTIVE
            if meta.get("dataset") != dataset:
                msg = (f"ARTIFACT MISMATCH: {path} was generated from "
                       f"'{meta.get('dataset')}' but '{dataset}' was requested.")
                if strict:
                    raise RuntimeError(msg)
                print("  WARNING: " + msg)
            # Objective is absent from artifacts written before it was tracked;
            # only flag a genuine disagreement, not a missing field.
            if meta.get("objective") not in (None, OBJECTIVE):
                msg = (f"ARTIFACT MISMATCH: {path} was generated under "
                       f"objective '{meta.get('objective')}' but '{OBJECTIVE}' "
                       f"was requested. The two configurations use different "
                       f"splits; regenerate with attacks/shadow_models.py.")
                if strict:
                    raise RuntimeError(msg)
                print("  WARNING: " + msg)
        return arr, path

    if not allow_legacy:
        raise FileNotFoundError(f"{path} not found (legacy fallback disabled)")

    for legacy in (os.path.join(EXP_DIR, f"{kind}_{cond_str(use_dp)}_{dataset}.npy"),
                   os.path.join(EXP_DIR, f"{kind}_{cond_str(use_dp)}.npy")):
        if os.path.exists(legacy):
            print(f"  WARNING: using untagged legacy artifact {legacy}.")
            print(f"           It carries no record of which dataset produced it.")
            print(f"           Regenerate to get {path}.")
            return np.load(legacy), legacy

    raise FileNotFoundError(
        f"No artifact for kind={kind} dataset={dataset} "
        f"cond={cond_str(use_dp)}. Expected {path}."
    )
