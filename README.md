# Membership Privacy Auditing in Centralized and Federated Learning: Multi-Attack Evaluation and DP-SGD Analysis

Code for *Membership Privacy Auditing in Centralized and Federated Learning: Multi-Attack Evaluation and DP-SGD Analysis*.

The Selection-Aware Membership Advantage (SAMA) introduced in this paper is the advantage of the
membership-inference attack and threshold chosen on a selection split and
evaluated once, unchanged, on a record-disjoint audit split, reported with a
cluster-bootstrap interval and the state of the audited model. This repository
reproduces every table and figure in the paper.

---

## Install

```bash
conda create -n sama python=3.10 && conda activate sama
pip install -r requirements.txt
```

Tested with PyTorch 2.5.1 (CUDA 12.1), Opacus, scikit-learn and Flower.
A GPU is not required; the largest job takes a few hours on CPU.

## Data

Three datasets download automatically on first use. Diabetes must be fetched
manually because of its licence:

| Dataset | Source | Action |
|---|---|---|
| Breast Cancer Wisconsin | scikit-learn built-in | none |
| Heart Disease (combined) | UCI id 45 | automatic |
| Cardiotocography | UCI id 193 | automatic |
| Diabetes 130-US Hospitals | [UCI id 296](https://archive.ics.uci.edu/dataset/296/) | see below |

```bash
mkdir -p data
# download diabetic_data.csv from the link above into data/
```

---

## Configuration

Two switches in `utils/config.py` select what runs. Every script reads them at
import, so set them before launching.

```python
DATASET   = "breast_cancer"   # breast_cancer | heart_disease
                              # cardiotocography | diabetes_hospital
OBJECTIVE = "unweighted"      # unweighted = memorization stress test
                              # weighted   = realistic configuration
NOISE_MULTIPLIER = 0.8        # sigma for the DP runs
```

`OBJECTIVE` does more than change the loss. `unweighted` is the stress test: a
fixed 50-epoch budget, a two-way split and no early stopping. `weighted` is the
realistic configuration: prevalence-weighted loss, a three-way split, and
stopping at the best validation AUROC. Validation rows are excluded from the
attack set, so the two configurations produce different audit sizes.

Only Heart Disease and Cardiotocography support `weighted`. Diabetes has no
learnable signal at this architecture, and Breast Cancer's 28 training records
cannot yield a usable validation split.

Artifacts are named by dataset, objective and privacy condition, and are
verified against the requested configuration on load, so one configuration
cannot be scored against another's artifacts.

Scripting the switches:

```bash
sed -i 's/^DATASET = .*/DATASET = "cardiotocography"/' utils/config.py
sed -i 's/^OBJECTIVE = .*/OBJECTIVE = "weighted"/'     utils/config.py
```

---

## Pipeline

Run in order. Each step consumes the previous step's artifacts.

```bash
# 1. shadow ensemble -> attack features
python attacks/shadow_models.py --no-dp
python attacks/shadow_models.py --dp

# 2. per-example reference statistics (difficulty-calibrated loss, offline LiRA)
python attacks/per_example_attacks.py --no-dp
python attacks/per_example_attacks.py --dp

# 3. white-box gradient features (optional; reported separately)
python attacks/whitebox_attack.py --no-dp
python attacks/whitebox_attack.py --dp

# 4. SAMA, row-level partition
python attacks/compute_fmpas.py --no-dp --seed 42 --n-bootstrap 2000
python attacks/compute_fmpas.py --dp    --seed 42 --n-bootstrap 2000

# 5. SAMA, record-disjoint partition with a cluster bootstrap (headline)
python attacks/compute_fmpas_v2.py --no-dp --seed 42 --n-bootstrap 2000
python attacks/compute_fmpas_v2.py --dp    --seed 42 --n-bootstrap 2000

# 6. paired test of the DP reduction
python attacks/compare_conditions.py --seed 42 --n-bootstrap 2000
```

Step 2 prints `row count match` and `label vector match`. **Both must read
YES.** They confirm that the replayed per-shadow splits line up with the saved
features; a NO means the artifacts were generated under a different
configuration and everything downstream is misaligned.

### Useful flags

| Flag | Script | Effect |
|---|---|---|
| `--bb-only` | `compute_fmpas` | skip the white-box block |
| `--no-perexample` | `compute_fmpas` | ablation: conventional attacks only |
| `--row-split` | `compute_fmpas_v2` | row-level rather than record-disjoint |
| `--fl` | `per_example_attacks`, `compute_fmpas*` | score the federated artifacts |
| `--b-fl` | `compare_conditions` | compare centralized against federated |
| `--b-no-dp` | `compare_conditions` | force condition B to no-DP |

---

## Reproducing the paper

**Utility, loss gaps and degeneracy counts.** Trains its own
targets and shadows; independent of the pipeline above.

```bash
python experiments/run_multiseed.py --seeds 5 --shadows 10 --no-dp
python experiments/run_multiseed.py --seeds 5 --shadows 10 --dp
```

**Federated comparison.** Cardiotocography only.

```bash
python attacks/fl_shadow_models.py
python attacks/per_example_attacks.py --no-dp --fl
python attacks/compute_fmpas.py --no-dp --fl --seed 42 --n-bootstrap 2000
python attacks/compare_conditions.py --b-fl --seed 42 --n-bootstrap 2000
```

**Privacy budget sweep.** Trains a fresh target and ensemble at
each noise level; several hours on Diabetes.

```bash
python experiments/run_epsilon_sweep.py --no-resume
```

`--no-resume` forces recomputation. Without it, cached results in
`experiments/eps_ckpt/` are reused and the sweep silently reports stale values.

---

## Reproducibility

Every generator is seeded per run. Seeding PyTorch alone is **not** sufficient
for the DP path, since Opacus draws its Gaussian noise from its own generator;
`seed_everything()` covers `random`, `numpy`, `torch` and CUDA. Two identical
DP invocations should produce byte-identical output, including the selected
stopping epoch and tuned threshold — worth checking after any change to the
private path.

Headline values use audit-split seed 42.

## Layout

```
attacks/      shadow ensembles, attack portfolios, scoring
  shadow_models.py        black-box features from the shadow ensemble
  per_example_attacks.py  difficulty-calibrated loss and offline LiRA
  whitebox_attack.py      gradient features from a single target model
  compute_fmpas.py        SAMA, row-level partition
  compute_fmpas_v2.py     SAMA, record-disjoint with cluster bootstrap
  compare_conditions.py   paired test between two conditions
  fl_shadow_models.py     FedAvg shadow ensembles
experiments/  multi-seed runs, noise sweep, plotting
models/       target architecture and DP-SGD training
utils/
  config.py         the two switches, split fractions, DP parameters
  splits.py         three-way split, early stopping, threshold tuning
  objective.py      unweighted and prevalence-weighted losses
  artifact_paths.py naming and provenance verification
  metrics.py        TPR-at-FPR and PPV
```

## Notes

- `utils/data_loader_diabetes_hospital.py` requires the `is_numeric_dtype`
  check for pandas ≥ 2.2; older code testing `dtype == "object"` crashes on the
  median imputation.
- Feature standardization and imputation are fitted on each model's own
  training split. This is a data-dependent step outside the DP-SGD mechanism,
  so a reported epsilon covers the optimization rather than the full pipeline.
- `experiments/*.npy` are not tracked. They are reproducible from the scripts
  and a seed, and the Diabetes attack set alone is 33 MB.
