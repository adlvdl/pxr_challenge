# pxr_challenge
Personal repository on my participation on the OpenADME PXR blind challenge. 
This challenge contains two strands: property prediction and structure elucidation. 
I plan to participate only in the property prediction strand.
More information about the challenge can be found at 
https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/

**Result at a glance:** the Phase-1 submission (a six-model weighted ensemble)
achieved **0.495 MAE** on the unblinded Phase-1 test set, ranking **~78 of 252**
(top third). See [Final submission](#final-submission) for the model details. This
repository is released under the [MIT License](LICENSE).

# Why am I doing this?

- To create an educational resource on the use of ML for drug discovery
- To create a demonstration of my work process

# Technical setup

- Use of open software: rdkit, matplotlib, chemprop, scikit-learn, ...
- Work on marimo notebooks
- Use Claude Code to write code, not direct the analysis
- Post thoughts, progress and ideas about the challenge at my blog https://www.delavega.ai/blog.html


# Final submission

The prediction submitted for the final analog set (analog set 2) is a **weighted
ensemble of five models, each retrained on a counter-screen–filtered and
semi-pure–augmented training set**, using **default hyperparameters**. It is
produced by Analysis 6 of
[`marimo_notebooks/6_ml_optimization_3.py`](marimo_notebooks/6_ml_optimization_3.py)
and written to `submissions/6_ens_default_augfilt_submission.csv`.

## Model architectures

The ensemble combines five complementary regressors. Each predicts pEC50 from a
different molecular representation:

| Tag | Model | Representation | Library |
|---|---|---|---|
| `cp` | Chemprop D-MPNN (trained from scratch) | Molecular graph | chemprop |
| `ch` | Chemprop D-MPNN fine-tuned from the **CheMeleon** foundation backbone | Molecular graph (warm-started) | chemprop |
| `xg` | XGBoost gradient-boosted trees | Mordred descriptors | xgboost |
| `mc` | Macau Bayesian matrix factorization | CheMeleon embeddings (side info) | smurff |
| `tf` | TabPFN in-context regressor | CheMeleon embeddings | tabpfn |

The two graph models (`cp`, `ch`) and TabPFN (`tf`) carry the most weight; the
tree and matrix-factorization models add representational diversity. A sixth
candidate, a Random Forest on Mordred descriptors, was found to have **zero
weight** in the cross-validated weight sweep (notebook 4) and is therefore
excluded.

## Weighting scheme

Predictions are combined as a **normalized weighted average**. The raw weights
(carried over from the best ensemble found by 5×5 cross-validation in notebook 4,
labelled `cp5·ch5·rf0·xg⅓·mc1·tf5`) are:

| Model | Raw weight | Normalized weight |
|---|---|---|
| `cp` (Chemprop scratch) | 5 | ≈ 0.306 |
| `ch` (CheMeleon) | 5 | ≈ 0.306 |
| `xg` (XGBoost) | 1/3 | ≈ 0.020 |
| `mc` (Macau) | 1 | ≈ 0.061 |
| `tf` (TabPFN) | 5 | ≈ 0.306 |

The final prediction for each compound is
`Σ (weightₘ / Σ weights) · predictionₘ`, with the raw weights summing to 16⅓.

## Augmentation and filtering steps

Both data interventions are applied to the **training set only**; the test
compounds are never altered.

1. **Counter-screen filtering.** Each compound's PXR pEC50 is compared against
   its counter-screen pEC50. Compounds where the counter-screen pEC50 is **≥**
   the PXR pEC50 are treated as **non-selective hits** (the apparent activity is
   not PXR-specific) and removed from training. This was the `counter` filter
   evaluated in Analysis 5 in notebook 6.

2. **Semi-pure augmentation.** The OpenADMET **96-compound semi-pure** set
   (≈ 94 usable compounds with purity-corrected pEC50 values, spanning the
   moderate-to-hit range where the models are weakest) is added to the training
   set. Compounds already present in the filtered training set are skipped to
   avoid duplicates. This was the `+semipure` augmentation from Analysis 4 in notebook 6.

After filtering and augmentation, every model is retrained on this single shared
training set (with a 10 % held-out split, seed = 42, used for early stopping in
the models that support it). 

# Blog posts

Each analysis notebook is accompanied by a write-up on
[my blog](https://www.delavega.ai/blog.html). The posts narrate the reasoning and
results; the notebooks contain the runnable code.

| Notebook(s) | Blog post |
|---|---|
| [`0_check_datasets.py`](marimo_notebooks/0_check_datasets.py) | [PXR challenge #0: First contact with the data](https://www.delavega.ai/posts/2026_04_07_first_contact_with_data.html) |
| [`1_sar_exploration.py`](marimo_notebooks/1_sar_exploration.py), [`1a`](marimo_notebooks/1a_data_preprocessing.py)–[`1e`](marimo_notebooks/1e_scaffold_analysis.py) | [PXR challenge #1: Exploring the data](https://www.delavega.ai/posts/2026_04_15_pxr_sar_exploration.html) |
| [`2_ml_baseline.py`](marimo_notebooks/2_ml_baseline.py) | [PXR Challenge #2: Training and Comparing ML Baseline Models](https://www.delavega.ai/posts/2026_04_22_ml_baseline.html) |
| [`3_ml_optimization.py`](marimo_notebooks/3_ml_optimization.py) | [PXR Challenge #3: Ensembling is all you need?](https://www.delavega.ai/posts/2026_05_01_ml_optimization.html) |
| [`4_ml_optimization_2.py`](marimo_notebooks/4_ml_optimization_2.py) | [PXR Challenge #4: Almost at the end of phase 1](https://www.delavega.ai/posts/2026_05_21_ml_optimization_2.html) |
| [`5_unblinded_analysis.py`](marimo_notebooks/5_unblinded_analysis.py) | [PXR Challenge #5: Unblinding the Phase 1 test set](https://www.delavega.ai/posts/2026_06_18_unblinded_analysis.html) |

# Work summary

A notebook-by-notebook account of the analysis that led to the final submission.
The narrative write-ups live in the [blog posts](#blog-posts) linked above.

**[`1a`](marimo_notebooks/1a_data_preprocessing.py) — Data ingestion and preprocessing.**
Reads the four raw challenge CSVs (single-dose, dose-response, counter-screen,
test), computes InChIKeys/InChIs to deduplicate across ~1,500 compounds, and
builds the matched molecular pairs (MMPs) with `mmpdb`. Writes the combined
`all_compounds_activity_data.csv` that every downstream notebook reads.

**[`1b`–`1e`](marimo_notebooks/1b_chemical_space_and_mmp.py) — SAR and chemical-space exploration.**
UMAP/t-SNE embeddings show high chemical diversity with no dominant scaffold
family; activity-cliff analysis finds only ~60 MMP cliffs (4%); train/test
similarity is indistinguishable from within-set similarity. Conclusion: a diverse,
mildly discontinuous dataset with scattered potent compounds — challenging to
model and with a test set well covered by training analogues.

**[`2_ml_baseline.py`](marimo_notebooks/2_ml_baseline.py) — Baseline models and CV protocol.**
Establishes the rigorous **5×5 nested cross-validation** (random split, seed 42)
used throughout, comparing RF, XGBoost, Chemprop and CheMeleon against mean/1-NN
baselines. Random, scaffold and temporal splits give near-identical rankings.
**CheMeleon** (CheMeleon-pretrained D-MPNN) is the clear winner: graph models with
pretrained initialization beat fingerprint models consistently.

**[`3_ml_optimization.py`](marimo_notebooks/3_ml_optimization.py) — Ensembling.**
Sweeps weighted ensembles of the baseline models. The best combinations give zero
weight to RF and XGBoost, favouring CheMeleon + Chemprop. The chosen ensemble cut
test-set MAE from 0.574 (single CheMeleon) to 0.507.

**[`4_ml_optimization_2.py`](marimo_notebooks/4_ml_optimization_2.py) — HPO, new models, Phase-1 submission.**
Adds **Macau** (Bayesian matrix factorization) and **TabPFN** (in-context tabular
model), and runs per-model hyperparameter optimization (XGBoost gained the most,
0.56 → 0.52 MAE). The Phase-1 submission is the six-model weighted ensemble
`cp5·ch5·rf0·xg⅓·mc1·tf5`, which reached **0.495 MAE** and ranked ~78/252 (top third).

**[`5_unblinded_analysis.py`](marimo_notebooks/5_unblinded_analysis.py) — Unblinding the Phase-1 test set.**
With the Phase-1 labels revealed, the ensemble held up best (MAE 0.495, ρ 0.784)
while the CV-champion CheMeleon underperformed. The dominant failure mode is
**regression to the mean**: every model underpredicts potent compounds (by up to
−1.0 log unit in the hit zone), driven by the 46% of test compounds that sit on
activity cliffs. Proposed fixes: reweighting, calibration and uncertainty.

**[`6_ml_optimization_3.py`](marimo_notebooks/6_ml_optimization_3.py) — Remedies and the final ensemble.**
Tests the proposed fixes under the same 5×5 CV — post-hoc calibration, loss
reweighting, oversampling the extremes, semi-pure data augmentation, and
training-set filtering — finding each helps only modestly on its own. The **final
submission** (described [above](#final-submission)) combines the two most promising
data-level interventions, retraining the full five-model ensemble on a
counter-filtered + semi-pure-augmented training set.

# Repository layout

```
data/
  raw/20260403/          # Initial challenge release (single-dose, dose-response, counter-screen, test)
  raw/20260409/          # Re-released challenge CSVs (the canonical training/test inputs)
  raw/20260528/          # Unblinded Phase-1 test labels (used by notebook 5)
  raw/20260619/          # 96-compound semi-pure set (used by notebook 6)
  processed/             # Derived files produced by notebook 1a (activity table + MMP files)
marimo_notebooks/        # Analysis notebooks (1a → 1e exploration, then 2 → 6 modeling)
predictions/             # Cached 5×5 CV out-of-fold predictions + Optuna HPO .db files
submissions/             # Challenge-format CSVs (SMILES, Molecule Name, pEC50)
plots/                   # Static PNG outputs, one subfolder per notebook
checkpoints/             # Saved model checkpoints
logs/                    # Subprocess logs (chemprop CLI, etc.)
html_notebooks/          # Exported HTML snapshots of the notebooks
posts/                   # Local Markdown drafts (published versions are the blog posts above)
```

All raw data, processed files, cached predictions and HPO databases are tracked in
git, so a fresh clone can reproduce the analysis without re-downloading or
re-running the slow steps.

# Getting started

## Prerequisites

- **Python 3.14** — required; earlier versions are not tested.
  If you use [pyenv](https://github.com/pyenv/pyenv), the included `.python-version`
  file will select the correct version automatically.
- **Git** — to clone the repository.

## Setup

```bash
# 1. Clone the repo
git clone <repo-url>
cd pxr_challenge

# 2. Create a virtual environment and install all dependencies
python3.14 -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate

pip install -r requirements.txt
```

All raw data files are already tracked in git — no separate download step is required.

## Running the notebooks

The notebooks use relative paths (e.g. `../data/...`) and must be launched
**from inside the `marimo_notebooks/` directory**. They fall into two stages:

```bash
cd marimo_notebooks

# Stage 1 — data preprocessing & exploration (fast, CPU-only)
marimo run 1a_data_preprocessing.py      # produces data/processed/ files (run first)
marimo run 1b_chemical_space_and_mmp.py
marimo run 1c_activity_cliffs.py
marimo run 1d_train_test_exploration.py
marimo run 1e_scaffold_analysis.py

# Stage 2 — modeling, ensembling & submission (slow; see note below)
marimo run 2_ml_baseline.py              # baselines + 5×5 CV protocol
marimo run 3_ml_optimization.py          # weighted ensembling
marimo run 4_ml_optimization_2.py        # HPO, Macau/TabPFN, Phase-1 submission
marimo run 5_unblinded_analysis.py       # unblinded Phase-1 analysis
marimo run 6_ml_optimization_3.py        # remedies + final submission
```

Run them in order: **1a must be run first** — it writes
`data/processed/all_compounds_activity_data.csv` and the MMP files that all
downstream notebooks read as input. The modeling notebooks (2–6) read predictions
cached by earlier notebooks from `predictions/`, so they are also best run in
sequence.

To open a notebook in edit mode (interactive cells, code visible):

```bash
marimo edit 1a_data_preprocessing.py
```

> **Note on the modeling notebooks (2–6).** These train Chemprop, CheMeleon,
> Macau and TabPFN by shelling out to subprocesses (the chemprop CLI, isolated
> Python workers) and benefit greatly from a GPU or Apple-Silicon MPS device.
> Training is slow, so each notebook **caches its 5×5 CV predictions to
> `predictions/` and skips retraining when the cache exists** — committed caches
> let a fresh clone reproduce the figures and submissions in minutes. Delete the
> relevant `predictions/*.csv.gz` file to force a retrain.

## MMP indexing (one-time, slow step)

Notebook `1a` calls `mmpdb fragment` and `mmpdb index` via subprocess.
Both steps are **skipped automatically if the output files already exist**
(`data/processed/all_compounds_mmp.frag` and `all_compounds_mmp.mmp.csv.gz`).
These pre-computed files are tracked in git, so the fragmentation step will be
skipped on a fresh clone.

If you need to rerun it (e.g. after updating the compound list), delete the
output files and re-run `1a`.

# License

This project is licensed under the MIT License — see the [LICENSE](LICENSE) file
for details.