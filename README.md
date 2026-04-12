# pxr_challenge
Personal repository on my participation on the OpenADME PXR blind challenge. 
This challenge contains two strands: property prediction and structure elucidation. 
I plan to participate only in the property prediction strand.
More information about the challenge can be found at 
https://openadmet.ghost.io/announcing-the-next-openadmet-blind-challenge-predicting-pxr-induction/

# Why am I doing this?

- To create an educational resource on the use of ML for drug discovery
- To create a demonstration of my work process

# Technical setup

- Use of open software: rdkit, matplotlib, chemprop, scikit-learn, ...
- Work on marimo notebooks
- Use Claude Code to write code, not direct the analysis
- Post thoughts, progress and ideas about the challenge at my blog https://adlvdl.github.io/blog.html

# Expected workflow

This is a preliminary plan of the work to be done:

1. Data analysis and preprocessing: downloading the data, exploring general SAR character of the dataset, think whether any compound data is better removed or altered for training, explore public datasets that might enhance the prediction

2. Generate data splits: as we will be comparing ML model predictions to chose the best one, I plan to follow the process outline in a recent paper by the Polaris group to generate 5x5 cross-validated data splits to make statistical sound comparisons. The main point to explore is whether to generate random, scaffold or time based splits

3. Generate single task baseline: this will likely be a comparison from different fingerprints used on RF and XGB as well as chemprop

4. Explore multitask settings and/or finetuning models for property prediction: previous challenges showed the impact of external data to improve predictions so this will be an important aspect. It is also possible that different data available in the challenge can be modeled separately in this manner

5. Provide predictions for analog set 1: this held out dataset will be unblinded in the middle of the challenge. This will provide important information for how the different models performed prospectively and might suggest alterations before submitting predictions for analog set 2

6. Provide predictions for analog set 2: this will be the final step and will be the set on which the participant will be ranked.

# Repository layout

```
data/
  raw/20260409/          # Challenge CSVs (tracked in git — no download needed)
  processed/             # Derived files produced by notebook 1a
marimo_notebooks/        # Analysis notebooks (run in order: 1a → 1b → … → 1e)
plots/1_sar_exploration/ # Static PNG outputs written by the notebooks
posts/                   # Markdown write-ups of each analysis stage
html_notebooks/          # Exported HTML snapshots of the notebooks
```

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
**from inside the `marimo_notebooks/` directory**:

```bash
cd marimo_notebooks
marimo run 1a_data_preprocessing.py      # produces data/processed/ files
marimo run 1b_chemical_space_and_mmp.py
marimo run 1c_activity_cliffs.py
marimo run 1d_train_test_exploration.py
marimo run 1e_scaffold_analysis.py
```

Run them in order: **1a must be run first** — it writes
`data/processed/all_compounds_activity_data.csv` and the MMP files that all
downstream notebooks read as input.

To open a notebook in edit mode (interactive cells, code visible):

```bash
marimo edit 1a_data_preprocessing.py
```

## MMP indexing (one-time, slow step)

Notebook `1a` calls `mmpdb fragment` and `mmpdb index` via subprocess.
Both steps are **skipped automatically if the output files already exist**
(`data/processed/all_compounds_mmp.frag` and `all_compounds_mmp.mmp.csv.gz`).
These pre-computed files are tracked in git, so the fragmentation step will be
skipped on a fresh clone.

If you need to rerun it (e.g. after updating the compound list), delete the
output files and re-run `1a`.