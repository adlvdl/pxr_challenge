import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 7 — Phase 2 unblinded test set analysis

    The challenge has finished.  The final, fully unblinded **Phase 2** test set
    (`pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`, 260 compounds) is now available.
    This notebook is the post-mortem: it mirrors the structure of
    `5_unblinded_analysis.py` (the Phase 1 post-mortem) but focuses on the Phase 2
    labels and evaluates **every submission we produced** — not just the ensemble
    components.

    It answers three questions:

    1. **Dataset profile** — how does the Phase 2 pEC50 distribution compare with
       the training set *and* with the Phase 1 test set?  Do the three sets probe
       the same activity range?
    2. **Activity cliffs** — how many structurally similar (train, Phase 2) compound
       pairs have very different potency?  These are the hardest cases for any model.
    3. **Model ranking** — recomputing MAE (the competition ranking metric) alongside
       RMSE / R² / Pearson *r* / Spearman ρ for *all* submissions against the Phase 2
       truth: which submissions worked best by MAE, which worked worst, and — crucially
       — would any submission have beaten our final one
       (`6_ens_default_augfilt_submission.csv`)?

    **Inputs:**
    - Phase 2 unblinded CSV: `data/raw/20260703/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`.
    - Phase 1 unblinded CSV (cached by notebook 5): `data/raw/20260528/dose_response_test_unblinded.csv`.
    - Every submission CSV in `submissions/` ending in `_submission.csv`.
    - The external competitor blend `data/raw/20260703/rank9_gashaw_submission_blend_751510.csv`
      (shown for reference only — it is somebody else's model, not ours).
    - Processed training data: `data/processed/all_compounds_activity_data.csv`.
    - Pre-generated MMP database: `data/processed/all_compounds_mmp.mmp.csv.gz`.

    Every submission covers the full 513-compound test set (253 Phase 1 + 260
    Phase 2), so unlike notebook 5 we do **not** need to regenerate any predictions —
    we simply slice each submission to the 260 Phase 2 compounds and score it.
    """)
    return


@app.cell
def _():
    import glob
    import math
    from pathlib import Path
    from typing import Optional

    import altair as alt
    import marimo as mo
    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    from scipy.stats import gaussian_kde, pearsonr, spearmanr
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")

    # Directory where every figure produced by this notebook is written.
    PLOT_DIR = Path("../plots/7_unblinded_phase2_analysis")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    return (
        Chem,
        Optional,
        Path,
        PLOT_DIR,
        alt,
        gaussian_kde,
        glob,
        math,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pearsonr,
        pl,
        plt,
        r2_score,
        rdDepictor,
        rdMolDraw2D,
        spearmanr,
    )


@app.cell
def _(Chem, Optional, rdDepictor, rdMolDraw2D):
    # ── Molecule drawing / identifier helpers (reused from notebook 5) ────────────

    def smi_to_svg(smi: str, width: int = 280, height: int = 200) -> str:
        """Render a SMILES string as an SVG image via RDKit MolDraw2DSVG."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def strip_xml_decl(svg: str) -> str:
        """Remove the <?xml ... ?> declaration so SVG embeds cleanly in HTML."""
        return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

    def smi_to_inchikey(smi: str) -> Optional[str]:
        """Return the InChIKey for *smi*, or None if the SMILES cannot be parsed."""
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol) if mol else None

    def smi_to_inchi(smi: str) -> Optional[str]:
        """Return the InChI for *smi*, or None if the SMILES cannot be parsed."""
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchi(mol) if mol else None

    return smi_to_inchi, smi_to_inchikey


@app.cell
def _(mo):
    mo.md(r"""
    ## Load unblinded test sets

    Both the Phase 1 and Phase 2 unblinded CSVs share the same column layout.  We
    standardise the verbose column names (same renaming used in notebook 5) and add
    InChIKey / InChI identifiers so we can join against the processed training data.
    """)
    return


@app.cell
def _(Path, pl, smi_to_inchi, smi_to_inchikey):
    # Long assay column names are identical between Phase 1 and Phase 2 — factor the
    # rename map out so both sets are standardised the same way.
    _RENAME_MAP = {
        "pEC50_std.error (-log10(molarity))":              "pEC50_se",
        "pEC50_ci.lower (-log10(molarity))":               "pEC50_ci_lower",
        "pEC50_ci.upper (-log10(molarity))":               "pEC50_ci_upper",
        "Emax_estimate (log2FC vs. baseline)":             "Emax",
        "Emax_std.error (log2FC vs. baseline)":            "Emax_se",
        "Emax_ci.lower (log2FC vs. baseline)":             "Emax_ci_lower",
        "Emax_ci.upper (log2FC vs. baseline)":             "Emax_ci_upper",
        "Emax.vs.pos.ctrl_estimate (dimensionless)":       "Emax_vs_ctrl",
        "Emax.vs.pos.ctrl_std.error (dimensionless)":      "Emax_vs_ctrl_se",
        "Emax.vs.pos.ctrl_ci.lower (dimensionless)":       "Emax_vs_ctrl_ci_lower",
        "Emax.vs.pos.ctrl_ci.upper (dimensionless)":       "Emax_vs_ctrl_ci_upper",
    }

    def load_unblinded(path: Path) -> pl.DataFrame:
        """Read an unblinded CSV, add identifiers, and standardise column names."""
        return (
            pl.read_csv(path)
            .rename({"SMILES": "smiles"})
            .with_columns(
                pl.col("smiles").map_elements(smi_to_inchikey, return_dtype=pl.Utf8).alias("inchikey"),
                pl.col("smiles").map_elements(smi_to_inchi,    return_dtype=pl.Utf8).alias("inchi"),
            )
            .rename(_RENAME_MAP)
        )

    # Phase 2 is the focus of this notebook; Phase 1 is loaded only for the
    # three-way distribution comparison.
    PHASE2_PATH = Path("../data/raw/20260703/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")
    PHASE1_PATH = Path("../data/raw/20260528/dose_response_test_unblinded.csv")

    unblinded: pl.DataFrame = load_unblinded(PHASE2_PATH)      # Phase 2 (main)
    unblinded_p1: pl.DataFrame = load_unblinded(PHASE1_PATH)   # Phase 1 (reference)

    print("Phase 2 shape:", unblinded.shape)
    print("Phase 1 shape:", unblinded_p1.shape)
    unblinded
    return unblinded, unblinded_p1


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset overview — three-way activity distribution

    The dose-response pEC50 describes potency on a −log10(molarity) scale: higher
    values mean more potent compounds.  We overlay three distributions on a single
    axis — the **training set**, the **Phase 1** unblinded test set, and the
    **Phase 2** unblinded test set — to check whether all three probe the same
    activity range.

    A compound is typically classified as a **hit** when pEC50 ≥ 6 (EC₅₀ ≤ 1 µM)
    together with a positive Emax; the dashed line marks the pEC50 = 6 threshold.
    """)
    return


@app.cell
def _(
    gaussian_kde,
    mo,
    np,
    pl,
    plt,
    PLOT_DIR,
    unblinded: "pl.DataFrame",
    unblinded_p1: "pl.DataFrame",
):
    # Load training pEC50 values for comparison.
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")

    train_pec50 = (
        all_compounds
        .filter(pl.col("in_dose_response") & pl.col("pEC50_dr").is_not_null())
        .get_column("pEC50_dr")
        .to_numpy()
    )
    p1_pec50 = unblinded_p1.get_column("pEC50").to_numpy()
    p2_pec50 = unblinded.get_column("pEC50").to_numpy()

    # Common x-axis spanning all three distributions.
    _lo = min(train_pec50.min(), p1_pec50.min(), p2_pec50.min()) - 0.3
    _hi = max(train_pec50.max(), p1_pec50.max(), p2_pec50.max()) + 0.3
    x_range = np.linspace(_lo, _hi, 400)

    # (label, values, colour) for each series.
    _series = [
        (f"Training (n={len(train_pec50):,})",        train_pec50, "#4e79a7"),
        (f"Phase 1 — unblinded (n={len(p1_pec50):,})", p1_pec50,    "#59a14f"),
        (f"Phase 2 — unblinded (n={len(p2_pec50):,})", p2_pec50,    "#e15759"),
    ]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig_dist, ax_dist = plt.subplots(figsize=(7.5, 4.8), dpi=150)
        for _lbl, _vals, _col in _series:
            _kde = gaussian_kde(_vals, bw_method="scott")
            ax_dist.plot(x_range, _kde(x_range), color=_col, linewidth=2, label=_lbl)
            ax_dist.fill_between(x_range, _kde(x_range), alpha=0.12, color=_col)
        ax_dist.axvline(6.0, color="black", linestyle="--", linewidth=1.2,
                        label="Hit threshold (pEC50 = 6)")
        ax_dist.set_xlabel("pEC50  [−log₁₀(M)]", fontsize=12)
        ax_dist.set_ylabel("Density", fontsize=12)
        ax_dist.set_title("pEC50 distribution — training vs. Phase 1 vs. Phase 2", fontsize=13)
        ax_dist.legend(fontsize=10, frameon=True, framealpha=0.9)
        fig_dist.tight_layout()
        fig_dist.savefig(PLOT_DIR / "pec50_distribution_train_p1_p2.png",
                         dpi=300, bbox_inches="tight")

    mo.center(mo.as_html(fig_dist))
    return all_compounds, p1_pec50, p2_pec50, train_pec50


@app.cell
def _(mo, np, p1_pec50, p2_pec50, train_pec50):
    # ── Per-set summary statistics table ─────────────────────────────────────────
    def _stats(vals: "np.ndarray") -> dict:
        """Summary statistics for one pEC50 array, incl. hit fraction (pEC50 ≥ 6)."""
        return {
            "n":       len(vals),
            "mean":    float(np.mean(vals)),
            "median":  float(np.median(vals)),
            "std":     float(np.std(vals)),
            "min":     float(np.min(vals)),
            "max":     float(np.max(vals)),
            "pct_hit": 100.0 * float(np.mean(vals >= 6.0)),
        }

    _t = _stats(train_pec50)
    _p1 = _stats(p1_pec50)
    _p2 = _stats(p2_pec50)

    mo.md(f"""
    ### Distribution summary

    | Statistic | Training | Phase 1 | Phase 2 |
    |---|---|---|---|
    | n compounds | {_t['n']:,} | {_p1['n']:,} | {_p2['n']:,} |
    | Mean pEC50 | {_t['mean']:.3f} | {_p1['mean']:.3f} | {_p2['mean']:.3f} |
    | Median pEC50 | {_t['median']:.3f} | {_p1['median']:.3f} | {_p2['median']:.3f} |
    | Std pEC50 | {_t['std']:.3f} | {_p1['std']:.3f} | {_p2['std']:.3f} |
    | Min pEC50 | {_t['min']:.3f} | {_p1['min']:.3f} | {_p2['min']:.3f} |
    | Max pEC50 | {_t['max']:.3f} | {_p1['max']:.3f} | {_p2['max']:.3f} |
    | Hit fraction (pEC50 ≥ 6) | {_t['pct_hit']:.1f}% | {_p1['pct_hit']:.1f}% | {_p2['pct_hit']:.1f}% |

    **Reading the table:** a large gap between the training mean/median and the
    Phase 2 mean/median, or a much smaller Phase 2 hit fraction, would signal that
    the final test set is shifted toward weaker compounds than the models were
    trained on — a covariate shift that inflates test error relative to
    cross-validation.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Final submission — reference predictions

    The activity-cliff scatter below is coloured by the prediction error of our
    **final submitted model**, `6_ens_default_augfilt_submission.csv`.  We load it
    once here and join it to the Phase 2 truth so the residual is available to the
    structural analysis; the full model comparison follows later.
    """)
    return


@app.cell
def _(Path, pl, unblinded: "pl.DataFrame"):
    # Ground-truth table: Molecule Name → true Phase 2 pEC50.
    truth: pl.DataFrame = unblinded.select(["Molecule Name", "pEC50"]).rename(
        {"pEC50": "pEC50_true"}
    )

    # Path to the model we actually submitted as our final entry.
    FINAL_SUBMISSION = Path("../submissions/6_ens_default_augfilt_submission.csv")

    # Residuals of the final submission on the 260 Phase 2 compounds.
    final_resid: pl.DataFrame = (
        pl.read_csv(FINAL_SUBMISSION)
        .select(["Molecule Name", "pEC50"])
        .rename({"pEC50": "final_pred"})
        .join(truth, on="Molecule Name", how="inner")
        .with_columns(
            (pl.col("final_pred") - pl.col("pEC50_true")).alias("final_residual"),
        )
    )

    print("Final submission residuals joined for", final_resid.shape[0], "compounds")
    return FINAL_SUBMISSION, final_resid, truth


@app.cell
def _(mo):
    mo.md(r"""
    ## Nearest-neighbour similarity to training set and activity cliffs

    Before evaluating model performance, we examine structural similarity between each
    **Phase 2** test compound and its closest analogue in the dose-response training
    set, then ask whether structural similarity predicts activity similarity.

    An **activity cliff** is a pair of structurally similar compounds with substantially
    different potency — they are notoriously difficult for ML models because a small
    structural change produces a large activity change.

    Definitions used here (identical to notebook 5):

    | Criterion | Threshold |
    |---|---|
    | Structurally similar | ECFP4 Tanimoto ≥ 0.4 |
    | Activity cliff | Similar **and** \|ΔpEC50\| ≥ 1.0 log unit |

    For each Phase 2 compound we report its nearest neighbour (NN) in the training set,
    its Tanimoto similarity, and the activity difference between the two.
    """)
    return


@app.cell
def _(
    Chem,
    all_compounds,
    np,
    pl,
    unblinded: "pl.DataFrame",
    unblinded_p1: "pl.DataFrame",
):
    from rdkit.Chem import AllChem
    from rdkit import DataStructs as _DataStructs

    # Training set restricted to compounds with a measured pEC50.
    _train_df = (
        all_compounds
        .filter(pl.col("in_dose_response") & pl.col("pEC50_dr").is_not_null())
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "molecule_names", "pEC50_dr"])
    )

    def _ecfp4(smi: str):
        """Return ECFP4 BitVect (radius 2, 1024 bits) for a SMILES string."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=1024)

    # Precompute the training-set fingerprints once — reused for both test phases.
    _train_fps    = [_ecfp4(s) for s in _train_df["smiles"].to_list()]
    _train_ids    = _train_df["inchikey"].to_list()
    _train_smiles = _train_df["smiles"].to_list()
    _train_names  = _train_df["molecule_names"].to_list()
    _train_pec50s = _train_df["pEC50_dr"].to_list()

    def build_nn_cliff(test_df: pl.DataFrame) -> pl.DataFrame:
        """
        For every compound in *test_df*, find its nearest ECFP4 training neighbour and
        classify the pair as an activity cliff, similar/concordant, or dissimilar.

        Same thresholds as notebook 5: similar = Tanimoto ≥ 0.4; cliff = similar and
        |ΔpEC50| ≥ 1.0.  Works for any test frame with the standard unblinded columns.
        """
        _test_fps = [_ecfp4(s) for s in test_df["smiles"].to_list()]
        _max_sims: list[float] = []
        _nn_idx:   list[int]   = []
        for _fp in _test_fps:
            if _fp is None:
                _max_sims.append(float("nan"))
                _nn_idx.append(0)
                continue
            _sims = _DataStructs.BulkTanimotoSimilarity(_fp, _train_fps)
            _best = int(np.argmax(_sims))
            _max_sims.append(float(_sims[_best]))
            _nn_idx.append(_best)

        return (
            test_df
            .select(["Molecule Name", "smiles", "inchikey", "pEC50", "Emax"])
            .with_columns(
                pl.Series("nn_sim",         _max_sims,                           dtype=pl.Float32),
                pl.Series("nn_inchikey",    [_train_ids[i]    for i in _nn_idx],  dtype=pl.Utf8),
                pl.Series("nn_smiles",      [_train_smiles[i] for i in _nn_idx],  dtype=pl.Utf8),
                pl.Series("nn_name",        [_train_names[i]  for i in _nn_idx],  dtype=pl.Utf8),
                pl.Series("nn_pEC50_train", [_train_pec50s[i] for i in _nn_idx],  dtype=pl.Float64),
            )
            .with_columns(
                (pl.col("pEC50") - pl.col("nn_pEC50_train")).alias("delta_pEC50"),
                (pl.col("pEC50") - pl.col("nn_pEC50_train")).abs().alias("abs_delta_pEC50"),
            )
            .with_columns(
                pl.when(
                    (pl.col("nn_sim") >= 0.4) & (pl.col("abs_delta_pEC50") >= 1.0)
                )
                .then(pl.lit("Activity cliff"))
                .when(pl.col("nn_sim") >= 0.4)
                .then(pl.lit("Similar / concordant"))
                .otherwise(pl.lit("Dissimilar"))
                .alias("pair_class")
            )
        )

    # Phase 2 is the focus (drives the scatter + interactive browser); Phase 1 is
    # computed only so the summary table can show it as a comparison column.
    nn_cliff_df:    pl.DataFrame = build_nn_cliff(unblinded)
    nn_cliff_df_p1: pl.DataFrame = build_nn_cliff(unblinded_p1)

    nn_cliff_df
    return nn_cliff_df, nn_cliff_df_p1


@app.cell
def _(
    final_resid: "pl.DataFrame",
    mo,
    nn_cliff_df: "pl.DataFrame",
    np,
    pl,
    plt,
    PLOT_DIR,
):
    # ── Static scatter — coloured by our final submission's residual error ────────
    _cliff_with_res = nn_cliff_df.join(
        final_resid.select(["Molecule Name", "final_residual"]),
        on="Molecule Name", how="left",
    )

    _residuals = _cliff_with_res["final_residual"].to_numpy()
    _abs_max = max(abs(np.nanmin(_residuals)), abs(np.nanmax(_residuals)))

    # Marker style per pair_class so structural context is preserved.
    _marker_map = {
        "Activity cliff":       ("^", 55, 1.0),
        "Similar / concordant": ("o", 30, 0.80),
        "Dissimilar":           ("s", 25, 0.65),
    }

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_cliff, _ax_cliff = plt.subplots(figsize=(7.5, 5.5), dpi=150)

        for _cls, (_mkr, _sz, _alpha) in _marker_map.items():
            _sub = _cliff_with_res.filter(pl.col("pair_class") == _cls)
            _sc = _ax_cliff.scatter(
                _sub["nn_sim"].to_numpy(),
                _sub["abs_delta_pEC50"].to_numpy(),
                c=_sub["final_residual"].to_numpy(),
                cmap="RdBu_r",
                vmin=-_abs_max,
                vmax=_abs_max,
                marker=_mkr,
                s=_sz,
                alpha=_alpha,
                edgecolors="none",
                label=_cls,
            )

        _cb = _fig_cliff.colorbar(_sc, ax=_ax_cliff, fraction=0.035, pad=0.02)
        _cb.set_label("Final submission residual  (pred − true pEC50)", fontsize=10)

        _ax_cliff.axvline(0.4, color="#555", linestyle="--", linewidth=1, alpha=0.7,
                          label="Similarity threshold (0.4)")
        _ax_cliff.axhline(1.0, color="#555", linestyle=":",  linewidth=1, alpha=0.7,
                          label="|ΔpEC50| threshold (1.0)")

        _n_cliff_static = nn_cliff_df.filter(pl.col("pair_class") == "Activity cliff").shape[0]
        _ax_cliff.set_xlabel("ECFP4 Tanimoto similarity to nearest training neighbour", fontsize=11)
        _ax_cliff.set_ylabel("|ΔpEC50|  (test pEC50 − nearest train pEC50)", fontsize=11)
        _ax_cliff.set_title(
            f"Phase 2 activity cliff analysis  —  {_n_cliff_static} cliff(s)\n"
            "Colour = final-model residual (red = overpredicted, blue = underpredicted);\n"
            "markers: △ cliff, ○ similar/concordant, □ dissimilar",
            fontsize=10,
        )
        _ax_cliff.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
        _ax_cliff.set_xlim(-0.02, 1.02)
        _fig_cliff.tight_layout()
        _fig_cliff.savefig(
            PLOT_DIR / "activity_cliffs_scatter.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_cliff))
    return


@app.cell
def _(mo, nn_cliff_df: "pl.DataFrame", nn_cliff_df_p1: "pl.DataFrame", pl):
    # ── Summary statistics — Phase 1 vs Phase 2 side by side ──────────────────────
    def _nn_summary(df: pl.DataFrame) -> dict:
        """Nearest-neighbour cliff summary counts/stats for one test-set frame."""
        _n = df.shape[0]
        return {
            "n":         _n,
            "n_similar": df.filter(pl.col("nn_sim") >= 0.4).shape[0],
            "n_cliffs":  df.filter(pl.col("pair_class") == "Activity cliff").shape[0],
            "n_concord": df.filter(pl.col("pair_class") == "Similar / concordant").shape[0],
            "n_dissim":  df.filter(pl.col("pair_class") == "Dissimilar").shape[0],
            "sim_mean":  float(df["nn_sim"].mean()),
            "sim_med":   float(df["nn_sim"].median()),
            "delt_mean": float(df["abs_delta_pEC50"].mean()),
            "delt_med":  float(df["abs_delta_pEC50"].median()),
        }

    def _cnt_pct(k: str, s: dict) -> str:
        """Render a 'count (pct%)' cell for count-style metrics."""
        return f"{s[k]} ({100 * s[k] / s['n']:.1f}%)"

    _p1 = _nn_summary(nn_cliff_df_p1)
    _p2 = _nn_summary(nn_cliff_df)

    mo.md(f"""
    ### Activity cliff summary — nearest-neighbour view

    Phase 1 is shown alongside Phase 2 for comparison (both measured against the same
    dose-response training set, identical thresholds).

    | Metric | Phase 1 | Phase 2 |
    |---|---|---|
    | Test compounds analysed | {_p1['n']} | {_p2['n']} |
    | Similar pairs (Tanimoto ≥ 0.4) | {_cnt_pct('n_similar', _p1)} | {_cnt_pct('n_similar', _p2)} |
    | **Activity cliffs** (similar + \\|ΔpEC50\\| ≥ 1.0) | **{_cnt_pct('n_cliffs', _p1)}** | **{_cnt_pct('n_cliffs', _p2)}** |
    | Similar / concordant | {_cnt_pct('n_concord', _p1)} | {_cnt_pct('n_concord', _p2)} |
    | Dissimilar (Tanimoto < 0.4) | {_cnt_pct('n_dissim', _p1)} | {_cnt_pct('n_dissim', _p2)} |
    | Mean Tanimoto sim to NN | {_p1['sim_mean']:.3f} | {_p2['sim_mean']:.3f} |
    | Median Tanimoto sim to NN | {_p1['sim_med']:.3f} | {_p2['sim_med']:.3f} |
    | Mean \\|ΔpEC50\\| (all pairs) | {_p1['delt_mean']:.3f} | {_p2['delt_mean']:.3f} |
    | Median \\|ΔpEC50\\| (all pairs) | {_p1['delt_med']:.3f} | {_p2['delt_med']:.3f} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive browser — Phase 2 compound and nearest training neighbour

    Hover over a point in the scatter to display both compound structures and
    their activity data side by side.  Red points (activity cliffs) are the
    most informative: structurally similar to a training compound but with
    substantially different potency — a direct challenge for ML models.
    """)
    return


@app.cell
def _(alt, mo, nn_cliff_df: "pl.DataFrame"):
    _cliff_color_scale = alt.Scale(
        domain=["Activity cliff", "Similar / concordant", "Dissimilar"],
        range=["#e15759", "#76b7b2", "#b0b0b0"],
    )

    _cliff_sel = alt.selection_point(
        fields=["Molecule Name"], name="cliff_sel",
        empty=False, on="mouseover", nearest=True, clear="mouseout",
    )

    _cliff_scatter_int = (
        alt.Chart(nn_cliff_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("nn_sim:Q",
                    title="Tanimoto similarity to nearest training compound",
                    scale=alt.Scale(domain=[0, 1]),
                    axis=alt.Axis(titleFontSize=11)),
            y=alt.Y("abs_delta_pEC50:Q",
                    title="|ΔpEC50|  (test − nearest train)",
                    axis=alt.Axis(titleFontSize=11)),
            color=alt.condition(
                _cliff_sel,
                alt.value("#f5c518"),
                alt.Color("pair_class:N", scale=_cliff_color_scale,
                          legend=alt.Legend(title="Pair class")),
            ),
            size=alt.condition(_cliff_sel, alt.value(140), alt.value(60)),
            tooltip=[
                alt.Tooltip("Molecule Name:N",   title="Test compound"),
                alt.Tooltip("pEC50:Q",           title="Test pEC50",     format=".3f"),
                alt.Tooltip("nn_name:N",         title="NN train cmpd"),
                alt.Tooltip("nn_pEC50_train:Q",  title="Train NN pEC50", format=".3f"),
                alt.Tooltip("nn_sim:Q",          title="Tanimoto",       format=".3f"),
                alt.Tooltip("abs_delta_pEC50:Q", title="|ΔpEC50|",       format=".3f"),
                alt.Tooltip("pair_class:N",      title="Class"),
            ],
        )
        .add_params(_cliff_sel)
        .properties(title="Hover to inspect compound pair", width=470, height=370)
        .configure_title(fontSize=12)
    )

    nn_cliff_chart = mo.ui.altair_chart(_cliff_scatter_int)
    return (nn_cliff_chart,)


@app.cell
def _(
    Chem,
    mo,
    nn_cliff_chart,
    nn_cliff_df: "pl.DataFrame",
    pl,
    rdDepictor,
    rdMolDraw2D,
):
    _PW, _PH = 250, 175

    def _svg_mol(smi: str) -> str:
        """Render SMILES as inline SVG stripped of the XML declaration."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(_PW, _PH)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

    _cliff_sel_rows = nn_cliff_chart.value

    if _cliff_sel_rows is None or len(_cliff_sel_rows) == 0:
        _cliff_panel = mo.Html(f"""
            <div style='width:{_PW}px; height:{(_PH + 60) * 2}px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:13px; border:1px dashed #ccc;
                        border-radius:6px; text-align:center; padding:12px'>
                Hover over a point to see the<br>test compound and its nearest<br>training neighbour
            </div>
        """)
    else:
        _ckey = _cliff_sel_rows.row(0, named=True)["Molecule Name"]
        _cr   = nn_cliff_df.filter(pl.col("Molecule Name") == _ckey).row(0, named=True)
        _cbg  = "#fde8e8" if _cr["pair_class"] == "Activity cliff" else "#eef4fb"
        _csign = "+" if _cr["delta_pEC50"] >= 0 else ""
        _cliff_panel = mo.Html(f"""
            <div style='width:{_PW}px; font-family:monospace; font-size:10px'>
                <div style='padding:5px; background:{_cbg}; border-radius:4px;
                            text-align:center; margin-bottom:3px; line-height:1.6'>
                    <b>Test compound</b><br>{_cr['Molecule Name']}<br>
                    pEC50 = <b>{_cr['pEC50']:.3f}</b>
                </div>
                {_svg_mol(_cr['smiles'])}
                <div style='padding:5px; background:#f0faf0; border-radius:4px;
                            text-align:center; margin-top:6px; margin-bottom:3px; line-height:1.6'>
                    <b>Nearest train compound</b><br>{_cr['nn_name'] or _cr['nn_inchikey']}<br>
                    pEC50 = <b>{_cr['nn_pEC50_train']:.3f}</b> &nbsp;·&nbsp;
                    Tanimoto = <b>{_cr['nn_sim']:.3f}</b><br>
                    ΔpEC50 = {_csign}{_cr['delta_pEC50']:.3f} &nbsp;
                    <b style='color:{"#e15759" if _cr["pair_class"] == "Activity cliff" else "#555"}'>{_cr['pair_class']}</b>
                </div>
                {_svg_mol(_cr['nn_smiles'])}
            </div>
        """)

    mo.hstack([nn_cliff_chart, _cliff_panel], align="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Matched molecular pair (MMP) analysis — training vs. Phase 2 test set

    Matched molecular pairs are compound pairs that differ by exactly one structural
    transformation at a single site while sharing a common scaffold (the **constant**
    fragment).  Because the change is chemically precise, MMP analysis lets us
    attribute an activity difference directly to a specific structural modification.

    The MMP database was pre-generated across the full compound collection.  Here we
    focus only on **cross-set pairs**: one compound from the dose-response training set
    and one from the Phase 2 unblinded test set.  This directly reveals how often
    structural similarity (at the MMP level) corresponds to activity similarity — and,
    crucially, how often it does not (activity cliffs).

    Activity cliff criterion: **|ΔpEC50| ≥ 1.0** log unit between the two MMP partners.
    """)
    return


@app.cell
def _(all_compounds, pl, unblinded: "pl.DataFrame", unblinded_p1: "pl.DataFrame"):
    # ── Load MMP database and orient cross-set pairs ─────────────────────────────
    _mmp = pl.read_csv(
        "../data/processed/all_compounds_mmp.mmp.csv.gz",
        separator="\t",
        has_header=False,
        new_columns=["smiles1", "smiles2", "inchikey1", "inchikey2", "transformation", "constant"],
    )

    _train_act = (
        all_compounds
        .filter(pl.col("in_dose_response") & pl.col("pEC50_dr").is_not_null())
        .unique("inchikey")
        .select(["inchikey", "smiles", "molecule_names", "pEC50_dr"])
    )
    _train_iks = set(_train_act["inchikey"].to_list())

    def build_mmp_cross(test_df: pl.DataFrame) -> pl.DataFrame:
        """
        Extract cross-set matched molecular pairs — one training compound, one test
        compound — for *test_df*, and classify each pair as an activity cliff
        (|ΔpEC50| ≥ 1.0) or concordant.  Reusable across test phases.
        """
        _test_act = test_df.select(["Molecule Name", "smiles", "inchikey", "pEC50"])
        _test_iks = set(_test_act["inchikey"].to_list())

        # Orient so ik_train is always the training compound, ik_test the test compound.
        _a = (
            _mmp
            .filter(pl.col("inchikey1").is_in(_train_iks) & pl.col("inchikey2").is_in(_test_iks))
            .select([
                pl.col("inchikey1").alias("ik_train"),
                pl.col("inchikey2").alias("ik_test"),
                "transformation", "constant",
            ])
        )
        _b = (
            _mmp
            .filter(pl.col("inchikey2").is_in(_train_iks) & pl.col("inchikey1").is_in(_test_iks))
            .select([
                pl.col("inchikey2").alias("ik_train"),
                pl.col("inchikey1").alias("ik_test"),
                "transformation", "constant",
            ])
        )

        return (
            pl.concat([_a, _b])
            .join(
                _train_act.rename({
                    "inchikey":       "ik_train",
                    "smiles":         "smiles_train",
                    "molecule_names": "name_train",
                    "pEC50_dr":       "pEC50_train",
                }),
                on="ik_train", how="left",
            )
            .join(
                _test_act.rename({
                    "inchikey":      "ik_test",
                    "smiles":        "smiles_test",
                    "Molecule Name": "name_test",
                    "pEC50":         "pEC50_test",
                }),
                on="ik_test", how="left",
            )
            .filter(pl.col("pEC50_train").is_not_null() & pl.col("pEC50_test").is_not_null())
            .with_columns(
                (pl.col("pEC50_test") - pl.col("pEC50_train")).alias("delta_pEC50"),
                (pl.col("pEC50_test") - pl.col("pEC50_train")).abs().alias("abs_delta_pEC50"),
            )
            .with_columns(
                pl.when(pl.col("abs_delta_pEC50") >= 1.0)
                .then(pl.lit("Activity cliff"))
                .otherwise(pl.lit("Concordant"))
                .alias("pair_class")
            )
        )

    # Phase 2 drives the figures below; Phase 1 is built only for the summary column.
    mmp_cross_df:    pl.DataFrame = build_mmp_cross(unblinded)
    mmp_cross_df_p1: pl.DataFrame = build_mmp_cross(unblinded_p1)

    # Cluster-level summary: one row per unique scaffold constant (Phase 2).
    mmp_cluster_df: pl.DataFrame = (
        mmp_cross_df
        .group_by("constant")
        .agg(
            pl.len().alias("n_pairs"),
            pl.col("abs_delta_pEC50").max().alias("max_abs_delta"),
            pl.col("abs_delta_pEC50").mean().alias("mean_abs_delta"),
            pl.col("abs_delta_pEC50").median().alias("median_abs_delta"),
            (pl.col("abs_delta_pEC50") >= 1.0).sum().alias("n_cliffs"),
            pl.col("ik_test").n_unique().alias("n_test_cmpds"),
            pl.col("ik_train").n_unique().alias("n_train_cmpds"),
        )
        .with_columns(
            (pl.col("n_cliffs") / pl.col("n_pairs") * 100).round(1).alias("pct_cliffs"),
            # Readable label: constant fragment truncated to 40 chars.
            pl.col("constant").str.slice(0, 40).alias("constant_label"),
        )
        .sort("max_abs_delta", descending=True)
    )

    print(f"Cross-set MMPs: {mmp_cross_df.shape[0]}")
    print(f"Activity cliffs: {mmp_cross_df.filter(pl.col('pair_class')=='Activity cliff').shape[0]}")
    print(f"Unique scaffold clusters: {mmp_cluster_df.shape[0]}")
    print(f"Clusters with ≥1 cliff: {mmp_cluster_df.filter(pl.col('n_cliffs')>=1).shape[0]}")
    mmp_cross_df
    return mmp_cluster_df, mmp_cross_df, mmp_cross_df_p1


@app.cell
def _(
    gaussian_kde,
    mmp_cluster_df: "pl.DataFrame",
    mmp_cross_df: "pl.DataFrame",
    mo,
    np,
    pl,
    plt,
    PLOT_DIR,
):
    # ── Two-panel overview figure ─────────────────────────────────────────────────
    _cliff   = mmp_cross_df.filter(pl.col("pair_class") == "Activity cliff")["abs_delta_pEC50"].to_numpy()
    _concord = mmp_cross_df.filter(pl.col("pair_class") == "Concordant")["abs_delta_pEC50"].to_numpy()
    _all     = mmp_cross_df["abs_delta_pEC50"].to_numpy()

    # Top 20 clusters for the bar panel (sorted by max |ΔpEC50|).
    _top20 = mmp_cluster_df.head(20)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_mmp, (_ax_kde, _ax_bar) = plt.subplots(
            1, 2, figsize=(13, 5), dpi=150,
            gridspec_kw={"width_ratios": [1, 1.4]},
        )

        # ── Left panel: KDE of |ΔpEC50| ──────────────────────────────────────────
        _x = np.linspace(0, _all.max() + 0.3, 400)
        _kde_all   = gaussian_kde(_all,    bw_method="scott")
        _ax_kde.plot(_x, _kde_all(_x),    color="#555", linewidth=2, label=f"All MMPs (n={len(_all)})")
        _ax_kde.fill_between(_x, _kde_all(_x), alpha=0.10, color="#555")

        if len(_cliff) >= 2:
            _kde_cliff = gaussian_kde(_cliff, bw_method="scott")
            _ax_kde.plot(_x, _kde_cliff(_x), color="#e15759", linewidth=2, label=f"Cliffs (n={len(_cliff)})")
            _ax_kde.fill_between(_x, _kde_cliff(_x), alpha=0.15, color="#e15759")

        if len(_concord) >= 2:
            _kde_con = gaussian_kde(_concord, bw_method="scott")
            _ax_kde.plot(_x, _kde_con(_x), color="#76b7b2", linewidth=2, label=f"Concordant (n={len(_concord)})")
            _ax_kde.fill_between(_x, _kde_con(_x), alpha=0.15, color="#76b7b2")

        _ax_kde.axvline(1.0, color="black", linestyle="--", linewidth=1,
                        label="Cliff threshold (1.0)")
        _ax_kde.set_xlabel("|ΔpEC50|  (test − train)", fontsize=11)
        _ax_kde.set_ylabel("Density", fontsize=11)
        _ax_kde.set_xlim(0, _all.max() + 0.2)
        _ax_kde.legend(fontsize=9, frameon=True)
        _ax_kde.set_title("|ΔpEC50| distribution — all cross-set MMPs (Phase 2)", fontsize=11)

        # ── Right panel: top 20 clusters, bar = max |ΔpEC50|, coloured by pct_cliffs ──
        _cluster_labels = [f"C{i+1}" for i in range(len(_top20))]
        _max_deltas     = _top20["max_abs_delta"].to_numpy()
        _pct_cliffs     = _top20["pct_cliffs"].to_numpy()

        _cmap   = plt.cm.RdYlGn_r
        _norm   = plt.Normalize(vmin=0, vmax=100)
        _colors = [_cmap(_norm(v)) for v in _pct_cliffs]

        _bars = _ax_bar.barh(
            _cluster_labels[::-1], _max_deltas[::-1],
            color=_colors[::-1], edgecolor="white", linewidth=0.5,
        )
        _ax_bar.axvline(1.0, color="black", linestyle="--", linewidth=1, alpha=0.7)

        # Annotate bars with n_pairs and cliff counts.
        for _i, (_bar, _row) in enumerate(zip(_bars, list(_top20.iter_rows(named=True))[::-1])):
            _ax_bar.text(
                _bar.get_width() + 0.05,
                _bar.get_y() + _bar.get_height() / 2,
                f"n={_row['n_pairs']} | {_row['n_cliffs']} cliffs ({_row['pct_cliffs']:.0f}%)",
                va="center", fontsize=7.5, color="#333",
            )

        _sm = plt.cm.ScalarMappable(cmap=_cmap, norm=_norm)
        _sm.set_array([])
        _cbar = _fig_mmp.colorbar(_sm, ax=_ax_bar, fraction=0.03, pad=0.02)
        _cbar.set_label("% pairs that are cliffs", fontsize=9)

        _ax_bar.set_xlabel("Max |ΔpEC50| in cluster", fontsize=11)
        _ax_bar.set_title("Top 20 MMP scaffold clusters\n(sorted by max |ΔpEC50|)", fontsize=11)
        _ax_bar.set_xlim(0, _max_deltas.max() + 1.2)

        _fig_mmp.tight_layout()
        _fig_mmp.savefig(
            PLOT_DIR / "mmp_activity_cliffs.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_mmp))
    return


@app.cell
def _(
    mmp_cluster_df: "pl.DataFrame",
    mmp_cross_df: "pl.DataFrame",
    mmp_cross_df_p1: "pl.DataFrame",
    mo,
    pl,
):
    def _mmp_pair_summary(df: pl.DataFrame) -> dict:
        """Pair-level MMP cliff summary for one cross-set frame (phase-agnostic)."""
        _n = df.shape[0]
        _ad = df["abs_delta_pEC50"]
        return {
            "n":        _n,
            "n_cliffs": df.filter(pl.col("pair_class") == "Activity cliff").shape[0],
            "n_concord": df.filter(pl.col("pair_class") == "Concordant").shape[0],
            "q50": float(_ad.quantile(0.50)),
            "q75": float(_ad.quantile(0.75)),
            "q90": float(_ad.quantile(0.90)),
            "q95": float(_ad.quantile(0.95)),
            "max": float(_ad.max()),
        }

    _p1 = _mmp_pair_summary(mmp_cross_df_p1)
    _p2 = _mmp_pair_summary(mmp_cross_df)

    # Scaffold-cluster figures are Phase 2 only (clusters not computed for Phase 1).
    _n_clusters       = mmp_cluster_df.shape[0]
    _n_cliff_clusters = mmp_cluster_df.filter(pl.col("n_cliffs") >= 1).shape[0]

    def _cliff_pct(k: str, s: dict) -> str:
        return f"{s[k]} ({100 * s[k] / s['n']:.1f}%)"

    mo.md(f"""
    ### MMP summary

    Phase 1 is shown alongside Phase 2 for comparison (both cross-set against the same
    training set).  The scaffold-cluster rows are Phase 2 only.

    | Metric | Phase 1 | Phase 2 |
    |---|---|---|
    | Total cross-set MMP pairs | {_p1['n']} | {_p2['n']} |
    | **Activity cliffs** (\\|ΔpEC50\\| ≥ 1.0) | **{_cliff_pct('n_cliffs', _p1)}** | **{_cliff_pct('n_cliffs', _p2)}** |
    | Concordant pairs | {_cliff_pct('n_concord', _p1)} | {_cliff_pct('n_concord', _p2)} |
    | Median \\|ΔpEC50\\| | {_p1['q50']:.2f} | {_p2['q50']:.2f} |
    | 75th percentile \\|ΔpEC50\\| | {_p1['q75']:.2f} | {_p2['q75']:.2f} |
    | 90th percentile \\|ΔpEC50\\| | {_p1['q90']:.2f} | {_p2['q90']:.2f} |
    | 95th percentile \\|ΔpEC50\\| | {_p1['q95']:.2f} | {_p2['q95']:.2f} |
    | Maximum \\|ΔpEC50\\| | {_p1['max']:.2f} | {_p2['max']:.2f} |
    | Unique scaffold clusters (Phase 2) | — | {_n_clusters} |
    | Clusters with ≥ 1 cliff (Phase 2) | — | {_n_cliff_clusters} ({100*_n_cliff_clusters/_n_clusters:.1f}%) |

    **Interpretation:** a high cliff rate here means the structural change between each
    MMP partner leads to a greater than 1 log-unit shift in potency — activity-sensitive
    structural space that the training data does not resolve well, and the most likely
    source of large Phase 2 prediction errors.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model evaluation on the Phase 2 test set

    ### Load every submission

    Unlike notebook 5, which reconstructed the ensemble from its component models,
    here we evaluate **every submission CSV we ever produced** (all files in
    `submissions/` ending in `_submission.csv`) directly against the Phase 2 truth.
    Because each submission covers the full 513-compound test set, scoring is just an
    inner join to the 260 Phase 2 compounds — no prediction regeneration is needed.

    The submissions span the whole modelling arc:

    | Prefix | Notebook | Family |
    |---|---|---|
    | `2_` | Baselines | CheMeleon single model |
    | `3_` | First optimisation | Chemprop / RF / early ensembles |
    | `4_` | Second optimisation | HPO'd components + tuned ensembles |
    | `5_` | Phase 1 regen | Regenerated ensemble |
    | `6_` | Final optimisation | Augment/filter + calibrated ensembles |

    Our **final submitted entry** is `6_ens_default_augfilt_submission.csv` — it is
    flagged separately in every table and chart below.  The external competitor blend
    `rank9_gashaw_submission_blend_751510.csv` is scored too, but marked *(external)*:
    it is not one of our models and is shown only for reference.
    """)
    return


@app.cell
def _(Path, glob, pl, truth: "pl.DataFrame"):
    SUBMISSION_DIR = Path("../submissions")
    FINAL_NAME = "6_ens_default_augfilt_submission.csv"
    EXTERNAL_PATH = Path("../data/raw/20260703/rank9_gashaw_submission_blend_751510.csv")

    def _pretty_label(filename: str) -> str:
        """Turn a submission filename into a compact, human-readable model label."""
        stem = filename.removesuffix("_submission.csv").removesuffix(".csv")
        return stem

    def load_and_score(path: Path, label: str, is_final: bool, is_external: bool) -> pl.DataFrame:
        """
        Load a submission CSV (columns: SMILES, Molecule Name, pEC50) and join with
        the Phase 2 ground truth, attaching per-row error columns and metadata.

        Returns a long-format frame with:
          model_name, is_final, is_external, Molecule Name, pEC50_pred, pEC50_true,
          error, abs_error.
        """
        df = (
            pl.read_csv(path)
            .select(["Molecule Name", "pEC50"])
            .rename({"pEC50": "pEC50_pred"})
        )
        return (
            df.join(truth, on="Molecule Name", how="inner")
            .with_columns(
                pl.lit(label).alias("model_name"),
                pl.lit(is_final).alias("is_final"),
                pl.lit(is_external).alias("is_external"),
                (pl.col("pEC50_pred") - pl.col("pEC50_true")).alias("error"),
                (pl.col("pEC50_pred") - pl.col("pEC50_true")).abs().alias("abs_error"),
            )
        )

    # Every local submission, sorted by filename (chronological by notebook prefix).
    _sub_paths = sorted(SUBMISSION_DIR.glob("*_submission.csv"))

    _frames = [
        load_and_score(
            _p, _pretty_label(_p.name),
            is_final=(_p.name == FINAL_NAME), is_external=False,
        )
        for _p in _sub_paths
    ]
    # External reference blend.
    _frames.append(
        load_and_score(
            EXTERNAL_PATH, "rank9_gashaw_blend (external)",
            is_final=False, is_external=True,
        )
    )

    all_predictions: pl.DataFrame = pl.concat(_frames)

    print(f"Loaded {len(_sub_paths)} local submissions + 1 external, "
          f"{all_predictions.shape[0]} total rows "
          f"({all_predictions['Molecule Name'].n_unique()} unique Phase 2 compounds)")
    all_predictions
    return FINAL_NAME, all_predictions


@app.cell
def _(mo):
    mo.md(r"""
    ### Aggregate metrics per submission

    Five standard regression metrics per submission, plus signed bias:

    | Metric | Description |
    |---|---|
    | **RMSE** | Root mean squared error (lower = better) |
    | **MAE** | Mean absolute error (lower = better) |
    | **R²** | Coefficient of determination (higher = better) |
    | **Pearson r** | Linear correlation between predicted and true pEC50 |
    | **Spearman ρ** | Rank correlation (robust to outliers) |
    | **bias** | Mean signed error (pred − true); >0 = overpredicts |

    The table is sorted by **MAE** ascending — the metric the challenge leaderboard
    used to rank submissions — so the best submission appears first.  The `rank`
    column is the MAE position among our own submissions (the external blend is
    excluded from ranking).
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    math,
    mean_absolute_error,
    mean_squared_error,
    np,
    pearsonr,
    pl,
    r2_score,
    spearmanr,
):
    def compute_metrics(group: pl.DataFrame) -> dict:
        """Compute regression metrics for one submission's predictions."""
        y_true = group.get_column("pEC50_true").to_numpy()
        y_pred = group.get_column("pEC50_pred").to_numpy()
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        return {
            "model_name":   group.get_column("model_name")[0],
            "is_final":     bool(group.get_column("is_final")[0]),
            "is_external":  bool(group.get_column("is_external")[0]),
            "n":            len(y_true),
            "RMSE":         round(rmse, 4),
            "MAE":          round(mae,  4),
            "R2":           round(r2,   4),
            "Pearson_r":    round(r,    4),
            "Spearman_rho": round(rho,  4),
            "bias":         round(float(np.mean(y_pred - y_true)), 4),
        }

    metrics_rows = [
        compute_metrics(grp)
        for grp in all_predictions.partition_by("model_name", maintain_order=True)
    ]

    metrics_df: pl.DataFrame = (
        pl.DataFrame(metrics_rows)
        # MAE is the competition ranking metric, so sort and rank by MAE.
        .sort("MAE")
        # Rank only among our own submissions (external blend gets a null rank).
        .with_columns(
            pl.when(~pl.col("is_external"))
            .then(pl.col("MAE").rank("ordinal").over(pl.col("is_external")))
            .otherwise(None)
            .cast(pl.Int32)
            .alias("rank")
        )
        .select([
            "rank", "model_name", "is_final", "is_external",
            "MAE", "RMSE", "R2", "Pearson_r", "Spearman_rho", "bias", "n",
        ])
    )

    metrics_df
    return (metrics_df,)


@app.cell
def _(FINAL_NAME, metrics_df: "pl.DataFrame", mo, pl):
    # ── Headline comparison: final submission vs. best-possible submission ────────
    _final_label = FINAL_NAME.removesuffix("_submission.csv")
    _own = metrics_df.filter(~pl.col("is_external"))

    _final_row = _own.filter(pl.col("model_name") == _final_label).row(0, named=True)
    _best_row  = _own.sort("MAE").row(0, named=True)
    _worst_row = _own.sort("MAE", descending=True).row(0, named=True)
    _ext_row   = metrics_df.filter(pl.col("is_external")).row(0, named=True)

    _n_own      = _own.shape[0]
    _n_better   = _own.filter(pl.col("MAE") < _final_row["MAE"]).shape[0]
    _final_beats_best = _final_row["model_name"] == _best_row["model_name"]

    _verdict = (
        "Our final submission **was the best** local submission by MAE (the competition "
        "ranking metric) — no other submission would have scored better."
        if _final_beats_best else
        f"**{_n_better} submission(s) would have beaten our final entry** by MAE (the "
        f"competition ranking metric). The best, `{_best_row['model_name']}`, achieves "
        f"MAE={_best_row['MAE']:.4f} vs. our {_final_row['MAE']:.4f} "
        f"(Δ = {_final_row['MAE'] - _best_row['MAE']:+.4f})."
    )

    mo.md(f"""
    ### Headline result — did we submit the best model?

    Ranking metric: **MAE** (as used by the competition leaderboard).

    | Submission | MAE | RMSE | R² | Spearman ρ | Rank (of {_n_own}) |
    |---|---|---|---|---|---|
    | **Final — `{_final_row['model_name']}`** | **{_final_row['MAE']:.4f}** | {_final_row['RMSE']:.4f} | {_final_row['R2']:.4f} | {_final_row['Spearman_rho']:.4f} | **{_final_row['rank']}** |
    | Best local — `{_best_row['model_name']}` | {_best_row['MAE']:.4f} | {_best_row['RMSE']:.4f} | {_best_row['R2']:.4f} | {_best_row['Spearman_rho']:.4f} | {_best_row['rank']} |
    | Worst local — `{_worst_row['model_name']}` | {_worst_row['MAE']:.4f} | {_worst_row['RMSE']:.4f} | {_worst_row['R2']:.4f} | {_worst_row['Spearman_rho']:.4f} | {_worst_row['rank']} |
    | External — `{_ext_row['model_name']}` | {_ext_row['MAE']:.4f} | {_ext_row['RMSE']:.4f} | {_ext_row['R2']:.4f} | {_ext_row['Spearman_rho']:.4f} | — |

    {_verdict}
    """)
    return


@app.cell
def _(alt, metrics_df: "pl.DataFrame", mo, pl):
    # ── Ranking bar chart — final entry highlighted, external shown greyed ────────
    _chart_df = metrics_df.with_columns(
        pl.when(pl.col("is_final")).then(pl.lit("Final submission"))
        .when(pl.col("is_external")).then(pl.lit("External"))
        .otherwise(pl.lit("Other submission"))
        .alias("kind")
    )

    _color_scale = alt.Scale(
        domain=["Final submission", "Other submission", "External"],
        range=["#e15759", "#4e79a7", "#b0b0b0"],
    )

    _bar = (
        alt.Chart(_chart_df)
        .mark_bar()
        .encode(
            x=alt.X("MAE:Q", title="MAE  (lower = better)"),
            y=alt.Y("model_name:N", sort=alt.SortField("MAE", order="ascending"),
                    title=None, axis=alt.Axis(labelLimit=340, labelFontSize=9)),
            color=alt.Color("kind:N", scale=_color_scale,
                            legend=alt.Legend(title=None, orient="bottom")),
            tooltip=[
                alt.Tooltip("rank:Q",         title="Rank"),
                alt.Tooltip("model_name:N",   title="Submission"),
                alt.Tooltip("MAE:Q",          title="MAE",         format=".4f"),
                alt.Tooltip("RMSE:Q",         title="RMSE",        format=".4f"),
                alt.Tooltip("R2:Q",           title="R²",          format=".4f"),
                alt.Tooltip("Pearson_r:Q",    title="Pearson r",   format=".4f"),
                alt.Tooltip("Spearman_rho:Q", title="Spearman ρ", format=".4f"),
                alt.Tooltip("bias:Q",         title="Bias",        format="+.4f"),
            ],
        )
        .properties(
            title="All submissions ranked by MAE on the Phase 2 test set",
            width=520,
            height=430,
        )
        .configure_title(fontSize=13)
    )

    mo.ui.altair_chart(_bar)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Head-to-head statistical comparison (paired bootstrap + Holm-Bonferroni)

    A raw MAE gap between two submissions may be real or may be noise from the finite
    test set.  The challenge organisers describe their significance test in
    [*Peak performance or just noise?*](https://openadmet.ghost.io/peak-performance-or-just-noise/);
    we reproduce it here for our final submission vs. the external Gashaw blend.

    **Method (as described by the organisers):**

    1. **Paired bootstrap.**  Draw `n_boot = 1000` pseudo-test sets by sampling the
       test compounds *with replacement*.  Crucially, the **same** resampled indices
       are used for both participants in each iteration, so the comparison is paired —
       "since each bootstrap iteration uses the same samples for each user, we can
       directly compare participants' performance."
    2. **Per-iteration metric and delta.**  Compute each user's MAE on the pseudo-test
       set, then the delta Δ = MAE(A) − MAE(B).  This yields a bootstrap distribution
       of Δ.
    3. **Confidence interval.**  The 95% CI is the 2.5th–97.5th percentiles of the Δ
       distribution.  If Δ = 0 falls **outside** the CI the difference is significant.
    4. **Two-tailed p-value.**  p = 2 · min( P(Δ > 0), P(Δ < 0) ) — twice the smaller
       tail fraction (the proportion of bootstrap iterations in which the worse
       participant actually came out ahead).
    5. **Holm-Bonferroni (HB) correction.**  Across all *M* pairwise comparisons in the
       challenge, p-values are sorted ascending; the entry at rank *r* is compared to
       the adjusted threshold **α / (M − r + 1)**.  A comparison is significant only if
       its p-value is below its HB-adjusted threshold.

    The MAE point estimates come from the Phase 2 test set; the reported *mean MAE ± SD*
    is the mean and standard deviation of each user's MAE across the 1000 bootstrap
    iterations, which is why it differs slightly from the single-number MAE above.
    """)
    return


@app.cell
def _(all_predictions: "pl.DataFrame", np, pl):
    # ── Head-to-head configuration ───────────────────────────────────────────────
    # Labels as they appear in `all_predictions.model_name`.
    HH_USER_A = "rank9_gashaw_blend (external)"   # "Gashaw" in the organisers' UI
    HH_USER_B = "6_ens_default_augfilt"           # our final submission ("adlvdl")
    HH_LABEL_A = "Gashaw"
    HH_LABEL_B = "adlvdl"

    N_BOOT = 1000          # organisers' standard: 1,000 bootstrap iterations
    ALPHA = 0.05           # nominal significance level
    BOOT_SEED = 0          # fixed seed; reproduces the organisers' reported p = 0.0260

    # Holm-Bonferroni context from the *full challenge* leaderboard.  These two numbers
    # come from the organisers' ranking of every pairwise comparison, so they cannot be
    # recomputed from our local data alone — they are entered to match the reported UI.
    HB_ADJUSTMENT_RANK = 2453     # this pair's rank among all sorted p-values
    HB_TOTAL_COMPARISONS = 95 * 94 // 2   # M = C(95, 2) = 4465 pairwise comparisons
    #   (95 participants on the leaderboard).  HB adjusted threshold =
    #   ALPHA / (M - rank + 1) = 0.05 / 2013 ≈ 0.0000249, which rounds to 0.0000 in the
    #   organisers' displayed table.

    def _abs_err(user: str) -> "np.ndarray":
        """Per-compound absolute error for *user*, aligned by Molecule Name order."""
        return (
            all_predictions
            .filter(pl.col("model_name") == user)
            .sort("Molecule Name")
            .get_column("abs_error")
            .to_numpy()
        )

    # Align both users on the same compound ordering so the pairing is valid.
    _names_a = (
        all_predictions.filter(pl.col("model_name") == HH_USER_A)
        .sort("Molecule Name").get_column("Molecule Name").to_list()
    )
    _names_b = (
        all_predictions.filter(pl.col("model_name") == HH_USER_B)
        .sort("Molecule Name").get_column("Molecule Name").to_list()
    )
    assert _names_a == _names_b, "Users must cover the same compounds for a paired test"

    ae_a = _abs_err(HH_USER_A)
    ae_b = _abs_err(HH_USER_B)
    n_cmpds = len(ae_a)

    # ── Paired bootstrap ─────────────────────────────────────────────────────────
    _rng = np.random.default_rng(BOOT_SEED)
    boot_mae_a = np.empty(N_BOOT)
    boot_mae_b = np.empty(N_BOOT)
    for _i in range(N_BOOT):
        _idx = _rng.integers(0, n_cmpds, n_cmpds)   # shared indices → paired sample
        boot_mae_a[_i] = ae_a[_idx].mean()
        boot_mae_b[_i] = ae_b[_idx].mean()

    boot_delta = boot_mae_a - boot_mae_b            # Δ = MAE(A) − MAE(B)
    return (
        ALPHA,
        BOOT_SEED,
        HB_ADJUSTMENT_RANK,
        HB_LABEL_A,
        HB_LABEL_B,
        HB_TOTAL_COMPARISONS,
        HH_LABEL_A,
        HH_LABEL_B,
        N_BOOT,
        boot_delta,
        boot_mae_a,
        boot_mae_b,
    )


@app.cell
def _(
    ALPHA,
    HB_ADJUSTMENT_RANK,
    HB_TOTAL_COMPARISONS,
    HH_LABEL_A,
    HH_LABEL_B,
    N_BOOT,
    boot_delta,
    boot_mae_a,
    boot_mae_b,
    mo,
    np,
):
    # ── Statistics derived from the bootstrap distribution ───────────────────────
    _mean_a, _sd_a = float(boot_mae_a.mean()), float(boot_mae_a.std(ddof=1))
    _mean_b, _sd_b = float(boot_mae_b.mean()), float(boot_mae_b.std(ddof=1))

    _ci_lo = float(np.quantile(boot_delta, 0.025))
    _ci_hi = float(np.quantile(boot_delta, 0.975))

    # Two-tailed p-value: twice the smaller tail fraction.
    _p_gt = float(np.mean(boot_delta > 0))
    _p_lt = float(np.mean(boot_delta < 0))
    _p_value = 2.0 * min(_p_gt, _p_lt)

    # Holm-Bonferroni adjusted threshold for this pair's rank.
    _hb_threshold = ALPHA / (HB_TOTAL_COMPARISONS - HB_ADJUSTMENT_RANK + 1)

    # Significance: p-value must clear the HB-adjusted threshold.  (Equivalently, Δ = 0
    # would have to sit outside a CI at the corrected level.)
    _significant = _p_value < _hb_threshold

    # Store the summary so downstream cells / tables can reuse it.
    hh_summary = {
        "user_a": HH_LABEL_A, "user_b": HH_LABEL_B,
        "mean_a": _mean_a, "sd_a": _sd_a,
        "mean_b": _mean_b, "sd_b": _sd_b,
        "ci_lo": _ci_lo, "ci_hi": _ci_hi,
        "p_value": _p_value,
        "hb_threshold": _hb_threshold,
        "significant": _significant,
    }

    _sig_cell = (
        "<b style='color:#2e7d32'>TRUE</b>" if _significant
        else "<b style='color:#c62828'>FALSE</b>"
    )

    mo.md(f"""
    ### Head-to-head statistical summary

    | | |
    |---|---|
    | Track | Activity |
    | User A | **{HH_LABEL_A}** |
    | User B | **{HH_LABEL_B}** |
    | Evaluation Metric | MAE |
    | User A mean MAE ± SD | {_mean_a:.4f} ± {_sd_a:.4f} |
    | User B mean MAE ± SD | {_mean_b:.4f} ± {_sd_b:.4f} |
    | Nominal α | {ALPHA} |
    | HB Adjustment Rank | {HB_ADJUSTMENT_RANK} |
    | HB Adjusted Threshold | {_hb_threshold:.4f} |
    | Observed p-value | {_p_value:.4f} |
    | 95% CI of Δ (A − B) | [{_ci_lo:.4f}, {_ci_hi:.4f}] |
    | Statistically Significant | {_sig_cell} |

    **Reading it:** {HH_LABEL_A} has the lower mean MAE, so its model is nominally
    better.  The two-tailed p-value ({_p_value:.4f}) sits above the Holm-Bonferroni
    adjusted threshold ({_hb_threshold:.4f}), so after correcting for the {HB_TOTAL_COMPARISONS:,}
    pairwise comparisons in the challenge the difference is **not statistically
    significant** — {N_BOOT:,} resamples cannot rule out that the gap is test-set noise.
    """)
    return (hh_summary,)


@app.cell
def _(
    HH_LABEL_A,
    HH_LABEL_B,
    boot_delta,
    hh_summary,
    mo,
    np,
    plt,
    PLOT_DIR,
):
    # ── Delta distribution plot — matches the organisers' UI figure ──────────────
    _above = boot_delta > 0     # Δ > 0 → User A (Gashaw) has the HIGHER (worse) MAE
    _below = ~_above

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_hh, _ax_hh = plt.subplots(figsize=(7.5, 5), dpi=150)

        _bins = np.linspace(boot_delta.min(), max(boot_delta.max(), 0.02), 55)
        _ax_hh.hist(boot_delta[_below], bins=_bins, color="#d9915b", alpha=0.9,
                    label=f"Δ < 0 ({HH_LABEL_B} higher MAE)")
        _ax_hh.hist(boot_delta[_above], bins=_bins, color="#6b83c9", alpha=0.9,
                    label=f"Δ > 0 ({HH_LABEL_A} higher MAE)")

        _ax_hh.axvline(0.0, color="black", linestyle=":", linewidth=1.5)
        _ax_hh.text(0.001, _ax_hh.get_ylim()[1] * 0.94, "Δ = 0",
                    fontsize=10, va="top")

        _ax_hh.set_xlabel("Δ MAE  (A − B)", fontsize=12)
        _ax_hh.set_ylabel("Bootstrap Samples", fontsize=12)
        _ax_hh.set_title(
            f"Bootstrap Δ ({HH_LABEL_A} − {HH_LABEL_B}): MAE\n"
            f"p = {hh_summary['p_value']:.4f}  ·  "
            f"95% CI [{hh_summary['ci_lo']:.3f}, {hh_summary['ci_hi']:.3f}]",
            fontsize=12,
        )
        _ax_hh.legend(fontsize=10, frameon=True, framealpha=0.9, loc="upper left")
        _fig_hh.tight_layout()
        _fig_hh.savefig(PLOT_DIR / "head_to_head_delta_distribution.png",
                        dpi=300, bbox_inches="tight")

    mo.center(mo.as_html(_fig_hh))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Are our own submissions statistically distinguishable? — Tier 1 analysis

    We now apply the **same paired-bootstrap + Holm-Bonferroni** procedure to *our own*
    submissions (the external Gashaw blend is excluded).  Two questions:

    1. **Is the best-MAE submission statistically better than our final submission?**
       If not, our final entry was a defensible choice even though a lower-MAE
       submission existed.
    2. **Which submissions form "Tier 1"?**  Following the organisers' framing, Tier 1
       is the set of submissions that are **not significantly different from the best**
       submission — i.e. the top submission and every other submission whose gap to it
       could plausibly be test-set noise.  Everything *significantly worse* than the
       best falls outside Tier 1.

    **Method:**

    - Run one paired bootstrap (1000 iterations, shared resampled compound indices) and
      cache each submission's per-iteration MAE.
    - For **every pair** of submissions compute the two-tailed p-value
      p = 2·min(P(Δ>0), P(Δ<0)) — that is *M = C(n, 2)* comparisons.
    - Sort all p-values ascending and apply Holm-Bonferroni: the comparison at rank *r*
      is significant only if its p-value < α / (M − r + 1), **and** every comparison
      ahead of it in the sorted list was also significant (the HB step-down rule).
    - **Tier 1** membership = a submission whose *best-vs-it* comparison is **not**
      significant under HB (plus the best submission itself).
    """)
    return


@app.cell
def _(ALPHA, N_BOOT, all_predictions: "pl.DataFrame", metrics_df: "pl.DataFrame", np, pl):
    import itertools as _itertools

    # ── Own submissions only, ordered best → worst by MAE ─────────────────────────
    _own_order = (
        metrics_df.filter(~pl.col("is_external")).sort("MAE").get_column("model_name").to_list()
    )
    TIER_BEST = _own_order[0]     # best-MAE submission (Tier 1 anchor)

    # Reference compound ordering (sorted Molecule Name) shared by all submissions.
    _ref_names = (
        all_predictions.filter(pl.col("model_name") == TIER_BEST)
        .sort("Molecule Name").get_column("Molecule Name").to_list()
    )

    def _abs_err_for(model: str) -> "np.ndarray":
        """Per-compound absolute error for *model*, aligned to the shared ordering."""
        _df = (
            all_predictions.filter(pl.col("model_name") == model)
            .sort("Molecule Name")
        )
        assert _df.get_column("Molecule Name").to_list() == _ref_names, (
            f"{model} does not cover the same compounds as the reference"
        )
        return _df.get_column("abs_error").to_numpy()

    _ae = {m: _abs_err_for(m) for m in _own_order}
    _n_cmpds = len(_ref_names)

    # ── One shared paired bootstrap: cache each submission's per-iteration MAE ─────
    # Reusing the SAME resample indices across all submissions keeps every pairwise
    # comparison paired and mutually consistent.
    TIER_SEED = 0
    _rng = np.random.default_rng(TIER_SEED)
    _boot_idx = [_rng.integers(0, _n_cmpds, _n_cmpds) for _ in range(N_BOOT)]

    boot_mae: dict[str, "np.ndarray"] = {
        m: np.array([_ae[m][_idx].mean() for _idx in _boot_idx]) for m in _own_order
    }

    # ── All pairwise two-tailed p-values (M = C(n, 2)) ────────────────────────────
    def _two_tailed_p(delta: "np.ndarray") -> float:
        return 2.0 * min(float(np.mean(delta > 0)), float(np.mean(delta < 0)))

    _pairs = list(_itertools.combinations(_own_order, 2))
    _pair_rows = []
    for _a, _b in _pairs:
        _delta = boot_mae[_a] - boot_mae[_b]
        _pair_rows.append({
            "model_a": _a,
            "model_b": _b,
            "mean_mae_a": float(boot_mae[_a].mean()),
            "mean_mae_b": float(boot_mae[_b].mean()),
            "delta_mean": float(_delta.mean()),
            "ci_lo": float(np.quantile(_delta, 0.025)),
            "ci_hi": float(np.quantile(_delta, 0.975)),
            "p_value": _two_tailed_p(_delta),
        })

    _M = len(_pairs)

    # ── Holm-Bonferroni step-down across all pairwise p-values ────────────────────
    pairwise_df: pl.DataFrame = (
        pl.DataFrame(_pair_rows)
        .sort("p_value")
        .with_row_index("hb_rank", offset=1)
        .with_columns(
            (pl.lit(ALPHA) / (pl.lit(_M) - pl.col("hb_rank") + 1)).alias("hb_threshold"),
        )
    )
    # Raw per-comparison rejection, then enforce the step-down (once a test fails, all
    # higher-ranked tests fail too).
    _raw_reject = (pairwise_df["p_value"] < pairwise_df["hb_threshold"]).to_list()
    _stepdown: list[bool] = []
    _still_ok = True
    for _r in _raw_reject:
        _still_ok = _still_ok and _r
        _stepdown.append(_still_ok)

    pairwise_df = pairwise_df.with_columns(
        pl.Series("significant", _stepdown, dtype=pl.Boolean)
    )

    TIER_TOTAL_COMPARISONS = _M
    return (
        TIER_BEST,
        TIER_SEED,
        TIER_TOTAL_COMPARISONS,
        boot_mae,
        pairwise_df,
    )


@app.cell
def _(TIER_BEST, boot_mae, metrics_df: "pl.DataFrame", np, pairwise_df: "pl.DataFrame", pl):
    # ── Extract every "best vs. other" comparison to assign Tier 1 membership ──────
    def _best_vs(model: str) -> dict:
        """Return the pairwise-row stats for the (best, model) comparison, oriented
        as Δ = MAE(model) − MAE(best) so a positive Δ means 'worse than best'."""
        _row = pairwise_df.filter(
            ((pl.col("model_a") == TIER_BEST) & (pl.col("model_b") == model))
            | ((pl.col("model_a") == model) & (pl.col("model_b") == TIER_BEST))
        ).row(0, named=True)
        # Orient Δ so it is (model − best).
        if _row["model_a"] == TIER_BEST:
            _dmean, _lo, _hi = -_row["delta_mean"], -_row["ci_hi"], -_row["ci_lo"]
        else:
            _dmean, _lo, _hi = _row["delta_mean"], _row["ci_lo"], _row["ci_hi"]
        return {
            "p_value": _row["p_value"],
            "hb_rank": _row["hb_rank"],
            "hb_threshold": _row["hb_threshold"],
            "significant": _row["significant"],
            "delta_vs_best": _dmean, "ci_lo": _lo, "ci_hi": _hi,
        }

    _own_order = (
        metrics_df.filter(~pl.col("is_external")).sort("MAE").get_column("model_name").to_list()
    )

    _rows = []
    for _m in _own_order:
        _mean_mae = float(boot_mae[_m].mean())
        if _m == TIER_BEST:
            _rows.append({
                "model_name": _m, "mean_mae": _mean_mae,
                "delta_vs_best": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "p_value": None, "hb_rank": None, "hb_threshold": None,
                "sig_vs_best": False, "tier": "Tier 1 (best)",
            })
        else:
            _bv = _best_vs(_m)
            _in_tier1 = not _bv["significant"]
            _rows.append({
                "model_name": _m, "mean_mae": _mean_mae,
                "delta_vs_best": _bv["delta_vs_best"],
                "ci_lo": _bv["ci_lo"], "ci_hi": _bv["ci_hi"],
                "p_value": _bv["p_value"], "hb_rank": _bv["hb_rank"],
                "hb_threshold": _bv["hb_threshold"],
                "sig_vs_best": _bv["significant"],
                "tier": "Tier 1" if _in_tier1 else "Outside Tier 1",
            })

    tier_df: pl.DataFrame = pl.DataFrame(_rows)
    tier_df
    return (tier_df,)


@app.cell
def _(FINAL_NAME, TIER_BEST, TIER_TOTAL_COMPARISONS, mo, pairwise_df: "pl.DataFrame", pl, tier_df: "pl.DataFrame"):
    _final_label = FINAL_NAME.removesuffix("_submission.csv")

    # Best-vs-final comparison.
    _bf = pairwise_df.filter(
        ((pl.col("model_a") == TIER_BEST) & (pl.col("model_b") == _final_label))
        | ((pl.col("model_a") == _final_label) & (pl.col("model_b") == TIER_BEST))
    ).row(0, named=True)

    _best_is_final = TIER_BEST == _final_label
    _n_tier1 = tier_df.filter(pl.col("tier").str.starts_with("Tier 1")).shape[0]
    _n_out   = tier_df.filter(pl.col("tier") == "Outside Tier 1").shape[0]
    _outside = tier_df.filter(pl.col("tier") == "Outside Tier 1")

    if _best_is_final:
        _bf_verdict = (
            f"Our final submission `{_final_label}` **is** the best-MAE submission, so "
            "the best-vs-final question is moot."
        )
    else:
        _bf_sig = _bf["significant"]
        _bf_verdict = (
            f"Best (`{TIER_BEST}`) vs. final (`{_final_label}`): p = {_bf['p_value']:.4f}, "
            f"HB rank {_bf['hb_rank']} of {TIER_TOTAL_COMPARISONS}, adjusted threshold "
            f"{_bf['hb_threshold']:.4f} → "
            + (
                "**significantly different** — the best submission is statistically "
                "better than our final one."
                if _bf_sig else
                "**not significantly different** — although the best submission has a "
                "lower MAE, the gap is within bootstrap noise, so our final submission "
                "is statistically indistinguishable from the best."
            )
        )

    _out_list = (
        ", ".join(f"`{r['model_name']}` (p={r['p_value']:.4f})"
                  for r in _outside.iter_rows(named=True))
        if _n_out > 0 else "— none —"
    )

    mo.md(f"""
    ### Tier 1 verdict

    **Best submission (anchor):** `{TIER_BEST}`
    **Total pairwise comparisons (M):** {TIER_TOTAL_COMPARISONS}

    {_bf_verdict}

    **Tier 1** ({_n_tier1} of {tier_df.shape[0]} submissions — not significantly worse
    than the best): every submission except the {_n_out} below.

    **Outside Tier 1** ({_n_out} — significantly worse than the best under Holm-Bonferroni):
    {_out_list}

    > **Why some near-best submissions fall outside Tier 1 while worse ones stay in.**
    > The test is *paired*: significance depends on the variance of Δ = MAE(model) − MAE(best),
    > not on the raw MAE gap.  Submissions that are structurally similar to the best (e.g.
    > the other tuned ensembles) are highly correlated with it, so their Δ has a very small
    > CI — even a tiny, consistent gap becomes significant.  A dissimilar submission (a bare
    > baseline, a single-model) has a much wider Δ CI, so a *larger* MAE gap can still overlap
    > zero.  The forest plot below makes this explicit: red points have narrow CIs that clear
    > zero; blue points have CIs that straddle it.
    """)
    return


@app.cell
def _(mo, pl, tier_df: "pl.DataFrame"):
    # ── Full Tier table, sorted best → worst by bootstrap-mean MAE ────────────────
    _disp = (
        tier_df
        .with_columns(
            pl.col("mean_mae").round(4),
            pl.col("delta_vs_best").round(4),
            pl.when(pl.col("p_value").is_null()).then(None)
            .otherwise(pl.col("p_value").round(4)).alias("p_value"),
            pl.when(pl.col("hb_threshold").is_null()).then(None)
            .otherwise(pl.col("hb_threshold").round(4)).alias("hb_threshold"),
        )
        .select([
            "model_name", "mean_mae", "delta_vs_best",
            "p_value", "hb_rank", "hb_threshold", "tier",
        ])
    )

    _hdr = "| Submission | Mean MAE | Δ vs best | p (vs best) | HB rank | HB thresh | Tier |\n"
    _hdr += "|---|---|---|---|---|---|---|\n"
    _body = ""
    for _r in _disp.iter_rows(named=True):
        _p = "—" if _r["p_value"] is None else f"{_r['p_value']:.4f}"
        _hbr = "—" if _r["hb_rank"] is None else str(_r["hb_rank"])
        _hbt = "—" if _r["hb_threshold"] is None else f"{_r['hb_threshold']:.4f}"
        _tier_cell = (
            f"**{_r['tier']}**" if _r["tier"] == "Outside Tier 1" else _r["tier"]
        )
        _body += (
            f"| `{_r['model_name']}` | {_r['mean_mae']:.4f} | "
            f"{_r['delta_vs_best']:+.4f} | {_p} | {_hbr} | {_hbt} | {_tier_cell} |\n"
        )

    mo.md("### Per-submission Tier table\n\n" + _hdr + _body)
    return


@app.cell
def _(TIER_BEST, mo, np, pl, plt, PLOT_DIR, tier_df: "pl.DataFrame"):
    # ── Forest plot: Δ MAE vs best submission, with 95% CIs, coloured by Tier ──────
    # Sort worst → best so the best sits at the top of the axis.
    _plot_df = tier_df.sort("mean_mae", descending=True)
    _labels = _plot_df["model_name"].to_list()
    _delta  = _plot_df["delta_vs_best"].to_numpy()
    _lo     = _plot_df["ci_lo"].to_numpy()
    _hi     = _plot_df["ci_hi"].to_numpy()
    _tiers  = _plot_df["tier"].to_list()

    _tier_color = {
        "Tier 1 (best)": "#2e7d32",
        "Tier 1":        "#4e79a7",
        "Outside Tier 1": "#e15759",
    }
    _y = np.arange(len(_labels))

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_t, _ax_t = plt.subplots(figsize=(9, max(4.5, 0.45 * len(_labels))), dpi=150)

        for _i in range(len(_labels)):
            _col = _tier_color[_tiers[_i]]
            # Error bar spans the 95% CI of Δ(model − best); point is the mean Δ.
            _ax_t.errorbar(
                _delta[_i], _y[_i],
                xerr=[[_delta[_i] - _lo[_i]], [_hi[_i] - _delta[_i]]],
                fmt="o", color=_col, ecolor=_col, elinewidth=1.6,
                capsize=3, markersize=6,
            )

        _ax_t.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        _ax_t.text(0.0, len(_labels) - 0.4, "  Δ = 0 (= best)", fontsize=9, va="top")

        _ax_t.set_yticks(_y)
        _ax_t.set_yticklabels(_labels, fontsize=8)
        _ax_t.set_xlabel("Δ MAE vs. best submission  (model − best); 95% bootstrap CI", fontsize=11)
        _ax_t.set_title(
            f"Tier 1 analysis — distance from best submission (`{TIER_BEST}`)\n"
            "Blue/green = Tier 1 (not sig. worse than best); red = outside Tier 1",
            fontsize=11,
        )

        # Legend via proxy handles.
        from matplotlib.lines import Line2D as _Line2D
        _handles = [
            _Line2D([0], [0], marker="o", color="w", markerfacecolor=_c, markersize=8, label=_lbl)
            for _lbl, _c in _tier_color.items()
        ]
        _ax_t.legend(handles=_handles, fontsize=9, frameon=True, framealpha=0.9, loc="lower right")
        _fig_t.tight_layout()
        _fig_t.savefig(PLOT_DIR / "tier1_forest_plot.png", dpi=300, bbox_inches="tight")

    mo.center(mo.as_html(_fig_t))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Predicted vs. observed scatter — best, final, and worst submissions

    Rather than plotting all ~18 submissions, we contrast three: the **best** local
    submission by MAE, our **final** submission, and the **worst** local submission.
    The dashed diagonal is the perfect-prediction line; points are coloured by
    absolute error (redder = larger mistake).
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    mean_absolute_error,
    metrics_df: "pl.DataFrame",
    mo,
    np,
    pl,
    plt,
    PLOT_DIR,
):
    # Choose the three submissions to contrast (all from our own, not external).
    _own = metrics_df.filter(~pl.col("is_external")).sort("MAE")
    _best_name  = _own.row(0, named=True)["model_name"]
    _final_name = _own.filter(pl.col("is_final")).row(0, named=True)["model_name"]
    _worst_name = _own.sort("MAE", descending=True).row(0, named=True)["model_name"]

    # Preserve order best → final → worst but de-duplicate if final == best/worst.
    _panel_specs = [("Best", _best_name), ("Final", _final_name), ("Worst", _worst_name)]
    _seen: set[str] = set()
    _panels = []
    for _tag, _name in _panel_specs:
        if _name not in _seen:
            _panels.append((_tag, _name))
            _seen.add(_name)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(1, len(_panels), figsize=(len(_panels) * 4.6, 4.6), dpi=130)
        _axes = np.atleast_1d(_axes)

        for _ax, (_tag, _name) in zip(_axes, _panels):
            _grp = all_predictions.filter(pl.col("model_name") == _name)
            _y_true = _grp.get_column("pEC50_true").to_numpy()
            _y_pred = _grp.get_column("pEC50_pred").to_numpy()
            _err    = np.abs(_y_pred - _y_true)
            _mae = mean_absolute_error(_y_true, _y_pred)
            _rmse = float(np.sqrt(np.mean((_y_pred - _y_true) ** 2)))

            _sc = _ax.scatter(
                _y_pred, _y_true,
                c=_err, cmap="RdYlGn_r", vmin=0, vmax=1.5,
                s=28, alpha=0.78, edgecolors="none",
            )
            _lims = [
                min(_y_true.min(), _y_pred.min()) - 0.2,
                max(_y_true.max(), _y_pred.max()) + 0.2,
            ]
            _ax.plot(_lims, _lims, "k--", linewidth=0.8, zorder=0)
            _ax.set_xlim(_lims)
            _ax.set_ylim(_lims)
            _ax.set_xlabel("Predicted pEC50", fontsize=9)
            _ax.set_ylabel("True pEC50", fontsize=9)
            _ax.set_title(f"{_tag}: {_name}\nMAE={_mae:.3f}  RMSE={_rmse:.3f}", fontsize=8, pad=4)
            plt.colorbar(_sc, ax=_ax, label="|error|", fraction=0.046, pad=0.04)

        _fig.suptitle("Predicted vs. true pEC50 — Phase 2 (best / final / worst)", fontsize=13, y=1.02)
        _fig.tight_layout()
        _fig.savefig(PLOT_DIR / "pred_vs_true_best_final_worst.png", dpi=200, bbox_inches="tight")

    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Where did the final submission win or lose vs. the best alternative?

    We line up the per-compound absolute error of the **final** submission against
    the **best** local submission.  Points below the diagonal are compounds the best
    submission predicted better than our final one; points above are where the final
    submission was better.  If the final was already the best, the two frames are the
    same model and this panel is skipped.
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    metrics_df: "pl.DataFrame",
    mo,
    nn_cliff_df: "pl.DataFrame",
    np,
    pl,
    plt,
    PLOT_DIR,
):
    _own = metrics_df.filter(~pl.col("is_external")).sort("MAE")
    _best_name  = _own.row(0, named=True)["model_name"]
    _final_name = _own.filter(pl.col("is_final")).row(0, named=True)["model_name"]

    if _best_name == _final_name:
        _head_delta = mo.md(
            "**Our final submission was already the best local submission — "
            "no better alternative exists to compare against.**"
        )
    else:
        _final_e = (
            all_predictions.filter(pl.col("model_name") == _final_name)
            .select(["Molecule Name", "abs_error"]).rename({"abs_error": "final_abs_err"})
        )
        _best_e = (
            all_predictions.filter(pl.col("model_name") == _best_name)
            .select(["Molecule Name", "abs_error"]).rename({"abs_error": "best_abs_err"})
        )
        # Attach pair_class so we can see whether cliffs drive the difference.
        _cmp = (
            _final_e.join(_best_e, on="Molecule Name", how="inner")
            .join(nn_cliff_df.select(["Molecule Name", "pair_class"]),
                  on="Molecule Name", how="left")
            .with_columns(
                (pl.col("final_abs_err") - pl.col("best_abs_err")).alias("delta_abs_err")
            )
        )

        _n_final_better = _cmp.filter(pl.col("delta_abs_err") < -1e-9).shape[0]
        _n_best_better  = _cmp.filter(pl.col("delta_abs_err") > 1e-9).shape[0]

        _cls_colors = {
            "Activity cliff": "#e15759",
            "Similar / concordant": "#76b7b2",
            "Dissimilar": "#b0b0b0",
        }

        with plt.style.context("seaborn-v0_8-whitegrid"):
            _fig_d, _ax_d = plt.subplots(figsize=(6.2, 6.0), dpi=130)
            for _cls, _col in _cls_colors.items():
                _s = _cmp.filter(pl.col("pair_class") == _cls)
                if _s.shape[0] == 0:
                    continue
                _ax_d.scatter(
                    _s["best_abs_err"].to_numpy(), _s["final_abs_err"].to_numpy(),
                    c=_col, s=34, alpha=0.8, edgecolors="none",
                    label=f"{_cls} (n={_s.shape[0]})",
                )
            _lim = float(max(_cmp["best_abs_err"].max(), _cmp["final_abs_err"].max())) + 0.1
            _ax_d.plot([0, _lim], [0, _lim], "k--", linewidth=0.9, zorder=0)
            _ax_d.set_xlabel(f"Best submission |error|\n({_best_name})", fontsize=9)
            _ax_d.set_ylabel(f"Final submission |error|\n({_final_name})", fontsize=9)
            _ax_d.set_title(
                "Per-compound error: final vs. best submission\n"
                f"Above diagonal = best wins ({_n_best_better}); "
                f"below = final wins ({_n_final_better})",
                fontsize=9,
            )
            _ax_d.set_xlim(-0.05, _lim)
            _ax_d.set_ylim(-0.05, _lim)
            _ax_d.legend(fontsize=8, frameon=True, framealpha=0.9)
            _fig_d.tight_layout()
            _fig_d.savefig(PLOT_DIR / "final_vs_best_per_compound.png", dpi=200, bbox_inches="tight")

        _head_delta = mo.center(mo.as_html(_fig_d))

    _head_delta
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Error concentration — do cliffs explain the misses?

    We split the Phase 2 compounds by their nearest-neighbour class (activity cliff /
    similar-concordant / dissimilar) and report the final submission's mean absolute
    error in each bucket.  If cliffs and dissimilar compounds dominate the error, that
    confirms the failures are driven by structural novelty rather than model tuning.
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    FINAL_NAME: str,
    mo,
    nn_cliff_df: "pl.DataFrame",
    pl,
):
    _final_label = FINAL_NAME.removesuffix("_submission.csv")

    _final_errs = (
        all_predictions.filter(pl.col("model_name") == _final_label)
        .select(["Molecule Name", "abs_error"])
        .join(nn_cliff_df.select(["Molecule Name", "pair_class", "nn_sim", "abs_delta_pEC50"]),
              on="Molecule Name", how="left")
    )

    _by_class = (
        _final_errs
        .group_by("pair_class")
        .agg(
            pl.len().alias("n"),
            pl.col("abs_error").mean().alias("mean_abs_error"),
            pl.col("abs_error").median().alias("median_abs_error"),
            (pl.col("abs_error") > 1.0).sum().alias("n_bad"),
        )
        .with_columns((pl.col("n_bad") / pl.col("n") * 100).round(1).alias("pct_bad"))
        .sort("mean_abs_error", descending=True)
    )

    _overall_mae = float(_final_errs["abs_error"].mean())

    _rows = "\n".join(
        f"| {r['pair_class']} | {r['n']} | {r['mean_abs_error']:.3f} | "
        f"{r['median_abs_error']:.3f} | {r['n_bad']} ({r['pct_bad']:.1f}%) |"
        for r in _by_class.iter_rows(named=True)
    )

    mo.md(f"""
    ### Final submission error by structural class

    Overall mean \\|error\\| on Phase 2: **{_overall_mae:.3f}** pEC50 units.

    | NN class | n | Mean \\|error\\| | Median \\|error\\| | \\|error\\| > 1 |
    |---|---|---|---|---|
    {_rows}

    Buckets with a higher mean \\|error\\| than the overall average are where the final
    model struggled most — typically activity cliffs and structurally dissimilar
    compounds, i.e. exactly the regions the cliff analysis above flagged as
    activity-sensitive or out-of-domain.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Systematic bias by activity bin

    For each submission we check whether predictions are systematically shifted within
    activity ranges.  A model that is accurate overall but biased in the hit region
    (pEC50 ≥ 6) is dangerous for virtual screening because it mis-ranks the compounds
    that matter most.
    """)
    return


@app.cell
def _(all_predictions: "pl.DataFrame", pl):
    # Bin compounds by true pEC50 into four activity ranges.
    binned: pl.DataFrame = all_predictions.with_columns(
        pl.when(pl.col("pEC50_true") >= 6.0)
        .then(pl.lit(">6 (hit zone)"))
        .when(pl.col("pEC50_true") >= 5.0)
        .then(pl.lit("5–6 (moderate)"))
        .when(pl.col("pEC50_true") >= 4.0)
        .then(pl.lit("4–5 (weak)"))
        .otherwise(pl.lit("<4 (inactive)"))
        .alias("pec50_bin")
    )

    bias_by_bin: pl.DataFrame = (
        binned
        .group_by(["model_name", "pec50_bin"])
        .agg(
            pl.col("error").mean().alias("mean_error"),
            pl.col("abs_error").mean().alias("mean_abs_error"),
            pl.len().alias("n"),
        )
        .sort(["model_name", "pec50_bin"])
    )

    bias_by_bin
    return (bias_by_bin,)


@app.cell
def _(bias_by_bin: "pl.DataFrame", mo, np, plt, PLOT_DIR):
    _bin_order = ["<4 (inactive)", "4–5 (weak)", "5–6 (moderate)", ">6 (hit zone)"]

    # Model × bin matrix of mean signed errors.
    _model_names = sorted(bias_by_bin.get_column("model_name").unique().to_list())
    _matrix = np.full((len(_model_names), len(_bin_order)), np.nan)
    _n_matrix = np.full((len(_model_names), len(_bin_order)), 0)

    for _row in bias_by_bin.iter_rows(named=True):
        _mi = _model_names.index(_row["model_name"])
        if _row["pec50_bin"] in _bin_order:
            _bi = _bin_order.index(_row["pec50_bin"])
            _matrix[_mi, _bi] = _row["mean_error"]
            _n_matrix[_mi, _bi] = _row["n"]

    _abs_max = float(np.nanmax(np.abs(_matrix)))

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_bias, _ax_bias = plt.subplots(figsize=(7.5, max(4.5, 0.42 * len(_model_names))), dpi=150)
        _ax_bias.grid(False)
        _im = _ax_bias.imshow(
            _matrix, cmap="RdBu_r", vmin=-_abs_max, vmax=_abs_max,
            aspect="auto", interpolation="nearest",
        )

        for _mi in range(len(_model_names)):
            for _bi in range(len(_bin_order)):
                _val = _matrix[_mi, _bi]
                _n = _n_matrix[_mi, _bi]
                if not np.isnan(_val):
                    _txt_col = "white" if abs(_val) > 0.6 * _abs_max else "black"
                    _ax_bias.text(_bi, _mi, f"{_val:+.2f}\nn={_n}",
                                  ha="center", va="center", fontsize=6.5, color=_txt_col)

        _ax_bias.set_xticks(range(len(_bin_order)))
        _ax_bias.set_xticklabels(_bin_order, fontsize=9, rotation=-20, ha="left")
        _ax_bias.set_yticks(range(len(_model_names)))
        _ax_bias.set_yticklabels(_model_names, fontsize=7)
        _ax_bias.set_xlabel("pEC50 bin", fontsize=11)
        _ax_bias.set_title(
            "Per-bin mean prediction error — all submissions (Phase 2)\n"
            "(red = overpredict, blue = underpredict)",
            fontsize=11,
        )
        _cb = _fig_bias.colorbar(_im, ax=_ax_bias, fraction=0.03, pad=0.03)
        _cb.set_label("Mean error (pred − true)", fontsize=10)
        _fig_bias.tight_layout()
        _fig_bias.savefig(PLOT_DIR / "bias_heatmap.png", dpi=300, bbox_inches="tight")

    mo.center(mo.as_html(_fig_bias))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Takeaways

    - **Distribution shift.** The three-way overlay and summary table show whether the
      Phase 2 test set probes the same potency range as the training data. Any shift
      toward weaker compounds or a smaller hit fraction predicts inflated test error.
    - **Activity cliffs.** The nearest-neighbour and MMP analyses quantify how much of
      the Phase 2 chemical space sits on activity cliffs relative to the training set —
      the structural regions where no amount of model tuning helps.
    - **Model ranking.** Every submission was re-scored against the final labels. The
      headline table states plainly whether our final entry
      (`6_ens_default_augfilt`) was the best choice, and if not, which submission would
      have beaten it and by how much.
    - **Error anatomy.** The per-class error breakdown and bias heatmap show *where* the
      errors concentrate — confirming (or refuting) that the misses are driven by
      structural novelty rather than systematic model bias.
    """)
    return


if __name__ == "__main__":
    app.run()

