import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 5 — Unblinded test set analysis

    This notebook analyses the newly available ground-truth labels for the Phase 1
    test set (`pxr-challenge_TEST_PHASE_1_UNBLINDED.csv`) to answer three questions:

    1. **Dataset profile** — how does the unblinded pEC50 distribution compare with
       the training set?  What fraction of test compounds would be classified as hits?
    2. **Model ranking** — how did each of our submitted predictions perform now that
       the true labels are revealed?  Metrics: RMSE, MAE, R², Pearson *r*, Spearman ρ.
    3. **Error anatomy** — which compounds were most / least accurately predicted, and
       are there structural patterns in the hard-to-predict cases?

    **Inputs:**
    - Unblinded CSV fetched from HuggingFace on first run and cached locally in
      `data/raw/20260528/`.
    - Submission CSVs in `submissions/` — every file ending in `_submission.csv`.
    - Processed training data: `data/processed/all_compounds_activity_data.csv`.
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
    return (
        Chem,
        Optional,
        Path,
        alt,
        gaussian_kde,
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
    # ── Molecule drawing helpers ─────────────────────────────────────────────────

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
    ## Load unblinded test set

    The CSV is fetched directly from HuggingFace and saved locally so subsequent
    runs are fast.  The file contains 253 test compounds with full dose-response
    parameters: pEC50, Emax, confidence intervals, and assay metadata.
    """)
    return


@app.cell
def _(Path, pl, smi_to_inchi, smi_to_inchikey):
    # Cache the unblinded CSV locally to avoid re-downloading each run.
    RAW_DIR = Path("../data/raw/20260528")
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    UNBLINDED_LOCAL = RAW_DIR / "dose_response_test_unblinded.csv"
    HF_URL = "hf://datasets/openadmet/pxr-challenge-train-test/pxr-challenge_TEST_PHASE_1_UNBLINDED.csv"

    if not UNBLINDED_LOCAL.exists():
        pl.read_csv(HF_URL).write_csv(UNBLINDED_LOCAL)

    unblinded_raw: pl.DataFrame = pl.read_csv(UNBLINDED_LOCAL)

    # Standardise column names and add chemoinformatics identifiers.
    unblinded: pl.DataFrame = (
        unblinded_raw
        .rename({"SMILES": "smiles"})
        .with_columns(
            pl.col("smiles").map_elements(smi_to_inchikey, return_dtype=pl.Utf8).alias("inchikey"),
            pl.col("smiles").map_elements(smi_to_inchi,    return_dtype=pl.Utf8).alias("inchi"),
        )
        # Shorten verbose column names for convenience inside this notebook.
        .rename({
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
        })
    )

    print("Shape:", unblinded.shape)
    print("Columns:", unblinded.columns)
    unblinded
    return (unblinded,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Dataset overview

    ### pEC50 distribution

    The dose-response pEC50 describes potency on a −log10(molarity) scale:
    higher values mean more potent compounds.  We compare the test set distribution
    with the training set to check whether the two sets probe the same activity range.

    A compound is typically classified as a **hit** when:
    - pEC50 ≥ 6 (IC₅₀ ≤ 1 µM), **and**
    - Emax above a baseline threshold (here Emax > 0 in log2FC units, meaning the
      compound causes at least a fold-change greater than the vehicle control).
    """)
    return


@app.cell
def _(gaussian_kde, mo, np, pl, plt, unblinded: "pl.DataFrame"):
    # Load training pEC50 values for comparison
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")

    train_pec50 = (
        all_compounds
        .filter(pl.col("in_dose_response") & pl.col("pEC50_dr").is_not_null())
        .get_column("pEC50_dr")
        .to_numpy()
    )
    test_pec50 = unblinded.get_column("pEC50").to_numpy()

    x_range = np.linspace(
        min(train_pec50.min(), test_pec50.min()) - 0.3,
        max(train_pec50.max(), test_pec50.max()) + 0.3,
        400,
    )

    kde_train = gaussian_kde(train_pec50, bw_method="scott")
    kde_test  = gaussian_kde(test_pec50,  bw_method="scott")

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig_dist, ax_dist = plt.subplots(figsize=(7, 4.5), dpi=150)
        ax_dist.plot(x_range, kde_train(x_range), color="#4e79a7", linewidth=2,
                     label=f"Training (n={len(train_pec50):,})")
        ax_dist.fill_between(x_range, kde_train(x_range), alpha=0.15, color="#4e79a7")
        ax_dist.plot(x_range, kde_test(x_range), color="#e15759", linewidth=2,
                     label=f"Test — unblinded (n={len(test_pec50):,})")
        ax_dist.fill_between(x_range, kde_test(x_range), alpha=0.15, color="#e15759")
        ax_dist.axvline(6.0, color="black", linestyle="--", linewidth=1.2,
                        label="Hit threshold (pEC50 = 6)")
        ax_dist.set_xlabel("pEC50  [−log₁₀(M)]", fontsize=12)
        ax_dist.set_ylabel("Density", fontsize=12)
        ax_dist.set_title("pEC50 distribution — training vs. unblinded test set", fontsize=13)
        ax_dist.legend(fontsize=11, frameon=True, framealpha=0.9)
        fig_dist.tight_layout()
        fig_dist.savefig("../plots/5_unblinded_analysis/pec50_distribution_train_test.png",
                         dpi=300, bbox_inches="tight")

    mo.center(mo.as_html(fig_dist))
    return (all_compounds,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Nearest-neighbour similarity to training set and activity cliffs

    Before evaluating model performance, we examine structural similarity between each
    unblinded test compound and its closest analogue in the dose-response training set,
    then ask whether structural similarity predicts activity similarity.

    An **activity cliff** is a pair of structurally similar compounds with substantially
    different potency — they are notoriously difficult for ML models because a small
    structural change produces a large activity change.

    Definitions used here:

    | Criterion | Threshold |
    |---|---|
    | Structurally similar | ECFP4 Tanimoto ≥ 0.4 |
    | Activity cliff | Similar **and** \|ΔpEC50\| ≥ 1.0 log unit |

    For each test compound we report its nearest neighbour (NN) in the training set,
    its Tanimoto similarity, and the activity difference between the two.
    """)
    return


@app.cell
def _(Chem, all_compounds, np, pl, unblinded: "pl.DataFrame"):
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

    _train_fps    = [_ecfp4(s) for s in _train_df["smiles"].to_list()]
    _test_fps     = [_ecfp4(s) for s in unblinded["smiles"].to_list()]
    _train_ids    = _train_df["inchikey"].to_list()
    _train_smiles = _train_df["smiles"].to_list()
    _train_names  = _train_df["molecule_names"].to_list()
    _train_pec50s = _train_df["pEC50_dr"].to_list()

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

    # Build the per-test-compound nearest-neighbour table.
    nn_cliff_df: pl.DataFrame = (
        unblinded
        .select(["Molecule Name", "smiles", "inchikey", "pEC50", "Emax"])
        .with_columns(
            pl.Series("nn_sim",         _max_sims,                                  dtype=pl.Float32),
            pl.Series("nn_inchikey",    [_train_ids[i]    for i in _nn_idx],        dtype=pl.Utf8),
            pl.Series("nn_smiles",      [_train_smiles[i] for i in _nn_idx],        dtype=pl.Utf8),
            pl.Series("nn_name",        [_train_names[i]  for i in _nn_idx],        dtype=pl.Utf8),
            pl.Series("nn_pEC50_train", [_train_pec50s[i] for i in _nn_idx],        dtype=pl.Float64),
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

    nn_cliff_df
    return (nn_cliff_df,)


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    mo,
    nn_cliff_df: "pl.DataFrame",
    np,
    pl,
    plt,
):
    # ── Static matplotlib scatter — coloured by ensemble residual error ──────────
    _ens_errors = (
        all_predictions
        .filter(pl.col("model_name") == "Ensemble (cp·ch·xg·mc·tf)")
        .select(["Molecule Name", "error"])
        .rename({"error": "ens_residual"})
    )

    _cliff_with_ens = nn_cliff_df.join(
        _ens_errors, on="Molecule Name", how="left",
    )

    _residuals = _cliff_with_ens["ens_residual"].to_numpy()
    _abs_max = max(abs(np.nanmin(_residuals)), abs(np.nanmax(_residuals)))

    # Marker style per pair_class so structural context is preserved.
    _marker_map = {
        "Activity cliff":       ("^", 50, 1.0),
        "Similar / concordant": ("o", 30, 0.80),
        "Dissimilar":           ("s", 25, 0.65),
    }

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_cliff, _ax_cliff = plt.subplots(figsize=(7.5, 5.5), dpi=150)

        for _cls, (_mkr, _sz, _alpha) in _marker_map.items():
            _sub = _cliff_with_ens.filter(pl.col("pair_class") == _cls)
            _sc = _ax_cliff.scatter(
                _sub["nn_sim"].to_numpy(),
                _sub["abs_delta_pEC50"].to_numpy(),
                c=_sub["ens_residual"].to_numpy(),
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
        _cb.set_label("Ensemble residual  (pred − true pEC50)", fontsize=10)

        _ax_cliff.axvline(0.4, color="#555", linestyle="--", linewidth=1, alpha=0.7,
                          label="Similarity threshold (0.4)")
        _ax_cliff.axhline(1.0, color="#555", linestyle=":",  linewidth=1, alpha=0.7,
                          label="|ΔpEC50| threshold (1.0)")

        _n_cliff_static = nn_cliff_df.filter(pl.col("pair_class") == "Activity cliff").shape[0]
        _ax_cliff.set_xlabel("ECFP4 Tanimoto similarity to nearest training neighbour", fontsize=11)
        _ax_cliff.set_ylabel("|ΔpEC50|  (test pEC50 − nearest train pEC50)", fontsize=11)
        _ax_cliff.set_title(
            f"Activity cliff analysis  —  {_n_cliff_static} cliff(s)\n"
            "Colour = ensemble residual (red = overpredicted, blue = underpredicted);\n"
            "markers: △ cliff, ○ similar/concordant, □ dissimilar",
            fontsize=10,
        )
        _ax_cliff.legend(fontsize=8, frameon=True, framealpha=0.9, loc="upper left")
        _ax_cliff.set_xlim(-0.02, 1.02)
        _fig_cliff.tight_layout()
        _fig_cliff.savefig(
            "../plots/5_unblinded_analysis/activity_cliffs_scatter.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_cliff))
    return


@app.cell
def _(mo, nn_cliff_df: "pl.DataFrame", pl):
    # ── Summary statistics ───────────────────────────────────────────────────────
    _n_total   = nn_cliff_df.shape[0]
    _n_similar = nn_cliff_df.filter(pl.col("nn_sim") >= 0.4).shape[0]
    _n_cliffs  = nn_cliff_df.filter(pl.col("pair_class") == "Activity cliff").shape[0]
    _n_concord = nn_cliff_df.filter(pl.col("pair_class") == "Similar / concordant").shape[0]
    _n_dissim  = nn_cliff_df.filter(pl.col("pair_class") == "Dissimilar").shape[0]
    _sim_mean  = float(nn_cliff_df["nn_sim"].mean())
    _sim_med   = float(nn_cliff_df["nn_sim"].median())
    _delt_mean = float(nn_cliff_df["abs_delta_pEC50"].mean())
    _delt_med  = float(nn_cliff_df["abs_delta_pEC50"].median())

    mo.md(f"""
    ### Activity cliff summary

    | Metric | Value |
    |---|---|
    | Test compounds analysed | {_n_total} |
    | Similar pairs (Tanimoto ≥ 0.4) | {_n_similar} ({100*_n_similar/_n_total:.1f}%) |
    | **Activity cliffs** (similar + \\|ΔpEC50\\| ≥ 1.0) | **{_n_cliffs} ({100*_n_cliffs/_n_total:.1f}%)** |
    | Similar / concordant | {_n_concord} ({100*_n_concord/_n_total:.1f}%) |
    | Dissimilar (Tanimoto < 0.4) | {_n_dissim} ({100*_n_dissim/_n_total:.1f}%) |
    | Mean Tanimoto sim to NN | {_sim_mean:.3f} |
    | Median Tanimoto sim to NN | {_sim_med:.3f} |
    | Mean \\|ΔpEC50\\| (all pairs) | {_delt_mean:.3f} |
    | Median \\|ΔpEC50\\| (all pairs) | {_delt_med:.3f} |
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Interactive browser — test compound and nearest training neighbour

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
    ## Matched molecular pair (MMP) analysis — training vs. unblinded test set

    Matched molecular pairs are compound pairs that differ by exactly one structural
    transformation at a single site while sharing a common scaffold (the **constant**
    fragment).  Because the change is chemically precise, MMP analysis lets us
    attribute an activity difference directly to a specific structural modification.

    The MMP database was pre-generated across the full compound collection.  Here we
    focus only on **cross-set pairs**: one compound from the dose-response training set
    and one from the unblinded test set.  This directly reveals how often structural
    similarity (at the MMP level) corresponds to activity similarity — and, crucially,
    how often it does not (activity cliffs).

    Activity cliff criterion: **|ΔpEC50| ≥ 1.0** log unit between the two MMP partners.
    """)
    return


@app.cell
def _(all_compounds, pl, unblinded: "pl.DataFrame"):
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
    _test_act = unblinded.select(["Molecule Name", "smiles", "inchikey", "pEC50"])

    _train_iks = set(_train_act["inchikey"].to_list())
    _test_iks  = set(_test_act["inchikey"].to_list())

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

    mmp_cross_df: pl.DataFrame = (
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

    # Cluster-level summary: one row per unique scaffold constant.
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
    return mmp_cluster_df, mmp_cross_df


@app.cell
def _(
    gaussian_kde,
    mmp_cluster_df: "pl.DataFrame",
    mmp_cross_df: "pl.DataFrame",
    mo,
    np,
    pl,
    plt,
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
        _ax_kde.set_title("|ΔpEC50| distribution — all cross-set MMPs", fontsize=11)

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
            "../plots/5_unblinded_analysis/mmp_activity_cliffs.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_mmp))
    return


@app.cell
def _(mmp_cluster_df: "pl.DataFrame", mmp_cross_df: "pl.DataFrame", mo, pl):
    _n_total   = mmp_cross_df.shape[0]
    _n_cliffs  = mmp_cross_df.filter(pl.col("pair_class") == "Activity cliff").shape[0]
    _n_concord = mmp_cross_df.filter(pl.col("pair_class") == "Concordant").shape[0]
    _n_clusters       = mmp_cluster_df.shape[0]
    _n_cliff_clusters = mmp_cluster_df.filter(pl.col("n_cliffs") >= 1).shape[0]

    _q25 = float(mmp_cross_df["abs_delta_pEC50"].quantile(0.25))
    _q50 = float(mmp_cross_df["abs_delta_pEC50"].quantile(0.50))
    _q75 = float(mmp_cross_df["abs_delta_pEC50"].quantile(0.75))
    _q90 = float(mmp_cross_df["abs_delta_pEC50"].quantile(0.90))
    _q95 = float(mmp_cross_df["abs_delta_pEC50"].quantile(0.95))
    _max_delta = float(mmp_cross_df["abs_delta_pEC50"].max())

    mo.md(f"""
    ### MMP summary

    | Metric | Value |
    |---|---|
    | Total cross-set MMP pairs | {_n_total} |
    | **Activity cliffs** (\\|ΔpEC50\\| ≥ 1.0) | **{_n_cliffs} ({100*_n_cliffs/_n_total:.1f}%)** |
    | Concordant pairs | {_n_concord} ({100*_n_concord/_n_total:.1f}%) |
    | Unique scaffold clusters | {_n_clusters} |
    | Clusters with ≥ 1 cliff | {_n_cliff_clusters} ({100*_n_cliff_clusters/_n_clusters:.1f}%) |
    | Median \\|ΔpEC50\\| | {_q50:.2f} |
    | 75th percentile \\|ΔpEC50\\| | {_q75:.2f} |
    | 90th percentile \\|ΔpEC50\\| | {_q90:.2f} |
    | 95th percentile \\|ΔpEC50\\| | {_q95:.2f} |
    | Maximum \\|ΔpEC50\\| | {_max_delta:.2f} |

    **Interpretation:** {100*_n_cliffs/_n_total:.0f}% of MMP pairs are activity cliffs — the structural
    change between each MMP partner leads to a greater than 1 log-unit shift in
    potency.  This is a very high cliff rate and suggests the test compounds probe
    activity-sensitive structural space that the training data does not resolve well.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model evaluation

    ### Load submissions

    Each ensemble component is sourced from the best available file.
    Original submission CSVs are preferred; where none exists the predictions
    regenerated by `scripts/5_regenerate_predictions.py` are used instead.

    | File | Source | Role | Ensemble weight |
    |------|--------|------|-----------------|
    | `submissions/4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv` | original | **Ensemble** | — |
    | `submissions/4_chemeleon_hpo_submission.csv` | original | CheMeleon HPO (`ch`) | 5 |
    | `submissions/4_tabpfn_chemeleon_submission.csv` | original | TabPFN / CheMeleon (`tf`) | 5 |
    | `submissions/4_macau_che_hpo_submission.csv` | original | Macau HPO / CheMeleon (`mc`) | 1 |
    | `predictions/5_regen_cp_preds.csv` | regenerated | Chemprop HPO (`cp`) | 5 |
    | `predictions/5_regen_xg_preds.csv` | regenerated | XGBoost / Mordred (`xg`) | ⅓ |

    Macau and TabPFN are deterministic (seeded) so their regenerated predictions
    are bit-for-bit identical to the originals; the originals are used here.
    Chemprop and XGBoost are stochastic — only regenerated files are available.
    """)
    return


@app.cell
def _(Path, pl, unblinded: "pl.DataFrame"):
    SUBMISSION_DIR = Path("../submissions")
    PRED_DIR       = Path("../predictions")

    # Each entry: (path, pred_col, label)
    # pred_col is the column holding predictions — "pEC50" for submission CSVs,
    # "pEC50_<tag>" for the regenerated component CSVs.
    SOURCES: list[tuple[Path, str, str]] = [
        (SUBMISSION_DIR / "4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv",
         "pEC50",    "Ensemble (cp·ch·xg·mc·tf)"),
        (SUBMISSION_DIR / "4_chemeleon_hpo_submission.csv",
         "pEC50",    "CheMeleon HPO (ch)"),
        (SUBMISSION_DIR / "4_tabpfn_chemeleon_submission.csv",
         "pEC50",    "TabPFN / CheMeleon (tf)"),
        (SUBMISSION_DIR / "4_macau_che_hpo_submission.csv",
         "pEC50",    "Macau HPO / CheMeleon (mc)"),
        (PRED_DIR / "5_regen_cp_preds.csv",
         "pEC50_cp", "Chemprop HPO (cp) [regen]"),
        (PRED_DIR / "5_regen_xg_preds.csv",
         "pEC50_xg", "XGBoost / Mordred (xg) [regen]"),
    ]

    # Build a truth table: Molecule Name → true pEC50
    truth: pl.DataFrame = unblinded.select(["Molecule Name", "pEC50"]).rename(
        {"pEC50": "pEC50_true"}
    )

    def load_submission(path: Path, pred_col: str, label: str) -> pl.DataFrame:
        """
        Load a prediction CSV and join with the ground-truth pEC50.

        Args:
            path: Path to the CSV file.
            pred_col: Column name that holds the predicted pEC50 values.
            label: Human-readable model name attached to every row.

        Returns:
            DataFrame with columns:
              model_name, Molecule Name, pEC50_pred, pEC50_true, error, abs_error.
        """
        df = (
            pl.read_csv(path)
            .rename({pred_col: "pEC50_pred"})
            .select(["Molecule Name", "pEC50_pred"])
        )
        return (
            df.join(truth, on="Molecule Name", how="inner")
            .with_columns(
                pl.lit(label).alias("model_name"),
                (pl.col("pEC50_pred") - pl.col("pEC50_true")).alias("error"),
                (pl.col("pEC50_pred") - pl.col("pEC50_true")).abs().alias("abs_error"),
            )
        )

    all_predictions: pl.DataFrame = pl.concat(
        [load_submission(path, pred_col, label) for path, pred_col, label in SOURCES]
    )

    print(f"Loaded {len(SOURCES)} sources, "
          f"{all_predictions.shape[0]} total rows")
    all_predictions
    return (all_predictions,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Aggregate metrics per model

    Five standard regression metrics are computed for each submission:

    | Metric | Description |
    |---|---|
    | **RMSE** | Root mean squared error (lower = better) |
    | **MAE** | Mean absolute error (lower = better) |
    | **R²** | Coefficient of determination (higher = better) |
    | **Pearson r** | Linear correlation between predicted and true pEC50 |
    | **Spearman ρ** | Rank correlation (robust to outliers) |

    The table is sorted by MAE ascending so the best-performing model appears first.
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
        """Compute regression metrics for one model's predictions."""
        y_true = group.get_column("pEC50_true").to_numpy()
        y_pred = group.get_column("pEC50_pred").to_numpy()
        rmse = math.sqrt(mean_squared_error(y_true, y_pred))
        mae  = mean_absolute_error(y_true, y_pred)
        r2   = r2_score(y_true, y_pred)
        r, _ = pearsonr(y_true, y_pred)
        rho, _ = spearmanr(y_true, y_pred)
        n    = len(y_true)
        bias = float(np.mean(y_pred - y_true))
        return {
            "model_name": group.get_column("model_name")[0],
            "n":          n,
            "RMSE":       round(rmse, 4),
            "MAE":        round(mae,  4),
            "R2":         round(r2,   4),
            "Pearson_r":  round(r,    4),
            "Spearman_rho": round(rho, 4),
            "bias":       round(bias, 4),
        }

    metrics_rows = [
        compute_metrics(grp)
        for grp in all_predictions.partition_by("model_name", maintain_order=True)
    ]

    metrics_df: pl.DataFrame = (
        pl.DataFrame(metrics_rows)
        .sort("MAE")
    )

    metrics_df
    return (metrics_df,)


@app.cell
def _(alt, metrics_df: "pl.DataFrame", mo):
    _bar = (
        alt.Chart(metrics_df)
        .mark_bar()
        .encode(
            x=alt.X("MAE:Q", title="MAE  (lower = better)"),
            y=alt.Y("model_name:N", sort=alt.SortField("MAE", order="ascending"),
                    title=None, axis=alt.Axis(labelLimit=300, labelFontSize=10)),
            color=alt.Color("MAE:Q", scale=alt.Scale(scheme="redyellowgreen",
                                                      reverse=True), legend=None),
            tooltip=[
                alt.Tooltip("model_name:N", title="Model"),
                alt.Tooltip("MAE:Q",         title="MAE",         format=".4f"),
                alt.Tooltip("RMSE:Q",        title="RMSE",        format=".4f"),
                alt.Tooltip("R2:Q",          title="R²",          format=".4f"),
                alt.Tooltip("Pearson_r:Q",   title="Pearson r",   format=".4f"),
                alt.Tooltip("Spearman_rho:Q",title="Spearman ρ", format=".4f"),
                alt.Tooltip("bias:Q",        title="Bias",        format="+.4f"),
            ],
        )
        .properties(
            title="Model ranking by MAE on unblinded test set",
            width=480,
            height=320,
        )
        .configure_title(fontSize=13)
    )

    mo.ui.altair_chart(_bar)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Predicted vs. observed scatter per model

    Each panel shows the predicted pEC50 on the x-axis and the true pEC50 on the
    y-axis.  The dashed diagonal is the perfect-prediction line.  Points are
    coloured by absolute error — the redder the point, the larger the mistake.
    Panels are sorted by MAE (best model top-left).
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    math,
    mean_absolute_error,
    metrics_df: "pl.DataFrame",
    mo,
    np,
    plt,
):
    _model_order = metrics_df.get_column("model_name").to_list()
    _n_models    = len(_model_order)
    _ncols       = 3
    _nrows       = math.ceil(_n_models / _ncols)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(
            _nrows, _ncols,
            figsize=(_ncols * 4.5, _nrows * 4.5),
            dpi=120,
        )
        _axes_flat = _axes.flatten() if _n_models > 1 else [_axes]

        for _i, _model in enumerate(_model_order):
            _ax = _axes_flat[_i]
            _grp = all_predictions.filter(
                all_predictions["model_name"] == _model
            )
            _y_true = _grp.get_column("pEC50_true").to_numpy()
            _y_pred = _grp.get_column("pEC50_pred").to_numpy()
            _err    = np.abs(_y_pred - _y_true)

            _mae = mean_absolute_error(_y_true, _y_pred)

            _sc = _ax.scatter(
                _y_pred, _y_true,
                c=_err, cmap="RdYlGn_r", vmin=0, vmax=1.5,
                s=25, alpha=0.75, edgecolors="none",
            )
            _lims = [
                min(_y_true.min(), _y_pred.min()) - 0.2,
                max(_y_true.max(), _y_pred.max()) + 0.2,
            ]
            _ax.plot(_lims, _lims, "k--", linewidth=0.8, zorder=0)
            _ax.set_xlim(_lims)
            _ax.set_ylim(_lims)
            _ax.set_xlabel("Predicted pEC50", fontsize=9)
            _ax.set_ylabel("True pEC50",      fontsize=9)
            _ax.set_title(f"{_model}\nMAE={_mae:.3f}", fontsize=8, pad=4)
            plt.colorbar(_sc, ax=_ax, label="|error|", fraction=0.046, pad=0.04)

        # Hide any unused axes.
        for _j in range(_i + 1, len(_axes_flat)):
            _axes_flat[_j].set_visible(False)

        _fig.suptitle("Predicted vs. true pEC50 — all submissions", fontsize=14, y=1.01)
        _fig.tight_layout()
        _fig.savefig(
            "../plots/5_unblinded_analysis/pred_vs_true_all_models.png",
            dpi=200, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Residual error vs. true pEC50

    Each panel shows the signed residual (predicted − true pEC50) on the y-axis
    against the true pEC50 on the x-axis.  The dashed horizontal line at zero is
    the ideal residual.  Points above the line are overpredicted; points below are
    underpredicted.  Colour encodes the absolute error so large deviations are
    immediately visible regardless of sign.
    """)
    return


@app.cell
def _(
    all_predictions: "pl.DataFrame",
    math,
    mean_absolute_error,
    metrics_df: "pl.DataFrame",
    mo,
    np,
    plt,
):
    _model_order_res = metrics_df.get_column("model_name").to_list()
    _n_models_res    = len(_model_order_res)
    _ncols_res       = 3
    _nrows_res       = math.ceil(_n_models_res / _ncols_res)

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(
            _nrows_res, _ncols_res,
            figsize=(_ncols_res * 4.5, _nrows_res * 3.5),
            dpi=120,
            sharey=False,
        )
        _axes_flat = _axes.flatten() if _n_models_res > 1 else [_axes]

        # Shared colour scale across all panels — consistent visual reference.
        _vmax = 1.5

        for _i, _model in enumerate(_model_order_res):
            _ax = _axes_flat[_i]
            _grp = all_predictions.filter(all_predictions["model_name"] == _model)
            _y_true = _grp.get_column("pEC50_true").to_numpy()
            _y_pred = _grp.get_column("pEC50_pred").to_numpy()
            _resid  = _y_pred - _y_true
            _abs_err = np.abs(_resid)
            _mae = mean_absolute_error(_y_true, _y_pred)

            _sc = _ax.scatter(
                _y_true, _resid,
                c=_abs_err, cmap="RdYlGn_r", vmin=0, vmax=_vmax,
                s=25, alpha=0.8, edgecolors="none",
            )
            _ax.axhline(0, color="black", linestyle="--", linewidth=0.9, zorder=0)
            _ax.set_xlabel("True pEC50", fontsize=9)
            _ax.set_ylabel("Residual (pred − true)", fontsize=9)
            _ax.set_title(f"{_model}\nMAE={_mae:.3f}", fontsize=8, pad=4)
            plt.colorbar(_sc, ax=_ax, label="|error|", fraction=0.046, pad=0.04)

        for _j in range(_i + 1, len(_axes_flat)):
            _axes_flat[_j].set_visible(False)

        _fig.suptitle("Residual error vs. true pEC50 — all submissions", fontsize=14, y=1.01)
        _fig.tight_layout()
        _fig.savefig(
            "../plots/5_unblinded_analysis/residuals_all_models.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Error analysis

    ### Hardest compounds to predict

    We identify compounds for which the **best-performing** model still produced
    a large absolute error.  For each compound we take the minimum absolute error
    across all submissions, then rank by that value.  Compounds that were hard for
    every model likely represent chemical or mechanistic novelty.
    """)
    return


@app.cell
def _(all_predictions: "pl.DataFrame", pl, unblinded: "pl.DataFrame"):
    # Per-compound: minimum absolute error across all models, and which model achieved it.
    compound_min_err: pl.DataFrame = (
        all_predictions
        .group_by("Molecule Name")
        .agg(
            pl.col("abs_error").min().alias("min_abs_error"),
            pl.col("abs_error").mean().alias("mean_abs_error"),
            pl.col("abs_error").std().alias("std_abs_error"),
            pl.col("pEC50_true").first().alias("pEC50_true"),
        )
        # Join SMILES and InChIKey from the unblinded dataset.
        .join(
            unblinded.select(["Molecule Name", "smiles", "inchikey", "Emax"]),
            on="Molecule Name",
            how="left",
        )
        .sort("mean_abs_error", descending=True)
    )

    print("Top 10 hardest compounds (largest mean absolute error across models):")
    compound_min_err.head(10)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### Model disagreement on large errors (|error| > 1 log unit)

    We now inspect **badly-predicted compounds** — those where at least one model's
    absolute error exceeds 1 pEC50 unit.  Three questions guide this analysis:

    1. **How many** compounds are badly predicted by each model?  Does the ensemble
       have fewer failures than its individual components?
    2. **Do models fail on the same compounds or on different ones?**  The Jaccard
       similarity of each pair of "bad-prediction sets" answers this directly.
    3. **Does the ensemble rescue individual mistakes?**  We classify every compound
       as *rescued*, *still bad*, *only ensemble bad*, or *good everywhere*.
    """)
    return


@app.cell
def _(all_predictions: "pl.DataFrame", pl):
    # ── Threshold and display constants ──────────────────────────────────────────
    BAD_THRESH = 1.0
    ENSEMBLE_NAME = "Ensemble (cp·ch·xg·mc·tf)"

    # Short axis labels used in heatmaps and bar charts
    SHORT_MODEL_NAMES: dict[str, str] = {
        "Ensemble (cp·ch·xg·mc·tf)":      "Ensemble",
        "CheMeleon HPO (ch)":              "CheMeleon\n(ch)",
        "TabPFN / CheMeleon (tf)":         "TabPFN\n(tf)",
        "Macau HPO / CheMeleon (mc)":      "Macau\n(mc)",
        "Chemprop HPO (cp) [regen]":       "Chemprop\n(cp)",
        "XGBoost / Mordred (xg) [regen]":  "XGBoost\n(xg)",
    }

    # Compounds where at least one model has abs_error > BAD_THRESH
    _bad_names = (
        all_predictions
        .filter(pl.col("abs_error") > BAD_THRESH)
        .get_column("Molecule Name")
        .unique()
    )

    # Long-format predictions restricted to these compounds (all models included)
    bad_preds_long: pl.DataFrame = all_predictions.filter(
        pl.col("Molecule Name").is_in(_bad_names)
    )

    # Wide format: Molecule Name × model_name → abs_error value
    error_wide: pl.DataFrame = bad_preds_long.pivot(
        values="abs_error",
        index="Molecule Name",
        on="model_name",
        aggregate_function="first",
    )

    # Count how many models flag each compound as bad
    _model_cols = [c for c in error_wide.columns if c != "Molecule Name"]
    error_wide = error_wide.with_columns(
        pl.sum_horizontal([
            (pl.col(m) > BAD_THRESH).cast(pl.Int32) for m in _model_cols
        ]).alias("n_models_bad")
    ).sort(["n_models_bad", "Molecule Name"], descending=[True, False])

    # Per-model tally
    per_model_bad: pl.DataFrame = (
        all_predictions
        .with_columns((pl.col("abs_error") > BAD_THRESH).alias("is_bad"))
        .group_by("model_name")
        .agg(
            pl.col("is_bad").sum().alias("n_bad"),
            pl.len().alias("n_total"),
        )
        .with_columns(
            (pl.col("n_bad") / pl.col("n_total") * 100).round(1).alias("pct_bad")
        )
        .sort("n_bad", descending=True)
    )

    print(f"Compounds with |error| > {BAD_THRESH} in ≥1 model: {_bad_names.len()}")
    per_model_bad
    return BAD_THRESH, ENSEMBLE_NAME


@app.cell
def _(
    BAD_THRESH,
    ENSEMBLE_NAME,
    all_predictions: "pl.DataFrame",
    mo,
    pl,
    unblinded: "pl.DataFrame",
):
    # ── Rescue analysis: compare ensemble against individual component models ─────
    _ens_preds = (
        all_predictions
        .filter(pl.col("model_name") == ENSEMBLE_NAME)
        .select(["Molecule Name", "pEC50_true", "pEC50_pred", "abs_error"])
        .rename({"abs_error": "ens_abs_error", "pEC50_pred": "ens_pred"})
    )

    _ind_stats = (
        all_predictions
        .filter(pl.col("model_name") != ENSEMBLE_NAME)
        .group_by("Molecule Name")
        .agg(
            (pl.col("abs_error") > BAD_THRESH).sum().alias("n_ind_bad"),
            pl.col("abs_error").mean().alias("mean_ind_abs_error"),
            pl.len().alias("n_ind_models"),
        )
    )

    rescue_df: pl.DataFrame = (
        _ens_preds
        .join(_ind_stats, on="Molecule Name", how="inner")
        .join(unblinded.select(["Molecule Name", "smiles", "inchikey"]),
              on="Molecule Name", how="left")
        .with_columns(
            pl.when(
                (pl.col("n_ind_bad") >= 1) & (pl.col("ens_abs_error") <= BAD_THRESH)
            ).then(pl.lit("Rescued by ensemble"))
            .when(
                (pl.col("n_ind_bad") >= 1) & (pl.col("ens_abs_error") > BAD_THRESH)
            ).then(pl.lit("Still bad in ensemble"))
            .when(
                (pl.col("n_ind_bad") == 0) & (pl.col("ens_abs_error") > BAD_THRESH)
            ).then(pl.lit("Only ensemble bad"))
            .otherwise(pl.lit("Good everywhere"))
            .alias("rescue_status")
        )
    )

    _n_rescued   = rescue_df.filter(pl.col("rescue_status") == "Rescued by ensemble").shape[0]
    _n_still_bad = rescue_df.filter(pl.col("rescue_status") == "Still bad in ensemble").shape[0]
    _n_ens_only  = rescue_df.filter(pl.col("rescue_status") == "Only ensemble bad").shape[0]
    _n_good      = rescue_df.filter(pl.col("rescue_status") == "Good everywhere").shape[0]
    _n_total     = rescue_df.shape[0]
    _n_ind       = int(rescue_df["n_ind_models"].max())

    mo.md(f"""
    ### Ensemble rescue analysis

    For each test compound we compare the ensemble against the {_n_ind} individual component
    models and assign one of four outcomes:

    | Status | n | % of test set |
    |---|---|---|
    | **Rescued by ensemble** — bad in ≥1 component, good in ensemble | **{_n_rescued}** | **{100*_n_rescued/_n_total:.1f}%** |
    | **Still bad in ensemble** — bad in ≥1 component AND bad in ensemble | {_n_still_bad} | {100*_n_still_bad/_n_total:.1f}% |
    | Only ensemble bad — all components fine, ensemble errs | {_n_ens_only} | {100*_n_ens_only/_n_total:.1f}% |
    | Good everywhere — all models predict well | {_n_good} | {100*_n_good/_n_total:.1f}% |

    **{_n_rescued} compounds** ({100*_n_rescued/_n_total:.0f}% of the test set) were corrected by
    averaging — at least one component model failed but the ensemble brought the error
    below {BAD_THRESH:.0f} pEC50 unit.  The **{_n_still_bad} "still bad"** compounds represent
    structural blind spots that no ensemble weighting can compensate for.
    """)
    return (rescue_df,)


@app.cell
def _(BAD_THRESH, mo, pl, plt, rescue_df: "pl.DataFrame"):
    # ── Scatter: ensemble error vs mean individual-model error ────────────────────
    _status_colors = {
        "Rescued by ensemble":   "#76b7b2",
        "Still bad in ensemble": "#e15759",
        "Only ensemble bad":     "#f28e2b",
        "Good everywhere":       "#b0b0b0",
    }

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_r, _ax_r = plt.subplots(figsize=(6.5, 5.5), dpi=130)

        for _status, _col in _status_colors.items():
            _sub = rescue_df.filter(pl.col("rescue_status") == _status)
            if _sub.shape[0] == 0:
                continue
            _ax_r.scatter(
                _sub["mean_ind_abs_error"].to_numpy(),
                _sub["ens_abs_error"].to_numpy(),
                c=_col, label=f"{_status}  (n={_sub.shape[0]})",
                s=40, alpha=0.85, edgecolors="none",
            )

        _lim = float(max(
            rescue_df["mean_ind_abs_error"].max(),
            rescue_df["ens_abs_error"].max(),
        )) + 0.15
        _ax_r.plot([0, _lim], [0, _lim], "k--", linewidth=0.9, zorder=0,
                   label="y = x  (ensemble = mean component)")
        _ax_r.axhline(BAD_THRESH, color="grey", linewidth=0.8, linestyle=":", zorder=0)
        _ax_r.axvline(BAD_THRESH, color="grey", linewidth=0.8, linestyle=":", zorder=0)
        _ax_r.text(BAD_THRESH + 0.03, 0.05, f"{BAD_THRESH}", color="grey", fontsize=8)
        _ax_r.text(0.05, BAD_THRESH + 0.04, f"{BAD_THRESH}", color="grey", fontsize=8)

        _ax_r.set_xlabel("Mean |error| across individual component models", fontsize=11)
        _ax_r.set_ylabel("Ensemble |error|", fontsize=11)
        _ax_r.set_title(
            "Ensemble vs. component model error per compound\n"
            "Below diagonal = ensemble outperforms mean component; "
            "dotted lines = |error| = 1 threshold",
            fontsize=9,
        )
        _ax_r.legend(fontsize=8.5, frameon=True, framealpha=0.9)
        _ax_r.set_xlim(-0.05, _lim)
        _ax_r.set_ylim(-0.05, _lim)
        _fig_r.tight_layout()
        _fig_r.savefig(
            "../plots/5_unblinded_analysis/ensemble_rescue_scatter.png",
            dpi=200, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_r))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ### UpSet plot — overlap in compounds predicted badly

    A Venn diagram becomes unwieldy with six models.  The UpSet plot below
    replaces it: each column in the intersection matrix represents a model,
    and each row is an intersection of models that *all* flagged a compound
    as badly predicted (|error| > 1 pEC50 unit).  The bar heights show
    how many compounds fall into each intersection — revealing, for example,
    how many compounds are uniquely missed by a single model versus missed
    by every model simultaneously.
    """)
    return


@app.cell
def _(BAD_THRESH, all_predictions: "pl.DataFrame", mo, pl, plt):
    from upsetplot import from_memberships

    _short_names: dict[str, str] = {
        "Ensemble (cp·ch·xg·mc·tf)":      "Ensemble",
        "CheMeleon HPO (ch)":              "CheMeleon",
        "TabPFN / CheMeleon (tf)":         "TabPFN",
        "Macau HPO / CheMeleon (mc)":      "Macau",
        "Chemprop HPO (cp) [regen]":       "Chemprop",
        "XGBoost / Mordred (xg) [regen]":  "XGBoost",
    }

    # For each compound, collect the set of models that predicted it badly.
    _bad = (
        all_predictions
        .filter(pl.col("abs_error") > BAD_THRESH)
        .with_columns(
            pl.col("model_name").replace(_short_names).alias("model_short"),
        )
        .group_by("Molecule Name")
        .agg(pl.col("model_short").sort())
    )

    # from_memberships expects a list of lists (category memberships per observation).
    _memberships: list[list[str]] = _bad.get_column("model_short").to_list()

    # from_memberships returns a Series with a boolean MultiIndex that may have
    # duplicate rows.  Collapse to unique groups with counts so UpSet is happy.
    _raw = from_memberships(_memberships)
    _upset_data = _raw.groupby(level=_raw.index.names).size()

    _upset = __import__("upsetplot").UpSet(
        _upset_data,
        show_counts=True,
        sort_by="cardinality",
        sort_categories_by="cardinality",
        min_subset_size=1,
        facecolor="#4e79a7",
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig_upset = plt.figure(figsize=(10, 5.5), dpi=150)
        _upset.plot(fig=_fig_upset)
        _fig_upset.suptitle(
            f"Model disagreement — compounds with |error| > {BAD_THRESH} pEC50 unit",
            fontsize=12, y=1.02,
        )
        _fig_upset.savefig(
            "../plots/5_unblinded_analysis/upset_bad_predictions.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_upset))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Systematic bias analysis

    For each model we look at whether predictions are systematically shifted
    (biased high or low) within different activity bins.  A model that performs
    well overall but is biased in the hit region is particularly dangerous for
    virtual screening.
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
def _(bias_by_bin: "pl.DataFrame", mo, np, plt):
    _bin_order = ["<4 (inactive)", "4–5 (weak)",
                  "5–6 (moderate)", ">6 (hit zone)"]

    _short = {
        "Ensemble (cp·ch·xg·mc·tf)":      "Ensemble",
        "CheMeleon HPO (ch)":              "CheMeleon (ch)",
        "TabPFN / CheMeleon (tf)":         "TabPFN (tf)",
        "Macau HPO / CheMeleon (mc)":      "Macau (mc)",
        "Chemprop HPO (cp) [regen]":       "Chemprop (cp)",
        "XGBoost / Mordred (xg) [regen]":  "XGBoost (xg)",
    }

    # Pivot to a model × bin matrix of mean errors.
    _model_names = sorted(bias_by_bin.get_column("model_name").unique().to_list(),
                          key=lambda m: _short.get(m, m))
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
        _fig_bias, _ax_bias = plt.subplots(figsize=(7, 4.5), dpi=150)
        _ax_bias.grid(False)
        _im = _ax_bias.imshow(
            _matrix, cmap="RdBu_r", vmin=-_abs_max, vmax=_abs_max,
            aspect="auto", interpolation="nearest",
        )

        # Annotate each cell with the mean error value and sample size.
        for _mi in range(len(_model_names)):
            for _bi in range(len(_bin_order)):
                _val = _matrix[_mi, _bi]
                _n = _n_matrix[_mi, _bi]
                if not np.isnan(_val):
                    _txt_col = "white" if abs(_val) > 0.6 * _abs_max else "black"
                    _ax_bias.text(_bi, _mi, f"{_val:+.3f}\nn={_n}",
                                  ha="center", va="center", fontsize=8, color=_txt_col)

        _ax_bias.set_xticks(range(len(_bin_order)))
        _ax_bias.set_xticklabels(_bin_order, fontsize=10, rotation=-20, ha="left")
        _ax_bias.set_yticks(range(len(_model_names)))
        _ax_bias.set_yticklabels([_short.get(m, m) for m in _model_names], fontsize=10)
        _ax_bias.set_xlabel("pEC50 bin", fontsize=12)
        _ax_bias.set_title(
            "Per-bin mean prediction error\n(red = overpredict, blue = underpredict)",
            fontsize=12,
        )
        _cb = _fig_bias.colorbar(_im, ax=_ax_bias, fraction=0.04, pad=0.03)
        _cb.set_label("Mean error (pred − true)", fontsize=10)
        _fig_bias.tight_layout()
        _fig_bias.savefig(
            "../plots/5_unblinded_analysis/bias_heatmap.png",
            dpi=300, bbox_inches="tight",
        )

    mo.center(mo.as_html(_fig_bias))
    return


if __name__ == "__main__":
    app.run()
