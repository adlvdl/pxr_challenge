import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 1d — Train / test set exploration

    Compares the chemical space of the dose-response training set and the blinded
    test set to understand:

    - How structurally similar the two sets are (pairwise Tanimoto KDE curves).
    - How much of the test set is covered by the training data (nearest-neighbour
      maximum similarity per test compound).
    - Where test compounds fall in UMAP space relative to their closest training
      analogue (interactive scatter with on-click structure panel).

    **Input:** `data/processed/all_compounds_activity_data.csv` (produced by **1a**).
    The raw test CSV is also read to access the blinded test molecules.
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    import numpy as np
    from pathlib import Path
    from typing import Optional, Callable

    import matplotlib.pyplot as plt
    import matplotlib.figure
    from scipy.stats import gaussian_kde

    from umap import UMAP

    from skfp.fingerprints import (
        ECFPFingerprint,
        MACCSFingerprint,
        TopologicalTorsionFingerprint,
        RDKitFingerprint,
        AtomPairFingerprint,
        AvalonFingerprint,
        MordredFingerprint,
        MQNsFingerprint,
        PubChemFingerprint,
    )
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D

    RDLogger.DisableLog("rdApp.*")
    return (
        AtomPairFingerprint,
        AvalonFingerprint,
        Callable,
        Chem,
        ConformerGenerator,
        DataStructs,
        ECFPFingerprint,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        Optional,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        TopologicalTorsionFingerprint,
        UMAP,
        alt,
        gaussian_kde,
        mo,
        np,
        pl,
        plt,
        rdDepictor,
        rdMolDraw2D,
    )


@app.cell
def _(
    AtomPairFingerprint,
    AvalonFingerprint,
    ConformerGenerator,
    ECFPFingerprint,
    MACCSFingerprint,
    MQNsFingerprint,
    MolFromSmilesTransformer,
    MordredFingerprint,
    PubChemFingerprint,
    RDKitFingerprint,
    TopologicalTorsionFingerprint,
    pl,
):
    _fp_dict = {
        "ecfp":     ECFPFingerprint,
        "morgan":   ECFPFingerprint,
        "maccs":    MACCSFingerprint,
        "torsion":  TopologicalTorsionFingerprint,
        "rdkit":    RDKitFingerprint,
        "atompair": AtomPairFingerprint,
        "avalon":   AvalonFingerprint,
        "mordred":  MordredFingerprint,
        "mqn":      MQNsFingerprint,
        "pubchem":  PubChemFingerprint,
    }

    def generate_fingerprint(df: pl.DataFrame, fingerprint_type: str, **kwargs) -> pl.DataFrame:
        """
        Generate molecular fingerprints and add them as a column to the DataFrame.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported fingerprint keys.
            **kwargs: Extra arguments forwarded to the skfp fingerprint class.

        Returns:
            DataFrame with an added column named after fingerprint_type.
        """
        if fingerprint_type not in _fp_dict:
            raise ValueError(
                f"Fingerprint type not recognised: {fingerprint_type!r}. "
                f"Valid values: {list(_fp_dict.keys())}"
            )
        fp_func = _fp_dict[fingerprint_type](**kwargs) if kwargs else _fp_dict[fingerprint_type]()
        if fp_func.requires_conformers:
            mol_from_smiles = MolFromSmilesTransformer()
            conf_gen = ConformerGenerator()
            mols_list = mol_from_smiles.transform(df.get_column("smiles"))
            mols_list = conf_gen.transform(mols_list)
        else:
            mols_list = df.get_column("smiles")
        fps = fp_func.transform(mols_list)
        return df.with_columns(pl.Series(values=fps, name=fingerprint_type))

    return (generate_fingerprint,)


@app.cell
def _(Callable, DataStructs, generate_fingerprint, np, pl):
    _METRIC_FNS: dict[str, Callable] = {
        "tanimoto": DataStructs.BulkTanimotoSimilarity,
        "dice":     DataStructs.BulkDiceSimilarity,
        "cosine":   DataStructs.BulkCosineSimilarity,
    }

    def _row_to_bitvect(row: np.ndarray) -> DataStructs.ExplicitBitVect:
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.where(row > 0)[0].tolist():
            bv.SetBit(bit)
        return bv

    def compute_pairwise_similarities(
        df: pl.DataFrame,
        id_col: str,
        fingerprint: str,
        metric: str,
        **fp_kwargs,
    ) -> pl.DataFrame:
        """
        Compute all unique pairwise similarities within a single DataFrame.

        Args:
            df: DataFrame with molecule data and a "smiles" column.
            id_col: Column with unique molecule identifiers.
            fingerprint: Fingerprint type accepted by generate_fingerprint().
            metric: One of "tanimoto", "dice", or "cosine".
            **fp_kwargs: Extra arguments forwarded to the fingerprint constructor.

        Returns:
            Polars DataFrame with columns ID1, ID2, fingerprint, metric, similarity.
        """
        if metric not in _METRIC_FNS:
            raise ValueError(f"metric must be one of {list(_METRIC_FNS.keys())!r}")

        if fingerprint not in df.columns:
            df = generate_fingerprint(df, fingerprint, **fp_kwargs)

        fp_array = np.vstack(df[fingerprint].to_list())
        ids = df[id_col].cast(pl.Utf8).to_list()
        n = len(ids)
        bitvects = [_row_to_bitvect(fp_array[i]) for i in range(n)]
        bulk_fn = _METRIC_FNS[metric]

        id1_list, id2_list, sim_list = [], [], []
        for i in range(n - 1):
            sims = bulk_fn(bitvects[i], bitvects[i + 1:])
            id1_list.extend([ids[i]] * len(sims))
            id2_list.extend(ids[i + 1:])
            sim_list.extend(sims)

        return pl.DataFrame({
            "ID1":         id1_list,
            "ID2":         id2_list,
            "fingerprint": pl.Series([fingerprint] * len(sim_list), dtype=pl.Utf8),
            "metric":      pl.Series([metric]      * len(sim_list), dtype=pl.Utf8),
            "similarity":  pl.Series(sim_list,                      dtype=pl.Float32),
        })

    return (compute_pairwise_similarities,)


@app.cell
def _(Optional, Path, gaussian_kde, np, pl, plt):
    def plot_similarity_distributions(
        sim_df: pl.DataFrame,
        group_col: str,
        group_order: list[str],
        colors: list[str],
        title: str,
        x_label: str = "Tanimoto similarity",
        save_path: Optional[str | Path] = None,
        figsize: tuple[float, float] = (6, 5),
        dpi: int = 300,
    ) -> "matplotlib.figure.Figure":
        """
        Plot overlapping KDE curves for Tanimoto similarity distributions.

        Args:
            sim_df: DataFrame with "similarity" and group_col columns.
            group_col: Column used to split into separate curves.
            group_order: Ordered list of group labels.
            colors: Hex/named colours, one per group.
            title: Plot title.
            x_label: x-axis label.
            save_path: If given, saves the figure here.
            figsize: Figure size in inches.
            dpi: Resolution for raster output.

        Returns:
            matplotlib.figure.Figure.
        """
        _x = np.linspace(0, 1, 500)
        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            for group, color in zip(group_order, colors):
                vals = (
                    sim_df.filter(pl.col(group_col) == group)
                    .get_column("similarity").to_numpy()
                )
                if len(vals) < 2:
                    continue
                kde = gaussian_kde(vals, bw_method="scott")
                ax.plot(_x, kde(_x), color=color, linewidth=2, label=group)
                ax.fill_between(_x, kde(_x), alpha=0.15, color=color)
            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=11, frameon=True, framealpha=0.9, edgecolor="0.8")
            ax.set_title(title, fontsize=13)
            fig.tight_layout()
        if save_path is not None:
            fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")
        return fig

    return (plot_similarity_distributions,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(Chem, pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    # Also read the raw test file to access the Molecule Name column
    test = pl.read_csv("../data/raw/20260409/dose_response_test.csv").rename({"SMILES": "smiles"})

    def _smi_to_inchikey(smi: str):
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol) if mol else None

    test = test.with_columns(
        pl.col("smiles").map_elements(_smi_to_inchikey, return_dtype=pl.Utf8).alias("inchikey")
    )

    all_compounds
    return all_compounds, test


@app.cell
def _(mo):
    mo.md(r"""
    ## Train / test similarity comparison

    Three ECFP4 Tanimoto similarity distributions are overlaid:

    - **Within train** — all unique pairs among dose-response training compounds
    - **Within test**  — all unique pairs among test-set compounds
    - **Train vs test** — every (train, test) cross-set pair

    A large gap between "within train" and "train vs test" would indicate that
    the test set probes chemical space not well represented by the training data.
    """)
    return


@app.cell
def _(
    DataStructs,
    all_compounds,
    compute_pairwise_similarities,
    generate_fingerprint,
    np,
    pl,
    plot_similarity_distributions,
    test,
):
    _train_df = (
        all_compounds
        .filter(pl.col("in_dose_response"))
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )
    _test_df = (
        test
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )

    _FP_KWARGS = {"fp_size": 1024, "radius": 2, "include_chirality": True}

    _sim_train = (
        compute_pairwise_similarities(
            _train_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto", **_FP_KWARGS,
        )
        .with_columns(pl.lit("within_train").alias("comparison"))
    )
    _sim_test = (
        compute_pairwise_similarities(
            _test_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto", **_FP_KWARGS,
        )
        .with_columns(pl.lit("within_test").alias("comparison"))
    )

    # Cross-set similarity matrix: every (train, test) pair
    def _to_bv(row: np.ndarray) -> DataStructs.ExplicitBitVect:
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.where(row > 0)[0].tolist():
            bv.SetBit(bit)
        return bv

    _train_fp = generate_fingerprint(_train_df, "ecfp", **_FP_KWARGS)
    _test_fp  = generate_fingerprint(_test_df,  "ecfp", **_FP_KWARGS)

    _train_arr = np.vstack(_train_fp["ecfp"].to_list())
    _test_arr  = np.vstack(_test_fp["ecfp"].to_list())

    _train_bvs = [_to_bv(_train_arr[i]) for i in range(len(_train_arr))]
    _test_bvs  = [_to_bv(_test_arr[i])  for i in range(len(_test_arr))]

    _cross_sims: list[float] = []
    for _bv in _train_bvs:
        _cross_sims.extend(DataStructs.BulkTanimotoSimilarity(_bv, _test_bvs))

    _sim_cross = pl.DataFrame({
        "similarity": pl.Series(_cross_sims, dtype=pl.Float32),
        "comparison": pl.Series(["train_vs_test"] * len(_cross_sims), dtype=pl.Utf8),
    })

    _sim_combined = pl.concat([
        _sim_train.select(["similarity", "comparison"]),
        _sim_test.select(["similarity",  "comparison"]),
        _sim_cross.select(["similarity", "comparison"]),
    ])

    plot_similarity_distributions(
        _sim_combined,
        group_col="comparison",
        group_order=["within_train", "within_test", "train_vs_test"],
        colors=["#4e79a7", "#59a14f", "#e15759"],
        title="ECFP4 Tanimoto similarity — train / test comparison",
        save_path="../plots/1_sar_exploration/density_sim_train_test.png",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Test set coverage by training compounds

    For each test compound we compute the **maximum ECFP4 Tanimoto similarity**
    to any compound in the dose-response training set. This gives us the
    nearest neighbor (NN) of the test compound within the training set.

    A high value means the test compound is well-represented in the training data;
    a low value flags a potential extrapolation challenge for any model.

    The scatter shows test compounds projected into 2D via **UMAP** (jaccard
    metric on ECFP4), coloured by nearest-neighbour similarity to the training
    set.  Hover over a point to see the test compound and its closest training
    analogue side by side.
    """)
    return


@app.cell
def _(DataStructs, UMAP, all_compounds, generate_fingerprint, np, pl, test):
    _ECFP4 = {"fp_size": 1024, "radius": 2, "include_chirality": True}

    _train_meta = (
        all_compounds
        .filter(pl.col("in_dose_response"))
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "molecule_names"])
    )
    _test_meta = (
        test
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "Molecule Name"])
    )

    _train_fp = generate_fingerprint(_train_meta, "ecfp", **_ECFP4)
    _test_fp  = generate_fingerprint(_test_meta,  "ecfp", **_ECFP4)

    _train_arr = np.vstack(_train_fp["ecfp"].to_list())
    _test_arr  = np.vstack(_test_fp["ecfp"].to_list())

    def _to_bv(row: np.ndarray):
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.where(row > 0)[0].tolist():
            bv.SetBit(bit)
        return bv

    _train_bvs = [_to_bv(_train_arr[i]) for i in range(len(_train_arr))]
    _test_bvs  = [_to_bv(_test_arr[i])  for i in range(len(_test_arr))]

    _max_sims: list[float] = []
    _nn_idx:   list[int]   = []
    for _bv in _test_bvs:
        _sims = DataStructs.BulkTanimotoSimilarity(_bv, _train_bvs)
        _best = int(np.argmax(_sims))
        _max_sims.append(float(_sims[_best]))
        _nn_idx.append(_best)

    _train_ids    = _train_meta["inchikey"].to_list()
    _train_smiles = _train_meta["smiles"].to_list()
    _train_names  = _train_meta["molecule_names"].to_list()

    _umap = UMAP(
        n_components=2,
        n_neighbors=min(15, len(_test_arr) - 1),
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )
    _umap_coords = _umap.fit_transform(_test_arr)

    test_coverage_df = pl.DataFrame({
        "inchikey":         _test_meta["inchikey"].to_list(),
        "smiles":           _test_meta["smiles"].to_list(),
        "molecule_names":   _test_meta["Molecule Name"].to_list(),
        "UMAP_x":           _umap_coords[:, 0].tolist(),
        "UMAP_y":           _umap_coords[:, 1].tolist(),
        "max_sim_to_train": pl.Series(_max_sims, dtype=pl.Float32),
        "nn_inchikey":      [_train_ids[i]    for i in _nn_idx],
        "nn_smiles":        [_train_smiles[i] for i in _nn_idx],
        "nn_name":          [_train_names[i]  for i in _nn_idx],
    })

    test_coverage_df
    return (test_coverage_df,)


@app.cell
def _(alt, mo, test_coverage_df):
    _sel = alt.selection_point(fields=["inchikey"], name="test_sel", empty=False, on="mouseover", nearest=True, clear="mouseout")

    _scatter = (
        alt.Chart(test_coverage_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X("UMAP_x:Q", title="UMAP 1",
                     axis=alt.Axis(titleFontSize=13, labelFontSize=11)),
            y=alt.Y("UMAP_y:Q", title="UMAP 2",
                     axis=alt.Axis(titleFontSize=13, labelFontSize=11)),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),
                alt.Color("max_sim_to_train:Q",
                           scale=alt.Scale(scheme="viridis"),
                           legend=alt.Legend(title="Max similarity\nto train")),
            ),
            size=alt.condition(_sel, alt.value(120), alt.value(60)),
            tooltip=[
                alt.Tooltip("molecule_names:N",   title="Name"),
                alt.Tooltip("inchikey:N",          title="InChIKey"),
                alt.Tooltip("max_sim_to_train:Q",  title="Max sim to train", format=".3f"),
                alt.Tooltip("nn_name:N",           title="Nearest train compound"),
            ],
        )
        .add_params(_sel)
        .properties(
            title="Test compounds — ECFP4 UMAP coloured by nearest-train similarity",
            width=500,
            height=400,
        )
        .configure_title(fontSize=12)
    )

    test_coverage_chart = mo.ui.altair_chart(_scatter)
    return (test_coverage_chart,)


@app.cell
def _(
    Chem,
    mo,
    pl,
    rdDepictor,
    rdMolDraw2D,
    test_coverage_chart,
    test_coverage_df,
):
    _PANEL_W = 300
    _PANEL_H = 200

    def _smi_to_svg(smi: str, width: int = _PANEL_W, height: int = _PANEL_H) -> str:
        """Render a single SMILES as an SVG string via RDKit MolDraw2DSVG."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def _strip_xml(svg: str) -> str:
        return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

    _sel_rows = test_coverage_chart.value

    if _sel_rows is None or len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:{(_PANEL_H + 40) * 2}px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px'>
                Hover over a point to see the test compound and its nearest training neighbour
            </div>
        """)
    else:
        _key = _sel_rows.row(0, named=True)["inchikey"]
        _r = test_coverage_df.filter(pl.col("inchikey") == _key).row(0, named=True)

        _svg_test = _smi_to_svg(_r["smiles"])
        _svg_nn   = _smi_to_svg(_r["nn_smiles"])

        _name_test = _r["molecule_names"] or _r["inchikey"]
        _name_nn   = _r["nn_name"]        or _r["nn_inchikey"]

        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; font-family:monospace; font-size:11px'>
                <div style='padding:5px; background:#eef4fb;
                            border-radius:4px; text-align:center; margin-bottom:2px'>
                    <b>Test compound</b><br>{_name_test}<br>
                    Max sim to train: <b>{_r['max_sim_to_train']:.3f}</b>
                </div>
                {_strip_xml(_svg_test)}
                <div style='padding:5px; background:#f0faf0;
                            border-radius:4px; text-align:center;
                            margin-top:6px; margin-bottom:2px'>
                    <b>Nearest train compound</b><br>{_name_nn}
                </div>
                {_strip_xml(_svg_nn)}
            </div>
        """)

    mo.hstack([test_coverage_chart, _panel], align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Based on our discussion in notebook 1c of how to define similar compounds,
    almost all test compounds have a NN in the training set that would be considered
    similar (ECFP4 Tanimoto over 0.4)
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Dose-response vs counter screen selectivity

    Compounds that have both a dose-response pEC50 and a counter screen pEC50 are
    plotted against each other to identify selective hits.

    - **Selective**: dose-response pEC50 is at least **1.5 units higher** than
      counter screen pEC50 (compound is active in the primary assay but not in the
      counter screen).
    - **Hit**: selective **and** dose-response pEC50 ≥ **6** (potent enough to
      matter).

    Three reference lines are drawn:
    - Diagonal — identity line (pEC50_dr = pEC50_counter).
    - Diagonal shifted down 1.5 units — the selectivity threshold.
    - Vertical line at pEC50_dr = 6 — the potency threshold.
    """)
    return


@app.cell
def _(all_compounds, pl):
    ## Build a dataset restricted to compounds with both pEC50 values available.
    # Filter to rows that have both dose-response and counter screen pEC50 values.
    dr_counter_df = (
        all_compounds
        .filter(
            pl.col("pEC50_dr").is_not_null()
            & pl.col("pEC50_counter").is_not_null()
        )
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "molecule_names", "pEC50_dr", "pEC50_counter"])
        # Mark a compound as selective if its dose-response pEC50 is >1.5 units
        # above the counter screen pEC50 — i.e. it is potent in the primary assay
        # but not in the counter assay.
        .with_columns(
            (pl.col("pEC50_dr") - pl.col("pEC50_counter") > 1.5).alias("selective")
        )
        # A hit must also clear the absolute potency threshold of pEC50_dr >= 6.
        .with_columns(
            (pl.col("selective") & (pl.col("pEC50_dr") >= 6.0)).alias("hit")
        )
        # A readable category label for tooltip / legend use.
        .with_columns(
            pl.when(pl.col("hit"))
            .then(pl.lit("Hit"))
            .when(pl.col("selective"))
            .then(pl.lit("Selective"))
            .otherwise(pl.lit("Non-selective"))
            .alias("category")
        )
    )

    dr_counter_df
    return (dr_counter_df,)


@app.cell
def _(alt, dr_counter_df, mo, np):
    ## Build reference line data for the three guide lines.
    # Determine axis range with a small margin so the lines extend past all points.
    _x_min = float(dr_counter_df["pEC50_dr"].min()) - 0.3
    _x_max = float(dr_counter_df["pEC50_dr"].max()) + 0.3
    _y_min = float(dr_counter_df["pEC50_counter"].min()) - 0.3
    _y_max = float(dr_counter_df["pEC50_counter"].max()) + 0.3

    # Both axes share the same global range so diagonal lines are meaningful.
    _global_min = min(_x_min, _y_min)
    _global_max = max(_x_max, _y_max)

    _line_pts = np.linspace(_global_min, _global_max, 200).tolist()

    # Identity diagonal: counter = dr
    _diag_data = {
        "x": _line_pts,
        "y": _line_pts,
        "line": ["Identity (counter = DR)"] * len(_line_pts),
    }
    # Selectivity threshold diagonal: counter = dr - 1.5
    _thresh_data = {
        "x": _line_pts,
        "y": [v - 1.5 for v in _line_pts],
        "line": ["Selectivity threshold (DR − 1.5)"] * len(_line_pts),
    }
    # Potency threshold: vertical line at pEC50_dr = 6
    _potency_pts = np.linspace(_global_min, _global_max, 200).tolist()
    _potency_data = {
        "x": [6.0] * len(_potency_pts),
        "y": _potency_pts,
        "line": ["Potency threshold (DR = 6)"] * len(_potency_pts),
    }

    # Colour palette: Non-selective = grey, Selective = orange, Hit = red.
    _color_scale = alt.Scale(
        domain=["Non-selective", "Selective", "Hit"],
        range=["#b0b0b0", "#f28e2b", "#e15759"],
    )
    # Dash pattern for each reference line.
    _dash_scale = alt.Scale(
        domain=["Identity (counter = DR)", "Selectivity threshold (DR − 1.5)", "Potency threshold (DR = 6)"],
        range=[[4, 4], [6, 3], [3, 3]],
    )

    # Click selection — one point at a time, keyed on inchikey.
    _sel = alt.selection_point(fields=["inchikey"], name="dr_sel", empty=False, on="mouseover", nearest=True, clear="mouseout")

    ## Scatter layer — one point per compound.
    _scatter_layer = (
        alt.Chart(dr_counter_df)
        .mark_circle(opacity=0.75, size=60)
        .encode(
            x=alt.X(
                "pEC50_dr:Q",
                title="Dose-response pEC50",
                scale=alt.Scale(domain=[_global_min, _global_max]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            y=alt.Y(
                "pEC50_counter:Q",
                title="Counter screen pEC50",
                scale=alt.Scale(domain=[_global_min, _global_max]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),
                alt.Color(
                    "category:N",
                    scale=_color_scale,
                    legend=alt.Legend(
                        title=None,
                        orient="bottom",
                        direction="horizontal",
                        titleFontSize=12,
                        labelFontSize=11,
                    ),
                ),
            ),
            size=alt.condition(_sel, alt.value(120), alt.value(60)),
            tooltip=[
                alt.Tooltip("molecule_names:N", title="Name"),
                alt.Tooltip("inchikey:N",        title="InChIKey"),
                alt.Tooltip("pEC50_dr:Q",         title="DR pEC50",      format=".2f"),
                alt.Tooltip("pEC50_counter:Q",    title="Counter pEC50", format=".2f"),
                alt.Tooltip("category:N",         title="Category"),
            ],
        )
        .add_params(_sel)
    )

    ## Reference line layers — diagonal, threshold, potency.
    _ref_diag = (
        alt.Chart(alt.Data(values=[{"x": x, "y": y, "line": l} for x, y, l in zip(_diag_data["x"], _diag_data["y"], _diag_data["line"])]))
        .mark_line(strokeWidth=1.5)
        .encode(
            x="x:Q",
            y="y:Q",
            strokeDash=alt.StrokeDash("line:N", scale=_dash_scale, legend=None),
            color=alt.value("#555555"),
        )
    )
    _ref_thresh = (
        alt.Chart(alt.Data(values=[{"x": x, "y": y, "line": l} for x, y, l in zip(_thresh_data["x"], _thresh_data["y"], _thresh_data["line"])]))
        .mark_line(strokeWidth=1.5)
        .encode(
            x="x:Q",
            y="y:Q",
            strokeDash=alt.StrokeDash("line:N", scale=_dash_scale, legend=None),
            color=alt.value("#555555"),
        )
    )
    _ref_potency = (
        alt.Chart(alt.Data(values=[{"x": x, "y": y, "line": l} for x, y, l in zip(_potency_data["x"], _potency_data["y"], _potency_data["line"])]))
        .mark_line(strokeWidth=1.5)
        .encode(
            x="x:Q",
            y="y:Q",
            strokeDash=alt.StrokeDash("line:N", scale=_dash_scale, legend=None),
            color=alt.value("#555555"),
        )
    )

    _chart = (
        alt.layer(_ref_diag, _ref_thresh, _ref_potency, _scatter_layer)
        .properties(
            title="Dose-response vs counter screen pEC50 — selectivity & hits",
            width=520,
            height=480,
        )
        .configure_title(fontSize=13)
        .configure_legend(orient="bottom", direction="horizontal", titleAnchor="middle")
        .configure_view(stroke="transparent")
    )

    dr_counter_chart = mo.ui.altair_chart(_chart)
    return (dr_counter_chart,)


@app.cell
def _(Chem, dr_counter_chart, dr_counter_df, mo, rdDepictor, rdMolDraw2D):
    _PANEL_W = 280
    _PANEL_H = 220

    def _smi_to_svg(smi: str, width: int = _PANEL_W, height: int = _PANEL_H) -> str:
        """Render a SMILES as an SVG string via RDKit MolDraw2DSVG."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    def _strip_xml(svg: str) -> str:
        return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

    # apply_selection filters dr_counter_df to the clicked point(s).
    # This is required for layered charts where .value returns UndefinedType.
    _sel_rows = dr_counter_chart.apply_selection(dr_counter_df)

    if len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:{_PANEL_H + 80}px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px; text-align:center; padding:12px'>
                Click a point to see the compound structure
            </div>
        """)
    else:
        _r = _sel_rows.row(0, named=True)
        _svg = _smi_to_svg(_r["smiles"])
        # Colour the info box header by category.
        _bg = {"Hit": "#fde8e8", "Selective": "#fef3e2", "Non-selective": "#f0f0f0"}[_r["category"]]
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; font-family:monospace; font-size:11px'>
                <div style='padding:6px; background:{_bg}; border-radius:4px;
                            margin-bottom:4px; line-height:1.8'>
                    <b>{_r['molecule_names'] or _r['inchikey']}</b><br>
                    <b>Category:</b> {_r['category']}<br>
                    <b>DR pEC50:</b> {_r['pEC50_dr']:.2f}<br>
                    <b>Counter pEC50:</b> {_r['pEC50_counter']:.2f}<br>
                    <b>ΔpEC50:</b> {_r['pEC50_dr'] - _r['pEC50_counter']:.2f}
                </div>
                {_strip_xml(_svg)}
            </div>
        """)

    mo.hstack([dr_counter_chart, _panel], align="start")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Counter-screen status of nearest training neighbours

    For each test compound we found its nearest neighbour (NN) in the dose-response
    training set.  Here we ask: **what do we know about that NN compound in the
    counter screen?**

    Four mutually exclusive categories are assigned to each NN:

    | Category | Meaning |
    |---|---|
    | **Not tested** | NN was never run in the counter assay |
    | **Non-selective** | NN has a counter pEC50 within 1.5 units of its DR pEC50 |
    | **Selective** | NN counter pEC50 is >1.5 units below DR pEC50, but DR pEC50 < 6 |
    | **Hit** | Selective *and* DR pEC50 ≥ 6 |

    This tells us how much selectivity context exists in the training data for the
    regions of chemical space where the test compounds live.
    """)
    return


@app.cell
def _(all_compounds, pl, test_coverage_df):
    # Build a lookup of counter-screen status for every dose-response train compound.
    # We use all_compounds which carries both in_counter and pEC50_dr / pEC50_counter.
    _train_status = (
        all_compounds
        .filter(pl.col("in_dose_response"))
        .unique(subset=["inchikey"])
        .select(["inchikey", "in_counter", "pEC50_dr", "pEC50_counter"])
        # Derive the same selectivity / hit logic used in dr_counter_df.
        .with_columns(
            pl.when(~pl.col("in_counter"))
            .then(pl.lit("Not tested"))
            .when(pl.col("pEC50_counter").is_null())
            .then(pl.lit("Not tested"))
            .when(pl.col("pEC50_dr").is_null())
            .then(pl.lit("Not tested"))
            .when((pl.col("pEC50_dr") - pl.col("pEC50_counter") > 1.5) & (pl.col("pEC50_dr") >= 6.0))
            .then(pl.lit("Hit"))
            .when(pl.col("pEC50_dr") - pl.col("pEC50_counter") > 1.5)
            .then(pl.lit("Selective"))
            .otherwise(pl.lit("Non-selective"))
            .alias("nn_counter_status")
        )
        .select(["inchikey", "nn_counter_status"])
    )

    # Join the NN inchikey in test_coverage_df to the status lookup.
    nn_status_df = (
        test_coverage_df
        .select(["inchikey", "molecule_names", "max_sim_to_train", "nn_inchikey", "nn_name"])
        .join(
            _train_status.rename({"inchikey": "nn_inchikey"}),
            on="nn_inchikey",
            how="left",
        )
        # NNs not present in the lookup at all (shouldn't happen, but guard against it).
        .with_columns(
            pl.col("nn_counter_status").fill_null("Not tested")
        )
    )

    # Summary counts per category.
    _order = ["Hit", "Selective", "Non-selective", "Not tested"]
    nn_status_summary = (
        nn_status_df
        .group_by("nn_counter_status")
        .agg(pl.len().alias("n_test_compounds"))
        .with_columns(
            (pl.col("n_test_compounds") / pl.col("n_test_compounds").sum() * 100)
            .round(1)
            .alias("pct")
        )
        .with_columns(
            pl.col("nn_counter_status")
            .cast(pl.Enum(_order))
        )
        .sort("nn_counter_status")
    )

    nn_status_summary
    return (nn_status_summary,)


@app.cell
def _(mo, nn_status_summary, plt):
    _order = ["Hit", "Selective", "Non-selective", "Not tested"]
    _colors = ["#e15759", "#f28e2b", "#b0b0b0", "#d3d3f5"]

    # Reorder rows to match the desired category order.
    _rows = {r["nn_counter_status"]: r for r in nn_status_summary.iter_rows(named=True)}
    _labels = [cat for cat in _order if cat in _rows]
    _counts = [_rows[cat]["n_test_compounds"] for cat in _labels]
    _pcts   = [_rows[cat]["pct"] for cat in _labels]
    _bar_colors = [_colors[_order.index(cat)] for cat in _labels]

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(figsize=(5, 4), dpi=150)
        _bars = _ax.bar(_labels, _counts, color=_bar_colors, edgecolor="white", linewidth=0.8)

        # Annotate each bar with count and percentage.
        for _bar, _n, _pct in zip(_bars, _counts, _pcts):
            _ax.text(
                _bar.get_x() + _bar.get_width() / 2,
                _bar.get_height() + 0.3,
                f"{_n}\n({_pct}%)",
                ha="center", va="bottom", fontsize=9,
            )

        _ax.set_xlabel("Counter-screen status of nearest train neighbour", fontsize=11)
        _ax.set_ylabel("Number of test compounds", fontsize=11)
        _ax.set_title("Counter-screen status of nearest training neighbour\nper test compound", fontsize=11)
        _ax.tick_params(axis="x", labelsize=10)
        _fig.tight_layout()

    _fig.savefig("../plots/1_sar_exploration/barchart_test_nn_train.png", 
                    dpi=300, bbox_inches="tight")
    mo.center(mo.as_html(_fig))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the majority of the test compounds seem to derive from selective compounds (> 90%) it is surprising
    to see that the hit category is not higher. This could be an artifact, it could happen that an analog of a hit
    compound is identified within a commercially available compound set has a NN that is different within our dataset.
    """)
    return


if __name__ == "__main__":
    app.run()
