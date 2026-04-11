import marimo

__generated_with = "0.23.0"
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
    to any compound in the dose-response training set.

    A high value means the test compound is well-represented in the training data;
    a low value flags a potential extrapolation challenge for any model.

    The scatter shows test compounds projected into 2D via **UMAP** (jaccard
    metric on ECFP4), coloured by nearest-neighbour similarity to the training
    set.  Click a point to see the test compound and its closest training
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
    _sel = alt.selection_point(fields=["inchikey"], name="test_sel", empty=False)

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
                Click a point to see the test compound and its nearest training neighbour
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


if __name__ == "__main__":
    app.run()
