import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 1c — Activity cliff analysis

    Activity cliffs (ACs) are pairs of molecules that are structurally very similar
    but have large differences in biological activity.  They are challenging for
    machine-learning models because they violate the "similar molecules ↔ similar
    activity" assumption that most models rely on.

    This notebook:
    1. Identifies ACs from the **MMP** table (transformation-based cliffs).
    2. Identifies ACs from **pairwise fingerprint similarity** (threshold-based).
    3. Displays an interactive scatter of all ECFP4 cliffs with side-by-side
       structure rendering.
    4. Shows a UMAP of the dose-response subset coloured by pEC50.

    **Input:** `data/processed/all_compounds_activity_data.csv` and
    `data/processed/all_compounds_mmp.mmp.csv.gz` (both produced by notebook **1a**).
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    import numpy as np
    import base64
    from pathlib import Path
    from typing import Optional, Callable

    import matplotlib.pyplot as plt
    import matplotlib.figure
    from matplotlib.colors import LinearSegmentedColormap
    from scipy.stats import gaussian_kde
    from matplotlib_venn import venn2

    from sklearn.decomposition import PCA
    from umap import UMAP
    from sklearn.manifold import TSNE

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdDepictor, CombineMols
    from rdkit.Chem.Draw import rdMolDraw2D

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

    RDLogger.DisableLog("rdApp.*")
    return (
        AtomPairFingerprint,
        AvalonFingerprint,
        Callable,
        Chem,
        CombineMols,
        ConformerGenerator,
        DataStructs,
        ECFPFingerprint,
        LinearSegmentedColormap,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        Optional,
        PCA,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        TSNE,
        TopologicalTorsionFingerprint,
        UMAP,
        alt,
        base64,
        gaussian_kde,
        mo,
        np,
        pl,
        plt,
        rdDepictor,
        rdMolDraw2D,
        venn2,
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
        """Convert a binary uint8 numpy fingerprint row to an RDKit ExplicitBitVect."""
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
        Compute pairwise molecular similarities for all unique compound pairs.

        Each unique (i, j) pair with i < j is evaluated once using RDKit bulk ops.
        Fingerprints are generated on demand if absent.

        Args:
            df: Polars DataFrame with molecule data and a "smiles" column.
            id_col: Column name with unique molecule identifiers.
            fingerprint: Fingerprint type accepted by generate_fingerprint().
            metric: One of "tanimoto", "dice", or "cosine".
            **fp_kwargs: Extra arguments forwarded to the fingerprint constructor.

        Returns:
            Polars DataFrame with columns ID1, ID2, fingerprint, metric, similarity.
        """
        if metric not in _METRIC_FNS:
            raise ValueError(f"metric must be one of {list(_METRIC_FNS.keys())!r}, got {metric!r}")

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
        Plot overlapping KDE curves for pairwise Tanimoto similarity distributions.

        Args:
            sim_df: Polars DataFrame with a "similarity" column and group_col.
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
def _(
    ECFPFingerprint,
    LinearSegmentedColormap,
    Optional,
    PCA,
    Path,
    TSNE,
    UMAP,
    np,
    pl,
    plt,
):
    def generate_embedding_plot_continuous(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: str,
        fp_column: str = "ecfp",
        method: str = "umap",
        umap_metric: str = "euclidean",
        title: Optional[str] = None,
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8.0, 7.0),
        point_size: float = 18.0,
        alpha: float = 0.8,
        cmap: str = "viridis",
    ) -> "matplotlib.figure.Figure":
        """
        Matplotlib scatter for a continuous colour column — recomputes embedding if absent.

        Args:
            df: DataFrame with molecule data.
            x_col / y_col: Embedding coordinate columns (computed if absent).
            color_col: Numeric column for colour encoding.
            fp_column: Fingerprint column (ECFP by default).
            method: "umap" or "tsne".
            umap_metric: Distance metric passed to UMAP.
            title: Optional plot title.
            save_path: If given, saves the figure here.
            dpi: Resolution for raster output.
            figsize: Figure size in inches.
            point_size: Marker area in points².
            alpha: Marker opacity.
            cmap: Matplotlib colourmap name or "ryg".

        Returns:
            matplotlib.figure.Figure.
        """

        if fp_column not in df.columns:
            fp_func = ECFPFingerprint(fp_size=4096, radius=3, include_chirality=True, count=True)
            fps = fp_func.transform(df.get_column("smiles"))
            df = df.with_columns(pl.Series(values=fps, name=fp_column))

        fp_array = np.vstack(df[fp_column].to_list())
        n_samples = fp_array.shape[0]

        if x_col not in df.columns or y_col not in df.columns:
            if method == "umap":
                umap = UMAP(
                    n_components=2,
                    n_neighbors=min(15, n_samples - 1),
                    min_dist=0.1,
                    metric=umap_metric,
                    random_state=42,
                )
                coords = umap.fit_transform(fp_array)
            else:
                n_pca = min(50, n_samples - 1)
                pca = PCA(n_components=n_pca, random_state=42)
                coords_pca = pca.fit_transform(fp_array)

                tsne = TSNE(n_components=2, random_state=42,
                             perplexity=min(30.0, float(n_samples - 1)),
                             init="pca", learning_rate="auto")
                coords = tsne.fit_transform(coords_pca)
            df = df.with_columns(
                pl.Series(name=x_col, values=coords[:, 0]),
                pl.Series(name=y_col, values=coords[:, 1]),
            )

        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()
        values = df[color_col].to_numpy().astype(float)

        _cmap = (
            LinearSegmentedColormap.from_list("ryg", ["#d73027", "#fee08b", "#1a9850"])
            if cmap == "ryg" else cmap
        )

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            sc = ax.scatter(x, y, c=values, cmap=_cmap, s=point_size, alpha=alpha, linewidths=0)
            cbar = fig.colorbar(sc, ax=ax, pad=0.02)
            cbar.set_label(color_col, fontsize=12)
            cbar.ax.tick_params(labelsize=10)
            ax.set_xlabel("UMAP 1" if method == "umap" else "t-SNE 1", fontsize=14, labelpad=8)
            ax.set_ylabel("UMAP 2" if method == "umap" else "t-SNE 2", fontsize=14, labelpad=8)
            ax.tick_params(axis="both", labelsize=11)
            if title:
                ax.set_title(title, fontsize=16, pad=12)
            fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

        return fig

    return (generate_embedding_plot_continuous,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load data
    """)
    return


@app.cell
def _(pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")

    mmp_raw = pl.read_csv(
        "../data/processed/all_compounds_mmp.mmp.csv.gz",
        separator="\t",
        has_header=False,
        new_columns=["smiles1", "smiles2", "ID1", "ID2", "transform", "core"],
    )
    all_compounds
    return all_compounds, mmp_raw


@app.cell
def _(Chem, mmp_raw, pl):
    _n_ha1, _n_ha2, _n_ha_core, _n_ha_frag1, _n_ha_frag2 = [], [], [], [], []
    for _s1, _s2, _core, _transform in mmp_raw.select(
        ["smiles1", "smiles2", "core", "transform"]
    ).iter_rows():
        _lhs, _rhs = _transform.split(">>")
        _m_lhs = Chem.MolFromSmiles(_lhs)
        _m_rhs = Chem.MolFromSmiles(_rhs)
        _n_ha1.append(Chem.MolFromSmiles(_s1).GetNumHeavyAtoms())
        _n_ha2.append(Chem.MolFromSmiles(_s2).GetNumHeavyAtoms())
        _n_ha_core.append(Chem.MolFromSmiles(_core).GetNumHeavyAtoms())
        _n_ha_frag1.append(_m_lhs.GetNumHeavyAtoms() if _m_lhs else None)
        _n_ha_frag2.append(_m_rhs.GetNumHeavyAtoms() if _m_rhs else None)

    mmp_df_filtered = (
        mmp_raw
        .with_columns(
            pl.Series("n_ha1",     _n_ha1,     dtype=pl.Int32),
            pl.Series("n_ha2",     _n_ha2,     dtype=pl.Int32),
            pl.Series("n_ha_core", _n_ha_core, dtype=pl.Int32),
            pl.Series("n_ha_frag1",_n_ha_frag1,dtype=pl.Int32),
            pl.Series("n_ha_frag2",_n_ha_frag2,dtype=pl.Int32),
        )
        .with_columns(
            (abs(pl.col("n_ha_frag1") - pl.col("n_ha_frag2"))).alias("size_diff_transform"),
            (pl.max_horizontal("n_ha_frag1", "n_ha_frag2") / pl.col("n_ha_core")).alias("core_transform_ratio"),
        )
        .filter(pl.col("core_transform_ratio") < 1.0)
    )
    mmp_df_filtered
    return (mmp_df_filtered,)


@app.cell
def _(mo):
    mo.md(r"""
    ## UMAP of dose-response compounds coloured by pEC50

    This plot uses only the compounds tested in the dose-response assay.
    The embedding is recomputed on this subset alone (different from the full-
    dataset embedding in notebook 1b) to maximise structural resolution within
    the active chemical space.
    """)
    return


@app.cell
def _(all_compounds, generate_embedding_plot_continuous, pl):
    _dr_df = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .drop(["UMAP_x", "UMAP_y", "TSNE_x", "TSNE_y"],
              strict=False)   # drop pre-computed coords if present
    )

    generate_embedding_plot_continuous(
        _dr_df,
        x_col="UMAP_x",
        y_col="UMAP_y",
        color_col="pEC50_dr",
        method="umap",
        umap_metric="jaccard",
        title="UMAP Chemical Space — Training pEC50 (dose-response)",
        point_size=20,
        alpha=0.85,
        cmap="viridis",
        save_path="../plots/1_sar_exploration/umap_pec50_chemical_space.png",
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    Similar to the plot on the whole chemical space, most low active compounds tend to
    cluster together.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## MMP-based activity cliffs

    One approach to identifying ACs is to use MMPs: we keep pairs from the
    filtered MMP table where both compounds were tested in the dose-response assay
    and their pEC50 values differ by **≥ 2 log units**.

    This dataset has low MMP propensity (high chemical diversity), so the number
    of MMP cliffs is small.
    """)
    return


@app.cell
def _(all_compounds, mmp_df_filtered, pl):
    _pec50 = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "pEC50_dr"])
    )

    mmp_activity_cliffs = (
        mmp_df_filtered
        .join(_pec50.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1"}), on="ID1", how="inner")
        .join(_pec50.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2"}), on="ID2", how="inner")
        .with_columns(
            (pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50")
        )
        .filter(pl.col("delta_pEC50") >= 2.0)
        .sort("delta_pEC50", descending=True)
    )

    mmp_activity_cliffs
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Fingerprint-based activity cliffs

    An alternative approach defines a similarity threshold to consider a pair of
    compounds "similar".  The threshold depends on the fingerprint and metric, so whenever
    someone says a Tanimoto similarity value without saying the fingerprint is providing useless
    information. Based on rule of thumb and previous experience, I generally use the following
    thresholds to define similar compounds:

    - MACCS + Tanimoto → threshold ≈ 0.8
    - ECFP4 + Tanimoto → threshold ≈ 0.4

    First, we compare the pairwise similarity distributions of three fingerprints
    across all dose-response training compounds.
    """)
    return


@app.cell
def _(
    all_compounds,
    compute_pairwise_similarities,
    pl,
    plot_similarity_distributions,
):
    _dr_df = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )

    _sim_maccs = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="maccs", metric="tanimoto",
    )
    _sim_ecfp4_1k = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto",
        fp_size=1024, radius=2, include_chirality=True,
    )
    _sim_ecfp4_4k = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto",
        fp_size=4096, radius=2, include_chirality=True,
    )
    _sim_ecfp6 = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto",
        fp_size=4096, radius=3, include_chirality=True,
    )

    # Relabel ECFP variants by their common names
    _sim_ecfp4_1k = _sim_ecfp4_1k.with_columns(pl.lit("ecfp4_1k").alias("fingerprint"))
    _sim_ecfp4_4k = _sim_ecfp4_4k.with_columns(pl.lit("ecfp4_4k").alias("fingerprint"))
    _sim_ecfp6 = _sim_ecfp6.with_columns(pl.lit("ecfp6").alias("fingerprint"))

    pairwise_similarities = pl.concat([_sim_maccs, _sim_ecfp4_1k, _sim_ecfp4_4k, _sim_ecfp6])

    plot_similarity_distributions(
        pairwise_similarities,
        group_col="fingerprint",
        group_order=["maccs", "ecfp4_1k", "ecfp4_4k", "ecfp6"],
        colors=["#4e79a7", "#f28e2b", "#e15759", "#59a14f"],
        title="Pairwise similarity distributions — dose-response compounds",
        save_path="../plots/1_sar_exploration/density_sim_fingerprints.png",
    )
    return (pairwise_similarities,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This plot demonstrates how different the similarity value distributions can be for different
    fingerprints.
    ECFP6 generally has lower similarity values.
    Even small changes such as the bit size for ECFP4 (1024 vs 4096) makes a difference, although a small one.
    It is always useful to check the similarity distribution within a dataset and decide if
    you need to adapt those rule of thumb thresholds.
    For the analysis below I decided to keep them.
    """)
    return


@app.cell
def _(np, pairwise_similarities, pl):
    """Summary statistics for each fingerprint distribution."""
    _fp_order = ["maccs", "ecfp4_1k", "ecfp4_4k", "ecfp6"]
    _rows = []
    for _fp in _fp_order:
        _vals = (
            pairwise_similarities.filter(pl.col("fingerprint") == _fp)
            .get_column("similarity").to_numpy()
        )
        _rows.append({
            "fingerprint": _fp.upper(),

            "Q25":    round(float(np.percentile(_vals, 25)),   3),
            "median": round(float(np.median(_vals)),           3),
            "Q75":    round(float(np.percentile(_vals, 75)),   3),
            "p95":    round(float(np.percentile(_vals, 95)),   3),
            "p99":    round(float(np.percentile(_vals, 99)),   3),
            "p99.9":   round(float(np.percentile(_vals, 99.9)),   3),
        })
    pl.DataFrame(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Cliff count summary (MACCS ≥ 0.8 and ECFP4 ≥ 0.4, |ΔpEC50| ≥ 2)
    """)
    return


@app.cell
def _(all_compounds, pairwise_similarities, pl):
    _pec50 = (
        all_compounds.filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "pEC50_dr"])
    )

    _sim_with_delta = (
        pairwise_similarities
        .join(_pec50.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1"}), on="ID1", how="inner")
        .join(_pec50.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2"}), on="ID2", how="inner")
        .with_columns((pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50"))
    )

    _thresholds = {"maccs": 0.8, "ecfp4_1k": 0.4, "ecfp4_4k": 0.4}
    _rows = []
    for _fp, _sim_thresh in _thresholds.items():
        _total_similar = (
            _sim_with_delta
            .filter((pl.col("fingerprint") == _fp) & (pl.col("similarity") >= _sim_thresh))
            .height
        )
        _n_cliffs = (
            _sim_with_delta
            .filter(
                (pl.col("fingerprint") == _fp)
                & (pl.col("similarity") >= _sim_thresh)
                & (pl.col("delta_pEC50") >= 2.0)
            )
            .height
        )
        _rows.append({
            "fingerprint":          _fp.upper(),
            "similarity threshold": _sim_thresh,
            "similar pairs":        _total_similar,
            "activity cliffs":      _n_cliffs,
            "cliff fraction (%)":   round(100 * _n_cliffs / _total_similar, 1) if _total_similar > 0 else 0.0,
        })
    pl.DataFrame(_rows)
    return


@app.cell
def _(Path, all_compounds, pairwise_similarities, pl, plt, venn2):
    """
    Venn diagram comparing which compound pairs are flagged as activity cliffs
    by MACCS (Tanimoto ≥ 0.8) vs ECFP4_1k (Tanimoto ≥ 0.4), both requiring
    |ΔpEC50| ≥ 2.  Each pair is represented as a frozenset of its two InChIKeys
    so that (A, B) and (B, A) collapse to the same element.
    """
    _DELTA_THRESH = 2.0

    _pec50 = (
        all_compounds.filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "pEC50_dr"])
    )

    _sim_with_delta = (
        pairwise_similarities
        .join(_pec50.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1"}), on="ID1", how="inner")
        .join(_pec50.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2"}), on="ID2", how="inner")
        .with_columns((pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50"))
    )

    def _cliff_pairs(fp_name: str, sim_thresh: float) -> set[frozenset]:
        """Return the set of cliff pairs (as frozensets) for a given fingerprint."""
        rows = (
            _sim_with_delta
            .filter(
                (pl.col("fingerprint") == fp_name)
                & (pl.col("similarity") >= sim_thresh)
                & (pl.col("delta_pEC50") >= _DELTA_THRESH)
            )
            .select(["ID1", "ID2"])
            .rows()
        )
        return {frozenset(pair) for pair in rows}

    _maccs_cliffs   = _cliff_pairs("maccs",     0.8)
    _ecfp4_1k_cliffs = _cliff_pairs("ecfp4_1k", 0.4)

    _fig, _ax = plt.subplots(figsize=(5, 4))
    venn2(
        subsets=(_maccs_cliffs, _ecfp4_1k_cliffs),
        set_labels=("MACCS ≥ 0.8", "ECFP4_1k ≥ 0.4"),
        set_colors=("#4e79a7", "#f28e2b"),
        alpha=0.6,
        ax=_ax,
    )
    _ax.set_title("Activity cliff overlap — MACCS vs ECFP4_1k\n(|ΔpEC50| ≥ 2)")
    _fig.tight_layout()
    _fig.savefig(
        Path("../plots/1_sar_exploration/venn_cliffs_maccs_ecfp4.png"),
        dpi=150, bbox_inches="tight",
    )
    _fig
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This also shows another complexity of activity cliff analysis, different fingerprints
    will identify different ACs, as their definition of similarity can be very different.
    You can look at consensus cliffs, ACs defined by more than one similarity threshold,
    in our case that would be 10 out of cumulatively more than 500 ACs.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Activity cliff scatter plot (ECFP4, Tanimoto ≥ 0.4)

    Each point is an activity cliff: Tanimoto similarity ≥ 0.4 (ECFP4) and
    |ΔpEC50| ≥ 2 log units. Hover over a point to see both molecules drawn side
    by side.
    """)
    return


@app.cell
def _(
    Chem,
    CombineMols,
    all_compounds,
    alt,
    base64,
    mo,
    pairwise_similarities,
    pl,
    rdDepictor,
    rdMolDraw2D,
):
    def _pair_smiles_to_base64_png(smi1: str, smi2: str, width: int = 300, height: int = 150) -> str:
        """Render two molecules side by side as a base64-encoded PNG data URI."""
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return ""
        combined = CombineMols(mol1, mol2)
        rdDepictor.Compute2DCoords(combined)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(combined)
        drawer.FinishDrawing()
        return f"data:image/png;base64,{base64.b64encode(drawer.GetDrawingText()).decode('ascii')}"

    _CLIFF_THRESHOLD = 2.0
    _SIM_THRESHOLD   = 0.4

    _pec50_smiles = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "smiles", "pEC50_dr", "molecule_names"])
        .unique(subset=["inchikey"])
    )

    _ecfp4_cliffs = (
        pairwise_similarities
        .filter(
            (pl.col("fingerprint") == "ecfp4_1k") & (pl.col("similarity") >= _SIM_THRESHOLD)
        )
        .join(
            _pec50_smiles.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1",
                                   "smiles": "smiles1", "molecule_names": "name1"}),
            on="ID1", how="inner",
        )
        .join(
            _pec50_smiles.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2",
                                   "smiles": "smiles2", "molecule_names": "name2"}),
            on="ID2", how="inner",
        )
        .with_columns((pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50"))
        .filter(pl.col("delta_pEC50") >= _CLIFF_THRESHOLD)
    )

    cliff_plot_df = _ecfp4_cliffs.drop(["fingerprint", "metric"]).with_columns(
        pl.max_horizontal("pEC50_1", "pEC50_2").alias("max_pEC50")
    )

    _sel = alt.selection_point(fields=["ID1", "ID2"], name="cliff_sel", empty=False, on="mouseover", nearest=True, clear="mouseout")

    _scatter = (
        alt.Chart(cliff_plot_df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X("similarity:Q", title="Tanimoto similarity (ECFP4)",
                     scale=alt.Scale(domain=[0.39, 1.0]),
                     axis=alt.Axis(titleFontSize=13, labelFontSize=11)),
            y=alt.Y("delta_pEC50:Q", title="|ΔpEC50| (dose-response)",
                     scale=alt.Scale(domain=[1.9, 4.0]),
                     axis=alt.Axis(titleFontSize=13, labelFontSize=11)),
            color=alt.condition(
                _sel, alt.value("#f5c518"),
                alt.Color("max_pEC50:Q", scale=alt.Scale(scheme="viridis"),
                           legend=alt.Legend(title="max pEC50")),
            ),
            size=alt.condition(_sel, alt.value(120), alt.value(60)),
            tooltip=[
                alt.Tooltip("name1:N",       title="Molecule 1"),
                alt.Tooltip("name2:N",       title="Molecule 2"),
                alt.Tooltip("similarity:Q",  title="Tanimoto",   format=".3f"),
                alt.Tooltip("pEC50_1:Q",     title="pEC50 mol1", format=".2f"),
                alt.Tooltip("pEC50_2:Q",     title="pEC50 mol2", format=".2f"),
                alt.Tooltip("delta_pEC50:Q", title="|ΔpEC50|",   format=".2f"),
                alt.Tooltip("max_pEC50:Q",   title="max pEC50",  format=".2f"),
            ],
        )
        .add_params(_sel)
        .properties(title="Activity cliffs — ECFP4 Tanimoto ≥ 0.4, |ΔpEC50| ≥ 2", width=500, height=400)
        .configure_title(fontSize=12)
    )

    cliff_chart = mo.ui.altair_chart(_scatter)
    return cliff_chart, cliff_plot_df


@app.cell
def _(
    Chem,
    CombineMols,
    cliff_chart,
    cliff_plot_df,
    mo,
    pl,
    rdDepictor,
    rdMolDraw2D,
):
    _PANEL_W = 350

    def _pair_to_svg(smi1: str, smi2: str, width: int = _PANEL_W, height: int = 210) -> str:
        """Render two molecules side by side as an SVG string."""
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return ""
        combined = CombineMols(mol1, mol2)
        rdDepictor.Compute2DCoords(combined)
        drawer = rdMolDraw2D.MolDraw2DSVG(width, height)
        drawer.DrawMolecule(combined)
        drawer.FinishDrawing()
        return drawer.GetDrawingText()

    _sel_rows = cliff_chart.value

    if _sel_rows is None or len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:400px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px'>
                Hover over a point to see the pair of structures
            </div>
        """)
    else:
        _row = _sel_rows.row(0, named=True)
        _match = cliff_plot_df.filter(
            (pl.col("ID1") == _row["ID1"]) & (pl.col("ID2") == _row["ID2"])
        )
        _r = _match.row(0, named=True)
        _svg = _pair_to_svg(_r["smiles1"], _r["smiles2"])
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px'>
                <div style='font-size:11px; font-family:monospace; margin-bottom:6px;
                            padding:6px; background:#f8f8f8; border-radius:4px'>
                    <b>Mol 1:</b> {_r['name1']} &nbsp; pEC50 = {_r['pEC50_1']:.2f}<br>
                    <b>Mol 2:</b> {_r['name2']} &nbsp; pEC50 = {_r['pEC50_2']:.2f}<br>
                    Tanimoto {_r['similarity']:.3f} &nbsp;|&nbsp; |ΔpEC50| {_r['delta_pEC50']:.2f}
                </div>
                {_svg}
            </div>
        """)

    mo.hstack([cliff_chart, _panel], align="start")
    return


if __name__ == "__main__":
    app.run()
