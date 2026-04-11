import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 1b — Chemical space embeddings & MMP network

    Reads `all_compounds_activity_data.csv` (produced by **1a**) and the pre-computed
    MMP pairs from `all_compounds_mmp.mmp.csv.gz`.

    Produces:
    - UMAP and t-SNE scatter plots of the full chemical space coloured by dataset membership
    - MMP network visualisation restricted to dose-response, counter-screen, and test compounds

    Static PNG outputs are saved to `plots/1_sar_exploration/`.
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import numpy as np
    from pathlib import Path
    from typing import Optional

    import matplotlib.pyplot as plt
    import matplotlib.figure
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    import networkx as nx

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    from rdkit import Chem, RDLogger

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
        Chem,
        ConformerGenerator,
        ECFPFingerprint,
        LineCollection,
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
        mo,
        mpatches,
        np,
        nx,
        pl,
        plt,
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
        Generate molecular fingerprints using scikit-fingerprints and add them as a column.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported types (ecfp/morgan, maccs, torsion,
                rdkit, atompair, avalon, mordred, mqn, pubchem).
            **kwargs: Additional keyword arguments forwarded to the skfp fingerprint class.

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
def _(PCA, TSNE, UMAP, np, pl):
    def add_tsne_columns(df: pl.DataFrame, fp_column: str = "ecfp") -> pl.DataFrame:
        """
        Adds TSNE_x and TSNE_y columns using PCA pre-reduction followed by t-SNE.

        Args:
            df: DataFrame containing the fingerprint column.
            fp_column: Name of the fingerprint array column.

        Returns:
            DataFrame with TSNE_x and TSNE_y columns added.
        """
        fp_array = np.vstack(df[fp_column].to_list())
        n_samples = fp_array.shape[0]

        if n_samples <= 1:
            return df.with_columns(
                pl.lit(float("nan")).alias("TSNE_x"),
                pl.lit(float("nan")).alias("TSNE_y"),
            )

        n_pca_components = min(50, n_samples - 1)
        pca = PCA(n_components=n_pca_components, random_state=42)
        fp_reduced = pca.fit_transform(fp_array)

        perplexity = min(30.0, float(n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity,
                    init="pca", learning_rate="auto")
        tsne_results = tsne.fit_transform(fp_reduced)

        return df.with_columns(
            pl.Series(name="TSNE_x", values=tsne_results[:, 0]),
            pl.Series(name="TSNE_y", values=tsne_results[:, 1]),
        )


    def add_umap_columns(
        df: pl.DataFrame,
        fp_column: str = "ecfp",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ) -> pl.DataFrame:
        """
        Adds UMAP_x and UMAP_y columns using UMAP dimensionality reduction.

        Args:
            df: DataFrame containing the fingerprint column.
            fp_column: Name of the fingerprint array column.
            n_neighbors: Controls local vs. global structure trade-off.
            min_dist: Controls how tightly points are packed.
            metric: Distance metric (euclidean, cosine, jaccard).

        Returns:
            DataFrame with UMAP_x and UMAP_y columns added.
        """
        fp_array = np.vstack(df[fp_column].to_list())
        n_samples = fp_array.shape[0]

        if n_samples <= 1:
            return df.with_columns(
                pl.lit(float("nan")).alias("UMAP_x"),
                pl.lit(float("nan")).alias("UMAP_y"),
            )

        umap = UMAP(
            n_components=2,
            n_neighbors=min(n_neighbors, n_samples - 1),
            min_dist=min_dist,
            metric=metric,
            random_state=42,
        )
        umap_results = umap.fit_transform(fp_array)

        return df.with_columns(
            pl.Series(name="UMAP_x", values=umap_results[:, 0]),
            pl.Series(name="UMAP_y", values=umap_results[:, 1]),
        )

    return add_tsne_columns, add_umap_columns


@app.cell
def _(
    LinearSegmentedColormap,
    Optional,
    Path,
    add_tsne_columns,
    add_umap_columns,
    generate_fingerprint,
    mpatches,
    pl,
    plt,
):
    def generate_embedding_plot(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        cutoff_value: Optional[float] = None,
        x_title: str = "X",
        y_title: str = "Y",
        fp_column: str = "ecfp",
        method: str = "tsne",
        umap_metric: str = "euclidean",
        title: Optional[str] = None,
        color_legend: Optional[dict[str, str]] = None,
        legend_loc: str = "best",
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8.0, 7.0),
        point_size: float = 18.0,
        alpha: float = 0.8,
        cmap: str = "viridis",
    ) -> "matplotlib.figure.Figure":
        """
        Generate a 2D scatter plot of a molecular embedding.

        Fingerprint and embedding columns are computed on demand if absent.

        Args:
            df: Polars DataFrame with molecule data.
            x_col: Column name for the x-axis coordinate.
            y_col: Column name for the y-axis coordinate.
            color_col: Optional column for point colour.
            cutoff_value: When set alongside color_col, splits points into two classes.
            x_title: x-axis label.
            y_title: y-axis label.
            fp_column: Fingerprint column; computed via generate_fingerprint if absent.
            method: Embedding method — "tsne" or "umap".
            umap_metric: Distance metric passed to UMAP.
            title: Optional plot title.
            color_legend: Dict mapping hex colour strings to legend labels.
            legend_loc: Legend location string (matplotlib convention).
            save_path: If given, saves the figure here.
            dpi: Resolution for raster output (default 300).
            figsize: Figure dimensions in inches.
            point_size: Scatter point size in points².
            alpha: Point opacity (0–1).
            cmap: Matplotlib colourmap name or "ryg" for red-yellow-green.

        Returns:
            matplotlib.figure.Figure
        """
        if fp_column not in df.columns:
            df = generate_fingerprint(df, fp_column)

        if x_col not in df.columns or y_col not in df.columns:
            if method == "umap":
                df = add_umap_columns(df, fp_column, metric=umap_metric)
            else:
                df = add_tsne_columns(df, fp_column)

        # ── Matplotlib path ──────────────────────────────────────────────────────
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if color_col:
                raw = df[color_col].to_numpy()
                is_color_string = df[color_col].dtype == pl.Utf8
                if is_color_string:
                    ax.scatter(x, y, c=raw, s=point_size, linewidths=0)
                    if color_legend:
                        ax.legend(
                            handles=[mpatches.Patch(color=hex_, label=label)
                                     for hex_, label in color_legend.items()],
                            loc=legend_loc, frameon=True, fontsize=10, markerscale=1.5,
                        )
                elif cutoff_value is not None:
                    values = raw.astype(float)
                    ax.scatter(x[values > cutoff_value], y[values > cutoff_value],
                               c="red", s=point_size, alpha=alpha, linewidths=0,
                               label=f"{color_col} > {cutoff_value}")
                    ax.scatter(x[values <= cutoff_value], y[values <= cutoff_value],
                               c="steelblue", s=point_size, alpha=alpha, linewidths=0,
                               label=f"{color_col} ≤ {cutoff_value}")
                    ax.legend(frameon=True, fontsize=11, markerscale=1.5)
                else:
                    values = raw.astype(float)
                    _cmap = (
                        LinearSegmentedColormap.from_list("ryg", ["#d73027", "#fee08b", "#1a9850"])
                        if cmap == "ryg" else cmap
                    )
                    sc = ax.scatter(x, y, c=values, cmap=_cmap,
                                    s=point_size, alpha=alpha, linewidths=0)
                    cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                    cbar.set_label(color_col, fontsize=12)
                    cbar.ax.tick_params(labelsize=10)
            else:
                ax.scatter(x, y, c="steelblue", s=point_size, alpha=alpha, linewidths=0)

            ax.set_xlabel(x_title, fontsize=14, labelpad=8)
            ax.set_ylabel(y_title, fontsize=14, labelpad=8)
            ax.tick_params(axis="both", labelsize=11)
            if title:
                ax.set_title(title, fontsize=16, pad=12)
            fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

        return fig

    return (generate_embedding_plot,)


@app.cell
def _(
    LineCollection,
    LinearSegmentedColormap,
    Optional,
    Path,
    mpatches,
    np,
    nx,
    pl,
    plt,
):
    def _run_graph_layout(
        node_ids: list[str],
        edges_df: pl.DataFrame,
        layout: str,
        iterations: int,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Build a networkx graph and compute a 2D force-directed layout.

        Multi-component graphs are laid out component-by-component and then
        packed into a grid so clusters are spatially separated.

        Args:
            node_ids: All molecule identifiers.
            edges_df: DataFrame with ID1 and ID2 columns (and optional weight).
            layout: "fruchterman_reingold" or "kamada_kawai".
            iterations: FR iteration count.
            seed: Random seed for reproducible FR layout.

        Returns:
            Dict mapping node_id → (x, y) position array.
        """
        g = nx.Graph()
        g.add_nodes_from(node_ids)

        weight_col: Optional[str] = None
        if "similarity" in edges_df.columns:
            weight_col = "similarity"
        elif "CoreSize" in edges_df.columns:
            max_core = edges_df["CoreSize"].max()
            if max_core and max_core > 0:
                edges_df = edges_df.with_columns(
                    (pl.col("CoreSize") / max_core).alias("_weight")
                )
                weight_col = "_weight"

        for row in edges_df.iter_rows(named=True):
            w = float(row[weight_col]) if weight_col else 1.0
            g.add_edge(row["ID1"], row["ID2"], weight=w)

        components = list(nx.connected_components(g))

        if len(components) == 1:
            if layout == "kamada_kawai":
                raw = nx.kamada_kawai_layout(g, weight="weight")
            else:
                raw = nx.fruchterman_reingold_layout(
                    g, weight="weight", seed=seed, iterations=iterations
                )
            return {node: np.array(pos) for node, pos in raw.items()}

        from tqdm import tqdm

        sorted_components = sorted(components, key=len, reverse=True)
        sub_layouts: list[tuple[list[str], np.ndarray, float]] = []

        for component in tqdm(sorted_components, desc="Laying out components", unit="component"):
            subg = g.subgraph(component)
            if len(component) == 1:
                sub_pos = {next(iter(component)): np.array([0.0, 0.0])}
            elif layout == "kamada_kawai":
                sub_pos = nx.kamada_kawai_layout(subg, weight="weight")
            else:
                sub_pos = nx.fruchterman_reingold_layout(
                    subg, weight="weight", seed=seed, iterations=iterations
                )

            coords = np.array(list(sub_pos.values()))
            if coords.ptp(axis=0).min() > 0:
                coords = (coords - coords.min(axis=0)) / coords.ptp(axis=0)
            else:
                coords = np.zeros_like(coords)

            scale = np.sqrt(len(component))
            sub_layouts.append((list(sub_pos.keys()), coords, scale))

        total_area = sum(s ** 2 for _, _, s in sub_layouts)
        target_row_width = np.sqrt(total_area)
        gap = 0.15

        all_positions: dict[str, np.ndarray] = {}
        cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0

        for nodes, coords, scale in sub_layouts:
            w = scale * (1 + gap)
            if cursor_x > 0 and cursor_x + scale > target_row_width:
                cursor_x = 0.0
                cursor_y += row_height + gap
                row_height = 0.0
            offset = np.array([cursor_x, cursor_y])
            for node, xy in zip(nodes, coords * scale):
                all_positions[node] = xy + offset
            cursor_x += w
            row_height = max(row_height, scale)

        return all_positions

    _RYG_COLORS = ["#d73027", "#fee08b", "#1a9850"]

    def generate_similarity_network(
        molecules_df: pl.DataFrame,
        id_col: str,
        smiles_col: str = "smiles",
        property_col: Optional[str] = None,
        edges_df: Optional[pl.DataFrame] = None,
        similarity_threshold: float = 0.4,
        layout: str = "fruchterman_reingold",
        layout_iterations: int = 100,
        node_size: int = 200,
        edge_opacity: float = 0.35,
        max_edges: Optional[int] = 5000,
        title: Optional[str] = None,
        property_title: Optional[str] = None,
        color_legend: Optional[dict[str, str]] = None,
        legend_loc: str = "lower center",
        legend_ncols: int = 4,
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8.0, 7.0),
    ) -> "matplotlib.figure.Figure":
        """
        Generate a molecular similarity network.

        Nodes represent molecules; edges connect structurally related pairs (from
        a pre-computed MMP edge list or pairwise fingerprint similarity).  Node
        colour encodes an optional numeric property (red → yellow → green).

        Args:
            molecules_df: DataFrame with molecule data.
            id_col: Column with unique molecule identifiers.
            smiles_col: Column containing SMILES strings.
            property_col: Optional numeric or hex-colour column for node colour.
            edges_df: Pre-computed edge list with ID1 and ID2 columns.
            similarity_threshold: Minimum similarity when edges_df is None.
            layout: "fruchterman_reingold" or "kamada_kawai".
            layout_iterations: FR layout iteration count.
            node_size: Marker area in points².
            edge_opacity: Edge line opacity (0–1).
            max_edges: Maximum edges to render (top-N by weight).
            title: Optional plot title.
            property_title: Colourbar/legend label.
            color_legend: Dict of hex→label pairs for categorical colour columns.
            legend_loc: Legend position string.
            legend_ncols: Number of legend columns.
            save_path: If given, saves the figure here.
            dpi: Resolution for raster output.
            figsize: Figure dimensions in inches.

        Returns:
            matplotlib.figure.Figure
        """
        if id_col not in molecules_df.columns:
            raise ValueError(f"id_col {id_col!r} not found in molecules_df")
        if smiles_col not in molecules_df.columns:
            raise ValueError(f"smiles_col {smiles_col!r} not found in molecules_df")

        node_ids: list[str] = molecules_df[id_col].cast(pl.Utf8).to_list()
        node_id_set = set(node_ids)

        resolved_edges = edges_df.filter(
            pl.col("ID1").cast(pl.Utf8).is_in(node_id_set)
            & pl.col("ID2").cast(pl.Utf8).is_in(node_id_set)
        )
        if max_edges is not None and len(resolved_edges) > max_edges:
            if "similarity" in resolved_edges.columns:
                resolved_edges = resolved_edges.sort("similarity", descending=True)
            elif "CoreSize" in resolved_edges.columns:
                resolved_edges = resolved_edges.sort("CoreSize", descending=True)
            print(f"Warning: keeping top {max_edges} edges by weight.")
            resolved_edges = resolved_edges.head(max_edges)

        positions = _run_graph_layout(node_ids, resolved_edges, layout, layout_iterations)
        node_x = [float(positions[nid][0]) for nid in node_ids]
        node_y = [float(positions[nid][1]) for nid in node_ids]
        colorbar_title = property_title if property_title is not None else property_col

        is_color_string = (
            property_col is not None
            and property_col in molecules_df.columns
            and molecules_df[property_col].dtype == pl.Utf8
        )

        # ── Matplotlib path ──────────────────────────────────────────────────────
        edge_segments = []
        for row in resolved_edges.iter_rows(named=True):
            id1, id2 = str(row["ID1"]), str(row["ID2"])
            x0, y0 = positions[id1]; x1, y1 = positions[id2]
            edge_segments.append([(float(x0), float(y0)), (float(x1), float(y1))])

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
            lc = LineCollection(edge_segments, colors=[(0.5, 0.5, 0.5, edge_opacity)], linewidths=0.8)
            ax.add_collection(lc)

            node_x_arr = np.array(node_x)
            node_y_arr = np.array(node_y)

            if is_color_string:
                ax.scatter(node_x_arr, node_y_arr,
                           c=molecules_df[property_col].to_numpy(),
                           s=node_size, linewidths=0.8, edgecolors="white", zorder=2)
                if color_legend:
                    _handles = [mpatches.Patch(color=hex_, label=label)
                                for hex_, label in color_legend.items()]
                    _outside = legend_loc in ("lower center", "upper center", "center left", "center right")
                    _anchor = {
                        "lower center": (0.5, -0.08), "upper center": (0.5, 1.08),
                        "center left": (-0.08, 0.5), "center right": (1.08, 0.5),
                    }.get(legend_loc)
                    ax.legend(handles=_handles, loc=legend_loc, bbox_to_anchor=_anchor,
                              ncols=legend_ncols, frameon=True, fontsize=10, borderaxespad=0.0)
                    if _outside:
                        fig.tight_layout()
            elif property_col is not None:
                ryg_cmap = LinearSegmentedColormap.from_list("ryg", _RYG_COLORS)
                sc = ax.scatter(node_x_arr, node_y_arr,
                                c=molecules_df[property_col].to_numpy().astype(float),
                                cmap=ryg_cmap, s=node_size, linewidths=0.8,
                                edgecolors="white", zorder=2)
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(colorbar_title, fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            else:
                ax.scatter(node_x_arr, node_y_arr, c="steelblue", s=node_size,
                           linewidths=0.8, edgecolors="white", zorder=2)

            ax.autoscale_view()
            ax.set_axis_off()
            if title:
                ax.set_title(title, fontsize=16, pad=12)
            fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

        return fig

    return (generate_similarity_network,)


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
def _(mo):
    mo.md(r"""
    ## Filter MMP pairs

    Keep only pairs where the variable section is smaller than the
    common core (core_transform_ratio < 1.0). This is a pretty broad definition
    of an MMP. If the dataset contains larger set of structurally similar analogs,
    I would set the ratio anywhere from 0.5 to 0.2 and maybe also restrict the size
    of the variable section further. However, in this dataset we have a very diverse
    set and we generate few enough MMPs with the more relaxed constraints as it is.
    """)
    return


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
def _(add_tsne_columns, add_umap_columns, generate_fingerprint, pl):
    def prepare_for_plotting(df: pl.DataFrame) -> pl.DataFrame:
        """
        Compute ECFP6 fingerprints, t-SNE and UMAP embeddings in one pass.

        Uses ECFP6 (radius=3, 4096-bit, chirality-aware, count-based) with
        Jaccard distance for UMAP to approximate Tanimoto similarity.
        """
        df = generate_fingerprint(df, "ecfp", fp_size=4096, radius=3,
                                   include_chirality=True, count=True)
        df = add_tsne_columns(df)
        df = add_umap_columns(df, metric="jaccard")
        return df

    return (prepare_for_plotting,)


@app.cell
def _(all_compounds, prepare_for_plotting):
    plot_df = prepare_for_plotting(all_compounds)
    plot_df
    return (plot_df,)


@app.cell
def _(pl, plot_df):
    # Priority order (highest wins): test > counter > dose-response > single-dose only
    plot_df_colored = plot_df.with_columns(
        pl.when(pl.col("in_test"))
          .then(pl.lit("#1a7a4a"))          # dark green  — test set
          .when(pl.col("in_counter"))
          .then(pl.lit("#1a3a6b"))          # dark blue   — counter screen
          .when(pl.col("in_dose_response"))
          .then(pl.lit("#6baed6"))          # light blue  — dose response
          .otherwise(pl.lit("#c0c0c080"))   # light grey 50% transparent — single dose only
          .alias("color")
    )
    plot_df_colored
    return (plot_df_colored,)


@app.cell
def _(mo):
    mo.md(r"""
    ## UMAP chemical space (coloured by dataset)
    """)
    return


@app.cell
def _(generate_embedding_plot, plot_df_colored):
    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    generate_embedding_plot(
        plot_df_colored, x_col="UMAP_x", y_col="UMAP_y",
        color_col="color", point_size=6,
        title="UMAP ECFP6 Chemical Space",
        color_legend=_legend,
        save_path="../plots/1_sar_exploration/umap_ecfp6_chemical_space.png",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## t-SNE chemical space (coloured by dataset)
    """)
    return


@app.cell
def _(generate_embedding_plot, plot_df_colored):
    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    generate_embedding_plot(
        plot_df_colored, x_col="TSNE_x", y_col="TSNE_y",
        color_col="color", point_size=6,
        title="tSNE ECFP6 Chemical Space",
        color_legend=_legend,
        save_path="../plots/1_sar_exploration/tsne_ecfp6_chemical_space.png",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## MMP network

    Nodes: dose-response, counter-screen, and test compounds that have at least
    one MMP partner.  Edges: filtered MMP pairs (core_transform_ratio < 1.0).
    Node colour encodes dataset membership (same palette as the embedding plots).
    """)
    return


@app.cell
def _(generate_similarity_network, mmp_df_filtered, pl, plot_df_colored):
    _candidate_keys = set(
        plot_df_colored
        .filter(pl.col("in_dose_response") | pl.col("in_counter") | pl.col("in_test"))
        .get_column("inchikey")
        .to_list()
    )

    _edges = mmp_df_filtered.filter(
        pl.col("ID1").is_in(_candidate_keys) & pl.col("ID2").is_in(_candidate_keys)
    )

    _connected_keys = set(_edges["ID1"].to_list()) | set(_edges["ID2"].to_list())
    _network_df = plot_df_colored.filter(pl.col("inchikey").is_in(_connected_keys))

    print(f"Nodes: {len(_network_df)}, Edges: {len(_edges)}")

    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    generate_similarity_network(
        _network_df, "inchikey",
        edges_df=_edges, node_size=16, layout_iterations=50,
        property_col="color", max_edges=15000, color_legend=_legend,
        save_path="../plots/1_sar_exploration/mmp_network.png",
    )
    return


if __name__ == "__main__":
    app.run()
