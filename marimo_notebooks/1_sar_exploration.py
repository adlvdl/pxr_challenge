import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Import libraries and define useful functions
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    import numpy as np
    import subprocess
    import sys
    import base64
    import itertools
    import networkx as nx
    from pathlib import Path
    from typing import Optional, Callable
    from collections import defaultdict, deque

    import matplotlib.pyplot as plt
    import matplotlib.figure
    from matplotlib.collections import LineCollection
    from matplotlib.colors import LinearSegmentedColormap
    import matplotlib.patches as mpatches

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from umap import UMAP

    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdDepictor, CombineMols
    from rdkit.Chem.Draw import rdMolDraw2D
    from scipy.stats import gaussian_kde

    # Suppress RDKit InChI warnings
    RDLogger.DisableLog('rdApp.*')

    return (
        Chem, DataStructs, Optional, Callable, PCA, Path, TSNE, UMAP,
        alt, base64, defaultdict, deque, itertools, gaussian_kde,
        LineCollection, LinearSegmentedColormap, mpatches,
        mo, np, nx, pl, plt, rdDepictor, rdMolDraw2D,
        CombineMols, subprocess, sys,
    )



@app.cell
def _(Chem):
    def smi_to_inchikey(smi: str):
        return Chem.MolToInchiKey(Chem.MolFromSmiles(smi))
    def smi_to_inchi(smi: str):
        return Chem.MolToInchi(Chem.MolFromSmiles(smi))

    return smi_to_inchi, smi_to_inchikey


@app.cell
def _(pl):
    from skfp.fingerprints import (
        ECFPFingerprint,
        MACCSFingerprint,
        TopologicalTorsionFingerprint,
        RDKitFingerprint,
        AtomPairFingerprint,
        AvalonFingerprint,
        E3FPFingerprint,
        MordredFingerprint,
        MQNsFingerprint,
        PubChemFingerprint,

    )
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

    _fp_dict={
        "ecfp": ECFPFingerprint,
        "morgan": ECFPFingerprint,
        "maccs": MACCSFingerprint,
        "torsion": TopologicalTorsionFingerprint,
        "rdkit": RDKitFingerprint,
        "atompair": AtomPairFingerprint,
        "avalon": AvalonFingerprint,
        "e3fp": E3FPFingerprint,
        "mordred": MordredFingerprint,
        "mqn": MQNsFingerprint,
        "pubchem": PubChemFingerprint,


    }

    def generate_fingerprint(df: pl.DataFrame, fingerprint_type: str, **kwargs) -> pl.DataFrame:
        """
        Generate molecular fingerprints using scikit-fingerprints and add them as a column.

        Dispatches to the appropriate skfp fingerprint class based on fingerprint_type.
        For fingerprint types that require 3D conformers (e.g., E3FP), conformers are
        generated automatically via RDKit ETKDGv3.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported types: "ecfp"/"morgan", "maccs",
                "torsion", "rdkit", "atompair", "avalon", "e3fp", "mordred", "mqn",
                "pubchem", "descriptors".
            **kwargs: Additional keyword arguments forwarded to the skfp fingerprint class
                constructor (e.g., radius=3, n_bits=1024 for ECFP).

        Returns:
            DataFrame with an added column named after fingerprint_type containing
            the computed fingerprint arrays.

        Raises:
            ValueError: If fingerprint_type is not a recognized key.
        """
        if fingerprint_type not in _fp_dict.keys():
            raise ValueError(f"Fingerprint type not recognized: {fingerprint_type!r}. Valid values: {list(_fp_dict.keys())}")

        if len(kwargs)==0:        
            fp_func = _fp_dict[fingerprint_type]()
        else:

            fp_func = _fp_dict[fingerprint_type](**kwargs)

        if fp_func.requires_conformers:
            mol_from_smiles = MolFromSmilesTransformer()
            conf_gen = ConformerGenerator()
            mols_list = mol_from_smiles.transform(df.get_column("smiles"))
            mols_list = conf_gen.transform(mols_list)

        else:
            mols_list = df.get_column("smiles")

        fps = fp_func.transform(mols_list)
        fps_col = pl.Series(values=fps, name=fingerprint_type)
        fps = df.with_columns( fps_col)
        return fps

    return (generate_fingerprint,)


@app.cell
def _(PCA, TSNE, UMAP, np, pl):
    def add_tsne_columns(df: pl.DataFrame, fp_column: str = "ecfp") -> pl.DataFrame:
        """
        Adds TSNE_x and TSNE_y columns to a DataFrame.

        This function calculates fingerprints (if not already present),
        reduces their dimensionality using PCA, and then applies t-SNE
        to generate 2D coordinates.

        Args:
            df: DataFrame to add the t-SNE columns to.
            fp_column: Name of the column containing fingerprints.
                       If it doesn't exist, it will be calculated using 'np_counts_fp' type.

        Returns:
            DataFrame with the added TSNE_x and TSNE_y columns.
        """


        fp_array = np.vstack(df[fp_column].to_list())
        n_samples = fp_array.shape[0]

        if n_samples <= 1:
            return df.with_columns(
                pl.lit(float("nan")).alias("TSNE_x"),
                pl.lit(float("nan")).alias("TSNE_y"),
            )

        # Reduce dimensions with PCA
        n_pca_components = min(50, n_samples - 1)
        pca = PCA(n_components=n_pca_components, random_state=42)
        fp_reduced = pca.fit_transform(fp_array)

        # Apply t-SNE
        # Perplexity must be less than n_samples
        perplexity = min(30.0, float(n_samples - 1))
        tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity, init='pca', learning_rate='auto')
        tsne_results = tsne.fit_transform(fp_reduced)
        df = df.with_columns(
            pl.Series(name="TSNE_x", values=tsne_results[:, 0]),
            pl.Series(name="TSNE_y", values=tsne_results[:, 1])
        )

        return df


    def add_umap_columns(
        df: pl.DataFrame,
        fp_column: str = "ecfp",
        n_neighbors: int = 15,
        min_dist: float = 0.1,
        metric: str = "euclidean",
    ) -> pl.DataFrame:
        """
        Adds UMAP_x and UMAP_y columns to a DataFrame.

        Reduces fingerprint vectors to 2D using UMAP. Faster than t-SNE on large
        datasets and better preserves global structure.

        Args:
            df: DataFrame containing the fingerprint column.
            fp_column: Name of the column containing fingerprint arrays.
            n_neighbors: Controls the local vs. global structure trade-off.
            min_dist: Controls how tightly points are packed in the embedding.
            metric: Distance metric passed to UMAP. Any metric supported by
                ``umap-learn`` is accepted (e.g. ``"euclidean"``, ``"cosine"``,
                ``"jaccard"``). ``"jaccard"`` is equivalent to Tanimoto distance
                for binary fingerprints and produces a chemically meaningful
                layout. Defaults to ``"euclidean"``.

        Returns:
            DataFrame with added UMAP_x and UMAP_y columns.


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
    Chem, LinearSegmentedColormap, Optional, Path,
    add_tsne_columns, add_umap_columns, alt, base64,
    generate_fingerprint, mo, mpatches, pl, plt, rdDepictor, rdMolDraw2D,
):
    def _smi_to_base64_png(smi: str, width: int = 200, height: int = 200) -> str:
        """Render a SMILES as a base64-encoded PNG data URI using RDKit MolDraw2DCairo."""
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return ""
        rdDepictor.Compute2DCoords(mol)
        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    def add_image_column(df: pl.DataFrame, image_column: str = "image", smiles_column: str = "smiles") -> pl.DataFrame:
        """
        Adds a column with base64-encoded PNG molecule images to a DataFrame.

        Renders each SMILES via RDKit MolDraw2DCairo and encodes as a data URI
        suitable for Altair image tooltips.

        Args:
            df: DataFrame to add the image column to.
            image_column: Name of the new image column.
            smiles_column: Name of the column containing SMILES strings.

        Returns:
            DataFrame with the added image column.
        """
        if image_column not in df.columns:
            image_list = df[smiles_column].map_elements(
                _smi_to_base64_png, return_dtype=pl.Utf8
            )
            df = df.with_columns(pl.Series(name=image_column, values=image_list))
        return df

    def generate_embedding_plot(
        df: pl.DataFrame,
        x_col: str,
        y_col: str,
        color_col: Optional[str] = None,
        cutoff_value: Optional[float] = None,
        x_title: str = "X",
        y_title: str = "Y",
        image_col: str = "image",
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
        for_marimo: bool = False,
    ) -> "matplotlib.figure.Figure | mo.ui.altair_chart":
        """
        Generates a 2D scatter plot of a molecular embedding.

        Two output modes are available, selected via the for_marimo parameter:

        - for_marimo=False (default): Returns a publication-quality
          matplotlib.figure.Figure with a seaborn whitegrid style. Color modes:
            - color_col + cutoff_value: binary red/blue split at the cutoff with a legend.
            - color_col only: continuous viridis colorbar.
            - neither: uniform steelblue.
          Supports save_path, dpi, figsize, point_size, and alpha.

        - for_marimo=True: Returns a mo.ui.altair_chart with image tooltips, designed
          for use inside a marimo notebook. Fingerprints, embeddings, and molecule
          images are computed on demand when the corresponding columns are absent.

        In both modes, fingerprint and embedding columns are computed on demand if absent.

        Args:
            df: Polars DataFrame with molecule data.
            x_col: Column name for the x-axis coordinate.
            y_col: Column name for the y-axis coordinate.
            color_col: Optional column to use for point colouring.
            cutoff_value: When provided alongside color_col, splits points into
                two classes (above = red, at/below = blue). Ignored for the
                marimo path.
            x_title: x-axis label.
            y_title: y-axis label.
            image_col: Column name for base64 molecule images (marimo path only).
                Computed from smiles via add_image_column() if absent.
            fp_column: Fingerprint column; computed via generate_fingerprint if absent.
            method: Embedding method — "tsne" or "umap".
            umap_metric: Distance metric passed to UMAP when ``method="umap"``.
                Any metric supported by ``umap-learn`` is accepted (e.g.
                ``"euclidean"``, ``"cosine"``, ``"jaccard"``). ``"jaccard"`` is
                equivalent to Tanimoto distance for binary fingerprints and
                produces a chemically meaningful layout. Ignored when
                ``method="tsne"``. Defaults to ``"euclidean"``.
            title: Optional plot title.
            save_path: If given, the matplotlib figure is saved to this path. The
                format is inferred from the file extension (png, pdf, svg). Only
                used when for_marimo=False.
            dpi: Resolution for raster output (default 300). Only used when
                for_marimo=False.
            figsize: Figure dimensions in inches (width, height). Only used when
                for_marimo=False.
            point_size: Scatter point size in points² (matplotlib `s` parameter).
                Only used when for_marimo=False.
            alpha: Point opacity (0–1). Only used when for_marimo=False.
            for_marimo: If False (default), returns a matplotlib Figure. If True,
                returns a mo.ui.altair_chart with image tooltips.

        Returns:
            matplotlib.figure.Figure when for_marimo=False, or
            mo.ui.altair_chart when for_marimo=True.

        Raises:
            ValueError: If method is not "tsne" or "umap".
        """


        if fp_column not in df.columns:
            df = generate_fingerprint(df, fp_column)

        if x_col not in df.columns or y_col not in df.columns:
            if method == "umap":
                df = add_umap_columns(df, fp_column, metric=umap_metric)
            else:
                df = add_tsne_columns(df, fp_column)

        # -----------------------------------------------------------------------
        # Altair path (for_marimo=True)
        # -----------------------------------------------------------------------
        if for_marimo:
            if image_col not in df.columns:
                df = add_image_column(df)

            tooltip_cols = [image_col]
            if color_col:
                tooltip_cols.append(color_col)

            chart = alt.Chart(df).mark_circle(size=60)

            if color_col:
                if cutoff_value is not None:
                    color_encoding = alt.condition(
                        alt.datum[color_col] > cutoff_value,
                        alt.value("red"),
                        alt.value("blue"),
                    )
                else:
                    color_encoding = alt.Color(
                        color_col,
                        scale=alt.Scale(scheme="viridis"),
                        legend=alt.Legend(title=color_col),
                    )
            else:
                color_encoding = alt.value("steelblue")

            final_chart = chart.encode(
                x=alt.X(
                    x_col,
                    title=x_title,
                    axis=alt.Axis(titleFontSize=16, labelFontSize=12),
                    scale=alt.Scale(domain=[df[x_col].min(), df[x_col].max()]),
                ),
                y=alt.Y(
                    y_col,
                    title=y_title,
                    axis=alt.Axis(titleFontSize=16, labelFontSize=12),
                    scale=alt.Scale(domain=[df[y_col].min(), df[y_col].max()]),
                ),
                color=color_encoding,
                tooltip=[alt.Tooltip(c) for c in tooltip_cols],
            )

            return mo.ui.altair_chart(final_chart)

        # -----------------------------------------------------------------------
        # Matplotlib path (for_marimo=False)
        # -----------------------------------------------------------------------
        x = df[x_col].to_numpy()
        y = df[y_col].to_numpy()

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            if color_col:
                raw = df[color_col].to_numpy()
                # Detect whether the column contains pre-computed color strings
                # (e.g. hex codes) rather than numeric values.
                is_color_string = df[color_col].dtype == pl.Utf8
                if is_color_string:
                    # Do NOT pass alpha — it overrides any per-color alpha
                    # already embedded in the hex string (8-digit RRGGBBAA).
                    ax.scatter(x, y, c=raw, s=point_size, linewidths=0)
                    if color_legend:
                        ax.legend(
                            handles=[mpatches.Patch(color=hex_, label=label)
                                     for hex_, label in color_legend.items()],
                            loc=legend_loc,
                            frameon=True, fontsize=10, markerscale=1.5,
                        )
                elif cutoff_value is not None:
                    values = raw.astype(float)
                    ax.scatter(
                        x[values > cutoff_value], y[values > cutoff_value],
                        c="red", s=point_size, alpha=alpha, linewidths=0,
                        label=f"{color_col} > {cutoff_value}",
                    )
                    ax.scatter(
                        x[values <= cutoff_value], y[values <= cutoff_value],
                        c="steelblue", s=point_size, alpha=alpha, linewidths=0,
                        label=f"{color_col} ≤ {cutoff_value}",
                    )
                    ax.legend(frameon=True, fontsize=11, markerscale=1.5)
                else:
                    values = raw.astype(float)
                    _cmap = (
                        LinearSegmentedColormap.from_list("ryg", ["#d73027", "#fee08b", "#1a9850"])
                        if cmap == "ryg"
                        else cmap
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
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    return add_image_column, generate_embedding_plot


@app.cell
def _(
    LineCollection, LinearSegmentedColormap, Optional, Path,
    add_image_column, alt, mo, mpatches, np, nx, pl, plt,
):
    def _run_graph_layout(
        node_ids: list[str],
        edges_df: pl.DataFrame,
        layout: str,
        iterations: int,
        seed: int = 42,
    ) -> dict[str, np.ndarray]:
        """
        Builds a networkx graph from an edge list and computes a 2D layout.

        All node IDs (including isolated nodes not present in any edge) are added
        to the graph before layout, ensuring every molecule gets a position.

        Args:
            node_ids: All molecule identifiers (superset of those in edges_df).
            edges_df: DataFrame with columns ID1, ID2, and optionally a weight
                column (similarity or CoreSize). If neither is present, weight
                defaults to 1.0.
            layout: "fruchterman_reingold" or "kamada_kawai".
            iterations: Number of iterations for Fruchterman-Reingold (ignored
                for Kamada-Kawai).
            seed: Random seed for reproducible FR layout.

        Returns:
            Dict mapping each node_id to its (x, y) position as a numpy array.

        Raises:
            ImportError: If networkx is not installed.
        """


        g = nx.Graph()
        g.add_nodes_from(node_ids)

        # Determine weight column
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

        # Two-level layout: FR converges quickly within a single connected
        # component but has no inter-component forces, so a graph with many
        # small components looks like random scatter regardless of iterations.
        # Fix: lay out each component independently, then arrange the component
        # bounding boxes on a grid so clusters are spatially separated.
        components = list(nx.connected_components(g))

        if len(components) == 1:
            # Single component — plain FR or KK
            if layout == "kamada_kawai":
                raw = nx.kamada_kawai_layout(g, weight="weight")
            else:
                raw = nx.fruchterman_reingold_layout(
                    g, weight="weight", seed=seed, iterations=iterations
                )
            return {node: np.array(pos) for node, pos in raw.items()}

        # Multiple components — lay out each one, then pack using a bin-strip
        # algorithm where each component's canvas size scales with sqrt(n_nodes),
        # so larger clusters occupy proportionally more visual space.
        from tqdm import tqdm

        sorted_components = sorted(components, key=len, reverse=True)

        # Lay out each subgraph and normalise to a unit square
        sub_layouts: list[tuple[list[str], np.ndarray, float]] = []
        for component in tqdm(
            sorted_components,
            desc="Laying out components",
            unit="component",
        ):
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

            # Scale factor: sqrt(n) so area ∝ n_nodes
            scale = np.sqrt(len(component))
            sub_layouts.append((list(sub_pos.keys()), coords, scale))

        # Strip packing: fill rows left-to-right; start a new row when the
        # cumulative width exceeds the target row width (sqrt of total area).
        total_area = sum(s ** 2 for _, _, s in sub_layouts)
        target_row_width = np.sqrt(total_area)
        gap = 0.15  # fractional gap between components

        all_positions: dict[str, np.ndarray] = {}
        cursor_x, cursor_y, row_height = 0.0, 0.0, 0.0

        for nodes, coords, scale in sub_layouts:
            w = scale * (1 + gap)
            if cursor_x > 0 and cursor_x + scale > target_row_width:
                # Start a new row
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
        similarity_metric: str = "tanimoto",
        fp_column: str = "ecfp",
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
        image_col: str = "image",
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8.0, 7.0),
        for_marimo: bool = True,
    ) -> "matplotlib.figure.Figure | mo.ui.altair_chart":
        """
        Generates an interactive molecular similarity network visualization.

        Nodes represent molecules; edges connect pairs that are similar (either
        from a pre-computed edge list such as MMP output, or from pairwise
        Tanimoto similarity above a threshold). Node positions are determined by a
        force-directed layout algorithm. Nodes are optionally colored by a numeric
        property using a red (low) → yellow (mid) → green (high) colorscale.
        Hovering a node shows the 2D molecular structure (marimo path only).

        When for_marimo=True the chart is rendered as an Altair layered chart
        (edges as mark_line, nodes as mark_circle) wrapped in mo.ui.altair_chart,
        consistent with the rest of the marimo visualization layer. When
        for_marimo=False a matplotlib Figure is returned: edges are drawn as a
        LineCollection and nodes as a scatter, with an optional colorbar when
        property_col is set.

        Args:
            molecules_df: Polars DataFrame containing molecule data.
            id_col: Column name with unique molecule identifiers.
            smiles_col: Column name containing SMILES strings.
            property_col: Optional numeric column for node color encoding.
            edges_df: Pre-computed edge list with ID1 and ID2 columns (e.g., MMP
                output from find_matched_molecular_pairs()). When None, edges are
                computed from pairwise similarity using fp_column and
                similarity_metric.
            similarity_threshold: Minimum similarity to connect two nodes. Only
                used when edges_df is None. Default 0.4.
            similarity_metric: Similarity metric used for automatic edge
                computation when edges_df is None. One of ``"tanimoto"``
                (default), ``"dice"``, or ``"cosine"``. Ignored when edges_df is
                provided.
            fp_column: Fingerprint column used for similarity edge computation;
                auto-generated via generate_fingerprint() if absent.
            layout: Force-directed layout algorithm — "fruchterman_reingold"
                (default, scales well to large graphs) or "kamada_kawai" (better
                aesthetics for small/dense graphs with fewer than ~150 nodes).
            layout_iterations: Number of iterations for Fruchterman-Reingold layout.
                Lower values (50) speed up layout for large graphs.
            node_size: Marker size for nodes. In Altair (for_marimo=True) this is
                the area in points²; in matplotlib (for_marimo=False) it is the
                area in points². Default 200.
            edge_opacity: Opacity of edge lines (0–1). Default 0.35.
            max_edges: Maximum number of edges to render. When exceeded, the top-N
                edges by similarity (or CoreSize for MMP input) are kept and a
                warning is printed. Set to None to disable the cap.
            title: Optional plot title.
            property_title: Label for the colorbar/legend. Defaults to property_col.
            image_col: Column name for base64 molecule images used in hover
                tooltips (marimo path only). Computed from smiles_col if absent.
            save_path: If given, the matplotlib figure is saved here. Only used
                when for_marimo=False.
            dpi: Resolution for raster output (default 300). Only used when
                for_marimo=False.
            figsize: Figure dimensions in inches (width, height). Only used when
                for_marimo=False.
            for_marimo: If True, returns a mo.ui.altair_chart for use in a marimo
                notebook. If False, returns a matplotlib Figure.

        Returns:
            matplotlib.figure.Figure (for_marimo=False) or
            mo.ui.altair_chart (for_marimo=True).

        Raises:
            ImportError: If networkx is not installed.
            ValueError: If layout is not "fruchterman_reingold" or "kamada_kawai".
            ValueError: If id_col or smiles_col are not in molecules_df.
            ValueError: If edges_df is None and no edges are found above the
                similarity threshold (suggests lowering the threshold).
        """


        _LAYOUT_METHODS = ("fruchterman_reingold", "kamada_kawai")
        if layout not in _LAYOUT_METHODS:
            raise ValueError(f"layout must be one of {_LAYOUT_METHODS}, got {layout!r}")


        if id_col not in molecules_df.columns:
            raise ValueError(f"id_col {id_col!r} not found in molecules_df columns: {molecules_df.columns}")

        if smiles_col not in molecules_df.columns:
            raise ValueError(f"smiles_col {smiles_col!r} not found in molecules_df columns: {molecules_df.columns}")

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
            print(
                f"Warning: edges_df has {len(edges_df)} edges; keeping top {max_edges} "
                f"by weight. Raise max_edges or filter edges_df to suppress this."
            )
            resolved_edges = resolved_edges.head(max_edges)

        # --- Molecule images ---
        if image_col not in molecules_df.columns and for_marimo:
            molecules_df = add_image_column(molecules_df, image_column=image_col, smiles_column=smiles_col)
            images = molecules_df[image_col].to_list()

        # --- Force-directed layout ---
        positions = _run_graph_layout(node_ids, resolved_edges, layout, layout_iterations)

        node_x = [float(positions[nid][0]) for nid in node_ids]
        node_y = [float(positions[nid][1]) for nid in node_ids]


        colorbar_title = property_title if property_title is not None else property_col

        # Detect whether property_col contains pre-computed hex color strings
        # (same logic as generate_embedding_plot)
        is_color_string = (
            property_col is not None
            and property_col in molecules_df.columns
            and molecules_df[property_col].dtype == pl.Utf8
        )

        # -----------------------------------------------------------------------
        # Altair path (for_marimo=True)
        # -----------------------------------------------------------------------
        if for_marimo:
            # Build node DataFrame — one row per molecule
            node_data: dict[str, list] = {
                "x": node_x,
                "y": node_y,
                id_col: node_ids,
                image_col: images,
            }
            if property_col is not None:
                node_data[property_col] = molecules_df[property_col].to_list()
            node_df = pl.DataFrame(node_data)

            # Build edge DataFrame — two rows per edge (one endpoint per row),
            # grouped by edge_id so mark_line draws each segment independently.
            edge_rows: list[dict] = []
            for edge_idx, row in enumerate(resolved_edges.iter_rows(named=True)):
                id1, id2 = str(row["ID1"]), str(row["ID2"])
                x0, y0 = positions[id1]
                x1, y1 = positions[id2]
                edge_rows.append({"x": float(x0), "y": float(y0), "edge_id": edge_idx})
                edge_rows.append({"x": float(x1), "y": float(y1), "edge_id": edge_idx})
            edge_df = pl.DataFrame(edge_rows) if edge_rows else pl.DataFrame({"x": [], "y": [], "edge_id": []})

            edge_layer = (
                alt.Chart(edge_df)
                .mark_line(color="gray", opacity=edge_opacity, strokeWidth=1)
                .encode(
                    x=alt.X("x:Q", axis=None),
                    y=alt.Y("y:Q", axis=None),
                    detail="edge_id:N",
                )
            )

            tooltip_cols = [alt.Tooltip(id_col), alt.Tooltip(image_col)]
            if property_col is not None:
                tooltip_cols.insert(1, alt.Tooltip(property_col))

            if is_color_string:
                # Hex color column — pass values directly, build legend from
                # unique hex→label pairs matching generate_embedding_plot style
                _domain = ["#c0c0c080", "#6baed6", "#1a3a6b", "#1a7a4a"]
                _labels = ["Single dose only", "Dose response", "Counter screen", "Test set"]
                color_encoding = alt.Color(
                    f"{property_col}:N",
                    scale=alt.Scale(domain=_domain, range=_domain),
                    legend=alt.Legend(
                        title="Dataset",
                        labelExpr=(
                            "datum.value === '#c0c0c080' ? 'Single dose only' : "
                            "datum.value === '#6baed6'   ? 'Dose response' : "
                            "datum.value === '#1a3a6b'   ? 'Counter screen' : "
                            "'Test set'"
                        ),
                    ),
                )
            elif property_col is not None:
                color_encoding = alt.Color(
                    property_col,
                    scale=alt.Scale(
                        domain=[
                            node_df[property_col].min(),
                            node_df[property_col].max(),
                        ],
                        range=["red", "yellow", "green"],
                    ),
                    legend=alt.Legend(title=colorbar_title),
                )
            else:
                color_encoding = alt.value("steelblue")

            node_layer = (
                alt.Chart(node_df)
                .mark_circle(size=node_size, stroke="white", strokeWidth=1)
                .encode(
                    x=alt.X("x:Q", axis=None),
                    y=alt.Y("y:Q", axis=None),
                    color=color_encoding,
                    tooltip=tooltip_cols,
                )
            )

            chart = alt.layer(edge_layer, node_layer).properties(
                title=title or "",
                width=600,
                height=500,
            )
            return mo.ui.altair_chart(chart)

        # -----------------------------------------------------------------------
        # Matplotlib path (for_marimo=False)
        # -----------------------------------------------------------------------
        edge_segments: list[list[tuple[float, float]]] = []
        for row in resolved_edges.iter_rows(named=True):
            id1, id2 = str(row["ID1"]), str(row["ID2"])
            x0, y0 = positions[id1]
            x1, y1 = positions[id2]
            edge_segments.append([(float(x0), float(y0)), (float(x1), float(y1))])

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            lc = LineCollection(
                edge_segments,
                colors=[(0.5, 0.5, 0.5, edge_opacity)],
                linewidths=0.8,
            )
            ax.add_collection(lc)

            node_x_arr = np.array(node_x)
            node_y_arr = np.array(node_y)

            if is_color_string:
                # Hex color strings with embedded alpha — do NOT pass scalar
                # alpha so per-point alpha (e.g. #c0c0c080) is preserved
                ax.scatter(
                    node_x_arr, node_y_arr,
                    c=molecules_df[property_col].to_numpy(),
                    s=node_size,
                    linewidths=0.8,
                    edgecolors="white",
                    zorder=2,
                )
                if color_legend:
                    _handles = [mpatches.Patch(color=hex_, label=label)
                                for hex_, label in color_legend.items()]
                    _outside = legend_loc in ("lower center", "upper center",
                                              "center left", "center right")
                    _anchor = {
                        "lower center": (0.5, -0.08),
                        "upper center": (0.5, 1.08),
                        "center left":  (-0.08, 0.5),
                        "center right": (1.08, 0.5),
                    }.get(legend_loc)
                    ax.legend(
                        handles=_handles,
                        loc=legend_loc,
                        bbox_to_anchor=_anchor,
                        ncols=legend_ncols,
                        frameon=True, fontsize=10,
                        borderaxespad=0.0,
                    )
                    if _outside:
                        fig.tight_layout()
            elif property_col is not None:
                property_values = molecules_df[property_col].to_numpy().astype(float)
                ryg_cmap = LinearSegmentedColormap.from_list("ryg", _RYG_COLORS)
                sc = ax.scatter(
                    node_x_arr, node_y_arr,
                    c=property_values,
                    cmap=ryg_cmap,
                    s=node_size,
                    linewidths=0.8,
                    edgecolors="white",
                    zorder=2,
                )
                cbar = fig.colorbar(sc, ax=ax, pad=0.02)
                cbar.set_label(colorbar_title, fontsize=12)
                cbar.ax.tick_params(labelsize=10)
            else:
                ax.scatter(
                    node_x_arr, node_y_arr,
                    c="steelblue",
                    s=node_size,
                    linewidths=0.8,
                    edgecolors="white",
                    zorder=2,
                )

            ax.autoscale_view()
            ax.set_axis_off()
            if title:
                ax.set_title(title, fontsize=16, pad=12)

            fig.tight_layout()

        if save_path is not None:
            save_path = Path(save_path)
            fig.savefig(save_path, dpi=dpi, bbox_inches="tight")

        return fig

    return (generate_similarity_network,)


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

        Each group in `group_order` produces one curve, coloured by the
        corresponding entry in `colors`.  The function is intentionally
        generic so it can be reused for different grouping variables:

        - **fingerprint comparison** — group_col="fingerprint", groups=["maccs", "ecfp4", "ecfp6"]
        - **dataset comparison**     — group_col="comparison",  groups=["within_train", "within_test", "train_vs_test"]

        Args:
            sim_df:      Polars DataFrame with at least a ``similarity`` column
                         (Float32/Float64) and the column named by ``group_col``.
            group_col:   Name of the column used to split curves (e.g. "fingerprint").
            group_order: Ordered list of group labels — one curve per label.
            colors:      List of hex/named colours, one per group.
            title:       Plot title.
            x_label:     x-axis label. Defaults to "Tanimoto similarity".
            save_path:   If given, the figure is saved to this path at ``dpi`` resolution.
            figsize:     Figure size in inches (width, height).
            dpi:         Resolution for raster output.

        Returns:
            matplotlib.figure.Figure with the KDE overlay.
        """
        _x = np.linspace(0, 1, 500)

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=figsize, dpi=dpi)

            for group, color in zip(group_order, colors):
                vals = (
                    sim_df
                    .filter(pl.col(group_col) == group)
                    .get_column("similarity")
                    .to_numpy()
                )
                if len(vals) < 2:
                    continue
                kde = gaussian_kde(vals, bw_method="scott")
                ax.plot(_x, kde(_x), color=color, linewidth=2, label=group)
                ax.fill_between(_x, kde(_x), alpha=0.15, color=color)

            ax.set_xlabel(x_label, fontsize=12)
            ax.set_ylabel("Density", fontsize=12)
            ax.set_xlim(0, 1)
            ax.legend(fontsize=11)
            ax.set_title(title, fontsize=13)
            fig.tight_layout()

        if save_path is not None:
            fig.savefig(Path(save_path), dpi=dpi, bbox_inches="tight")

        return fig

    return (plot_similarity_distributions,)


@app.cell
def _(Callable, DataStructs, generate_fingerprint, np, pl):
    # Maps metric name → RDKit BulkXxxSimilarity function
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

        Fingerprints are generated on demand if the corresponding column is absent.
        Each unique (i, j) pair with i < j is evaluated once — no duplicates,
        no self-comparisons. The result is long-format so that outputs from
        different fingerprints or metrics can be concatenated directly for
        combined analyses.

        Args:
            df: Polars DataFrame containing molecule data. Must include id_col and
                a "smiles" column (used if the fingerprint column is absent).
            id_col: Column name with unique molecule identifiers (e.g. "inchikey").
            fingerprint: Fingerprint type accepted by generate_fingerprint()
                (e.g. "ecfp", "maccs", "rdkit"). Additional keyword arguments
                are forwarded to the fingerprint constructor via fp_kwargs.
            metric: Similarity metric — one of "tanimoto", "dice", or "cosine".
            **fp_kwargs: Extra arguments passed to the fingerprint constructor
                (e.g. fp_size=2048, radius=3 for ECFP).

        Returns:
            Polars DataFrame with columns:
                - ID1, ID2  : identifier pair (ID1 < ID2 in input order)
                - fingerprint: fingerprint name (string literal)
                - metric    : similarity metric name (string literal)
                - similarity: computed similarity value (Float32)

        Raises:
            ValueError: If metric is not one of the supported options.
        """
        if metric not in _METRIC_FNS:
            raise ValueError(
                f"metric must be one of {list(_METRIC_FNS.keys())!r}, got {metric!r}"
            )

        # Generate fingerprints if not already present
        if fingerprint not in df.columns:
            df = generate_fingerprint(df, fingerprint, **fp_kwargs)

        fp_array = np.vstack(df[fingerprint].to_list())
        ids = df[id_col].cast(pl.Utf8).to_list()
        n = len(ids)

        # Pre-convert all rows to ExplicitBitVect for RDKit bulk ops
        bitvects = [_row_to_bitvect(fp_array[i]) for i in range(n)]

        bulk_fn = _METRIC_FNS[metric]

        id1_list: list[str] = []
        id2_list: list[str] = []
        sim_list: list[float] = []

        # For each molecule i, compute its similarity to all j > i in one bulk call
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
def _(pl, smi_to_inchi, smi_to_inchikey):
    # the rename is there because some datasets use smiles and others SMILES
    # and also I prefer lowercase
    def process_dataset(df:pl.DataFrame)-> pl.DataFrame:
        tmp = df.rename({"SMILES":"smiles"}).with_columns(
            pl.col("smiles").map_elements(smi_to_inchikey).alias("inchikey"),
            pl.col("smiles").map_elements(smi_to_inchi).alias("inchi"),
            # pl.col("smiles").map_elements(smitosvg).alias("svg")
        )
        # tmp = generate_fingerprint(tmp, "ecfp", 
        #                     fp_size=4096, radius =3,
        #                     include_chirality=True, count=True)
        # tmp = add_tsne_columns(tmp)
        # tmp = add_umap_columns(tmp, metric="jaccard")
        return tmp


    return (process_dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    # Read and preprocess the data
    """)
    return


@app.cell
def _(pl, process_dataset):
    train = process_dataset(pl.read_csv("../data/raw/20260409/dose_response_train.csv"))
    test = process_dataset(pl.read_csv("../data/raw/20260409/dose_response_test.csv"))
    train_counter = process_dataset(pl.read_csv("../data/raw/20260409/counter_screen_train.csv"))
    train_single = process_dataset(pl.read_csv("../data/raw/20260409/single_dose_train.csv"))
    return test, train, train_counter, train_single


@app.cell
def _(mo):
    mo.md(r"""
    ## Pivot single-dose dataset by concentration

    Each molecule (identified by `inchikey`) may have been tested at multiple concentrations.
    Here we:
    1. Aggregate molecule identifiers: keep `smiles`, `inchi`, `inchikey` as unique entities,
       concatenate all `Molecule Name` and `OCNT Batch` values seen for that molecule.
    2. Add a boolean `is_hit` flag per row: `median_log2_fc > 1` AND `fdr_bh < 0.05`.
    3. Pivot so that each `concentration_uM` becomes its own set of columns, keeping
       the **median** of `median_log2_fc` and the `is_hit` flag (true if any replicate
       is a hit at that concentration).
    """)
    return


@app.cell
def _(pl, train_single):
    # ── Step 1: build the identifier lookup (smiles / inchi / inchikey) ──────────
    # train_single already has inchi and inchikey added by process_dataset()

    # Aggregate molecule-level metadata: one row per inchikey
    _mol_meta = (
        train_single
        .group_by("inchikey")
        .agg(
            # Take the first non-null smiles and inchi (all rows for the same
            # inchikey should be identical, so first() is fine)
            pl.col("smiles").first(),
            pl.col("inchi").first(),
            # Collect all unique Molecule Name values (nulls excluded) and join
            pl.col("Molecule Name")
              .drop_nulls()
              .unique()
              .sort()
              .str.join("|")
              .alias("molecule_names"),
            # Same for OCNT Batch
            pl.col("OCNT Batch")
              .drop_nulls()
              .unique()
              .sort()
              .str.join("|")
              .alias("ocnt_batches"),
        )
    )

    # Nominal concentration labels (µM): map messy float values to round numbers
    # so pivot column names are clean (e.g. "median_log2_fc_1" not "median_log2_fc_0.9203")
    _conc_labels = {99.01: 100, 33.0: 30, 8.251: 10, 0.9803: 1}

    # ── Step 2: add is_hit flag, then aggregate per (inchikey, concentration_uM) ───
    # is_hit: median_log2_fc > 1 AND fdr_bh < 0.05 at the replicate level.
    # After grouping, take the median log2fc and propagate hit status with any()
    # (a compound is a hit at that concentration if any replicate qualifies).
    _agg_values = (
        train_single
        .with_columns(
            (10**6 * pl.col("concentration_M")).round(4).alias("concentration_uM"),
            ((pl.col("median_log2_fc") > 1) & (pl.col("fdr_bh") < 0.05)).alias("is_hit"),
        )
        .with_columns(
            pl.col("concentration_uM")
              .replace(_conc_labels)
              .alias("concentration_uM")
        )
        .group_by(["inchikey", "concentration_uM"])
        .agg(
            pl.col("median_log2_fc").median(),
            pl.col("is_hit").any(),
        )
    )

    # ── Step 3: pivot so each concentration becomes its own column ───────────────
    # Pivot the two value columns separately to preserve their native dtypes:
    # - median_log2_fc: "median" aggregation → float
    # - is_hit: "first" aggregation → boolean (already one row per group above;
    #   using "median" would silently cast bool to float 0.0/1.0)
    _pivoted_fc = _agg_values.pivot(
        on="concentration_uM",
        index="inchikey",
        values="median_log2_fc",
        aggregate_function="median",
    )
    _pivoted_hit = _agg_values.pivot(
        on="concentration_uM",
        index="inchikey",
        values="is_hit",
        aggregate_function="first",
    )
    # suffix="_is_hit" makes is_hit pivot columns like "100.0_is_hit";
    # rename the bare concentration columns from the fc pivot to match: "100.0_log2_fc"
    _pivoted = _pivoted_fc.join(_pivoted_hit, on="inchikey", how="left", suffix="_is_hit")
    _pivoted = _pivoted.rename(
        {c: f"{c}_log2_fc" for c in _pivoted.columns if c != "inchikey" and "_is_hit" not in c}
    )

    # ── Step 4: join molecule metadata back ──────────────────────────────────────
    train_single_pivoted: pl.DataFrame = _mol_meta.join(
        _pivoted, on="inchikey", how="left"
    )

    train_single_pivoted
    return (train_single_pivoted,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Enrich single-dose pivot with dose-response and counter-screen activity

    Join `pEC50` and `Emax_estimate` from:
    - **train** (primary dose-response assay) → suffixed `_dr`
    - **train_counter** (counter-screen assay) → suffixed `_counter`

    Both datasets are first reduced to one row per `inchikey` (median of the two
    activity columns) before joining, to guard against any duplicate records.
    """)
    return


@app.cell
def _(pl, train, train_counter, train_single_pivoted: "pl.DataFrame"):
    _emax_col = "Emax_estimate (log2FC vs. baseline)"

    # ── Dose-response: one row per inchikey ───────────────────────────────────────
    _dr = (
        train
        .group_by("inchikey")
        .agg(
            pl.col("smiles").first(),
            pl.col("inchi").first(),
            pl.col("Molecule Name").drop_nulls().unique().sort().str.join("|").alias("molecule_names"),
            pl.col("OCNT Batch").drop_nulls().unique().sort().str.join("|").alias("ocnt_batches"),
            pl.col("pEC50").median().alias("pEC50_dr"),
            pl.col(_emax_col).median().alias("Emax_dr"),
        )
    )

    # ── Counter-screen: one row per inchikey ──────────────────────────────────────
    _counter = (
        train_counter
        .group_by("inchikey")
        .agg(
            pl.col("smiles").first(),
            pl.col("inchi").first(),
            pl.col("Molecule Name").drop_nulls().unique().sort().str.join("|").alias("molecule_names"),
            pl.col("OCNT Batch").drop_nulls().unique().sort().str.join("|").alias("ocnt_batches"),
            pl.col("pEC50").median().alias("pEC50_counter"),
            pl.col(_emax_col).median().alias("Emax_counter"),
        )
    )

    # ── Join both onto the pivoted single-dose table ───────────────────────────────
    # full outer join so inchikeys only present in train/_counter are also included.
    # After each full join, use coalesce to fill nulls in metadata columns from
    # the right-hand side (rows new to this join have nulls on the left).
    _meta_cols = ["smiles", "inchi", "molecule_names", "ocnt_batches"]

    train_single_enriched = (
        train_single_pivoted
        .join(_dr, on="inchikey", how="full", coalesce=True, suffix="_dr_r")
        .with_columns([
            pl.coalesce([c, f"{c}_dr_r"]).alias(c) for c in _meta_cols
        ])
        .drop([f"{c}_dr_r" for c in _meta_cols])
        .join(_counter, on="inchikey", how="full", coalesce=True, suffix="_ctr_r")
        .with_columns([
            pl.coalesce([c, f"{c}_ctr_r"]).alias(c) for c in _meta_cols
        ])
        .drop([f"{c}_ctr_r" for c in _meta_cols])
    )

    train_single_enriched
    return (train_single_enriched,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Full chemical space

    Combine all four datasets into one table, adding boolean membership flags:
    - `in_single_dose` — compound was in the single-dose screen
    - `in_dose_response` — compound was in the dose-response screen
    - `in_counter` — compound was in the counter-screen
    - `in_test` — compound is in the blinded test set
    """)
    return


@app.cell
def _(pl, test, train, train_counter, train_single_enriched):
    _meta_cols = ["smiles", "inchi", "molecule_names"]

    # ── Collect the inchikey sets for membership flags ────────────────────────────
    _single_keys  = set(train_single_enriched.get_column("inchikey").drop_nulls().to_list())
    _dr_keys      = set(train.get_column("inchikey").drop_nulls().to_list())
    _counter_keys = set(train_counter.get_column("inchikey").drop_nulls().to_list())
    _test_keys    = set(test.get_column("inchikey").drop_nulls().to_list())

    # ── Slim test table: one row per inchikey with metadata only ──────────────────
    _test_meta = (
        test
        .group_by("inchikey")
        .agg(
            pl.col("smiles").first(),
            pl.col("inchi").first(),
            pl.col("Molecule Name").drop_nulls().unique().sort().str.join("|").alias("molecule_names"),

        )
    )

    # ── Full outer join of test onto the enriched table ───────────────────────────
    _full = (
        train_single_enriched
        .join(_test_meta, on="inchikey", how="full", coalesce=True, suffix="_tst_r")
        .with_columns([
            pl.coalesce([c, f"{c}_tst_r"]).alias(c) for c in _meta_cols
        ])
        .drop([f"{c}_tst_r" for c in _meta_cols])
    )

    # ── Add boolean membership flags ──────────────────────────────────────────────
    all_compounds = _full.with_columns(
        pl.col("inchikey").is_in(_single_keys).alias("in_single_dose"),
        pl.col("inchikey").is_in(_dr_keys).alias("in_dose_response"),
        pl.col("inchikey").is_in(_counter_keys).alias("in_counter"),
        pl.col("inchikey").is_in(_test_keys).alias("in_test"),
    )

    all_compounds
    return (all_compounds,)


@app.cell
def _(all_compounds):
    all_compounds.write_csv("../data/processed/all_compounds_activity_data.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Plot and explore chemical space
    """)
    return


@app.cell
def _(add_tsne_columns, add_umap_columns, generate_fingerprint, pl):
    def prepare_for_plotting(df: pl.DataFrame) -> pl.DataFrame:
        df = generate_fingerprint(df, "ecfp", 
                            fp_size=4096, radius =3,
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
          .then(pl.lit("#1a7a4a"))           # dark green  — test set
          .when(pl.col("in_counter"))
          .then(pl.lit("#1a3a6b"))           # dark blue   — counter screen
          .when(pl.col("in_dose_response"))
          .then(pl.lit("#6baed6"))           # light blue  — dose response
          .otherwise(pl.lit("#c0c0c080"))    # light grey 50% transparent — single dose only
          .alias("color")
    )
    plot_df_colored
    return (plot_df_colored,)


@app.cell
def _(generate_embedding_plot, plot_df_colored):
    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    generate_embedding_plot(plot_df_colored, x_col="UMAP_x", y_col="UMAP_y",
                        color_col="color", point_size=6,
                        title="UMAP ECFP6 Chemical Space",
                        color_legend=_legend,
                        save_path="../plots/1_sar_exploration/umap_ecfp6_chemical_space.png")
    return


@app.cell
def _(generate_embedding_plot, plot_df_colored):
    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    generate_embedding_plot(plot_df_colored, x_col="TSNE_x", y_col="TSNE_y",
                        color_col="color", point_size=6,
                        title="tSNE ECFP6 Chemical Space",
                        color_legend=_legend,
                        save_path="../plots/1_sar_exploration/tsne_ecfp6_chemical_space.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Compute MMPs
    """)
    return


@app.cell
def _(Chem, all_compounds, pl):
    # Canonicalise SMILES through RDKit before writing the mmpdb input file.
    # This strips CXSMILES extensions (e.g. atom-map notation "|&1:10,17|")
    # that Polars would write as extra space-separated tokens, causing mmpdb
    # to misparse the title column and raise a UNIQUE constraint error in SQLite.
    _canonical = all_compounds.with_columns(
        pl.col("smiles")
          .map_elements(
              lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)),
              return_dtype=pl.Utf8,
          )
          .alias("smiles_canonical")
    )

    _canonical.select(["smiles_canonical", "inchikey"]).write_csv(
        "../data/processed/all_compounds_mmp.smi",
        separator=" ",
        include_header=False,
        quote_style="never",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compute Matched Molecular Pairs (MMP)

    Two steps using [mmpdb](https://github.com/rdkit/mmpdb):

    1. **fragment** — breaks each molecule into core/substituent pairs at single
       rotatable bonds; writes a `.frag` file.
    2. **index** — finds all pairs of fragments that share the same core (constant
       part) but differ in one substituent (variable part); writes a `.csv.gz`
       for easier reading into the notebook.

    Both commands are run as subprocesses so the notebook is self-contained and
    the full processing pipeline can be reproduced end-to-end.
    """)
    return


@app.cell
def _(Path, mo, subprocess, sys):
    _smi  = "../data/processed/all_compounds_mmp.smi"
    _frag = "../data/processed/all_compounds_mmp.frag"

    if Path(_frag).exists():
        mo.md(f"**fragment** skipped — `{_frag}` already exists")
    else:
        _result = subprocess.run(
            [sys.executable, "-m", "mmpdblib", "fragment", _smi, "-o", _frag],
            capture_output=True,
            text=True,
        )

        if _result.returncode != 0:
            raise RuntimeError(f"mmpdb fragment failed:\n{_result.stderr}")

        mo.md(f"**fragment** finished — output: `{_frag}`")
    return


@app.cell
def _(Path, mo, subprocess, sys):
    _frag  = "../data/processed/all_compounds_mmp.frag"
    _mmpdb = "../data/processed/all_compounds_mmp.mmp.csv.gz"

    if Path(_mmpdb).exists():
        mo.md(f"**index** skipped — `{_mmpdb}` already exists")
    else:
        _result = subprocess.run(
            [sys.executable, "-m", "mmpdblib", "index", _frag, "-out csv.gz", "-o", _mmpdb],
            capture_output=True,
            text=True,
        )

        if _result.returncode != 0:
            raise RuntimeError(f"mmpdb index failed:\n{_result.stderr}")

        mo.md(f"**index** finished — MMP database: `{_mmpdb}`")
    return


@app.cell
def _(pl):
    mmp_df = pl.read_csv("../data/processed/all_compounds_mmp.mmp.csv.gz", separator="\t",
                            has_header=False, new_columns=["smiles1","smiles2", "ID1",
                                                        "ID2", "transform", "core" ])
    mmp_df
    return (mmp_df,)


@app.cell
def _(mo):
    mo.md(r"""
    Once we have all computed MMPs it is normally useful to filter out some of them,
    based on our definition of what is considered a valid transformation.
    Here the main filtering step is to only allow MMPs when the variable section (frag) is
    at most 50% as large as the common section (core).
    Another typical filter is to restrict the size of the variable section to smaller fragments,
    but the limit already set by mmdb seems sensible here.
    Notice we lose the majority of MMPs (from 68K to 2.6K) and that is fine.
    It is similar idea of varying the similarity threshold value when using fingerprints.
    """)
    return


@app.cell
def _(Chem, mmp_df, pl):
    # Compute all heavy-atom counts in a single Python pass over the rows,
    # parsing each SMILES only once — avoids 5 separate map_elements scans
    # over 68K rows which was taking ~10 minutes.
    _n_ha1, _n_ha2, _n_ha_core, _n_ha_frag1, _n_ha_frag2 = [], [], [], [], []
    for _s1, _s2, _core, _transform in mmp_df.select(
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
        mmp_df
        .with_columns(
            pl.Series("n_ha1",    _n_ha1,    dtype=pl.Int32),
            pl.Series("n_ha2",    _n_ha2,    dtype=pl.Int32),
            pl.Series("n_ha_core",_n_ha_core,dtype=pl.Int32),
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
def _(generate_similarity_network, mmp_df_filtered, pl, plot_df_colored):
    #I was having issues with the image being larger than the 10MB marimo normal limit for output
    #mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 100_000_000

    # Step 1: candidate nodes — dose-response, counter-screen, or test only
    _candidate_keys = set(
        plot_df_colored
        .filter(pl.col("in_dose_response") | pl.col("in_counter") | pl.col("in_test"))
        .get_column("inchikey")
        .to_list()
    )

    # Step 2: keep only edges where BOTH endpoints are candidates
    _edges = mmp_df_filtered.filter(
        pl.col("ID1").is_in(_candidate_keys) & pl.col("ID2").is_in(_candidate_keys)
    )

    # Step 3: keep only nodes that have at least one edge — isolated nodes have
    # no forces in FR and bloat the layout with random scatter
    _connected_keys = set(_edges["ID1"].to_list()) | set(_edges["ID2"].to_list())
    _network_df = plot_df_colored.filter(pl.col("inchikey").is_in(_connected_keys))

    print(f"Nodes: {len(_network_df)}, Edges: {len(_edges)}")

    _legend = {
        "#c0c0c080": "Single dose only",
        "#6baed6":   "Dose response",
        "#1a3a6b":   "Counter screen",
        "#1a7a4a":   "Test set",
    }
    #The interactive version of this plot (and most others in this notebook) is very heavy
    #but I ran it once during DEA to have a brief look at the structures
    #mo._runtime.context.get_context().marimo_config["runtime"]["output_max_bytes"] = 30_000_000
    generate_similarity_network(_network_df, "inchikey",
            edges_df=_edges, node_size=16, layout_iterations=50, property_col="color",
            max_edges=15000, color_legend=_legend, for_marimo=False,
            save_path="../plots/1_sar_exploration/mmp_network.png")
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Activity cliff analysis

    Activity cliffs (ACs) are pairs of molecules that are very similar
    but have large differences in their activity.
    They are challenging to predict for many general ML models,
    as they break the assumption that similar molecules will have similar activity.
    In general, datasets with large number of ACs are more difficult to model.
    """)
    return


@app.cell
def _(generate_embedding_plot, pl, plot_df):
    # Filter to training compounds with pEC50 dose-response data only, then
    # drop the pre-computed embedding columns so generate_embedding_plot
    # recomputes UMAP and t-SNE on this subset alone.
    _dr_df = (
        plot_df
        .filter(pl.col("pEC50_dr").is_not_null())
        .drop(["UMAP_x", "UMAP_y", "TSNE_x", "TSNE_y"])
    )

    generate_embedding_plot(
        _dr_df,
        x_col="UMAP_x",
        y_col="UMAP_y",
        color_col="pEC50_dr",
        x_title="UMAP 1",
        y_title="UMAP 2",
        method="umap",
        umap_metric="jaccard",
        title="UMAP Chemical Space — Training pEC50 (dose-response)",
        point_size=20,
        alpha=0.85,
        cmap="viridis",
        for_marimo=False,
        save_path="../plots/1_sar_exploration/umap_pec50_chemical_space.png",
    )
    return


@app.cell
def _(mo):
    mo.md(r"""
    One way to compute ACs is based on MMPs, this can be useful because you can always point
    to a single transformation (either of the central section or a terminal section) of the molecule.
    In this dataset, as we have seen there is generally low propensity to MMP formation due to the
    large amount of chemical diversity so the number of MMP cliffs is small.
    """)
    return


@app.cell
def _(all_compounds, mmp_df_filtered, pl):
    # Build a lookup: inchikey → pEC50_dr (training dose-response only)
    _pec50 = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "pEC50_dr"])
    )

    # Join pEC50 for both endpoints of each MMP pair, then keep only pairs
    # where both compounds were tested in the dose-response assay and their
    # pEC50 values differ by at least 2 log units (activity cliff threshold).
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
    Another way is to define a similarity threshold value to consider a pair of compounds
    similar.
    It is important to note that the threshold value depends both on the similarity metric and
     the molecular representation used.
    For example, for Tanimoto similarity using ECFP4 a threshold value of 0.35 or 0.4 is appropriate
    while for MACCS you would need to increase it to 0.8.
    """)
    return


@app.cell
def _(
    all_compounds,
    compute_pairwise_similarities,
    pl,
    plot_similarity_distributions,
):
    # Dose-response compounds only — one row per unique inchikey
    _dr_df = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )

    # MACCS keys (167-bit, no extra parameters needed)
    _sim_maccs = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="maccs", metric="tanimoto",
    )

    # ECFP4: radius=2, 1024-bit, chirality-aware
    _sim_ecfp4 = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto",
        fp_size=1024, radius=2, include_chirality=True,
    )

    # ECFP6: radius=3, 4096-bit, chirality-aware
    _sim_ecfp6 = compute_pairwise_similarities(
        _dr_df, id_col="inchikey", fingerprint="ecfp", metric="tanimoto",
        fp_size=4096, radius=3, include_chirality=True,
    )

    # Label ECFP variants by their common names before concatenating
    _sim_ecfp4 = _sim_ecfp4.with_columns(pl.lit("ecfp4").alias("fingerprint"))
    _sim_ecfp6 = _sim_ecfp6.with_columns(pl.lit("ecfp6").alias("fingerprint"))

    pairwise_similarities = pl.concat([_sim_maccs, _sim_ecfp4, _sim_ecfp6])

    # Overlay KDE curves for all three fingerprints on a single axes for direct comparison
    plot_similarity_distributions(
        pairwise_similarities,
        group_col="fingerprint",
        group_order=["maccs", "ecfp4", "ecfp6"],
        colors=["#4e79a7", "#f28e2b", "#59a14f"],
        title="Pairwise similarity distributions — dose-response compounds",
        save_path="../plots/1_sar_exploration/density_sim_fingerprints.png",
    )
    return (pairwise_similarities,)


@app.cell
def _(np, pairwise_similarities, pl):
    _fp_order = ["maccs", "ecfp4", "ecfp6"]

    _rows = []
    for _fp in _fp_order:
        _vals = (
            pairwise_similarities
            .filter(pl.col("fingerprint") == _fp)
            .get_column("similarity")
            .to_numpy()
        )
        _mean   = float(np.mean(_vals))
        _median = float(np.median(_vals))
        _q25    = float(np.percentile(_vals, 25))
        _q75    = float(np.percentile(_vals, 75))
        _rows.append({
            "fingerprint":    _fp.upper(),
            "mean":           round(_mean,            3),
            "median":         round(_median,          3),
            "Q25":            round(_q25,             3),
            "Q75":            round(_q75,             3),
            "IQR":            round(_q75 - _q25,      3),
            "p95":  round(float(np.percentile(_vals, 95)), 3),
            "p99": round(float(np.percentile(_vals, 99)), 3),
        })

    pl.DataFrame(_rows)
    return


@app.cell
def _(pairwise_similarities, pl):
    pairwise_similarities.filter(
        (pl.col("fingerprint") == "maccs") & (pl.col("similarity") >= 0.8)
    ).height
    return


@app.cell
def _(pairwise_similarities, pl):
    pairwise_similarities.filter(
        (pl.col("fingerprint") == "ecfp4") & (pl.col("similarity") >= 0.4)
    ).height
    return


@app.cell
def _(all_compounds, pairwise_similarities, pl):
    # pEC50 lookup: one row per inchikey from the dose-response training set
    _pec50 = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "pEC50_dr"])
    )

    # Join pEC50 onto both ends of every similar pair, then count cliffs
    _sim_with_delta = (
        pairwise_similarities
        .join(_pec50.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1"}), on="ID1", how="inner")
        .join(_pec50.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2"}), on="ID2", how="inner")
        .with_columns(
            (pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50")
        )
    )

    _thresholds = {"maccs": 0.8, "ecfp4": 0.4}

    _rows = []
    for _fp, _sim_thresh in _thresholds.items():
        _total_similar = (
            _sim_with_delta
            .filter(
                (pl.col("fingerprint") == _fp)
                & (pl.col("similarity") >= _sim_thresh)
            )
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
            "fingerprint":        _fp.upper(),
            "similarity threshold": _sim_thresh,
            "similar pairs":      _total_similar,
            "activity cliffs":    _n_cliffs,
            "cliff fraction (%)": round(100 * _n_cliffs / _total_similar, 1) if _total_similar > 0 else 0.0,
        })

    pl.DataFrame(_rows)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Activity cliff scatter plot (ECFP4, Tanimoto ≥ 0.4)

    Each point is an **activity cliff**: a pair with Tanimoto similarity ≥ 0.4
    (ECFP4) and |ΔpEC50| ≥ 2 log units (dose-response assay).
    The x-axis shows the pairwise similarity and the y-axis shows the activity
    difference. Hover over a point to see both molecules drawn side by side.
    """)
    return


@app.cell
def _(
    Chem, CombineMols, all_compounds, alt, base64, mo, pairwise_similarities,
    pl, rdDepictor, rdMolDraw2D,
):
    def _pair_smiles_to_base64_png(
        smi1: str,
        smi2: str,
        width: int = 300,
        height: int = 150,
    ) -> str:
        """
        Render two molecules side by side as a base64-encoded PNG data URI.

        Uses ``Chem.CombineMols`` to safely join the two molecule objects before
        drawing — this avoids the ambiguity of concatenating SMILES strings with
        a dot when either input already contains disconnected fragments (salts,
        counterions) whose SMILES also use dots.

        Args:
            smi1: SMILES of the first molecule.
            smi2: SMILES of the second molecule.
            width: Image width in pixels (shared across both structures).
            height: Image height in pixels.

        Returns:
            Data URI string ``"data:image/png;base64,<bytes>"``, or ``""`` on
            parse failure.
        """
        mol1 = Chem.MolFromSmiles(smi1)
        mol2 = Chem.MolFromSmiles(smi2)
        if mol1 is None or mol2 is None:
            return ""

        # CombineMols merges two RDKit Mol objects into one disconnected molecule
        # without any SMILES re-parsing, so dots already in smi1/smi2 are irrelevant.
        combined = CombineMols(mol1, mol2)
        rdDepictor.Compute2DCoords(combined)

        drawer = rdMolDraw2D.MolDraw2DCairo(width, height)
        drawer.DrawMolecule(combined)
        drawer.FinishDrawing()
        png_bytes = drawer.GetDrawingText()
        encoded = base64.b64encode(png_bytes).decode("ascii")
        return f"data:image/png;base64,{encoded}"

    # ── Build activity cliff pairs only ─────────────────────────────────────────
    _CLIFF_THRESHOLD = 2.0
    _SIM_THRESHOLD   = 0.4

    # pEC50 lookup: one row per inchikey with smiles and molecule name
    _pec50_smiles = (
        all_compounds
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["inchikey", "smiles", "pEC50_dr", "molecule_names"])
        .unique(subset=["inchikey"])
    )

    # Filter to activity cliffs only: similar (≥ 0.4) AND large activity gap (≥ 2 pEC50 units)
    _ecfp4_cliffs = (
        pairwise_similarities
        .filter(
            (pl.col("fingerprint") == "ecfp4")
            & (pl.col("similarity") >= _SIM_THRESHOLD)
        )
        .join(
            _pec50_smiles.rename({"inchikey": "ID1", "pEC50_dr": "pEC50_1", "smiles": "smiles1", "molecule_names": "name1"}),
            on="ID1", how="inner",
        )
        .join(
            _pec50_smiles.rename({"inchikey": "ID2", "pEC50_dr": "pEC50_2", "smiles": "smiles2", "molecule_names": "name2"}),
            on="ID2", how="inner",
        )
        .with_columns(
            (pl.col("pEC50_1") - pl.col("pEC50_2")).abs().alias("delta_pEC50"),
        )
        .filter(pl.col("delta_pEC50") >= _CLIFF_THRESHOLD)
    )

    # ── Build plot DataFrame ─────────────────────────────────────────────────────
    cliff_plot_df = _ecfp4_cliffs.drop(["fingerprint", "metric"]).with_columns(
        pl.max_horizontal("pEC50_1", "pEC50_2").alias("max_pEC50")
    )

    # ── Altair scatter with point selection ──────────────────────────────────────
    # empty=False: when nothing is clicked the selection matches no points,
    # so all points fall through to the viridis colour encoding (if_false branch).
    _sel = alt.selection_point(fields=["ID1", "ID2"], name="cliff_sel", empty=False)

    _scatter = (
        alt.Chart(cliff_plot_df)
        .mark_circle(size=60, opacity=0.8)
        .encode(
            x=alt.X(
                "similarity:Q",
                title="Tanimoto similarity (ECFP4)",
                scale=alt.Scale(domain=[0.39, 1.0]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            y=alt.Y(
                "delta_pEC50:Q",
                title="|ΔpEC50| (dose-response)",
                scale=alt.Scale(domain=[1.9, 4.0]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),
                alt.Color(
                    "max_pEC50:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(title="max pEC50"),
                ),
            ),
            size=alt.condition(_sel, alt.value(120), alt.value(60)),
            tooltip=[
                alt.Tooltip("name1:N",       title="Molecule 1"),
                alt.Tooltip("name2:N",       title="Molecule 2"),
                alt.Tooltip("similarity:Q",  title="Tanimoto",   format=".3f"),
                alt.Tooltip("pEC50_1:Q",     title="pEC50 mol1", format=".2f"),
                alt.Tooltip("pEC50_2:Q",     title="pEC50 mol2", format=".2f"),
                alt.Tooltip("delta_pEC50:Q", title="|ΔpEC50|",   format=".2f"),
                alt.Tooltip("max_pEC50:Q",  title="max pEC50",  format=".2f"),
            ],
        )
        .add_params(_sel)
        .properties(title="Activity cliffs — ECFP4 Tanimoto ≥ 0.4, |ΔpEC50| ≥ 2", width=500, height=400)
        .configure_title(fontSize=12)
    )

    # Exported so the next cell can (a) read .value and (b) place it in hstack
    cliff_chart = mo.ui.altair_chart(_scatter)
    return cliff_chart, cliff_plot_df


@app.cell
def _(Chem, CombineMols, cliff_chart, cliff_plot_df, mo, pl, rdDepictor, rdMolDraw2D):
    _PANEL_W = 350   # panel width in px; SVG canvas matches this exactly

    def _pair_to_svg(smi1: str, smi2: str, width: int = _PANEL_W, height: int = 210) -> str:
        """Render two molecules side by side as an SVG string using CombineMols."""
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

    # ── Structure panel — reacts to the clicked point ────────────────────────────
    # cliff_chart.value is a Polars DataFrame of the currently selected rows.
    # This cell is re-run by marimo whenever the selection changes.
    _sel_rows = cliff_chart.value

    if _sel_rows is None or len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:400px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px'>
                Click a point to see the pair of structures
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

    # cliff_chart is passed here (not created here) so marimo allows rendering it
    # alongside the panel in the same hstack output
    mo.hstack([cliff_chart, _panel], align="start")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    If you explore a bit this plot you will come to see many pairs that are considered
    highly similar by ECFP4 and Tanimoto similarity but have large changes in their structure
    and might not be considered similar by a chemist.
    This is one of the limitation of any similarity metric.
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Exploration of test set
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Train / Test similarity comparison

    To understand how representative the training set is of the test set, we
    compare three ECFP4 Tanimoto similarity distributions:

    - **Within train** — all unique pairs among dose-response training compounds
    - **Within test**  — all unique pairs among test-set compounds
    - **Train vs test** — every (train, test) pair (cross-set, no self-comparisons)

    A large gap between "within train" and "train vs test" would indicate that
    the test set probes chemical space not well covered by the training data,
    which is a known challenge for generalisation.
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

    # ── Prepare deduplicated train and test sets ──────────────────────────────────
    # Training set: dose-response compounds, one row per unique inchikey
    _train_df = (
        all_compounds
        .filter(pl.col("in_dose_response"))
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )

    # Test set: one row per unique inchikey
    _test_df = (
        test
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles"])
    )

    # ── ECFP4 parameters (radius=2, 1024-bit, chirality-aware) ───────────────────
    _FP_KWARGS = {"fp_size": 1024, "radius": 2, "include_chirality": True}

    # ── Within-train pairwise similarities ───────────────────────────────────────
    _sim_train = (
        compute_pairwise_similarities(
            _train_df, id_col="inchikey", fingerprint="ecfp",
            metric="tanimoto", **_FP_KWARGS,
        )
        .with_columns(pl.lit("within_train").alias("comparison"))
    )

    # ── Within-test pairwise similarities ────────────────────────────────────────
    _sim_test = (
        compute_pairwise_similarities(
            _test_df, id_col="inchikey", fingerprint="ecfp",
            metric="tanimoto", **_FP_KWARGS,
        )
        .with_columns(pl.lit("within_test").alias("comparison"))
    )

    # ── Cross-set similarities (every train–test pair) ───────────────────────────
    # compute_pairwise_similarities handles within-set pairs only, so we compute
    # the cross-set matrix directly using the same RDKit bulk approach.

    def _row_to_bitvect(row: np.ndarray):
        """Convert a binary uint8 numpy fingerprint row to an RDKit ExplicitBitVect."""
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.where(row > 0)[0].tolist():
            bv.SetBit(bit)
        return bv

    # Generate ECFP4 for both sets (column named "ecfp")
    _train_fp = generate_fingerprint(_train_df, "ecfp", **_FP_KWARGS)
    _test_fp  = generate_fingerprint(_test_df,  "ecfp", **_FP_KWARGS)

    _train_arr = np.vstack(_train_fp["ecfp"].to_list())
    _test_arr  = np.vstack(_test_fp["ecfp"].to_list())

    _train_bvs = [_row_to_bitvect(_train_arr[i]) for i in range(len(_train_arr))]
    _test_bvs  = [_row_to_bitvect(_test_arr[i])  for i in range(len(_test_arr))]

    # For each train molecule, compute its Tanimoto to every test molecule
    _cross_sims: list[float] = []
    for _bv in _train_bvs:
        _cross_sims.extend(DataStructs.BulkTanimotoSimilarity(_bv, _test_bvs))

    _sim_cross = pl.DataFrame({
        "similarity":  pl.Series(_cross_sims, dtype=pl.Float32),
        "comparison":  pl.Series(["train_vs_test"] * len(_cross_sims), dtype=pl.Utf8),
    })

    # ── Combine and plot ──────────────────────────────────────────────────────────
    _sim_combined = pl.concat([
        _sim_train.select(["similarity", "comparison"]),
        _sim_test.select(["similarity", "comparison"]),
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

    The scatter shows the test compounds projected into 2D via **UMAP** (jaccard
    metric on ECFP4), coloured by their nearest-neighbour similarity to the
    training set.  Click a point to inspect the test compound and its closest
    training-set neighbour side by side.
    """)
    return


@app.cell
def _(DataStructs, UMAP, all_compounds, generate_fingerprint, np, pl, test):

    # ── ECFP4 parameters used throughout this section ────────────────────────────
    _ECFP4 = {"fp_size": 1024, "radius": 2, "include_chirality": True}

    # ── Deduplicated train and test sets ─────────────────────────────────────────
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

    # ── Generate ECFP4 fingerprints for both sets ─────────────────────────────────
    _train_fp = generate_fingerprint(_train_meta, "ecfp", **_ECFP4)
    _test_fp  = generate_fingerprint(_test_meta,  "ecfp", **_ECFP4)

    _train_arr = np.vstack(_train_fp["ecfp"].to_list())   # (n_train, 1024)
    _test_arr  = np.vstack(_test_fp["ecfp"].to_list())    # (n_test,  1024)

    # ── Convert to RDKit ExplicitBitVect for bulk Tanimoto ────────────────────────
    def _to_bv(row: np.ndarray):
        """Convert binary uint8 numpy row to RDKit ExplicitBitVect."""
        bv = DataStructs.ExplicitBitVect(int(row.shape[0]))
        for bit in np.where(row > 0)[0].tolist():
            bv.SetBit(bit)
        return bv

    _train_bvs = [_to_bv(_train_arr[i]) for i in range(len(_train_arr))]
    _test_bvs  = [_to_bv(_test_arr[i])  for i in range(len(_test_arr))]

    # ── For each test compound: max similarity + index of nearest train compound ──
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

    # ── UMAP embedding of test compounds (jaccard ≡ Tanimoto for binary FPs) ─────
    _umap = UMAP(
        n_components=2,
        n_neighbors=min(15, len(_test_arr) - 1),
        min_dist=0.1,
        metric="jaccard",
        random_state=42,
    )
    _umap_coords = _umap.fit_transform(_test_arr)   # (n_test, 2)

    # ── Assemble the plot DataFrame ───────────────────────────────────────────────
    test_coverage_df = pl.DataFrame({
        "inchikey":         _test_meta["inchikey"].to_list(),
        "smiles":           _test_meta["smiles"].to_list(),
        "molecule_names":   _test_meta["Molecule Name"].to_list(),
        "UMAP_x":           _umap_coords[:, 0].tolist(),
        "UMAP_y":           _umap_coords[:, 1].tolist(),
        "max_sim_to_train": pl.Series(_max_sims, dtype=pl.Float32),
        # Nearest-neighbour info from the training set
        "nn_inchikey":      [_train_ids[i]    for i in _nn_idx],
        "nn_smiles":        [_train_smiles[i] for i in _nn_idx],
        "nn_name":          [_train_names[i]  for i in _nn_idx],
    })

    test_coverage_df
    return (test_coverage_df,)


@app.cell
def _(alt, mo, test_coverage_df):
    # ── Altair scatter — test compounds in UMAP space coloured by max similarity ──
    # empty=False: nothing selected → all points use the viridis colour encoding.
    _sel = alt.selection_point(fields=["inchikey"], name="test_sel", empty=False)

    _scatter = (
        alt.Chart(test_coverage_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X(
                "UMAP_x:Q",
                title="UMAP 1",
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            y=alt.Y(
                "UMAP_y:Q",
                title="UMAP 2",
                axis=alt.Axis(titleFontSize=13, labelFontSize=11),
            ),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),          # bright gold for the selected point
                alt.Color(
                    "max_sim_to_train:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(title="Max similarity\nto train"),
                ),
            ),
            size=alt.condition(_sel, alt.value(120), alt.value(60)),
            tooltip=[
                alt.Tooltip("molecule_names:N", title="Name"),
                alt.Tooltip("inchikey:N",        title="InChIKey"),
                alt.Tooltip("max_sim_to_train:Q", title="Max sim to train", format=".3f"),
                alt.Tooltip("nn_name:N",          title="Nearest train compound"),
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

    # Export so the next cell can read .value
    test_coverage_chart = mo.ui.altair_chart(_scatter)
    return (test_coverage_chart,)


@app.cell
def _(Chem, mo, pl, rdDepictor, rdMolDraw2D, test_coverage_chart, test_coverage_df):
    _PANEL_W  = 300   # width of each molecule panel (px); panels are stacked vertically
    _PANEL_H  = 200   # height of each individual molecule panel (px)

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

    # ── Structure panel — reacts to clicked point ────────────────────────────────
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

        # Trim the XML declaration that rdMolDraw2DSVG prepends so we can embed
        # both SVGs inline without duplicate declarations.
        def _strip_xml(svg: str) -> str:
            return svg.split("?>", 1)[-1].strip() if "?>" in svg else svg

        _name_test = _r["molecule_names"] or _r["inchikey"]
        _name_nn   = _r["nn_name"]        or _r["nn_inchikey"]

        # Test compound on top, nearest training compound below
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
    # Ring system and partial scaffold analysis

    In this analysis I plan to use RDKit to decompose a molecule into a collection of substructures
    from individual rings to partial scaffold segments.
    This provide a rich view of the structural variety in the dataset.
    The interesting section tends to be looking for such fragments that are large and are present
    in a large number of compound.
    That can be a way to computationally analyse an unknown or not well characterised dataset in search
    for analog series.
    """)
    return


@app.cell
def _(Chem, Optional, defaultdict, deque, itertools, pl):
    # ── Internal helpers ──────────────────────────────────────────────────────────

    def _build_ring_systems(mol: Chem.Mol) -> list[frozenset[int]]:
        """
        Return one frozenset of atom indices per ring system.

        A ring system is the union of all SSSR rings that share at least one atom
        (i.e. fused, bridged or spiro rings merge into one system).  Each isolated
        single ring is its own system.
        """
        atom_rings = mol.GetRingInfo().AtomRings()
        if not atom_rings:
            return []

        n = len(atom_rings)
        adj: dict[int, set[int]] = defaultdict(set)
        for i, j in itertools.combinations(range(n), 2):
            if set(atom_rings[i]) & set(atom_rings[j]):   # shared atom → same system
                adj[i].add(j)
                adj[j].add(i)

        visited = [False] * n
        systems: list[frozenset[int]] = []
        for start in range(n):
            if visited[start]:
                continue
            component: list[int] = []
            queue = deque([start])
            while queue:
                node = queue.popleft()
                if visited[node]:
                    continue
                visited[node] = True
                component.append(node)
                queue.extend(adj[node])
            systems.append(frozenset(a for ri in component for a in atom_rings[ri]))
        return systems

    def _linker_atoms(
        mol_adj: dict[int, set[int]],
        rs1: frozenset[int],
        rs2: frozenset[int],
        non_ring: set[int],
    ) -> Optional[frozenset[int]]:
        """
        Find the set of non-ring atoms that form the shortest path between two
        ring systems.

        The search starts at every ring-1 atom, crosses only non-ring atoms, and
        stops as soon as it reaches a ring-2 atom.  A direct ring–ring bond (zero
        linker atoms) returns an empty frozenset.  Returns None when the two
        systems are not connected through non-ring atoms at all.
        """
        # Direct ring-ring bond → zero-atom linker
        for a in rs1:
            if any(nb in rs2 for nb in mol_adj[a]):
                return frozenset()

        # BFS: start from rs1 boundary, traverse non-ring atoms, stop at rs2
        prev: dict[int, int] = {}
        queue = deque()
        for a in rs1:
            for nb in mol_adj[a]:
                if nb in non_ring and nb not in prev:
                    prev[nb] = a
                    queue.append(nb)

        while queue:
            cur = queue.popleft()
            if cur in rs2:
                # Trace back and collect only the non-ring atoms
                path: list[int] = []
                node = cur
                while node not in rs1:
                    if node in non_ring:
                        path.append(node)
                    node = prev[node]
                return frozenset(path)
            for nb in mol_adj[cur]:
                if nb not in prev and (nb in non_ring or nb in rs2):
                    prev[nb] = cur
                    queue.append(nb)
        return None

    def _to_canonical(mol: Chem.Mol, atom_set: frozenset[int]) -> Optional[str]:
        """
        Canonical SMILES for an atom subset; None on failure or empty set.

        Aromatic ring fragments extracted with ``MolFragmentToSmiles`` lose their
        ring context, so RDKit cannot re-perceive aromaticity and returns broken
        lowercase SMILES (e.g. ``c1nncn1``) that fail a subsequent
        ``MolFromSmiles`` call.  The fix is to Kekulize the parent molecule first
        so that ``MolFragmentToSmiles`` emits explicit alternating bond notation
        (e.g. ``C1=NN=CN=1``), which ``MolFromSmiles`` can always parse and then
        re-aromatize correctly.
        """
        if not atom_set:
            return None
        try:
            mol_kek = Chem.RWMol(mol)
            Chem.Kekulize(mol_kek, clearAromaticFlags=False)
            smi = Chem.MolFragmentToSmiles(mol_kek, sorted(atom_set), kekuleSmiles=True)
            if not smi:
                return None
            m = Chem.MolFromSmiles(smi)
            return Chem.MolToSmiles(m) if m is not None else None
        except Exception:
            return None

    # ── Public function ───────────────────────────────────────────────────────────

    def decompose_scaffold_network(smiles: list[str]) -> pl.DataFrame:
        """
        Decompose each molecule into three exhaustive, non-redundant levels of
        ring-based scaffolds.

        Every output scaffold contains only atoms that belong to rings or to bonds
        directly connecting two ring systems (linkers).  No dangling substituents
        or partial terminal fragments are included.

        **Scaffold types**

        ``ring_system``
            A fused or bridged polycyclic ring system (or a single isolated ring)
            with all substituents removed.  Each disconnected ring system in the
            molecule produces exactly one entry.  The system is the union of all
            SSSR rings that share at least one atom.

        ``linked_ring_systems``
            A pair of ring systems connected by a linker.  The linker is the
            shortest path through non-ring atoms between the two systems (zero
            atoms when the rings are directly bonded).  One entry is produced for
            every pair of ring systems that are connected in the molecular graph
            through non-ring atoms or via a direct ring–ring bond.

        ``full_scaffold``
            All ring systems in the molecule plus all linkers connecting them,
            assembled into a single scaffold.  Only produced when the molecule
            contains three or more ring systems that are all mutually reachable.
            When all ring systems form a single connected component this is the
            Murcko scaffold core.

        Args:
            smiles: Input SMILES strings.  Invalid SMILES are skipped and
                recorded with ``scaffold_type=null`` and a message in
                ``parse_error``.

        Returns:
            Long-format Polars DataFrame with columns:

            - ``smiles``              : original input SMILES (Utf8)
            - ``scaffold_smiles``     : canonical SMILES of the scaffold (Utf8),
                                        null on parse failure
            - ``scaffold_type``       : ``"ring_system"``,
                                        ``"linked_ring_systems"``, or
                                        ``"full_scaffold"`` (Utf8), null on failure
            - ``scaffold_heavy_atoms``: heavy-atom count of the scaffold (Int32),
                                        null on failure
            - ``parse_error``         : error message or null (Utf8)

            Molecules with no rings produce zero scaffold rows.
            All canonical SMILES are deduplicated per input molecule.
        """
        in_smi_col:    list[str]            = []
        sc_smi_col:    list[Optional[str]]  = []
        sc_type_col:   list[Optional[str]]  = []
        sc_ha_col:     list[Optional[int]]  = []
        err_col:       list[Optional[str]]  = []

        for smi in smiles:
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                in_smi_col.append(smi);  sc_smi_col.append(None)
                sc_type_col.append(None); sc_ha_col.append(None)
                err_col.append(f"Invalid SMILES: {smi!r}")
                continue

            ring_systems = _build_ring_systems(mol)
            if not ring_systems:
                continue   # acyclic molecule — no scaffold rows

            ring_atom_set: set[int] = set(a for rs in ring_systems for a in rs)
            non_ring: set[int] = set(range(mol.GetNumAtoms())) - ring_atom_set

            # Molecular adjacency list
            mol_adj: dict[int, set[int]] = defaultdict(set)
            for bond in mol.GetBonds():
                u, v = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                mol_adj[u].add(v)
                mol_adj[v].add(u)

            # Deduplicate scaffolds within this molecule with a dict {canonical_smi: type}
            seen: dict[str, str] = {}

            def _emit(atom_set: frozenset[int], stype: str) -> None:
                csmi = _to_canonical(mol, atom_set)
                if csmi and csmi not in seen:
                    seen[csmi] = stype

            # ── Level 1: individual ring systems ─────────────────────────────────
            for rs in ring_systems:
                _emit(rs, "ring_system")

            # ── Level 2: pairs of ring systems with their linker ─────────────────
            # Also build the linker map for later use in the full scaffold
            linker_map: dict[tuple[int, int], Optional[frozenset[int]]] = {}
            rs_connected: dict[int, set[int]] = defaultdict(set)  # adjacency in RS-graph

            for i, rs1 in enumerate(ring_systems):
                for j, rs2 in enumerate(ring_systems):
                    if j <= i:
                        continue
                    linker = _linker_atoms(mol_adj, rs1, rs2, non_ring)
                    linker_map[(i, j)] = linker
                    if linker is not None:
                        rs_connected[i].add(j)
                        rs_connected[j].add(i)
                        _emit(rs1 | rs2 | linker, "linked_ring_systems")

            # ── Level 3: full scaffold (≥ 3 ring systems, all connected) ─────────
            if len(ring_systems) >= 3:
                # Find the connected component(s) of the RS-graph
                rs_visited: list[bool] = [False] * len(ring_systems)
                for rs_start in range(len(ring_systems)):
                    if rs_visited[rs_start]:
                        continue
                    comp: list[int] = []
                    q = deque([rs_start])
                    while q:
                        node = q.popleft()
                        if rs_visited[node]:
                            continue
                        rs_visited[node] = True
                        comp.append(node)
                        q.extend(rs_connected[node])

                    if len(comp) >= 3:
                        all_atoms: frozenset[int] = frozenset(
                            a
                            for rs_idx in comp
                            for a in ring_systems[rs_idx]
                        )
                        for a, b in itertools.combinations(comp, 2):
                            lk = linker_map.get((min(a, b), max(a, b)))
                            if lk:
                                all_atoms = all_atoms | lk
                        _emit(all_atoms, "full_scaffold")

            # ── Flush collected scaffolds for this molecule ───────────────────────
            # csmi is always a valid canonical SMILES produced by _to_canonical,
            # so MolFromSmiles should never return None here.
            for csmi, stype in seen.items():
                sc_mol = Chem.MolFromSmiles(csmi)
                ha = sc_mol.GetNumHeavyAtoms() if sc_mol is not None else None
                in_smi_col.append(smi);       sc_smi_col.append(csmi)
                sc_type_col.append(stype);    sc_ha_col.append(ha)
                err_col.append(None)

        return pl.DataFrame({
            "smiles":               pl.Series(in_smi_col,  dtype=pl.Utf8),
            "scaffold_smiles":      pl.Series(sc_smi_col,  dtype=pl.Utf8),
            "scaffold_type":        pl.Series(sc_type_col, dtype=pl.Utf8),
            "scaffold_heavy_atoms": pl.Series(sc_ha_col,   dtype=pl.Int32),
            "parse_error":          pl.Series(err_col,     dtype=pl.Utf8),
        })

    return (decompose_scaffold_network,)


@app.cell
def _(all_compounds, decompose_scaffold_network, pl):
    # ── Fragment every unique molecule across all datasets ────────────────────────
    # One row per (inchikey, dataset membership flags) — deduplication avoids
    # running the expensive ring-perception step multiple times for the same structure.
    _unique_mols = (
        all_compounds
        .unique(subset=["inchikey"])
        .select(["inchikey", "smiles", "in_single_dose", "in_dose_response",
                 "in_counter", "in_test"])
        .drop_nulls(subset=["smiles"])
    )

    # decompose_scaffold_network takes a list of SMILES and returns long-format rows
    # with (smiles, scaffold_smiles, scaffold_type, scaffold_heavy_atoms).
    # We pass all unique SMILES at once so the function iterates only once.
    _scaffold_long = decompose_scaffold_network(
        _unique_mols.get_column("smiles").to_list()
    )

    # Join the dataset membership flags back onto the long-format scaffold table
    # using the original SMILES as the key.
    _membership = _unique_mols.select(
        ["smiles", "in_single_dose", "in_dose_response", "in_counter", "in_test"]
    )

    scaffold_hits = (
        _scaffold_long
        .filter(pl.col("parse_error").is_null())           # drop failed parses
        .join(_membership, on="smiles", how="left")
    )

    scaffold_hits
    return (scaffold_hits,)


@app.cell
def _(pl, scaffold_hits):
    # ── Aggregate: one row per unique scaffold ────────────────────────────────────
    # For each scaffold SMILES we count:
    #   n_total        — distinct molecules containing this scaffold (across all datasets)
    #   n_single_dose  — molecules in the single-dose screen
    #   n_dose_response— molecules in the dose-response screen
    #   n_counter      — molecules in the counter screen
    #   n_test         — molecules in the test set
    #
    # Because a molecule can belong to multiple datasets, the per-dataset counts
    # can sum to more than n_total — that is intentional and informative.

    scaffold_counts = (
        scaffold_hits
        # One row per (scaffold_smiles, smiles) pair — deduplicate within molecule
        # in case the same scaffold appears twice for the same input (shouldn't happen
        # given decompose_scaffold_network already deduplicates, but be defensive).
        .unique(subset=["scaffold_smiles", "smiles"])
        # Group only on the two identity columns; take scaffold_heavy_atoms as first()
        # so a stray null in that column cannot split a scaffold into multiple groups.
        .group_by(["scaffold_smiles", "scaffold_type"])
        .agg(
            pl.col("scaffold_heavy_atoms").first().alias("scaffold_heavy_atoms"),
            pl.col("smiles").n_unique().alias("n_total"),
            pl.col("in_single_dose") .filter(pl.col("in_single_dose")) .len().alias("n_single_dose"),
            pl.col("in_dose_response").filter(pl.col("in_dose_response")).len().alias("n_dose_response"),
            pl.col("in_counter")     .filter(pl.col("in_counter"))      .len().alias("n_counter"),
            pl.col("in_test")        .filter(pl.col("in_test"))         .len().alias("n_test"),
        )
        .select([
            "scaffold_smiles", "scaffold_type", "scaffold_heavy_atoms",
            "n_total", "n_single_dose", "n_dose_response", "n_counter", "n_test",
        ])
        .sort(["scaffold_type", "n_total"], descending=[False, True])
    )

    scaffold_counts
    return (scaffold_counts,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Scaffold coverage of the test set by the dose-response training set

    Each point is a scaffold present in **at least one test compound** and
    **at least one dose-response training compound**.

    - **x-axis** — number of dose-response training molecules containing this scaffold
    - **y-axis** — number of test molecules containing this scaffold
    - **colour**  — scaffold size in heavy atoms

    Scaffolds sitting high on the y-axis but far left are test-set enriched
    (low training coverage); scaffolds far right and low are training-enriched.
    Click a point to see the scaffold structure in the panel on the right.
    """)
    return


@app.cell
def _(alt, mo, pl, scaffold_counts):
    # ── Filter to scaffolds present in both test and dose-response sets ───────────
    _plot_df = (
        scaffold_counts
        .filter(
            (pl.col("n_test") >= 1) & (pl.col("n_dose_response") >= 1)
        )
    )

    # ── Altair scatter — one point per scaffold ───────────────────────────────────
    _sel = alt.selection_point(fields=["scaffold_smiles"], name="scaffold_sel", empty=False)

    _scatter = (
        alt.Chart(_plot_df)
        .mark_circle(opacity=0.85)
        .encode(
            x=alt.X(
                "n_dose_response:Q",
                title="Dose-response training molecules",
                scale=alt.Scale(type="log", base=10, domain=[0.9, 5000]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11,
                              values=[1, 10, 100, 1000, 5000], format="~s"),
            ),
            y=alt.Y(
                "n_test:Q",
                title="Test-set molecules",
                scale=alt.Scale(type="log", base=10, domain=[0.9, 500]),
                axis=alt.Axis(titleFontSize=13, labelFontSize=11,
                              values=[1, 10, 100, 500], format="~s"),
            ),
            color=alt.condition(
                _sel,
                alt.value("#f5c518"),          # gold when selected
                alt.Color(
                    "scaffold_heavy_atoms:Q",
                    scale=alt.Scale(scheme="viridis"),
                    legend=alt.Legend(title="Heavy atoms"),
                ),
            ),
            size=alt.condition(_sel, alt.value(150), alt.value(60)),
            tooltip=[
                alt.Tooltip("scaffold_smiles:N",      title="Scaffold"),
                alt.Tooltip("scaffold_type:N",        title="Type"),
                alt.Tooltip("scaffold_heavy_atoms:Q", title="Heavy atoms"),
                alt.Tooltip("n_dose_response:Q",      title="# dose-response"),
                alt.Tooltip("n_test:Q",               title="# test"),
                alt.Tooltip("n_total:Q",              title="# total"),
            ],
        )
        .add_params(_sel)
        .properties(
            title="Shared scaffolds — dose-response training vs. test set",
            width=500,
            height=400,
        )
        .configure_title(fontSize=12)
    )

    scaffold_coverage_chart = mo.ui.altair_chart(_scatter)
    return scaffold_coverage_chart, _plot_df


@app.cell
def _(Chem, mo, pl, rdDepictor, rdMolDraw2D, scaffold_counts, scaffold_coverage_chart):
    _PANEL_W = 300
    _PANEL_H = 260

    def _scaffold_to_svg(smi: str, width: int = _PANEL_W, height: int = _PANEL_H) -> str:
        """Render a scaffold SMILES as an SVG string."""
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

    _sel_rows = scaffold_coverage_chart.value

    if _sel_rows is None or len(_sel_rows) == 0:
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; height:{_PANEL_H + 80}px; display:flex;
                        align-items:center; justify-content:center;
                        color:grey; font-size:14px; border:1px dashed #ccc;
                        border-radius:6px'>
                Click a point to see the scaffold structure
            </div>
        """)
    else:
        _scaffold_smi = _sel_rows.row(0, named=True)["scaffold_smiles"]
        _r = (
            scaffold_counts
            .filter(pl.col("scaffold_smiles") == _scaffold_smi)
            .row(0, named=True)
        )
        _svg = _scaffold_to_svg(_r["scaffold_smiles"])
        _panel = mo.Html(f"""
            <div style='width:{_PANEL_W}px; font-family:monospace; font-size:11px'>
                <div style='padding:6px; background:#f5f5f5; border-radius:4px;
                            margin-bottom:4px; line-height:1.6'>
                    <b>Type:</b> {_r['scaffold_type']}<br>
                    <b>Heavy atoms:</b> {_r['scaffold_heavy_atoms']}<br>
                    <b># dose-response:</b> {_r['n_dose_response']}<br>
                    <b># test:</b> {_r['n_test']}<br>
                    <b># total:</b> {_r['n_total']}
                </div>
                {_strip_xml(_svg)}
            </div>
        """)

    mo.hstack([scaffold_coverage_chart, _panel], align="start")
    return


if __name__ == "__main__":
    app.run()
