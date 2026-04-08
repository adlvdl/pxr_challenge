import marimo

__generated_with = "0.22.4"
app = marimo.App()


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Import libraries
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import altair as alt
    from rdkit import Chem
    import useful_rdkit_utils as uru
    import mols2grid

    from typing import Optional
    import matplotlib.pyplot as plt
    import matplotlib.figure
    from pathlib import Path

    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    import numpy as np
    from umap import UMAP


    #This removes some warnings that appear when calculating INCHIs
    from rdkit import RDLogger 
    RDLogger.DisableLog('rdApp.*')
    return Chem, Optional, PCA, Path, TSNE, UMAP, alt, mo, np, pl, plt, uru


@app.cell
def _(Chem):
    #adapted from https://rdkit.blogspot.com/2015/02/new-drawing-code.html
    #first seen at https://iwatobipen.wordpress.com/2024/01/19/new-type-of-python-notebook-marimo-cheminformatics-rdkit/
    from rdkit.Chem import rdDepictor
    from rdkit.Chem.Draw import rdMolDraw2D
    def smitosvg(smi,molSize=(200,200),kekulize=True):
        mc = Chem.MolFromSmiles(smi)
        if kekulize:
            try:
                Chem.Kekulize(mc)
            except:
                mc = Chem.MolFromSmiles(smi)
        if not mc.GetNumConformers():
            rdDepictor.Compute2DCoords(mc)
        drawer = rdMolDraw2D.MolDraw2DSVG(molSize[0],molSize[1])
        drawer.DrawMolecule(mc)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # It seems that the svg renderer used doesn't quite hit the spec.
        # Here are some fixes to make it work in the notebook, although I think
        # the underlying issue needs to be resolved at the generation step
        return svg.replace('svg:','')

    return


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
    Optional,
    Path,
    add_tsne_columns,
    add_umap_columns,
    alt,
    generate_fingerprint,
    mo,
    pl,
    plt,
    uru,
):


    def add_image_column(df: pl.DataFrame, image_column: str = "image", smiles_column: str = "smiles") -> pl.DataFrame:
        """
        Adds a column with molecule images to a DataFrame.

        Args:
            df: DataFrame to add the image column to.
            image_column: Name of the new image column.
            smiles_column: Name of the column containing SMILES strings.

        Returns:
            DataFrame with the added image column.
        """

        if image_column not in df.columns:
            image_list = df[smiles_column].map_elements(lambda x: uru.smi_to_base64_image(x, target='altair'), return_dtype=pl.Utf8)
            df = df.with_columns(pl.Series(name=image_column, values=image_list) )  
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
        save_path: Optional[str | Path] = None,
        dpi: int = 300,
        figsize: tuple[float, float] = (8.0, 7.0),
        point_size: float = 18.0,
        alpha: float = 0.8,
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
                    sc = ax.scatter(x, y, c=values, cmap="viridis",
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

    return (generate_embedding_plot,)


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Read and preprocess the data
    """)
    return


@app.cell
def _(pl, process_dataset):
    train = process_dataset(pl.read_csv("../data/raw/20260403/dose_response_train.csv"))
    test = process_dataset(pl.read_csv("../data/raw/20260403/dose_response_test.csv"))
    train_counter = process_dataset(pl.read_csv("../data/raw/20260403/counter_screen_train.csv"))
    train_single = process_dataset(pl.read_csv("../data/raw/20260403/single_dose_train.csv"))
    return test, train, train_counter, train_single


@app.cell
def _(mo):
    mo.md(r"""
    ## Pivot single-dose dataset by concentration

    Each molecule (identified by `inchikey`) may have been tested at multiple concentrations.
    Here we:
    1. Aggregate molecule identifiers: keep `smiles`, `inchi`, `inchikey` as unique entities,
       concatenate all `Molecule Name` and `OCNT Batch` values seen for that molecule.
    2. Pivot so that each `concentration_M` becomes its own set of columns, computing the
       **median** of `median_log2_fc`, `cohens_d`, and `log2_fc_estimate` across replicates/batches.
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

    # ── Step 2: compute per-(inchikey, concentration_M) medians ──────────────────
    _agg_values = (
        train_single
        .with_columns((10**6 *  pl.col("concentration_M")).alias("concentration_uM"))
        .group_by(["inchikey", "concentration_uM"])
        .agg(
            pl.col("median_log2_fc").median(),
            pl.col("cohens_d").median(),
            pl.col("log2_fc_estimate").median(),
        )
    )

    # ── Step 3: pivot so each concentration becomes its own column ───────────────
    # Polars pivot produces one column per (value_column × unique_concentration).
    # The column names will be e.g. "median_log2_fc_1e-05", "cohens_d_9.901e-05" …
    _pivoted = _agg_values.pivot(
        on="concentration_uM",          # each unique concentration → new column(s)
        index="inchikey",
        values=["median_log2_fc", "cohens_d", "log2_fc_estimate"],
        aggregate_function="median",   # already aggregated above, but keeps pivot safe
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Plot and explore chemical space
    """)
    return


@app.cell
def _(pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    return (all_compounds,)


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
    generate_embedding_plot(plot_df_colored, x_col="UMAP_x", y_col="UMAP_y", 
                        color_col="color", point_size=6,
                        title="UMAP ECFP6 Chemical Space")
    return


@app.cell
def _(generate_embedding_plot, plot_df_colored):
    generate_embedding_plot(plot_df_colored, x_col="TSNE_x", y_col="TSNE_y", 
                        color_col="color", point_size=6,
                        title="tSNE ECFP6 Chemical Space")
    return


if __name__ == "__main__":
    app.run()
