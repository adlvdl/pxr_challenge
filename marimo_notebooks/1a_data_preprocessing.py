import marimo

__generated_with = "0.23.0"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # 1a — Data ingestion and preprocessing

    Reads the four raw CSV datasets, computes InChIKeys and InChIs, pivots and
    enriches the single-dose screen, and produces the combined `all_compounds`
    DataFrame that is written to `data/processed/all_compounds_activity_data.csv`.

    All downstream notebooks (1b–1e) read that CSV as their starting point.
    """)
    return


@app.cell
def _():
    import polars as pl
    import marimo as mo
    import subprocess
    import sys
    import base64
    from pathlib import Path
    from typing import Optional

    from rdkit import Chem, RDLogger

    # Suppress RDKit InChI warnings
    RDLogger.DisableLog("rdApp.*")
    return Chem, Optional, Path, mo, pl, subprocess, sys


@app.cell
def _(Chem, Optional):
    def smi_to_inchikey(smi: str) -> Optional[str]:
        """Return InChIKey for *smi*, or None on parse failure."""
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(mol) if mol else None

    def smi_to_inchi(smi: str) -> Optional[str]:
        """Return InChI for *smi*, or None on parse failure."""
        mol = Chem.MolFromSmiles(smi)
        return Chem.MolToInchi(mol) if mol else None

    return smi_to_inchi, smi_to_inchikey


@app.cell
def _(pl, smi_to_inchi, smi_to_inchikey):
    def process_dataset(df: pl.DataFrame) -> pl.DataFrame:
        """
        Standardise a raw dataset CSV into a common schema.

        - Renames the ``SMILES`` column to lowercase ``smiles``.
        - Appends ``inchikey`` and ``inchi`` columns computed via RDKit.

        Args:
            df: Raw Polars DataFrame loaded from one of the challenge CSVs.

        Returns:
            DataFrame with added ``smiles``, ``inchikey``, and ``inchi`` columns.
        """
        return df.rename({"SMILES": "smiles"}).with_columns(
            pl.col("smiles").map_elements(smi_to_inchikey, return_dtype=pl.Utf8).alias("inchikey"),
            pl.col("smiles").map_elements(smi_to_inchi,    return_dtype=pl.Utf8).alias("inchi"),
        )

    return (process_dataset,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Read raw datasets
    """)
    return


@app.cell
def _(pl, process_dataset):
    train         = process_dataset(pl.read_csv("../data/raw/20260409/dose_response_train.csv"))
    test          = process_dataset(pl.read_csv("../data/raw/20260409/dose_response_test.csv"))
    train_counter = process_dataset(pl.read_csv("../data/raw/20260409/counter_screen_train.csv"))
    train_single  = process_dataset(pl.read_csv("../data/raw/20260409/single_dose_train.csv"))
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
            pl.col("smiles").first(),
            pl.col("inchi").first(),
            pl.col("Molecule Name")
              .drop_nulls()
              .unique()
              .sort()
              .str.join("|")
              .alias("molecule_names"),
            pl.col("OCNT Batch")
              .drop_nulls()
              .unique()
              .sort()
              .str.join("|")
              .alias("ocnt_batches"),
        )
    )

    # Nominal concentration labels (µM): map messy float values to round numbers
    _conc_labels = {99.01: 100, 33.0: 30, 8.251: 10, 0.9803: 1}

    # ── Step 2: add is_hit flag, then aggregate per (inchikey, concentration_uM) ──
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
    """Write the combined dataset to disk for consumption by notebooks 1b–1e."""
    all_compounds.write_csv("../data/processed/all_compounds_activity_data.csv")
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Compute MMP input file

    Canonicalise SMILES through RDKit (strips CXSMILES extensions that confuse
    mmpdb) then write the two-column `.smi` file required by `mmpdb fragment`.
    If the stripping of CXSMILES is not done, it will trip up mmpdb and cause issues.
    Probably another way to solve would have been to use a different separator in the `.smi`
    file and adapt the call to `mmpdb fragment`.
    """)
    return


@app.cell
def _(Chem, all_compounds, pl):
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
    ## Run mmpdb: fragment + index

    Two steps using [mmpdb](https://github.com/rdkit/mmpdb):

    1. **fragment** — breaks each molecule at single rotatable bonds; writes a `.frag` file.
    2. **index** — finds all matched molecular pairs sharing the same core; writes a `.csv.gz`.

    Both steps are skipped automatically if the output file already exists.
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


if __name__ == "__main__":
    app.run()
