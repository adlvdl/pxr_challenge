import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Webinar plots

    This notebook adapts previously generated plots for a webinar presentation.
    Where possible, existing tables/files are read directly instead of recomputing
    values. When recomputation is unavoidable, the original code from the source
    notebook is reused as-is so the underlying data matches exactly.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Imports

    Minimal set of imports needed to reproduce the adapted plots, copied from
    `2_ml_baseline.py`.
    """)
    return


@app.cell
def _():
    import math
    import warnings
    from pathlib import Path
    from typing import Iterator, Optional

    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import numpy as np
    import pandas as pd
    import polars as pl
    import marimo as mo
    import pingouin as pg
    import seaborn as sns

    from scipy.stats import spearmanr
    from sklearn.model_selection._split import _BaseKFold as BaseKFold
    from statsmodels.stats.libqsturng import psturng, qsturng

    from rdkit import Chem, DataStructs
    from rdkit.Chem.Scaffolds import MurckoScaffold
    from rdkit.DataStructs import ExplicitBitVect

    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer
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

    return (
        AtomPairFingerprint,
        AvalonFingerprint,
        BaseKFold,
        Chem,
        ConformerGenerator,
        DataStructs,
        E3FPFingerprint,
        ECFPFingerprint,
        ExplicitBitVect,
        Iterator,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        MurckoScaffold,
        Optional,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        TopologicalTorsionFingerprint,
        math,
        mo,
        mpatches,
        np,
        pd,
        pg,
        pl,
        plt,
        psturng,
        qsturng,
        sns,
        spearmanr,
        warnings,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Data splitting utilities

    Copied verbatim from `2_ml_baseline.py` — only the **random** and **scaffold**
    strategies are needed here (the temporal split is intentionally excluded from
    the webinar version of this plot).
    """)
    return


@app.cell
def _(BaseKFold, Iterator, Optional, np, pl):
    # ── helpers (copied from 2_ml_baseline.py) ──────────────────────────────────

    def split_dataset_random(
        df: pl.DataFrame,
        p_test: float = 0.2,
        seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """
        Randomly split a DataFrame into train and test subsets.

        Args:
            df: Input DataFrame.
            p_test: Fraction of rows allocated to the test set.
            seed: Random seed for reproducibility.

        Returns:
            Tuple of (train_df, test_df).
        """
        rng = np.random.default_rng(seed)
        idx = rng.permutation(df.shape[0])
        n_test = int(len(idx) * p_test)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return df[train_idx].clone(), df[test_idx].clone()

    class GroupKFoldShuffle(BaseKFold):
        """
        K-fold cross-validator that respects group boundaries and supports shuffling.

        Args:
            n_splits: Number of folds.
            shuffle: Whether to shuffle groups before splitting.
            random_state: Random seed used when shuffle=True.
        """

        def __init__(
            self,
            n_splits: int = 5,
            *,
            shuffle: bool = False,
            random_state: Optional[int] = None,
        ) -> None:
            super().__init__(n_splits=n_splits, shuffle=shuffle, random_state=random_state)

        def split(self, X, y=None, groups=None) -> Iterator:
            unique_groups = np.unique(groups)

            if self.shuffle:
                rng = np.random.default_rng(self.random_state)
                unique_groups = rng.permutation(unique_groups)

            split_groups = np.array_split(unique_groups, self.n_splits)

            for test_group_ids in split_groups:
                test_mask = np.isin(groups, test_group_ids)
                train_mask = ~test_mask
                yield np.where(train_mask)[0], np.where(test_mask)[0]

    return GroupKFoldShuffle, split_dataset_random


@app.cell
def _(GroupKFoldShuffle, Iterator, pl, split_dataset_random):
    def generate_cv_splits_random(
        df: pl.DataFrame,
        n_outer: int = 5,
        n_inner: int = 5,
        seed: int = 42,
        p_val: float = 0,
    ) -> Iterator:
        """
        Generate nested 5x5 CV splits using a **random** molecule assignment.

        Args:
            df: Polars DataFrame to split.
            n_outer: Number of outer CV folds.
            n_inner: Number of inner CV folds per outer iteration.
            seed: Random seed for GroupKFoldShuffle.
            p_val: Fraction of the training set reserved as a validation split.

        Yields:
            Tuples of (fold_index, outer_index, inner_index, train_df, val_df, test_df).
        """
        for i in range(n_outer):
            kf = GroupKFoldShuffle(n_splits=n_inner, random_state=seed + i, shuffle=True)
            groups = list(range(df.shape[0]))
            for j, (train_idx, test_idx) in enumerate(kf.split(df, groups=groups)):
                fold = i * n_inner + j
                train = df[train_idx].clone()
                test = df[test_idx].clone()

                val = None
                if p_val > 0:
                    train, val = split_dataset_random(train, p_test=p_val, seed=seed + fold)
                yield fold, i, j, train, val, test

    return (generate_cv_splits_random,)


@app.cell
def _(
    Chem,
    GroupKFoldShuffle,
    Iterator,
    MurckoScaffold,
    np,
    pl,
    split_dataset_random,
):
    def _get_bemis_murcko_scaffold(smiles: str) -> str:
        """
        Return the canonical Bemis-Murcko scaffold SMILES for a molecule.

        Args:
            smiles: Input SMILES string.

        Returns:
            Canonical scaffold SMILES, or "" on failure.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return ""
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return Chem.MolToSmiles(scaffold, canonical=True)

    def generate_cv_splits_scaffold(
        df: pl.DataFrame,
        n_outer: int = 5,
        n_inner: int = 5,
        seed: int = 42,
        p_val: float = 0,
    ) -> Iterator:
        """
        Generate nested 5x5 CV splits using **Bemis-Murcko scaffold** assignment.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            n_outer: Number of outer CV folds.
            n_inner: Number of inner CV folds per outer iteration.
            seed: Random seed for GroupKFoldShuffle.
            p_val: Fraction of the training set reserved as a validation split.

        Yields:
            Tuples of (fold_index, outer_index, inner_index, train_df, val_df, test_df).
        """
        scaffolds = [_get_bemis_murcko_scaffold(s) for s in df["smiles"].to_list()]
        unique_scaffolds = list(dict.fromkeys(scaffolds))
        scaffold_to_id = {s: i for i, s in enumerate(unique_scaffolds)}
        groups = np.array([scaffold_to_id[s] for s in scaffolds])

        for i in range(n_outer):
            kf = GroupKFoldShuffle(n_splits=n_inner, random_state=seed + i, shuffle=True)
            for j, (train_idx, test_idx) in enumerate(kf.split(df, groups=groups)):
                fold = i * n_inner + j
                train = df[train_idx].clone()
                test = df[test_idx].clone()

                val = None
                if p_val > 0:
                    train, val = split_dataset_random(train, p_test=p_val, seed=seed + fold)
                yield fold, i, j, train, val, test

    return (generate_cv_splits_scaffold,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Fingerprint generation

    Copied verbatim from `2_ml_baseline.py`.
    """)
    return


@app.cell
def _(
    AtomPairFingerprint,
    AvalonFingerprint,
    ConformerGenerator,
    E3FPFingerprint,
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

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported types (see _fp_dict keys).
            **kwargs: Additional keyword arguments forwarded to the skfp fingerprint class.

        Returns:
            DataFrame with an added column named after fingerprint_type.
        """
        if fingerprint_type not in _fp_dict.keys():
            raise ValueError(
                f"Fingerprint type not recognized: {fingerprint_type!r}. "
                f"Valid values: {list(_fp_dict.keys())}"
            )

        if len(kwargs) == 0:
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
        fps = df.with_columns(fps_col)
        return fps

    return (generate_fingerprint,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Load training data

    Same source file as `2_ml_baseline.py`: `data/processed/all_compounds_activity_data.csv`,
    filtered to compounds with a measured `pEC50_dr`.
    """)
    return


@app.cell
def _(pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    whole_train = all_compounds.filter(pl.col("pEC50_dr").is_not_null())
    return (whole_train,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Train/test Tanimoto similarity — random vs. scaffold split

    Adapted from `plots/2_ml_baseline/tanimoto_similarity_splits.png`: the **temporal**
    split is dropped, keeping only **random** and **scaffold**, since those are the two
    strategies relevant for the webinar narrative. The similarity computation itself
    (ECFP4, radius 2, 2048 bits, one outer round of 5-fold CV per strategy) is identical
    to the original notebook — it is not persisted as a table anywhere, so it is
    recomputed here using the exact same helper functions and seed.
    """)
    return


@app.cell
def _(
    DataStructs,
    ExplicitBitVect,
    generate_cv_splits_random,
    generate_cv_splits_scaffold,
    generate_fingerprint,
    np,
    pl,
    whole_train,
):
    def _to_rdkit_bitvects(df: pl.DataFrame) -> list[ExplicitBitVect]:
        """
        Convert the "ecfp" uint8 numpy array column (added by generate_fingerprint)
        to a list of RDKit ExplicitBitVect objects required by BulkTanimotoSimilarity.
        """
        fp_size = len(df["ecfp"][0])
        bitvects = []
        for arr in df["ecfp"].to_list():
            bv = ExplicitBitVect(fp_size)
            for i in np.flatnonzero(arr):
                bv.SetBit(int(i))
            bitvects.append(bv)
        return bitvects

    def _fold_similarities(train_df: pl.DataFrame, test_df: pl.DataFrame) -> dict[str, list[float]]:
        """
        Compute all-pairs and nearest-neighbour Tanimoto similarities between
        test and train fingerprints for a single fold.
        """
        train_fps = _to_rdkit_bitvects(generate_fingerprint(train_df, "ecfp", radius=2, fp_size=2048))
        test_fps  = _to_rdkit_bitvects(generate_fingerprint(test_df,  "ecfp", radius=2, fp_size=2048))

        all_sims: list[float] = []
        nn_sims:  list[float] = []

        for test_fp in test_fps:
            sims = DataStructs.BulkTanimotoSimilarity(test_fp, train_fps)
            all_sims.extend(sims)
            nn_sims.append(float(np.max(sims)))

        return {"all": all_sims, "nn": nn_sims}

    # ── collect similarities for random and scaffold splits only ───────────────

    records: list[dict] = []

    # Random 1x5 CV  (n_outer=1, n_inner=5)
    for _fold, _outer, _inner, _train, _val, _test in generate_cv_splits_random(
        whole_train, n_outer=1, n_inner=5, seed=42
    ):
        _sims = _fold_similarities(_train, _test)
        for _s in _sims["all"]:
            records.append({"split": "random", "mode": "all pairs",        "tanimoto": _s})
        for _s in _sims["nn"]:
            records.append({"split": "random", "mode": "nearest neighbour","tanimoto": _s})

    # Scaffold 1x5 CV
    for _fold, _outer, _inner, _train, _val, _test in generate_cv_splits_scaffold(
        whole_train, n_outer=1, n_inner=5, seed=42
    ):
        _sims = _fold_similarities(_train, _test)
        for _s in _sims["all"]:
            records.append({"split": "scaffold", "mode": "all pairs",        "tanimoto": _s})
        for _s in _sims["nn"]:
            records.append({"split": "scaffold", "mode": "nearest neighbour","tanimoto": _s})

    sim_df = pl.DataFrame(records)
    return (sim_df,)


@app.cell
def _(Path, mo, mpatches, plt, sim_df):
    _splits  = ["random", "scaffold"]
    _modes   = ["all pairs", "nearest neighbour"]
    _colors  = {"random": "#4C78A8", "scaffold": "#F58518"}
    _labels  = {"all pairs": "All pairs", "nearest neighbour": "Nearest neighbour"}

    _fig, _axes = plt.subplots(2, 1, figsize=(6, 9), sharex=True)
    _fig.suptitle("Train/test Tanimoto similarity by split strategy (ECFP4)", fontsize=13)

    for _ax, _mode in zip(_axes, _modes):
        _data = [
            sim_df.filter((sim_df["split"] == _s) & (sim_df["mode"] == _mode))["tanimoto"].to_list()
            for _s in _splits
        ]
        _bp = _ax.boxplot(
            _data,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "black", "linewidth": 1.5},
            whiskerprops={"linewidth": 1.2},
            capprops={"linewidth": 1.2},
        )
        for _patch, _split in zip(_bp["boxes"], _splits):
            _patch.set_facecolor(_colors[_split])
            _patch.set_alpha(0.8)

        _ax.set_title(_labels[_mode], fontsize=12)
        _ax.set_xticks([1, 2])
        _ax.set_xticklabels(_splits, fontsize=11)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("Tanimoto similarity", fontsize=11)
        _ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        _ax.set_axisbelow(True)

    _axes[-1].set_xlabel("Split strategy", fontsize=11)

    _handles = [mpatches.Patch(facecolor=_colors[s], alpha=0.8, label=s) for s in _splits]
    _fig.legend(handles=_handles, loc="lower center", ncol=2, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.04))
    _fig.tight_layout()

    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_PLOT_DIR / "tanimoto_similarity_splits.png", bbox_inches="tight")

    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Chemprop size comparison — reference-style CI plot

    Adapted from `plots/3_ml_optimization/chemprop_size_mcs.png`. The original MCS
    grid plot (4 metrics x pairwise comparisons) is dense and its labels overlap when
    scaled down for slides, and the accompanying boxplots don't communicate
    statistical significance directly. This version keeps only the **MAE** metric and
    shows, for each chemprop-size variant, its mean MAE difference from the reference
    method (`ch_base` — the baseline chemprop architecture from the size-test run)
    with a 95% Tukey HSD confidence interval:

    - **Blue** — the reference method itself (`ch_base`)
    - **Red** — significantly worse than the reference (Tukey HSD p-adj < 0.05, higher MAE)
    - **Green** — significantly better than the reference (Tukey HSD p-adj < 0.05, lower MAE)
    - **Gray** — not significantly different from the reference

    The underlying predictions are read directly from
    `predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz` and
    `predictions/3_larger_chemprop_size_test.csv.gz`, and the metric/statistics
    computation (`calc_regression_metrics`, `rm_tukey_hsd`) is copied verbatim from
    `3_ml_optimization.py` so the numbers match the original plot exactly.
    """)
    return


@app.cell
def _(pl, spearmanr, warnings):
    def calc_regression_metrics(
        df: pl.DataFrame,
        cycle_col: str,
        val_col: str,
        pred_col: str,
        thresh: float,
    ) -> pl.DataFrame:
        """
        Calculate regression metrics (MAE, MSE, R2, rho, prec, recall) for each method and split.

        Copied from `3_ml_optimization.py`, with the NaN-prediction guard from the
        `4_ml_optimization_2.py` version added (needed for Macau's occasional
        numerical instability on a couple of Mordred-descriptor compounds).

        Args:
            df: Polars DataFrame with columns [method, split] plus the columns named in the
                remaining arguments.
            cycle_col: Column indicating the cross-validation fold.
            val_col: Column with the ground truth values.
            pred_col: Column with the model predictions.
            thresh: Decision threshold used to binarise continuous values for precision/recall.

        Returns:
            Polars DataFrame with columns [cv_cycle, method, split, mae, mse, r2, rho, prec, recall].
        """
        from sklearn.metrics import (
            mean_absolute_error,
            mean_squared_error,
            r2_score,
            precision_score,
            recall_score,
        )

        # Drop rows where predictions are NaN (e.g. Macau numerical instability)
        # before any metric computation to avoid downstream sklearn errors.
        df_in = df.filter(pl.col(pred_col).is_not_nan() & pl.col(pred_col).is_not_null())
        df_in = df_in.with_columns([
            (pl.col(val_col) > thresh).alias("true_class"),
            (pl.col(pred_col) > thresh).alias("pred_class"),
        ])

        assert df_in["true_class"].n_unique() == 2, "Binary classification requires two classes"

        metric_list: list[dict] = []

        for group_keys, group_df in df_in.group_by([cycle_col, "method", "split"]):
            cycle, method, split = group_keys
            y_true = group_df[val_col].to_numpy()
            y_pred = group_df[pred_col].to_numpy()
            y_true_cls = group_df["true_class"].to_numpy()
            y_pred_cls = group_df["pred_class"].to_numpy()

            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                rho, _ = spearmanr(y_true, y_pred)
            metric_list.append({
                "cv_cycle": cycle,
                "method": method,
                "split": split,
                "mae": mean_absolute_error(y_true, y_pred),
                "mse": mean_squared_error(y_true, y_pred),
                "r2": r2_score(y_true, y_pred),
                "rho": rho,
                "prec": precision_score(y_true_cls, y_pred_cls, zero_division=0),
                "recall": recall_score(y_true_cls, y_pred_cls, zero_division=0),
            })

        return pl.DataFrame(metric_list)

    return (calc_regression_metrics,)


@app.cell
def _(Optional, np, pd, pg, pl, psturng, qsturng, warnings):
    def rm_tukey_hsd(
        df: pl.DataFrame,
        metric: str,
        group_col: str,
        alpha: float = 0.05,
        sort: bool = False,
        direction_dict: Optional[dict] = None,
    ) -> tuple:
        """
        Perform repeated measures Tukey HSD test on the given Polars DataFrame.

        Copied verbatim from `3_ml_optimization.py`. Internally converts to pandas
        for pingouin/statsmodels compatibility.

        Args:
            df: Polars DataFrame with columns [cv_cycle, group_col, metric].
            metric: Column name of the metric to test.
            group_col: Column name indicating the comparison groups.
            alpha: Significance level for the test.
            sort: Whether to sort groups by their mean metric value.
            direction_dict: Maps metric names to "maximize" or "minimize" for sort direction.

        Returns:
            Tuple of (result_tab, df_means, df_means_diff, pc) — all pandas DataFrames.
        """
        df_pd = df.to_pandas()

        if sort and direction_dict and metric in direction_dict:
            if direction_dict[metric] == 'maximize':
                df_means = df_pd.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=False)
            elif direction_dict[metric] == 'minimize':
                df_means = df_pd.groupby(group_col).mean(numeric_only=True).sort_values(metric, ascending=True)
            else:
                raise ValueError("Invalid direction. Expected 'maximize' or 'minimize'.")
        else:
            df_means = df_pd.groupby(group_col).mean(numeric_only=True)

        with warnings.catch_warnings():
            warnings.filterwarnings('ignore', category=RuntimeWarning,
                                    message='divide by zero encountered in scalar divide')
            aov = pg.rm_anova(dv=metric, within=group_col, subject='cv_cycle', data=df_pd, detailed=True)
        mse = aov.loc[1, 'MS']
        df_resid = aov.loc[1, 'DF']

        methods = df_means.index
        n_groups = len(methods)
        n_per_group = df_pd[group_col].value_counts().mean()

        tukey_se = np.sqrt(2 * mse / n_per_group)
        q = qsturng(1 - alpha, n_groups, df_resid)

        num_comparisons = len(methods) * (len(methods) - 1) // 2
        result_tab = pd.DataFrame(index=range(num_comparisons),
                                  columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])

        df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
        pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

        row_idx = 0
        for i, method1 in enumerate(methods):
            for j, method2 in enumerate(methods):
                if i < j:
                    group1 = df_pd[df_pd[group_col] == method1][metric]
                    group2 = df_pd[df_pd[group_col] == method2][metric]
                    mean_diff = group1.mean() - group2.mean()
                    studentized_range = np.abs(mean_diff) / tukey_se
                    adjusted_p = psturng(studentized_range * np.sqrt(2), n_groups, df_resid)
                    if isinstance(adjusted_p, np.ndarray):
                        adjusted_p = adjusted_p[0]
                    lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                    upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                    result_tab.loc[row_idx] = [method1, method2, mean_diff, lower, upper, adjusted_p]
                    pc.loc[method1, method2] = adjusted_p
                    pc.loc[method2, method1] = adjusted_p
                    df_means_diff.loc[method1, method2] = mean_diff
                    df_means_diff.loc[method2, method1] = -mean_diff
                    row_idx += 1

        df_means_diff = df_means_diff.astype(float)

        result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
        result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])

        result_tab.index = result_tab['group1'] + ' - ' + result_tab['group2']

        return result_tab, df_means, df_means_diff, pc

    return (rm_tukey_hsd,)


@app.cell
def _(mo):
    mo.md(r"""
    ### Reference-style CI plot

    New helpers (not present in the original notebooks) that reshape a `rm_tukey_hsd`
    result table into a "method vs. reference" view: one row per method, showing its
    mean difference from a chosen reference method with a Tukey HSD confidence
    interval, colored by significance relative to that reference.

    `make_reference_ci_plot` draws a single panel; `make_stacked_reference_ci_plot`
    stacks several such panels vertically (sharing the x-axis), each with its own
    data and reference method — used below for the filtering analysis, one panel
    per model.
    """)
    return


@app.cell
def _(Optional, Path, np, pd, plt, rm_tukey_hsd):
    _REF_COLORS = {
        "reference": "#4C78A8",  # blue — method under comparison
        "worse": "#D62728",      # red — significantly worse
        "better": "#2CA02C",     # green — significantly better
        "similar": "#7F7F7F",    # gray — not significantly different
    }

    def _reference_ci_rows(
        df_in,
        metric: str,
        reference: str,
        group_col: str,
        alpha: float,
    ):
        """
        Build a per-method table of mean difference from `reference` with a Tukey
        HSD confidence interval, shared by make_reference_ci_plot and
        make_stacked_reference_ci_plot.

        Returns:
            (methods, plot_df) — plot_df indexed by method with columns
            [meandiff, lower, upper, p_adj].
        """
        result_tab, df_means, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col, alpha=alpha)

        methods = list(df_means.index)
        rows = []
        for method in methods:
            if method == reference:
                rows.append({"method": method, "meandiff": 0.0, "lower": 0.0, "upper": 0.0, "p_adj": np.nan})
                continue
            # rm_tukey_hsd stores each unordered pair once; meandiff is group1 - group2
            fwd = result_tab[(result_tab["group1"] == reference) & (result_tab["group2"] == method)]
            rev = result_tab[(result_tab["group1"] == method) & (result_tab["group2"] == reference)]
            if len(fwd) == 1:
                # meandiff is reference - method; flip sign so positive means method is worse (higher metric)
                row = fwd.iloc[0]
                rows.append({
                    "method": method,
                    "meandiff": -row["meandiff"],
                    "lower": -row["upper"],
                    "upper": -row["lower"],
                    "p_adj": row["p-adj"],
                })
            else:
                row = rev.iloc[0]
                rows.append({
                    "method": method,
                    "meandiff": row["meandiff"],
                    "lower": row["lower"],
                    "upper": row["upper"],
                    "p_adj": row["p-adj"],
                })

        plot_df = pd.DataFrame(rows).set_index("method").loc[methods]
        return methods, plot_df

    def _reference_ci_color(row, reference: str, alpha: float, higher_is_better: bool) -> str:
        if row.name == reference:
            return _REF_COLORS["reference"]
        if row["p_adj"] < alpha:
            # meandiff > 0 means "method minus reference" is positive.
            # Whether that's worse or better depends on the metric's direction.
            is_worse = (row["meandiff"] > 0) != higher_is_better
            return _REF_COLORS["worse"] if is_worse else _REF_COLORS["better"]
        return _REF_COLORS["similar"]

    def _draw_reference_ci(
        ax, methods, plot_df, reference: str, alpha: float, higher_is_better: bool,
        method_order=None, labels=None, group_bands=None,
    ) -> None:
        """
        Args:
            method_order: Optional explicit top-to-bottom row order (list of method
                names). Defaults to the order in `methods`.
            labels: Optional dict mapping method name -> display label for the
                y-tick (e.g. to shorten "xgb_mordred_rand_counter" to "counter (random)").
            group_bands: Optional list of ints, one per row (same order as
                method_order), used to shade alternating groups of related rows
                (e.g. a filter and its matched random control) with a faint
                background band so paired rows are visually linked.
        """
        order = method_order or methods
        y_pos = np.arange(len(order))

        if group_bands is not None:
            # Draw one contiguous span per run of same-band rows rather than one
            # span per row — overlapping per-row spans would stack their alpha at
            # shared edges and show up as a visible seam between paired rows.
            _run_start = 0
            for i in range(1, len(group_bands) + 1):
                if i == len(group_bands) or group_bands[i] != group_bands[_run_start]:
                    if group_bands[_run_start] % 2 == 1:
                        ax.axhspan(
                            y_pos[_run_start] - 0.5, y_pos[i - 1] + 0.5,
                            color="#000000", alpha=0.04, zorder=0,
                        )
                    _run_start = i

        for y, method in zip(y_pos, order):
            row = plot_df.loc[method]
            color = _reference_ci_color(row, reference, alpha, higher_is_better)
            if method == reference:
                ax.scatter([row["meandiff"]], [y], color=color, zorder=3, s=50)
            else:
                err = np.array([[row["meandiff"] - row["lower"]], [row["upper"] - row["meandiff"]]])
                ax.errorbar(
                    [row["meandiff"]], [y], xerr=err,
                    fmt="o", color=color, ecolor=color, capsize=4, markersize=6, zorder=3,
                )
        ax.set_yticks(y_pos)
        ax.set_yticklabels([labels[m] if labels and m in labels else m for m in order])
        ax.set_ylim(-0.5, len(order) - 0.5)
        ax.axvline(0, color="black", ls="--", lw=1, zorder=1)

    def _reference_ci_legend_handles() -> list:
        return [
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_REF_COLORS["reference"], markersize=8, label="Method under comparison"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_REF_COLORS["worse"], markersize=8, label="Significantly worse"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_REF_COLORS["better"], markersize=8, label="Significantly better"),
            plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=_REF_COLORS["similar"], markersize=8, label="Similar"),
        ]

    def make_reference_ci_plot(
        df_in,
        metric: str,
        reference: str,
        group_col: str = "method",
        alpha: float = 0.05,
        higher_is_better: bool = False,
        figsize: tuple[int, int] = (7, 5),
        xlabel: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot each method's mean difference from a reference method with a Tukey HSD CI.

        Args:
            df_in: Polars DataFrame passed through to rm_tukey_hsd.
            metric: Metric column name to compare (e.g. "mae").
            reference: Value in group_col to use as the reference ("method under comparison").
            group_col: Column indicating the comparison groups. Default is "method".
            alpha: Significance level used by rm_tukey_hsd. Default is 0.05.
            higher_is_better: Whether a higher metric value is the better outcome
                (e.g. True for r2/rho, False for mae/mse). Controls which side of
                zero is colored red (worse) vs. green (better). Default False.
            figsize: Figure size as (width, height).
            xlabel: Custom x-axis label. Defaults to f"Mean {metric.upper()} difference from {reference}".
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        methods, plot_df = _reference_ci_rows(df_in, metric, reference, group_col, alpha)

        fig, ax = plt.subplots(figsize=figsize)
        _draw_reference_ci(ax, methods, plot_df, reference, alpha, higher_is_better)
        ax.set_xlabel(xlabel or f"Mean {metric.upper()} difference from {reference}")
        ax.set_ylabel("Method")
        ax.legend(handles=_reference_ci_legend_handles(), loc="best", frameon=True)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    def make_stacked_reference_ci_plot(
        panels: list[dict],
        metric: str,
        group_col: str = "method",
        alpha: float = 0.05,
        higher_is_better: bool = False,
        figsize: tuple[int, int] = (7, 8),
        xlabel: Optional[str] = None,
        suptitle: Optional[str] = None,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Stack several make_reference_ci_plot-style panels vertically, sharing the x-axis.

        Args:
            panels: List of dicts, one per stacked panel, each with keys:
                - "title": panel title (str).
                - "df_in": Polars DataFrame passed to rm_tukey_hsd for this panel.
                - "reference": Value in group_col to use as the reference for this panel.
                - "method_order" (optional): explicit top-to-bottom row order.
                - "labels" (optional): dict mapping method name -> display label,
                    e.g. to shorten "xgb_mordred_rand_counter" to "counter (random)".
                - "group_bands" (optional): list of ints (same order as
                    method_order) used to shade alternating groups of related rows
                    with a faint background band, so e.g. a filter and its matched
                    random control are visually linked.
                Each panel's df_in/reference pair is passed to rm_tukey_hsd
                independently, so different panels may use different reference
                methods or method sets.
            metric: Metric column name to compare (e.g. "mae").
            group_col: Column indicating the comparison groups. Default is "method".
            alpha: Significance level used by rm_tukey_hsd. Default is 0.05.
            higher_is_better: Whether a higher metric value is the better outcome.
            figsize: Figure size as (width, height).
            xlabel: Custom x-axis label for the bottom panel. Defaults to
                f"Mean {metric.upper()} difference from reference".
            suptitle: Optional figure-level title.
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        fig, axes = plt.subplots(len(panels), 1, figsize=figsize, sharex=True)
        if len(panels) == 1:
            axes = [axes]

        for ax, panel in zip(axes, panels):
            reference = panel["reference"]
            methods, plot_df = _reference_ci_rows(panel["df_in"], metric, reference, group_col, alpha)
            _draw_reference_ci(
                ax, methods, plot_df, reference, alpha, higher_is_better,
                method_order=panel.get("method_order"),
                labels=panel.get("labels"),
                group_bands=panel.get("group_bands"),
            )
            ax.set_title(panel["title"], fontsize=11)
            ax.set_ylabel("Method")

        axes[-1].set_xlabel(xlabel or f"Mean {metric.upper()} difference from reference")
        if suptitle:
            fig.suptitle(suptitle, fontsize=13)
        fig.legend(handles=_reference_ci_legend_handles(), loc="lower center", ncol=4,
                   fontsize=9, frameon=False, bbox_to_anchor=(0.5, -0.02))
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    return make_reference_ci_plot, make_stacked_reference_ci_plot


@app.cell
def _(Path, calc_regression_metrics, mo, pl):
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")

    _chemeleon_ref = (
        pl.read_csv(_BASELINE)
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _chemprop_size_df = (
        pl.read_csv(Path("../predictions/3_larger_chemprop_size_test.csv.gz"))
        .with_columns(pl.col("model").str.replace("chemprop", "ch"))
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _cmp_df = pl.concat([_chemprop_size_df, _chemeleon_ref])
    chemprop_size_metrics = calc_regression_metrics(
        _cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0
    )
    mo.md("Loaded and recomputed metrics for the chemprop size comparison (same pipeline as `3_ml_optimization.py`).")
    return (chemprop_size_metrics,)


@app.cell
def _(Path, chemprop_size_metrics, make_reference_ci_plot, mo):
    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_reference_ci_plot(
        chemprop_size_metrics,
        metric="mae",
        reference="ch_base",
        xlabel="Mean MAE difference from ch_base",
        save_path=_PLOT_DIR / "chemprop_size_mae_ci.png",
    )
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Multitask comparison — reference-style CI plot

    Adapted from `plots/3_ml_optimization/multitask_ci_mae.png`, using the same
    `make_reference_ci_plot` helper as the chemprop-size comparison above so both
    webinar slides share one visual language. Reference method is `st_pec50` — the
    single-task chemprop control trained only on pEC50 — so the plot answers
    "does adding multitask targets help or hurt MAE relative to the single-task
    model?"

    The underlying predictions are read directly from
    `predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz` and
    `predictions/3_multitask_test.csv.gz`, reusing the same `calc_regression_metrics`
    pipeline copied from `3_ml_optimization.py`.
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl):
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")

    _chemeleon_ref = (
        pl.read_csv(_BASELINE)
        .filter(pl.col("model") == "chemeleon")
        .with_columns(pl.lit("chemel").alias("model"))
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _mt_cmp_df = (
        pl.read_csv(Path("../predictions/3_multitask_test.csv.gz"))
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _cmp_df = pl.concat([_mt_cmp_df, _chemeleon_ref])
    multitask_metrics = calc_regression_metrics(
        _cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0
    )
    mo.md("Loaded and recomputed metrics for the multitask comparison (same pipeline as `3_ml_optimization.py`).")
    return (multitask_metrics,)


@app.cell
def _(Path, make_reference_ci_plot, mo, multitask_metrics):
    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_reference_ci_plot(
        multitask_metrics,
        metric="mae",
        reference="st_pec50",
        figsize=(7, 6),
        xlabel="Mean MAE difference from st_pec50",
        save_path=_PLOT_DIR / "multitask_mae_ci.png",
    )
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Training-set filtering — reference-style CI plot

    Adapted from `plots/6_ml_optimization_3/analysis5_filtering_delta_mae.png`. The
    original is a bar chart of point-estimate ΔMAE vs. the full-data baseline, with a
    filter bar next to its size-matched random-removal twin — it doesn't show whether
    any difference is statistically significant. This version uses the same
    reference-CI style as the other webinar plots, with the full-data baseline as the
    reference (blue) and each filter (`counter`, `cliff`, `knn`) and its matched
    random control (`rand_counter`, `rand_cliff`, `rand_knn`) as its own row with a
    Tukey HSD confidence interval. The two models (`xgb_mordred`, `macau_chemeleon`)
    are stacked one on top of the other instead of side by side.

    Predictions are read directly from `predictions/6_filtering_cv.csv.gz`, with the
    full-data baselines pulled from `predictions/6_reweighting_cv.csv.gz` and
    `predictions/4_fp_model_comparison_1.csv.gz` — the same join used in
    `6_ml_optimization_3.py`, copied verbatim.
    """)
    return


@app.cell
def _(Path, pl):
    _PRED_DIR = Path("../predictions")

    def _baseline(path: str, col: str, val: str, model_key: str) -> pl.DataFrame:
        return (
            pl.read_csv(_PRED_DIR / path)
            .filter(pl.col(col) == val)
            .select(["fold", "y_true", "y_pred"])
            .with_columns(pl.format("{}_baseline", pl.lit(model_key)).alias("method"))
        )

    _base = pl.concat([
        _baseline("6_reweighting_cv.csv.gz", "method", "xgb_mordred_uniform", "xgb_mordred"),
        _baseline("4_fp_model_comparison_1.csv.gz", "method", "macau_chemeleon", "macau_chemeleon"),
    ])
    _filtering_cv = pl.read_csv(_PRED_DIR / "6_filtering_cv.csv.gz")
    _cols = ["fold", "method", "y_true", "y_pred"]
    filt_all = pl.concat([_base.select(_cols), _filtering_cv.select(_cols)])
    return (filt_all,)


@app.cell
def _(calc_regression_metrics, filt_all, pl):
    filt_metrics = calc_regression_metrics(
        filt_all.rename({"fold": "cv_cycle"}).with_columns(pl.lit("random").alias("split")),
        cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0,
    )
    return (filt_metrics,)


@app.cell
def _(Path, filt_metrics, make_stacked_reference_ci_plot, mo, pl):
    # Row order groups each filter directly above its size-matched random control
    # (e.g. "counter" sits right above "counter (random)"), with the baseline last
    # at the bottom as the zero reference. Alternating background bands (via
    # group_bands) reinforce which random control belongs to which filter.
    _FILTERS = ["counter", "cliff", "knn"]

    def _panel_spec(model: str, title: str) -> dict:
        method_order = [f"{model}_baseline"]
        labels = {f"{model}_baseline": "full-data baseline"}
        group_bands = [len(_FILTERS)]  # baseline gets its own band index
        for i, f in enumerate(_FILTERS):
            method_order = [f"{model}_{f}", f"{model}_rand_{f}"] + method_order
            labels[f"{model}_{f}"] = f
            labels[f"{model}_rand_{f}"] = f"{f} (random, matched N)"
            group_bands = [i, i] + group_bands
        methods_used = list(labels.keys())
        return {
            "title": title,
            "df_in": filt_metrics.filter(pl.col("method").is_in(methods_used)),
            "reference": f"{model}_baseline",
            "method_order": method_order,
            "labels": labels,
            "group_bands": group_bands,
        }

    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    _panels = [
        _panel_spec("xgb_mordred", "xgb_mordred"),
        _panel_spec("macau_chemeleon", "macau_chemeleon"),
    ]

    _fig = make_stacked_reference_ci_plot(
        _panels,
        metric="mae",
        figsize=(7, 9),
        xlabel="Mean MAE difference from full-data baseline",
        suptitle="Training-set filtering — ΔMAE vs. full-data baseline\n(each filter paired with its size-matched random control)",
        save_path=_PLOT_DIR / "filtering_delta_mae_ci.png",
    )
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## HPO comparison — MAE multiple-comparison-of-means (MCS) heatmap

    Adapted from `plots/4_ml_optimization_2/analysis5_hpo_mcs_mae.png`, restricted to
    the following methods:

    - `chemeleon` and `chemprop` — baselines from `2_ml_baseline.py`
    - `rf` (renamed `rf_baseline`) — the RF baseline from `2_ml_baseline.py`
    - `rf_mordred` and `xgboost_mordred` (renamed `xgb_mordred`) — Analysis 1
      fingerprint-comparison baselines from `4_ml_optimization_2.py`
    - `rf_mordred_hpo` and `xgb_mordred_hpo` (renamed `rf_hpo`/`xgb_hpo` to avoid
      label overlap in the heatmap) — hyperparameter-optimised models from the
      same notebook

    This is the same all-vs-all pairwise Tukey HSD heatmap style as the original
    (`make_mcs_plot_grid`/`mcs_plot`, copied verbatim from `3_ml_optimization.py`),
    just with a different method subset and no recomputation — all predictions are
    read directly from `predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz`,
    `predictions/4_fp_model_comparison_1.csv.gz`, and
    `predictions/4_hpo_a5_best_5x5cv.csv.gz`.
    """)
    return


@app.cell
def _(np, sns):
    def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
                 ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
                 show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
        """
        Create a multiple comparison of means plot using a heatmap.

        Copied verbatim from `3_ml_optimization.py`.

        Parameters:
        pc (pd.DataFrame): DataFrame containing p-values for pairwise comparisons.
        effect_size (pd.DataFrame): DataFrame containing effect sizes for pairwise comparisons.
        means (pd.Series): Series containing mean values for each group.
        labels (bool): Whether to show labels on the axes. Default is True.
        cmap (str): Colormap to use for the heatmap. Default is None.
        cbar_ax_bbox (tuple): Bounding box for the colorbar axis. Default is None.
        ax (matplotlib.axes.Axes): The axes on which to plot the heatmap. Default is None.
        show_diff (bool): Whether to show the mean differences in the plot. Default is True.
        cell_text_size (int): Font size for the cell text. Default is 16.
        axis_text_size (int): Font size for the axis text. Default is 12.
        show_cbar (bool): Whether to show the colorbar. Default is True.
        reverse_cmap (bool): Whether to reverse the colormap. Default is False.
        vlim (float): Limit for the colormap. Default is None.
        **kwargs: Additional keyword arguments for the heatmap.

        Returns:
        matplotlib.axes.Axes: The axes with the heatmap.
        """
        for key in ['cbar', 'vmin', 'vmax', 'center']:
            if key in kwargs:
                del kwargs[key]

        if not cmap:
            cmap = "coolwarm"
        if reverse_cmap:
            cmap = cmap + "_r"

        significance = pc.copy().astype(object)
        significance[(pc < 0.001) & (pc >= 0)] = '***'
        significance[(pc < 0.01) & (pc >= 0.001)] = '**'
        significance[(pc < 0.05) & (pc >= 0.01)] = '*'
        significance[(pc >= 0.05)] = ''

        np.fill_diagonal(significance.values, '')

        if show_diff:
            annotations = effect_size.round(3).astype(str) + significance
        else:
            annotations = significance

        hax = sns.heatmap(effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
                          annot_kws={"size": cell_text_size},
                          vmin=-2*vlim if vlim else None, vmax=2*vlim if vlim else None, **kwargs)

        if labels:
            label_list = list(means.index)
            x_label_list = [x + f'\n{means.loc[x].round(2)}' for x in label_list]
            y_label_list = [x + f'\n{means.loc[x].round(2)}\n' for x in label_list]
            hax.set_xticklabels(x_label_list, size=axis_text_size, ha='center', va='top', rotation=0,
                                rotation_mode='anchor')
            hax.set_yticklabels(y_label_list, size=axis_text_size, ha='center', va='center', rotation=90,
                                rotation_mode='anchor')

        hax.set_xlabel('')
        hax.set_ylabel('')

        return hax

    return (mcs_plot,)


@app.cell
def _(Optional, Path, math, mcs_plot, plt, rm_tukey_hsd):
    def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                           figsize=(20, 10), direction_dict={}, effect_dict={}, show_diff=True,
                           cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False,
                           save_path: Optional[Path] = None):
        """
        Create a grid of multiple comparison of means plots using Tukey HSD test results.

        Copied verbatim from `3_ml_optimization.py`.

        Parameters:
        df (pd.DataFrame): Input dataframe containing the data.
        stats (list of str): List of statistical metrics to create plots for.
        group_col (str): The column name indicating the groups.
        alpha (float): Significance level for the Tukey HSD test. Default is 0.05.
        figsize (tuple): Size of the figure. Default is (20, 10).
        direction_dict (dict): Dictionary indicating whether to minimize or maximize each metric.
        effect_dict (dict): Dictionary with effect size limits for each metric.
        show_diff (bool): Whether to show the mean differences in the plot. Default is True.
        cell_text_size (int): Font size for the cell text. Default is 16.
        axis_text_size (int): Font size for the axis text. Default is 12.
        title_text_size (int): Font size for the title text. Default is 16.
        sort (bool): Whether to sort the axes. Default is False.
        save_path (Path | None): If provided, the figure is saved to this path before returning.

        Returns:
        plt.Figure: The figure with the grid of heatmaps.
        """
        ncol = 1 if len(stats) == 1 else (2 if len(stats) == 4 else 3)
        nrow = math.ceil(len(stats) / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

        for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
            direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')

        for key in ['r2', 'rho', 'prec', 'recall']:
            effect_dict.setdefault(key, 0.1)
        effect_dict.setdefault('mae', 0.5)
        effect_dict.setdefault('mse', 1.0)

        direction_dict = {k.lower(): v for k, v in direction_dict.items()}
        effect_dict = {k.lower(): v for k, v in effect_dict.items()}

        for i, stat in enumerate(stats):
            stat = stat.lower()

            row = i // ncol
            col = i % ncol

            if stat not in direction_dict:
                raise ValueError(f"Stat '{stat}' is missing in direction_dict. Please set its value.")
            if stat not in effect_dict:
                raise ValueError(f"Stat '{stat}' is missing in effect_dict. Please set its value.")

            reverse_cmap = False
            if direction_dict[stat] == 'minimize':
                reverse_cmap = True

            _, df_means, df_means_diff, pc = rm_tukey_hsd(df, stat, group_col, alpha,
                                                           sort_axes, direction_dict)

            hax = mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                           show_diff=show_diff, ax=ax[row, col], cbar=True,
                           cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                           reverse_cmap=reverse_cmap, vlim=effect_dict[stat])
            hax.set_title(stat.upper(), fontsize=title_text_size)

        if (len(stats) % ncol) != 0:
            for i in range(len(stats), nrow * ncol):
                row = i // ncol
                col = i % ncol
                ax[row, col].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    return (make_mcs_plot_grid,)


@app.cell
def _(Path, calc_regression_metrics, pl):
    _baseline2 = (
        pl.read_csv(Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz"))
        .filter(pl.col("model").is_in(["chemeleon", "chemprop", "rf"]))
        .with_columns(
            pl.when(pl.col("model") == "rf").then(pl.lit("rf_baseline")).otherwise(pl.col("model")).alias("method")
        )
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _fp_comparison = (
        pl.read_csv(Path("../predictions/4_fp_model_comparison_1.csv.gz"))
        .filter(pl.col("method").is_in(["rf_mordred", "xgboost_mordred"]))
        .with_columns(
            pl.when(pl.col("method") == "xgboost_mordred").then(pl.lit("xgb_mordred")).otherwise(pl.col("method")).alias("method")
        )
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _hpo = (
        pl.read_csv(Path("../predictions/4_hpo_a5_best_5x5cv.csv.gz"))
        .filter(pl.col("method").is_in(["rf_mordred_hpo", "xgb_mordred_hpo"]))
        .with_columns(pl.col("method").str.replace("_mordred_hpo", "_hpo"))
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _hpo_cmp_df = pl.concat([_baseline2, _fp_comparison, _hpo])
    hpo_mcs_metrics = calc_regression_metrics(
        _hpo_cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0
    )
    return (hpo_mcs_metrics,)


@app.cell
def _(Path, hpo_mcs_metrics, make_mcs_plot_grid, mo):
    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_mcs_plot_grid(
        hpo_mcs_metrics,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOT_DIR / "hpo_mcs_mae.png",
    )
    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Submission comparison — 5×5 CV, Phase 1, and Phase 2 MAE

    For every submission ever saved to `submissions/*_submission.csv`, this table
    reports:

    - **Phase 1 MAE** — scored directly against
      `data/raw/20260528/dose_response_test_unblinded.csv` (253 compounds), joined
      on `Molecule Name`. Same pattern as `5_unblinded_analysis.py`.
    - **Phase 2 MAE** — scored directly against
      `data/raw/20260703/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv` (260 compounds).
      Same pattern as `7_unblinded_phase2_analysis.py`.
    - **5×5 CV MAE** — only available where the submission's exact model/ensemble
      configuration was evaluated under 5×5 cross-validation and the result saved
      somewhere in `predictions/`; blank where a submission was trained once and
      scored only against the unblinded sets (no matching CV run exists).
    """)
    return


@app.cell
def _(Path, pl):
    _PHASE1_PATH = Path("../data/raw/20260528/dose_response_test_unblinded.csv")
    _PHASE2_PATH = Path("../data/raw/20260703/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv")

    truth_p1 = pl.read_csv(_PHASE1_PATH).select(["Molecule Name", "pEC50"]).rename({"pEC50": "pEC50_true"})
    truth_p2 = pl.read_csv(_PHASE2_PATH).select(["Molecule Name", "pEC50"]).rename({"pEC50": "pEC50_true"})
    return truth_p1, truth_p2


@app.cell
def _(Path, pl, truth_p1, truth_p2):
    def _score_submission(path: Path, truth: pl.DataFrame) -> float | None:
        """Mean absolute error of one submission CSV against a truth table, joined on Molecule Name."""
        pred = (
            pl.read_csv(path)
            .select(["Molecule Name", "pEC50"])
            .rename({"pEC50": "pEC50_pred"})
        )
        joined = pred.join(truth, on="Molecule Name", how="inner")
        if joined.shape[0] == 0:
            return None
        return float((joined["pEC50_pred"] - joined["pEC50_true"]).abs().mean())

    _SUBMISSION_DIR = Path("../submissions")
    _sub_paths = sorted(_SUBMISSION_DIR.glob("*_submission.csv"))

    submission_scores = pl.DataFrame([
        {
            "submission": p.name,
            "phase1_mae": _score_submission(p, truth_p1),
            "phase2_mae": _score_submission(p, truth_p2),
        }
        for p in _sub_paths
    ])
    submission_scores
    return (submission_scores,)


@app.cell
def _(mo):
    mo.md(r"""
    ### 5×5 CV MAE per submission

    Traced back to the exact model/ensemble configuration behind each submission
    file (see notebooks 2–6). Computed directly from the saved out-of-fold
    predictions in `predictions/` wherever an exact-configuration CV run exists —
    mean of the 25 per-fold MAEs, matching `calc_regression_metrics`.

    Two groups use the **closest available** CV run rather than an exact
    configuration match (flagged with *):

    - `3_ens_*` (early test-set ensembles, notebook 3)* — same ensemble weights
      (e.g. `rf0_gbm0_cp1_ch1`) are swept in
      `predictions/3_prediction_ensemble_test.csv.gz`, but that sweep ensembles
      *different* underlying models (default RF/chemprop) than the submission
      (chemprop_depth6 + CheMeleon). Closest available, not exact.
    - `3_rf_mordred3d_test_submission.csv`* — closest CV run
      (`predictions/3_rf_fingerprint_comparison.csv.gz`, `mordred_3d`) used
      `n_estimators=500`, while the submission was trained with the default
      `n_estimators=100`.
    - `5_regen_*`* — reuses the CV MAE of `4_ens_cp5_ch5_rf0_xg13_mc1_tf5`, whose
      tuned weights it shares; its chemprop/CheMeleon/XGBoost components were
      stochastically retrained, so this is representative, not exact.

    Left blank where no CV run exists at all:

    - `6_ens_*_augfilt` — trained once on counter-filtered + semi-pure-augmented
      data and evaluated only against the unblinded sets; no CV run exists.
    - `6_ensemble_calibrated_linear` — reconstructed by reproducing the notebook's
      leakage-free cross-fit linear calibration on the ensemble's out-of-fold
      predictions (`6_ml_optimization_3.py`, `ensemble_oof` / `crossfit_calibrate`),
      since it isn't stored in a `predictions/*.csv.gz` file directly.
    """)
    return


@app.cell
def _(Path, pl):
    def _cv_mae(path: str, col: str, val: str) -> float:
        """Mean of per-fold MAE for one method/model value in a saved 5x5 CV predictions file."""
        df = pl.read_csv(Path("../predictions") / path).filter(pl.col(col) == val)
        return float(
            df.group_by("fold")
            .agg((pl.col("y_pred") - pl.col("y_true")).abs().mean().alias("fold_mae"))["fold_mae"]
            .mean()
        )

    def _cv_mae_ensemble_sweep(w_cp, w_ch, w_rf, w_xg, w_mc, w_tf) -> float:
        sweep = pl.read_csv(Path("../predictions/4_ensemble_sweep_metrics.csv.gz"))
        row = sweep.filter(
            (pl.col("w_cp") == w_cp) & (pl.col("w_ch") == w_ch) & (pl.col("w_rf") == w_rf)
            & (pl.col("w_xg") == w_xg) & (pl.col("w_mc") == w_mc) & (pl.col("w_tf") == w_tf)
        )
        return float(row["mae"][0])

    cv_mae_by_submission: dict[str, float] = {
        "2_ml_baseline_chemeleon_test_submission.csv":
            _cv_mae("2_ml_baseline_5x5cv_random_predictions.csv.gz", "model", "chemeleon"),
        "3_chemprop_depth6_test_submission.csv":
            _cv_mae("3_larger_chemprop_size_test.csv.gz", "model", "chemprop_depth6"),
        "4_chemeleon_hpo_submission.csv":
            _cv_mae("4_hpo_best_5x5cv.csv.gz", "method", "chemeleon_hpo"),
        "4_ens_cp4_ch5_rf0_xg1_mc0_tf5_submission.csv":
            _cv_mae_ensemble_sweep(4, 5, 0, 1, 0, 5),
        "4_ens_cp4_ch5_rf0_xg1_mc1_tf5_submission.csv":
            _cv_mae_ensemble_sweep(4, 5, 0, 1, 1, 5),
        "4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv":
            _cv_mae_ensemble_sweep(5, 5, 0, 1 / 3, 1, 5),
        "4_macau_che_hpo_submission.csv":
            _cv_mae("4_hpo_a5_best_5x5cv.csv.gz", "method", "macau_che_hpo"),
        "4_tabpfn_chemeleon_submission.csv":
            _cv_mae("4_fp_model_comparison_2.csv.gz", "method", "tabpfn_chemeleon"),
        # Closest available CV run — same ensemble weights, different underlying
        # component models (default RF/chemprop rather than chemprop_depth6/CheMeleon).
        "3_ens_rf0_gbm0_cp1_ch1_submission.csv":
            _cv_mae("3_prediction_ensemble_test.csv.gz", "model", "ens_rf0_gbm0_cp1_ch1"),
        "3_ens_rf0_gbm0_cp1_ch2_submission.csv":
            _cv_mae("3_prediction_ensemble_test.csv.gz", "model", "ens_rf0_gbm0_cp1_ch2"),
        "3_ens_rf1_gbm0_cp1_ch2_submission.csv":
            _cv_mae("3_prediction_ensemble_test.csv.gz", "model", "ens_rf1_gbm0_cp1_ch2"),
        "3_ens_rf1_gbm0_cp2_ch2_submission.csv":
            _cv_mae("3_prediction_ensemble_test.csv.gz", "model", "ens_rf1_gbm0_cp2_ch2"),
        # Closest available CV run — same fingerprint/model family, different n_estimators.
        "3_rf_mordred3d_test_submission.csv":
            _cv_mae("3_rf_fingerprint_comparison.csv.gz", "model", "mordred_3d"),
        # Reconstructed (not read from a predictions/*.csv.gz file) — see markdown note above.
        "6_ensemble_calibrated_linear_submission.csv": 0.4710,
    }
    # 5_regen_* shares its tuned weights with 4_ens_cp5_ch5_rf0_xg13_mc1_tf5 — reuse its CV MAE.
    cv_mae_by_submission["5_regen_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv"] = (
        cv_mae_by_submission["4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv"]
    )
    return (cv_mae_by_submission,)


@app.cell
def _(cv_mae_by_submission: dict[str, float], mo, pl, submission_scores):
    submission_comparison = submission_scores.with_columns(
        pl.col("submission").replace_strict(cv_mae_by_submission, default=None).alias("cv_5x5_mae")
    ).select(["submission", "cv_5x5_mae", "phase1_mae", "phase2_mae"])

    mo.ui.table(submission_comparison.to_pandas(), selection=None, pagination=False)
    return (submission_comparison,)


@app.cell
def _(mo):
    mo.md(r"""
    ### CV → holdout generalisation gap

    Scatter of how much each submission's error grew going from 5×5 CV to the two
    unblinded holdout sets: x = Phase 1 MAE − CV MAE, y = Phase 2 MAE − CV MAE.
    Points above/right of the origin did worse on the holdout set than CV predicted;
    the dashed diagonal marks where the Phase 1 and Phase 2 gaps are equal.

    For the two `6_ens_*_augfilt` submissions with no CV run at all, a CV MAE of
    **0.47** is assumed (the tuned-ensemble CV MAE, representative of the
    configuration family they were built from) purely to place them on this plot —
    their gap should be read as approximate, not measured.

    Points are colored by originating notebook (the submission filename's numeric
    prefix).
    """)
    return


@app.cell
def _(Path, plt, submission_comparison):
    _ASSUMED_CV_MAE = 0.47
    _NOTEBOOK_COLORS = {
        "2": "#4C78A8", "3": "#F58518", "4": "#54A24B", "5": "#B279A2", "6": "#E45756",
    }

    _df = submission_comparison.with_columns(
        submission_comparison["submission"].str.split("_").list.get(0).alias("notebook")
    )
    _cv = _df["cv_5x5_mae"].to_list()
    _cv_filled = [v if v is not None else _ASSUMED_CV_MAE for v in _cv]
    _is_assumed = [v is None for v in _cv]
    _p1 = _df["phase1_mae"].to_numpy()
    _p2 = _df["phase2_mae"].to_numpy()
    _notebooks = _df["notebook"].to_list()

    _x = [p1 - cv for p1, cv in zip(_p1, _cv_filled)]
    _y = [p2 - cv for p2, cv in zip(_p2, _cv_filled)]

    fig, ax = plt.subplots(figsize=(7, 6))

    for _nb in sorted(_NOTEBOOK_COLORS):
        _idx = [i for i, n in enumerate(_notebooks) if n == _nb]
        if not _idx:
            continue
        _marker_x = [_x[i] for i in _idx]
        _marker_y = [_y[i] for i in _idx]
        _facecolors = ["none" if _is_assumed[i] else _NOTEBOOK_COLORS[_nb] for i in _idx]
        ax.scatter(
            _marker_x, _marker_y,
            facecolors=_facecolors, edgecolors=_NOTEBOOK_COLORS[_nb],
            linewidths=1.5, s=70, label=f"notebook {_nb}", zorder=3,
        )

    _lim = max(abs(v) for v in _x + _y) * 1.15
    ax.plot([-_lim, _lim], [-_lim, _lim], ls="--", color="gray", lw=1, zorder=1)
    ax.axhline(0, color="black", lw=0.8, zorder=1)
    ax.axvline(0, color="black", lw=0.8, zorder=1)
    ax.set_xlim(-_lim, _lim)
    ax.set_ylim(-_lim, _lim)
    ax.set_xlabel("Phase 1 MAE − 5×5 CV MAE")
    ax.set_ylabel("Phase 2 MAE − 5×5 CV MAE")
    ax.set_title("CV → holdout generalisation gap, by submission\n(open markers = assumed CV MAE = 0.47)")
    ax.legend(title="Notebook", loc="best", frameon=True)
    ax.set_aspect("equal")
    fig.tight_layout()

    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(_PLOT_DIR / "submission_cv_holdout_gap.png", dpi=300, bbox_inches="tight")
    fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Tier analysis anchored on the final submission

    Adapted from `plots/7_unblinded_phase2_analysis/tier1_forest_plot.png`. The
    original forest plot anchors the paired-bootstrap + Holm-Bonferroni tier test on
    the **best-MAE** submission — "is every other submission significantly worse
    than the best one?" This version keeps the exact same statistical procedure
    (1000-iteration paired bootstrap on Phase 2 absolute error, shared resample
    indices, all-pairs Holm-Bonferroni step-down) but anchors on our **final
    submission**, `6_ens_default_augfilt`, instead — "is every other submission
    significantly worse than the one we actually shipped?"

    - **Blue** — Tier 1: not significantly worse than the final submission
      (includes the final submission itself, at Δ = 0)
    - **Red** — Outside Tier 1: significantly worse than the final submission

    Predictions are read directly from every `submissions/*_submission.csv` file
    (the external Gashaw blend excluded, same as the original), scored against
    `data/raw/20260703/pxr-challenge_TEST_PHASE_2_UNBLINDED.csv`.
    """)
    return


@app.cell
def _(Path, pl, truth_p2):
    _SUBMISSION_DIR = Path("../submissions")
    _FINAL_NAME = "6_ens_default_augfilt_submission.csv"

    def _load_and_score(path: Path) -> pl.DataFrame:
        df = (
            pl.read_csv(path)
            .select(["Molecule Name", "pEC50"])
            .rename({"pEC50": "pEC50_pred"})
        )
        return (
            df.join(truth_p2, on="Molecule Name", how="inner")
            .with_columns(
                pl.lit(path.name.removesuffix("_submission.csv")).alias("model_name"),
                (pl.col("pEC50_pred") - pl.col("pEC50_true")).abs().alias("abs_error"),
            )
        )

    _sub_paths = sorted(_SUBMISSION_DIR.glob("*_submission.csv"))
    tier_predictions = pl.concat([_load_and_score(p) for p in _sub_paths])
    FINAL_LABEL = _FINAL_NAME.removesuffix("_submission.csv")
    return FINAL_LABEL, tier_predictions


@app.cell
def _(FINAL_LABEL, np, pl, tier_predictions):
    import itertools as _itertools

    _N_BOOT = 1000
    _ALPHA = 0.05
    _TIER_SEED = 0

    _own_order = tier_predictions["model_name"].unique().sort().to_list()

    _ref_names = (
        tier_predictions.filter(pl.col("model_name") == FINAL_LABEL)
        .sort("Molecule Name")["Molecule Name"].to_list()
    )

    def _abs_err_for(model: str) -> np.ndarray:
        _df = tier_predictions.filter(pl.col("model_name") == model).sort("Molecule Name")
        assert _df["Molecule Name"].to_list() == _ref_names, (
            f"{model} does not cover the same compounds as the reference"
        )
        return _df["abs_error"].to_numpy()

    _ae = {m: _abs_err_for(m) for m in _own_order}
    _n_cmpds = len(_ref_names)

    _rng = np.random.default_rng(_TIER_SEED)
    _boot_idx = [_rng.integers(0, _n_cmpds, _n_cmpds) for _ in range(_N_BOOT)]

    boot_mae_final: dict[str, np.ndarray] = {
        m: np.array([_ae[m][idx].mean() for idx in _boot_idx]) for m in _own_order
    }

    def _two_tailed_p(delta: np.ndarray) -> float:
        return 2.0 * min(float(np.mean(delta > 0)), float(np.mean(delta < 0)))

    _pairs = list(_itertools.combinations(_own_order, 2))
    _pair_rows = []
    for _a, _b in _pairs:
        _delta = boot_mae_final[_a] - boot_mae_final[_b]
        _pair_rows.append({
            "model_a": _a, "model_b": _b,
            "delta_mean": float(_delta.mean()),
            "ci_lo": float(np.quantile(_delta, 0.025)),
            "ci_hi": float(np.quantile(_delta, 0.975)),
            "p_value": _two_tailed_p(_delta),
        })

    _M = len(_pairs)
    pairwise_df_final = (
        pl.DataFrame(_pair_rows)
        .sort("p_value")
        .with_row_index("hb_rank", offset=1)
        .with_columns((pl.lit(_ALPHA) / (pl.lit(_M) - pl.col("hb_rank") + 1)).alias("hb_threshold"))
    )
    _raw_reject = (pairwise_df_final["p_value"] < pairwise_df_final["hb_threshold"]).to_list()
    _stepdown: list[bool] = []
    _still_ok = True
    for _r in _raw_reject:
        _still_ok = _still_ok and _r
        _stepdown.append(_still_ok)
    pairwise_df_final = pairwise_df_final.with_columns(pl.Series("significant", _stepdown, dtype=pl.Boolean))
    return boot_mae_final, pairwise_df_final


@app.cell
def _(
    FINAL_LABEL,
    boot_mae_final: "dict[str, np.ndarray]",
    pairwise_df_final,
    pl,
):
    def _final_vs(model: str) -> dict:
        """Pairwise-row stats for (final, model), oriented as Δ = MAE(model) − MAE(final)."""
        row = pairwise_df_final.filter(
            ((pl.col("model_a") == FINAL_LABEL) & (pl.col("model_b") == model))
            | ((pl.col("model_a") == model) & (pl.col("model_b") == FINAL_LABEL))
        ).row(0, named=True)
        if row["model_a"] == FINAL_LABEL:
            dmean, lo, hi = -row["delta_mean"], -row["ci_hi"], -row["ci_lo"]
        else:
            dmean, lo, hi = row["delta_mean"], row["ci_lo"], row["ci_hi"]
        return {
            "p_value": row["p_value"], "significant": row["significant"],
            "delta_vs_final": dmean, "ci_lo": lo, "ci_hi": hi,
        }

    _own_order = sorted(boot_mae_final.keys())
    _rows = []
    for _m in _own_order:
        _mean_mae = float(boot_mae_final[_m].mean())
        if _m == FINAL_LABEL:
            _rows.append({
                "model_name": _m, "mean_mae": _mean_mae,
                "delta_vs_final": 0.0, "ci_lo": 0.0, "ci_hi": 0.0,
                "tier": "Tier 1 (final)",
            })
        else:
            _fv = _final_vs(_m)
            _rows.append({
                "model_name": _m, "mean_mae": _mean_mae,
                "delta_vs_final": _fv["delta_vs_final"],
                "ci_lo": _fv["ci_lo"], "ci_hi": _fv["ci_hi"],
                "tier": "Tier 1" if not _fv["significant"] else "Outside Tier 1",
            })

    tier_df_final = pl.DataFrame(_rows)
    return (tier_df_final,)


@app.cell
def _(Path, np, plt, tier_df_final):
    _plot_df = tier_df_final.sort("mean_mae", descending=True)
    _labels = _plot_df["model_name"].to_list()
    _delta = _plot_df["delta_vs_final"].to_numpy()
    _lo = _plot_df["ci_lo"].to_numpy()
    _hi = _plot_df["ci_hi"].to_numpy()
    _tiers = _plot_df["tier"].to_list()

    _tier_color = {
        "Tier 1 (final)": "#4e79a7",
        "Tier 1": "#4e79a7",
        "Outside Tier 1": "#e15759",
    }
    _y = np.arange(len(_labels))

    with plt.style.context("seaborn-v0_8-whitegrid"):
        tier_fig, tier_ax = plt.subplots(figsize=(9, max(4.5, 0.45 * len(_labels))), dpi=150)

        for i in range(len(_labels)):
            col = _tier_color[_tiers[i]]
            tier_ax.errorbar(
                _delta[i], _y[i],
                xerr=[[_delta[i] - _lo[i]], [_hi[i] - _delta[i]]],
                fmt="o", color=col, ecolor=col, elinewidth=1.6,
                capsize=3, markersize=6,
            )

        tier_ax.axvline(0.0, color="black", linestyle="--", linewidth=1.2)
        tier_ax.text(0.0, len(_labels) - 0.4, "  Δ = 0 (= final submission)", fontsize=9, va="top")

        tier_ax.set_yticks(_y)
        tier_ax.set_yticklabels(_labels, fontsize=8)
        tier_ax.set_xlabel("Δ MAE vs. final submission  (model − final); 95% bootstrap CI", fontsize=11)
        tier_ax.set_title(
            "Tier 1 analysis — distance from final submission (`6_ens_default_augfilt`)\n"
            "Blue = Tier 1 (not sig. worse than final); red = outside Tier 1",
            fontsize=11,
        )

        from matplotlib.lines import Line2D as _Line2D
        _handles = [
            _Line2D([0], [0], marker="o", color="w", markerfacecolor="#4e79a7", markersize=8, label="Tier 1"),
            _Line2D([0], [0], marker="o", color="w", markerfacecolor="#e15759", markersize=8, label="Outside Tier 1"),
        ]
        tier_ax.legend(handles=_handles, fontsize=9, frameon=True, framealpha=0.9, loc="lower right")
        tier_fig.tight_layout()

        _PLOT_DIR = Path("../plots/webinar")
        _PLOT_DIR.mkdir(parents=True, exist_ok=True)
        tier_fig.savefig(_PLOT_DIR / "tier1_forest_plot_final_anchor.png", dpi=300, bbox_inches="tight")

    tier_fig
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Model × fingerprint comparison — MAE MCS heatmap

    Adapted from Analysis 1 and Analysis 3 in `4_ml_optimization_2.py`
    (`plots/4_ml_optimization_2/analysis1_mcs_mae.png` /
    `analysis3_tabpfn_mcs_mae.png`), combined into a single grid restricted to:

    - `chemeleon` — the CheMeleon baseline from `2_ml_baseline.py`
    - `rf_mordred`, `rf_chemeleon` — Random Forest on Mordred / CheMeleon
      fingerprints (Analysis 1)
    - `macau_mordred`, `macau_chemeleon` — Macau on Mordred / CheMeleon
      fingerprints (Analysis 1)
    - `tabpfn_mordred`, `tabpfn_chemeleon` — TabPFN on Mordred / CheMeleon
      fingerprints (Analysis 3)

    Same all-vs-all pairwise Tukey HSD heatmap style as the other MCS plots in this
    notebook (`make_mcs_plot_grid`/`mcs_plot`), sorted by MAE. No recomputation —
    predictions are read directly from
    `predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz`,
    `predictions/4_fp_model_comparison_1.csv.gz`, and
    `predictions/4_fp_model_comparison_2.csv.gz`.
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, pl):
    _baseline3 = (
        pl.read_csv(Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz"))
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _comparison1 = (
        pl.read_csv(Path("../predictions/4_fp_model_comparison_1.csv.gz"))
        .filter(pl.col("method").is_in(["rf_mordred", "rf_chemeleon", "macau_mordred", "macau_chemeleon"]))
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _comparison2 = (
        pl.read_csv(Path("../predictions/4_fp_model_comparison_2.csv.gz"))
        .filter(pl.col("method").is_in(["tabpfn_mordred", "tabpfn_chemeleon"]))
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
        .select(["cv_cycle", "method", "split", "y_true", "y_pred"])
    )
    _fp_cmp_df = pl.concat([_baseline3, _comparison1, _comparison2]).with_columns(
        pl.col("method")
        .str.replace("tabpfn", "tfn")
        .str.replace("macau", "mc")
        .str.replace("_chemeleon", "_chemel")
    )
    fp_mcs_metrics = calc_regression_metrics(
        _fp_cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0
    )
    return (fp_mcs_metrics,)


@app.cell
def _(Path, fp_mcs_metrics, make_mcs_plot_grid, mo):
    _PLOT_DIR = Path("../plots/webinar")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_mcs_plot_grid(
        fp_mcs_metrics,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOT_DIR / "fp_model_mcs_mae.png",
    )
    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
