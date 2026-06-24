import marimo

__generated_with = "0.23.9"
app = marimo.App(width="medium")


@app.cell
def _(mo):
    mo.md(r"""
    # 6 — ML optimization 3: calibration, reweighting and extra data

    The unblinded analysis (notebook 5 / the
    [blog post](https://github.com/adlvdl/pxr_challenge)) exposed two correctable
    failure modes in the Phase-1 models:

    1. **Regression to the mean at the activity extremes.** Every model overpredicts
       inactive compounds and underpredicts the rare potent ones. On the unblinded
       set the hit zone (pEC50 > 6) was underpredicted by up to a full log unit —
       precisely the region where a screening decision is made.
    2. **Unused auxiliary assay data.** Beyond the dose-response labels, OpenADMET
       released additional measurements that other participants found useful.

    This notebook tests three concrete remedies the blog post proposed as next steps:

    | Analysis | Idea | Targets failure |
    |---|---|---|
    | **1 — Post-hoc calibration** | De-shrink predictions (linear + isotonic) | #1 |
    | **2 — Loss/sample reweighting** | Up-weight extreme-activity compounds in training | #1 |
    | **3 — Oversampling extremes** | Duplicate extreme compounds (works for every model) | #1 |
    | **4 — Extra assay data** | Augment training with the 96-compound *semi-pure* set | #2 |
    | **5 — Filtering training data** | Remove unreliable points (counter / cliffs / kNN noise) | label noise |

    **Protocol.** Strategies are compared with the same **5×5 cross-validation** used
    throughout notebooks 2–4 (random split, seed = 42). To avoid biasing decisions
    toward the now-known Phase-1 labels, **only the single best strategy chosen by CV
    is applied to the unblinded test set** at the very end.

    > Uncertainty estimation — the third idea in the blog's *Next steps* — is left for
    > a later notebook: it does not directly improve the single-point predictions the
    > submission requires.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import os
    # Let PyTorch's libomp and sklearn's libomp coexist in one process (needed
    # because Chemprop's device check imports torch while RF/XGBoost run in-process).
    os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
    # Force the non-interactive Agg backend: figures are only saved / rendered to
    # HTML, never shown in a GUI. Avoids the macOS backend import error under
    # headless execution (and is exactly what marimo needs to render figures).
    os.environ.setdefault("MPLBACKEND", "Agg")

    import gc
    import gzip
    import json
    import math
    import shutil
    import subprocess
    import sys
    import tempfile
    import warnings
    from pathlib import Path
    from typing import Iterator, Optional

    import marimo as mo
    import matplotlib.pyplot as plt
    # Initialise the matplotlib backend + ft2font C extension NOW, before the heavy
    # native libraries (torch/smurff/skfp) load — otherwise a shared-library symbol
    # clash breaks ft2font's import at the first figure.
    plt.close(plt.figure())
    import numpy as np
    import pandas as pd
    import pingouin as pg
    import polars as pl
    import seaborn as sns
    from tqdm.auto import tqdm

    from scipy.stats import spearmanr
    from statsmodels.stats.libqsturng import psturng, qsturng

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    from sklearn.isotonic import IsotonicRegression
    from sklearn.linear_model import LinearRegression
    from sklearn.metrics import (
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        r2_score,
        recall_score,
    )
    from sklearn.model_selection._split import _BaseKFold as BaseKFold

    import xgboost as xgb
    import torch
    import smurff
    import scipy.sparse as sp

    from rdkit import Chem, RDLogger
    from skfp.fingerprints import (
        ECFPFingerprint,
        MACCSFingerprint,
        MordredFingerprint,
        MQNsFingerprint,
    )
    from skfp.preprocessing import ConformerGenerator, MolFromSmilesTransformer

    RDLogger.DisableLog("rdApp.*")
    return (
        BaseKFold,
        Chem,
        ConformerGenerator,
        ECFPFingerprint,
        IsotonicRegression,
        Iterator,
        LinearRegression,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        Optional,
        Path,
        RandomForestClassifier,
        RandomForestRegressor,
        gc,
        gzip,
        json,
        math,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pd,
        pg,
        pl,
        plt,
        precision_score,
        psturng,
        qsturng,
        r2_score,
        recall_score,
        shutil,
        smurff,
        sns,
        sp,
        spearmanr,
        subprocess,
        sys,
        tempfile,
        torch,
        tqdm,
        warnings,
        xgb,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Shared infrastructure

    The model classes, fingerprint helpers, cross-validation splitter and the
    Tukey-HSD multiple-comparison plots are copied verbatim from notebook 4 so the
    5×5 CV results are directly comparable. Two small extensions are made:

    - `RandomForestModel.train` and `BoostedTreesModel.train` accept an optional
      `sample_weight`, needed for the reweighting analysis.
    """)
    return


@app.cell
def _(Path, json, np, subprocess, sys, tempfile):
    # ── CheMeleon embedding subprocess script ─────────────────────────────────
    # CheMeleon (PyTorch) must run in an isolated subprocess to avoid the OpenMP
    # runtime collision between PyTorch's libkmp and sklearn in the parent process.
    # The script is written once here and reused by all analysis cells.
    _CHEMELEON_SCRIPT = "\n".join([
        "import os, json, sys, numpy as np",
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
        "from pathlib import Path",
        "import torch",
        "from chemprop import featurizers",
        "from chemprop import nn as chemnn",
        "from chemprop.models import MPNN",
        "from chemprop.nn import RegressionFFN",
        "from chemprop.data import BatchMolGraph",
        "from rdkit.Chem import MolFromSmiles",
        "smiles_file, out_train, out_test = sys.argv[1], sys.argv[2], sys.argv[3]",
        "with open(smiles_file) as f:",
        "    data = json.load(f)",
        "mp_path = Path.home() / '.chemprop' / 'chemeleon_mp.pt'",
        "ckpt = torch.load(mp_path, weights_only=True)",
        "mp = chemnn.BondMessagePassing(**ckpt['hyper_parameters'])",
        "mp.load_state_dict(ckpt['state_dict'])",
        "model = MPNN(mp, chemnn.MeanAggregation(), RegressionFFN(input_dim=mp.output_dim))",
        "model.eval()",
        "feat = featurizers.SimpleMoleculeMolGraphFeaturizer()",
        "def embed(smiles):",
        "    bmg = BatchMolGraph([feat(MolFromSmiles(s)) for s in smiles])",
        "    with torch.no_grad():",
        "        return model.fingerprint(bmg).numpy(force=True)",
        "np.save(out_train, embed(data['train']))",
        "np.save(out_test,  embed(data['test']))",
    ])
    _CHEMELEON_SCRIPT_PATH = Path(tempfile.gettempdir()) / "chemeleon_embed.py"
    _CHEMELEON_SCRIPT_PATH.write_text(_CHEMELEON_SCRIPT)

    def chemeleon_embed(
        smiles_train: list[str],
        smiles_test: list[str],
        prefix: str = "chemeleon",
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Generate CheMeleon embeddings for train and test SMILES in an isolated
        subprocess, returning (X_train, X_test) as float32 numpy arrays.

        Args:
            smiles_train: SMILES strings for the training set.
            smiles_test:  SMILES strings for the test set.
            prefix: Prefix for the temp files written by this call.

        Returns:
            Tuple (X_train, X_test) of shape (n_train, 2048) and (n_test, 2048).
        """
        tmp = Path(tempfile.gettempdir())
        smi_file   = tmp / f"{prefix}_smiles.json"
        train_file = tmp / f"{prefix}_train"
        test_file  = tmp / f"{prefix}_test"
        smi_file.write_text(json.dumps({"train": smiles_train, "test": smiles_test}))
        result = subprocess.run(
            [sys.executable, str(_CHEMELEON_SCRIPT_PATH),
             str(smi_file), str(train_file), str(test_file)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(f"CheMeleon subprocess failed:\n{result.stderr}")
        X_train = np.load(str(train_file) + ".npy")
        X_test  = np.load(str(test_file)  + ".npy")
        for p in [smi_file, Path(str(train_file) + ".npy"), Path(str(test_file) + ".npy")]:
            p.unlink(missing_ok=True)
        return X_train, X_test

    return (chemeleon_embed,)


@app.cell
def _(np, pl):
    def extract_fp_matrix(df: pl.DataFrame, fp_col: str) -> np.ndarray:
        """Extract a 2-D float32 feature matrix from a fingerprint column."""
        return np.stack(df[fp_col].to_list()).astype(np.float32)

    return (extract_fp_matrix,)


@app.cell
def _(
    ConformerGenerator,
    ECFPFingerprint,
    MACCSFingerprint,
    MQNsFingerprint,
    MolFromSmilesTransformer,
    MordredFingerprint,
    pl,
):
    _fp_dict = {
        "ecfp": ECFPFingerprint,
        "morgan": ECFPFingerprint,
        "maccs": MACCSFingerprint,
        "mordred": MordredFingerprint,
        "mqn": MQNsFingerprint,
    }

    def generate_fingerprint(df: pl.DataFrame, fingerprint_type: str, **kwargs) -> pl.DataFrame:
        """
        Generate molecular fingerprints and add them as a new column to the DataFrame.

        CheMeleon embeddings are not handled here — use the `chemeleon_embed`
        function which runs inference in an isolated subprocess.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the keys in `_fp_dict`.
            **kwargs: Forwarded to the fingerprint class constructor.

        Returns:
            DataFrame with an added column named after `fingerprint_type`.
        """
        if fingerprint_type not in _fp_dict:
            raise ValueError(
                f"Fingerprint type not recognized: {fingerprint_type!r}. "
                f"Valid values: {list(_fp_dict.keys())}"
            )

        smiles_list = df.get_column("smiles").to_list()
        fp_func = _fp_dict[fingerprint_type](**kwargs)

        if fp_func.requires_conformers:
            mol_from_smiles = MolFromSmilesTransformer()
            conf_gen = ConformerGenerator()
            mols_list = conf_gen.transform(mol_from_smiles.transform(smiles_list))
        else:
            mols_list = smiles_list

        fps = fp_func.transform(mols_list)
        return df.with_columns(pl.Series(values=fps, name=fingerprint_type))

    return (generate_fingerprint,)


@app.cell
def _(RandomForestClassifier, RandomForestRegressor, np):
    class RandomForestModel:
        """Scikit-learn Random Forest with a unified fit/predict interface."""

        def __init__(
            self,
            pred_type: str = "regression",
            n_estimators: int = 500,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: str | float = "sqrt",
            random_state: int | None = 42,
            n_jobs: int = -1,
        ) -> None:
            self.pred_type = pred_type
            common = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            if pred_type == "regression":
                self.model = RandomForestRegressor(**common)
            elif pred_type == "classification":
                self.model = RandomForestClassifier(**common)
            else:
                raise ValueError("pred_type must be 'classification' or 'regression'")

        def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            """Fit the forest, optionally weighting individual training samples."""
            self.model.fit(X_train, y_train, sample_weight=sample_weight)

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            if self.pred_type == "classification":
                return self.model.predict_proba(X_test)[:, 1]
            return self.model.predict(X_test)

    return (RandomForestModel,)


@app.cell
def _(np, xgb):
    class BoostedTreesModel:
        """XGBoost gradient-boosted trees with a unified fit/predict interface."""

        def __init__(
            self,
            pred_type: str = "regression",
            n_estimators: int = 500,
            max_depth: int = 6,
            learning_rate: float = 0.05,
            subsample: float = 0.8,
            colsample_bytree: float = 0.8,
            min_child_weight: float = 1.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 1.0,
            early_stopping_rounds: int = 30,
        ) -> None:
            self.pred_type = pred_type
            common = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                early_stopping_rounds=early_stopping_rounds,
                tree_method="hist",
                n_jobs=-1,
            )
            if pred_type == "regression":
                self.model = xgb.XGBRegressor(**common)
            elif pred_type == "classification":
                self.model = xgb.XGBClassifier(**common)
            else:
                raise ValueError("pred_type must be 'classification' or 'regression'")

        def train(
            self,
            X_train: np.ndarray,
            y_train: np.ndarray,
            X_val: np.ndarray,
            y_val: np.ndarray,
            sample_weight: np.ndarray | None = None,
        ) -> None:
            """Fit with an evaluation set for early stopping, optionally weighted."""
            self.model.fit(
                X_train, y_train,
                sample_weight=sample_weight,
                eval_set=[(X_val, y_val)], verbose=False,
            )

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            if self.pred_type == "classification":
                return self.model.predict_proba(X_test)[:, 1]
            return self.model.predict(X_test)

    return (BoostedTreesModel,)


@app.cell
def _(Optional, Path, np, pl, shutil, subprocess, sys, tempfile, torch):
    # ── Chemprop (scratch D-MPNN) via CLI subprocess, with sample weights ─────────
    # Copied from notebook 2's baseline ChempropModel so the `uniform` reference
    # (notebook 2's chemprop CV) and the reweighted runs here share an identical
    # architecture. Training runs in a subprocess (the chemprop CLI) to keep
    # PyTorch out of the notebook kernel. The only extension is optional per-sample
    # `sample_weight`, written as a `weight` column and passed to `chemprop train -w`
    # — Chemprop's native per-datapoint loss weighting.
    _CHEMPROP_BIN = Path(sys.executable).parent / "chemprop"
    _CHEMPROP_LOG = Path("../logs/6_chemprop_cli.log")
    _CHEMPROP_LOG.parent.mkdir(parents=True, exist_ok=True)
    _CHEMPROP_MODEL_DIR = Path(tempfile.gettempdir()) / "chemprop_reweight_model"

    def _chemprop_device() -> str:
        """Best available accelerator for the chemprop CLI."""
        return (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _write_chemprop_csv(
        smiles: list[str],
        targets: "np.ndarray | None",
        path: Path,
        target_col: str,
        weights: "np.ndarray | None" = None,
    ) -> None:
        """Write a chemprop input CSV with smiles, optional target and weight columns."""
        cols: dict = {"smiles": smiles}
        if targets is not None:
            cols[target_col] = targets.flatten().tolist()
        if weights is not None:
            cols["weight"] = weights.flatten().tolist()
        pl.DataFrame(cols).write_csv(path)

    def _run_chemprop_cli(args: list[str]) -> None:
        """Run the chemprop CLI, logging to file; raise with a tail on failure."""
        cmd = [str(_CHEMPROP_BIN)] + args
        with open(_CHEMPROP_LOG, "a") as log:
            log.write(f"\n{'=' * 60}\nCMD: {' '.join(cmd)}\n{'=' * 60}\n")
            result = subprocess.run(cmd, stdout=log, stderr=log, text=True)
        if result.returncode != 0:
            print("\n".join(_CHEMPROP_LOG.read_text().splitlines()[-30:]))
            raise RuntimeError(
                f"chemprop CLI failed (exit {result.returncode}). Log: {_CHEMPROP_LOG}")

    class ChempropModel:
        """Chemprop D-MPNN trained from scratch via the CLI, with optional weights."""

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _CHEMPROP_MODEL_DIR,
            epochs: int = 50,
        ) -> None:
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type = pred_type
            self.model_dir = model_dir
            self.epochs = epochs
            self.target_col: Optional[str] = None

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val: list[str],
            y_val: np.ndarray,
            target_col: str = "target",
            sample_weight: np.ndarray | None = None,
        ) -> None:
            """
            Train via `chemprop train`. When `sample_weight` is given it is written
            as a `weight` column and passed with `-w`; the validation set is given
            unit weights so early stopping tracks an unweighted val loss.
            """
            self.target_col = target_col
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "6_chemprop_train.csv"
            val_csv = tmp / "6_chemprop_val.csv"

            _write_chemprop_csv(X_train, y_train, train_csv, target_col, sample_weight)
            _val_w = np.ones(len(X_val)) if sample_weight is not None else None
            _write_chemprop_csv(X_val, y_val, val_csv, target_col, _val_w)

            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)

            task_type = "regression" if self.pred_type == "regression" else "binary"
            # Pass val_csv twice (val + dummy test) so the CLI tracks val_loss.
            args = [
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _chemprop_device(),
                "--epochs", str(self.epochs),
                "--save-dir", str(self.model_dir),
            ]
            if sample_weight is not None:
                args += ["-w", "weight"]
            _run_chemprop_cli(args)

            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            """Run inference via `chemprop predict`."""
            tmp = Path(tempfile.gettempdir())
            test_csv = tmp / "6_chemprop_test.csv"
            pred_csv = tmp / "6_chemprop_preds.csv"
            model_pt = self.model_dir / "model_0" / "best.pt"

            _write_chemprop_csv(X_test, None, test_csv, self.target_col)
            _run_chemprop_cli([
                "predict",
                "--test-path", str(test_csv),
                "--model-path", str(model_pt),
                "--preds-path", str(pred_csv),
            ])
            preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()
            test_csv.unlink(missing_ok=True)
            pred_csv.unlink(missing_ok=True)
            return preds.flatten()

    return (ChempropModel,)


@app.cell
def _(np, smurff, sp, tempfile):
    # ── Macau (Bayesian matrix factorization, smurff) ────────────────────────────
    # Copied from notebook 4. Uses the fingerprint matrix as row side information.
    # Default params match the `macau_chemeleon` baseline reused as the 1× row.
    class MacauModel:
        """Bayesian matrix factorization (Macau) with fingerprint side information."""

        def __init__(
            self,
            num_latent: int = 16,
            burnin: int = 100,
            nsamples: int = 200,
            univariate: bool = False,
            direct: bool = True,
            num_threads: int | None = None,
            seed: int = 42,
        ) -> None:
            self.num_latent = num_latent
            self.burnin = burnin
            self.nsamples = nsamples
            self.univariate = univariate
            self.direct = direct
            self.num_threads = num_threads
            self.seed = seed
            self._predict_session = None

        def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """Fit via Gibbs sampling; y_train is a 1-D target turned into an (n×1) matrix."""
            n = len(y_train)
            _vals = y_train.flatten().astype(np.float64)
            _mask = ~np.isnan(_vals)
            Y_train = sp.coo_matrix(
                (_vals[_mask], (np.where(_mask)[0], np.zeros(_mask.sum(), dtype=int))),
                shape=(n, 1),
            )
            with tempfile.TemporaryDirectory() as tmpdir:
                import os, logging
                save_name = os.path.join(tmpdir, "smurff_model.hdf5")
                _smurff_logger = logging.getLogger("smurff")
                _root_logger = logging.getLogger()
                _prev_s, _prev_r = _smurff_logger.level, _root_logger.level
                _smurff_logger.setLevel(logging.ERROR)
                _root_logger.setLevel(logging.ERROR)
                try:
                    session = smurff.MacauSession(
                        Ytrain=Y_train,
                        side_info=[X_train.astype(np.float64), None],
                        num_latent=self.num_latent,
                        burnin=self.burnin,
                        nsamples=self.nsamples,
                        univariate=self.univariate,
                        direct=self.direct,
                        num_threads=self.num_threads,
                        seed=self.seed,
                        verbose=0,
                        save_name=save_name,
                        save_freq=1,
                    )
                    session.init()
                    while session.step():
                        pass
                finally:
                    _smurff_logger.setLevel(_prev_s)
                    _root_logger.setLevel(_prev_r)
                self._predict_session = session.makePredictSession()

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """Average posterior Gibbs samples for the test compounds."""
            if self._predict_session is None:
                raise RuntimeError("Call train() before predict().")
            sample_arrays = self._predict_session.predict(
                (X_test.astype(np.float64), slice(None)))
            return np.mean(np.array(sample_arrays), axis=0).flatten()

    return (MacauModel,)


@app.cell
def _(Path, np, subprocess, sys, tempfile):
    # ── TabPFN predictor via CPU subprocess ──────────────────────────────────────
    # TabPFN (torch) runs in an isolated subprocess on CPU — matching notebook 4's
    # baseline (n_estimators=8, ignore_pretraining_limits=True) so the reused 1×
    # `tabpfn_chemeleon` row is comparable. CPU avoids the MPS allocator ceiling.
    _TABPFN_SCRIPT = Path(tempfile.gettempdir()) / "6_tabpfn.py"
    _TABPFN_SCRIPT.write_text("\n".join([
        "import os, sys, numpy as np",
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
        "from dotenv import load_dotenv; from pathlib import Path",
        "load_dotenv(Path('.env'))",
        "import torch",
        "torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))",
        "from tabpfn import TabPFNRegressor",
        "X_train = np.load(sys.argv[1]); y_train = np.load(sys.argv[2])",
        "X_test = np.load(sys.argv[3]); out = sys.argv[4]",
        "model = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=True, device='cpu')",
        "model.fit(X_train, y_train)",
        "np.save(out, model.predict(X_test))",
    ]))

    def tabpfn_predict(
        X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray,
    ) -> np.ndarray:
        """Fit TabPFN on (X_train, y_train) and predict X_test, via CPU subprocess."""
        tmp = Path(tempfile.gettempdir())
        f_xtr, f_ytr, f_xte = tmp / "6_tf_Xtr.npy", tmp / "6_tf_ytr.npy", tmp / "6_tf_Xte.npy"
        f_out = tmp / "6_tf_preds"
        np.save(str(f_xtr), X_train)
        np.save(str(f_ytr), y_train)
        np.save(str(f_xte), X_test)
        res = subprocess.run(
            [sys.executable, str(_TABPFN_SCRIPT),
             str(f_xtr), str(f_ytr), str(f_xte), str(f_out)],
            capture_output=True, text=True, cwd=str(Path("../").resolve()),
        )
        if res.returncode != 0:
            raise RuntimeError(f"TabPFN subprocess failed:\n{res.stderr}")
        preds = np.load(str(f_out) + ".npy")
        for p in [f_xtr, f_ytr, f_xte, Path(str(f_out) + ".npy")]:
            p.unlink(missing_ok=True)
        return preds

    return (tabpfn_predict,)


@app.cell
def _(BaseKFold, Iterator, Optional, np, pl):
    def split_dataset_random(
        df: pl.DataFrame, p_test: float = 0.2, seed: int = 42,
    ) -> tuple[pl.DataFrame, pl.DataFrame]:
        """Randomly split a DataFrame into (train, test) subsets."""
        rng = np.random.default_rng(seed)
        idx = rng.permutation(df.shape[0])
        n_test = int(len(idx) * p_test)
        test_idx, train_idx = idx[:n_test], idx[n_test:]
        return df[train_idx].clone(), df[test_idx].clone()

    class GroupKFoldShuffle(BaseKFold):
        """GroupKFold that shuffles groups before splitting (reproducible)."""

        def __init__(
            self, n_splits: int = 5, *, shuffle: bool = False,
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
                yield np.where(~test_mask)[0], np.where(test_mask)[0]

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
        Generate nested 5×5 CV splits with a random per-molecule assignment.

        Yields:
            (fold_index, outer_index, inner_index, train_df, val_df, test_df).
            `val_df` is None when p_val == 0.
        """
        for i in range(n_outer):
            kf = GroupKFoldShuffle(n_splits=n_inner, random_state=seed + i, shuffle=True)
            groups = list(range(df.shape[0]))  # each molecule is its own group
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
    mean_absolute_error,
    mean_squared_error,
    pl,
    precision_score,
    r2_score,
    recall_score,
    spearmanr,
    warnings,
):
    def calc_regression_metrics(
        df: pl.DataFrame,
        cycle_col: str,
        val_col: str,
        pred_col: str,
        thresh: float,
    ) -> pl.DataFrame:
        """
        Per (cv_cycle, method, split) regression metrics: MAE, MSE, R², ρ, prec, recall.

        Precision/recall use a binary hit label derived from `thresh`.
        """
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
                "rho": float(rho),
                "prec": precision_score(group_df["true_class"].to_numpy(),
                                        group_df["pred_class"].to_numpy()),
                "recall": recall_score(group_df["true_class"].to_numpy(),
                                       group_df["pred_class"].to_numpy()),
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
        """Repeated-measures Tukey HSD across CV folds (subject = cv_cycle)."""
        df_pd = df.to_pandas()

        if sort and direction_dict and metric in direction_dict:
            ascending = direction_dict[metric] == "minimize"
            df_means = df_pd.groupby(group_col).mean(numeric_only=True).sort_values(
                metric, ascending=ascending)
        else:
            df_means = df_pd.groupby(group_col).mean(numeric_only=True)

        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            aov = pg.rm_anova(dv=metric, within=group_col, subject="cv_cycle",
                              data=df_pd, detailed=True)
        mse = aov.loc[1, "MS"]
        df_resid = aov.loc[1, "DF"]

        methods = df_means.index
        n_groups = len(methods)
        n_per_group = df_pd[group_col].value_counts().mean()
        tukey_se = np.sqrt(2 * mse / n_per_group)
        q = qsturng(1 - alpha, n_groups, df_resid)

        num_comparisons = n_groups * (n_groups - 1) // 2
        result_tab = pd.DataFrame(index=range(num_comparisons),
                                  columns=["group1", "group2", "meandiff", "lower", "upper", "p-adj"])
        df_means_diff = pd.DataFrame(index=methods, columns=methods, data=0.0)
        pc = pd.DataFrame(index=methods, columns=methods, data=1.0)

        row_idx = 0
        for i, m1 in enumerate(methods):
            for j, m2 in enumerate(methods):
                if i < j:
                    g1 = df_pd[df_pd[group_col] == m1][metric]
                    g2 = df_pd[df_pd[group_col] == m2][metric]
                    mean_diff = g1.mean() - g2.mean()
                    studentized = np.abs(mean_diff) / tukey_se
                    adjusted_p = psturng(studentized * np.sqrt(2), n_groups, df_resid)
                    if isinstance(adjusted_p, np.ndarray):
                        adjusted_p = adjusted_p[0]
                    lower = mean_diff - (q / np.sqrt(2) * tukey_se)
                    upper = mean_diff + (q / np.sqrt(2) * tukey_se)
                    result_tab.loc[row_idx] = [m1, m2, mean_diff, lower, upper, adjusted_p]
                    pc.loc[m1, m2] = pc.loc[m2, m1] = adjusted_p
                    df_means_diff.loc[m1, m2] = mean_diff
                    df_means_diff.loc[m2, m1] = -mean_diff
                    row_idx += 1

        df_means_diff = df_means_diff.astype(float)
        result_tab["group1_mean"] = result_tab["group1"].map(df_means[metric])
        result_tab["group2_mean"] = result_tab["group2"].map(df_means[metric])
        result_tab.index = result_tab["group1"] + " - " + result_tab["group2"]
        return result_tab, df_means, df_means_diff, pc

    return (rm_tukey_hsd,)


@app.cell
def _(Optional, Path, math, np, plt, rm_tukey_hsd, sns):
    def mcs_plot(pc, effect_size, means, labels=True, cmap=None, ax=None,
                 show_diff=True, cell_text_size=16, axis_text_size=12,
                 show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
        """Multiple-comparison-of-means heatmap (Tukey HSD)."""
        for key in ["cbar", "vmin", "vmax", "center"]:
            kwargs.pop(key, None)
        cmap = cmap or "coolwarm"
        if reverse_cmap:
            cmap = cmap + "_r"

        significance = pc.copy().astype(object)
        significance[(pc < 0.001) & (pc >= 0)] = "***"
        significance[(pc < 0.01) & (pc >= 0.001)] = "**"
        significance[(pc < 0.05) & (pc >= 0.01)] = "*"
        significance[(pc >= 0.05)] = ""
        np.fill_diagonal(significance.values, "")
        annotations = effect_size.round(2).astype(str) + significance if show_diff else significance

        hax = sns.heatmap(
            effect_size, cmap=cmap, annot=annotations, fmt="", cbar=show_cbar, ax=ax,
            annot_kws={"size": cell_text_size},
            vmin=-2 * vlim if vlim else None, vmax=2 * vlim if vlim else None, **kwargs,
        )
        if labels:
            label_list = list(means.index)
            hax.set_xticklabels([f"{x}\n{means.loc[x]:.3f}" for x in label_list],
                                size=axis_text_size, ha="center", va="top", rotation=0)
            hax.set_yticklabels([f"{x}\n{means.loc[x]:.3f}\n" for x in label_list],
                                size=axis_text_size, ha="center", va="center", rotation=90)
        hax.set_xlabel("")
        hax.set_ylabel("")
        return hax

    def make_mcs_plot_grid(
        df, stats: list[str], group_col: str, alpha: float = 0.05,
        figsize: tuple = (10, 8), direction_dict: dict | None = None,
        effect_dict: dict | None = None, show_diff: bool = True,
        cell_text_size: int = 14, axis_text_size: int = 11, title_text_size: int = 15,
        sort_axes: bool = True, save_path: Optional[Path] = None,
    ) -> "plt.Figure":
        """Grid of Tukey-HSD MCS heatmaps, one panel per metric."""
        direction_dict = dict(direction_dict or {})
        effect_dict = dict(effect_dict or {})
        for key in ["r2", "rho", "prec", "recall", "mae", "mse"]:
            direction_dict.setdefault(
                key, "maximize" if key in ["r2", "rho", "prec", "recall"] else "minimize")
        for key in ["r2", "rho", "prec", "recall"]:
            effect_dict.setdefault(key, 0.1)
        effect_dict.setdefault("mae", 0.05)
        effect_dict.setdefault("mse", 0.1)

        ncol = 1 if len(stats) == 1 else (2 if len(stats) == 4 else 3)
        nrow = math.ceil(len(stats) / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

        for i, stat in enumerate(stats):
            stat = stat.lower()
            _, df_means, df_means_diff, pc = rm_tukey_hsd(
                df, stat, group_col, alpha, sort_axes, direction_dict)
            mcs_plot(pc, effect_size=df_means_diff, means=df_means[stat],
                     show_diff=show_diff, ax=ax[i // ncol, i % ncol], cbar=True,
                     cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                     reverse_cmap=(direction_dict.get(stat) == "minimize"),
                     vlim=effect_dict.get(stat))
            ax[i // ncol, i % ncol].set_title(stat.upper(), fontsize=title_text_size)

        for i in range(len(stats), nrow * ncol):
            ax[i // ncol, i % ncol].set_visible(False)
        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    return make_mcs_plot_grid, mcs_plot


@app.cell
def _(np, pl, plt):
    # ── Activity-bin helpers (shared by all analyses) ────────────────────────────
    # The same four pEC50 bins used in the unblinded analysis (notebook 5).
    BIN_ORDER = ["<4 (inactive)", "4–5 (weak)", "5–6 (moderate)", ">6 (hit zone)"]

    def activity_bin_expr(col: str = "y_true") -> pl.Expr:
        """Polars expression assigning each row to one of the four pEC50 bins."""
        return (
            pl.when(pl.col(col) >= 6.0).then(pl.lit(">6 (hit zone)"))
            .when(pl.col(col) >= 5.0).then(pl.lit("5–6 (moderate)"))
            .when(pl.col(col) >= 4.0).then(pl.lit("4–5 (weak)"))
            .otherwise(pl.lit("<4 (inactive)"))
            .alias("pec50_bin")
        )

    def bias_by_bin_table(
        df: pl.DataFrame, method_col: str, true_col: str, pred_col: str,
    ) -> pl.DataFrame:
        """Mean signed error (pred − true) per method × activity bin."""
        return (
            df.with_columns(activity_bin_expr(true_col))
            .group_by([method_col, "pec50_bin"])
            .agg(
                (pl.col(pred_col) - pl.col(true_col)).mean().alias("mean_error"),
                (pl.col(pred_col) - pl.col(true_col)).abs().mean().alias("mean_abs_error"),
                pl.len().alias("n"),
            )
        )

    def plot_bias_heatmap(
        bias_df: pl.DataFrame, method_order: list[str], title: str, save_path,
    ) -> "plt.Figure":
        """Heatmap of mean signed error: rows = methods, columns = activity bins."""
        matrix = np.full((len(method_order), len(BIN_ORDER)), np.nan)
        for row in bias_df.iter_rows(named=True):
            if row["method"] in method_order and row["pec50_bin"] in BIN_ORDER:
                mi = method_order.index(row["method"])
                bi = BIN_ORDER.index(row["pec50_bin"])
                matrix[mi, bi] = row["mean_error"]
        abs_max = float(np.nanmax(np.abs(matrix)))

        with plt.style.context("seaborn-v0_8-whitegrid"):
            fig, ax = plt.subplots(figsize=(7, 0.7 * len(method_order) + 1.8), dpi=150)
            ax.grid(False)
            im = ax.imshow(matrix, cmap="RdBu_r", vmin=-abs_max, vmax=abs_max,
                           aspect="auto", interpolation="nearest")
            for mi in range(len(method_order)):
                for bi in range(len(BIN_ORDER)):
                    val = matrix[mi, bi]
                    if not np.isnan(val):
                        col = "white" if abs(val) > 0.6 * abs_max else "black"
                        ax.text(bi, mi, f"{val:+.3f}", ha="center", va="center",
                                fontsize=9, color=col)
            ax.set_xticks(range(len(BIN_ORDER)))
            ax.set_xticklabels(BIN_ORDER, fontsize=9, rotation=-20, ha="left")
            ax.set_yticks(range(len(method_order)))
            ax.set_yticklabels(method_order, fontsize=10)
            ax.set_title(title, fontsize=12)
            cb = fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03)
            cb.set_label("Mean error (pred − true)", fontsize=9)
            fig.tight_layout()
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig

    return BIN_ORDER, bias_by_bin_table, plot_bias_heatmap


@app.cell
def _(Path, gc, pl):
    # ── Dose-response training set — the single shared target for every analysis ──
    DR_TRAIN = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "molecule_names", "pEC50_dr"])
    )
    PLOTS_DIR = Path("../plots/6_ml_optimization_3")
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    PRED_DIR = Path("../predictions")
    gc.collect()
    print(f"DR training compounds: {DR_TRAIN.shape[0]}")
    DR_TRAIN.head()
    return DR_TRAIN, PLOTS_DIR, PRED_DIR


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 1 — Post-hoc calibration (de-shrinking)

    The cheapest fix for regression-to-the-mean is a one-dimensional map applied
    *after* training that pulls predictions back toward the truth: low predictions
    down, high predictions up. Two variants are tested, exactly as proposed in the
    blog post:

    - **Linear de-shrink** — `LinearRegression` of true on predicted pEC50. A fitted
      slope > 1 expands the prediction range.
    - **Isotonic** — `IsotonicRegression`, a flexible monotonic map that can correct
      curvature the linear fit cannot.

    **Leakage-free evaluation.** Calibration is model-agnostic, so it reuses the
    out-of-fold 5×5 CV predictions already saved in notebook 4 — no retraining. For
    each test fold the calibrator is fit on the *sibling* inner folds of the same
    outer repeat (a disjoint set of compounds) and then applied to the held-out
    fold. This keeps the compounds used to fit the calibrator separate from those
    used to score it, so the reported numbers are honest.

    Calibration is evaluated for all six HPO component models **and** the submitted
    ensemble (`cp5·ch5·rf0·xg⅓·mc1·tf5`), reconstructed from the per-model OOF
    predictions.
    """)
    return


@app.cell
def _(PRED_DIR, pl):
    # ── Load the saved 5×5 CV out-of-fold predictions for all six models ─────────
    def _load_oof(path: str, methods: list[str]) -> pl.DataFrame:
        return (
            pl.read_csv(PRED_DIR / path)
            .filter(pl.col("method").is_in(methods))
            .select(["inchikey", "molecule_names", "fold", "outer_fold",
                     "inner_fold", "method", "y_true", "y_pred"])
        )

    oof_components = pl.concat([
        _load_oof("4_hpo_best_5x5cv.csv.gz", ["chemprop_hpo", "chemeleon_hpo"]),
        _load_oof("4_hpo_a5_best_5x5cv.csv.gz", ["macau_che_hpo", "rf_mordred_hpo", "xgb_mordred_hpo"]),
        _load_oof("4_fp_model_comparison_2.csv.gz", ["tabpfn_chemeleon"]),
    ])

    # ── Reconstruct the submitted ensemble OOF: weighted average per compound ────
    # Weights from notebook 4's best submission (rf weight is 0, so rf is excluded).
    _ENS_WEIGHTS = {
        "chemprop_hpo": 5.0, "chemeleon_hpo": 5.0,
        "xgb_mordred_hpo": 1.0 / 3.0, "macau_che_hpo": 1.0, "tabpfn_chemeleon": 5.0,
    }
    _wide = (
        oof_components
        .filter(pl.col("method").is_in(list(_ENS_WEIGHTS)))
        .pivot(values="y_pred",
               index=["inchikey", "molecule_names", "fold", "outer_fold", "inner_fold"],
               on="method", aggregate_function="first")
    )
    _truth = oof_components.select(["inchikey", "outer_fold", "y_true"]).unique(
        subset=["inchikey", "outer_fold"])
    _num = sum(w * pl.col(m) for m, w in _ENS_WEIGHTS.items())
    _den = sum(_ENS_WEIGHTS.values())
    ensemble_oof = (
        _wide.join(_truth, on=["inchikey", "outer_fold"], how="left")
        .with_columns((_num / _den).alias("y_pred"), pl.lit("ensemble").alias("method"))
        .select(["inchikey", "molecule_names", "fold", "outer_fold", "inner_fold",
                 "method", "y_true", "y_pred"])
    )

    cv_oof_all = pl.concat([oof_components, ensemble_oof])
    print("OOF rows per method:")
    print(cv_oof_all.group_by("method").len().sort("method"))
    return cv_oof_all, ensemble_oof


@app.cell
def _(IsotonicRegression, LinearRegression, np, pl):
    def crossfit_calibrate(df_method: pl.DataFrame, kind: str) -> pl.DataFrame:
        """
        Apply leakage-free cross-fit calibration to one method's OOF predictions.

        For each of the 25 folds the calibrator is fit on the other four inner
        folds of the same outer repeat (disjoint compounds) and applied to the
        held-out fold.

        Args:
            df_method: OOF rows for a single method (cols: fold, outer_fold,
                inner_fold, y_true, y_pred, ...).
            kind: "raw" (identity), "linear" or "isotonic".

        Returns:
            The input rows with an added `y_cal` column.
        """
        parts: list[pl.DataFrame] = []
        for fold in range(25):
            outer = fold // 5
            fit = df_method.filter((pl.col("outer_fold") == outer) & (pl.col("fold") != fold))
            test = df_method.filter(pl.col("fold") == fold)
            xp_fit = fit["y_pred"].to_numpy()
            yt_fit = fit["y_true"].to_numpy()
            xp_test = test["y_pred"].to_numpy()

            if kind == "linear":
                cal = LinearRegression().fit(xp_fit.reshape(-1, 1), yt_fit)
                y_cal = cal.predict(xp_test.reshape(-1, 1))
            elif kind == "isotonic":
                cal = IsotonicRegression(out_of_bounds="clip").fit(xp_fit, yt_fit)
                y_cal = cal.predict(xp_test)
            elif kind == "raw":
                y_cal = xp_test
            else:
                raise ValueError(f"Unknown calibration kind: {kind!r}")

            parts.append(test.with_columns(pl.Series("y_cal", np.asarray(y_cal))))
        return pl.concat(parts)

    return (crossfit_calibrate,)


@app.cell
def _(crossfit_calibrate, cv_oof_all, pl):
    # ── Run all three calibration variants for every method ──────────────────────
    _records: list[pl.DataFrame] = []
    for _method in cv_oof_all["method"].unique().to_list():
        _dfm = cv_oof_all.filter(pl.col("method") == _method)
        for _kind in ["raw", "linear", "isotonic"]:
            _cal = crossfit_calibrate(_dfm, _kind).with_columns(
                pl.lit(_method).alias("base_model"),
                pl.lit(_kind).alias("calibration"),
                pl.format("{}__{}", pl.lit(_method), pl.lit(_kind)).alias("method_cal"),
            )
            _records.append(_cal)

    calib_cv = pl.concat(_records)
    print(f"Calibrated CV rows: {calib_cv.shape[0]:,} "
          f"({calib_cv['base_model'].n_unique()} models × 3 variants)")
    calib_cv
    return (calib_cv,)


@app.cell
def _(calib_cv, mean_absolute_error, mo, pl):
    # ── Summary: overall + extreme-zone metrics per model × calibration ──────────
    def _zone_mae(df: pl.DataFrame, lo: float | None, hi: float | None) -> float:
        sub = df
        if lo is not None:
            sub = sub.filter(pl.col("y_true") >= lo)
        if hi is not None:
            sub = sub.filter(pl.col("y_true") < hi)
        if sub.shape[0] == 0:
            return float("nan")
        return mean_absolute_error(sub["y_true"].to_numpy(), sub["y_cal"].to_numpy())

    _rows = []
    for (_base, _cal), _grp in calib_cv.group_by(["base_model", "calibration"]):
        _yt = _grp["y_true"].to_numpy()
        _yc = _grp["y_cal"].to_numpy()
        _hit = _grp.filter(pl.col("y_true") >= 6.0)
        _inact = _grp.filter(pl.col("y_true") < 4.0)
        _rows.append({
            "base_model": _base,
            "calibration": _cal,
            "MAE": round(mean_absolute_error(_yt, _yc), 4),
            "hitzone_MAE": round(_zone_mae(_grp, 6.0, None), 3),
            "hitzone_bias": round(float((_hit["y_cal"] - _hit["y_true"]).mean()), 3),
            "inactive_bias": round(float((_inact["y_cal"] - _inact["y_true"]).mean()), 3),
        })

    calib_summary = (
        pl.DataFrame(_rows)
        .with_columns(pl.col("calibration").cast(pl.Enum(["raw", "linear", "isotonic"])))
        .sort(["base_model", "calibration"])
    )

    mo.vstack([
        mo.md("### Calibration summary — overall MAE and extreme-zone behaviour\n"
              "`hitzone` = pEC50 ≥ 6; `inactive` = pEC50 < 4. "
              "Bias is mean signed error (pred − true)."),
        mo.ui.table(calib_summary.to_pandas(), selection=None, pagination=False),
    ])
    return (calib_summary,)


@app.cell
def _(PLOTS_DIR, calib_summary, mo, np, plt):
    # ── Visual comparison of calibration effect across models and ensemble ────────
    _models = calib_summary["base_model"].unique().sort().to_list()
    _calibrations = ["raw", "linear", "isotonic"]
    _metrics = ["MAE", "hitzone_MAE", "hitzone_bias", "inactive_bias"]
    _titles = ["Overall MAE", "Hit-zone MAE (pEC50 ≥ 6)",
               "Hit-zone bias (pred − true)", "Inactive bias (pred − true)"]
    _colors = {"raw": "#4e79a7", "linear": "#e15759", "isotonic": "#59a14f"}

    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig_cal, axes_cal = plt.subplots(2, 2, figsize=(14, 9), dpi=140)
        x_pos = np.arange(len(_models))
        bar_w = 0.25

        for ax, metric, title in zip(axes_cal.flatten(), _metrics, _titles):
            for k, cal in enumerate(_calibrations):
                vals = []
                for model in _models:
                    row = calib_summary.filter(
                        (calib_summary["base_model"] == model)
                        & (calib_summary["calibration"] == cal)
                    )
                    vals.append(float(row[metric][0]) if row.shape[0] else float("nan"))
                offset = (k - 1) * bar_w
                ax.bar(x_pos + offset, vals, bar_w, label=cal,
                       color=_colors[cal], edgecolor="white", linewidth=0.5)

            if metric in ("hitzone_bias", "inactive_bias"):
                ax.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=0)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(_models, fontsize=8, rotation=30, ha="right")
            ax.set_title(title, fontsize=11)

        handles, labels = axes_cal[0, 0].get_legend_handles_labels()
        fig_cal.legend(handles, labels, loc="upper center", ncol=3, fontsize=10,
                       title="calibration", title_fontsize=10,
                       framealpha=1.0, facecolor="white", edgecolor="#cccccc",
                       bbox_to_anchor=(0.5, 1.02))
        fig_cal.suptitle(
            "Effect of post-hoc calibration across models and ensemble",
            fontsize=13, y=1.06)
        fig_cal.tight_layout()
        fig_cal.savefig(PLOTS_DIR / "analysis1_calibration_comparison.png",
                        dpi=300, bbox_inches="tight")

    mo.vstack([
        mo.md("### Calibration comparison across all models\n"
              "Grouped bars show the raw (uncalibrated), linear, and isotonic "
              "calibration variants for each base model. The bias panels show signed "
              "error: positive = overprediction, negative = underprediction. Ideal "
              "bias is zero."),
        mo.as_html(fig_cal),
    ])
    return


@app.cell
def _(
    PLOTS_DIR,
    calc_regression_metrics,
    calib_cv,
    make_mcs_plot_grid,
    mo,
    pl,
):
    # ── MCS (Tukey HSD) on the ensemble: raw vs linear vs isotonic ───────────────
    _ens = (
        calib_cv
        .filter(pl.col("base_model") == "ensemble")
        .select([
            pl.col("fold").alias("cv_cycle"),
            pl.col("calibration").alias("method"),
            pl.lit("random").alias("split"),
            "y_true",
            pl.col("y_cal").alias("y_pred"),
        ])
    )
    _metrics = calc_regression_metrics(_ens, "cv_cycle", "y_true", "y_pred", thresh=4.0)

    _fig = make_mcs_plot_grid(
        _metrics, stats=["mae"], group_col="method",
        figsize=(7, 6), effect_dict={"mae": 0.02},
        save_path=PLOTS_DIR / "analysis1_calibration_mcs_mae.png",
    )
    mo.vstack([
        mo.md("### Ensemble — does calibration significantly change CV MAE?\n"
              "Tukey HSD across the 25 folds. Cells show the mean MAE difference; "
              "`*` marks p < 0.05."),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(PLOTS_DIR, bias_by_bin_table, calib_cv, mo, pl, plot_bias_heatmap):
    # ── Per-bin signed-error heatmap for the ensemble (raw vs linear vs isotonic) ─
    _ens = calib_cv.filter(pl.col("base_model") == "ensemble").with_columns(
        pl.col("calibration").alias("method"))
    _bias = bias_by_bin_table(_ens, "method", "y_true", "y_cal")
    _fig = plot_bias_heatmap(
        _bias, method_order=["raw", "linear", "isotonic"],
        title="Ensemble mean signed error by activity bin\n(red = overpredict, blue = underpredict)",
        save_path=PLOTS_DIR / "analysis1_calibration_bias_heatmap.png",
    )
    mo.vstack([
        mo.md("### Where does calibration act?\n"
              "If de-shrinking works, the `<4` column should move toward 0 from the "
              "right (less overprediction) and the `>6` column toward 0 from the left "
              "(less underprediction)."),
        mo.as_html(_fig),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While the methods we used worked in that the bias is closer to zero in all cases, the effect is very small.
    The isotonic method was slightly better than linear.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 2 — Loss / sample reweighting

    Calibration acts after the fact; reweighting changes what the model learns. By
    up-weighting compounds at the activity extremes, the model is pushed to commit
    to potent and inactive compounds instead of hedging toward the dense middle of
    the distribution.

    Three weighting schemes are compared:

    | Scheme | Weight for a compound with label *y* |
    |---|---|
    | `uniform` | 1 (baseline) |
    | `invdensity` | ∝ 1 / (population of *y*'s pEC50 bin) — up-weights rare regions |
    | `distance` | 1 + \|*y* − median(*y*)\| — grows linearly toward the extremes |

    **Which models can be reweighted?** Only models with a per-sample loss weight:

    | Model | Reweighting | Mechanism |
    |---|---|---|
    | XGBoost · Mordred | ✓ | sklearn `sample_weight` |
    | RF · CheMeleon | ✓ | sklearn `sample_weight` |
    | Chemprop · scratch | ✓ | `MoleculeDatapoint.weight` / `chemprop train -w` |
    | TabPFN | ✗ | in-context model — no trainable loss to weight |
    | Macau (smurff) | ✗ | no per-observation weight in the API |

    So **TabPFN and Macau are out**; Chemprop is added as the deep-model case. To save
    compute, only `invdensity` and `distance` are trained for Chemprop here — its
    `uniform` baseline is taken from notebook 2 (identical folds and architecture).

    All weights are normalised to mean 1 so the effective regularisation strength is
    unchanged. Same 5×5 CV splits as everywhere else (seed = 42, 10 % inner
    validation for early stopping). Tree-model features are pre-computed once and
    indexed per fold so the only thing that varies between schemes is the weighting.
    """)
    return


@app.cell
def _(
    Chem,
    DR_TRAIN,
    PRED_DIR,
    chemeleon_embed,
    extract_fp_matrix,
    generate_fingerprint,
    np,
    pl,
):
    # ── Pre-compute Mordred + CheMeleon features for every compound we will need ──
    # The feature of a molecule never changes between folds, so we embed/fingerprint
    # the union of (DR training set + semi-pure set) exactly once and cache to disk.
    _SEMIPURE_RAW = pl.read_csv(
        "../data/raw/20260619/pxr-challenge_96-compound-uscale-semi-pure_TRAIN.csv")

    def _ik(smi: str) -> str | None:
        m = Chem.MolFromSmiles(smi)
        return Chem.MolToInchiKey(m) if m else None

    semipure = (
        _SEMIPURE_RAW
        .rename({"SMILES": "smiles",
                 "Corrected Semi-Pure pEC50 (log)": "pEC50_corrected"})
        .with_columns(
            pl.col("pEC50_corrected").cast(pl.Float64, strict=False),
            pl.col("smiles").map_elements(_ik, return_dtype=pl.Utf8).alias("inchikey"),
        )
        .filter(pl.col("pEC50_corrected").is_not_null() & pl.col("inchikey").is_not_null())
        .select(["smiles", "inchikey", "pEC50_corrected"])
        .unique(subset=["inchikey"], keep="first")
    )

    _feat_df = (
        pl.concat([
            DR_TRAIN.select(["inchikey", "smiles"]),
            semipure.select(["inchikey", "smiles"]),
        ])
        .unique(subset=["inchikey"], keep="first")
    )

    FEATURE_CACHE_PATH = PRED_DIR / "6_feature_cache.npz"
    if FEATURE_CACHE_PATH.exists():
        _cache = np.load(FEATURE_CACHE_PATH, allow_pickle=True)
        feature_cache = {
            "iks": list(_cache["iks"]),
            "mordred": _cache["mordred"],
            "chemeleon": _cache["chemeleon"],
        }
        print(f"Loaded feature cache: {FEATURE_CACHE_PATH.name}")
    else:
        print(f"Computing features for {_feat_df.shape[0]} unique compounds …")
        _che, _ = chemeleon_embed(_feat_df["smiles"].to_list(),
                                  _feat_df["smiles"].to_list()[:1], prefix="a2cache")
        _mor_df = generate_fingerprint(_feat_df, "mordred")
        _mor = extract_fp_matrix(_mor_df, "mordred")
        # Drop descriptor columns that are NaN for any compound (consistent mask).
        _mask = ~np.isnan(_mor).any(axis=0)
        _mor = _mor[:, _mask]
        feature_cache = {
            "iks": _feat_df["inchikey"].to_list(),
            "mordred": _mor.astype(np.float32),
            "chemeleon": _che.astype(np.float32),
        }
        np.savez_compressed(
            FEATURE_CACHE_PATH,
            iks=np.array(feature_cache["iks"], dtype=object),
            mordred=feature_cache["mordred"],
            chemeleon=feature_cache["chemeleon"],
        )
        print(f"Cached → {FEATURE_CACHE_PATH.name}  "
              f"(mordred {feature_cache['mordred'].shape}, "
              f"chemeleon {feature_cache['chemeleon'].shape})")

    _ik2idx = {ik: i for i, ik in enumerate(feature_cache["iks"])}

    def gather_X(inchikeys: list[str], kind: str) -> np.ndarray:
        """Return the cached feature matrix (kind = 'mordred'|'chemeleon') for rows."""
        idx = [_ik2idx[ik] for ik in inchikeys]
        return feature_cache[kind][idx]

    print(f"semi-pure usable compounds: {semipure.shape[0]}")
    return gather_X, semipure


@app.cell
def _(np):
    def compute_sample_weights(
        y: np.ndarray, scheme: str, n_bins: int = 20, alpha: float = 1.0,
    ) -> np.ndarray:
        """
        Per-sample training weights for a reweighting scheme (mean-normalised to 1).

        Args:
            y: Training labels (pEC50).
            scheme: "uniform", "invdensity" or "distance".
            n_bins: Number of histogram bins for inverse-density weighting.
            alpha: Slope for the distance-from-median scheme.
        """
        if scheme == "uniform":
            w = np.ones_like(y, dtype=float)
        elif scheme == "invdensity":
            counts, edges = np.histogram(y, bins=n_bins)
            idx = np.clip(np.digitize(y, edges[1:-1]), 0, n_bins - 1)
            w = 1.0 / np.maximum(counts[idx], 1).astype(float)
        elif scheme == "distance":
            w = 1.0 + alpha * np.abs(y - np.median(y))
        else:
            raise ValueError(f"Unknown weighting scheme: {scheme!r}")
        return w * (len(w) / w.sum())

    return (compute_sample_weights,)


@app.cell
def _(
    BoostedTreesModel,
    DR_TRAIN,
    PRED_DIR,
    RandomForestModel,
    compute_sample_weights,
    gather_X,
    gc,
    generate_cv_splits_random,
    gzip,
    pl,
    tqdm,
):
    # ── 5×5 CV: two models × three weighting schemes ─────────────────────────────
    _OUT = PRED_DIR / "6_reweighting_cv.csv.gz"
    _SCHEMES = ["uniform", "invdensity", "distance"]
    _MODELS = [("xgb_mordred", "mordred"), ("rf_chemeleon", "chemeleon")]

    if _OUT.exists():
        reweight_cv = pl.read_csv(_OUT)
        print(f"Found {_OUT.name} — skipping training ({reweight_cv.shape[0]:,} rows).")
    else:
        _records: list[dict] = []
        for _fold, _outer, _inner, _tr, _va, _te in tqdm(
            generate_cv_splits_random(DR_TRAIN, n_outer=5, n_inner=5, seed=42, p_val=0.1),
            total=25, desc="reweighting CV", unit="fold",
        ):
            _y_tr = _tr["pEC50_dr"].to_numpy()
            _y_va = _va["pEC50_dr"].to_numpy()
            _y_te = _te["pEC50_dr"].to_numpy()

            for _model_key, _kind in _MODELS:
                _X_tr = gather_X(_tr["inchikey"].to_list(), _kind)
                _X_va = gather_X(_va["inchikey"].to_list(), _kind)
                _X_te = gather_X(_te["inchikey"].to_list(), _kind)

                for _scheme in _SCHEMES:
                    _w = compute_sample_weights(_y_tr, _scheme)
                    if _model_key == "xgb_mordred":
                        _m = BoostedTreesModel(pred_type="regression")
                        _m.train(_X_tr, _y_tr, _X_va, _y_va, sample_weight=_w)
                    else:
                        _m = RandomForestModel(pred_type="regression")
                        _m.train(_X_tr, _y_tr, sample_weight=_w)
                    _pred = _m.predict(_X_te)
                    del _m
                    gc.collect()

                    for _ik, _yt, _yp in zip(_te["inchikey"].to_list(),
                                             _y_te.tolist(), _pred.tolist()):
                        _records.append({
                            "inchikey": _ik, "fold": _fold,
                            "model": _model_key, "scheme": _scheme,
                            "method": f"{_model_key}_{_scheme}",
                            "y_true": _yt, "y_pred": _yp,
                        })

        reweight_cv = pl.DataFrame(_records)
        with gzip.open(_OUT, "wb") as _f:
            reweight_cv.write_csv(_f)
        print(f"Wrote {reweight_cv.shape[0]:,} rows → {_OUT.name}")
    return (reweight_cv,)


@app.cell
def _(
    ChempropModel,
    DR_TRAIN,
    PRED_DIR,
    compute_sample_weights,
    generate_cv_splits_random,
    gzip,
    pl,
    tqdm,
):
    # ── Reweighting on a scratch Chemprop D-MPNN (slow — checkpointed per fold) ───
    # Chemprop is the one deep model that supports per-sample loss weights natively.
    # Only the non-trivial schemes are trained here; the `uniform` reference is the
    # scratch-chemprop 5×5 CV from notebook 2 (identical splits, identical
    # architecture), loaded in the summary cell below.
    _OUT = PRED_DIR / "6_reweighting_chemprop_cv.csv.gz"
    _CKPT = _OUT.with_suffix(".ckpt.gz")
    _SCHEMES = ["invdensity", "distance"]

    if _OUT.exists():
        reweight_chemprop_cv = pl.read_csv(_OUT)
        print(f"Found {_OUT.name} — skipping Chemprop training "
              f"({reweight_chemprop_cv.shape[0]:,} rows).")
    else:
        if _CKPT.exists():
            _records = pl.read_csv(_CKPT).to_dicts()
            _done = {(r["fold"], r["scheme"]) for r in _records}
            print(f"Resuming Chemprop reweighting from checkpoint "
                  f"({len(_records):,} rows, {len(_done)} fold×scheme done).")
        else:
            _records, _done = [], set()

        for _fold, _outer, _inner, _tr, _va, _te in tqdm(
            generate_cv_splits_random(DR_TRAIN, n_outer=5, n_inner=5, seed=42, p_val=0.1),
            total=25, desc="Chemprop reweighting CV", unit="fold",
        ):
            _y_tr = _tr["pEC50_dr"].to_numpy()
            _y_va = _va["pEC50_dr"].to_numpy()
            _y_te = _te["pEC50_dr"].to_numpy()

            for _scheme in _SCHEMES:
                if (_fold, _scheme) in _done:
                    continue
                _w = compute_sample_weights(_y_tr, _scheme)
                _m = ChempropModel(pred_type="regression", epochs=50)
                _m.train(_tr["smiles"].to_list(), _y_tr,
                         _va["smiles"].to_list(), _y_va,
                         target_col="pEC50_dr", sample_weight=_w)
                _pred = _m.predict(_te["smiles"].to_list())
                del _m

                for _ik, _yt, _yp in zip(_te["inchikey"].to_list(),
                                         _y_te.tolist(), _pred.tolist()):
                    _records.append({
                        "inchikey": _ik, "fold": _fold,
                        "model": "chemprop_scratch", "scheme": _scheme,
                        "method": f"chemprop_scratch_{_scheme}",
                        "y_true": _yt, "y_pred": _yp,
                    })
                # Checkpoint after every (fold, scheme) so the long run can resume.
                with gzip.open(_CKPT, "wb") as _f:
                    pl.DataFrame(_records).write_csv(_f)

        reweight_chemprop_cv = pl.DataFrame(_records)
        with gzip.open(_OUT, "wb") as _f:
            reweight_chemprop_cv.write_csv(_f)
        _CKPT.unlink(missing_ok=True)
        print(f"Wrote {reweight_chemprop_cv.shape[0]:,} rows → {_OUT.name}")
    return (reweight_chemprop_cv,)


@app.cell
def _(
    BIN_ORDER,
    PLOTS_DIR,
    PRED_DIR,
    bias_by_bin_table,
    calc_regression_metrics,
    mcs_plot,
    mean_absolute_error,
    mo,
    np,
    pl,
    plt,
    reweight_chemprop_cv,
    reweight_cv,
    rm_tukey_hsd,
):
    # ── Combine tree models (here) + Chemprop (here) + Chemprop uniform (nb 2) ────
    # The scratch-chemprop `uniform` baseline is taken from notebook 2's 5×5 CV,
    # which used identical folds (seed 42) and an identical architecture, so it is a
    # fair reference for the reweighted chemprop runs.
    _chemprop_uniform = (
        pl.read_csv(PRED_DIR / "2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .filter(pl.col("model") == "chemprop")
        .select(["inchikey", "fold", "y_true", "y_pred"])
        .with_columns(pl.lit("chemprop_scratch_uniform").alias("method"))
    )
    _cols = ["inchikey", "fold", "method", "y_true", "y_pred"]
    reweight_all = pl.concat([
        reweight_cv.select(_cols),
        reweight_chemprop_cv.select(_cols),
        _chemprop_uniform.select(_cols),
    ])

    # ── Summary table, MCS and per-bin bias ──────────────────────────────────────
    _rows = []
    for (_method,), _grp in reweight_all.group_by(["method"]):
        _yt = _grp["y_true"].to_numpy()
        _yp = _grp["y_pred"].to_numpy()
        _hit = _grp.filter(pl.col("y_true") >= 6.0)
        _inact = _grp.filter(pl.col("y_true") < 4.0)
        _rows.append({
            "method": _method,
            "MAE": round(mean_absolute_error(_yt, _yp), 4),
            "hitzone_MAE": round(mean_absolute_error(
                _hit["y_true"].to_numpy(), _hit["y_pred"].to_numpy()), 3),
            "hitzone_bias": round(float((_hit["y_pred"] - _hit["y_true"]).mean()), 3),
            "inactive_bias": round(float((_inact["y_pred"] - _inact["y_true"]).mean()), 3),
        })
    reweight_summary = pl.DataFrame(_rows).sort("MAE")

    # ── MCS heatmaps: one subplot per model, scheme-only labels ─────────────────
    _MODEL_TITLES = {
        "xgb_mordred": "XGBoost · Mordred",
        "rf_chemeleon": "RF · CheMeleon",
        "chemprop_scratch": "Chemprop · scratch",
    }
    _SCHEMES = ["uniform", "distance", "invdensity"]

    _scheme_df = reweight_all.with_columns(
        pl.col("method").str.extract(r"(uniform|invdensity|distance)$").alias("scheme"),
        pl.col("method").str.replace(r"_(uniform|invdensity|distance)$", "").alias("model"),
    )

    fig_mcs, axes_mcs = plt.subplots(1, 3, figsize=(18, 5.5), dpi=140)
    for _ax, (_model_key, _title) in zip(axes_mcs, _MODEL_TITLES.items()):
        _sub = _scheme_df.filter(pl.col("model") == _model_key)
        _met = calc_regression_metrics(
            _sub.select([
                pl.col("fold").alias("cv_cycle"),
                pl.col("scheme").alias("method"),
                pl.lit("random").alias("split"),
                "y_true", "y_pred",
            ]),
            "cv_cycle", "y_true", "y_pred", thresh=4.0,
        )
        _, _means, _diffs, _pc = rm_tukey_hsd(
            _met, "mae", "method", 0.05, True, {"mae": "minimize"})
        mcs_plot(_pc, effect_size=_diffs, means=_means["mae"],
                 show_diff=True, ax=_ax, cbar=True,
                 cell_text_size=14, axis_text_size=11,
                 reverse_cmap=True, vlim=0.03)
        _ax.set_title(_title, fontsize=13)
    fig_mcs.suptitle("MAE — Tukey HSD per model (reweighting schemes)",
                     fontsize=14, y=1.02)
    fig_mcs.tight_layout()
    fig_mcs.savefig(PLOTS_DIR / "analysis2_reweighting_mcs_mae.png",
                    dpi=300, bbox_inches="tight")

    # ── Per-bin signed-error heatmaps: one subplot per model ────────────────────
    _bias_all = bias_by_bin_table(_scheme_df, "method", "y_true", "y_pred")
    _bias_all = _bias_all.with_columns(
        pl.col("method").str.extract(r"(uniform|invdensity|distance)$").alias("scheme"),
        pl.col("method").str.replace(r"_(uniform|invdensity|distance)$", "").alias("model"),
    )

    fig_bias, axes_bias = plt.subplots(
        1, 3, figsize=(18, 3.8), dpi=150)
    for _ax, (_model_key, _title) in zip(axes_bias, _MODEL_TITLES.items()):
        _sub = _bias_all.filter(pl.col("model") == _model_key)
        _matrix = np.full((len(_SCHEMES), len(BIN_ORDER)), np.nan)
        for _row in _sub.iter_rows(named=True):
            if _row["scheme"] in _SCHEMES and _row["pec50_bin"] in BIN_ORDER:
                _mi = _SCHEMES.index(_row["scheme"])
                _bi = BIN_ORDER.index(_row["pec50_bin"])
                _matrix[_mi, _bi] = _row["mean_error"]
        _abs_max = float(np.nanmax(np.abs(_matrix)))
        _ax.grid(False)
        _im = _ax.imshow(_matrix, cmap="RdBu_r", vmin=-_abs_max, vmax=_abs_max,
                         aspect="auto", interpolation="nearest")
        for _mi in range(len(_SCHEMES)):
            for _bi in range(len(BIN_ORDER)):
                _val = _matrix[_mi, _bi]
                if not np.isnan(_val):
                    _col = "white" if abs(_val) > 0.6 * _abs_max else "black"
                    _ax.text(_bi, _mi, f"{_val:+.3f}", ha="center", va="center",
                             fontsize=9, color=_col)
        _ax.set_xticks(range(len(BIN_ORDER)))
        _ax.set_xticklabels(BIN_ORDER, fontsize=8, rotation=-20, ha="left")
        _ax.set_yticks(range(len(_SCHEMES)))
        if _ax is axes_bias[0]:
            _ax.set_yticklabels(_SCHEMES, fontsize=10)
        else:
            _ax.set_yticklabels([])
        _ax.set_title(_title, fontsize=11)
        fig_bias.colorbar(_im, ax=_ax, fraction=0.04, pad=0.03).set_label(
            "Mean error (pred − true)", fontsize=8)
    fig_bias.suptitle(
        "Reweighting — mean signed error by activity bin\n"
        "(red = overpredict, blue = underpredict)",
        fontsize=12, y=1.06)
    fig_bias.tight_layout()
    fig_bias.savefig(PLOTS_DIR / "analysis2_reweighting_bias_heatmap.png",
                     dpi=300, bbox_inches="tight")

    mo.vstack([
        mo.md("### Reweighting results — XGBoost·Mordred, RF·CheMeleon, Chemprop·scratch\n"
              "Inverse-density and distance weighting trade overall MAE for reduced "
              "bias at the extremes — watch the `>6` and `<4` columns versus `uniform`. "
              "Chemprop is the one deep model that supports loss weighting; its "
              "`uniform` row is notebook 2's baseline (same folds, same architecture). "
              "Chemprop is stochastic, so small differences may reflect seed noise."),
        mo.ui.table(reweight_summary.to_pandas(), selection=None, pagination=False),
        mo.as_html(fig_mcs),
        mo.as_html(fig_bias),
    ])
    return (reweight_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 3 — Oversampling the activity extremes

    Reweighting (Analysis 2) only works for models with a per-sample loss weight.
    **Oversampling** achieves the same goal at the *data* level — duplicating the
    rare extreme compounds in the training set — and therefore works for **every**
    model, including TabPFN and Macau which have no loss weight to set.

    The extremes are the **active** tail (pEC50 > 5.5, ≈ 9 % of the set) and the
    **inactive** tail (pEC50 < 3.5, ≈ 23 %); the dense middle (3.5–5.5) is left
    untouched. Each extreme compound in the *training* fold is replicated ×k:

    | k | extreme share of training set |
    |---|---|
    | 1 (baseline) | 32 % |
    | 2 | 49 % |
    | 3 | 59 % |
    | 5 | 71 % |

    Four models are tested — **XGBoost·Mordred, Macau·CheMeleon, TabPFN·CheMeleon,
    Chemprop·scratch**. To save compute, the **1× baseline is reused** from earlier
    config-matched 5×5 CV runs (XGBoost from Analysis 2, Macau/TabPFN from notebook
    4, Chemprop from notebook 2); only k = 2, 3, 5 are trained here. Oversampling is
    applied to the training split only, so the test folds are unchanged and every
    comparison stays on the same dose-response compounds.
    """)
    return


@app.cell
def _(np):
    def oversample_extremes(
        y: np.ndarray, factor: int, hi: float = 5.5, lo: float = 3.5,
    ) -> np.ndarray:
        """
        Row indices that replicate each extreme-activity compound `factor` times.

        Args:
            y: Training labels (pEC50).
            factor: Replication count for extreme rows (1 = no oversampling).
            hi: Active-tail threshold (compounds with y > hi are extreme).
            lo: Inactive-tail threshold (compounds with y < lo are extreme).

        Returns:
            Index array into the training arrays: all rows once, plus (factor − 1)
            extra copies of the extreme rows.
        """
        base = np.arange(len(y))
        if factor <= 1:
            return base
        extreme = np.where((y > hi) | (y < lo))[0]
        extra = np.repeat(extreme, factor - 1)
        return np.concatenate([base, extra])

    return (oversample_extremes,)


@app.cell
def _(
    BoostedTreesModel,
    ChempropModel,
    DR_TRAIN,
    MacauModel,
    PRED_DIR,
    gather_X,
    gc,
    generate_cv_splits_random,
    gzip,
    oversample_extremes,
    pl,
    tabpfn_predict,
    tqdm,
):
    # ── 5×5 CV: oversample the extremes ×{2,3,5} for four model families ──────────
    # The 1× baseline is reused in the summary cell, so only k>1 is trained here.
    # The slow models are Chemprop (CLI) and TabPFN (CPU subprocess); the loop is
    # checkpointed per (fold, model, factor) so it survives interruptions.
    _OUT = PRED_DIR / "6_oversampling_cv.csv.gz"
    _CKPT = _OUT.with_suffix(".ckpt.gz")
    _FACTORS = [2, 3, 5]

    if _OUT.exists():
        oversampling_cv = pl.read_csv(_OUT)
        print(f"Found {_OUT.name} — skipping ({oversampling_cv.shape[0]:,} rows).")
    else:
        if _CKPT.exists():
            _records = pl.read_csv(_CKPT).to_dicts()
            _done = {(r["fold"], r["method"]) for r in _records}
            print(f"Resuming oversampling CV from checkpoint "
                  f"({len(_records):,} rows, {len(_done)} fold×method done).")
        else:
            _records, _done = [], set()

        for _fold, _outer, _inner, _tr, _va, _te in tqdm(
            generate_cv_splits_random(DR_TRAIN, n_outer=5, n_inner=5, seed=42, p_val=0.1),
            total=25, desc="oversampling CV", unit="fold",
        ):
            _y_tr = _tr["pEC50_dr"].to_numpy()
            _y_va = _va["pEC50_dr"].to_numpy()
            _y_te = _te["pEC50_dr"].to_numpy()
            _te_iks = _te["inchikey"].to_list()

            # Features gathered once per fold (rows duplicated per factor via _idx).
            _Xtr_mor = gather_X(_tr["inchikey"].to_list(), "mordred")
            _Xva_mor = gather_X(_va["inchikey"].to_list(), "mordred")
            _Xte_mor = gather_X(_te_iks, "mordred")
            _Xtr_che = gather_X(_tr["inchikey"].to_list(), "chemeleon")
            _Xte_che = gather_X(_te_iks, "chemeleon")
            _smi_tr = _tr["smiles"].to_list()
            _smi_va = _va["smiles"].to_list()
            _smi_te = _te["smiles"].to_list()

            def _save(model_key: str, factor: int, preds) -> None:
                for _ik, _yt, _yp in zip(_te_iks, _y_te.tolist(), list(preds)):
                    _records.append({
                        "inchikey": _ik, "fold": _fold, "model": model_key,
                        "factor": factor, "method": f"{model_key}_os{factor}",
                        "y_true": _yt, "y_pred": float(_yp),
                    })
                with gzip.open(_CKPT, "wb") as _f:
                    pl.DataFrame(_records).write_csv(_f)

            for _factor in _FACTORS:
                _idx = oversample_extremes(_y_tr, _factor)
                _yo = _y_tr[_idx]

                if (_fold, f"xgb_mordred_os{_factor}") not in _done:
                    _m = BoostedTreesModel(pred_type="regression")
                    _m.train(_Xtr_mor[_idx], _yo, _Xva_mor, _y_va)
                    _save("xgb_mordred", _factor, _m.predict(_Xte_mor))
                    del _m; gc.collect()

                if (_fold, f"macau_chemeleon_os{_factor}") not in _done:
                    _m = MacauModel(seed=42)
                    _m.train(_Xtr_che[_idx], _yo)
                    _save("macau_chemeleon", _factor, _m.predict(_Xte_che))
                    del _m; gc.collect()

                if (_fold, f"tabpfn_chemeleon_os{_factor}") not in _done:
                    _save("tabpfn_chemeleon", _factor,
                          tabpfn_predict(_Xtr_che[_idx], _yo, _Xte_che))

                if (_fold, f"chemprop_scratch_os{_factor}") not in _done:
                    _m = ChempropModel(pred_type="regression", epochs=50)
                    _m.train([_smi_tr[i] for i in _idx], _yo, _smi_va, _y_va,
                             target_col="pEC50_dr")
                    _save("chemprop_scratch", _factor, _m.predict(_smi_te))
                    del _m; gc.collect()

        oversampling_cv = pl.DataFrame(_records)
        with gzip.open(_OUT, "wb") as _f:
            oversampling_cv.write_csv(_f)
        _CKPT.unlink(missing_ok=True)
        print(f"Wrote {oversampling_cv.shape[0]:,} rows → {_OUT.name}")
    return (oversampling_cv,)


@app.cell
def _(PLOTS_DIR, PRED_DIR, mean_absolute_error, mo, oversampling_cv, pl, plt):
    # ── Combine reused 1× baselines with the oversampled runs ────────────────────
    # Note: notebook 2's file labels models in a `model` column; the others use
    # `method` — so the filter column is passed explicitly.
    def _baseline(path: str, col: str, val: str, model_key: str) -> pl.DataFrame:
        return (
            pl.read_csv(PRED_DIR / path)
            .filter(pl.col(col) == val)
            .select(["fold", "y_true", "y_pred"])
            .with_columns(pl.lit(model_key).alias("model"), pl.lit(1).cast(pl.Int64).alias("factor"))
        )

    _base = pl.concat([
        _baseline("6_reweighting_cv.csv.gz", "method", "xgb_mordred_uniform", "xgb_mordred"),
        _baseline("4_fp_model_comparison_1.csv.gz", "method", "macau_chemeleon", "macau_chemeleon"),
        _baseline("4_fp_model_comparison_2.csv.gz", "method", "tabpfn_chemeleon", "tabpfn_chemeleon"),
        _baseline("2_ml_baseline_5x5cv_random_predictions.csv.gz", "model", "chemprop", "chemprop_scratch"),
    ])
    _cols = ["fold", "model", "factor", "y_true", "y_pred"]
    os_all = pl.concat([
        _base.select(_cols),
        oversampling_cv.select(_cols),
    ]).with_columns(pl.format("{}_os{}", pl.col("model"), pl.col("factor")).alias("method"))

    # ── Pooled metrics per (model, factor) ───────────────────────────────────────
    _rows = []
    for (_model, _factor), _grp in os_all.group_by(["model", "factor"]):
        _yt = _grp["y_true"].to_numpy()
        _yp = _grp["y_pred"].to_numpy()
        _hit = _grp.filter(pl.col("y_true") > 5.5)
        _inact = _grp.filter(pl.col("y_true") < 3.5)
        _rows.append({
            "model": _model, "factor": _factor,
            "method": f"{_model}_os{_factor}",
            "MAE": round(mean_absolute_error(_yt, _yp), 4),
            "hitzone_bias": round(float((_hit["y_pred"] - _hit["y_true"]).mean()), 3),
            "inactive_bias": round(float((_inact["y_pred"] - _inact["y_true"]).mean()), 3),
        })
    oversampling_summary = pl.DataFrame(_rows).sort(["model", "factor"])

    # Per-fold MAE (mean ± std across folds) for the MAE-vs-factor curve.
    _per_fold = []
    for (_model, _factor, _fold), _grp in os_all.group_by(["model", "factor", "fold"]):
        _per_fold.append({"model": _model, "factor": _factor,
                          "mae": mean_absolute_error(_grp["y_true"].to_numpy(),
                                                     _grp["y_pred"].to_numpy())})
    _pf = (
        pl.DataFrame(_per_fold)
        .group_by(["model", "factor"])
        .agg(pl.col("mae").mean().alias("mae_mean"), pl.col("mae").std().alias("mae_std"))
    )

    _MODELS = ["xgb_mordred", "macau_chemeleon", "tabpfn_chemeleon", "chemprop_scratch"]
    _COLORS = {"xgb_mordred": "#4e79a7", "macau_chemeleon": "#e15759",
               "tabpfn_chemeleon": "#59a14f", "chemprop_scratch": "#b07aa1"}

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, (_ax1, _ax2) = plt.subplots(1, 2, figsize=(13, 5), dpi=140)
        for _model in _MODELS:
            _s = _pf.filter(pl.col("model") == _model).sort("factor")
            _ax1.errorbar(_s["factor"].to_numpy(), _s["mae_mean"].to_numpy(),
                          yerr=_s["mae_std"].to_numpy(), marker="o", capsize=3,
                          color=_COLORS[_model], label=_model)
            _b = oversampling_summary.filter(pl.col("model") == _model).sort("factor")
            _ax2.plot(_b["factor"].to_numpy(), _b["hitzone_bias"].to_numpy(),
                      marker="o", color=_COLORS[_model], label=f"{_model} (hit)")
            _ax2.plot(_b["factor"].to_numpy(), _b["inactive_bias"].to_numpy(),
                      marker="s", linestyle="--", color=_COLORS[_model], alpha=0.6)
        _ax1.set_xlabel("oversampling factor k", fontsize=11)
        _ax1.set_ylabel("CV MAE (mean ± std over folds)", fontsize=11)
        _ax1.set_title("Overall MAE vs oversampling", fontsize=12)
        _ax1.set_xticks([1, 2, 3, 5])
        _ax1.legend(fontsize=9)
        _ax2.axhline(0, color="black", linewidth=0.9, linestyle="--", zorder=0)
        _ax2.set_xlabel("oversampling factor k", fontsize=11)
        _ax2.set_ylabel("signed bias (pred − true)", fontsize=11)
        _ax2.set_title("Extreme-zone bias vs oversampling\n(○ hit > 5.5, □ inactive < 3.5)",
                       fontsize=12)
        _ax2.set_xticks([1, 2, 3, 5])
        _ax2.legend(fontsize=8, ncol=2)
        _fig.tight_layout()
        _fig.savefig(PLOTS_DIR / "analysis3_oversampling_curves.png",
                     dpi=300, bbox_inches="tight")

    mo.vstack([
        mo.md("### Oversampling results\n"
              "Left: does oversampling lower overall MAE? Right: does it pull the "
              "hit-zone bias (○) up toward 0 and the inactive bias (□) down toward 0? "
              "k = 1 is the reused baseline."),
        mo.ui.table(oversampling_summary.to_pandas(), selection=None, pagination=False),
        mo.as_html(_fig),
    ])
    return (oversampling_summary,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    While oversampling reduces bias in the extremes of the distribution in a more impactful way than previous methods, in general all models perform much worse. tabPFN see some of the biggest decreases in performance and also see an increase in the bias
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 4 — Extra assay data (96-compound semi-pure set)

    OpenADMET released a small but high-quality dataset: 96 compounds re-measured
    from **semi-pure** microscale synthesis with purity-corrected pEC50 values
    (≈ 94 usable, spanning pEC50 4.0–7.0 — exactly the moderate-to-hit range where
    the models struggle). Only **5 of these overlap the dose-response training set**,
    so they are almost entirely new measurements.

    Two questions:

    1. **Label-noise audit** — for the handful of overlapping compounds, how far is
       the original training pEC50 from the corrected semi-pure value? This bounds
       how much of the model error is irreducible label noise.
    2. **Augmentation** — does adding these compounds to the training fold improve
       5×5 CV on the dose-response set? The semi-pure compounds are added to the
       *training* split only (and never when they collide with a test-fold compound),
       so evaluation stays on the dose-response compounds and folds stay comparable.
    """)
    return


@app.cell
def _(DR_TRAIN, PLOTS_DIR, mo, pl, plt, semipure):
    # ── Label-noise audit on the overlap with the DR training set ────────────────
    _overlap = (
        semipure.join(
            DR_TRAIN.select(["inchikey", "pEC50_dr", "molecule_names"]),
            on="inchikey", how="inner")
        .with_columns(
            (pl.col("pEC50_corrected") - pl.col("pEC50_dr")).alias("delta"))
        .sort(pl.col("delta").abs(), descending=True)
    )

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _ax = plt.subplots(figsize=(5.5, 5.5), dpi=150)
        _x = _overlap["pEC50_dr"].to_numpy()
        _y = _overlap["pEC50_corrected"].to_numpy()
        _ax.scatter(_x, _y, s=70, color="#4e79a7", edgecolors="white", zorder=3)
        _lims = [min(_x.min(), _y.min()) - 0.3, max(_x.max(), _y.max()) + 0.3]
        _ax.plot(_lims, _lims, "k--", linewidth=1, zorder=0, label="identity")
        for _r in _overlap.iter_rows(named=True):
            _ax.annotate(f"{_r['delta']:+.2f}", (_r["pEC50_dr"], _r["pEC50_corrected"]),
                         fontsize=8, xytext=(4, 4), textcoords="offset points")
        _ax.set_xlim(_lims); _ax.set_ylim(_lims)
        _ax.set_xlabel("Original training pEC50 (pEC50_dr)", fontsize=11)
        _ax.set_ylabel("Corrected semi-pure pEC50", fontsize=11)
        _ax.set_title(f"Label-noise audit — {_overlap.shape[0]} overlapping compounds", fontsize=12)
        _ax.legend(fontsize=10)
        _fig.tight_layout()
        _fig.savefig(PLOTS_DIR / "analysis4_label_noise.png", dpi=300, bbox_inches="tight")

    _mad = float(_overlap["delta"].abs().mean()) if _overlap.shape[0] else float("nan")
    mo.vstack([
        mo.md(f"### Label-noise audit\n"
              f"Mean absolute discrepancy between original and corrected pEC50: "
              f"**{_mad:.2f} log units** across {_overlap.shape[0]} overlapping compounds. "
              f"For reference, the best ensemble CV MAE is ≈ 0.47 — so part of the model "
              f"error is simply measurement noise in the labels."),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(
    BoostedTreesModel,
    DR_TRAIN,
    PRED_DIR,
    RandomForestModel,
    gather_X,
    gc,
    generate_cv_splits_random,
    gzip,
    np,
    pl,
    semipure,
    tqdm,
):
    # ── 5×5 CV: baseline vs +semi-pure augmentation ──────────────────────────────
    _OUT = PRED_DIR / "6_extradata_cv.csv.gz"
    _MODELS = [("xgb_mordred", "mordred"), ("rf_chemeleon", "chemeleon")]
    _sp_iks = semipure["inchikey"].to_list()
    _sp_y = semipure["pEC50_corrected"].to_numpy()

    if _OUT.exists():
        extradata_cv = pl.read_csv(_OUT)
        print(f"Found {_OUT.name} — skipping training ({extradata_cv.shape[0]:,} rows).")
    else:
        _records: list[dict] = []
        for _fold, _outer, _inner, _tr, _va, _te in tqdm(
            generate_cv_splits_random(DR_TRAIN, n_outer=5, n_inner=5, seed=42, p_val=0.1),
            total=25, desc="extra-data CV", unit="fold",
        ):
            _y_tr = _tr["pEC50_dr"].to_numpy()
            _y_va = _va["pEC50_dr"].to_numpy()
            _y_te = _te["pEC50_dr"].to_numpy()
            _te_iks = set(_te["inchikey"].to_list())

            # Semi-pure rows usable this fold: exclude any colliding with the test fold.
            _keep = [i for i, ik in enumerate(_sp_iks) if ik not in _te_iks]
            _sp_keep_iks = [_sp_iks[i] for i in _keep]
            _sp_keep_y = _sp_y[_keep]

            for _model_key, _kind in _MODELS:
                _X_tr = gather_X(_tr["inchikey"].to_list(), _kind)
                _X_va = gather_X(_va["inchikey"].to_list(), _kind)
                _X_te = gather_X(_te["inchikey"].to_list(), _kind)
                _X_sp = gather_X(_sp_keep_iks, _kind)

                for _scenario in ["base", "+semipure"]:
                    if _scenario == "base":
                        _Xt, _yt = _X_tr, _y_tr
                    else:
                        _Xt = np.vstack([_X_tr, _X_sp])
                        _yt = np.concatenate([_y_tr, _sp_keep_y])

                    if _model_key == "xgb_mordred":
                        _m = BoostedTreesModel(pred_type="regression")
                        _m.train(_Xt, _yt, _X_va, _y_va)
                    else:
                        _m = RandomForestModel(pred_type="regression")
                        _m.train(_Xt, _yt)
                    _pred = _m.predict(_X_te)
                    del _m
                    gc.collect()

                    for _ik, _ytrue, _yp in zip(_te["inchikey"].to_list(),
                                                _y_te.tolist(), _pred.tolist()):
                        _records.append({
                            "inchikey": _ik, "fold": _fold,
                            "model": _model_key, "scenario": _scenario,
                            "method": f"{_model_key}_{_scenario}",
                            "y_true": _ytrue, "y_pred": _yp,
                        })

        extradata_cv = pl.DataFrame(_records)
        with gzip.open(_OUT, "wb") as _f:
            extradata_cv.write_csv(_f)
        print(f"Wrote {extradata_cv.shape[0]:,} rows → {_OUT.name}")
    return (extradata_cv,)


@app.cell
def _(
    PLOTS_DIR,
    calc_regression_metrics,
    extradata_cv,
    make_mcs_plot_grid,
    mean_absolute_error,
    mo,
    pl,
):
    # ── Summary + MCS for the augmentation experiment ────────────────────────────
    _rows = []
    for (_method,), _grp in extradata_cv.group_by(["method"]):
        _yt = _grp["y_true"].to_numpy()
        _yp = _grp["y_pred"].to_numpy()
        _hit = _grp.filter(pl.col("y_true") >= 6.0)
        _rows.append({
            "method": _method,
            "MAE": round(mean_absolute_error(_yt, _yp), 4),
            "hitzone_MAE": round(mean_absolute_error(
                _hit["y_true"].to_numpy(), _hit["y_pred"].to_numpy()), 3),
            "hitzone_bias": round(float((_hit["y_pred"] - _hit["y_true"]).mean()), 3),
        })
    extradata_summary = pl.DataFrame(_rows).sort("MAE")

    _metrics = calc_regression_metrics(
        extradata_cv.select([
            pl.col("fold").alias("cv_cycle"), "method",
            pl.lit("random").alias("split"), "y_true", "y_pred"]),
        "cv_cycle", "y_true", "y_pred", thresh=4.0)
    _fig = make_mcs_plot_grid(
        _metrics, stats=["mae"], group_col="method",
        figsize=(10, 9), effect_dict={"mae": 0.02},
        cell_text_size=11, axis_text_size=9, title_text_size=13,
        save_path=PLOTS_DIR / "analysis4_extradata_mcs_mae.png")

    mo.vstack([
        mo.md("### Augmentation results\n"
              "`+semipure` adds ≈ 90 high-quality compounds to each training fold. "
              "With only a ~2 % increase in training size, any gain is expected to be small."),
        mo.ui.table(extradata_summary.to_pandas(), selection=None, pagination=False),
        mo.as_html(_fig),
    ])
    return (extradata_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 5 — Filtering (cleaning) the training set

    The previous analyses *add* or *reweight* data. This one *removes* likely
    unreliable training points and asks whether a cleaner set generalises better.
    Under the same 5×5 CV the **training fold is filtered** (the test fold is left
    intact) and compared against (a) the full-data baseline and (b) a **size-matched
    random-removal control** — so we can tell genuine cleaning from the mere effect
    of training on less data.

    Three filters are compared, each removing ≈ 10 % of the training fold:

    | Filter | Rule | Rationale |
    |---|---|---|
    | `counter` | drop where counter-screen pEC50 ≥ PXR pEC50 | the hit is not PXR-specific |
    | `cliff` | ECFP4 Tanimoto ≥ 0.4 & \|ΔpEC50\| ≥ 1.0 → drop the member nearer the mean | remove contradictory SAR, keep the informative extreme |
    | `knn` | drop the 10 % whose pEC50 deviates most from their 5 nearest ECFP4 neighbours | local label-noise removal |

    Every filter is paired with a `rand_*` control that drops the **same number** of
    random rows. Two models are tested — XGBoost·Mordred, Macau·CheMeleon — with
    the 1× (full-data) baseline reused from earlier config-matched runs. Filtering
    is computed inside each fold from the training compounds only, so there is no
    test-set leakage.
    """)
    return


@app.cell
def _(Chem, np, pl):
    # ── Training table with counter data + ECFP4 fingerprints + filter functions ──
    from rdkit.Chem import AllChem as _AllChem
    from rdkit import DataStructs as _DataStructs

    # Same rows/order as DR_TRAIN (so folds match the reused baselines) plus counter.
    filter_train = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "molecule_names", "pEC50_dr", "pEC50_counter"])
    )

    def _ecfp4(smi: str):
        mol = Chem.MolFromSmiles(smi)
        return _AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048) if mol else None

    ecfp4_by_ik = {
        ik: _ecfp4(s)
        for ik, s in zip(filter_train["inchikey"].to_list(), filter_train["smiles"].to_list())
    }
    print(f"ECFP4 cached for {len(ecfp4_by_ik)} compounds")

    def counter_remove(tr: pl.DataFrame) -> set:
        """Positions of compounds whose counter pEC50 ≥ their PXR pEC50 (non-selective)."""
        _rm = tr.with_columns(
            (pl.col("pEC50_counter").is_not_null()
             & (pl.col("pEC50_counter") >= pl.col("pEC50_dr"))).alias("_rm"))
        return set(np.where(_rm["_rm"].to_numpy())[0].tolist())

    def cliff_remove(fps: list, y: np.ndarray, tan: float = 0.4, dthr: float = 1.0) -> set:
        """Activity-cliff members to drop (the one nearer the global mean of each pair)."""
        mean = float(y.mean())
        remove: set = set()
        for i in range(len(fps)):
            if fps[i] is None:
                continue
            sims = _DataStructs.BulkTanimotoSimilarity(fps[i], fps[i + 1:])
            for off, s in enumerate(sims):
                j = i + 1 + off
                if s >= tan and abs(y[i] - y[j]) >= dthr:
                    remove.add(i if abs(y[i] - mean) < abs(y[j] - mean) else j)
        return remove

    def knn_remove(fps: list, y: np.ndarray, k: int = 5, q: float = 0.10) -> set:
        """Drop the top-q fraction by deviation from their k nearest ECFP4 neighbours."""
        n = len(fps)
        dev = np.zeros(n)
        for i in range(n):
            if fps[i] is None:
                dev[i] = -1.0
                continue
            sims = np.array(_DataStructs.BulkTanimotoSimilarity(fps[i], fps))
            sims[i] = -1.0
            nn = np.argpartition(sims, -k)[-k:]
            dev[i] = abs(y[i] - y[nn].mean())
        n_remove = int(round(q * n))
        return set(np.argsort(dev)[-n_remove:].tolist())

    return cliff_remove, counter_remove, ecfp4_by_ik, filter_train, knn_remove


@app.cell
def _(
    BoostedTreesModel,
    MacauModel,
    PRED_DIR,
    cliff_remove,
    counter_remove,
    ecfp4_by_ik,
    filter_train,
    gather_X,
    gc,
    generate_cv_splits_random,
    gzip,
    knn_remove,
    np,
    pl,
    tqdm,
):
    # ── 5×5 CV: filter the training fold, two models, filter + matched random ─────
    # 1× (full-data) baseline is reused in the summary cell; only filtered/random
    # variants are trained here. Checkpointed per (fold, method).
    _OUT = PRED_DIR / "6_filtering_cv.csv.gz"
    _CKPT = _OUT.with_suffix(".ckpt.gz")
    _FILTERS = ["counter", "cliff", "knn"]
    _MODELS = [
        ("xgb_mordred", "mordred"),
        ("macau_chemeleon", "chemeleon"),
    ]

    if _OUT.exists():
        filtering_cv = pl.read_csv(_OUT)
        print(f"Found {_OUT.name} — skipping ({filtering_cv.shape[0]:,} rows).")
    else:
        if _CKPT.exists():
            _records = pl.read_csv(_CKPT).to_dicts()
            _done = {(r["fold"], r["method"]) for r in _records}
            print(f"Resuming filtering CV ({len(_records):,} rows, {len(_done)} done).")
        else:
            _records, _done = [], set()

        for _fold, _outer, _inner, _tr, _va, _te in tqdm(
            generate_cv_splits_random(filter_train, n_outer=5, n_inner=5, seed=42, p_val=0.1),
            total=25, desc="filtering CV", unit="fold",
        ):
            _y_tr = _tr["pEC50_dr"].to_numpy()
            _y_va = _va["pEC50_dr"].to_numpy()
            _y_te = _te["pEC50_dr"].to_numpy()
            _tr_iks = _tr["inchikey"].to_list()
            _te_iks = _te["inchikey"].to_list()
            _n_tr = len(_tr_iks)
            _fps = [ecfp4_by_ik[ik] for ik in _tr_iks]

            # Remove-sets for the three filters and their size-matched random twins.
            _removes = {
                "counter": counter_remove(_tr),
                "cliff": cliff_remove(_fps, _y_tr),
                "knn": knn_remove(_fps, _y_tr),
            }
            _rng = np.random.default_rng(1000 + _fold)
            _variants: dict[str, set] = {}
            for _f in _FILTERS:
                _variants[_f] = _removes[_f]
                _variants[f"rand_{_f}"] = set(
                    _rng.choice(_n_tr, size=len(_removes[_f]), replace=False).tolist())

            # Features gathered once per fold; rows subset via _kept.
            _Xtr_mor = gather_X(_tr_iks, "mordred")
            _Xva_mor = gather_X(_va["inchikey"].to_list(), "mordred")
            _Xte_mor = gather_X(_te_iks, "mordred")
            _Xtr_che = gather_X(_tr_iks, "chemeleon")
            _Xte_che = gather_X(_te_iks, "chemeleon")

            def _save(method: str, preds) -> None:
                for _ik, _yt, _yp in zip(_te_iks, _y_te.tolist(), list(preds)):
                    _records.append({
                        "inchikey": _ik, "fold": _fold, "method": method,
                        "y_true": _yt, "y_pred": float(_yp),
                    })
                with gzip.open(_CKPT, "wb") as _f:
                    pl.DataFrame(_records).write_csv(_f)

            for _variant, _rm in _variants.items():
                _kept = np.array([p for p in range(_n_tr) if p not in _rm])
                _yk = _y_tr[_kept]
                for _model_key, _kind in _MODELS:
                    _method = f"{_model_key}_{_variant}"
                    if (_fold, _method) in _done:
                        continue
                    if _model_key == "xgb_mordred":
                        _m = BoostedTreesModel(pred_type="regression")
                        _m.train(_Xtr_mor[_kept], _yk, _Xva_mor, _y_va)
                        _save(_method, _m.predict(_Xte_mor)); del _m
                    elif _model_key == "macau_chemeleon":
                        _m = MacauModel(seed=42)
                        _m.train(_Xtr_che[_kept], _yk)
                        _save(_method, _m.predict(_Xte_che)); del _m
                    gc.collect()

        filtering_cv = pl.DataFrame(_records)
        with gzip.open(_OUT, "wb") as _f:
            filtering_cv.write_csv(_f)
        _CKPT.unlink(missing_ok=True)
        print(f"Wrote {filtering_cv.shape[0]:,} rows → {_OUT.name}")
    return (filtering_cv,)


@app.cell
def _(PLOTS_DIR, PRED_DIR, filtering_cv, mean_absolute_error, mo, np, pl, plt):
    # ── Combine reused full-data baselines with the filtered / random variants ────
    # notebook 2 labels models in a `model` column; the others use `method`.
    def _baseline(path: str, col: str, val: str, model_key: str) -> pl.DataFrame:
        return (
            pl.read_csv(PRED_DIR / path)
            .filter(pl.col(col) == val)
            .select(["fold", "y_true", "y_pred"])
            .with_columns(pl.format("{}_baseline", pl.lit(model_key)).alias("method"))
        )

    _base = pl.concat([
        _baseline("6_reweighting_cv.csv.gz", "method", "xgb_mordred_uniform", "xgb_mordred"),
        _baseline("4_fp_model_comparison_1.csv.gz", "method", "macau_chemeleon", "macau_chemeleon"),
    ])
    _cols = ["fold", "method", "y_true", "y_pred"]
    filt_all = pl.concat([_base.select(_cols), filtering_cv.select(_cols)])

    # ── Summary: MAE per method ──────────────────────────────────────────────────
    _rows = []
    for (_method,), _grp in filt_all.group_by(["method"]):
        _rows.append({"method": _method,
                      "MAE": round(mean_absolute_error(_grp["y_true"].to_numpy(),
                                                       _grp["y_pred"].to_numpy()), 4)})
    filtering_summary = pl.DataFrame(_rows).sort("method")

    _MODELS = ["xgb_mordred", "macau_chemeleon"]
    _FILTERS = ["counter", "cliff", "knn"]

    def _mae(method: str) -> float:
        _r = filtering_summary.filter(pl.col("method") == method)
        return float(_r["MAE"][0]) if _r.shape[0] else float("nan")

    # ── Per-model ΔMAE vs baseline: filter bar next to its matched random twin ────
    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(1, 2, figsize=(12, 5), dpi=140)
        for _ax, _model in zip(_axes.flatten(), _MODELS):
            _base_mae = _mae(f"{_model}_baseline")
            _x = np.arange(len(_FILTERS))
            _filt_d = [_mae(f"{_model}_{f}") - _base_mae for f in _FILTERS]
            _rand_d = [_mae(f"{_model}_rand_{f}") - _base_mae for f in _FILTERS]
            _ax.bar(_x - 0.2, _filt_d, 0.4, label="filter", color="#4e79a7")
            _ax.bar(_x + 0.2, _rand_d, 0.4, label="random (matched N)", color="#bab0ac")
            _ax.axhline(0, color="black", linewidth=0.9)
            _ax.set_xticks(_x)
            _ax.set_xticklabels(_FILTERS)
            _ax.set_ylabel("ΔMAE vs full-data baseline")
            _ax.set_title(f"{_model}  (baseline MAE={_base_mae:.3f})", fontsize=10)
            _ax.legend(fontsize=8)
        _fig.suptitle("Training-set filtering — ΔMAE vs baseline "
                      "(negative = improvement; beat the grey random control to count)",
                      fontsize=12, y=1.01)
        _fig.tight_layout()
        _fig.savefig(PLOTS_DIR / "analysis5_filtering_delta_mae.png",
                     dpi=300, bbox_inches="tight")

    mo.vstack([
        mo.md("### Filtering results\n"
              "A filter is worthwhile only if its blue bar is **below 0** (beats the "
              "full-data baseline) **and below its grey random twin** (the gain is from "
              "cleaning, not from a smaller training set)."),
        mo.ui.table(filtering_summary.to_pandas(), selection=None, pagination=False),
        mo.as_html(_fig),
    ])
    return (filtering_summary,)


@app.cell
def _(mo):
    mo.md(r"""
    # Strategy selection and application to the unblinded test set

    The five analyses are compared on a common footing: the CV MAE **improvement of
    each strategy over its own baseline**. Comparing absolute MAE across analyses
    would be unfair, because the calibration strategy operates on the strong ensemble
    while reweighting, oversampling, augmentation and filtering operate on weaker
    single component models.

    **What can actually be applied to the submission?** The submitted model is the
    ensemble. Of the three ideas, only **calibration** operates directly on it —
    reweighting and augmentation modify individual components that are each weaker
    than the ensemble, so they are CV experiments that would only pay off by
    *rebuilding* the ensemble from reweighted/augmented components (future work).

    Therefore, honouring the rule that the unblinded labels must not drive model
    selection, the strategy applied to the unblinded set is the **best calibration
    variant chosen purely by 5×5 CV** (which may be `raw`, i.e. no change). Its effect
    on the unblinded labels is then reported once, for confirmation only.
    """)
    return


@app.cell
def _(
    calib_summary,
    extradata_summary,
    filtering_summary,
    mo,
    oversampling_summary,
    pl,
    reweight_summary,
):
    # ── Assemble the cross-analysis improvement table ────────────────────────────
    def _best_delta(summary: pl.DataFrame, baseline_method: str,
                    candidates: list[str]) -> tuple[str, float, float]:
        base_mae = summary.filter(pl.col("method") == baseline_method)["MAE"][0]
        best_method, best_mae = baseline_method, base_mae
        for cand in candidates:
            mae = summary.filter(pl.col("method") == cand)["MAE"][0]
            if mae < best_mae:
                best_method, best_mae = cand, mae
        return best_method, best_mae, best_mae - base_mae

    # Calibration — evaluate on the ensemble (the submitted model).
    _cal = calib_summary.filter(pl.col("base_model") == "ensemble")
    _cal_base = _cal.filter(pl.col("calibration") == "raw")["MAE"][0]
    _cal_best_row = _cal.sort("MAE").row(0, named=True)
    _cal_best, _cal_mae, _cal_delta = (
        _cal_best_row["calibration"], _cal_best_row["MAE"], _cal_best_row["MAE"] - _cal_base)

    # Reweighting — best scheme per model relative to that model's uniform baseline.
    _rw_xgb = _best_delta(reweight_summary, "xgb_mordred_uniform",
                          ["xgb_mordred_invdensity", "xgb_mordred_distance"])
    _rw_rf = _best_delta(reweight_summary, "rf_chemeleon_uniform",
                         ["rf_chemeleon_invdensity", "rf_chemeleon_distance"])
    _rw_cp = _best_delta(reweight_summary, "chemprop_scratch_uniform",
                         ["chemprop_scratch_invdensity", "chemprop_scratch_distance"])

    # Oversampling — best factor (os2/os3/os5) vs the reused 1× baseline per model.
    def _os_delta(model_key: str) -> tuple[str, float, float]:
        return _best_delta(
            oversampling_summary, f"{model_key}_os1",
            [f"{model_key}_os2", f"{model_key}_os3", f"{model_key}_os5"])
    _os_xgb = _os_delta("xgb_mordred")
    _os_mac = _os_delta("macau_chemeleon")
    _os_tf = _os_delta("tabpfn_chemeleon")
    _os_cp = _os_delta("chemprop_scratch")

    # Augmentation — +semipure vs base per model.
    _ed_xgb = _best_delta(extradata_summary, "xgb_mordred_base", ["xgb_mordred_+semipure"])
    _ed_rf = _best_delta(extradata_summary, "rf_chemeleon_base", ["rf_chemeleon_+semipure"])

    # Filtering — best filter (counter/cliff/knn) vs the full-data baseline per model.
    def _filt_delta(model_key: str) -> tuple[str, float, float]:
        return _best_delta(
            filtering_summary, f"{model_key}_baseline",
            [f"{model_key}_counter", f"{model_key}_cliff", f"{model_key}_knn"])
    _fl_xgb = _filt_delta("xgb_mordred")
    _fl_mac = _filt_delta("macau_chemeleon")

    def _row(analysis: str, model: str, res: tuple) -> dict:
        return {"analysis": analysis, "model": model, "best_variant": res[0],
                "cv_mae": round(res[1], 4), "delta_mae": round(res[2], 4)}

    selection_table = pl.DataFrame([
        {"analysis": "1 calibration", "model": "ensemble", "best_variant": _cal_best,
         "cv_mae": round(_cal_mae, 4), "delta_mae": round(_cal_delta, 4)},
        _row("2 reweighting", "xgb_mordred", _rw_xgb),
        _row("2 reweighting", "rf_chemeleon", _rw_rf),
        _row("2 reweighting", "chemprop_scratch", _rw_cp),
        _row("3 oversampling", "xgb_mordred", _os_xgb),
        _row("3 oversampling", "macau_chemeleon", _os_mac),
        _row("3 oversampling", "tabpfn_chemeleon", _os_tf),
        _row("3 oversampling", "chemprop_scratch", _os_cp),
        _row("4 extra data", "xgb_mordred", _ed_xgb),
        _row("4 extra data", "rf_chemeleon", _ed_rf),
        _row("5 filtering", "xgb_mordred", _fl_xgb),
        _row("5 filtering", "macau_chemeleon", _fl_mac),
    ]).sort("delta_mae")

    # The biggest CV improvement of any strategy over its own baseline (for insight).
    winner = selection_table.row(0, named=True)
    # The strategy actually applied to the ensemble submission: best calibration by CV.
    best_calibration = _cal_best

    mo.vstack([
        mo.md("### Cross-analysis comparison — CV MAE improvement over each baseline\n"
              "Lower `delta_mae` (more negative) is better."),
        mo.ui.table(selection_table.to_pandas(), selection=None, pagination=False),
        mo.md(f"**Largest CV improvement (any strategy):** `{winner['analysis']}` → "
              f"`{winner['best_variant']}` (ΔMAE = {winner['delta_mae']:+.4f}).\n\n"
              f"**Applied to the ensemble submission (best calibration by CV):** "
              f"`{best_calibration}` (ΔMAE = {_cal_delta:+.4f})."),
    ])
    return best_calibration, winner


@app.cell
def _(IsotonicRegression, LinearRegression, ensemble_oof, pl):
    # ── Load the unblinded labels and the submitted ensemble test predictions ────
    unblinded = (
        pl.read_csv("../data/raw/20260528/dose_response_test_unblinded.csv")
        .select(["Molecule Name", "pEC50"])
        .rename({"pEC50": "pEC50_true"})
    )
    ens_test = (
        pl.read_csv("../submissions/4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv")
        .select(["Molecule Name", "SMILES", "pEC50"])
        .rename({"pEC50": "pEC50_pred"})
    )
    unblinded_eval = ens_test.join(unblinded, on="Molecule Name", how="inner")

    # Fit GLOBAL calibrators on all ensemble OOF predictions (training data only).
    _xp = ensemble_oof["y_pred"].to_numpy()
    _yt = ensemble_oof["y_true"].to_numpy()
    calib_linear = LinearRegression().fit(_xp.reshape(-1, 1), _yt)
    calib_isotonic = IsotonicRegression(out_of_bounds="clip").fit(_xp, _yt)

    _raw = unblinded_eval["pEC50_pred"].to_numpy()
    unblinded_eval = unblinded_eval.with_columns(
        pl.Series("pred_linear", calib_linear.predict(_raw.reshape(-1, 1))),
        pl.Series("pred_isotonic", calib_isotonic.predict(_raw)),
    )

    print(f"Unblinded compounds evaluated: {unblinded_eval.shape[0]}")
    print(f"Linear de-shrink slope: {float(calib_linear.coef_[0]):.3f} "
          f"(>1 expands the range), intercept {float(calib_linear.intercept_):.3f}")
    return calib_isotonic, calib_linear, ens_test, unblinded_eval


@app.cell
def _(PLOTS_DIR, mean_absolute_error, mo, np, pl, plt, unblinded_eval):
    # ── Effect of calibration on the unblinded set (overall + hit zone) ──────────
    def _row(name: str, col: str) -> dict:
        yt = unblinded_eval["pEC50_true"].to_numpy()
        yp = unblinded_eval[col].to_numpy()
        hit = unblinded_eval.filter(pl.col("pEC50_true") >= 6.0)
        inact = unblinded_eval.filter(pl.col("pEC50_true") < 4.0)
        return {
            "variant": name,
            "MAE": round(mean_absolute_error(yt, yp), 4),
            "bias": round(float(np.mean(yp - yt)), 4),
            "hitzone_MAE": round(mean_absolute_error(
                hit["pEC50_true"].to_numpy(), hit[col].to_numpy()), 3),
            "hitzone_bias": round(float((hit[col] - hit["pEC50_true"]).mean()), 3),
            "inactive_bias": round(float((inact[col] - inact["pEC50_true"]).mean()), 3),
        }

    unblinded_summary = pl.DataFrame([
        _row("ensemble (raw)", "pEC50_pred"),
        _row("ensemble + linear", "pred_linear"),
        _row("ensemble + isotonic", "pred_isotonic"),
    ])

    with plt.style.context("seaborn-v0_8-whitegrid"):
        _fig, _axes = plt.subplots(1, 3, figsize=(13.5, 4.6), dpi=140, sharex=True, sharey=True)
        _yt = unblinded_eval["pEC50_true"].to_numpy()
        for _ax, (_name, _col) in zip(_axes, [
            ("Raw ensemble", "pEC50_pred"),
            ("+ linear de-shrink", "pred_linear"),
            ("+ isotonic", "pred_isotonic"),
        ]):
            _yp = unblinded_eval[_col].to_numpy()
            _err = np.abs(_yp - _yt)
            _sc = _ax.scatter(_yp, _yt, c=_err, cmap="RdYlGn_r", vmin=0, vmax=1.5,
                              s=22, alpha=0.8, edgecolors="none")
            _lims = [min(_yt.min(), _yp.min()) - 0.2, max(_yt.max(), _yp.max()) + 0.2]
            _ax.plot(_lims, _lims, "k--", linewidth=0.8, zorder=0)
            _ax.axhline(6.0, color="#888", linestyle=":", linewidth=0.8)
            _ax.set_xlim(_lims); _ax.set_ylim(_lims)
            _ax.set_xlabel("Predicted pEC50", fontsize=10)
            _ax.set_title(f"{_name}\nMAE={mean_absolute_error(_yt, _yp):.3f}", fontsize=10)
        _axes[0].set_ylabel("True pEC50 (unblinded)", fontsize=10)
        _fig.colorbar(_sc, ax=_axes, label="|error|", fraction=0.025, pad=0.02)
        _fig.suptitle("Calibration on the unblinded test set", fontsize=13, y=1.02)
        _fig.savefig(PLOTS_DIR / "unblinded_calibration.png", dpi=300, bbox_inches="tight")

    mo.vstack([
        mo.md("### Calibration on the unblinded set\n"
              "The decision-relevant question is the **hit zone**: does de-shrinking "
              "reduce the underprediction of genuinely potent compounds without hurting "
              "overall MAE?"),
        mo.ui.table(unblinded_summary.to_pandas(), selection=None, pagination=False),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(
    best_calibration,
    calib_isotonic,
    calib_linear,
    ens_test,
    mo,
    pl,
    winner,
):
    # ── Write the final submission: the CV-selected calibration of the ensemble ──
    # Only this single, CV-chosen strategy is applied to the held-out test set.
    from pathlib import Path as _Path
    _sub_dir = _Path("../submissions")
    _sub_dir.mkdir(parents=True, exist_ok=True)

    _raw = ens_test["pEC50_pred"].to_numpy()
    if best_calibration == "linear":
        _cal = calib_linear.predict(_raw.reshape(-1, 1))
    elif best_calibration == "isotonic":
        _cal = calib_isotonic.predict(_raw)
    else:
        _cal = _raw

    _out = _sub_dir / f"6_ensemble_calibrated_{best_calibration}_submission.csv"
    ens_test.with_columns(pl.Series("pEC50", _cal)).select(
        ["SMILES", "Molecule Name", "pEC50"]).write_csv(_out)

    if best_calibration == "raw":
        _msg = (f"5×5 CV selected **`raw`** — no calibration variant beat the "
                f"uncalibrated ensemble, so the submission is unchanged "
                f"(written as `{_out.name}` for completeness).")
    else:
        _msg = (f"Applied the CV-selected calibration **`{best_calibration}`** to all "
                f"{ens_test.shape[0]} test compounds → `{_out.name}`.")

    mo.md(f"""
    ## Conclusion

    {_msg}

    The largest CV improvement of *any* strategy over its own baseline was
    `{winner['analysis']} / {winner['best_variant']}` (ΔMAE = {winner['delta_mae']:+.4f}),
    but it acts on a single component model weaker than the ensemble and so is not
    used for the submission.

    **What the 5×5 CV told us, and what held up on the unblinded set:**

    - Post-hoc **de-shrinking** is the cleanest lever for the regression-to-the-mean
      bias, but its effect on overall MAE is small — the bias is conditional on
      activity, and a global 1-D map can only partly correct it. Its value is
      concentrated in the extreme bins (notably reduced hit-zone underprediction).
    - **Reweighting** shifts error from the dense middle to the extremes; whether
      that is worth a small overall-MAE cost depends on whether the hit zone is the
      priority (for screening, it usually is). The natural next step is to rebuild
      the ensemble from reweighted components.
    - **Oversampling** is the data-level twin of reweighting and is the only one of
      these that also reaches TabPFN and Macau; the MAE/bias-vs-factor curves show
      how far each model can be pushed before duplicate extremes start to hurt.
    - The **semi-pure data** is too small to move 5×5 CV much, but the label-noise
      audit is the more useful product: it quantifies how much of the residual error
      is irreducible measurement noise rather than model error.
    - **Filtering** is only worthwhile when a filter beats both the full-data baseline
      *and* its size-matched random control; the paired bars make that test explicit,
      separating genuine cleaning from the cost of simply training on fewer points.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 6 — Improved ensembles (augmentation + counter-filtering)

    Analyses 2–5 tested reweighting, oversampling, augmentation, and filtering
    *individually* on single-component models. This final analysis combines the two
    most promising data-level interventions —

    1. **Augmentation** — adding the 96-compound semi-pure set to training.
    2. **Counter-filtering** — removing compounds whose counter-screen pEC50 ≥ PXR
       pEC50 (non-selective hits).

    — and retrains *all five ensemble components* (Chemprop, CheMeleon, XGBoost,
    Macau, TabPFN) on the cleaned + augmented training set.

    Two ensembles are built with the same model mixture and weights as the
    submitted ensemble (`cp5·ch5·rf0·xg⅓·mc1·tf5`):

    | Ensemble | Models |
    |---|---|
    | **Default** | All components use default hyperparameters |
    | **HPO** | All components use the HPO-tuned hyperparameters from notebook 4 |

    Both ensembles are evaluated on the unblinded test set and compared against the
    original submitted ensemble.
    """)
    return


@app.cell
def _(
    Optional,
    Path,
    np,
    pl,
    shutil,
    subprocess,
    sys,
    tempfile,
    torch,
):
    # ── Chemprop model classes for Analysis 6 (scratch + CheMeleon) ─────────────
    # Both mirror notebook 4's implementations: full HPO parameter surface passed
    # as CLI args.  Shared helpers are cell-private; the two classes are exported.
    _A6_BIN = Path(sys.executable).parent / "chemprop"
    _A6_LOG = Path("../logs/6_a6_chemprop_cli.log")
    _A6_LOG.parent.mkdir(parents=True, exist_ok=True)
    _A6_SCRATCH_DIR = Path(tempfile.gettempdir()) / "6_a6_scratch_model"
    _A6_CHEMELEON_DIR = Path(tempfile.gettempdir()) / "6_a6_chemeleon_model"

    def _a6_device() -> str:
        return (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _a6_write_csv(
        smiles: list[str], targets: "np.ndarray | None",
        path: Path, target_col: str,
    ) -> None:
        cols: dict = {"smiles": smiles}
        if targets is not None:
            cols[target_col] = targets.flatten().tolist()
        pl.DataFrame(cols).write_csv(path)

    def _a6_run_cli(args: list[str]) -> None:
        cmd = [str(_A6_BIN)] + args
        with open(_A6_LOG, "a") as log:
            log.write(f"\n{'=' * 60}\nCMD: {' '.join(cmd)}\n{'=' * 60}\n")
            result = subprocess.run(cmd, stdout=log, stderr=log, text=True)
        if result.returncode != 0:
            print("\n".join(_A6_LOG.read_text().splitlines()[-30:]))
            raise RuntimeError(
                f"chemprop CLI failed (exit {result.returncode}). Log: {_A6_LOG}")

    class ChempropScratchModel:
        """Chemprop D-MPNN trained from scratch via CLI, with full HPO params."""

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _A6_SCRATCH_DIR,
            epochs: int = 50,
            message_hidden_dim: int = 300,
            depth: int = 3,
            dropout: float = 0.0,
            ffn_hidden_dim: int = 300,
            ffn_num_layers: int = 2,
            batch_size: int = 64,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
        ) -> None:
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type = pred_type
            self.model_dir = model_dir
            self.epochs = epochs
            self.message_hidden_dim = message_hidden_dim
            self.depth = depth
            self.dropout = dropout
            self.ffn_hidden_dim = ffn_hidden_dim
            self.ffn_num_layers = ffn_num_layers
            self.batch_size = batch_size
            self.init_lr = init_lr
            self.max_lr = max_lr
            self.final_lr = final_lr
            self.target_col: Optional[str] = None

        def _base_train_args(self, task_type: str, target_col: str) -> list[str]:
            return [
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _a6_device(),
                "--message-hidden-dim", str(self.message_hidden_dim),
                "--depth", str(self.depth),
                "--dropout", str(self.dropout),
                "--ffn-hidden-dim", str(self.ffn_hidden_dim),
                "--ffn-num-layers", str(self.ffn_num_layers),
                "--batch-size", str(self.batch_size),
                "--init-lr", str(self.init_lr),
                "--max-lr", str(self.max_lr),
                "--final-lr", str(self.final_lr),
            ]

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val: list[str],
            y_val: np.ndarray,
            target_col: str = "target",
        ) -> None:
            self.target_col = target_col
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "6_a6_scratch_train.csv"
            val_csv = tmp / "6_a6_scratch_val.csv"
            _a6_write_csv(X_train, y_train, train_csv, target_col)
            _a6_write_csv(X_val, y_val, val_csv, target_col)
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)
            task_type = "regression" if self.pred_type == "regression" else "binary"
            _a6_run_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.epochs),
                "--save-dir", str(self.model_dir),
            ])
            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            tmp = Path(tempfile.gettempdir())
            test_csv = tmp / "6_a6_scratch_test.csv"
            pred_csv = tmp / "6_a6_scratch_preds.csv"
            model_pt = self.model_dir / "model_0" / "best.pt"
            _a6_write_csv(X_test, None, test_csv, self.target_col)
            _a6_run_cli([
                "predict",
                "--test-path", str(test_csv),
                "--model-path", str(model_pt),
                "--preds-path", str(pred_csv),
            ])
            preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()
            test_csv.unlink(missing_ok=True)
            pred_csv.unlink(missing_ok=True)
            return preds.flatten()

    class ChempropChemeleonModel:
        """Chemprop D-MPNN fine-tuned from CheMeleon pretrained backbone via CLI."""

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _A6_CHEMELEON_DIR,
            epochs: int = 50,
            dropout: float = 0.0,
            ffn_hidden_dim: int = 900,
            ffn_num_layers: int = 2,
            batch_size: int = 64,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
        ) -> None:
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type = pred_type
            self.model_dir = model_dir
            self.epochs = epochs
            self.dropout = dropout
            self.ffn_hidden_dim = ffn_hidden_dim
            self.ffn_num_layers = ffn_num_layers
            self.batch_size = batch_size
            self.init_lr = init_lr
            self.max_lr = max_lr
            self.final_lr = final_lr
            self.target_col: Optional[str] = None

        def _base_train_args(self, task_type: str, target_col: str) -> list[str]:
            return [
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _a6_device(),
                "--dropout", str(self.dropout),
                "--ffn-hidden-dim", str(self.ffn_hidden_dim),
                "--ffn-num-layers", str(self.ffn_num_layers),
                "--batch-size", str(self.batch_size),
                "--init-lr", str(self.init_lr),
                "--max-lr", str(self.max_lr),
                "--final-lr", str(self.final_lr),
            ]

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val: list[str],
            y_val: np.ndarray,
            target_col: str = "target",
        ) -> None:
            self.target_col = target_col
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "6_a6_che_train.csv"
            val_csv = tmp / "6_a6_che_val.csv"
            _a6_write_csv(X_train, y_train, train_csv, target_col)
            _a6_write_csv(X_val, y_val, val_csv, target_col)
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)
            task_type = "regression" if self.pred_type == "regression" else "binary"
            _a6_run_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.epochs),
                "--from-foundation", "CHEMELEON",
                "--save-dir", str(self.model_dir),
            ])
            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            tmp = Path(tempfile.gettempdir())
            test_csv = tmp / "6_a6_che_test.csv"
            pred_csv = tmp / "6_a6_che_preds.csv"
            model_pt = self.model_dir / "model_0" / "best.pt"
            _a6_write_csv(X_test, None, test_csv, self.target_col)
            _a6_run_cli([
                "predict",
                "--test-path", str(test_csv),
                "--model-path", str(model_pt),
                "--preds-path", str(pred_csv),
            ])
            preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()
            test_csv.unlink(missing_ok=True)
            pred_csv.unlink(missing_ok=True)
            return preds.flatten()

    return ChempropChemeleonModel, ChempropScratchModel


@app.cell
def _(
    BoostedTreesModel,
    ChempropChemeleonModel,
    ChempropScratchModel,
    MacauModel,
    Path,
    chemeleon_embed,
    counter_remove,
    extract_fp_matrix,
    filter_train,
    gc,
    generate_fingerprint,
    mean_absolute_error,
    mo,
    np,
    pl,
    semipure,
    tabpfn_predict,
    unblinded_eval,
):
    # ── Prepare the augmented + counter-filtered training set ────────────────────
    _TARGET_COL = "pEC50_dr"
    _SEED = 42
    _SUB_DIR = Path("../submissions")
    _SUB_DIR.mkdir(parents=True, exist_ok=True)

    # Counter-filter: remove compounds where counter pEC50 >= PXR pEC50
    _rm_idx = counter_remove(filter_train)
    _keep_idx = sorted(set(range(filter_train.shape[0])) - _rm_idx)
    _train_filtered = filter_train[_keep_idx].select(
        ["smiles", "inchikey", "molecule_names", _TARGET_COL])
    print(f"Counter-filtered: {filter_train.shape[0]} → {_train_filtered.shape[0]} "
          f"(removed {len(_rm_idx)})")

    # Augment with semi-pure compounds (excluding any already in training)
    _existing_iks = set(_train_filtered["inchikey"].to_list())
    _sp_new = semipure.filter(~pl.col("inchikey").is_in(list(_existing_iks)))
    _train_aug = pl.concat([
        _train_filtered,
        _sp_new.select([
            "smiles", "inchikey",
            pl.lit("").alias("molecule_names"),
            pl.col("pEC50_corrected").alias(_TARGET_COL),
        ]),
    ])
    print(f"Augmented: {_train_filtered.shape[0]} + {_sp_new.shape[0]} semi-pure "
          f"= {_train_aug.shape[0]}")

    # 10% val split for early stopping
    _rng = np.random.default_rng(_SEED)
    _n = len(_train_aug)
    _val_idx = _rng.choice(_n, size=int(_n * 0.1), replace=False)
    _tr_idx = np.setdiff1d(np.arange(_n), _val_idx)
    _train_sub = _train_aug[_tr_idx.tolist()]
    _val_sub = _train_aug[_val_idx.tolist()]

    # Test set
    _test_df = pl.read_csv("../data/raw/20260409/dose_response_test.csv")
    _test_smiles = _test_df["SMILES"].to_list()
    _test_names = _test_df["Molecule Name"].to_list()

    # ── Features ────────────────────────────────────────────────────────────────
    # Mordred fingerprints
    _fp_tr = generate_fingerprint(_train_sub, "mordred")
    _fp_va = generate_fingerprint(_val_sub, "mordred")
    _fp_te = generate_fingerprint(
        pl.DataFrame({"smiles": _test_smiles, "inchikey": _test_names,
                      "molecule_names": _test_names}), "mordred")
    _Xtr_mor = extract_fp_matrix(_fp_tr, "mordred")
    _Xva_mor = extract_fp_matrix(_fp_va, "mordred")
    _Xte_mor = extract_fp_matrix(_fp_te, "mordred")
    del _fp_tr, _fp_va, _fp_te
    # Drop NaN columns (consistent mask across train/val/test)
    _valid_mor = ~np.isnan(_Xtr_mor).any(axis=0)
    _Xtr_mor = _Xtr_mor[:, _valid_mor]
    _Xva_mor = _Xva_mor[:, _valid_mor]
    _Xte_mor = _Xte_mor[:, _valid_mor]

    # CheMeleon embeddings
    _Xtr_che_full, _Xte_che = chemeleon_embed(
        _train_aug["smiles"].to_list(), _test_smiles, prefix="a6_ens")
    _Xtr_che = _Xtr_che_full[_tr_idx]
    _Xva_che = _Xtr_che_full[_val_idx]

    _y_tr = _train_sub[_TARGET_COL].to_numpy()
    _y_va = _val_sub[_TARGET_COL].to_numpy()
    _y_full = _train_aug[_TARGET_COL].to_numpy()

    # ── Ensemble weights (same as submitted: cp5·ch5·rf0·xg⅓·mc1·tf5) ──────────
    _ENS_WEIGHTS = {"cp": 5.0, "ch": 5.0, "xg": 1.0 / 3.0, "mc": 1.0, "tf": 5.0}
    _W_TOTAL = sum(_ENS_WEIGHTS.values())

    # ── HPO best parameters ─────────────────────────────────────────────────────
    # Chemprop scratch HPO: all params including message_hidden_dim and depth
    _CHEMPROP_HPO = {
        "epochs": 50, "batch_size": 32, "dropout": 0.0,
        "ffn_hidden_dim": 1024, "ffn_num_layers": 4,
        "max_lr": 0.00014521767021847913,
    }
    # CheMeleon HPO: same as chemprop but without encoder architecture params
    # (fixed by the CheMeleon backbone)
    _CHEMELEON_HPO = {
        k: v for k, v in _CHEMPROP_HPO.items()
        if k not in ("message_hidden_dim", "depth")
    }
    _XGB_HPO = {
        "colsample_bytree": 0.7280642324424045,
        "learning_rate": 0.019940375676697552,
        "max_depth": 5, "n_estimators": 800,
        "reg_alpha": 0.006750974160271454,
        "subsample": 0.7328142736138993,
    }
    _MACAU_HPO = {"burnin": 400, "nsamples": 700, "num_latent": 8}

    # ── Train both ensemble variants ────────────────────────────────────────────
    _results: dict[str, dict[str, np.ndarray]] = {}

    for _variant, _config in [
        ("default", {
            "cp_kw": {"epochs": 50},
            "ch_kw": {},
            "xg_kw": {},
            "mc_kw": {"seed": _SEED},
        }),
        ("hpo", {
            "cp_kw": _CHEMPROP_HPO,
            "ch_kw": _CHEMELEON_HPO,
            "xg_kw": _XGB_HPO,
            "mc_kw": {"seed": _SEED, **_MACAU_HPO},
        }),
    ]:
        print(f"\n{'=' * 50}\nTraining {_variant} ensemble\n{'=' * 50}")
        _preds: dict[str, np.ndarray] = {}

        # cp — Chemprop scratch
        print(f"  [{_variant}] cp — Chemprop scratch …")
        _cp = ChempropScratchModel(pred_type="regression", **_config["cp_kw"])
        _cp.train(
            _train_sub["smiles"].to_list(), _y_tr,
            _val_sub["smiles"].to_list(), _y_va,
            target_col=_TARGET_COL,
        )
        _preds["cp"] = _cp.predict(_test_smiles)
        del _cp; gc.collect()

        # ch — CheMeleon fine-tuned
        print(f"  [{_variant}] ch — CheMeleon …")
        _ch = ChempropChemeleonModel(pred_type="regression", **_config["ch_kw"])
        _ch.train(
            _train_sub["smiles"].to_list(), _y_tr,
            _val_sub["smiles"].to_list(), _y_va,
            target_col=_TARGET_COL,
        )
        _preds["ch"] = _ch.predict(_test_smiles)
        del _ch; gc.collect()

        # xg — XGBoost on Mordred
        print(f"  [{_variant}] xg — XGBoost Mordred …")
        _xg = BoostedTreesModel(pred_type="regression", **_config["xg_kw"])
        _xg.train(_Xtr_mor, _y_tr, _Xva_mor, _y_va)
        _preds["xg"] = _xg.predict(_Xte_mor)
        del _xg; gc.collect()

        # mc — Macau on CheMeleon FP (uses full training set, no early stopping)
        print(f"  [{_variant}] mc — Macau CheMeleon …")
        _mc = MacauModel(**_config["mc_kw"])
        _mc.train(_Xtr_che_full, _y_full)
        _preds["mc"] = _mc.predict(_Xte_che)
        del _mc; gc.collect()

        # tf — TabPFN on CheMeleon FP (no HPO — in-context model)
        print(f"  [{_variant}] tf — TabPFN CheMeleon …")
        _preds["tf"] = tabpfn_predict(_Xtr_che_full, _y_full, _Xte_che)
        gc.collect()

        _results[_variant] = _preds

        # Compute weighted ensemble
        _ens_pred = sum(
            _preds[tag] * (w / _W_TOTAL)
            for tag, w in _ENS_WEIGHTS.items()
        )
        # Save submission
        _sub_path = _SUB_DIR / f"6_ens_{_variant}_augfilt_submission.csv"
        pl.DataFrame({
            "SMILES": _test_smiles,
            "Molecule Name": _test_names,
            "pEC50": _ens_pred.tolist(),
        }).write_csv(_sub_path)
        print(f"  → {_sub_path.name}")

    # ── Evaluate on unblinded test set ──────────────────────────────────────────
    _eval_rows = []
    _yt = unblinded_eval["pEC50_true"].to_numpy()

    # Original ensemble
    _yp_orig = unblinded_eval["pEC50_pred"].to_numpy()
    _hit_orig = unblinded_eval.filter(pl.col("pEC50_true") >= 6.0)
    _eval_rows.append({
        "ensemble": "original (nb 4)",
        "MAE": round(mean_absolute_error(_yt, _yp_orig), 4),
        "hitzone_bias": round(float(
            (_hit_orig["pEC50_pred"] - _hit_orig["pEC50_true"]).mean()), 3),
    })

    for _variant in ["default", "hpo"]:
        _ens_pred = sum(
            _results[_variant][tag] * (w / _W_TOTAL)
            for tag, w in _ENS_WEIGHTS.items()
        )
        # Join with unblinded to align compounds
        _pred_df = pl.DataFrame({
            "Molecule Name": _test_names,
            "pEC50_new": _ens_pred.tolist(),
        })
        _joined = unblinded_eval.join(_pred_df, on="Molecule Name", how="inner")
        _yt_j = _joined["pEC50_true"].to_numpy()
        _yp_j = _joined["pEC50_new"].to_numpy()
        _hit = _joined.filter(pl.col("pEC50_true") >= 6.0)
        _eval_rows.append({
            "ensemble": f"aug+filt ({_variant})",
            "MAE": round(mean_absolute_error(_yt_j, _yp_j), 4),
            "hitzone_bias": round(float(
                (_hit["pEC50_new"] - _hit["pEC50_true"]).mean()), 3),
        })

    ens_comparison = pl.DataFrame(_eval_rows)

    mo.vstack([
        mo.md("### Improved ensemble comparison on unblinded test set\n"
              "Both new ensembles use the same weights as the original submission "
              "(`cp5·ch5·xg⅓·mc1·tf5`) but are retrained on counter-filtered + "
              "semi-pure-augmented data."),
        mo.ui.table(ens_comparison.to_pandas(), selection=None, pagination=False),
    ])
    return


if __name__ == "__main__":
    app.run()
