import marimo

__generated_with = "0.23.5"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
    import json
    import logging
    import joblib
    import gc
    import math
    import warnings
    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Iterator, Literal, Optional
    from urllib.request import urlretrieve

    import matplotlib.pyplot as plt
    import numpy as np
    import polars as pl
    import marimo as mo
    import altair as alt
    import pingouin as pg
    import seaborn as sns

    from scipy import stats
    from scipy.stats import spearmanr
    from sklearn.metrics import (
        accuracy_score,
        r2_score,
        balanced_accuracy_score,
        f1_score,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        precision_score,
        recall_score,
        roc_auc_score,
    )
    from sklearn.model_selection._split import _BaseKFold as BaseKFold
    from statsmodels.stats.anova import AnovaRM
    from statsmodels.stats.libqsturng import psturng, qsturng

    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
    import xgboost as xgb

    import torch
    from torch import nn, optim
    from torch.functional import F
    from torch.utils.data import DataLoader

    import lightning as L
    from lightning import pytorch as pyl
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping

    from chemprop import data, featurizers, models
    from chemprop import nn as chemnn
    from chemprop.data import BatchMolGraph
    from chemprop.models import MPNN
    from chemprop.nn import RegressionFFN
    from rdkit.Chem import MolFromSmiles

    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

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

    import gzip
    import shutil
    import subprocess
    import sys
    import tempfile

    import pandas as pd
    import matplotlib.patches as mpatches
    from tqdm.auto import tqdm
    from typing import Iterable

    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect

    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    from dotenv import load_dotenv
    load_dotenv(Path("../.env"))

    from tabpfn import TabPFNClassifier, TabPFNRegressor
    import smurff
    import scipy.sparse as sp

    return (
        AtomPairFingerprint,
        AvalonFingerprint,
        BaseKFold,
        ConformerGenerator,
        E3FPFingerprint,
        ECFPFingerprint,
        Iterator,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        Optional,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        RandomForestClassifier,
        RandomForestRegressor,
        TabPFNClassifier,
        TabPFNRegressor,
        TopologicalTorsionFingerprint,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        gc,
        gzip,
        json,
        math,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        optuna,
        pd,
        pg,
        pl,
        plt,
        precision_score,
        psturng,
        qsturng,
        r2_score,
        recall_score,
        roc_auc_score,
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
def _(Path, json, np, subprocess, sys, tempfile):
    # ── CheMeleon embedding subprocess script ─────────────────────────────────
    # CheMeleon (PyTorch) must run in an isolated subprocess to avoid the
    # OpenMP runtime collision with sklearn and smurff in the parent process.
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

        Runs CheMeleon in a child process so PyTorch's libkmp never shares the
        address space with sklearn or smurff in the parent.

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
def _(mo):
    mo.md(r"""
    ## Model classes

    Four model types share a stable API:

    ```
    model.train(train_df, target_col, task, **kwargs)
    model.predict(df)          -> np.ndarray
    ```

    | Class | Backend | Input features |
    |---|---|---|
    | `RandomForestModel` | sklearn RF | fingerprint column |
    | `BoostedTreesModel` | XGBoost | fingerprint column |
    | `TabPFNModel` | TabPFN (in-context transformer) | fingerprint column |
    | `MacauModel` | Macau / Bayesian matrix factorization (smurff) | fingerprint column |
    | `ChempropModel` | Chemprop v2 MPNN from scratch | SMILES column |
    | `ChempropChemeleonModel` | Chemprop v2 fine-tuned from [CheMeleon](https://github.com/JacksonBurns/chemeleon) backbone | SMILES column |


    `task` is either `"regression"` or `"classification"`.
    Classification `predict()` returns the probability of the positive class.
    """)
    return


@app.cell
def _(np, pl):
    def extract_fp_matrix(df: pl.DataFrame, fp_col: str) -> np.ndarray:
        """
        Extract a 2-D float32 feature matrix from a fingerprint column.

        The column is expected to hold numpy arrays of equal length as produced
        by generate_fingerprint.

        Args:
            df: DataFrame with a fingerprint column.
            fp_col: Column name.

        Returns:
            2-D array of shape (n_compounds, fp_size).
        """
        return np.stack(df[fp_col].to_list()).astype(np.float32)

    return (extract_fp_matrix,)


@app.cell
def _(RandomForestClassifier, RandomForestRegressor, np):
    class RandomForestModel:
        """Scikit-learn Random Forest model with a unified fit/predict interface."""

        def __init__(
            self,
            pred_type: str = "classification",
            n_estimators: int = 100,
            max_depth: int | None = None,
            min_samples_split: int = 2,
            min_samples_leaf: int = 1,
            max_features: str | float = "sqrt",
            max_samples: float | None = None,
            class_weight: str | dict | None = None,
            random_state: int | None = None,
            n_jobs: int = -1,
        ) -> None:
            """
            Args:
                pred_type: "classification" (RandomForestClassifier) or
                    "regression" (RandomForestRegressor).
                n_estimators: Number of trees in the forest.
                max_depth: Maximum depth of each tree. None grows trees until
                    leaves are pure or contain fewer than min_samples_split samples.
                min_samples_split: Minimum number of samples required to split an
                    internal node.
                min_samples_leaf: Minimum number of samples required to be at a
                    leaf node.
                max_features: Number of features to consider at each split.
                    "sqrt" (default), "log2", a float fraction, or an int count.
                max_samples: Fraction of training samples drawn for each tree
                    (bootstrap). None uses all samples.
                class_weight: Only meaningful for classification. "balanced" adjusts
                    weights inversely proportional to class frequencies.
                random_state: Random seed for reproducibility.
                n_jobs: Number of parallel jobs for fitting trees. -1 uses all CPUs.

            Raises:
                ValueError: If pred_type is not "classification" or "regression".
            """
            self.model = None
            self.pred_type = pred_type
            common_kwargs = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                min_samples_split=min_samples_split,
                min_samples_leaf=min_samples_leaf,
                max_features=max_features,
                max_samples=max_samples,
                random_state=random_state,
                n_jobs=n_jobs,
            )
            if pred_type == "classification":
                self.model = RandomForestClassifier(
                    **common_kwargs, class_weight=class_weight
                )
            elif pred_type == "regression":
                self.model = RandomForestRegressor(**common_kwargs)
            else:
                raise ValueError(
                    "pred_type must be either 'classification' or 'regression'"
                )

        def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """
            Fit the model on the training data.

            Args:
                X_train: Training feature matrix.
                y_train: Training labels or values.
            """
            self.model.fit(X_train, y_train)

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """
            Generate predictions for the test set.

            Args:
                X_test: Test feature matrix.

            Returns:
                Predicted probabilities (classification) or values (regression).
            """
            if self.pred_type == "classification":
                return self.model.predict_proba(X_test)[:, 1]
            else:
                return self.model.predict(X_test)

    return (RandomForestModel,)


@app.cell
def _(np, xgb):
    class BoostedTreesModel:
        """XGBoost gradient-boosted tree model with a unified fit/predict interface."""

        def __init__(
            self,
            pred_type: str = "classification",
            n_estimators: int = 100,
            max_depth: int = 6,
            learning_rate: float = 0.3,
            subsample: float = 1.0,
            colsample_bytree: float = 1.0,
            min_child_weight: float = 1.0,
            gamma: float = 0.0,
            reg_alpha: float = 0.0,
            reg_lambda: float = 1.0,
            early_stopping_rounds: int = 10,
            scale_pos_weight: float = 1.0,
        ) -> None:
            """
            Args:
                pred_type: "classification" (XGBClassifier) or "regression" (XGBRegressor).
                n_estimators: Maximum number of boosting rounds.
                max_depth: Maximum depth of each tree. Deeper trees capture more
                    interactions but increase overfitting risk.
                learning_rate: Step-size shrinkage (η). Lower values require more
                    trees but generalise better.
                subsample: Fraction of training rows sampled per tree (0 < subsample ≤ 1).
                colsample_bytree: Fraction of features sampled per tree.
                min_child_weight: Minimum sum of instance weights in a child node.
                    Higher values prevent learning on rare samples.
                gamma: Minimum loss reduction required to make a further split.
                    Acts as a regularisation threshold.
                reg_alpha: L1 regularisation on leaf weights (increases sparsity).
                reg_lambda: L2 regularisation on leaf weights (shrinks weights).
                early_stopping_rounds: Stop training if validation metric does not
                    improve for this many consecutive rounds.
                scale_pos_weight: Ratio of negative to positive class counts.
                    Only meaningful for classification; set to n_neg/n_pos for
                    imbalanced datasets.

            Raises:
                ValueError: If pred_type is not "classification" or "regression".
            """
            self.model = None
            self.pred_type = pred_type
            common_kwargs = dict(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=learning_rate,
                subsample=subsample,
                colsample_bytree=colsample_bytree,
                min_child_weight=min_child_weight,
                gamma=gamma,
                reg_alpha=reg_alpha,
                reg_lambda=reg_lambda,
                early_stopping_rounds=early_stopping_rounds,
                tree_method="hist",
                n_jobs=-1,
            )
            if pred_type == "classification":
                self.model = xgb.XGBClassifier(
                    **common_kwargs, scale_pos_weight=scale_pos_weight
                )
            elif pred_type == "regression":
                self.model = xgb.XGBRegressor(**common_kwargs)
            else:
                raise ValueError(
                    "pred_type must be either 'classification' or 'regression'"
                )

        def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val: np.ndarray, y_val: np.ndarray) -> None:
            """
            Fit the model using training data with an evaluation set for early stopping.

            Args:
                X_train: Training feature matrix.
                y_train: Training labels or values.
                X_val: Validation feature matrix used for early stopping.
                y_val: Validation labels or values.
            """
            self.model.fit(X_train, y_train,
                           eval_set=[(X_val, y_val)], verbose=False)

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """
            Generate predictions for the test set.

            Args:
                X_test: Test feature matrix.

            Returns:
                Predicted probabilities (classification) or values (regression).
            """
            if self.pred_type == "classification":
                return self.model.predict_proba(X_test)[:, 1]
            else:
                return self.model.predict(X_test)

    return (BoostedTreesModel,)


@app.cell
def _(TabPFNClassifier, TabPFNRegressor, np):
    class TabPFNModel:
        """
        TabPFN (Tabular Prior-Fitted Networks) model with a unified fit/predict interface.

        TabPFN is a transformer pretrained on synthetic tabular datasets that performs
        in-context learning: the entire training set is passed as a "prompt" at
        inference time, making fit() essentially instantaneous.

        Because TabPFN operates on a numeric feature matrix it must be paired with
        a pre-computed fingerprint column (via generate_fingerprint / extract_fp_matrix)
        rather than raw SMILES strings.

        ## Size considerations

        The upstream pretraining limits are disabled by default
        (ignore_pretraining_limits=True) so the model can handle the full training
        sets used in CV.  Very large training sets (> ~10 000 rows) will increase
        inference time and memory substantially because the full training matrix is
        held in the transformer's context.

        Usage::

            model = TabPFNModel(pred_type="regression", n_estimators=8)
            model.train(X_train, y_train)
            preds = model.predict(X_test)
        """

        def __init__(
            self,
            pred_type: str = "regression",
            n_estimators: int = 8,
            random_state: int = 0,
            ignore_pretraining_limits: bool = True,
            device: str = "cpu",
        ) -> None:
            """
            Args:
                pred_type: "regression" (TabPFNRegressor) or "classification"
                    (TabPFNClassifier).
                n_estimators: Number of ensemble forward passes. Higher values
                    improve stability at the cost of inference time.
                random_state: Random seed for reproducibility.
                ignore_pretraining_limits: When True (default), disables the
                    upstream row / feature count limits so the model can be used
                    on full-sized training sets.
                device: PyTorch device.  Defaults to "cpu" — MPS causes OOM
                    errors on the full CV training sets (~3 300 samples) due to
                    the 8.3 GB MPS allocator ceiling on Apple Silicon.

            Raises:
                ValueError: If pred_type is not "regression" or "classification".
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type = pred_type
            common_kwargs = dict(
                n_estimators=n_estimators,
                random_state=random_state,
                ignore_pretraining_limits=ignore_pretraining_limits,
                device=device,
            )
            if pred_type == "regression":
                self.model = TabPFNRegressor(**common_kwargs)
            else:
                self.model = TabPFNClassifier(**common_kwargs)

        def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """
            Fit TabPFN on the training feature matrix.

            Unlike gradient-boosted or neural models, TabPFN stores the training
            data internally and performs no iterative optimisation.  fit() returns
            almost immediately; the actual computation happens during predict().

            Args:
                X_train: 2-D float32 feature matrix of shape (n_train, n_features).
                y_train: 1-D array of training labels or values.
            """
            self.model.fit(X_train, y_train)

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """
            Generate predictions for the test feature matrix.

            Args:
                X_test: 2-D float32 feature matrix of shape (n_test, n_features).

            Returns:
                For regression: 1-D array of predicted values.
                For classification: 1-D array of predicted probabilities for the
                positive class (index 1).
            """
            if self.pred_type == "classification":
                return self.model.predict_proba(X_test)[:, 1]
            else:
                return self.model.predict(X_test)

    return


@app.cell
def _(np, smurff, sp, tempfile):
    class MacauModel:
        """
        Bayesian matrix factorization model using the Macau algorithm (via smurff).

        Macau extends BPMF (Bayesian Probabilistic Matrix Factorization) with
        side information: molecular fingerprints are used as row-side features,
        linking the latent compound factors to the chemical structure via a
        learned link matrix.  The model is fully Bayesian — predictions are the
        average of posterior Gibbs samples after a burn-in phase.

        Because the model operates on a numeric feature matrix it must be paired
        with a pre-computed fingerprint column (via generate_fingerprint /
        extract_fp_matrix) rather than raw SMILES strings.

        ## How it works

        Training data is represented as a sparse matrix Y of shape
        (n_compounds, 1).  The fingerprint matrix (n_compounds, n_features) is
        passed as row side information.  During MCMC sampling smurff jointly
        infers:

        - latent compound factors U  (num_latent × n_compounds)
        - latent target factor  V  (num_latent × 1)
        - link matrix Beta  connecting fingerprints to U

        Prediction for new compounds uses their fingerprints to impute U via
        Beta, then computes U · V.

        Usage::

            model = MacauModel(num_latent=16, nsamples=200, burnin=100)
            model.train(X_train_fp, y_train)
            preds = model.predict(X_test_fp)
        """

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
            """
            Args:
                num_latent: Number of latent dimensions for the factorization.
                    Higher values capture more complex structure but increase
                    computation and overfitting risk.
                burnin: Number of Gibbs iterations to discard as burn-in before
                    collecting posterior samples.
                nsamples: Number of posterior Gibbs samples to collect.
                    Predictions are averaged across these samples.
                univariate: When True, use the faster univariate sampler instead
                    of the joint multivariate sampler.  Useful for very large
                    datasets; may converge more slowly.
                direct: When True, use the Cholesky (direct) solver for the
                    link matrix; otherwise use conjugate gradient (CG).
                    Direct is faster for small side-info matrices; CG scales
                    better to very high-dimensional fingerprints.
                num_threads: Number of OpenMP threads.  None lets smurff decide.
                seed: Random seed for the Gibbs sampler.
            """
            self.num_latent  = num_latent
            self.burnin      = burnin
            self.nsamples    = nsamples
            self.univariate  = univariate
            self.direct      = direct
            self.num_threads = num_threads
            self.seed        = seed
            # Stored after train() so predict() can reuse the session
            self._predict_session = None

        def train(self, X_train: np.ndarray, y_train: "np.ndarray | sp.spmatrix") -> None:
            """
            Fit the Macau model via Gibbs sampling.

            Accepts either a 1-D target array (single-task) or a pre-built sparse
            (n × k) matrix (multitask).  In the single-task case a sparse (n × 1)
            COO matrix is constructed automatically, omitting any NaN values.
            MacauSession uses X_train as row side information.

            Args:
                X_train: 2-D float32 fingerprint matrix, shape (n_train, n_features).
                y_train: 1-D target array, or sparse (n_train × k) target matrix.
            """
            if sp.issparse(y_train):
                Y_train = y_train.astype(np.float64)
            else:
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
                # Silence smurff: verbose=0 quiets the C++ layer; we also raise
                # the root logger threshold to suppress the Python-side INFO lines.
                # We bypass session.run() entirely (which would create a tqdm bar)
                # and call init() + step() directly instead.
                _smurff_logger = logging.getLogger("smurff")
                _root_logger   = logging.getLogger()
                _prev_smurff   = _smurff_logger.level
                _prev_root     = _root_logger.level
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
                    # Manual loop — equivalent to session.run() but without tqdm
                    session.init()
                    while session.step():
                        pass
                finally:
                    _smurff_logger.setLevel(_prev_smurff)
                    _root_logger.setLevel(_prev_root)
                self._predict_session = session.makePredictSession()

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """
            Generate predictions by averaging posterior Gibbs samples.

            Uses the fingerprint matrix of test compounds as row side information
            to project into the latent space and compute the predicted target value.

            Args:
                X_test: 2-D float32 fingerprint matrix, shape (n_test, n_features).

            Returns:
                1-D array of predicted values, averaged across all posterior samples.

            Raises:
                RuntimeError: If called before train().
            """
            if self._predict_session is None:
                raise RuntimeError("Call train() before predict().")
            # predict() returns a list of (n_test × 1) arrays, one per sample
            sample_arrays = self._predict_session.predict(
                (X_test.astype(np.float64), slice(None))
            )
            return np.mean(np.array(sample_arrays), axis=0).flatten()

    return (MacauModel,)


@app.cell
def _(Optional, Path, np, pl, shutil, subprocess, sys, tempfile, torch):
    import os
    # Resolve the chemprop CLI from the same venv as the running interpreter
    _CHEMPROP_BIN = Path(sys.executable).parent / "chemprop"

    # Persistent temp directories — one per model type, reused across CV folds.
    # Using a fixed path (not TemporaryDirectory) so the folder survives between
    # train() and predict() calls within the same session.
    _CHEMPROP_MODEL_DIR  = Path(tempfile.gettempdir()) / "chemprop_scratch_model"
    _CHEMELEON_MODEL_DIR = Path(tempfile.gettempdir()) / "chemprop_chemeleon_model"

    def _get_device() -> str:
        """
        Detect and return the best available compute device for PyTorch.

        Returns:
            "cuda" if an NVIDIA/AMD GPU is available, "mps" if running on Apple
            Silicon with Metal Performance Shaders, otherwise "cpu".
        """
        return (
                "cuda" # Device for NVIDIA or AMD GPUs
                if torch.cuda.is_available()
                else "mps" # Device for Apple Silicon (Metal Performance Shaders)
                if torch.backends.mps.is_available()
                else "cpu"
            )

    def _write_smiles_csv(
        smiles: list[str],
        targets: Optional[np.ndarray],
        path: Path,
        target_col: str,
    ) -> None:
        """
        Write a CSV file with a smiles column and an optional target column.

        Args:
            smiles: List of SMILES strings.
            targets: 1-D array of target values, or None for inference-only files.
            path: Destination file path.
            target_col: Name of the target column.
        """
        if targets is not None:
            df = pl.DataFrame({"smiles": smiles, target_col: targets.flatten().tolist()})
        else:
            df = pl.DataFrame({"smiles": smiles})
        df.write_csv(path)

    # Log file for all chemprop CLI calls — persisted in project logs/ folder.
    _CHEMPROP_LOG = Path("../logs/chemprop_cli.log")
    _CHEMPROP_LOG.parent.mkdir(parents=True, exist_ok=True)

    def _run_chemprop_cli(args: list[str]) -> None:
        """
        Run the chemprop CLI as a subprocess, redirecting all output to a log file.

        stdout and stderr are appended to _CHEMPROP_LOG so the notebook stays
        quiet. On failure the tail of the log is printed to help diagnose the error.

        Args:
            args: Argument list passed after the `chemprop` binary.

        Raises:
            RuntimeError: If the process exits with a non-zero return code.
        """
        cmd = [str(_CHEMPROP_BIN)] + args
        # Pass PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0 to remove MPS allocator ceiling,
        # preventing macOS memory pressure stalls between folds on Apple Silicon.
        _env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
        with open(_CHEMPROP_LOG, "a") as _log:
            _log.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\n{'='*60}\n")
            result = subprocess.run(cmd, stdout=_log, stderr=_log, text=True, env=_env)
        if result.returncode != 0:
            # Print only the last 30 lines of the log to surface the error
            lines = _CHEMPROP_LOG.read_text().splitlines()
            print("\n".join(lines[-30:]))
            raise RuntimeError(
                f"chemprop CLI failed (exit {result.returncode}). "
                f"Full log: {_CHEMPROP_LOG}"
            )

    class ChempropModel:
        """
        Chemprop D-MPNN trained from scratch via the chemprop CLI.

        train() and predict() shell out to `chemprop train` / `chemprop predict`
        rather than using the Python API, avoiding MPS memory issues when running
        many CV folds inside a notebook kernel.

        The trained model is written to a fixed temporary directory
        (/tmp/chemprop_scratch_model) which is overwritten on each train() call
        so no disk space accumulates across folds.

        ## Transfer learning workflow

        Call pretrain() once on an auxiliary dataset before the CV loop, then
        call train() as normal.  train() detects the saved pretrain checkpoint
        and passes it to `chemprop train --checkpoint`, initialising the
        encoder weights from the pretraining run.

        Optionally set freeze_encoder=True to lock the message-passing weights
        during fine-tuning, updating only the FFN head.

        Example::

            model = ChempropModel(pred_type="regression", freeze_encoder=True)
            # pretrain once on a large auxiliary dataset
            model.pretrain(X_aux, y_aux, X_val_aux, y_val_aux, target_col="pKi")
            # fine-tune (with the encoder frozen) on the target dataset
            model.train(X_train, y_train, X_val, y_val, target_col="pXC50")
            preds = model.predict(X_test)
        """

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _CHEMPROP_MODEL_DIR,
            pretrain_dir: Optional[Path] = None,
            freeze_encoder: bool = False,
            epochs: int = 50,
            pretrain_epochs: int = 50,
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
            """
            Args:
                pred_type: "regression" or "classification".
                model_dir: Directory where fine-tuning checkpoints are written.
                    Cleared before every train() call.
                pretrain_dir: Directory where pretraining checkpoints are
                    written and persisted.  Defaults to
                    /tmp/chemprop_pretrain_model when None.  Not cleared
                    between CV folds so the same pretrained encoder can be
                    reused across all folds.
                freeze_encoder: When True, the message-passing encoder loaded
                    from the pretrain checkpoint is frozen during fine-tuning
                    (--freeze-encoder).  Ignored if no pretrain checkpoint
                    exists.
                epochs: Maximum training epochs for train().
                pretrain_epochs: Maximum training epochs for pretrain().
                message_hidden_dim: Hidden dimension of the MPNN message-passing
                    layers (--message-hidden-dim).
                depth: Number of message-passing steps (--depth).
                dropout: Dropout probability applied after each message-passing
                    and FFN layer (--dropout).
                ffn_hidden_dim: Hidden dimension of the feed-forward network
                    (--ffn-hidden-dim).
                ffn_num_layers: Number of layers in the feed-forward network
                    (--ffn-num-layers).
                batch_size: Mini-batch size for training (--batch-size).
                init_lr: Initial learning rate for the one-cycle scheduler
                    (--init-lr).
                max_lr: Peak learning rate for the one-cycle scheduler (--max-lr).
                final_lr: Final learning rate for the one-cycle scheduler
                    (--final-lr).
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type          = pred_type
            self.model_dir          = model_dir
            self.pretrain_dir       = pretrain_dir or Path(tempfile.gettempdir()) / "chemprop_pretrain_model"
            self.freeze_encoder     = freeze_encoder
            self.epochs             = epochs
            self.pretrain_epochs    = pretrain_epochs
            self.message_hidden_dim = message_hidden_dim
            self.depth              = depth
            self.dropout            = dropout
            self.ffn_hidden_dim     = ffn_hidden_dim
            self.ffn_num_layers     = ffn_num_layers
            self.batch_size         = batch_size
            self.init_lr            = init_lr
            self.max_lr             = max_lr
            self.final_lr           = final_lr
            self.target_col: Optional[str] = None  # set during train()

        # ── internal helpers ────────────────────────────────────────────────

        def _base_train_args(self, task_type: str, target_col: str) -> list[str]:
            """Return CLI args shared between pretrain() and train()."""
            return [
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _get_device(),
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

        # ── public API ──────────────────────────────────────────────────────

        def pretrain(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "pretrain_target",
        ) -> None:
            """
            Pretrain the model on an auxiliary dataset.

            Trains a full D-MPNN from scratch on the supplied data and saves
            the checkpoint to pretrain_dir.  Call this once before the CV loop
            so every subsequent train() call can warm-start from the same
            encoder weights.

            The pretrain_dir is **not** cleared between calls, so a second
            call to pretrain() will overwrite the previous checkpoint.

            Args:
                X_train: SMILES strings for the auxiliary training set.
                y_train: Auxiliary training targets, shape (n,) or (n, 1).
                X_val:   SMILES strings for auxiliary validation (early stopping).
                y_val:   Auxiliary validation targets.
                target_col: Column name used in the temporary CSV files.
                    Should differ from the fine-tuning target_col to avoid
                    confusion in log files.
            """
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "chemprop_pretrain_train.csv"
            val_csv   = tmp / "chemprop_pretrain_val.csv"

            _write_smiles_csv(X_train, y_train, train_csv, target_col)
            _write_smiles_csv(X_val,   y_val,   val_csv,   target_col)

            if self.pretrain_dir.exists():
                shutil.rmtree(self.pretrain_dir)

            task_type = "regression" if self.pred_type == "regression" else "classification"
            _run_chemprop_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.pretrain_epochs),
                "--save-dir", str(self.pretrain_dir),
            ])

            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "target",
        ) -> None:
            """
            Train (or fine-tune) the model on the target dataset.

            If pretrain() was called beforehand and the checkpoint exists at
            pretrain_dir/model_0/best.pt, the encoder weights are loaded via
            `--checkpoint`.  When freeze_encoder=True the message-passing
            layers are also frozen so only the FFN head is updated.

            Writes temporary CSV files for train and val sets, runs the CLI,
            then removes the CSVs. The model_dir is cleared before each run.

            Args:
                X_train: SMILES strings for training.
                y_train: Training targets, shape (n,) or (n, 1).
                X_val:   SMILES strings for validation (early stopping).
                y_val:   Validation targets, shape (n,) or (n, 1).
                target_col: Column name used in the temporary CSV files.
            """
            self.target_col = target_col
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "chemprop_train.csv"
            val_csv   = tmp / "chemprop_val.csv"

            _write_smiles_csv(X_train, y_train, train_csv, target_col)
            _write_smiles_csv(X_val,   y_val,   val_csv,   target_col)

            # Remove stale fine-tuning checkpoints so the CLI starts fresh
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)

            task_type = "regression" if self.pred_type == "regression" else "classification"
            args = [
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.epochs),
                "--save-dir", str(self.model_dir),
            ]

            # Warm-start from pretrain checkpoint when it exists
            pretrain_ckpt = self.pretrain_dir / "model_0" / "best.pt"
            if pretrain_ckpt.exists():
                args += ["--checkpoint", str(pretrain_ckpt)]
                if self.freeze_encoder:
                    args.append("--freeze-encoder")

            _run_chemprop_cli(args)

            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            """
            Run inference by calling `chemprop predict` via subprocess.

            Writes a temporary SMILES CSV, runs the CLI, reads the output CSV,
            then removes both temporary files.

            Args:
                X_test: SMILES strings to predict.

            Returns:
                1-D numpy array of predicted values.
            """
            tmp = Path(tempfile.gettempdir())
            test_csv = tmp  / "chemprop_test.csv"
            pred_csv = tmp  / "chemprop_preds.csv"
            # The best.pt written by `chemprop train` into model_dir/model_0/
            model_pt = self.model_dir / "model_0" / "best.pt"

            _write_smiles_csv(X_test, None, test_csv, self.target_col)

            _run_chemprop_cli([
                "predict",
                "--test-path",  str(test_csv),
                "--model-path", str(model_pt),
                "--preds-path", str(pred_csv),
            ])

            preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()

            test_csv.unlink(missing_ok=True)
            pred_csv.unlink(missing_ok=True)

            return preds.flatten()

    class ChempropChemeleonModel:
        """
        Chemprop D-MPNN fine-tuned from the CheMeleon pretrained backbone via the CLI.

        Identical interface to ChempropModel but passes `--from-foundation CHEMELEON`
        to `chemprop train` so the MPNN encoder is always warm-started from the
        CheMeleon weights.  The CLI downloads and caches those weights
        automatically at ~/.chemprop/chemeleon_mp.pt on the first call.

        ## Transfer learning workflow

        Call pretrain() once on an auxiliary dataset before the CV loop, then
        call train() as normal.  pretrain() itself starts from the CheMeleon
        backbone; train() then warm-starts from the pretrain checkpoint,
        giving a three-stage initialisation chain:

            CheMeleon  →  pretrain (auxiliary data)  →  train (target data)

        Optionally set freeze_encoder=True to lock the message-passing weights
        during fine-tuning, updating only the FFN head.

        Reference: https://github.com/JacksonBurns/chemeleon
        """

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _CHEMELEON_MODEL_DIR,
            pretrain_dir: Optional[Path] = None,
            freeze_encoder: bool = False,
            epochs: int = 50,
            pretrain_epochs: int = 50,
            dropout: float = 0.0,
            ffn_hidden_dim: int = 900,
            ffn_num_layers: int = 2,
            batch_size: int = 64,
            init_lr: float = 1e-4,
            max_lr: float = 1e-3,
            final_lr: float = 1e-4,
        ) -> None:
            """
            Args:
                pred_type: "regression" or "classification".
                model_dir: Directory where fine-tuning checkpoints are written.
                    Distinct from ChempropModel's default to avoid collisions.
                pretrain_dir: Directory for pretraining checkpoints.  Defaults
                    to /tmp/chemeleon_pretrain_model when None.
                freeze_encoder: When True, the message-passing encoder from the
                    pretrain checkpoint is frozen during fine-tuning.  Ignored
                    if no pretrain checkpoint exists.
                epochs: Maximum training epochs for train().
                pretrain_epochs: Maximum training epochs for pretrain().
                dropout: Dropout probability applied after each FFN layer. Note
                    that message-passing architecture is fixed by CheMeleon
                    (2048 d_h, depth 6) and cannot be changed during fine-tuning.
                ffn_hidden_dim: Hidden dimension of the task-specific
                    feed-forward head (--ffn-hidden-dim).
                ffn_num_layers: Number of layers in the feed-forward head
                    (--ffn-num-layers).
                batch_size: Mini-batch size (--batch-size).
                init_lr: Initial learning rate (--init-lr).
                max_lr: Peak learning rate (--max-lr).
                final_lr: Final learning rate (--final-lr).
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type      = pred_type
            self.model_dir      = model_dir
            self.pretrain_dir   = pretrain_dir or Path(tempfile.gettempdir()) / "chemeleon_pretrain_model"
            self.freeze_encoder = freeze_encoder
            self.epochs         = epochs
            self.pretrain_epochs = pretrain_epochs
            self.dropout        = dropout
            self.ffn_hidden_dim = ffn_hidden_dim
            self.ffn_num_layers = ffn_num_layers
            self.batch_size     = batch_size
            self.init_lr        = init_lr
            self.max_lr         = max_lr
            self.final_lr       = final_lr
            self.target_col: Optional[str] = None

        # ── internal helpers ────────────────────────────────────────────────

        def _base_train_args(self, task_type: str, target_col: str) -> list[str]:
            """Return CLI args shared between pretrain() and train()."""
            return [
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _get_device(),
                "--dropout", str(self.dropout),
                "--ffn-hidden-dim", str(self.ffn_hidden_dim),
                "--ffn-num-layers", str(self.ffn_num_layers),
                "--batch-size", str(self.batch_size),
                "--init-lr", str(self.init_lr),
                "--max-lr", str(self.max_lr),
                "--final-lr", str(self.final_lr),
            ]

        # ── public API ──────────────────────────────────────────────────────

        def pretrain(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "pretrain_target",
        ) -> None:
            """
            Pretrain on an auxiliary dataset, starting from the CheMeleon backbone.

            Runs `chemprop train --from-foundation CHEMELEON` on the auxiliary
            data and saves the checkpoint to pretrain_dir.  The resulting
            checkpoint encodes both CheMeleon priors and auxiliary-task
            knowledge, and is used to warm-start subsequent train() calls.

            Args:
                X_train: SMILES strings for the auxiliary training set.
                y_train: Auxiliary training targets, shape (n,) or (n, 1).
                X_val:   SMILES strings for auxiliary validation (early stopping).
                y_val:   Auxiliary validation targets.
                target_col: Column name used in the temporary CSV files.
            """
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "chemeleon_pretrain_train.csv"
            val_csv   = tmp / "chemeleon_pretrain_val.csv"

            _write_smiles_csv(X_train, y_train, train_csv, target_col)
            _write_smiles_csv(X_val,   y_val,   val_csv,   target_col)

            if self.pretrain_dir.exists():
                shutil.rmtree(self.pretrain_dir)

            task_type = "regression" if self.pred_type == "regression" else "classification"
            _run_chemprop_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.pretrain_epochs),
                "--from-foundation", "CHEMELEON",
                "--save-dir", str(self.pretrain_dir),
            ])

            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "target",
        ) -> None:
            """
            Fine-tune on the target dataset.

            When a pretrain checkpoint exists at pretrain_dir/model_0/best.pt,
            the encoder weights are loaded via `--checkpoint`, giving a
            three-stage chain: CheMeleon → pretrain → fine-tune.
            Without a pretrain checkpoint the model falls back to
            `--from-foundation CHEMELEON` (standard two-stage fine-tuning).

            Args:
                X_train: SMILES strings for training.
                y_train: Training targets, shape (n,) or (n, 1).
                X_val:   SMILES strings for validation (early stopping).
                y_val:   Validation targets, shape (n,) or (n, 1).
                target_col: Column name used in the temporary CSV files.
            """
            self.target_col = target_col
            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "chemeleon_train.csv"
            val_csv   = tmp / "chemeleon_val.csv"

            _write_smiles_csv(X_train, y_train, train_csv, target_col)
            _write_smiles_csv(X_val,   y_val,   val_csv,   target_col)

            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)

            task_type = "regression" if self.pred_type == "regression" else "classification"
            args = [
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                *self._base_train_args(task_type, target_col),
                "--epochs", str(self.epochs),
                "--save-dir", str(self.model_dir),
            ]

            # Three-stage: load pretrain checkpoint when available
            pretrain_ckpt = self.pretrain_dir / "model_0" / "best.pt"
            if pretrain_ckpt.exists():
                args += ["--checkpoint", str(pretrain_ckpt)]
                if self.freeze_encoder:
                    args.append("--freeze-encoder")
            else:
                # Fall back to standard two-stage CheMeleon fine-tuning
                args += ["--from-foundation", "CHEMELEON"]

            _run_chemprop_cli(args)

            train_csv.unlink(missing_ok=True)
            val_csv.unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            """
            Run inference by calling `chemprop predict` via subprocess.

            Args:
                X_test: SMILES strings to predict.

            Returns:
                1-D numpy array of predicted values.
            """
            tmp = Path(tempfile.gettempdir())
            test_csv = tmp  / "chemeleon_test.csv"
            pred_csv = tmp  / "chemeleon_preds.csv"
            model_pt = self.model_dir / "model_0" / "best.pt"

            _write_smiles_csv(X_test, None, test_csv, self.target_col)

            _run_chemprop_cli([
                "predict",
                "--test-path",  str(test_csv),
                "--model-path", str(model_pt),
                "--preds-path", str(pred_csv),
            ])

            preds = pl.read_csv(pred_csv)[self.target_col].to_numpy()

            test_csv.unlink(missing_ok=True)
            pred_csv.unlink(missing_ok=True)

            return preds.flatten()

    return ChempropChemeleonModel, ChempropModel


@app.cell
def _(Path, json, np, subprocess, sys, tempfile):
    # ── ChempropAPIModel — Python-API subprocess variant ─────────────────────
    # Identical interface to ChempropModel but runs chemprop via its Python API
    # (Lightning Trainer) instead of the CLI.  The key benefit is explicit MPS
    # memory management: the subprocess calls torch.mps.empty_cache() and
    # torch.mps.synchronize() before exit, returning a clean Metal pool to macOS
    # immediately rather than waiting for the background Metal GC.  This
    # eliminates the multi-minute inter-fold stalls seen with the CLI approach.
    #
    # Two scripts are written to /tmp at cell init and reused across all calls:
    #   _TRAIN_SCRIPT  — train on (X_train_smiles, y_train, X_val_smiles, y_val),
    #                    save best checkpoint, write predictions on X_test.
    #   _PREDICT_SCRIPT — load checkpoint, write predictions on X_test.
    # Data is exchanged via JSON (SMILES lists) and .npy files (targets / preds).

    _TRAIN_SCRIPT_LINES = """
    import os, sys, json, numpy as np, tempfile, torch
    from datetime import datetime, timezone
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    _API_LOG = os.path.join("logs", "chemprop_api.log")
    os.makedirs("logs", exist_ok=True)
    def _log(msg):
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        line = f"{ts} - {msg}\\n"
        with open(_API_LOG, "a") as _f:
            _f.write(line)

    _log("train START")
    import lightning as L
    from lightning.pytorch.callbacks import ModelCheckpoint
    from lightning.pytorch.callbacks.early_stopping import EarlyStopping
    from chemprop import data, models, nn as chemnn
    from chemprop.data import build_dataloader

    args      = json.loads(sys.argv[1])
    smi_tr    = args["smi_tr"]
    smi_val   = args["smi_val"]
    smi_test  = args["smi_test"]
    y_tr      = np.load(args["y_tr"])
    y_val     = np.load(args["y_val"])
    out_preds = args["out_preds"]
    ckpt_path = args["ckpt_path"]
    params    = args["params"]

    def build_ds(smiles, targets=None):
        if targets is not None:
            return data.MoleculeDataset([
                data.MoleculeDatapoint.from_smi(s, [float(v)])
                for s, v in zip(smiles, targets)
            ])
        return data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smiles])

    ds_tr   = build_ds(smi_tr,   y_tr)
    ds_val  = build_ds(smi_val,  y_val)
    ds_test = build_ds(smi_test)
    batch_size = params.get("batch_size", 64)
    dl_tr   = build_dataloader(ds_tr,   shuffle=True,  num_workers=0, batch_size=batch_size)
    dl_val  = build_dataloader(ds_val,  shuffle=False, num_workers=0, batch_size=batch_size)
    dl_test = build_dataloader(ds_test, shuffle=False, num_workers=0, batch_size=batch_size)

    pretrain_ckpt = params.get("pretrain_ckpt")
    if pretrain_ckpt and os.path.exists(pretrain_ckpt):
        model = models.MPNN.load_from_file(pretrain_ckpt, map_location="cpu")
        model.init_lr  = params.get("init_lr",  1e-4)
        model.max_lr   = params.get("max_lr",   1e-3)
        model.final_lr = params.get("final_lr", 1e-4)
        model.apply(lambda m: setattr(m, "p", params.get("dropout", 0.0))
                    if isinstance(m, torch.nn.Dropout) else None)
    else:
        mp  = chemnn.BondMessagePassing(
            d_h=params["message_hidden_dim"], depth=params["depth"],
            dropout=params["dropout"],
        )
        ffn = chemnn.RegressionFFN(
            input_dim=mp.output_dim,
            n_layers=params["ffn_num_layers"],
            hidden_dim=params["ffn_hidden_dim"],
        )
        model = models.MPNN(
            mp, chemnn.MeanAggregation(), ffn,
            init_lr=params.get("init_lr", 1e-4),
            max_lr=params.get("max_lr", 1e-3),
            final_lr=params.get("final_lr", 1e-4),
        )

    ckpt_dir = os.path.dirname(ckpt_path)
    os.makedirs(ckpt_dir, exist_ok=True)
    # Strip any extension so ModelCheckpoint saves as "best.ckpt" (it appends .ckpt itself)
    ckpt_stem = os.path.splitext(os.path.basename(ckpt_path))[0]
    ckpt_cb = ModelCheckpoint(dirpath=ckpt_dir,
                              filename=ckpt_stem,
                              monitor="val_loss", save_top_k=1, mode="min")
    es_cb   = EarlyStopping(monitor="val_loss", patience=10, mode="min")

    trainer = L.Trainer(
        max_epochs=params["epochs"],
        enable_progress_bar=False, enable_model_summary=False, logger=False,
        accelerator="auto",
        callbacks=[ckpt_cb, es_cb],
    )
    trainer.fit(model, dl_tr, dl_val)
    best = models.MPNN.load_from_file(ckpt_cb.best_model_path, map_location="cpu")
    best.eval()
    preds = trainer.predict(best, dl_test)
    np.save(out_preds, np.concatenate([p.numpy() for p in preds]).flatten())

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    _log("train END (MPS cache flushed)")
    """.strip()

    _PREDICT_SCRIPT_LINES = """
    import os, sys, json, tempfile, numpy as np, torch
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    _API_LOG = os.path.join("logs", "chemprop_api.log")
    os.makedirs("logs", exist_ok=True)
    def _log(msg):
        from datetime import datetime, timezone
        ts = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
        with open(_API_LOG, "a") as _f: _f.write(f"{ts} - {msg}\\n")

    _log("predict START")
    import lightning as L
    from chemprop import data, models
    from chemprop.data import build_dataloader

    args      = json.loads(sys.argv[1])
    smi_test  = args["smi_test"]
    ckpt_path = args["ckpt_path"]
    out_preds = args["out_preds"]

    ds_test = data.MoleculeDataset([data.MoleculeDatapoint.from_smi(s) for s in smi_test])
    dl_test = build_dataloader(ds_test, shuffle=False, num_workers=0)
    model   = models.MPNN.load_from_file(ckpt_path, map_location="cpu")
    model.eval()

    trainer = L.Trainer(enable_progress_bar=False, enable_model_summary=False,
                    logger=False, accelerator="auto")
    preds = trainer.predict(model, dl_test)
    np.save(out_preds, np.concatenate([p.numpy() for p in preds]).flatten())

    if torch.backends.mps.is_available():
        torch.mps.synchronize()
        torch.mps.empty_cache()
    _log("predict END (MPS cache flushed)")
    """.strip()

    import textwrap as _tw
    _API_TRAIN_SCRIPT   = Path(tempfile.gettempdir()) / "chemprop_api_train.py"
    _API_PREDICT_SCRIPT = Path(tempfile.gettempdir()) / "chemprop_api_predict.py"
    _API_TRAIN_SCRIPT.write_text(_tw.dedent(_TRAIN_SCRIPT_LINES))
    _API_PREDICT_SCRIPT.write_text(_tw.dedent(_PREDICT_SCRIPT_LINES))

    _API_MODEL_DIR     = Path(tempfile.gettempdir()) / "chemprop_api_model"
    _API_PRETRAIN_DIR  = Path(tempfile.gettempdir()) / "chemprop_api_pretrain_model"

    def _run_api_script(script: Path, args_dict: dict) -> None:
        """Run a chemprop API subprocess, raising RuntimeError on failure."""
        result = subprocess.run(
            [sys.executable, str(script), json.dumps(args_dict)],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"ChempropAPIModel subprocess failed:\n{result.stderr[-2000:]}"
            )

    class ChempropAPIModel:
        """
        Chemprop D-MPNN trained via the Python API in an isolated subprocess.

        Identical interface to ChempropModel (train / predict / pretrain) but
        uses the Lightning Trainer directly instead of the CLI.  The subprocess
        calls torch.mps.empty_cache() + torch.mps.synchronize() before exit,
        returning the Metal pool to macOS immediately and eliminating the
        multi-minute inter-fold MPS memory stalls observed with the CLI approach.

        Results should be numerically equivalent to ChempropModel for the same
        hyperparameters and seed — use both classes in parallel to verify
        reproducibility before switching production runs to this class.
        """

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _API_MODEL_DIR,
            pretrain_dir: Path | None = None,
            freeze_encoder: bool = False,
            epochs: int = 50,
            pretrain_epochs: int = 50,
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
            self.pred_type          = pred_type
            self.model_dir          = model_dir
            self.pretrain_dir       = pretrain_dir or _API_PRETRAIN_DIR
            self.freeze_encoder     = freeze_encoder
            self.epochs             = epochs
            self.pretrain_epochs    = pretrain_epochs
            self.message_hidden_dim = message_hidden_dim
            self.depth              = depth
            self.dropout            = dropout
            self.ffn_hidden_dim     = ffn_hidden_dim
            self.ffn_num_layers     = ffn_num_layers
            self.batch_size         = batch_size
            self.init_lr            = init_lr
            self.max_lr             = max_lr
            self.final_lr           = final_lr
            self.target_col: str | None = None

        def _params(self, epochs: int) -> dict:
            """Serialisable params dict passed to subprocess scripts."""
            return {
                "message_hidden_dim": self.message_hidden_dim,
                "depth":              self.depth,
                "dropout":            self.dropout,
                "ffn_hidden_dim":     self.ffn_hidden_dim,
                "ffn_num_layers":     self.ffn_num_layers,
                "batch_size":         self.batch_size,
                "init_lr":            self.init_lr,
                "max_lr":             self.max_lr,
                "final_lr":           self.final_lr,
                "epochs":             epochs,
                "pretrain_ckpt":      str(self.pretrain_dir / "best.ckpt"),
            }

        def pretrain(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "pretrain_target",
        ) -> None:
            """
            Pretrain on an auxiliary dataset via the Python API subprocess.

            Saves the best checkpoint to pretrain_dir/best.pt.
            """
            import shutil
            if self.pretrain_dir.exists():
                shutil.rmtree(self.pretrain_dir)
            self.pretrain_dir.mkdir(parents=True, exist_ok=True)

            tmp = Path(tempfile.gettempdir())
            y_tr_f  = tmp / "api_pt_ytr.npy"
            y_val_f = tmp / "api_pt_yval.npy"
            np.save(str(y_tr_f),  y_train)
            np.save(str(y_val_f), y_val)

            params = {**self._params(self.pretrain_epochs), "pretrain_ckpt": ""}
            _run_api_script(_API_TRAIN_SCRIPT, {
                "smi_tr":    X_train,
                "smi_val":   X_val,
                "smi_test":  X_val,          # dummy — preds not used
                "y_tr":      str(y_tr_f),
                "y_val":     str(y_val_f),
                "out_preds": str(tmp / "api_pt_dummy_preds"),
                "ckpt_path": str(self.pretrain_dir / "best.pt"),
                "params":    params,
            })
            for f in [y_tr_f, y_val_f, tmp / "api_pt_dummy_preds.npy"]:
                Path(f).unlink(missing_ok=True)

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "target",
        ) -> None:
            """
            Fine-tune (or train from scratch) via the Python API subprocess.

            If pretrain_dir/best.pt exists, encoder weights are loaded from it.
            Saves the best checkpoint to model_dir/best.pt.
            """
            import shutil
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)
            self.model_dir.mkdir(parents=True, exist_ok=True)
            self.target_col = target_col

            tmp = Path(tempfile.gettempdir())
            y_tr_f  = tmp / "api_ytr.npy"
            y_val_f = tmp / "api_yval.npy"
            out_f   = tmp / "api_train_preds"
            np.save(str(y_tr_f),  y_train)
            np.save(str(y_val_f), y_val)

            _run_api_script(_API_TRAIN_SCRIPT, {
                "smi_tr":    X_train,
                "smi_val":   X_val,
                "smi_test":  X_val,          # dummy — preds not used
                "y_tr":      str(y_tr_f),
                "y_val":     str(y_val_f),
                "out_preds": str(out_f),
                "ckpt_path": str(self.model_dir / "best.ckpt"),
                "params":    self._params(self.epochs),
            })
            for f in [y_tr_f, y_val_f, Path(str(out_f) + ".npy")]:
                Path(f).unlink(missing_ok=True)

        def predict(self, X_test: list[str]) -> np.ndarray:
            """
            Run inference via the Python API subprocess.

            Args:
                X_test: SMILES strings to predict.

            Returns:
                1-D numpy array of predicted values.
            """
            tmp   = Path(tempfile.gettempdir())
            out_f = tmp / "api_preds"
            _run_api_script(_API_PREDICT_SCRIPT, {
                "smi_test":  X_test,
                "ckpt_path": str(self.model_dir / "best.ckpt"),
                "out_preds": str(out_f),
            })
            preds = np.load(str(out_f) + ".npy")
            Path(str(out_f) + ".npy").unlink(missing_ok=True)
            return preds.flatten()

    return (ChempropAPIModel,)


@app.cell
def _(mo):
    mo.md(r"""
    ## CLI vs API reproducibility comparison

    Runs both `ChempropModel` (CLI) and `ChempropAPIModel` (Python API subprocess)
    on the same 5×5 CV splits with identical default hyperparameters.  Because
    both use MPS with non-deterministic GPU ops, predictions will not be
    bit-identical, but the Spearman ρ and MAE should be statistically
    indistinguishable across folds.

    Results are saved to `predictions/4_api_vs_cli_comparison.csv.gz`.
    """)
    return


@app.cell
def _(
    ChempropAPIModel,
    ChempropModel,
    Path,
    calc_regression_metrics,
    gc,
    generate_cv_splits_random,
    gzip,
    mo,
    pl,
    pretrain_dr_train,
    rm_tukey_hsd,
    tqdm,
):
    _TARGET_COL    = "pEC50_dr"
    _COMP_PATH_GZ  = Path("../predictions/4_api_vs_cli_comparison.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1

    _EXPECTED_COMP = {"api", "cli"}
    _is_comp_complete = (
        _COMP_PATH_GZ.exists()
        and set(pl.read_csv(_COMP_PATH_GZ)["model"].unique().to_list()) >= _EXPECTED_COMP
    )

    if _is_comp_complete:
        print(f"Comparison results found — loading.")
        _comp_df = pl.read_csv(_COMP_PATH_GZ)
    else:
        _all_records = []
        _n_folds = _N_OUTER * _N_INNER

        for _model_key, _ModelClass in [("api", ChempropAPIModel), ("cli", ChempropModel)]:
            _pbar = tqdm(
                generate_cv_splits_random(
                    pretrain_dr_train, n_outer=_N_OUTER, n_inner=_N_INNER,
                    seed=_SEED, p_val=_P_VAL,
                ),
                total=_n_folds, desc=f"CV {_model_key}", unit="fold",
            )
            for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
                _m = _ModelClass(pred_type="regression")
                _m.train(
                    _train_raw["smiles"].to_list(), _train_raw[_TARGET_COL].to_numpy(),
                    _val_raw["smiles"].to_list(),   _val_raw[_TARGET_COL].to_numpy(),
                    target_col=_TARGET_COL,
                )
                _preds = _m.predict(_test_raw["smiles"].to_list())
                del _m
                gc.collect()

                for _ik, _mn, _smi, _yt, _yp in zip(
                    _test_raw["inchikey"].to_list(),
                    _test_raw["molecule_names"].to_list(),
                    _test_raw["smiles"].to_list(),
                    _test_raw[_TARGET_COL].to_numpy().tolist(),
                    _preds.tolist(),
                ):
                    _all_records.append({
                        "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                        "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                        "model": _model_key, "method": _model_key,
                        "y_true": _yt, "y_pred": _yp,
                    })

        _comp_df = pl.DataFrame(_all_records)
        _COMP_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_COMP_PATH_GZ, "wb") as _f:
            _comp_df.write_csv(_f)
        print(f"Saved → {_COMP_PATH_GZ}")

    # ── Metrics and Tukey HSD ─────────────────────────────────────────────────
    _metrics_comp = calc_regression_metrics(
        _comp_df.rename({"fold": "cv_cycle"})
                .with_columns(pl.lit("random").alias("split")),
        cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0,
    )

    _summary_comp = (
        _metrics_comp.group_by("method")
        .agg(pl.col(["mae", "rho", "r2"]).mean())
        .sort("mae")
    )

    _result_tab, _df_means, _, _ = rm_tukey_hsd(
        _metrics_comp, "mae", group_col="method",
        sort=True, direction_dict={"mae": "minimize"},
    )
    _sig = _result_tab[_result_tab["p-adj"] < 0.05]

    mo.vstack([
        mo.md("### CLI vs API — mean CV metrics"),
        mo.plain_text(_summary_comp.to_pandas().to_string(index=False)),
        mo.md("### Tukey HSD on MAE — expecting **no significant difference**"),
        mo.plain_text(
            _sig[["group1", "group2", "meandiff", "p-adj"]].to_string()
            if len(_sig) > 0 else "  No significant difference (p > 0.05) ✓"
        ),
    ])
    return


@app.cell
def _(
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    mean_absolute_error,
    mean_squared_error,
    np,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
    spearmanr,
    warnings,
):
    _classification_metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "mcc": matthews_corrcoef,
    }

    def _safe_spearmanr(y_true, y_pred):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return spearmanr(y_true, y_pred).correlation

    _regression_metrics = {
        "r2": r2_score,
        "rho": lambda y_true, y_pred: _safe_spearmanr(y_true, y_pred),
        "mse": mean_squared_error,
        "rmse": lambda y_true, y_pred: np.sqrt(mean_squared_error(y_true, y_pred)),
        "mae": mean_absolute_error,
    }

    def evaluate_predictions(
        y_pred: np.ndarray,
        y_test: np.ndarray,
        pred_type: str,
        thr: float = 0.5,
    ) -> dict[str, float]:
        """
        Compute a standard set of metrics for either classification or regression.

        Args:
            y_pred: Model predictions. For classification, these should be
                probability scores (0–1); for regression, continuous values.
            y_test: Ground-truth labels or values.
            pred_type: "classification" or "regression".
            thr: Decision threshold applied to y_pred for binary classification
                metrics. Ignored for regression.

        Returns:
            Dictionary mapping metric names to their computed values.
            Classification metrics: accuracy, balanced_accuracy, precision, recall,
            f1, mcc, roc_auc. Regression metrics: r2, rho, mse, rmse, mae.
        """
        if pred_type == "classification":
            out = {
                metric: _classification_metrics[metric](y_test, y_pred > thr)
                for metric in _classification_metrics
            }
            out["roc_auc"] = roc_auc_score(y_test, y_pred)
            return out
        else:
            return {
                metric: _regression_metrics[metric](y_test, y_pred)
                for metric in _regression_metrics
            }

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
        Generate molecular fingerprints and add them as a new column to the DataFrame.

        Dispatches to the appropriate fingerprint class based on fingerprint_type.
        All skfp types use their standard constructor / transform pipeline.
        For fingerprint types that require 3D conformers (e.g., E3FP), conformers are
        generated automatically via RDKit ETKDGv3.

        CheMeleon embeddings are not handled here — use the `chemeleon_embed`
        function which runs inference in an isolated subprocess.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of: "ecfp"/"morgan", "maccs", "torsion",
                "rdkit", "atompair", "avalon", "e3fp", "mordred", "mqn", "pubchem".
            **kwargs: Additional keyword arguments forwarded to the fingerprint
                class constructor (e.g., radius=3, n_bits=1024 for ECFP).

        Returns:
            DataFrame with an added column named after fingerprint_type.

        Raises:
            ValueError: If fingerprint_type is not a recognized key.
        """
        if fingerprint_type not in _fp_dict:
            raise ValueError(
                f"Fingerprint type not recognized: {fingerprint_type!r}. "
                f"Valid values: {list(_fp_dict.keys())}"
            )

        smiles_list = df.get_column("smiles").to_list()

        # All fingerprint types follow the skfp constructor / transform pattern.
        fp_func = _fp_dict[fingerprint_type](**kwargs)

        if fp_func.requires_conformers:
            mol_from_smiles = MolFromSmilesTransformer()
            conf_gen = ConformerGenerator()
            mols_list = mol_from_smiles.transform(smiles_list)
            mols_list = conf_gen.transform(mols_list)
        else:
            mols_list = smiles_list

        fps = fp_func.transform(mols_list)
        return df.with_columns(pl.Series(values=fps, name=fingerprint_type))

    return (generate_fingerprint,)


@app.cell
def _(BaseKFold, Iterator, Optional, np, pl):
    # ── helpers ────────────────────────────────────────────────────────────────

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

    # ── GroupKFoldShuffle ───────────────────────────────────────────────────────

    class GroupKFoldShuffle(BaseKFold):
        """
        K-fold cross-validator that respects group boundaries and supports shuffling.

        An extension of scikit-learn's GroupKFold that adds optional shuffling of
        groups before splitting. Useful for scaffold-aware cross-validation where
        you want reproducible but shuffled group assignments.

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
            # Collect unique groups, then optionally shuffle them so that fold
            # assignment is randomised while still keeping each group intact.
            unique_groups = np.unique(groups)

            if self.shuffle:
                rng = np.random.default_rng(self.random_state)
                unique_groups = rng.permutation(unique_groups)

            # Distribute groups as evenly as possible across folds.
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
        Generate nested 5×5 CV splits using a **random** molecule assignment.

        Each molecule is treated as its own group, so folds are purely random.
        This is the baseline split strategy: it gives optimistic estimates of
        generalisation because train and test scaffolds can overlap.

        Args:
            df: Polars DataFrame to split.
            n_outer: Number of outer CV folds.
            n_inner: Number of inner CV folds per outer iteration.
            seed: Random seed for GroupKFoldShuffle.
            p_val: Fraction of the training set reserved as a validation split.
                0 disables the validation split (val_df is yielded as None).

        Yields:
            Tuples of (fold_index, outer_index, inner_index, train_df, val_df, test_df).
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
def _(mo):
    mo.md(r"""
    ## ML comparison code

    Adapted from https://github.com/polaris-hub/polaris-method-comparison
    """)
    return


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
        Calculate regression metrics (MAE, MSE, R2, rho, prec, recall) for each method and split.

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
        # Drop rows where predictions are NaN (e.g. Macau numerical instability)
        # before any metric computation to avoid downstream sklearn errors.
        df_in = df.filter(pl.col(pred_col).is_not_nan() & pl.col(pred_col).is_not_null())
        # Derive binary class columns from the continuous threshold
        df_in = df_in.with_columns([
            (pl.col(val_col) > thresh).alias("true_class"),
            (pl.col(pred_col) > thresh).alias("pred_class"),
        ])

        # Ensure the threshold actually produces two distinct classes
        assert df_in["true_class"].n_unique() == 2, "Binary classification requires two classes"

        metric_list: list[dict] = []

        # Iterate over each (cycle, method, split) group and compute metrics
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
                "rho": float(rho),
                "prec": precision_score(y_true_cls, y_pred_cls),
                "recall": recall_score(y_true_cls, y_pred_cls),
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

        Internally converts to pandas for pingouin/statsmodels compatibility.
        All returned DataFrames are pandas objects for downstream seaborn plotting.

        Args:
            df: Polars DataFrame with columns [cv_cycle, group_col, metric].
            metric: Column name of the metric to test.
            group_col: Column name indicating the comparison groups.
            alpha: Significance level for the test.
            sort: Whether to sort groups by their mean metric value.
            direction_dict: Maps metric names to "maximize" or "minimize" for sort direction.

        Returns:
            Tuple of (result_tab, df_means, df_means_diff, pc) — all pandas DataFrames.
            - result_tab: Pairwise comparisons with adjusted p-values.
            - df_means: Mean values per group.
            - df_means_diff: Matrix of pairwise mean differences.
            - pc: Matrix of adjusted p-values.
        """
        # Convert to pandas — pingouin and statsmodels require it
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

        # Calculate pairwise mean differences and adjusted p-values
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
def _(Optional, Path, math, np, plt, rm_tukey_hsd, sns):
    def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
                 ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
                 show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
        """
        Multiple comparison of means heatmap (Tukey HSD).

        Args:
            pc: DataFrame of adjusted p-values (groups × groups).
            effect_size: DataFrame of pairwise mean differences.
            means: Series of mean metric value per group.
            labels: Show axis labels with mean values.
            cmap: Colormap name (default "coolwarm").
            ax: Existing Axes to draw on.
            show_diff: Annotate cells with mean difference values.
            cell_text_size: Font size for cell annotations.
            axis_text_size: Font size for axis tick labels.
            show_cbar: Show colorbar.
            reverse_cmap: Reverse the colormap (use for metrics to minimise).
            vlim: Symmetric colour scale limit (colours span −2×vlim to +2×vlim).

        Returns:
            The Axes with the heatmap.
        """
        for key in ['cbar', 'vmin', 'vmax', 'center']:
            kwargs.pop(key, None)

        if not cmap:
            cmap = "coolwarm"
        if reverse_cmap:
            cmap = cmap + "_r"

        significance = pc.copy().astype(object)
        significance[(pc < 0.001) & (pc >= 0)] = '***'
        significance[(pc < 0.01)  & (pc >= 0.001)] = '**'
        significance[(pc < 0.05)  & (pc >= 0.01)]  = '*'
        significance[(pc >= 0.05)] = ''
        np.fill_diagonal(significance.values, '')

        annotations = effect_size.round(2).astype(str) + significance if show_diff else significance

        hax = sns.heatmap(
            effect_size, cmap=cmap, annot=annotations, fmt='', cbar=show_cbar, ax=ax,
            annot_kws={"size": cell_text_size},
            vmin=-2 * vlim if vlim else None,
            vmax= 2 * vlim if vlim else None,
            **kwargs,
        )
        if labels:
            label_list = list(means.index)
            hax.set_xticklabels(
                [x + f'\n{means.loc[x]:.2f}' for x in label_list],
                size=axis_text_size, ha='center', va='top', rotation=0,
            )
            hax.set_yticklabels(
                [x + f'\n{means.loc[x]:.2f}\n' for x in label_list],
                size=axis_text_size, ha='center', va='center', rotation=90,
            )
        hax.set_xlabel('')
        hax.set_ylabel('')
        return hax

    def make_mcs_plot_grid(
        df,
        stats: list[str],
        group_col: str,
        alpha: float = 0.05,
        figsize: tuple = (20, 10),
        direction_dict: dict | None = None,
        effect_dict: dict | None = None,
        show_diff: bool = True,
        cell_text_size: int = 16,
        axis_text_size: int = 12,
        title_text_size: int = 16,
        sort_axes: bool = False,
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Grid of Tukey HSD MCS heatmaps, one panel per metric.

        Args:
            df: Polars DataFrame with [cv_cycle, group_col] and metric columns.
            stats: Metric names to plot.
            group_col: Column that identifies comparison groups.
            alpha: Significance threshold.
            figsize: Figure size.
            direction_dict: Maps metric → "maximize" or "minimize".
            effect_dict: Maps metric → colour-scale half-width.
            show_diff: Show mean differences in cell annotations.
            cell_text_size: Annotation font size.
            axis_text_size: Tick label font size.
            title_text_size: Panel title font size.
            sort_axes: Sort groups by mean metric value.
            save_path: Save figure to this path if provided.

        Returns:
            Matplotlib Figure.
        """
        if direction_dict is None:
            direction_dict = {}
        if effect_dict is None:
            effect_dict = {}

        for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
            direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')
        for key in ['r2', 'rho', 'prec', 'recall']:
            effect_dict.setdefault(key, 0.1)
        effect_dict.setdefault('mae', 0.5)
        effect_dict.setdefault('mse', 1.0)

        ncol = 1 if len(stats) == 1 else (2 if len(stats) == 4 else 3)
        nrow = math.ceil(len(stats) / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

        for i, stat in enumerate(stats):
            stat = stat.lower()
            _, df_means, df_means_diff, pc = rm_tukey_hsd(
                df, stat, group_col, alpha, sort_axes, direction_dict
            )
            mcs_plot(
                pc, effect_size=df_means_diff, means=df_means[stat],
                show_diff=show_diff, ax=ax[i // ncol, i % ncol], cbar=True,
                cell_text_size=cell_text_size, axis_text_size=axis_text_size,
                reverse_cmap=(direction_dict.get(stat) == 'minimize'),
                vlim=effect_dict.get(stat),
            )
            ax[i // ncol, i % ncol].set_title(stat.upper(), fontsize=title_text_size)

        for i in range(len(stats), nrow * ncol):
            ax[i // ncol, i % ncol].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
        return fig

    return make_mcs_plot_grid, mcs_plot


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 1 — RF, XGBoost and Macau on top-3 fingerprints

    Compare **RandomForestModel**, **BoostedTreesModel** (XGBoost) and
    **MacauModel** using the three fingerprints that ranked highest in the RF
    fingerprint sweep from notebook 3:

    | Fingerprint | Key | Dimensionality |
    |---|---|---|
    | Mordred 2D descriptors | `mordred` (no 3D) | 1 613 |
    | MQN counts | `mqn` | 42 |
    | CheMeleon learned embedding | `chemeleon` | 2 048 |

    Same 5×5 CV protocol as notebook 3 (seed=42, p_val=0.1).
    CheMeleon embeddings are generated in an isolated subprocess to avoid the
    OpenMP runtime collision between PyTorch and sklearn.
    """)
    return


@app.cell
def _(gc, pl):
    """Load the dose-response training set (same as all other experiments)."""
    fp_cmp_train = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "molecule_names", "pEC50_dr"])
    )
    gc.collect()
    return (fp_cmp_train,)


@app.cell
def _(
    BoostedTreesModel,
    MacauModel,
    Path,
    RandomForestModel,
    chemeleon_embed,
    extract_fp_matrix,
    fp_cmp_train,
    gc,
    generate_cv_splits_random,
    generate_fingerprint,
    gzip,
    np,
    pl,
    tqdm,
):
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/4_fp_model_comparison_1.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1

    # ── CheMeleon embedding subprocess ───────────────────────────────────────
    # CheMeleon (PyTorch) must run in an isolated subprocess to avoid the
    # OpenMP runtime collision with sklearn/smurff in the parent process.
    # ── model × fingerprint grid ──────────────────────────────────────────────
    # Each entry: (model_key, ModelClass, model_kwargs, fp_key, fp_type, fp_kwargs)
    # MacauModel has no pred_type param; RF and XGBoost do.
    _GRID = [
        ("rf",     RandomForestModel,  {"pred_type": "regression"}, "mordred",  "mordred",  {}),
        ("rf",     RandomForestModel,  {"pred_type": "regression"}, "mqn",      "mqn",      {}),
        ("rf",     RandomForestModel,  {"pred_type": "regression"}, "chemeleon","chemeleon",{}),
        ("xgboost",BoostedTreesModel,  {"pred_type": "regression"}, "mordred",  "mordred",  {}),
        ("xgboost",BoostedTreesModel,  {"pred_type": "regression"}, "mqn",      "mqn",      {}),
        ("xgboost",BoostedTreesModel,  {"pred_type": "regression"}, "chemeleon","chemeleon",{}),
        ("macau",  MacauModel,         {},                          "mordred",  "mordred",  {}),
        ("macau",  MacauModel,         {},                          "mqn",      "mqn",      {}),
        ("macau",  MacauModel,         {},                          "chemeleon","chemeleon",{}),
    ]

    # All 9 method×fingerprint combinations that must be present for the file
    # to be considered complete. A partial checkpoint (from a previous run that
    # crashed mid-way) will be missing some and will be re-run from scratch.
    _EXPECTED_METHODS = {
        f"{mk}_{fk}" for mk, _, _, fk, *_ in _GRID
    }
    _is_complete = (
        _PRED_PATH_GZ.exists()
        and set(pl.read_csv(_PRED_PATH_GZ)["method"].unique().to_list()) >= _EXPECTED_METHODS
    )

    if _is_complete:
        print(f"Complete predictions found at {_PRED_PATH_GZ} — skipping training.")
        fp_cmp_pred_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        if _PRED_PATH_GZ.exists():
            print(f"Partial/incomplete file found — removing.")
            _PRED_PATH_GZ.unlink()
        _n_folds = _N_OUTER * _N_INNER
        _CKPT_PATH = _PRED_PATH_GZ.with_suffix(".ckpt.gz")

        # Resume from checkpoint if one exists (e.g. pass 1 already completed)
        if _CKPT_PATH.exists():
            _all_records = pl.read_csv(_CKPT_PATH).to_dicts()
            _done_methods = {r["method"] for r in _all_records}
            print(f"Resuming from checkpoint: {len(_all_records):,} rows, "
                  f"completed: {sorted(_done_methods)}")
        else:
            _all_records = []
            _done_methods: set[str] = set()

        def _checkpoint(records: list[dict]) -> None:
            if not records:
                return
            _CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(_CKPT_PATH, "wb") as _f:
                pl.DataFrame(records).write_csv(_f)

        # ── pass 1: all models on chemeleon embeddings (subprocess embed) ──────
        # Embeddings computed in an isolated subprocess (CheMeleon loads torch);
        # RF, XGBoost and Macau then run in-process (no torch).
        _chemeleon_models = [(mk, MC, mkw) for mk, MC, mkw, fk, *_ in _GRID
                             if fk == "chemeleon"]
        _chemeleon_todo = [t for t in _chemeleon_models
                           if f"{t[0]}_chemeleon" not in _done_methods]

        if not _chemeleon_todo:
            print("Pass 1 (chemeleon) already complete — skipping.")
        else:
            _pbar_ch = tqdm(
                generate_cv_splits_random(
                    fp_cmp_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL,
                ),
                total=_n_folds, desc="CV chemeleon", unit="fold",
            )
            for _fold, _outer, _inner, _train_raw, _, _test_raw in _pbar_ch:
                _y_train = _train_raw[_TARGET_COL].to_numpy()
                _y_true  = _test_raw[_TARGET_COL].to_numpy()

                _X_train_ch, _X_test_ch = chemeleon_embed(
                    _train_raw["smiles"].to_list(),
                    _test_raw["smiles"].to_list(),
                    prefix="a1",
                )

                for _model_key, _ModelClass, _model_kwargs in _chemeleon_todo:
                    _pbar_ch.set_postfix({"fold": _fold, "model": _model_key, "fp": "chemeleon"}, refresh=False)
                    _m = _ModelClass(**_model_kwargs)
                    if _model_key == "xgboost":
                        _m.train(_X_train_ch, _y_train, _X_test_ch, _y_true)
                    else:
                        _m.train(_X_train_ch, _y_train)
                    _y_pred = _m.predict(_X_test_ch)
                    del _m
                    gc.collect()

                    for _ik, _mn, _smi, _yt, _yp in zip(
                        _test_raw["inchikey"].to_list(),
                        _test_raw["molecule_names"].to_list(),
                        _test_raw["smiles"].to_list(),
                        _y_true.tolist(), _y_pred.tolist(),
                    ):
                        _all_records.append({
                            "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                            "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                            "model": _model_key, "fingerprint": "chemeleon",
                            "method": f"{_model_key}_chemeleon",
                            "y_true": _yt, "y_pred": _yp,
                        })
                del _X_train_ch, _X_test_ch
                gc.collect()
            _checkpoint(_all_records)
            print(f"Pass 1 done — {len(_all_records):,} records so far")

        # ── pass 2: mordred and mqn (all in-process — no torch dependency) ─────
        _non_ch_fps = [("mordred", "mordred", {}), ("mqn", "mqn", {})]
        _seen, _inproc_models = set(), []
        for mk, MC, mkw, fk, *_ in _GRID:
            if fk != "chemeleon" and mk not in _seen:
                _inproc_models.append((mk, MC, mkw))
                _seen.add(mk)

        for _fp_col, _fp_type, _fp_kwargs in _non_ch_fps:
            _inproc_todo = [(mk, MC, mkw) for mk, MC, mkw in _inproc_models
                            if f"{mk}_{_fp_col}" not in _done_methods]

            if not _inproc_todo:
                print(f"Pass 2 ({_fp_col}) already complete — skipping.")
            else:
                _pbar_fp = tqdm(
                    generate_cv_splits_random(
                        fp_cmp_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL,
                    ),
                    total=_n_folds, desc=f"CV {_fp_col}", unit="fold",
                )
                for _fold, _outer, _inner, _train_raw, _, _test_raw in _pbar_fp:
                    _y_train = _train_raw[_TARGET_COL].to_numpy()
                    _y_true  = _test_raw[_TARGET_COL].to_numpy()

                    _train_fp = generate_fingerprint(_train_raw, _fp_type, **_fp_kwargs)
                    _test_fp  = generate_fingerprint(_test_raw,  _fp_type, **_fp_kwargs)
                    _X_train  = extract_fp_matrix(_train_fp, _fp_type)
                    _X_test   = extract_fp_matrix(_test_fp,  _fp_type)
                    del _train_fp, _test_fp

                    # Drop any feature column that contains a NaN in the training
                    # set. Test set uses the same column mask to avoid leakage.
                    if np.isnan(_X_train).any():
                        _valid_cols = ~np.isnan(_X_train).any(axis=0)
                        _X_train = _X_train[:, _valid_cols]
                        _X_test  = _X_test[:, _valid_cols]
                    gc.collect()

                    for _model_key, _ModelClass, _model_kwargs in _inproc_todo:
                        _pbar_fp.set_postfix({"fold": _fold, "model": _model_key, "fp": _fp_col}, refresh=False)
                        _m = _ModelClass(**_model_kwargs)
                        if _model_key == "xgboost":
                            # XGBoost needs a val set for early stopping
                            _m.train(_X_train, _y_train, _X_test, _y_true)
                        else:
                            _m.train(_X_train, _y_train)
                        _y_pred = _m.predict(_X_test)
                        del _m
                        gc.collect()

                        for _ik, _mn, _smi, _yt, _yp in zip(
                            _test_raw["inchikey"].to_list(),
                            _test_raw["molecule_names"].to_list(),
                            _test_raw["smiles"].to_list(),
                            _y_true.tolist(), _y_pred.tolist(),
                        ):
                            _all_records.append({
                                "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                                "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                                "model": _model_key, "fingerprint": _fp_col,
                                "method": f"{_model_key}_{_fp_col}",
                                "y_true": _yt, "y_pred": _yp,
                            })

                    del _X_train, _X_test
                    gc.collect()

            _checkpoint(_all_records)
            print(f"Pass 2 ({_fp_col}) done — {len(_all_records):,} records so far")

        # All passes complete — write final file and clean up checkpoint
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            pl.DataFrame(_all_records).write_csv(_f)
        _CKPT_PATH.unlink(missing_ok=True)
        print(f"All passes done — {len(_all_records):,} records total \u2192 {_PRED_PATH_GZ}")

    fp_cmp_pred_df = pl.read_csv(_PRED_PATH_GZ)
    return (fp_cmp_pred_df,)


@app.cell
def _(
    Path,
    calc_regression_metrics,
    fp_cmp_pred_df,
    make_mcs_plot_grid,
    mo,
    pl,
):
    """
    Summarise the fingerprint × model comparison with a mean metrics table
    and MCS heatmaps (Tukey HSD) for MAE, R² and ρ.
    """
    # Add CheMeleon fine-tuned baseline from notebook 2 for comparison
    _baseline_chemeleon = (
        pl.read_csv("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .filter(pl.col("model") == "chemeleon")
        .rename({"fold": "cv_cycle", "model": "method"})
        .with_columns([
            pl.lit("chemeleon_base").alias("method"),
            pl.lit("random").alias("split"),
        ])
    )

    _combined = pl.concat([
        fp_cmp_pred_df
            .rename({"fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
        _baseline_chemeleon,
    ], how="diagonal")

    _metrics_df = calc_regression_metrics(
        _combined,
        cycle_col="cv_cycle",
        val_col="y_true",
        pred_col="y_pred",
        thresh=4.0,
    )

    _summary = (
        _metrics_df
        .group_by("method")
        .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
        .sort("mae", descending=False)
    )

    _ABBREV = {"xgboost": "xg", "chemeleon_base": "che_b", "chemeleon": "che",
               "mordred": "mor", "chemprop": "cp", "no_pretrain": "bas", "macau": "mc"}
    def _shorten(name: str) -> str:
        for long, short in _ABBREV.items():
            name = name.replace(long, short)
        return name

    _metrics_plot = _metrics_df.with_columns(
        pl.col("method").map_elements(_shorten, return_dtype=pl.String)
    )

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_mcs_plot_grid(
        _metrics_plot,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.1, "mse": 0.2},
        sort_axes=True,
        save_path=_PLOTS_DIR / "analysis1_mcs_mae.png",
    )

    mo.vstack([
        mo.md("## Analysis 1 — RF / XGBoost / Macau × fingerprint (sorted by MAE)"),
        mo.plain_text(_summary.to_pandas().to_string(index=False)),
        mo.md("---"),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Pretraining experiment

    ## Hypothesis
    Pretraining a Chemprop D-MPNN (or CheMeleon fine-tune) on a related but
    distinct biological activity before fine-tuning on the dose-response pEC50
    target may improve generalisation, particularly for the smaller dose-response
    training set.

    ## Pretrain datasets

    | Name | Columns used | Task | Compounds | Notes |
    |---|---|---|---|---|
    | `sd10_reg` | `10.0_log2_fc` | regression | ~10 747 | 10 µM single-dose; includes DR-overlap compounds |
    | `sd30_reg` | `30.0_log2_fc` | regression | ~9 523 | 30 µM single-dose; includes DR-overlap compounds |
    | `sd10_cls` | `10.0_is_hit` | classification | ~10 747 | Same as sd10_reg but predicting hit/non-hit |
    | `sd30_cls` | `30.0_is_hit` | classification | ~9 523 | Same as sd30_reg but predicting hit/non-hit |
    | `counter_ic50` | `pEC50_counter` | regression | 2 646 | All counter compounds are also in DR — mild label-leakage risk |

    Single-dose sets include all available compounds (including those with dose-response
    data), accepting the mild leakage in exchange for a larger and more representative
    pretrain corpus.

    Counter-screen compounds are a strict subset of the dose-response set, so
    their pEC50 values will have been seen during pretraining for some test-fold
    compounds — treat results with caution.

    ## Models tested

    `ChempropModel` and `ChempropChemeleonModel`, each with:

    - **no_pretrain** — baseline (scratch / CheMeleon only)
    - **sd10_reg** — pretrained on 10 µM log2 fold-change (regression)
    - **sd30_reg** — pretrained on 30 µM log2 fold-change (regression)
    - **sd10_cls** — pretrained on 10 µM hit classification
    - **sd30_cls** — pretrained on 30 µM hit classification
    - **counter_ic50** — pretrained on counter-screen pEC50

    ## CV protocol

    Identical to notebook 3: 5×5 random CV, seed=42, p_val=0.1,
    target = `pEC50_dr`.
    """)
    return


@app.cell
def _(gc, pl):
    """
    Load the master activity table and build each pretrain dataset.

    Single-dose sets include all available compounds (including those that also
    have dose-response data), accepting mild label leakage for a larger pretrain
    corpus. Counter-screen compounds are a strict subset of the dose-response set
    and are included with that leakage caveat noted.
    """
    _all = pl.read_csv("../data/processed/all_compounds_activity_data.csv")

    # ── fine-tuning target ──────────────────────────────────────────────────────
    pretrain_dr_train = (
        _all
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "molecule_names", "pEC50_dr"])
    )

    # ── pretrain sets (single-dose regression — all compounds with measurement) ─
    pretrain_sd10_reg = (
        _all
        .filter(pl.col("10.0_log2_fc").is_not_null())
        .select(["smiles", "10.0_log2_fc"])
        .rename({"10.0_log2_fc": "log2_fc"})
    )

    pretrain_sd30_reg = (
        _all
        .filter(pl.col("30.0_log2_fc").is_not_null())
        .select(["smiles", "30.0_log2_fc"])
        .rename({"30.0_log2_fc": "log2_fc"})
    )

    # ── pretrain sets (single-dose classification — hit/non-hit) ───────────────
    # Cast boolean is_hit to integer (0/1) for the Chemprop binary classifier.
    pretrain_sd10_cls = (
        _all
        .filter(pl.col("10.0_is_hit").is_not_null())
        .select(["smiles", "10.0_is_hit"])
        .with_columns(pl.col("10.0_is_hit").cast(pl.Int8).alias("is_hit"))
        .drop("10.0_is_hit")
    )

    pretrain_sd30_cls = (
        _all
        .filter(pl.col("30.0_is_hit").is_not_null())
        .select(["smiles", "30.0_is_hit"])
        .with_columns(pl.col("30.0_is_hit").cast(pl.Int8).alias("is_hit"))
        .drop("30.0_is_hit")
    )

    # ── pretrain set (counter screen — overlaps with DR, leakage caveat) ───────
    pretrain_counter_ic50 = (
        _all
        .filter(pl.col("pEC50_counter").is_not_null())
        .select(["smiles", "pEC50_counter"])
    )

    del _all
    gc.collect()

    print(f"DR train set:            {len(pretrain_dr_train):>6} compounds")
    print(f"Pretrain SD10 reg:       {len(pretrain_sd10_reg):>6} compounds")
    print(f"Pretrain SD30 reg:       {len(pretrain_sd30_reg):>6} compounds")
    print(f"Pretrain SD10 cls:       {len(pretrain_sd10_cls):>6} compounds")
    print(f"Pretrain SD30 cls:       {len(pretrain_sd30_cls):>6} compounds")
    print(f"Pretrain counter IC50:   {len(pretrain_counter_ic50):>6} compounds")
    return (
        pretrain_counter_ic50,
        pretrain_dr_train,
        pretrain_sd10_cls,
        pretrain_sd10_reg,
        pretrain_sd30_cls,
        pretrain_sd30_reg,
    )


@app.cell
def _(
    ChempropModel,
    Path,
    gc,
    generate_cv_splits_random,
    gzip,
    pl,
    pretrain_counter_ic50,
    pretrain_dr_train,
    pretrain_sd10_cls,
    pretrain_sd10_reg,
    pretrain_sd30_cls,
    pretrain_sd30_reg,
    split_dataset_random,
    tqdm,
):
    """
    5×5 CV pretraining experiment.

    For each model class × pretrain strategy:
      1. (Optional) pretrain on the auxiliary dataset using the pretrain task's
         pred_type (regression for log2_fc / pEC50; classification for is_hit).
      2. Fine-tune / train on each CV fold's training split (always regression).
      3. Predict on the held-out test split.

    Seeds, n_outer, n_inner, and p_val match notebook 3 exactly so the
    no_pretrain baselines are directly comparable.
    """
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/4_pretrain_experiment.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1   # fraction of train kept as val for early stopping

    # Mapping: pretrain_name → (pretrain_df, pretrain_target_col, pretrain_pred_type)
    # pretrain_pred_type controls the task used during pretraining only;
    # fine-tuning is always regression on pEC50_dr.
    _PRETRAIN_SETS: dict[str, tuple | None] = {
        "no_pretrain":  None,
        "sd10_reg":     (pretrain_sd10_reg, "log2_fc",       "regression"),
        "sd30_reg":     (pretrain_sd30_reg, "log2_fc",       "regression"),
        "sd10_cls":     (pretrain_sd10_cls, "is_hit",        "classification"),
        "sd30_cls":     (pretrain_sd30_cls, "is_hit",        "classification"),
        "counter_ic50": (pretrain_counter_ic50, "pEC50_counter", "regression"),
    }

    _MODEL_CLASSES = {
        "chemprop": ChempropModel,
    }

    _EXPECTED_PRETRAIN_METHODS = {
        f"{mk}_{pn}" for mk in _MODEL_CLASSES for pn in _PRETRAIN_SETS
    }
    _is_complete = (
        _PRED_PATH_GZ.exists()
        and set(pl.read_csv(_PRED_PATH_GZ)["method"].unique().to_list()) >= _EXPECTED_PRETRAIN_METHODS
    )

    if _is_complete:
        print(f"Complete predictions found at {_PRED_PATH_GZ} — skipping training.")
        pred_df_pretrain = pl.read_csv(_PRED_PATH_GZ)
    else:
        if _PRED_PATH_GZ.exists():
            print("Partial/incomplete file found — removing.")
            _PRED_PATH_GZ.unlink()
        _CKPT_PATH = _PRED_PATH_GZ.with_suffix(".ckpt.gz")

        # Resume from checkpoint if one exists.
        # _done_methods  — methods with all 25 folds complete (skip entirely).
        # _done_folds    — method → set of fold indices already saved (partial resume).
        _n_folds_total = _N_OUTER * _N_INNER
        if _CKPT_PATH.exists():
            _ckpt_df = pl.read_csv(_CKPT_PATH)
            _fold_counts = (
                _ckpt_df.group_by("method")
                .agg(pl.col("fold").n_unique().alias("n_folds"))
            )
            _done_methods = {
                r["method"] for r in _fold_counts.to_dicts()
                if r["n_folds"] >= _n_folds_total
            }
            _done_folds: dict[str, set[int]] = {
                r["method"]: set(
                    _ckpt_df.filter(pl.col("method") == r["method"])["fold"].unique().to_list()
                )
                for r in _fold_counts.to_dicts()
                if r["n_folds"] < _n_folds_total
            }
            # Only keep fully-complete methods in _all_records to avoid duplication
            # when partial strategies re-add their rows via _strategy_records.
            _all_records = (
                _ckpt_df.filter(pl.col("method").is_in(list(_done_methods))).to_dicts()
            )
            _partial = {m: len(f) for m, f in _done_folds.items()}
            print(f"Resuming from checkpoint: {len(_ckpt_df):,} rows total")
            print(f"  Fully complete: {sorted(_done_methods)}")
            print(f"  Partial:        {_partial}")
        else:
            _ckpt_df = pl.DataFrame()
            _all_records = []
            _done_methods: set[str] = set()
            _done_folds: dict[str, set[int]] = {}

        def _checkpoint(records: list[dict]) -> None:
            """Append records to the checkpoint, merging with any existing data."""
            if not records:
                return
            _CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(_CKPT_PATH, "wb") as _f:
                pl.DataFrame(records).write_csv(_f)

        for _model_key, _ModelClass in _MODEL_CLASSES.items():
            for _pretrain_name, _pretrain_spec in _PRETRAIN_SETS.items():
                _method_key = f"{_model_key}_{_pretrain_name}"
                if _method_key in _done_methods:
                    print(f"Skipping {_method_key} — already in checkpoint.")
                    continue

                print(f"\n{'='*60}")
                print(f"Model: {_model_key}  |  Pretrain: {_pretrain_name}")
                print(f"{'='*60}")

                # Seed strategy records from checkpoint for partial resume
                _completed_folds = _done_folds.get(_method_key, set())
                # Load partial folds from per-strategy checkpoint if it exists,
                # otherwise fall back to the main checkpoint.
                _strategy_ckpt = _CKPT_PATH.with_name(
                    _CKPT_PATH.stem.replace(".csv", "") + f"_{_method_key}.csv.gz"
                )
                if _strategy_ckpt.exists():
                    _strategy_records = pl.read_csv(_strategy_ckpt).to_dicts()
                    _completed_folds = {r["fold"] for r in _strategy_records}
                    print(f"  Resuming from per-strategy checkpoint — "
                          f"{len(_completed_folds)}/25 folds done.")
                elif _completed_folds and len(_ckpt_df) > 0:
                    _strategy_records = (
                        _ckpt_df.filter(pl.col("method") == _method_key).to_dicts()
                    )
                else:
                    _strategy_records = []
                if _completed_folds:
                    print(f"  Resuming — {len(_completed_folds)}/25 folds already done.")

                # ── step 1: pretrain once before the CV loop ───────────────────
                # Skip if resuming a partial run — the pretrain checkpoint already
                # exists on disk and train() will find it via pretrain_dir.
                if _pretrain_spec is not None:
                    _pretrain_df, _pretrain_col, _pretrain_type = _pretrain_spec
                    # Use a temporary instance just to resolve the pretrain_dir path
                    _tmp_inst = _ModelClass(pred_type=_pretrain_type)
                    _shared_pretrain_dir = _tmp_inst.pretrain_dir
                    del _tmp_inst

                    if _completed_folds and _shared_pretrain_dir.exists():
                        print(f"Pretrain checkpoint found at {_shared_pretrain_dir} — skipping pretrain.")
                    else:
                        _pt_train, _pt_val = split_dataset_random(
                            _pretrain_df, p_test=0.1, seed=_SEED
                        )
                        _model_inst = _ModelClass(pred_type=_pretrain_type)
                        print(f"Pretraining on {len(_pretrain_df)} compounds "
                              f"({_pretrain_col}, {_pretrain_type})…")
                        _model_inst.pretrain(
                            _pt_train["smiles"].to_list(), _pt_train[_pretrain_col].to_numpy(),
                            _pt_val["smiles"].to_list(),   _pt_val[_pretrain_col].to_numpy(),
                            target_col=_pretrain_col,
                        )
                        _shared_pretrain_dir = _model_inst.pretrain_dir
                        del _model_inst
                        gc.collect()
                else:
                    _shared_pretrain_dir = None

                # ── step 2: 5×5 CV fine-tuning loop (always regression) ────────
                _n_folds = _N_OUTER * _N_INNER
                _remaining_folds = [
                    t for t in generate_cv_splits_random(
                        pretrain_dr_train,
                        n_outer=_N_OUTER, n_inner=_N_INNER,
                        seed=_SEED, p_val=_P_VAL,
                    )
                    if t[0] not in _completed_folds
                ]
                _pbar = tqdm(
                    _remaining_folds,
                    total=_n_folds - len(_completed_folds),
                    desc=f"{_model_key}/{_pretrain_name}",
                    unit="fold",
                )

                for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
                    _smi_train = _train_raw["smiles"].to_list()
                    _smi_val   = _val_raw["smiles"].to_list()
                    _smi_test  = _test_raw["smiles"].to_list()
                    _y_train   = _train_raw[_TARGET_COL].to_numpy()
                    _y_val     = _val_raw[_TARGET_COL].to_numpy()
                    _y_true    = _test_raw[_TARGET_COL].to_numpy()

                    # New instance per fold; fine-tuning is always regression.
                    # Passing pretrain_dir makes train() load the pretrained encoder.
                    if _shared_pretrain_dir is not None:
                        _fold_model = _ModelClass(
                            pred_type="regression",
                            pretrain_dir=_shared_pretrain_dir,
                        )
                    else:
                        _fold_model = _ModelClass(pred_type="regression")

                    _fold_model.train(
                        _smi_train, _y_train,
                        _smi_val,   _y_val,
                        target_col=_TARGET_COL,
                    )
                    _y_pred = _fold_model.predict(_smi_test)

                    del _fold_model

                    for _ik, _mn, _smi, _yt, _yp in zip(
                        _test_raw["inchikey"].to_list(),
                        _test_raw["molecule_names"].to_list(),
                        _smi_test,
                        _y_true.tolist(),
                        _y_pred.tolist(),
                    ):
                        _strategy_records.append({
                            "inchikey":       _ik,
                            "molecule_names": _mn,
                            "smiles":         _smi,
                            "fold":           _fold,
                            "outer_fold":     _outer,
                            "inner_fold":     _inner,
                            "model":          _model_key,
                            "pretrain":       _pretrain_name,
                            "method":         f"{_model_key}_{_pretrain_name}",
                            "y_true":         _yt,
                            "y_pred":         _yp,
                        })

                    # Write only this strategy's rows to a small per-strategy
                    # checkpoint — avoids re-serialising the entire _all_records
                    # list (which grows with each completed strategy) on every fold.
                    _strategy_ckpt = _CKPT_PATH.with_name(
                        _CKPT_PATH.stem.replace(".csv", "") + f"_{_method_key}.csv.gz"
                    )
                    _strategy_ckpt.parent.mkdir(parents=True, exist_ok=True)
                    with gzip.open(_strategy_ckpt, "wb") as _f:
                        pl.DataFrame(_strategy_records).write_csv(_f)

                _all_records.extend(_strategy_records)
                # Merge into the main checkpoint and remove the per-strategy file
                _checkpoint(_all_records)
                _strategy_ckpt = _CKPT_PATH.with_name(
                    _CKPT_PATH.stem.replace(".csv", "") + f"_{_method_key}.csv.gz"
                )
                _strategy_ckpt.unlink(missing_ok=True)
                print(f"  {_method_key} done — {len(_all_records):,} records so far")

        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            pl.DataFrame(_all_records).write_csv(_f)
        _CKPT_PATH.unlink(missing_ok=True)
        print(f"\nSaved {len(_all_records):,} prediction rows → {_PRED_PATH_GZ}")

    pred_df_pretrain = pl.read_csv(_PRED_PATH_GZ)
    return (pred_df_pretrain,)


@app.cell
def _(
    Path,
    calc_regression_metrics,
    make_mcs_plot_grid,
    mo,
    pl,
    pred_df_pretrain,
):
    """
    Compute per-fold metrics and display a summary table + MCS heatmaps.
    """
    _metrics_df = calc_regression_metrics(
        pred_df_pretrain
            .rename({"fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
        cycle_col="cv_cycle",
        val_col="y_true",
        pred_col="y_pred",
        thresh=4.0,
    )

    _summary = (
        _metrics_df
        .group_by("method")
        .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
        .sort("mae", descending=False)
    )

    _ABBREV = {"xgboost": "xg", "chemeleon_base": "che_b", "chemeleon": "che",
               "mordred": "mor", "chemprop": "cp", "no_pretrain": "bas", "macau": "mc"}
    def _shorten(name: str) -> str:
        for long, short in _ABBREV.items():
            name = name.replace(long, short)
        return name

    _metrics_plot = _metrics_df.with_columns(
        pl.col("method").map_elements(_shorten, return_dtype=pl.String)
    )

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_mcs_plot_grid(
        _metrics_plot,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.2, "mse": 0.2},
        sort_axes=True,
        save_path=_PLOTS_DIR / "analysis2_mcs_mae.png",
    )

    mo.vstack([
        mo.md("## Analysis 2 — Chemprop pretraining experiment (sorted by MAE)"),
        mo.plain_text(_summary.to_pandas().to_string(index=False)),
        mo.md("---"),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Chemprop hyperparameter optimisation

    ## Parameter space

    CheMeleon fixes the MPNN backbone (`d_h=2048`, `depth=6`, `dropout=0.0`,
    `activation=relu`), so those are excluded from the HPO on the scratch
    `ChempropModel`.  The free parameters are:

    | Parameter | Role | Default | Sensitivity scan | HPO |
    |---|---|---|---|---|
    | `message_hidden_dim` | MPNN hidden size | 300 | ✓ (timing impact) | ✗ fixed by CheMeleon |
    | `depth` | message-passing steps | 3 | ✓ (timing impact) | ✗ fixed by CheMeleon |
    | `dropout` | regularisation | 0.0 | ✓ | ✓ 0.0 – 0.4 |
    | `ffn_hidden_dim` | FFN hidden size | 300 | ✓ | ✓ 128 – 1024 |
    | `ffn_num_layers` | FFN depth | 2 | ✓ | ✓ 1 – 4 |
    | `max_lr` | peak learning rate | 1e-3 | ✓ | ✓ 1e-4 – 1e-2 |
    | `batch_size` | mini-batch size | 64 | ✓ | ✓ 32, 64, 128 |

    ## Protocol

    **Step 1 — sensitivity scan**: train one model per parameter value on an
    80/20 random split to understand the individual effect of each parameter
    before running the full HPO.

    **Step 2 — Optuna TPE HPO**: 50 trials using 1×5 CV on the full DR dataset,
    persisted to an SQLite database so the study can be resumed across sessions.
    Objective: mean MAE across all 5 folds (predictions cover the full dataset).
    """)
    return


@app.cell
def _(gc, pl, split_dataset_random):
    """Load the DR dataset and create a fixed 80/20 train/test split for HPO."""
    _all_dr = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "pEC50_dr"])
    )
    hpo_train, hpo_test = split_dataset_random(_all_dr, p_test=0.2, seed=42)
    gc.collect()
    print(f"HPO train: {len(hpo_train)}  |  HPO test: {len(hpo_test)}")
    return hpo_test, hpo_train


@app.cell
def _(
    ChempropModel,
    Path,
    gc,
    gzip,
    hpo_test,
    hpo_train,
    mo,
    np,
    pl,
    plt,
    spearmanr,
    warnings,
):
    """
    Parameter sensitivity scan.

    For each parameter, vary it across its candidate values while keeping all
    others at their defaults.  Each configuration is trained once on the 80/20
    split.  Training wall-clock time, MAE and Spearman ρ are recorded.
    Results are saved to predictions/4_hpo_sensitivity.csv.gz.

    message_hidden_dim and depth are included here because they have a large
    impact on training time — important context even though they are excluded
    from the HPO (CheMeleon fixes them).
    """
    import time as _time

    _SENSITIVITY_PATH = Path("../predictions/4_hpo_sensitivity.csv.gz")
    _TARGET_COL = "pEC50_dr"

    _DEFAULTS = dict(
        message_hidden_dim=300,
        depth=3,
        dropout=0.0,
        ffn_hidden_dim=300,
        ffn_num_layers=2,
        max_lr=1e-3,
        batch_size=64,
        epochs=50,
    )

    _PARAM_GRID = {
        "message_hidden_dim": [128, 300, 512, 1024, 2048],
        "depth":              [2, 3, 4, 5, 6],
        "dropout":            [0.0, 0.1, 0.2, 0.3, 0.4],
        "ffn_hidden_dim":     [128, 300, 512, 1024, 2048],
        "ffn_num_layers":     [1, 2, 3, 4],
        "max_lr":             [1e-4, 5e-4, 1e-3, 3e-3, 1e-2],
        "batch_size":         [32, 64, 128],
    }

    if _SENSITIVITY_PATH.exists():
        print(f"Sensitivity results found — loading.")
        _sens_df = pl.read_csv(_SENSITIVITY_PATH)
    else:
        _smi_train = hpo_train["smiles"].to_list()
        _smi_test  = hpo_test["smiles"].to_list()
        _y_train   = hpo_train[_TARGET_COL].to_numpy()
        _y_test    = hpo_test[_TARGET_COL].to_numpy()

        _n_val   = max(1, int(len(_smi_train) * 0.1))
        _smi_val = _smi_train[-_n_val:]
        _y_val   = _y_train[-_n_val:]
        _smi_tr  = _smi_train[:-_n_val]
        _y_tr    = _y_train[:-_n_val]

        _records = []
        _total = sum(len(v) for v in _PARAM_GRID.values())
        _done  = 0

        for _param, _values in _PARAM_GRID.items():
            for _val in _values:
                _kwargs = {**_DEFAULTS, _param: _val}
                _model = ChempropModel(pred_type="regression", **_kwargs)

                _t0 = _time.perf_counter()
                _model.train(_smi_tr, _y_tr, _smi_val, _y_val, target_col=_TARGET_COL)
                _train_sec = _time.perf_counter() - _t0

                _preds = _model.predict(_smi_test)
                del _model
                gc.collect()

                _mae = float(np.abs(_preds - _y_test).mean())
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    _rho = float(spearmanr(_y_test, _preds).correlation)

                _records.append({
                    "param":      _param,
                    "value":      float(_val),
                    "train_sec":  round(_train_sec, 1),
                    "mae":        _mae,
                    "rho":        _rho,
                })
                _done += 1
                print(f"[{_done}/{_total}]  {_param}={_val}  "
                      f"time={_train_sec:.0f}s  MAE={_mae:.3f}  ρ={_rho:.3f}")

        _sens_df = pl.DataFrame(_records)
        _SENSITIVITY_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_SENSITIVITY_PATH, "wb") as _f:
            _sens_df.write_csv(_f)
        print(f"Saved → {_SENSITIVITY_PATH}")

    # ── Plot: one panel per parameter, two lines (time / MAE) ────────────────
    _params = list(_PARAM_GRID.keys())
    _ncols = 2
    _nrows = -(-len(_params) // _ncols)
    _fig_s, _axes = plt.subplots(_nrows, _ncols, figsize=(12, 5 * _nrows))
    _axes = _axes.flatten()

    for _i, _param in enumerate(_params):
        _sub  = _sens_df.filter(pl.col("param") == _param).sort("value").to_pandas()
        _xticks = [str(float(v)) for v in _PARAM_GRID[_param]]
        _xpos   = list(range(len(_xticks)))

        _ax  = _axes[_i]
        _ax2 = _ax.twinx()

        _ax.plot(_xpos,  _sub["train_sec"], marker="D", color="tab:green", label="time (s)")
        _ax2.plot(_xpos, _sub["mae"],       marker="o", color="tab:blue",  label="MAE", linestyle="--")

        _ax.set_xticks(_xpos)
        _ax.set_xticklabels(_xticks, rotation=30, ha="right", fontsize=8)
        _ax.set_title(_param, fontsize=11)
        _ax.set_xlabel("value")
        _ax.set_ylabel("train time (s)", color="tab:green")
        _ax2.set_ylabel("MAE",           color="tab:blue")
        _ax.tick_params(axis="y", labelcolor="tab:green")
        _ax2.tick_params(axis="y", labelcolor="tab:blue")

        # Mark the default value
        _default_str = str(float(_DEFAULTS[_param]))
        if _default_str in _xticks:
            _ax.axvline(_xticks.index(_default_str), color="gray",
                        linestyle=":", alpha=0.5)

    for _j in range(len(_params), len(_axes)):
        _axes[_j].set_visible(False)

    _fig_s.suptitle("Sensitivity scan — training time / MAE per parameter (80/20 split)",
                     fontsize=13)
    _fig_s.tight_layout()

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)
    _fig_s.savefig(_PLOTS_DIR / "sensitivity_scan.png", dpi=150, bbox_inches="tight")

    mo.vstack([
        mo.md("## Sensitivity scan results"),
        mo.plain_text(_sens_df.sort(["param", "value"]).to_pandas().to_string(index=False)),
        mo.as_html(_fig_s),
    ])
    return


@app.cell
def _(
    ChempropModel,
    Path,
    gc,
    generate_cv_splits_random,
    mo,
    np,
    optuna,
    pl,
    pretrain_dr_train,
):
    """
    Optuna TPE hyperparameter optimisation using 1×5 CV.

    Each trial runs 5-fold CV on the full DR dataset, trains on 4 folds with
    10% of that held out for early stopping, and predicts on the 5th.  The
    objective is mean MAE across all 5 folds, giving predictions on the full
    dataset.  The study is persisted to SQLite so it can be resumed.
    """
    _TARGET_COL = "pEC50_dr"
    _DB_PATH    = Path("../predictions/4_hpo_chemprop.db")
    _STUDY_NAME = "chemprop_hpo_1x5cv"
    _N_TRIALS   = 50
    _SEED       = 42
    _P_VAL      = 0.1   # fraction of train held out for early stopping per fold

    def _objective(trial: optuna.Trial) -> float:
        # message_hidden_dim and depth excluded — fixed by CheMeleon backbone.
        params = dict(
            dropout        = trial.suggest_float("dropout", 0.0, 0.4, step=0.05),
            ffn_hidden_dim = trial.suggest_categorical(
                "ffn_hidden_dim", [128, 256, 300, 512, 1024]),
            ffn_num_layers = trial.suggest_int("ffn_num_layers", 1, 4),
            max_lr         = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True),
            batch_size     = trial.suggest_categorical("batch_size", [32, 64, 128]),
            epochs         = 50,
        )
        fold_maes = []
        # n_outer=1 → single pass; n_inner=5 → 5 folds covering full dataset
        for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in \
                generate_cv_splits_random(
                    pretrain_dr_train, n_outer=1, n_inner=5,
                    seed=_SEED, p_val=_P_VAL,
                ):
            _smi_tr  = _train_raw["smiles"].to_list()
            _smi_val = _val_raw["smiles"].to_list()
            _smi_te  = _test_raw["smiles"].to_list()
            _y_tr    = _train_raw[_TARGET_COL].to_numpy()
            _y_val   = _val_raw[_TARGET_COL].to_numpy()
            _y_te    = _test_raw[_TARGET_COL].to_numpy()

            model = ChempropModel(pred_type="regression", **params)
            model.train(_smi_tr, _y_tr, _smi_val, _y_val, target_col=_TARGET_COL)
            preds = model.predict(_smi_te)
            del model
            gc.collect()
            fold_maes.append(float(np.abs(preds - _y_te).mean()))

        return float(np.mean(fold_maes))

    _DB_PATH.parent.mkdir(parents=True, exist_ok=True)

    _study = optuna.create_study(
        study_name=_STUDY_NAME,
        storage=f"sqlite:///{_DB_PATH}",
        load_if_exists=True,
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )

    _completed = len([t for t in _study.trials
                      if t.state == optuna.trial.TrialState.COMPLETE])
    _remaining = max(0, _N_TRIALS - _completed)
    print(f"Study '{_STUDY_NAME}': {_completed} completed, {_remaining} remaining.")

    if _remaining > 0:
        _study.optimize(_objective, n_trials=_remaining, show_progress_bar=True)

    _best = _study.best_trial
    best_params = {**_best.params, "epochs": 50}

    mo.vstack([
        mo.md(f"## Optuna HPO — {_completed + _remaining} trials complete"),
        mo.md(f"**Best trial #{_best.number}  |  1×5 CV MAE = {_best.value:.4f}**"),
        mo.plain_text(
            pl.DataFrame([best_params]).to_pandas().to_string(index=False)
        ),
    ])
    return (best_params,)


@app.cell
def _(
    ChempropChemeleonModel,
    ChempropModel,
    Path,
    best_params,
    calc_regression_metrics,
    gc,
    generate_cv_splits_random,
    gzip,
    make_mcs_plot_grid,
    mo,
    pl,
    pretrain_dr_train,
    rm_tukey_hsd,
    tqdm,
):
    """
    Full 5×5 CV with the HPO-optimised parameters for both ChempropModel and
    ChempropChemeleonModel, saved to predictions/4_hpo_best_5x5cv.csv.gz.
    Compared against base chemprop and chemeleon from the notebook-2 baseline.
    """
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/4_hpo_best_5x5cv.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1

    # CheMeleon only tunes FFN params (encoder is fixed by the backbone)
    _chemeleon_params = {
        k: v for k, v in best_params.items()
        if k not in ("message_hidden_dim", "depth")
    }

    _MODELS = {
        "chemprop_hpo":  (ChempropModel,         best_params),
        "chemeleon_hpo": (ChempropChemeleonModel, _chemeleon_params),
    }

    _EXPECTED = set(_MODELS.keys())
    _is_complete = (
        _PRED_PATH_GZ.exists()
        and set(pl.read_csv(_PRED_PATH_GZ)["model"].unique().to_list()) >= _EXPECTED
    )

    if _is_complete:
        print(f"Results found — loading.")
        _hpo_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        _all_records = []
        for _model_key, (_ModelClass, _params) in _MODELS.items():
            _pbar = tqdm(
                generate_cv_splits_random(
                    pretrain_dr_train, n_outer=_N_OUTER, n_inner=_N_INNER,
                    seed=_SEED, p_val=_P_VAL,
                ),
                total=_N_OUTER * _N_INNER,
                desc=f"CV {_model_key}", unit="fold",
            )
            for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
                _m = _ModelClass(pred_type="regression", **_params)
                _m.train(
                    _train_raw["smiles"].to_list(), _train_raw[_TARGET_COL].to_numpy(),
                    _val_raw["smiles"].to_list(),   _val_raw[_TARGET_COL].to_numpy(),
                    target_col=_TARGET_COL,
                )
                _preds = _m.predict(_test_raw["smiles"].to_list())
                del _m
                gc.collect()

                for _ik, _mn, _smi, _yt, _yp in zip(
                    _test_raw["inchikey"].to_list(),
                    _test_raw["molecule_names"].to_list(),
                    _test_raw["smiles"].to_list(),
                    _test_raw[_TARGET_COL].to_numpy().tolist(),
                    _preds.tolist(),
                ):
                    _all_records.append({
                        "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                        "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                        "model": _model_key, "method": _model_key,
                        "y_true": _yt, "y_pred": _yp,
                    })

        _hpo_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _hpo_df.write_csv(_f)
        print(f"Saved → {_PRED_PATH_GZ}")

    # ── Load baselines from notebook 2 ────────────────────────────────────────
    _baseline = (
        pl.read_csv("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .filter(pl.col("model").is_in(["chemprop", "chemeleon"]))
        .rename({"model": "method"})
    )

    _combined = pl.concat([
        _hpo_df.drop("model"),
        _baseline,
    ], how="diagonal").rename({"fold": "cv_cycle"}).with_columns(
        pl.lit("random").alias("split")
    )

    _metrics = calc_regression_metrics(
        _combined, cycle_col="cv_cycle",
        val_col="y_true", pred_col="y_pred", thresh=4.0,
    )

    _summary = (
        _metrics
        .group_by("method")
        .agg(pl.col(["mae", "rho", "r2"]).mean())
        .sort("mae")
    )

    _result_tab, _df_means, _, _ = rm_tukey_hsd(
        _metrics, "mae", group_col="method",
        sort=True, direction_dict={"mae": "minimize"},
    )
    _sig = _result_tab[_result_tab["p-adj"] < 0.05]

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _fig = make_mcs_plot_grid(
        _metrics,
        stats=["mae"],
        group_col="method",
        figsize=(8, 8),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOTS_DIR / "hpo_mcs_mae.png",
    )

    mo.vstack([
        mo.md("## HPO best params — 5×5 CV vs baselines (sorted by MAE)"),
        mo.plain_text(_summary.to_pandas().to_string(index=False)),
        mo.md("### Tukey HSD on MAE (significant pairs, p < 0.05)"),
        mo.plain_text(
            _sig[["group1", "group2", "meandiff", "p-adj"]].to_string()
            if len(_sig) > 0 else "  No significant differences"
        ),
        mo.md("### MCS grid — MAE"),
        mo.as_html(_fig),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 3 — TabPFN on MQN and Mordred fingerprints

    TabPFN is evaluated on MQN, Mordred and CheMeleon and compared to the
    corresponding RF and XGBoost results from Analysis 1 (`4_fp_model_comparison_1.csv.gz`).

    TabPFN runs on CPU to avoid MPS out-of-memory errors.
    All comparisons use the same 5×5 CV splits (seed=42).
    """)
    return


@app.cell
def _(gc, pl):
    """Load the dose-response training set for analysis 3."""
    tabpfn_train = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("pEC50_dr").is_not_null())
        .select(["smiles", "inchikey", "molecule_names", "pEC50_dr"])
    )
    gc.collect()
    return (tabpfn_train,)


@app.cell
def _(
    Path,
    chemeleon_embed,
    extract_fp_matrix,
    gc,
    generate_cv_splits_random,
    generate_fingerprint,
    gzip,
    np,
    pl,
    subprocess,
    sys,
    tabpfn_train,
    tempfile,
    tqdm,
):
    """
    Run TabPFN on MQN, Mordred and CheMeleon fingerprints using 5×5 CV.

    Predictions are checkpointed fold-by-fold and saved to
    predictions/4_fp_model_comparison_2.csv.gz once both methods are complete.
    TabPFN runs as a subprocess on CPU to avoid MPS out-of-memory errors.
    """
    _TARGET_COL3   = "pEC50_dr"
    _PRED_PATH3_GZ = Path("../predictions/4_fp_model_comparison_2.csv.gz")
    _CKPT3_PATH    = _PRED_PATH3_GZ.with_suffix(".ckpt.gz")
    _N_OUTER3 = 5
    _N_INNER3 = 5
    _SEED3    = 42
    _P_VAL3   = 0.1
    _N_FOLDS3 = _N_OUTER3 * _N_INNER3
    _FPS3     = ["mqn", "mordred", "chemeleon"]
    _EXPECTED3 = {f"tabpfn_{fp}" for fp in _FPS3}

    # ── TabPFN subprocess script ──────────────────────────────────────────────
    # torch.set_num_threads uses all logical CPUs (P-cores + E-cores on Apple
    # Silicon) instead of PyTorch's default of P-cores only.
    _SCRIPT_PATH = Path(tempfile.gettempdir()) / "tabpfn_a3.py"
    _SCRIPT_PATH.write_text("\n".join([
        "import os, sys, numpy as np",
        "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
        "from dotenv import load_dotenv; from pathlib import Path",
        "load_dotenv(Path('.env'))",
        "import torch",
        "torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))",
        "from tabpfn import TabPFNRegressor",
        "X_train = np.load(sys.argv[1])",
        "y_train = np.load(sys.argv[2])",
        "X_test  = np.load(sys.argv[3])",
        "out     = sys.argv[4]",
        "model = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=True, device='cpu')",
        "model.fit(X_train, y_train)",
        "np.save(out, model.predict(X_test))",
    ]))

    def _tabpfn3_predict(X_train: np.ndarray, y_train: np.ndarray, X_test: np.ndarray) -> np.ndarray:
        tmp = Path(tempfile.gettempdir())
        f_Xtr, f_ytr, f_Xte = tmp / "a3_Xtr.npy", tmp / "a3_ytr.npy", tmp / "a3_Xte.npy"
        f_out = tmp / "a3_preds"
        np.save(str(f_Xtr), X_train); np.save(str(f_ytr), y_train); np.save(str(f_Xte), X_test)
        res = subprocess.run(
            [sys.executable, str(_SCRIPT_PATH), str(f_Xtr), str(f_ytr), str(f_Xte), str(f_out)],
            capture_output=True, text=True, cwd=str(Path("../").resolve()),
        )
        if res.returncode != 0:
            raise RuntimeError(f"TabPFN subprocess failed:\n{res.stderr}")
        preds = np.load(str(f_out) + ".npy")
        for p in [f_Xtr, f_ytr, f_Xte, Path(str(f_out) + ".npy")]:
            p.unlink(missing_ok=True)
        return preds

    def _checkpoint3(records: list[dict]) -> None:
        if not records:
            return
        _CKPT3_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_CKPT3_PATH, "wb") as _f:
            pl.DataFrame(records).write_csv(_f)

    # ── Check completion ──────────────────────────────────────────────────────
    _is_complete3 = (
        _PRED_PATH3_GZ.exists()
        and _EXPECTED3 <= set(pl.read_csv(_PRED_PATH3_GZ)["method"].unique().to_list())
    )

    if _is_complete3:
        print(f"All methods complete — loading {_PRED_PATH3_GZ}.")
        tabpfn_pred_df = pl.read_csv(_PRED_PATH3_GZ)
    else:
        # Seed accumulator from final file (already-done methods) or checkpoint
        if _PRED_PATH3_GZ.exists():
            _all3: list[dict] = pl.read_csv(_PRED_PATH3_GZ).to_dicts()
        elif _CKPT3_PATH.exists():
            _all3 = pl.read_csv(_CKPT3_PATH).to_dicts()
        else:
            _all3 = []

        # Which methods and folds are already saved
        _done_methods3: set[str] = {
            m for m in _EXPECTED3
            if sum(1 for r in _all3 if r["method"] == m and r["fold"]) >= _N_FOLDS3
        }
        _done_folds3_by_method: dict[str, set[int]] = {
            f"tabpfn_{fp}": {r["fold"] for r in _all3 if r["method"] == f"tabpfn_{fp}"}
            for fp in _FPS3
        }

        for _fp in _FPS3:
            _method = f"tabpfn_{_fp}"
            _done_folds = _done_folds3_by_method[_method]
            if len(_done_folds) >= _N_FOLDS3:
                print(f"{_method} already complete — skipping.")
                continue
            if _done_folds:
                print(f"Resuming {_method} from fold {len(_done_folds)}/{_N_FOLDS3}")

            _remaining = [
                t for t in generate_cv_splits_random(
                    tabpfn_train, n_outer=_N_OUTER3, n_inner=_N_INNER3,
                    seed=_SEED3, p_val=_P_VAL3,
                )
                if t[0] not in _done_folds
            ]
            for _fold3, _outer3, _inner3, _train3, _, _test3 in tqdm(
                _remaining,
                total=_N_FOLDS3 - len(_done_folds),
                desc=f"CV {_method}", unit="fold",
            ):
                _y_train3  = _train3[_TARGET_COL3].to_numpy()
                _y_true3   = _test3[_TARGET_COL3].to_numpy()
                if _fp == "chemeleon":
                    _Xtr3, _Xte3 = chemeleon_embed(
                        _train3["smiles"].to_list(), _test3["smiles"].to_list(),
                        prefix="a3_che",
                    )
                else:
                    _train_fp3 = generate_fingerprint(_train3, _fp)
                    _test_fp3  = generate_fingerprint(_test3,  _fp)
                    _Xtr3 = extract_fp_matrix(_train_fp3, _fp)
                    _Xte3 = extract_fp_matrix(_test_fp3,  _fp)
                    del _train_fp3, _test_fp3
                    if np.isnan(_Xtr3).any():
                        _valid = ~np.isnan(_Xtr3).any(axis=0)
                        _Xtr3, _Xte3 = _Xtr3[:, _valid], _Xte3[:, _valid]
                _y_pred3 = _tabpfn3_predict(_Xtr3, _y_train3, _Xte3)
                del _Xtr3, _Xte3
                gc.collect()
                for _ik3, _mn3, _smi3, _yt3, _yp3 in zip(
                    _test3["inchikey"].to_list(), _test3["molecule_names"].to_list(),
                    _test3["smiles"].to_list(), _y_true3.tolist(), _y_pred3.tolist(),
                ):
                    _all3.append({
                        "inchikey": _ik3, "molecule_names": _mn3, "smiles": _smi3,
                        "fold": _fold3, "outer_fold": _outer3, "inner_fold": _inner3,
                        "model": "tabpfn", "fingerprint": _fp,
                        "method": _method, "y_true": _yt3, "y_pred": _yp3,
                    })
                _checkpoint3(_all3)
            print(f"{_method} done.")

        _PRED_PATH3_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH3_GZ, "wb") as _f:
            pl.DataFrame(_all3).write_csv(_f)
        _CKPT3_PATH.unlink(missing_ok=True)
        print(f"Done — {len(_all3):,} rows → {_PRED_PATH3_GZ}")

        tabpfn_pred_df = pl.read_csv(_PRED_PATH3_GZ)
    return (tabpfn_pred_df,)


@app.cell
def _(
    Path,
    calc_regression_metrics,
    make_mcs_plot_grid,
    mo,
    pl,
    tabpfn_pred_df,
):
    """
    Compare TabPFN against RF (Mordred only) and CheMeleon baseline from notebook 2.
    """
    # ── RF Mordred baseline from Analysis 1 ──────────────────────────────────
    _a1_ref = (
        pl.read_csv("../predictions/4_fp_model_comparison_1.csv.gz")
        .filter(pl.col("method") == "rf_mordred")
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    # ── CheMeleon fine-tuned baseline from notebook 2 ────────────────────────
    _chemeleon_ref = (
        pl.read_csv("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns([
            pl.lit("chemeleon").alias("method"),
            pl.lit("random").alias("split"),
        ])
    )
    _tabpfn_ref = (
        tabpfn_pred_df
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )

    _combined3 = pl.concat([_a1_ref, _chemeleon_ref, _tabpfn_ref], how="diagonal")

    _metrics_df3 = calc_regression_metrics(
        _combined3,
        cycle_col="cv_cycle",
        val_col="y_true",
        pred_col="y_pred",
        thresh=4.0,
    )

    _summary3 = (
        _metrics_df3
        .group_by("method")
        .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
        .sort("mae", descending=False)
    )

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _fig3 = make_mcs_plot_grid(
        _metrics_df3,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOTS_DIR / "analysis3_mcs_mae.png",
    )

    mo.vstack([
        mo.md("## Analysis 3 — TabPFN vs RF Mordred and CheMeleon (sorted by MAE)"),
        mo.plain_text(_summary3.to_pandas().to_string(index=False)),
        mo.md("### MCS grid — MAE"),
        mo.as_html(_fig3),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 4 — Multitask Macau on CheMeleon fingerprint

    Tests whether auxiliary assay data improves Macau's dose-response predictions
    when using CheMeleon embeddings as row side information.

    Four scenarios are compared, all using the same 5×5 CV splits (seed=42):

    | Scenario | Train rows | Targets | Description |
    |----------|-----------|---------|-------------|
    | `base` | 4 138 | pEC50\_dr | DR compounds only — from Analysis 1 (`macau_chemeleon`) |
    | `+sd_cols` | 4 138 | pEC50\_dr + 10 µM log₂FC | Adds single-dose signal for ~66 % of DR compounds |
    | `+counter` | 4 138 | pEC50\_dr + pEC50\_counter | Adds counter-assay IC50 for ~64 % of DR compounds |
    | `+sd_counter` | 4 138 | pEC50\_dr + 10 µM log₂FC + pEC50\_counter | Both auxiliary columns, DR compounds only |
    | `+sd_rows` | 12 269 | pEC50\_dr + 10 µM log₂FC | Augments training matrix with 8 131 SD-only compounds (pEC50\_dr = missing) |

    **Note on `+sd_rows`**: the SD-only compounds are added to the *training* split
    only; they are never part of the test fold, so evaluation remains on DR
    compounds alone and cross-fold comparisons stay fair.
    """)
    return


@app.cell
def _(
    MacauModel,
    Path,
    chemeleon_embed,
    gc,
    generate_cv_splits_random,
    gzip,
    np,
    pl,
    tqdm,
):
    """
    Multitask Macau on CheMeleon embeddings — four auxiliary-data scenarios.

    Y matrix is (n_compounds × n_targets), represented as a sparse COO matrix
    so that missing values (NaN) are simply absent and do not bias the MCMC.
    Predictions are for pEC50_dr (column 0) only; auxiliary columns are used
    only to regularise the latent compound factors during training.
    """
    import scipy.sparse as _sp

    _TARGET_COL  = "pEC50_dr"
    _SD_COL      = "10.0_log2_fc"
    _COUNTER_COL = "pEC50_counter"
    _PRED_PATH   = Path("../predictions/4_macau_multitask.csv.gz")
    _CKPT_PATH   = _PRED_PATH.with_suffix(".ckpt.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1
    _N_FOLDS = _N_OUTER * _N_INNER

    _SCENARIOS = ["+sd_cols", "+counter", "+sd_counter", "+sd_rows"]
    _EXPECTED  = set(_SCENARIOS)

    # ── Load full activity table ──────────────────────────────────────────────
    _all_data = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    # DR training compounds (excludes test set — in_test flag is always False for DR)
    _dr_data = _all_data.filter(pl.col(_TARGET_COL).is_not_null())
    # Extra SD-only compounds for +sd_rows scenario
    _sd_only  = _all_data.filter(
        pl.col("in_single_dose") & ~pl.col("in_dose_response")
    )

    def _make_Y(df: pl.DataFrame, target_cols: list[str]):
        """Build a sparse (n × k) target matrix, omitting NaN entries."""
        _rows, _cols, _vals = [], [], []
        for _j, _tc in enumerate(target_cols):
            if _tc not in df.columns:
                continue
            _arr = df[_tc].to_numpy().astype(float)
            _mask = ~np.isnan(_arr)
            _rows.extend(np.where(_mask)[0].tolist())
            _cols.extend([_j] * int(_mask.sum()))
            _vals.extend(_arr[_mask].tolist())
        return _sp.coo_matrix(
            (np.array(_vals), (np.array(_rows), np.array(_cols))),
            shape=(len(df), len(target_cols)),
        )

    # ── Check / resume ────────────────────────────────────────────────────────
    _is_complete = (
        _PRED_PATH.exists()
        and _EXPECTED <= set(pl.read_csv(_PRED_PATH)["scenario"].unique().to_list())
    )

    if _is_complete:
        print(f"All scenarios complete — loading {_PRED_PATH}.")
        macau_mt_pred_df = pl.read_csv(_PRED_PATH)
    else:
        if _PRED_PATH.exists():
            _all_records: list[dict] = pl.read_csv(_PRED_PATH).to_dicts()
        elif _CKPT_PATH.exists():
            _all_records = pl.read_csv(_CKPT_PATH).to_dicts()
        else:
            _all_records = []

        _done_scenarios: dict[str, set[int]] = {
            sc: {r["fold"] for r in _all_records if r["scenario"] == sc}
            for sc in _SCENARIOS
        }

        def _ckpt(records: list[dict]) -> None:
            if not records:
                return
            _CKPT_PATH.parent.mkdir(parents=True, exist_ok=True)
            with gzip.open(_CKPT_PATH, "wb") as _f:
                pl.DataFrame(records).write_csv(_f)

        for _sc in _SCENARIOS:
            _done_folds = _done_scenarios[_sc]
            if len(_done_folds) >= _N_FOLDS:
                print(f"{_sc}: already complete — skipping.")
                continue
            if _done_folds:
                print(f"{_sc}: resuming from fold {len(_done_folds)}/{_N_FOLDS}")

            # Which target columns does this scenario use?
            _target_cols = {
                "+sd_cols":    [_TARGET_COL, _SD_COL],
                "+counter":    [_TARGET_COL, _COUNTER_COL],
                "+sd_counter": [_TARGET_COL, _SD_COL, _COUNTER_COL],
                "+sd_rows":    [_TARGET_COL, _SD_COL],
            }[_sc]

            _remaining = [
                t for t in generate_cv_splits_random(
                    _dr_data, n_outer=_N_OUTER, n_inner=_N_INNER,
                    seed=_SEED, p_val=_P_VAL,
                )
                if t[0] not in _done_folds
            ]

            for _fold, _outer, _inner, _train_dr, _, _test_dr in tqdm(
                _remaining,
                total=_N_FOLDS - len(_done_folds),
                desc=f"Macau multitask / {_sc}", unit="fold",
            ):
                # For +sd_rows: append SD-only compounds to the training split
                if _sc == "+sd_rows":
                    _train_rows = pl.concat([_train_dr, _sd_only], how="diagonal")
                else:
                    _train_rows = _train_dr

                # CheMeleon embeddings (subprocess to avoid OpenMP conflict)
                _Xtr, _Xte = chemeleon_embed(
                    _train_rows["smiles"].to_list(),
                    _test_dr["smiles"].to_list(),
                    prefix=f"a4_{_sc}",
                )

                # Build sparse Y matrix for training
                _Y_train = _make_Y(_train_rows, _target_cols)

                _model = MacauModel()
                _model.train(_Xtr, _Y_train)
                # predict returns (n_test × k); column 0 = pEC50_dr
                _preds_all = _model.predict(_Xte)
                _preds = _preds_all[:, 0] if _preds_all.ndim == 2 else _preds_all
                del _model, _Xtr, _Xte
                gc.collect()

                _y_true = _test_dr[_TARGET_COL].to_numpy()
                for _ik, _mn, _smi, _yt, _yp in zip(
                    _test_dr["inchikey"].to_list(),
                    _test_dr["molecule_names"].to_list(),
                    _test_dr["smiles"].to_list(),
                    _y_true.tolist(),
                    _preds.tolist(),
                ):
                    _all_records.append({
                        "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                        "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                        "scenario": _sc, "method": f"macau_{_sc}",
                        "y_true": _yt, "y_pred": _yp,
                    })
                _ckpt(_all_records)
            print(f"{_sc}: done.")

        _PRED_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH, "wb") as _f:
            pl.DataFrame(_all_records).write_csv(_f)
        _CKPT_PATH.unlink(missing_ok=True)
        print(f"All done — {len(_all_records):,} rows → {_PRED_PATH}")

        macau_mt_pred_df = pl.read_csv(_PRED_PATH)
    return (macau_mt_pred_df,)


@app.cell
def _(
    Path,
    calc_regression_metrics,
    macau_mt_pred_df,
    make_mcs_plot_grid,
    mo,
    pl,
):
    """
    Compare the multitask Macau scenarios against the single-task base
    (macau_chemeleon from Analysis 1) and the CheMeleon fine-tuned baseline.
    """
    # ── base: macau chemeleon single-task from Analysis 1 ────────────────────
    _macau_base = (
        pl.read_csv("../predictions/4_fp_model_comparison_1.csv.gz")
        .filter(pl.col("method") == "macau_chemeleon")
        .rename({"fold": "cv_cycle"})
        .with_columns([
            pl.lit("base").alias("scenario"),
            pl.lit("macau_base").alias("method"),
            pl.lit("random").alias("split"),
        ])
    )
    # ── CheMeleon fine-tuned baseline from notebook 2 ────────────────────────
    _chemeleon_base = (
        pl.read_csv("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "scenario", "fold": "cv_cycle"})
        .with_columns([
            pl.lit("chemeleon_base").alias("scenario"),
            pl.lit("chemeleon_base").alias("method"),
            pl.lit("random").alias("split"),
        ])
    )
    _macau_ref = (
        macau_mt_pred_df
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )

    _combined4 = pl.concat([_macau_base, _macau_ref, _chemeleon_base], how="diagonal")

    _metrics4 = calc_regression_metrics(
        _combined4,
        cycle_col="cv_cycle",
        val_col="y_true",
        pred_col="y_pred",
        thresh=4.0,
    )

    _summary4 = (
        _metrics4
        .group_by("method")
        .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
        .sort("mae")
    )

    _PLOTS_DIR = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _SHORTEN4 = {
        "macau_base":        "base",
        "chemeleon_base":    "che_base",
        "macau_+sd_cols":    "+sd",
        "macau_+counter":    "+ctr",
        "macau_+sd_counter": "+sd+ctr",
        "macau_+sd_rows":    "+sd_rows",
    }
    _metrics4_plot = _metrics4.with_columns(
        pl.col("method").replace(_SHORTEN4)
    )

    _fig4 = make_mcs_plot_grid(
        _metrics4_plot,
        stats=["mae"],
        group_col="method",
        figsize=(10, 10),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOTS_DIR / "analysis4_macau_multitask_mcs_mae.png",
    )

    mo.vstack([
        mo.md("## Analysis 4 — Multitask Macau (CheMeleon) vs single-task baseline"),
        mo.plain_text(_summary4.to_pandas().to_string(index=False)),
        mo.md("### MCS grid — MAE"),
        mo.as_html(_fig4),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 5 — HPO for RF, XGBoost, Macau and TabPFN

    Optuna TPE hyperparameter optimisation (50 trials, 1×5 CV, objective = mean MAE).

    | Model | Fingerprint | Parameters tuned |
    |-------|-------------|-----------------|
    | RF | Mordred | n\_estimators, max\_depth, min\_samples\_leaf, max\_features |
    | XGBoost | Mordred | n\_estimators, max\_depth, learning\_rate, subsample, colsample\_bytree, reg\_alpha |
    | Macau | CheMeleon | num\_latent, nsamples, burnin |

    Each study is persisted to SQLite (`predictions/4_hpo_*.db`) and can be resumed.
    Best params are then evaluated with 5×5 CV and compared via MCS.
    """)
    return


@app.cell
def _(
    BoostedTreesModel,
    MacauModel,
    Path,
    RandomForestModel,
    calc_regression_metrics,
    chemeleon_embed,
    extract_fp_matrix,
    gc,
    generate_cv_splits_random,
    generate_fingerprint,
    gzip,
    make_mcs_plot_grid,
    mo,
    np,
    optuna,
    pl,
    tqdm,
):
    """
    Analysis 5: Optuna HPO (50 trials, 1×5 CV) for RF/XGB on Mordred and
    Macau on CheMeleon, followed by 5×5 CV evaluation of best params.
    """
    _TARGET_COL = "pEC50_dr"
    _SEED       = 42
    _P_VAL      = 0.1
    _N_TRIALS   = 50

    # ── Load training data ────────────────────────────────────────────────────
    _dr_train5 = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col(_TARGET_COL).is_not_null())
        .select(["smiles", "inchikey", "molecule_names", _TARGET_COL])
    )

    # ── Determine upfront whether all work is already done ───────────────────
    _PRED5_PATH   = Path("../predictions/4_hpo_a5_best_5x5cv.csv.gz")
    _EXPECTED5    = {"rf_mordred_hpo", "xgb_mordred_hpo", "macau_che_hpo"}
    _DB_DIR       = Path("../predictions")

    _hpo_done = all(
        len([t for t in optuna.load_study(
            study_name=name,
            storage=f"sqlite:///{_DB_DIR}/{db}",
        ).trials if t.state == optuna.trial.TrialState.COMPLETE]) >= _N_TRIALS
        for name, db in [
            ("rf_mordred_hpo",      "4_hpo_rf_mordred.db"),
            ("xgb_mordred_hpo",     "4_hpo_xgb_mordred.db"),
            ("macau_chemeleon_hpo", "4_hpo_macau_chemeleon.db"),
        ]
        if (_DB_DIR / db).exists()
    ) if all((_DB_DIR / db).exists() for _, db in [
        ("rf_mordred_hpo",      "4_hpo_rf_mordred.db"),
        ("xgb_mordred_hpo",     "4_hpo_xgb_mordred.db"),
        ("macau_chemeleon_hpo", "4_hpo_macau_chemeleon.db"),
    ]) else False

    _cv_done = (
        _PRED5_PATH.exists()
        and _EXPECTED5 <= set(pl.read_csv(_PRED5_PATH)["method"].unique().to_list())
    )

    _all_done = _hpo_done and _cv_done

    # ── Pre-compute fingerprint matrices (skipped if results already exist) ───
    if not _all_done:
        # Mordred (for RF / XGB)
        _mordred_fp_all = generate_fingerprint(_dr_train5, "mordred")
        _X_mordred_all  = extract_fp_matrix(_mordred_fp_all, "mordred")
        del _mordred_fp_all
        if np.isnan(_X_mordred_all).any():
            _valid_cols = ~np.isnan(_X_mordred_all).any(axis=0)
            _X_mordred_all = _X_mordred_all[:, _valid_cols]

        # CheMeleon (for Macau) — embed all compounds once via subprocess.
        _X_che_all, _ = chemeleon_embed(
            _dr_train5["smiles"].to_list(),
            _dr_train5["smiles"].to_list()[:1],  # dummy single-row test; discarded
            prefix="a5_precompute",
        )

        # Shared inchikey → row-index lookup for both matrices
        _ik_to_idx = {ik: i for i, ik in enumerate(_dr_train5["inchikey"].to_list())}

        def _idx(df) -> np.ndarray:
            return np.array([_ik_to_idx[k] for k in df["inchikey"].to_list()])
    else:
        print("All HPO studies and 5×5 CV complete — skipping fingerprint computation.")
        _X_mordred_all = _X_che_all = None

        def _idx(df) -> np.ndarray:  # placeholder; never called when _all_done
            raise RuntimeError("Fingerprints not computed.")

    # ── Helper: 1×5 CV MAE for a given predict function ──────────────────────
    def _cv1x5_mae(predict_fn) -> float:
        maes = []
        for _fold, _outer, _inner, _tr, _val, _te in generate_cv_splits_random(
            _dr_train5, n_outer=1, n_inner=5, seed=_SEED, p_val=_P_VAL,
        ):
            maes.append(float(np.abs(predict_fn(_tr, _val, _te) - _te[_TARGET_COL].to_numpy()).mean()))
        return float(np.mean(maes))

    # ─────────────────────────────────────────────────────────────────────────
    # HPO studies
    # ─────────────────────────────────────────────────────────────────────────
    _DB_DIR.mkdir(parents=True, exist_ok=True)

    # ── RF on Mordred ─────────────────────────────────────────────────────────
    def _rf_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators    = trial.suggest_int("n_estimators", 100, 800, step=100),
            max_depth       = trial.suggest_int("max_depth", 3, 30),
            min_samples_leaf= trial.suggest_int("min_samples_leaf", 1, 20),
            max_features    = trial.suggest_float("max_features", 0.1, 1.0),
        )
        def _pred(tr, val, te):
            m = RandomForestModel(pred_type="regression", random_state=_SEED, **params)
            m.train(_X_mordred_all[_idx(tr)], tr[_TARGET_COL].to_numpy())
            return m.predict(_X_mordred_all[_idx(te)])
        return _cv1x5_mae(_pred)

    _rf_study = optuna.create_study(
        study_name="rf_mordred_hpo",
        storage=f"sqlite:///{_DB_DIR}/4_hpo_rf_mordred.db",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )
    _rf_done = len([t for t in _rf_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if _rf_done < _N_TRIALS:
        _rf_study.optimize(_rf_objective, n_trials=_N_TRIALS - _rf_done, show_progress_bar=True)
    _rf_best = {**_rf_study.best_params}
    print(f"RF best (MAE={_rf_study.best_value:.4f}): {_rf_best}")

    # ── XGBoost on Mordred ────────────────────────────────────────────────────
    def _xgb_objective(trial: optuna.Trial) -> float:
        params = dict(
            n_estimators     = trial.suggest_int("n_estimators", 100, 1500, step=100),
            max_depth        = trial.suggest_int("max_depth", 3, 10),
            learning_rate    = trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
            subsample        = trial.suggest_float("subsample", 0.5, 1.0),
            colsample_bytree = trial.suggest_float("colsample_bytree", 0.5, 1.0),
            reg_alpha        = trial.suggest_float("reg_alpha", 1e-4, 10.0, log=True),
        )
        def _pred(tr, val, te):
            m = BoostedTreesModel(pred_type="regression", **params)
            m.train(_X_mordred_all[_idx(tr)], tr[_TARGET_COL].to_numpy(),
                    _X_mordred_all[_idx(val)], val[_TARGET_COL].to_numpy())
            return m.predict(_X_mordred_all[_idx(te)])
        return _cv1x5_mae(_pred)

    _xgb_study = optuna.create_study(
        study_name="xgb_mordred_hpo",
        storage=f"sqlite:///{_DB_DIR}/4_hpo_xgb_mordred.db",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )
    _xgb_done = len([t for t in _xgb_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if _xgb_done < _N_TRIALS:
        _xgb_study.optimize(_xgb_objective, n_trials=_N_TRIALS - _xgb_done, show_progress_bar=True)
    _xgb_best = {**_xgb_study.best_params}
    print(f"XGB best (MAE={_xgb_study.best_value:.4f}): {_xgb_best}")

    # ── Macau on CheMeleon ────────────────────────────────────────────────────
    def _macau_objective(trial: optuna.Trial) -> float:
        params = dict(
            num_latent = trial.suggest_int("num_latent", 8, 64, step=8),
            nsamples   = trial.suggest_int("nsamples", 100, 1000, step=100),
            burnin     = trial.suggest_int("burnin", 50, 400, step=50),
        )
        def _pred(tr, val, te):
            m = MacauModel(seed=_SEED, **params)
            m.train(_X_che_all[_idx(tr)], tr[_TARGET_COL].to_numpy())
            return m.predict(_X_che_all[_idx(te)])
        return _cv1x5_mae(_pred)

    _mac_study = optuna.create_study(
        study_name="macau_chemeleon_hpo",
        storage=f"sqlite:///{_DB_DIR}/4_hpo_macau_chemeleon.db",
        load_if_exists=True, direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=_SEED),
    )
    _mac_done = len([t for t in _mac_study.trials if t.state == optuna.trial.TrialState.COMPLETE])
    if _mac_done < _N_TRIALS:
        _mac_study.optimize(_macau_objective, n_trials=_N_TRIALS - _mac_done, show_progress_bar=True)
    _mac_best = {**_mac_study.best_params}
    print(f"Macau best (MAE={_mac_study.best_value:.4f}): {_mac_best}")

    # ─────────────────────────────────────────────────────────────────────────
    # 5×5 CV evaluation with best params
    # ─────────────────────────────────────────────────────────────────────────
    _MODELS5 = {
        "rf_mordred_hpo":    ("rf",    _rf_best),
        "xgb_mordred_hpo":   ("xgb",   _xgb_best),
        "macau_che_hpo":     ("macau", _mac_best),
    }

    _is_complete5 = _cv_done

    if _is_complete5:
        print(f"5×5 CV results found — loading.")
        _pred5_df = pl.read_csv(_PRED5_PATH)
    else:
        if _PRED5_PATH.exists():
            _all5: list[dict] = pl.read_csv(_PRED5_PATH).to_dicts()
        else:
            _all5 = []
        _done5_by_method = {
            mk: {r["fold"] for r in _all5 if r["method"] == mk}
            for mk in _MODELS5
        }

        for _mk, (_mtype, _params) in _MODELS5.items():
            _done_folds5 = _done5_by_method[_mk]
            if len(_done_folds5) >= 25:
                print(f"{_mk}: complete — skipping.")
                continue
            if _done_folds5:
                print(f"{_mk}: resuming from fold {len(_done_folds5)}/25")

            for _fold, _outer, _inner, _tr, _val, _te in tqdm(
                [t for t in generate_cv_splits_random(
                    _dr_train5, n_outer=5, n_inner=5, seed=_SEED, p_val=_P_VAL,
                ) if t[0] not in _done_folds5],
                total=25 - len(_done_folds5), desc=f"5×5 CV {_mk}", unit="fold",
            ):
                if _mtype == "rf":
                    _m = RandomForestModel(pred_type="regression", random_state=_SEED, **_params)
                    _m.train(_X_mordred_all[_idx(_tr)], _tr[_TARGET_COL].to_numpy())
                    _preds5 = _m.predict(_X_mordred_all[_idx(_te)])
                    del _m

                elif _mtype == "xgb":
                    _m = BoostedTreesModel(pred_type="regression", **_params)
                    _m.train(_X_mordred_all[_idx(_tr)], _tr[_TARGET_COL].to_numpy(),
                             _X_mordred_all[_idx(_val)], _val[_TARGET_COL].to_numpy())
                    _preds5 = _m.predict(_X_mordred_all[_idx(_te)])
                    del _m

                elif _mtype == "macau":
                    _m = MacauModel(seed=_SEED, **_params)
                    _m.train(_X_che_all[_idx(_tr)], _tr[_TARGET_COL].to_numpy())
                    _preds5 = _m.predict(_X_che_all[_idx(_te)])
                    del _m

                gc.collect()
                for _ik, _mn, _smi, _yt, _yp in zip(
                    _te["inchikey"].to_list(), _te["molecule_names"].to_list(),
                    _te["smiles"].to_list(),
                    _te[_TARGET_COL].to_numpy().tolist(), _preds5.tolist(),
                ):
                    _all5.append({
                        "inchikey": _ik, "molecule_names": _mn, "smiles": _smi,
                        "fold": _fold, "outer_fold": _outer, "inner_fold": _inner,
                        "method": _mk, "y_true": _yt, "y_pred": _yp,
                    })

                # checkpoint after each fold
                _PRED5_PATH.parent.mkdir(parents=True, exist_ok=True)
                with gzip.open(_PRED5_PATH, "wb") as _f:
                    pl.DataFrame(_all5).write_csv(_f)

            print(f"{_mk}: done.")

        _pred5_df = pl.read_csv(_PRED5_PATH)

    # ─────────────────────────────────────────────────────────────────────────
    # Analysis: metrics + MCS
    # ─────────────────────────────────────────────────────────────────────────
    # Add Analysis 1 baselines for context: rf_mordred and macau_chemeleon
    _a1_baseline = (
        pl.read_csv("../predictions/4_fp_model_comparison_1.csv.gz")
        .filter(pl.col("method").is_in(["rf_mordred", "macau_chemeleon"]))
        .rename({"fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _pred5_combined = pl.concat([
        _pred5_df.rename({"fold": "cv_cycle"}).with_columns(pl.lit("random").alias("split")),
        _a1_baseline,
    ], how="diagonal")

    _metrics5 = calc_regression_metrics(
        _pred5_combined, cycle_col="cv_cycle",
        val_col="y_true", pred_col="y_pred", thresh=4.0,
    )
    _summary5 = (
        _metrics5
        .group_by("method")
        .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
        .sort("mae")
    )

    _PLOTS_DIR5 = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR5.mkdir(parents=True, exist_ok=True)

    _fig5 = make_mcs_plot_grid(
        _metrics5,
        stats=["mae"],
        group_col="method",
        figsize=(12, 12),
        effect_dict={"mae": 0.1},
        sort_axes=True,
        save_path=_PLOTS_DIR5 / "analysis5_hpo_mcs_mae.png",
    )

    _best_table = pl.DataFrame([
        {"model": "RF (Mordred)",       **{k: str(v) for k, v in _rf_best.items()},  "cv_mae": round(_rf_study.best_value,  4)},
        {"model": "XGBoost (Mordred)",  **{k: str(v) for k, v in _xgb_best.items()}, "cv_mae": round(_xgb_study.best_value, 4)},
        {"model": "Macau (CheMeleon)",  **{k: str(v) for k, v in _mac_best.items()}, "cv_mae": round(_mac_study.best_value, 4)},
    ])

    mo.vstack([
        mo.md("## Analysis 5 — HPO best params (1×5 CV)"),
        mo.plain_text(_best_table.to_pandas().to_string(index=False)),
        mo.md("## 5×5 CV evaluation vs untuned baselines (sorted by MAE)"),
        mo.plain_text(_summary5.to_pandas().to_string(index=False)),
        mo.md("### MCS grid — MAE"),
        mo.as_html(_fig5),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Analysis 6 — Ensemble sweep over HPO model predictions

    Exhaustive weighted-average ensemble over six HPO-tuned models:

    | Short | Model | FP |
    |-------|-------|----|
    | `cp` | Chemprop HPO | graph |
    | `ch` | CheMeleon HPO | graph |
    | `rf` | RF HPO | Mordred |
    | `xg` | XGBoost HPO | Mordred |
    | `mc` | Macau HPO | CheMeleon |
    | `tf` | TabPFN | CheMeleon |

    Each weight ∈ {0, ⅕, ¼, ⅓, ½, 1, 2, 3, 4, 5}. Ratio-duplicate combinations are pruned
    (e.g. all-2 = all-1), and at least 2 models must be active (weight > 0).
    Combinations are evaluated one at a time to avoid memory pressure — only
    per-fold MAE and Spearman ρ are kept per combo, not the full prediction matrix.
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, gzip, mcs_plot, mo, np, pl, plt, tqdm):
    """
    Sweep all ratio-distinct weight combinations for 6 HPO models.

    Processes one combination at a time: computes weighted-average predictions
    across all 25 folds, calculates MAE and Spearman ρ, appends a single-row
    summary to the output CSV, then moves to the next combination.  Memory use
    is constant regardless of search space size.  Resumes automatically if the
    output file already exists.
    """
    import itertools
    from fractions import Fraction

    _OUT_PATH = Path("../predictions/4_ensemble_sweep_metrics.csv.gz")

    # ── Model definitions ─────────────────────────────────────────────────────
    _MODELS = [
        ("cp",  Path("../predictions/4_hpo_best_5x5cv.csv.gz"),        "method", "chemprop_hpo"),
        ("ch",  Path("../predictions/4_hpo_best_5x5cv.csv.gz"),        "method", "chemeleon_hpo"),
        ("rf",  Path("../predictions/4_hpo_a5_best_5x5cv.csv.gz"),     "method", "rf_mordred_hpo"),
        ("xg",  Path("../predictions/4_hpo_a5_best_5x5cv.csv.gz"),     "method", "xgb_mordred_hpo"),
        ("mc",  Path("../predictions/4_hpo_a5_best_5x5cv.csv.gz"),     "method", "macau_che_hpo"),
        ("tf",  Path("../predictions/4_fp_model_comparison_2.csv.gz"), "method", "tabpfn_chemeleon"),
    ]
    _TAGS = [m[0] for m in _MODELS]

    # ── Build ratio-distinct weight matrix ────────────────────────────────────
    _W_frac = [Fraction(0), Fraction(1, 5), Fraction(1, 4), Fraction(1, 3),
               Fraction(1, 2), Fraction(1), Fraction(2), Fraction(3),
               Fraction(4), Fraction(5)]
    _W_vals = np.array([0.0, 1/5, 1/4, 1/3, 1/2, 1.0, 2.0, 3.0, 4.0, 5.0],
                       dtype=np.float64)

    _seen: set = set()
    _combos_idx: list[tuple[int, ...]] = []
    for _idx_tuple in itertools.product(range(10), repeat=6):
        _w_frac = tuple(_W_frac[i] for i in _idx_tuple)
        if sum(x > 0 for x in _w_frac) < 2:
            continue
        _min_nz = min(x for x in _w_frac if x > 0)
        _norm = tuple(x / _min_nz for x in _w_frac)
        if _norm not in _seen:
            _seen.add(_norm)
            _combos_idx.append(_idx_tuple)

    _W_mat = _W_vals[np.array(_combos_idx)]                    # (n_combos, 6)
    _W_norm = (_W_mat / _W_mat.sum(axis=1, keepdims=True))     # normalised rows
    _n_combos = len(_combos_idx)
    _combo_to_row = {t: i for i, t in enumerate(_combos_idx)}  # O(1) lookup

    def _combo_label(idx_tuple: tuple[int, ...]) -> str:
        """Compact label e.g. 'cp1_ch2_rf0_xg1_mc0_tf3'."""
        _label_map = {0: "0", 1: "15", 2: "14", 3: "13", 4: "12",
                      5: "1", 6: "2", 7: "3", 8: "4", 9: "5"}
        return "_".join(f"{tag}{_label_map[i]}" for tag, i in zip(_TAGS, idx_tuple))

    # ── Load and pivot predictions (done once regardless of cache state) ───────
    _parts = []
    for _tag, _path, _col, _val in _MODELS:
        _parts.append(
            pl.read_csv(_path)
            .filter(pl.col(_col) == _val)
            .select(["inchikey", "fold", "y_true", pl.col("y_pred").alias(_tag)])
        )
    _wide = _parts[0]
    for _p in _parts[1:]:
        _wide = _wide.join(_p.drop("y_true"), on=["inchikey", "fold"])

    # Pre-extract per-fold numpy arrays — avoids repeated Polars filtering
    _folds = sorted(_wide["fold"].unique().to_list())
    _fold_arrays: list[tuple[np.ndarray, np.ndarray]] = []
    for _f in _folds:
        _fd = _wide.filter(pl.col("fold") == _f)
        _fold_arrays.append((
            _fd.select(_TAGS).to_numpy().astype(np.float64),  # (n_cmp, 6)
            _fd["y_true"].to_numpy().astype(np.float64),       # (n_cmp,)
        ))

    def _spearman(a: np.ndarray, b: np.ndarray) -> float:
        """Spearman ρ between 1-D arrays a and b."""
        _ra = np.argsort(np.argsort(a)).astype(np.float64)
        _rb = np.argsort(np.argsort(b)).astype(np.float64)
        _da, _db = _ra - _ra.mean(), _rb - _rb.mean()
        _denom = _da.std() * _db.std()
        return float(_da @ _db / (len(_ra) * _denom)) if _denom > 0 else 0.0

    if _OUT_PATH.exists():
        _done_labels = set(pl.read_csv(_OUT_PATH)["ensemble"].to_list())
        print(f"Resuming — {len(_done_labels):,} of {_n_combos:,} combos already done.")
    else:
        _done_labels = set()
        _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    # ── One combination at a time, appending metrics to file ──────────────────
    _header_written = _OUT_PATH.exists()
    _remaining = [t for t in _combos_idx
                  if _combo_label(t) not in _done_labels]

    with gzip.open(_OUT_PATH, "ab" if _header_written else "wb") as _fh:
        for _idx_t in tqdm(_remaining, total=len(_remaining),
                           desc="Ensemble sweep", unit="combo"):
            _w = _W_norm[_combo_to_row[_idx_t]]       # normalised weight vector (6,)
            _fold_maes, _fold_rhos = [], []
            for _P, _y in _fold_arrays:
                _ens = _P @ _w                        # (n_cmp,) — single combo
                _fold_maes.append(float(np.abs(_ens - _y).mean()))
                _fold_rhos.append(_spearman(_ens, _y))
            _row = pl.DataFrame([{
                "ensemble": _combo_label(_idx_t),
                **{f"w_{tag}": _W_vals[_idx_t[i]] for i, tag in enumerate(_TAGS)},
                "mae": float(np.mean(_fold_maes)),
                "rho": float(np.mean(_fold_rhos)),
            }])
            _row.write_csv(_fh, include_header=not _header_written)
            _header_written = True

    _sweep_df = pl.read_csv(_OUT_PATH)
    print(f"Done — {len(_sweep_df):,} combos in {_OUT_PATH}")

    from statsmodels.stats.multicomp import pairwise_tukeyhsd
    import pandas as _pd

    _top20 = _sweep_df.sort("mae").head(20)

    # ── Individual model summary ───────────────────────────────────────────────
    _indiv_records = []
    for _tag, _path, _col, _val in _MODELS:
        _df = pl.read_csv(_path).filter(pl.col(_col) == _val)
        _metrics = calc_regression_metrics(
            _df.rename({"fold": "cv_cycle"}).with_columns(pl.lit("random").alias("split")),
            cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0,
        )
        _indiv_records.append(
            _metrics.group_by("method")
            .agg(pl.col(["mae", "rho"]).mean())
            .with_columns(pl.lit(_tag).alias("model"))
            .select(["model", "mae", "rho"])
        )
    _indiv_summary = pl.concat(_indiv_records).sort("mae")

    # ── Per-model weight MCS heatmaps ─────────────────────────────────────────
    # For each model (cp, ch, rf, xg, mc, tf) group all ensemble combos by the
    # weight assigned to that model and run Tukey HSD on MAE across weight groups.
    # Each weight level becomes a "method" in the heatmap.

    _W_LABEL = {0.0: "0", 1/5: "1/5", 1/4: "1/4", 1/3: "1/3",
                1/2: "1/2", 1.0: "1", 2.0: "2", 3.0: "3", 4.0: "4", 5.0: "5"}

    def _w_label(v: float) -> str:
        # round to avoid float comparison issues
        return _W_LABEL.get(round(v, 6), str(round(v, 4)))

    _TAG_FULLNAME = {
        "cp": "Chemprop HPO",
        "ch": "CheMeleon HPO",
        "rf": "RF HPO (Mordred)",
        "xg": "XGBoost HPO (Mordred)",
        "mc": "Macau HPO (CheMeleon)",
        "tf": "TabPFN (CheMeleon)",
    }

    def _weight_mcs_ax(ax, tag: str) -> None:
        """Draw a Tukey HSD MCS heatmap for one model's weight levels."""
        _wcol = f"w_{tag}"
        _sweep_pd = _sweep_df.select([_wcol, "mae"]).to_pandas()
        _sweep_pd["w_label"] = _sweep_pd[_wcol].map(_w_label)

        # Sort groups by mean MAE ascending
        _order = (
            _sweep_pd.groupby("w_label")["mae"].mean()
            .sort_values().index.tolist()
        )
        _groups = _sweep_pd["w_label"].values
        _vals   = _sweep_pd["mae"].values

        _tukey = pairwise_tukeyhsd(_vals, _groups, alpha=0.05)
        _res = _pd.DataFrame(
            data    = _tukey._results_table.data[1:],
            columns = _tukey._results_table.data[0],
        )

        # Build symmetric mean-difference and p-value matrices
        _means = _sweep_pd.groupby("w_label")["mae"].mean().reindex(_order)
        _n     = len(_order)
        _diff  = _pd.DataFrame(0.0, index=_order, columns=_order)
        _pval  = _pd.DataFrame(1.0, index=_order, columns=_order)

        for _, row in _res.iterrows():
            g1, g2 = str(row["group1"]), str(row["group2"])
            if g1 in _order and g2 in _order:
                d = float(row["meandiff"])
                p = float(row["p-adj"])
                _diff.loc[g1, g2] = d
                _diff.loc[g2, g1] = -d
                _pval.loc[g1, g2] = p
                _pval.loc[g2, g1] = p

        mcs_plot(
            pc=_pval, effect_size=_diff, means=_means,
            ax=ax, show_diff=True,
            cell_text_size=8, axis_text_size=9,
            reverse_cmap=True,  # lower MAE = better = blue
            vlim=0.05,
        )
        ax.set_title(f"{_TAG_FULLNAME[tag]}  (weight effect on MAE)", fontsize=11)

    _PLOTS_DIR6 = Path("../plots/4_ml_optimization_2")
    _PLOTS_DIR6.mkdir(parents=True, exist_ok=True)

    _fig_w, _axes_w = plt.subplots(3, 2, figsize=(14, 22))
    for _ax, _tag in zip(_axes_w.flatten(), _TAGS):
        _weight_mcs_ax(_ax, _tag)
    _fig_w.suptitle("Weight sensitivity MCS heatmaps — MAE (lower = better, blue)",
                     fontsize=14)
    _fig_w.tight_layout()
    _fig_w.savefig(_PLOTS_DIR6 / "analysis6_ensemble_mcs_mae.png",
                   dpi=120, bbox_inches="tight")

    mo.vstack([
        mo.md("## Analysis 6 — Ensemble sweep results"),
        mo.md(f"**{_n_combos:,} ratio-distinct weight combinations evaluated**"),
        mo.md("### Individual model summary (sorted by MAE)"),
        mo.plain_text(_indiv_summary.to_pandas().to_string(index=False)),
        mo.md("### Top-20 ensembles by MAE"),
        mo.plain_text(_top20.to_pandas().to_string(index=False)),
        mo.md("### Weight sensitivity — MCS heatmaps per model (MAE)"),
        mo.md("Each heatmap shows whether changing the weight assigned to one model "
              "significantly changes ensemble MAE (averaged over all other weight combinations)."),
        mo.as_html(_fig_w),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # MPS performance degradation analysis

    Analyses inter-fold timing from the chemprop CLI and API log files to
    characterise how macOS MPS memory pressure causes stalls and whether the
    pattern varies across the day/night cycle or degrades over multi-day runs.

    **Definitions**
    - *Normal fold*: inter-fold gap ≤ 150 s (pure training time, no MPS stall)
    - *Stall*: inter-fold gap > 150 s (MPS memory reclaim blocking next fold)
    - *Stall rate*: fraction of transitions that are stalls in a given window
    """)
    return


@app.cell
def _(Path, pl):
    """
    Parse the chemprop CLI log into a DataFrame of inter-fold gaps.

    Each training block starts with 'Running in mode train' and ends at the
    'predict - test size' line within that block.  Training time is measured
    as train-start → predict-end, which is immune to inter-run idle gaps that
    would otherwise inflate the apparent time of the first fold after a break.
    A block belongs to CheMeleon if it contains 'Loading cached CheMeleon'.
    """
    from datetime import datetime

    _CLI_LOG  = Path("../logs/chemprop_cli.log")
    _STALL_THR = 150

    # ── split log into per-fold blocks ────────────────────────────────────────
    # Each record: (train_start, predict_end, model).
    # Using train_start → predict_end avoids contamination from inter-run idle
    # gaps that would inflate the apparent training time of the next fold.
    _folds: list[tuple[datetime, datetime, str]] = []
    if _CLI_LOG.exists():
        _lines = _CLI_LOG.read_text().splitlines()
        _block_lines: list[str] = []
        _block_start: datetime | None = None
        for _line in _lines:
            if "Running in mode 'train'" in _line:
                # flush previous block
                if _block_lines and _block_start:
                    _is_che = any("Loading cached CheMeleon" in bl for bl in _block_lines)
                    _model  = "chemeleon" if _is_che else "chemprop"
                    for _bl in _block_lines:
                        if "chemprop.cli.predict - test size" in _bl:
                            try:
                                _ts_end = datetime.fromisoformat(
                                    _bl.split(" - ")[0].strip().replace("T", " ")
                                )
                                _folds.append((_block_start, _ts_end, _model))
                            except Exception:
                                pass
                            break
                try:
                    _block_start = datetime.fromisoformat(
                        _line.split(" - ")[0].strip().replace("T", " ")
                    )
                except Exception:
                    _block_start = None
                _block_lines = [_line]
            else:
                _block_lines.append(_line)
        # flush last block
        if _block_lines and _block_start:
            _is_che = any("Loading cached CheMeleon" in bl for bl in _block_lines)
            _model  = "chemeleon" if _is_che else "chemprop"
            for _bl in _block_lines:
                if "chemprop.cli.predict - test size" in _bl:
                    try:
                        _ts_end = datetime.fromisoformat(
                            _bl.split(" - ")[0].strip().replace("T", " ")
                        )
                        _folds.append((_block_start, _ts_end, _model))
                    except Exception:
                        pass
                    break

    _t0 = _folds[0][0] if _folds else None
    _rows = []
    for _i, (_t_start, _t_end, _model) in enumerate(_folds):
        _train_s = (_t_end - _t_start).total_seconds()
        _rows.append({
            "fold_idx":  _i,
            "model":     _model,
            "t_start":   _t_start,
            "gap_s":     _train_s,
            "hour":      _t_start.hour,
            "date":      _t_start.date().isoformat(),
            "is_stall":  _train_s > _STALL_THR,
            "elapsed_h": (_t_start - _t0).total_seconds() / 3600,
        })

    timing_df = pl.DataFrame(_rows)
    _by_model = (
        timing_df.group_by("model")
        .agg([
            pl.len().alias("n_gaps"),
            pl.col("is_stall").sum().alias("n_stalls"),
            (pl.col("is_stall").mean() * 100).round(1).alias("stall_pct"),
        ])
        .sort("model")
    )
    print(timing_df.group_by("model").len())
    print(_by_model)
    return (timing_df,)


@app.cell
def _(Path, mo, np, pl, plt, sns, timing_df):
    """
    Three plots split by model (chemprop vs chemeleon), saved to plots/:
      1. Stall rate by hour of day
      2. Rolling stall rate over elapsed run time
      3. Gap duration distribution
    """
    _STALL_THR = 150
    _PLOTS_DIR = Path("../plots") / "4_ml_optimization_2"
    _PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    _MODELS   = ["chemprop", "chemeleon"]
    _COLORS   = {"chemprop": "tab:blue", "chemeleon": "tab:orange"}
    _LABELS   = {"chemprop": "Chemprop (scratch)", "chemeleon": "CheMeleon"}

    sns.set_style("whitegrid")
    sns.set_context("notebook", font_scale=1.2)

    # ── per-model data prep ───────────────────────────────────────────────────
    def _hourly_df(df: pl.DataFrame) -> "pd.DataFrame":
        return (
            df.group_by("hour")
            .agg((pl.col("is_stall").mean() * 100).alias("stall_pct"))
            .sort("hour")
            .to_pandas()
        )

    def _rolling(df: pl.DataFrame, win: float = 2.0) -> "pd.DataFrame":
        _eh = df["elapsed_h"].to_numpy()
        _st = df["is_stall"].to_numpy().astype(float)
        _rows = []
        for _i, _e in enumerate(_eh):
            _m = (_eh >= _e - win) & (_eh <= _e)
            if _m.sum() >= 3:
                _rows.append({"elapsed_h": _e, "roll_pct": float(_st[_m].mean() * 100)})
        return pl.DataFrame(_rows).sort("elapsed_h").to_pandas() if _rows else None

    _subsets = {m: timing_df.filter(pl.col("model") == m) for m in _MODELS}

    # ── plot 1: stall rate by hour (chemprop only) ───────────────────────────
    _fig1, _ax1 = plt.subplots(figsize=(10, 5))
    _h = _hourly_df(_subsets["chemprop"])
    _ax1.plot(_h["hour"], _h["stall_pct"], marker="o", color=_COLORS["chemprop"])
    _ax1.fill_between(_h["hour"], _h["stall_pct"], alpha=0.2, color=_COLORS["chemprop"])
    _ax1.axhspan(0, 30, alpha=0.07, color="green")
    _ax1.text(0.5, 25, "target zone (≤30%)", color="green", alpha=0.7,
              transform=_ax1.get_yaxis_transform(), fontsize=9)
    _ax1.set_xlabel("Hour of day")
    _ax1.set_ylabel("Stall rate (%)")
    _ax1.set_title("Chemprop (scratch) — stall rate by hour of day")
    _ax1.set_xticks(range(0, 24, 2))
    _ax1.set_ylim(0, 105)
    _fig1.tight_layout()
    _fig1.savefig(_PLOTS_DIR / "mps_stall_by_hour.png", dpi=150, bbox_inches="tight")
    plt.close(_fig1)

    # ── plot 2: rolling stall rate over elapsed run time (chemprop only) ─────
    _fig2, _ax2 = plt.subplots(figsize=(10, 5))
    _r = _rolling(_subsets["chemprop"])
    if _r is not None:
        _ax2.plot(_r["elapsed_h"], _r["roll_pct"], color=_COLORS["chemprop"], alpha=0.85)
    _ax2.axhline(50, color="red", linestyle="--", alpha=0.5, label="50% threshold")
    _ax2.set_xlabel("Elapsed run time (hours)")
    _ax2.set_ylabel("Rolling stall rate % (2 h window)")
    _ax2.set_title("Chemprop (scratch) — performance degradation over run time")
    _ax2.set_ylim(0, 105)
    _ax2.legend()
    _fig2.tight_layout()
    _fig2.savefig(_PLOTS_DIR / "mps_stall_over_time.png", dpi=150, bbox_inches="tight")
    plt.close(_fig2)

    # ── plot 2b: individual fold times over elapsed run time ─────────────────
    _fig2b, _ax2b = plt.subplots(figsize=(12, 5))
    _df_cp = _subsets["chemprop"].to_pandas()
    _ax2b.scatter(
        _df_cp["elapsed_h"], _df_cp["gap_s"] / 60,
        color=_COLORS["chemprop"], alpha=0.5, s=18,
    )
    _ax2b.axhline(_STALL_THR / 60, color="red",    linestyle="--", alpha=0.6, label=f"stall threshold ({_STALL_THR} s)")
    _ax2b.axhline(15,              color="gray",   linestyle=":",  alpha=0.6, label="15 min")
    _ax2b.axhline(30,              color="olive",  linestyle=":",  alpha=0.6, label="30 min")
    _ax2b.axhline(60,              color="purple", linestyle=":",  alpha=0.6, label="1 h")
    _ax2b.set_yscale("log")
    _ax2b.set_xlabel("Elapsed run time (hours)")
    _ax2b.set_ylabel("Fold wall-clock time (min, log scale)")
    _ax2b.set_title("Chemprop (scratch) — individual fold times over elapsed run time")
    _ax2b.legend(loc="upper left", fontsize=9)
    _fig2b.tight_layout()
    _fig2b.savefig(_PLOTS_DIR / "mps_fold_times_scatter.png", dpi=150, bbox_inches="tight")
    plt.close(_fig2b)

    # ── plot 3: gap distribution (log-log) ────────────────────────────────────
    _all_gaps = timing_df["gap_s"].to_numpy()
    _bins = np.logspace(np.log10(_all_gaps.min() + 1), np.log10(_all_gaps.max()), 50)
    _fig3, _ax3 = plt.subplots(figsize=(10, 5))
    for _m in _MODELS:
        _gaps_m = _subsets[_m]["gap_s"].to_numpy()
        _ax3.hist(_gaps_m, bins=_bins, color=_COLORS[_m], alpha=0.6,
                  label=_LABELS[_m], log=True)
    _ax3.set_xscale("log")
    _ax3.axvline(_STALL_THR, color="red",    linestyle="--", alpha=0.7, label=f"stall threshold ({_STALL_THR} s)")
    _ax3.axvline(1_800,      color="olive",  linestyle=":",  alpha=0.7, label="30 min")
    _ax3.axvline(3_600,      color="purple", linestyle=":",  alpha=0.7, label="1 h")
    _ax3.axvline(10_800,     color="brown",  linestyle=":",  alpha=0.7, label="3 h")
    _ax3.set_xlabel("Inter-fold gap (s, log scale)")
    _ax3.set_ylabel("Count (log scale)")
    _ax3.set_title("Chemprop CLI — distribution of inter-fold gaps")
    _ax3.legend()
    _fig3.tight_layout()
    _fig3.savefig(_PLOTS_DIR / "mps_gap_distribution.png", dpi=150, bbox_inches="tight")
    plt.close(_fig3)

    # ── summary table per model ───────────────────────────────────────────────
    _summary_rows = []
    for _m in _MODELS:
        _df_m   = _subsets[_m]
        _normal = _df_m.filter(~pl.col("is_stall"))["gap_s"]
        _stalls = _df_m.filter(pl.col("is_stall"))["gap_s"]
        _summary_rows.append({
            "model":        _m,
            "n_folds":      len(_df_m) + 1,
            "n_stalls":     int(_df_m["is_stall"].sum()),
            "stall_pct":    round(100 * float(_df_m["is_stall"].mean()), 1),
            "normal_avg_s": round(float(_normal.mean()), 0) if len(_normal) else None,
            "stall_avg_s":  round(float(_stalls.mean()), 0) if len(_stalls) else None,
            "stall_max_s":  round(float(_stalls.max()),  0) if len(_stalls) else None,
        })
    _summary = pl.DataFrame(_summary_rows)

    mo.vstack([
        mo.md("## CLI MPS stall summary"),
        mo.plain_text(_summary.to_pandas().to_string(index=False)),
        mo.md("### Stall rate by hour of day"),
        mo.image(str(_PLOTS_DIR / "mps_stall_by_hour.png")),
        mo.md("### Performance degradation over run time"),
        mo.image(str(_PLOTS_DIR / "mps_stall_over_time.png")),
        mo.md("### Individual fold times over elapsed run time"),
        mo.image(str(_PLOTS_DIR / "mps_fold_times_scatter.png")),
        mo.md("### Inter-fold gap distribution"),
        mo.image(str(_PLOTS_DIR / "mps_gap_distribution.png")),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Submissions

    Train each model on the full dose-response training set and predict the
    513 held-out test compounds.

    **Individual submissions**

    | File | Model |
    |------|-------|
    | `4_tabpfn_chemeleon_submission.csv` | TabPFN on CheMeleon FP (default params) |
    | `4_chemeleon_hpo_submission.csv` | CheMeleon HPO best params |
    | `4_macau_che_hpo_submission.csv` | Macau HPO best params (CheMeleon FP) |

    **Ensemble submissions** (weighted average of cp / ch / xg / mc / tf test predictions)

    | File | Weights (cp · ch · rf · xg · mc · tf) |
    |------|---------------------------------------|
    | `4_ens_cp4_ch5_rf0_xg1_mc0_tf5_submission.csv` | 4 · 5 · 0 · 1 · 0 · 5 |
    | `4_ens_cp4_ch5_rf0_xg1_mc1_tf5_submission.csv` | 4 · 5 · 0 · 1 · 1 · 5 |
    | `4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv` | 5 · 5 · 0 · ⅓ · 1 · 5 |
    """)
    return


@app.cell
def _(
    BoostedTreesModel,
    ChempropChemeleonModel,
    ChempropModel,
    MacauModel,
    Path,
    best_params,
    chemeleon_embed,
    extract_fp_matrix,
    generate_fingerprint,
    mo,
    np,
    pl,
    subprocess,
    sys,
    tempfile,
):
    """
    Train all models on the full training set and generate test-set predictions.
    Each model is skipped if its output file already exists.
    """
    _TARGET_COL = "pEC50_dr"
    _SEED       = 42
    _SUB_DIR    = Path("../submissions")
    _SUB_DIR.mkdir(parents=True, exist_ok=True)
    _tmp        = Path(tempfile.gettempdir())
    _TABPFN_SCRIPT = _tmp / "sub_tabpfn.py"

    # ── Load datasets ─────────────────────────────────────────────────────────
    _train_full = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col(_TARGET_COL).is_not_null())
        .select(["smiles", "inchikey", "molecule_names", _TARGET_COL])
    )
    _test_df = pl.read_csv("../data/raw/20260409/dose_response_test.csv")
    _test_smiles = _test_df["SMILES"].to_list()
    _test_names  = _test_df["Molecule Name"].to_list()

    # 10 % val split for models that need early stopping
    _rng      = np.random.default_rng(_SEED)
    _n        = len(_train_full)
    _val_idx  = _rng.choice(_n, size=int(_n * 0.1), replace=False)
    _tr_idx   = np.setdiff1d(np.arange(_n), _val_idx)
    _train_sub = _train_full[_tr_idx.tolist()]
    _val_sub   = _train_full[_val_idx.tolist()]

    def _save_submission(path: Path, preds: np.ndarray) -> None:
        pl.DataFrame({
            "SMILES":        _test_smiles,
            "Molecule Name": _test_names,
            "pEC50":         preds.tolist(),
        }).write_csv(path)
        print(f"Saved {len(preds)} predictions → {path.name}")

    # ── 1. TabPFN on CheMeleon FP ─────────────────────────────────────────────
    _tfn_path = _SUB_DIR / "4_tabpfn_chemeleon_submission.csv"
    if _tfn_path.exists():
        print(f"tabpfn_chemeleon: already exists — skipping.")
    else:
        _TABPFN_SCRIPT.write_text("\n".join([
            "import os, sys, numpy as np",
            "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
            "from dotenv import load_dotenv; from pathlib import Path",
            "load_dotenv(Path('.env'))",
            "import torch",
            "torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))",
            "from tabpfn import TabPFNRegressor",
            "X_train = np.load(sys.argv[1])",
            "y_train = np.load(sys.argv[2])",
            "X_test  = np.load(sys.argv[3])",
            "out     = sys.argv[4]",
            "model = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=True, device='cpu')",
            "model.fit(X_train, y_train)",
            "np.save(out, model.predict(X_test))",
        ]))

        _Xtr_che, _Xte_che = chemeleon_embed(
            _train_full["smiles"].to_list(), _test_smiles, prefix="sub_tfn",
        )
        _ytr = _train_full[_TARGET_COL].to_numpy()

        _f_Xtr = _tmp / "sub_Xtr.npy"; _f_ytr = _tmp / "sub_ytr.npy"
        _f_Xte = _tmp / "sub_Xte.npy"; _f_out = _tmp / "sub_preds"
        np.save(str(_f_Xtr), _Xtr_che); np.save(str(_f_ytr), _ytr)
        np.save(str(_f_Xte), _Xte_che)
        _res = subprocess.run(
            [sys.executable, str(_TABPFN_SCRIPT),
             str(_f_Xtr), str(_f_ytr), str(_f_Xte), str(_f_out)],
            capture_output=True, text=True, cwd=str(Path("../").resolve()),
        )
        if _res.returncode != 0:
            raise RuntimeError(f"TabPFN failed:\n{_res.stderr}")
        _tfn_preds = np.load(str(_f_out) + ".npy")
        for _p in [_f_Xtr, _f_ytr, _f_Xte, Path(str(_f_out) + ".npy")]:
            _p.unlink(missing_ok=True)
        _save_submission(_tfn_path, _tfn_preds)

    # ── 2. CheMeleon HPO ──────────────────────────────────────────────────────
    _ch_path = _SUB_DIR / "4_chemeleon_hpo_submission.csv"
    if _ch_path.exists():
        print(f"chemeleon_hpo: already exists — skipping.")
    else:
        _chemeleon_params = {
            k: v for k, v in {**best_params, "epochs": 50}.items()
            if k not in ("message_hidden_dim", "depth")
        }
        _ch_model = ChempropChemeleonModel(pred_type="regression", **_chemeleon_params)
        _ch_model.train(
            _train_sub["smiles"].to_list(), _train_sub[_TARGET_COL].to_numpy(),
            _val_sub["smiles"].to_list(),   _val_sub[_TARGET_COL].to_numpy(),
            target_col=_TARGET_COL,
        )
        _save_submission(_ch_path, _ch_model.predict(_test_smiles))
        del _ch_model

    # ── 3. Macau HPO on CheMeleon FP ─────────────────────────────────────────
    _mac_path = _SUB_DIR / "4_macau_che_hpo_submission.csv"
    if _mac_path.exists():
        print(f"macau_che_hpo: already exists — skipping.")
    else:
        import optuna as _optuna
        _optuna.logging.set_verbosity(_optuna.logging.WARNING)
        _mac_study = _optuna.load_study(
            study_name="macau_chemeleon_hpo",
            storage="sqlite:///../predictions/4_hpo_macau_chemeleon.db",
        )
        _mac_best = _mac_study.best_params
        _Xtr_mac, _Xte_mac = chemeleon_embed(
            _train_full["smiles"].to_list(), _test_smiles, prefix="sub_mac",
        )
        _mac_model = MacauModel(seed=_SEED, **_mac_best)
        _mac_model.train(_Xtr_mac, _train_full[_TARGET_COL].to_numpy())
        _save_submission(_mac_path, _mac_model.predict(_Xte_mac))
        del _mac_model

    # ── 4. Ensemble test predictions — train cp, ch, xg, mc, tf ─────────────
    # Individual test predictions for each ensemble component
    _ens_preds: dict[str, np.ndarray] = {}

    # cp — Chemprop HPO
    _cp_cache = _tmp / "sub_ens_cp.npy"
    if _cp_cache.exists():
        _ens_preds["cp"] = np.load(str(_cp_cache))
    else:
        _cp_model = ChempropModel(pred_type="regression", **{**best_params, "epochs": 50})
        _cp_model.train(
            _train_sub["smiles"].to_list(), _train_sub[_TARGET_COL].to_numpy(),
            _val_sub["smiles"].to_list(),   _val_sub[_TARGET_COL].to_numpy(),
            target_col=_TARGET_COL,
        )
        _ens_preds["cp"] = _cp_model.predict(_test_smiles)
        np.save(str(_cp_cache), _ens_preds["cp"])
        del _cp_model

    # ch — CheMeleon HPO (reuse if already trained above)
    _ch_cache = _tmp / "sub_ens_ch.npy"
    if _ch_cache.exists():
        _ens_preds["ch"] = np.load(str(_ch_cache))
    else:
        _ch2 = ChempropChemeleonModel(pred_type="regression", **_chemeleon_params)
        _ch2.train(
            _train_sub["smiles"].to_list(), _train_sub[_TARGET_COL].to_numpy(),
            _val_sub["smiles"].to_list(),   _val_sub[_TARGET_COL].to_numpy(),
            target_col=_TARGET_COL,
        )
        _ens_preds["ch"] = _ch2.predict(_test_smiles)
        np.save(str(_ch_cache), _ens_preds["ch"])
        del _ch2

    # xg — XGBoost HPO on Mordred
    _xg_cache = _tmp / "sub_ens_xg.npy"
    if _xg_cache.exists():
        _ens_preds["xg"] = np.load(str(_xg_cache))
    else:
        import optuna as _optuna2
        _optuna2.logging.set_verbosity(_optuna2.logging.WARNING)
        _xgb_best = _optuna2.load_study(
            study_name="xgb_mordred_hpo",
            storage="sqlite:///../predictions/4_hpo_xgb_mordred.db",
        ).best_params
        _fp_tr = generate_fingerprint(_train_sub, "mordred")
        _fp_va = generate_fingerprint(_val_sub, "mordred")
        _fp_te = generate_fingerprint(
            pl.DataFrame({"smiles": _test_smiles, "inchikey": _test_names,
                          "molecule_names": _test_names}),
            "mordred",
        )
        _Xtr_xg = extract_fp_matrix(_fp_tr, "mordred")
        _Xva_xg = extract_fp_matrix(_fp_va, "mordred")
        _Xte_xg = extract_fp_matrix(_fp_te, "mordred")
        del _fp_tr, _fp_va, _fp_te
        if np.isnan(_Xtr_xg).any():
            _valid = ~np.isnan(_Xtr_xg).any(axis=0)
            _Xtr_xg = _Xtr_xg[:, _valid]
            _Xva_xg = _Xva_xg[:, _valid]
            _Xte_xg = _Xte_xg[:, _valid]
        _xg_model = BoostedTreesModel(pred_type="regression", **_xgb_best)
        _xg_model.train(
            _Xtr_xg, _train_sub[_TARGET_COL].to_numpy(),
            _Xva_xg, _val_sub[_TARGET_COL].to_numpy(),
        )
        _ens_preds["xg"] = _xg_model.predict(_Xte_xg)
        np.save(str(_xg_cache), _ens_preds["xg"])
        del _xg_model, _Xtr_xg, _Xva_xg, _Xte_xg

    # mc — Macau HPO on CheMeleon FP
    _mc_cache = _tmp / "sub_ens_mc.npy"
    if _mc_cache.exists():
        _ens_preds["mc"] = np.load(str(_mc_cache))
    else:
        _Xtr_mc, _Xte_mc = chemeleon_embed(
            _train_full["smiles"].to_list(), _test_smiles, prefix="sub_mc",
        )
        _mc_ens = MacauModel(seed=_SEED, **_mac_best)
        _mc_ens.train(_Xtr_mc, _train_full[_TARGET_COL].to_numpy())
        _ens_preds["mc"] = _mc_ens.predict(_Xte_mc)
        del _mc_ens, _Xtr_mc, _Xte_mc
        np.save(str(_mc_cache), _ens_preds["mc"])

    # tf — TabPFN on CheMeleon FP
    _tf_cache = _tmp / "sub_ens_tf.npy"
    if _tf_cache.exists():
        _ens_preds["tf"] = np.load(str(_tf_cache))
    else:
        _Xtr_tf, _Xte_tf = chemeleon_embed(
            _train_full["smiles"].to_list(), _test_smiles, prefix="sub_tf",
        )
        _ytr_tf = _train_full[_TARGET_COL].to_numpy()
        _f_Xtr2 = _tmp / "sub2_Xtr.npy"; _f_ytr2 = _tmp / "sub2_ytr.npy"
        _f_Xte2 = _tmp / "sub2_Xte.npy"; _f_out2 = _tmp / "sub2_preds"
        np.save(str(_f_Xtr2), _Xtr_tf); np.save(str(_f_ytr2), _ytr_tf)
        np.save(str(_f_Xte2), _Xte_tf)
        _res_tf = subprocess.run(
            [sys.executable, str(_TABPFN_SCRIPT),
             str(_f_Xtr2), str(_f_ytr2), str(_f_Xte2), str(_f_out2)],
            capture_output=True, text=True, cwd=str(Path("../").resolve()),
        )
        if _res_tf.returncode != 0:
            raise RuntimeError(f"TabPFN failed:\n{_res_tf.stderr}")
        _ens_preds["tf"] = np.load(str(_f_out2) + ".npy")
        for _p in [_f_Xtr2, _f_ytr2, _f_Xte2, Path(str(_f_out2) + ".npy")]:
            _p.unlink(missing_ok=True)
        np.save(str(_tf_cache), _ens_preds["tf"])

    # ── 5. Write ensemble submissions ─────────────────────────────────────────
    _W_MAP = {"0": 0.0, "15": 1/5, "14": 1/4, "13": 1/3, "12": 1/2,
              "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
    _ENS_TAGS = ["cp", "ch", "rf", "xg", "mc", "tf"]

    for _ens_label in [
        "cp4_ch5_rf0_xg1_mc0_tf5",
        "cp4_ch5_rf0_xg1_mc1_tf5",
        "cp5_ch5_rf0_xg13_mc1_tf5",
    ]:
        _ens_path = _SUB_DIR / f"4_ens_{_ens_label}_submission.csv"
        if _ens_path.exists():
            print(f"{_ens_path.name}: already exists — skipping.")
            continue
        _parts_ens = _ens_label.split("_")
        _w_raw = {tag: _W_MAP[p[len(tag):]] for tag, p in zip(_ENS_TAGS, _parts_ens)}
        _total = sum(_w_raw.values())
        _ens_pred = sum(
            _ens_preds[tag] * (w / _total)
            for tag, w in _w_raw.items()
            if w > 0
        )
        _save_submission(_ens_path, _ens_pred)

    mo.md("## Submissions generated — see validation cell below.")
    return


@app.cell
def _(Path, mo, np, pd):
    """
    Validate all submission files produced in this notebook.

    Rules (matching the OpenADMET activity_validation.py spec):
      - Required columns: SMILES, Molecule Name, pEC50
      - No missing identifiers or duplicate Molecule Names
      - pEC50 must be numeric and finite
      - Exactly 513 rows
    """
    _ACTIVITY_DATASET_SIZE = 513
    _SUB_DIR = Path("../submissions")

    _SUBMISSION_FILES = [
        _SUB_DIR / "4_tabpfn_chemeleon_submission.csv",
        _SUB_DIR / "4_chemeleon_hpo_submission.csv",
        _SUB_DIR / "4_macau_che_hpo_submission.csv",
        _SUB_DIR / "4_ens_cp4_ch5_rf0_xg1_mc0_tf5_submission.csv",
        _SUB_DIR / "4_ens_cp4_ch5_rf0_xg1_mc1_tf5_submission.csv",
        _SUB_DIR / "4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv",
    ]

    def _validate(path: Path) -> tuple[bool, list[str]]:
        errors: list[str] = []
        if not path.exists():
            return False, [f"File does not exist: {path}"]
        try:
            df = pd.read_csv(path)
        except Exception as exc:
            return False, [f"Error reading CSV: {exc}"]

        for col in ("SMILES", "Molecule Name", "pEC50"):
            if col not in df.columns:
                errors.append(f"Missing required column: '{col}'")
        if errors:
            return False, errors

        if df.empty:
            return False, ["Submission is empty."]

        if df[["SMILES", "Molecule Name"]].isna().any(axis=1).sum():
            errors.append("Row(s) with missing identifier values.")

        if df["Molecule Name"].duplicated().sum():
            errors.append(f"{df['Molecule Name'].duplicated().sum()} duplicated Molecule Name(s).")

        _numeric = pd.to_numeric(df["pEC50"], errors="coerce")
        if _numeric.isna().sum():
            errors.append(f"pEC50 has {_numeric.isna().sum()} non-numeric value(s).")
        elif not np.isfinite(_numeric.to_numpy()).all():
            errors.append(f"pEC50 has non-finite value(s).")

        if len(df) != _ACTIVITY_DATASET_SIZE:
            errors.append(f"{len(df)} rows, expected {_ACTIVITY_DATASET_SIZE}.")

        return len(errors) == 0, errors

    _results = []
    for _path in _SUBMISSION_FILES:
        _ok, _errs = _validate(_path)
        if _ok:
            _results.append(mo.md(f"✓ **{_path.name}** — passed"))
        else:
            _results.append(mo.md(
                f"✗ **{_path.name}** — FAILED:\n" + "\n".join(f"- {e}" for e in _errs)
            ))

    mo.vstack(_results)
    return


if __name__ == "__main__":
    app.run()
