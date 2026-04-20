import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Imports
    """)
    return


@app.cell
def _():
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

    return (
        AnovaRM,
        AtomPairFingerprint,
        AvalonFingerprint,
        BaseKFold,
        Chem,
        ConformerGenerator,
        DataStructs,
        E3FPFingerprint,
        ECFPFingerprint,
        ExplicitBitVect,
        Iterable,
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
        RandomForestClassifier,
        RandomForestRegressor,
        TopologicalTorsionFingerprint,
        accuracy_score,
        balanced_accuracy_score,
        f1_score,
        gc,
        gzip,
        math,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        mo,
        mpatches,
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
        roc_auc_score,
        shutil,
        sns,
        spearmanr,
        stats,
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
    | `ChempropModel` | Chemprop v2 MPNN from scratch | SMILES column |
    | `ChempropChemeleonModel` | Chemprop v2 fine-tuned from [CheMeleon](https://github.com/JacksonBurns/chemeleon) backbone | SMILES column |
    | `MeanBaseline` | Predict all values as mean pEC50 on train set | None |
    | `NearestNeighbourBaseline` | Predict value as pEC50 of NN in train set | fingerprint column |

    `task` is either `"regression"` or `"classification"`.
    Classification `predict()` returns the probability of the positive class.

    The last two models are not expected to provide good performance, but can be a useful
    way to obtain a worst-case MAE value.
    The `MeanBaseline` model should provide an R² of zero and a rho value of NaN.
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

        def __init__(self, pred_type: str = "classification") -> None:
            """
            Args:
                pred_type: "classification" (RandomForestClassifier) or
                    "regression" (RandomForestRegressor).

            Raises:
                ValueError: If pred_type is not "classification" or "regression".
            """
            self.model = None
            self.pred_type = pred_type
            if pred_type == "classification":
                self.model = RandomForestClassifier(n_jobs=-1)
            elif pred_type == "regression":
                self.model = RandomForestRegressor(n_jobs=-1)
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

        def __init__(self, pred_type: str = "classification") -> None:
            """
            Args:
                pred_type: "classification" (XGBClassifier) or "regression" (XGBRegressor).

            Raises:
                ValueError: If pred_type is not "classification" or "regression".
            """
            self.model = None
            self.pred_type = pred_type
            if pred_type == "classification":
                self.model = xgb.XGBClassifier(tree_method="hist", 
                                               early_stopping_rounds=2,
                                               n_jobs=-1)
            elif pred_type == "regression":
                self.model = xgb.XGBRegressor(tree_method="hist", 
                                              early_stopping_rounds=2,
                                              n_jobs=-1)
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
def _(Optional, np):
    class MeanBaseline:
        """
        Non-ML baseline that predicts the training-set mean for every test compound.

        This is the simplest possible baseline: it ignores all molecular structure
        and returns a constant prediction equal to the mean of the training labels.
        Any useful model must beat this.
        """

        def __init__(self) -> None:
            self._mean: Optional[float] = None

        def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """
            Store the training-set mean.

            Args:
                X_train: Feature matrix (unused — kept for API compatibility).
                y_train: Training labels or values.
            """
            self._mean = float(np.mean(y_train))

        def predict(self, X_test: np.ndarray) -> np.ndarray:
            """
            Return the training-set mean for every test compound.

            Args:
                X_test: Test feature matrix (only shape is used).

            Returns:
                1-D array of length n_test filled with the training mean.
            """
            return np.full(X_test.shape[0], self._mean, dtype=np.float32)

    class NearestNeighbourBaseline:
        """
        Simple ML baseline that predicts the training label of the most similar
        training compound (1-NN regression by Tanimoto similarity).

        Fingerprints are treated as binary vectors. Tanimoto similarity is computed
        efficiently with matrix operations:

            T(a, b) = |a ∩ b| / |a ∪ b|  =  dot(a, b) / (|a| + |b| - dot(a, b))

        where |a| = sum of set bits = dot(a, a).
        """

        def __init__(self) -> None:
            self._X_train: Optional[np.ndarray] = None
            self._y_train: Optional[np.ndarray] = None

        def train(self, X_train: np.ndarray, y_train: np.ndarray) -> None:
            """
            Store the training fingerprints and labels.

            Args:
                X_train: Binary fingerprint matrix of shape (n_train, fp_size).
                y_train: Training target values of shape (n_train,).
            """
            self._X_train = X_train.astype(np.float32)
            self._y_train = y_train.astype(np.float32)

        def predict(self, X_test: np.ndarray, chunk_size: int = 64) -> np.ndarray:
            """
            Return the training label of the nearest neighbour for each test compound.

            Tanimoto similarity is computed in row-chunks of test compounds so that
            only a (chunk_size × n_train) matrix exists in memory at any time, rather
            than the full (n_test × n_train) matrix.

            Args:
                X_test: Binary fingerprint matrix of shape (n_test, fp_size).
                chunk_size: Number of test compounds processed per chunk.

            Returns:
                1-D array of shape (n_test,) with the nearest-neighbour predictions.
            """
            X_test = X_test.astype(np.float32)
            train_counts = self._X_train.sum(axis=1)  # (n_train,) — computed once
            nn_idx = np.empty(X_test.shape[0], dtype=np.intp)

            for start in range(0, X_test.shape[0], chunk_size):
                chunk = X_test[start : start + chunk_size]
                dot = chunk @ self._X_train.T                           # (chunk, n_train)
                test_counts = chunk.sum(axis=1)                         # (chunk,)
                union = test_counts[:, None] + train_counts[None, :] - dot
                tanimoto = np.where(union > 0, dot / union, 0.0)
                nn_idx[start : start + chunk_size] = np.argmax(tanimoto, axis=1)

            return self._y_train[nn_idx]

    return MeanBaseline, NearestNeighbourBaseline


@app.cell
def _(Optional, Path, np, pl, shutil, subprocess, sys, tempfile, torch):
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

    # Single log file for all chemprop CLI calls — appended across folds.
    _CHEMPROP_LOG = Path(tempfile.gettempdir()) / "chemprop_cli.log"

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
        with open(_CHEMPROP_LOG, "a") as _log:
            _log.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\n{'='*60}\n")
            result = subprocess.run(cmd, stdout=_log, stderr=_log, text=True)
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
        """

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _CHEMPROP_MODEL_DIR,
            epochs: int = 50,
        ) -> None:
            """
            Args:
                pred_type: "regression" or "classification".
                model_dir: Directory where the CLI writes model checkpoints.
                    Reused (overwritten) on every train() call.
                epochs: Maximum number of training epochs.
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type = pred_type
            self.model_dir = model_dir
            self.epochs    = epochs
            self.target_col: Optional[str] = None  # set during train()

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "target",
        ) -> None:
            """
            Train the model by calling `chemprop train` via subprocess.

            Writes temporary CSV files for train and val sets, runs the CLI,
            then removes the CSVs. The model directory is cleared before each
            run so old checkpoints do not accumulate.

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

            # Remove stale checkpoints so the CLI starts fresh each fold
            if self.model_dir.exists():
                shutil.rmtree(self.model_dir)

            task_type = "regression" if self.pred_type == "regression" else "binary"
            # Pass val_csv twice (as val and as dummy test) so the CLI tracks
            # val_loss for early stopping. Two-file mode triggers a validation
            # error unless --split-sizes is also set.
            _run_chemprop_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _get_device(),
                "--epochs", str(self.epochs),
                "--save-dir", str(self.model_dir),
            ])

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
        to `chemprop train`.  The CLI downloads and caches the CheMeleon weights
        automatically at ~/.chemprop/chemeleon_mp.pt on the first call.

        Reference: https://github.com/JacksonBurns/chemeleon
        """

        def __init__(
            self,
            pred_type: str = "regression",
            model_dir: Path = _CHEMELEON_MODEL_DIR,
            epochs: int = 50,
        ) -> None:
            """
            Args:
                pred_type: "regression" or "classification".
                model_dir: Directory where the CLI writes model checkpoints.
                    Distinct from ChempropModel's default to avoid collisions.
                epochs: Maximum number of training epochs.
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type  = pred_type
            self.model_dir  = model_dir
            self.epochs     = epochs
            self.target_col: Optional[str] = None

        def train(
            self,
            X_train: list[str],
            y_train: np.ndarray,
            X_val:   list[str],
            y_val:   np.ndarray,
            target_col: str = "target",
        ) -> None:
            """
            Fine-tune from CheMeleon by calling `chemprop train --from-foundation CHEMELEON`.

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

            task_type = "regression" if self.pred_type == "regression" else "binary"
            # Pass val_csv twice (as val and as dummy test) — same reason as ChempropModel.
            _run_chemprop_cli([
                "train",
                "--data-path", str(train_csv), str(val_csv), str(val_csv),
                "--smiles-columns", "smiles",
                "--target-columns", target_col,
                "--task-type", task_type,
                "--accelerator", _get_device(),
                "--epochs", str(self.epochs),
                "--from-foundation", "CHEMELEON",
                "--save-dir", str(self.model_dir),
            ])

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
        Generate molecular fingerprints using scikit-fingerprints and add them as a column.

        Dispatches to the appropriate skfp fingerprint class based on fingerprint_type.
        For fingerprint types that require 3D conformers (e.g., E3FP), conformers are
        generated automatically via RDKit ETKDGv3.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported types: "ecfp"/"morgan", "maccs",
                "torsion", "rdkit", "atompair", "avalon", "e3fp", "mordred", "mqn",
                "pubchem".
            **kwargs: Additional keyword arguments forwarded to the skfp fingerprint class
                constructor (e.g., radius=3, n_bits=1024 for ECFP).

        Returns:
            DataFrame with an added column named after fingerprint_type containing
            the computed fingerprint arrays.

        Raises:
            ValueError: If fingerprint_type is not a recognized key.
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
    ## Data splitting utilities

    Three complementary CV strategies are implemented here to assess how well models
    generalise across different notions of "unseen" data:

    | Strategy | What it tests |
    |---|---|
    | **Random** | Baseline — molecules are shuffled at random across folds |
    | **Scaffold** | Chemical generalisation — folds are split by Bemis–Murcko scaffold so the model never sees a scaffold at test time that it trained on |
    | **Temporal** | Prospective generalisation — molecules are ordered by their numeric ID (a proxy for acquisition time)  |

    Random and scaffold CV share the nested generator interface:
    `(fold_index, outer_index, inner_index, train_df, val_df, test_df)`

    Temporal CV uses a simpler walk-forward interface:
    `(fold_index, train_df, val_df, test_df)`
    """)
    return


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
        Return the canonical Bemis–Murcko scaffold SMILES for a molecule.

        Molecules that fail to parse or have no ring system return an empty string,
        which causes them to be pooled together into a single "no scaffold" group.

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
        Generate nested 5×5 CV splits using **Bemis–Murcko scaffold** assignment.

        All molecules sharing the same scaffold are kept in the same fold, so the
        model never encounters a scaffold at test time that appeared during training.
        This gives a more realistic estimate of performance on genuinely novel
        chemical series.

        Scaffold groups are shuffled (but not ordered by size) before fold assignment
        so that very large scaffolds are randomly distributed.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            n_outer: Number of outer CV folds.
            n_inner: Number of inner CV folds per outer iteration.
            seed: Random seed for GroupKFoldShuffle.
            p_val: Fraction of the training set reserved as a validation split.

        Yields:
            Tuples of (fold_index, outer_index, inner_index, train_df, val_df, test_df).
        """
        # Compute scaffold label for every molecule once (reused across all folds).
        scaffolds = [_get_bemis_murcko_scaffold(s) for s in df["smiles"].to_list()]
        # Map each unique scaffold SMILES to an integer group id.
        unique_scaffolds = list(dict.fromkeys(scaffolds))  # preserves first-seen order
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
def _(Iterator, np, pl, split_dataset_random):
    def _extract_molecule_number(name: str) -> int:
        """
        Extract the numeric portion from a molecule name used as a temporal proxy.

        The molecule_names column contains identifiers such as "OADMET-0003144",
        where the numeric field after the dash encodes acquisition order.
        Molecules with an empty or unparsable name return 0 and are sorted
        to the earliest positions.

        Args:
            name: Molecule identifier string (e.g. "OADMET-0003144").

        Returns:
            Integer index representing acquisition order.
        """
        try:
            return int(name.split("-")[1])
        except (IndexError, ValueError):
            return 0

    def generate_cv_splits_temporal(
        df: pl.DataFrame,
        n_folds: int = 5,
        seed: int = 42,
        p_val: float = 0,
    ) -> Iterator:
        """
        Generate CV splits by chunking molecules in temporal order.

        Molecules are sorted by the numeric component of their molecule_names value
        (e.g. "0003144" in "OADMET-0003144"), which is assumed to reflect acquisition
        order. The sorted dataset is divided into n_folds equal-ish chunks. Each fold
        uses one chunk as the test set and the concatenation of all remaining chunks
        as the training set.

        Note: this differs from a strict walk-forward split — training data for a
        given fold may include molecules acquired *after* the test chunk. The intent
        is purely to use the numeric ID as a proxy for diversity/batches rather than
        to enforce a hard temporal boundary.

        Molecules with empty or unparsable names are assigned order 0 and sort
        to the beginning of the sequence (treated as the earliest acquired).

        Args:
            df: Polars DataFrame containing a "molecule_names" column.
            n_folds: Number of folds / chunks.
            seed: Random seed used only for the optional validation split.
            p_val: Fraction of the training set reserved as a validation split
                (sampled randomly).

        Yields:
            Tuples of (fold_index, train_df, val_df, test_df).
            val_df is None when p_val == 0.
        """
        # Sort molecules by numeric ID so earlier IDs appear first.
        mol_numbers = np.array(
            [_extract_molecule_number(n) for n in df["molecule_names"].to_list()]
        )
        sorted_idx = np.argsort(mol_numbers, kind="stable")
        df_sorted = df[sorted_idx].clone()
        n = df_sorted.shape[0]

        # Cut the sorted data into n_folds equal-ish chunks.
        boundaries = np.linspace(0, n, n_folds + 1, dtype=int)
        chunks = [
            df_sorted[boundaries[i]:boundaries[i + 1]].clone()
            for i in range(n_folds)
        ]

        for fold in range(n_folds):
            test = chunks[fold]
            train = pl.concat([chunks[i] for i in range(n_folds) if i != fold])

            val = None
            if p_val > 0:
                train, val = split_dataset_random(train, p_test=p_val, seed=seed + fold)

            yield fold, train, val, test

    return (generate_cv_splits_temporal,)


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
        # Derive binary class columns from the continuous threshold
        df_in = df.with_columns([
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
def _(AnovaRM, Optional, Path, pg, pl, plt, sns):
    def make_boxplots_parametric(
        df: pl.DataFrame,
        metric_ls: list[str],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create boxplots for each metric using repeated measures ANOVA.

        Converts to pandas internally because statsmodels AnovaRM and seaborn
        require pandas DataFrames.

        Args:
            df: Polars DataFrame with columns [cv_cycle, method] plus metric columns.
            metric_ls: List of metric column names to create boxplots for.
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        # AnovaRM and seaborn both require pandas
        df_pd = df.to_pandas()

        sns.set_context('notebook')
        sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
        sns.set_style('whitegrid')
        figure, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(14, 10))

        for i, stat in enumerate(metric_ls):
            model = AnovaRM(data=df_pd, depvar=stat, subject='cv_cycle', within=['method']).fit()
            p_value = model.anova_table['Pr > F'].iloc[0]
            ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i // 2, i % 2], data=df_pd, palette="Set2", legend=False)
            title = stat.upper()
            ax.set_title(f"p={p_value:.1e}")
            ax.set_xlabel("")
            ax.set_ylabel(title)
            x_tick_labels = ax.get_xticklabels()
            label_text_list = [x.get_text() for x in x_tick_labels]
            new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
            ax.set_xticks(list(range(0, len(x_tick_labels))))
            ax.set_xticklabels(new_xtick_labels)
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
        return figure

    def make_boxplots_nonparametric(
        df: pl.DataFrame,
        metric_ls: list[str],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create boxplots for each metric using the Friedman non-parametric test.

        Converts to pandas internally because pingouin and seaborn require pandas.

        Args:
            df: Polars DataFrame with columns [cv_cycle, method] plus metric columns.
            metric_ls: List of metric column names to create boxplots for.
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        # pingouin and seaborn both require pandas
        df_pd = df.to_pandas()

        sns.set_context('notebook')
        sns.set(rc={'figure.figsize': (4, 3)}, font_scale=1.5)
        sns.set_style('whitegrid')
        figure, axes = plt.subplots(2, 2, sharex=False, sharey=False, figsize=(14, 10))

        for i, stat in enumerate(metric_ls):
            friedman = pg.friedman(df_pd, dv=stat, within="method", subject="cv_cycle")['p_unc'].values[0]
            ax = sns.boxplot(y=stat, x="method", hue="method", ax=axes[i // 2, i % 2], data=df_pd, palette="Set2", legend=False)
            title = stat.replace("_", " ").upper()
            ax.set_title(f"p={friedman:.1e}")
            ax.set_xlabel("")
            ax.set_ylabel(title)
            x_tick_labels = ax.get_xticklabels()
            label_text_list = [x.get_text() for x in x_tick_labels]
            new_xtick_labels = ["\n".join(x.split("_")) for x in label_text_list]
            ax.set_xticks(list(range(0, len(x_tick_labels))))
            ax.set_xticklabels(new_xtick_labels)
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
        return figure

    return make_boxplots_nonparametric, make_boxplots_parametric


@app.cell
def _(Optional, Path, math, np, pl, plt, rm_tukey_hsd, sns, stats):
    def make_normality_diagnostic(
        df: pl.DataFrame,
        metric_ls: list[str],
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a normality diagnostic plot grid with histograms and QQ plots for the given metrics.

        Residuals are computed by subtracting each group's mean (per method) so that
        the normality assumption of the repeated-measures ANOVA can be assessed.

        Args:
            df: Polars DataFrame with columns [cv_cycle, method, split] plus metric columns.
            metric_ls: List of metric column names to assess for normality.
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        # Subtract per-method group mean from each metric (mean-centre within method)
        group_means = df.group_by("method").agg([
            pl.col(m).mean().alias(f"_mean_{m}") for m in metric_ls
        ])
        df_norm = df.join(group_means, on="method", how="left")
        df_norm = df_norm.with_columns([
            (pl.col(m) - pl.col(f"_mean_{m}")).alias(m) for m in metric_ls
        ]).drop([f"_mean_{m}" for m in metric_ls])

        # Unpivot (melt) to long format for easy per-metric iteration
        df_long = df_norm.unpivot(
            on=metric_ls,
            index=["cv_cycle", "method", "split"],
            variable_name="metric",
            value_name="value",
        )

        # Convert to pandas for seaborn and scipy.stats.probplot
        df_long_pd = df_long.to_pandas()

        sns.set_context('notebook', font_scale=1.5)
        sns.set_style('whitegrid')

        metrics = df_long_pd['metric'].unique()
        n_metrics = len(metrics)

        fig, axes = plt.subplots(2, n_metrics, figsize=(20, 10))

        for i, metric in enumerate(metrics):
            ax = axes[0, i]
            sns.histplot(df_long_pd[df_long_pd['metric'] == metric]['value'], kde=True, ax=ax)
            ax.set_title(f'{metric}', fontsize=16)

        for i, metric in enumerate(metrics):
            ax = axes[1, i]
            metric_data = df_long_pd[df_long_pd['metric'] == metric]['value']
            stats.probplot(metric_data, dist="norm", plot=ax)
            ax.set_title("")

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


    def mcs_plot(pc, effect_size, means, labels=True, cmap=None, cbar_ax_bbox=None,
                 ax=None, show_diff=True, cell_text_size=16, axis_text_size=12,
                 show_cbar=True, reverse_cmap=False, vlim=None, **kwargs):
        """
        Create a multiple comparison of means plot using a heatmap.

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

        # Create a DataFrame for the annotations
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


    def make_mcs_plot_grid(df, stats, group_col, alpha=.05,
                           figsize=(20, 10), direction_dict={}, effect_dict={}, show_diff=True,
                           cell_text_size=16, axis_text_size=12, title_text_size=16, sort_axes=False,
                           save_path: Optional[Path] = None):
        """
        Create a grid of multiple comparison of means plots using Tukey HSD test results.

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
        # Use a 2-column grid when 4 stats are given (perfect 2×2); otherwise 3 columns.
        ncol = 2 if len(stats) == 4 else 3
        nrow = math.ceil(len(stats) / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize)

        # Set defaults
        for key in ['r2', 'rho', 'prec', 'recall', 'mae', 'mse']:
            direction_dict.setdefault(key, 'maximize' if key in ['r2', 'rho', 'prec', 'recall'] else 'minimize')

        for key in ['r2', 'rho', 'prec', 'recall']:
            effect_dict.setdefault(key, 0.1)

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

        # If there are less plots than cells in the grid, hide the remaining cells
        if (len(stats) % ncol) != 0:
            for i in range(len(stats), nrow * ncol):
                row = i // ncol
                col = i % ncol
                ax[row, col].set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


    return make_mcs_plot_grid, make_normality_diagnostic


@app.cell
def _(
    Optional,
    Path,
    calc_regression_metrics,
    np,
    pl,
    plt,
    precision_score,
    recall_score,
    rm_tukey_hsd,
    sns,
):
    def make_scatterplot(
        df: pl.DataFrame,
        val_col: str,
        pred_col: str,
        thresh: float,
        cycle_col: str = "cv_cycle",
        group_col: str = "method",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create scatter plots for each method showing the relationship between predicted and measured values.

        Args:
            df: Polars DataFrame with columns [group_col, cycle_col, val_col, pred_col].
            val_col: Column name for the ground truth values.
            pred_col: Column name for the model predictions.
            thresh: Decision threshold for binary precision/recall computation.
            cycle_col: Column indicating the cross-validation fold. Default is "cv_cycle".
            group_col: Column indicating the comparison groups/methods. Default is "method".
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        df_split_metrics = calc_regression_metrics(
            df, cycle_col=cycle_col, val_col=val_col, pred_col=pred_col, thresh=thresh
        )
        methods = df[group_col].unique().to_list()

        fig, axs = plt.subplots(nrows=3, ncols=2, figsize=(14, 18))
        axs_flat = axs.flatten()

        for ax, method in zip(axs_flat, methods):
            # Filter using Polars expressions
            df_method = df.filter(pl.col(group_col) == method)
            df_metrics = df_split_metrics.filter(pl.col(group_col) == method)

            y_true_vals = df_method[val_col].to_numpy()
            y_pred_vals = df_method[pred_col].to_numpy()

            ax.scatter(y_pred_vals, y_true_vals, alpha=0.3)
            ax.plot(
                [y_true_vals.min(), y_true_vals.max()],
                [y_true_vals.min(), y_true_vals.max()],
                'k--', lw=1,
            )

            ax.axhline(y=thresh, color='r', linestyle='--')
            ax.axvline(x=thresh, color='r', linestyle='--')
            ax.set_title(method)

            precision = precision_score(y_true_vals > thresh, y_pred_vals > thresh)
            recall = recall_score(y_true_vals > thresh, y_pred_vals > thresh)

            # Aggregate mean metrics across CV folds for the annotation
            mae_mean  = df_metrics["mae"].mean()
            mse_mean  = df_metrics["mse"].mean()
            r2_mean   = df_metrics["r2"].mean()
            rho_mean  = df_metrics["rho"].mean()
            metrics_text = (
                f"MAE: {mae_mean:.2f}\nMSE: {mse_mean:.2f}\n"
                f"R2: {r2_mean:.2f}\nrho: {rho_mean:.2f}\n"
                f"Precision: {precision:.2f}\nRecall: {recall:.2f}"
            )
            ax.text(0.05, .5, metrics_text, transform=ax.transAxes, verticalalignment='top')
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Measured')

        for ax in axs_flat[len(methods):]:
            ax.set_visible(False)

        fig.tight_layout()
        if save_path is not None:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
        return fig


    def ci_plot(result_tab, ax_in, name: str, show_ylabel: bool = True) -> None:
        """
        Create a confidence interval plot for the given result table.

        result_tab is a pandas DataFrame produced by rm_tukey_hsd — seaborn's
        pointplot and errorbar require pandas Series for its index labels.

        Args:
            result_tab: pandas DataFrame with columns ['meandiff', 'lower', 'upper'].
            ax_in: Matplotlib Axes on which to draw the plot.
            name: Title string for the subplot.
            show_ylabel: Whether to show y-axis tick labels. Set False for right-column axes.
        """
        result_err = np.array([
            result_tab['meandiff'] - result_tab['lower'],
            result_tab['upper'] - result_tab['meandiff'],
        ])
        sns.set(rc={'figure.figsize': (6, 2)})
        sns.set_context('notebook')
        sns.set_style('whitegrid')
        ax = sns.pointplot(x=result_tab.meandiff, y=result_tab.index, marker='o', linestyle='', ax=ax_in)
        ax.errorbar(y=result_tab.index, x=result_tab['meandiff'], xerr=result_err, fmt='o', capsize=5)
        ax.axvline(0, ls="--", lw=3)
        ax.set_xlabel("Mean Difference")
        ax.set_ylabel("")
        ax.set_title(name)
        ax.set_xlim(-0.2, 0.2)
        if not show_ylabel:
            ax.set_yticklabels([])


    def make_ci_plot_grid(
        df_in: pl.DataFrame,
        metric_list: list[str],
        group_col: str = "method",
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

        Args:
            df_in: Polars DataFrame passed through to rm_tukey_hsd (converted internally).
            metric_list: List of metric column names to create confidence interval plots for.
            group_col: Column indicating the comparison groups. Default is "method".
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        figure, axes = plt.subplots(2, 2, figsize=(14, 12), sharex=False)
        for i, metric in enumerate(metric_list):
            row, col = i // 2, i % 2
            df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
            ci_plot(df_tukey, ax_in=axes[row, col], name=metric, show_ylabel=(col == 0))
        for ax in axes.flatten()[len(metric_list):]:
            ax.set_visible(False)
        figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
        return figure

    return make_ci_plot_grid, make_scatterplot


@app.cell
def _(mo):
    mo.md(r"""
    # Read train dataset and test different data splits
    """)
    return


@app.cell
def _(pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    all_compounds
    return (all_compounds,)


@app.cell
def _(all_compounds, pl):
    whole_train = all_compounds.filter(pl.col("pEC50_dr").is_not_null())
    whole_train
    return (whole_train,)


@app.cell
def _(mo):
    mo.md(r"""
    # Comparing data split strategies via train/test Tanimoto similarity

    For each split method we run one outer round of 5-fold CV and compute, for every
    test compound, its Tanimoto similarity to all training compounds (ECFP4, radius 2,
    2048 bits).  Two views are compared:

    - **All pairs** — every (test, train) similarity value, giving a sense of the full
      similarity distribution the model is exposed to
    - **Nearest neighbour** — only the maximum similarity per test compound, which
      directly measures how "close" to training data each prediction will be

    A well-separated scaffold or temporal split should shift both distributions leftward
    relative to the random split.
    """)
    return


@app.cell
def _(
    DataStructs,
    ExplicitBitVect,
    generate_cv_splits_random,
    generate_cv_splits_scaffold,
    generate_cv_splits_temporal,
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

        Fingerprints are ECFP4 (radius=2, fp_size=2048) generated via
        generate_fingerprint and converted to RDKit ExplicitBitVect for
        BulkTanimotoSimilarity.

        Returns a dict with keys "all" and "nn", each a flat list of floats.
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

    # ── collect similarities across all three split methods ────────────────────

    records: list[dict] = []

    # Random 1×5 CV  (n_outer=1, n_inner=5)
    for _fold, _outer, _inner, _train, _val, _test in generate_cv_splits_random(
        whole_train, n_outer=1, n_inner=5, seed=42
    ):
        _sims = _fold_similarities(_train, _test)
        for _s in _sims["all"]:
            records.append({"split": "random", "mode": "all pairs",        "tanimoto": _s})
        for _s in _sims["nn"]:
            records.append({"split": "random", "mode": "nearest neighbour","tanimoto": _s})

    # Scaffold 1×5 CV
    for _fold, _outer, _inner, _train, _val, _test in generate_cv_splits_scaffold(
        whole_train, n_outer=1, n_inner=5, seed=42
    ):
        _sims = _fold_similarities(_train, _test)
        for _s in _sims["all"]:
            records.append({"split": "scaffold", "mode": "all pairs",        "tanimoto": _s})
        for _s in _sims["nn"]:
            records.append({"split": "scaffold", "mode": "nearest neighbour","tanimoto": _s})

    # Temporal 5-fold CV
    for _fold, _train, _val, _test in generate_cv_splits_temporal(
        whole_train, n_folds=5, seed=42
    ):
        _sims = _fold_similarities(_train, _test)
        for _s in _sims["all"]:
            records.append({"split": "temporal", "mode": "all pairs",        "tanimoto": _s})
        for _s in _sims["nn"]:
            records.append({"split": "temporal", "mode": "nearest neighbour","tanimoto": _s})

    sim_df = pl.DataFrame(records)
    return (sim_df,)


@app.cell
def _(Path, mo, mpatches, plt, sim_df):
    _splits  = ["random", "scaffold", "temporal"]
    _modes   = ["all pairs", "nearest neighbour"]
    _colors  = {"random": "#4C78A8", "scaffold": "#F58518", "temporal": "#54A24B"}
    _labels  = {"all pairs": "All pairs", "nearest neighbour": "Nearest neighbour"}

    _fig, _axes = plt.subplots(1, 2, figsize=(10, 5), sharey=True)
    _fig.suptitle("Train/test Tanimoto similarity by split strategy (ECFP4)", fontsize=13)

    for _ax, _mode in zip(_axes, _modes):
        # Build one data list and one colour list per split, in fixed order
        _data   = [
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
        _ax.set_xticks([1, 2, 3])
        _ax.set_xticklabels(_splits, fontsize=11)
        _ax.set_ylim(0, 1)
        _ax.set_ylabel("Tanimoto similarity", fontsize=11)
        _ax.set_xlabel("Split strategy", fontsize=11)
        _ax.yaxis.grid(True, linestyle="--", alpha=0.6)
        _ax.set_axisbelow(True)

    # shared legend
    _handles = [mpatches.Patch(facecolor=_colors[s], alpha=0.8, label=s) for s in _splits]
    _fig.legend(handles=_handles, loc="lower center", ncol=3, fontsize=11, frameon=False, bbox_to_anchor=(0.5, -0.04))
    _fig.tight_layout()

    _PLOT_DIR = Path("../plots/2_ml_baseline")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)
    _fig.savefig(_PLOT_DIR / "tanimoto_similarity_splits.png", bbox_inches="tight")

    mo.as_html(_fig)
    return


@app.cell
def _(mo):
    mo.md(r"""
    All three splits are very similar.
    For the scaffold split, this is likely because the majority of scaffolds are present
    in only one or a very small number of molecules.
    For a more challenging structure-based data split, you could perform clustering such as
    hierarchical clustering and then extract five distinct sets of molecules.

    The temporal split is traditionally considered a more realistic and pessimistic split,
    but that assumes data from a cyclical process — a series of DMTL (Design, Make, Test, Learn) cycles — where later entries
    expand and learn from previous ones.
    That is not the case here, as the training set is largely a subset of a screening library.
    Nonetheless, there is a small shift to lower values, seen especially in the NN plot.
    If I wanted to examine this in more detail, I would check whether the training compounds
    that were not in the single-dose set have ID numbers after those of the single-dose compounds.
    That might explain the slight shift.
    """)
    return


@app.cell
def _(gc, sim_df):
    del sim_df
    gc.collect()
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Train single task baseline models

    ## 5 × 5 random cross-validation

    We iterate over 25 folds (5 outer × 5 inner) produced by `generate_cv_splits_random`.
    For **each fold** we:

    1. Generate ECFP fingerprints on the fold's train and test subsets (needed by the
       fingerprint-based models).
    2. Instantiate and train all four model classes and the two baselines (Mean and NN):
       - `RandomForestModel` (ECFP, 500 trees)
       - `BoostedTreesModel` (ECFP, XGBoost)
       - `ChempropModel` (MPNN from scratch, 50 epochs)
       - `ChempropChemeleonModel` (fine-tuned CheMeleon, 30 epochs)
    3. Predict `pEC50_dr` on the test compounds.
    4. Collect every prediction into a long-format Polars DataFrame and write it to
       `predictions/cv_predictions.csv` once all folds are done.

    The CSV schema is:
    ```
    inchikey | molecule_names | smiles | fold | outer_fold | inner_fold |
    model    | y_true         | y_pred
    ```
    """)
    return


@app.cell
def _(
    BoostedTreesModel,
    ChempropChemeleonModel,
    ChempropModel,
    MeanBaseline,
    NearestNeighbourBaseline,
    Path,
    RandomForestModel,
    extract_fp_matrix,
    gc,
    generate_cv_splits_random,
    generate_fingerprint,
    gzip,
    pl,
    tqdm,
    whole_train,
):
    # ── constants ──────────────────────────────────────────────────────────────
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _N_OUTER    = 5
    _N_INNER    = 5
    _SEED       = 42
    _P_VAL      = 0.1          # fraction of train kept as validation (XGBoost / Chemprop early stopping)
    _FP_TYPE    = "ecfp"
    _FP_KWARGS  = {"radius": 2, "fp_size": 2048}

    # ── model names ───────────────────────────────────────────────────────────
    _MODEL_NAMES = ["mean", "nn1", "rf", "gbm", "chemprop", "chemeleon"]

    if _PRED_PATH_GZ.exists():
        print(f"Predictions already exist at {_PRED_PATH_GZ} — skipping training.")
        _pred_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        # ── run all 25 folds ──────────────────────────────────────────────────
        #_debug_df = whole_train.sample(n=100, seed=_SEED)  # TODO: remove for full run
        _all_records: list[dict] = []
        _n_folds = _N_OUTER * _N_INNER

        _pbar = tqdm(
            generate_cv_splits_random(
                whole_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL
            ),
            total=_n_folds,
            desc="CV folds",
            unit="fold",
        )

        for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
            # Generate fingerprints once per fold — RF and GBM share these arrays
            _train_fp = generate_fingerprint(_train_raw, _FP_TYPE, **_FP_KWARGS)
            _val_fp   = generate_fingerprint(_val_raw,   _FP_TYPE, **_FP_KWARGS)
            _test_fp  = generate_fingerprint(_test_raw,  _FP_TYPE, **_FP_KWARGS)

            # Extract numpy arrays used by fingerprint-based models
            _X_train = extract_fp_matrix(_train_fp, _FP_TYPE)
            _X_val   = extract_fp_matrix(_val_fp,   _FP_TYPE)
            _X_test  = extract_fp_matrix(_test_fp,  _FP_TYPE)
            _y_train = _train_raw[_TARGET_COL].to_numpy()
            _y_val   = _val_raw[_TARGET_COL].to_numpy()
            _y_true  = _test_raw[_TARGET_COL].to_numpy()

            # Extract SMILES lists used by Chemprop-based models
            _smi_train = _train_raw["smiles"].to_list()
            _smi_val   = _val_raw["smiles"].to_list()
            _smi_test  = _test_raw["smiles"].to_list()

            # ── train & predict each model ────────────────────────────────────
            for _model_name in _MODEL_NAMES:
                _pbar.set_postfix({"fold": _fold, "o": _outer, "i": _inner, "model": _model_name}, refresh=False)

                if _model_name == "mean":
                    _model = MeanBaseline()
                    _model.train(_X_train, _y_train)
                    _y_pred = _model.predict(_X_test)

                elif _model_name == "nn1":
                    _model = NearestNeighbourBaseline()
                    _model.train(_X_train, _y_train)
                    _y_pred = _model.predict(_X_test)

                elif _model_name == "rf":
                    _model = RandomForestModel(pred_type="regression")
                    _model.train(_X_train, _y_train)
                    _y_pred = _model.predict(_X_test)

                elif _model_name == "gbm":
                    _model = BoostedTreesModel(pred_type="regression")
                    _model.train(_X_train, _y_train, _X_val, _y_val)
                    _y_pred = _model.predict(_X_test)

                elif _model_name == "chemprop":
                    _model = ChempropModel(pred_type="regression")
                    _model.train(_smi_train, _y_train, _smi_val, _y_val, target_col=_TARGET_COL)
                    _y_pred = _model.predict(_smi_test)

                elif _model_name == "chemeleon":
                    _model = ChempropChemeleonModel(pred_type="regression")
                    _model.train(_smi_train, _y_train, _smi_val, _y_val, target_col=_TARGET_COL)
                    _y_pred = _model.predict(_smi_test)

                # Free model memory before accumulating results
                del _model

                # Accumulate one row per test compound
                for _ik, _mn, _smi, _yt, _yp in zip(
                    _test_raw["inchikey"].to_list(),
                    _test_raw["molecule_names"].to_list(),
                    _test_raw["smiles"].to_list(),
                    _y_true.tolist(),
                    _y_pred.tolist(),
                ):
                    _all_records.append({
                        "inchikey":       _ik,
                        "molecule_names": _mn,
                        "smiles":         _smi,
                        "fold":           _fold,
                        "outer_fold":     _outer,
                        "inner_fold":     _inner,
                        "model":          _model_name,
                        "y_true":         _yt,
                        "y_pred":         _yp,
                    })

            # Release per-fold fingerprint arrays before the next fold
            del _train_fp, _val_fp, _test_fp, _X_train, _X_val, _X_test
            gc.collect()

        # ── write predictions (gzip-compressed directly) ──────────────────────
        _pred_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _pred_df.write_csv(_f)
        print(f"\nSaved {len(_pred_df):,} prediction rows → {_PRED_PATH_GZ}")

    _pred_df
    return


@app.cell
def _(mo):
    mo.md(r"""
    On a base M4 Mac Mini, the previous cell took around 7h to run
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Model comparison analysis

    We follow the approach from
    [polaris-hub/polaris-method-comparison](https://github.com/polaris-hub/polaris-method-comparison):

    > **Never compare models using only the mean of a metric over folds.**
    > Distributions carry the information needed to assess statistical significance.

    The workflow is:
    1. Load predictions → compute per-fold regression metrics
    2. Inspect normality of the residuals (histogram + QQ plot)
    3. If normally distributed: **repeated-measures ANOVA + Tukey HSD**
    4. If not: **Friedman test** (non-parametric equivalent)
    5. Visualise results: boxplots, confidence-interval plots, multiple-comparison heatmaps
    """)
    return


@app.cell
def _(pl):
    """Load the saved 5×5 CV predictions and rename the model column to 'method'
    so it matches the column name expected by all comparison functions."""
    pred_df = (
        pl.read_csv("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    pred_df
    return (pred_df,)


@app.cell
def _(calc_regression_metrics, pred_df):
    """Compute MAE, MSE, R², Spearman ρ (rho), precision and recall for each
    (cv_cycle, method) group. """
    THRESH = 4.0

    metrics_df = calc_regression_metrics(
        pred_df,
        cycle_col="cv_cycle",
        val_col="y_true",
        pred_col="y_pred",
        thresh=THRESH,
    )
    metrics_df
    return THRESH, metrics_df


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 1 — Scatter plots: predicted vs measured

    Each panel shows one model's predictions pooled across all 25 folds.
    The diagonal dashed line is the identity (perfect prediction); red dashed lines
    mark the activity threshold.  Metric values shown are fold-averaged.
    """)
    return


@app.cell
def _(Path, THRESH, make_scatterplot, mo, pred_df):
    PLOT_DIR = Path("../plots/2_ml_baseline")
    PLOT_DIR.mkdir(parents=True, exist_ok=True)
    mo.as_html(make_scatterplot(
        pred_df,
        val_col="y_true",
        pred_col="y_pred",
        thresh=THRESH,
        cycle_col="cv_cycle",
        group_col="method",
        save_path=PLOT_DIR / "scatterplot.png",
    ))
    return (PLOT_DIR,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 2 — Normality diagnostics

    Repeated-measures ANOVA assumes that the **residuals within each method group**
    are approximately normally distributed.  We assess this visually via:

    - **Histogram + KDE** (top row) — residuals should look bell-shaped
    - **Q–Q plot** (bottom row) — points should fall on the diagonal

    If the residuals look skewed or heavy-tailed, use the Friedman test instead
    (see Step 4).
    """)
    return


@app.cell
def _(PLOT_DIR, make_normality_diagnostic, metrics_df, mo):
    METRIC_LIST = ["mae", "mse", "r2", "rho"]

    mo.as_html(make_normality_diagnostic(
        metrics_df,
        METRIC_LIST,
        save_path=PLOT_DIR / "normality_diagnostic.png",
    ))
    return (METRIC_LIST,)


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 3 — Boxplots with repeated-measures ANOVA p-values

    Each box shows the cross-fold distribution of a metric for one model.
    The title of each panel reports the **repeated-measures ANOVA p-value**,
    which tests whether at least one model differs significantly from the others
    while accounting for the shared folds (the repeated-measures structure).

    A small p-value (< 0.05) means the models are not equivalent; see the
    Tukey HSD heatmaps below for pairwise comparisons.
    """)
    return


@app.cell
def _(METRIC_LIST, PLOT_DIR, make_boxplots_parametric, metrics_df, mo):

    mo.as_html(make_boxplots_parametric(
        metrics_df,
        METRIC_LIST,
        save_path=PLOT_DIR / "boxplots_parametric.png",
    ))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 4 — Boxplots with Friedman test p-values (non-parametric)

    Same layout as Step 3 but using the **Friedman test**, which makes no normality
    assumption.  Compare these p-values with the ANOVA p-values above: if they agree,
    the ANOVA result is trustworthy; if they differ substantially, prefer the Friedman
    result.
    """)
    return


@app.cell
def _(METRIC_LIST, PLOT_DIR, make_boxplots_nonparametric, metrics_df, mo):

    mo.as_html(make_boxplots_nonparametric(
        metrics_df,
        METRIC_LIST,
        save_path=PLOT_DIR / "boxplots_nonparametric.png",
    ))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 5 — Tukey HSD confidence-interval plots

    These plots show pairwise mean differences between models with 95 % simultaneous
    confidence intervals (Tukey HSD, family-wise error rate = 0.05).

    - Intervals that **do not cross zero** indicate a statistically significant
      difference between that pair.
    - Intervals are symmetric around the observed mean difference; the dashed vertical
      line is the null (no difference).
    """)
    return


@app.cell
def _(METRIC_LIST, PLOT_DIR, make_ci_plot_grid, metrics_df, mo):

    mo.as_html(make_ci_plot_grid(
        metrics_df,
        METRIC_LIST,
        group_col="method",
        save_path=PLOT_DIR / "ci_plot_grid.png",
    ))
    return


@app.cell
def _(mo):
    mo.md(r"""
    ## Step 6 — Multiple-comparison heatmaps (Tukey HSD)

    Each cell shows the **mean difference** between the row model and the column model,
    annotated with the Tukey-adjusted p-value significance stars:

    | Stars | p-value |
    |---|---|
    | `***` | < 0.001 |
    | `**`  | < 0.01  |
    | `*`   | < 0.05  |
    | (none)| ≥ 0.05  |

    The colour encodes the direction and magnitude of the difference.  For metrics to
    **maximise** (R², ρ, precision, recall) a warm colour (red) means the row model
    is better; for metrics to **minimise** (MAE, MSE) the colourmap is reversed so
    warm still means the row model is worse.
    """)
    return


@app.cell
def _(METRIC_LIST, PLOT_DIR, make_mcs_plot_grid, metrics_df, mo):

    mo.as_html(make_mcs_plot_grid(
        metrics_df,
        stats=METRIC_LIST,
        group_col="method",
        figsize=(13, 12),
        direction_dict={
            "r2": "maximize",
            "rho": "maximize",
            "prec": "maximize",
            "recall": "maximize",
            "mae": "minimize",
            "mse": "minimize",
        },
        effect_dict={
            "r2": 0.2,
            "rho": 0.2,
            "prec": 0.2,
            "recall": 0.2,
            "mae": 0.5,
            "mse": 1.0,
        },
        show_diff=True,
        sort_axes=True,
        save_path=PLOT_DIR / "mcs_plot_grid.png",
    ))
    return


@app.cell
def _(mo):
    mo.md(r"""
    This last plot shows the Tukey HSD results as a heatmap: cell colour and numbers inside cells encode effect size, and stars indicate significance level.
    For our comparison, CheMeleon is the best across all metrics and against all other models.
    Chemprop is better than RF and GBM across all metrics.
    GBM beats RF only on one metric, rho.

    It is not surprising that the mean and NN baselines were much worse than the other models.
    As expected, the R² of the mean model was 0 and rho was NaN.
    The worst-case MAE was 0.91, meaning predictions were off by roughly one order of magnitude on average.
    The NN model performed slightly better in terms of MAE, but R² was slightly below zero.

    Given the results, in the next cells we will train a Chemeleon model on the whole training set
    and predict the test set.
    """)
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Train Chemeleon on whole train dataset and submit predictions
    """)
    return


@app.cell
def _(ChempropChemeleonModel, Path, np, pl, whole_train):
    """
    Train a CheMeleon model on the entire training set (all compounds with a measured
    pEC50_dr) and generate predictions for the 513 held-out test compounds.

    The model uses a 10 % internal validation split drawn from the training data for
    early stopping — this split is *not* the competition test set.

    The output CSV matches the format required by validate_activity_submission():
        SMILES | Molecule Name | pEC50
    """


    _TARGET_COL   = "pEC50_dr"
    _SEED         = 42
    _PRED_OUT     = Path("../predictions/2_ml_baseline_chemeleon_test_submission.csv")

    # ── load data ──────────────────────────────────────────────────────────────
    # Test set loaded directly from the most recent raw release file.
    # dose_response_test.csv already has columns named "SMILES" and "Molecule Name".
    _test_df = pl.read_csv("../data/raw/20260409/dose_response_test.csv")
    # whole_train is already filtered to rows with measured pEC50_dr

    # ── validation split for early stopping (10 % of train) ───────────────────

    _rng      = np.random.default_rng(_SEED)
    _n        = whole_train.shape[0]
    _val_idx  = _rng.choice(_n, size=int(_n * 0.1), replace=False)
    _train_idx = np.setdiff1d(np.arange(_n), _val_idx)

    _train_sub = whole_train[_train_idx]
    _val_sub   = whole_train[_val_idx]

    _X_train = _train_sub["smiles"].to_list()
    _y_train = _train_sub[_TARGET_COL].to_numpy()
    _X_val   = _val_sub["smiles"].to_list()
    _y_val   = _val_sub[_TARGET_COL].to_numpy()
    _X_test  = _test_df["SMILES"].to_list()

    # ── train ──────────────────────────────────────────────────────────────────
    if _PRED_OUT.exists():
        print(f"Submission file already exists at {_PRED_OUT} — skipping training.")
    else:
        _model = ChempropChemeleonModel(pred_type="regression", epochs=50)
        _model.train(_X_train, _y_train, _X_val, _y_val, target_col=_TARGET_COL)

        # ── predict and build submission DataFrame ─────────────────────────────
        _y_pred = _model.predict(_X_test)

        _submission = pl.DataFrame({
            "SMILES":        _test_df["SMILES"].to_list(),
            "Molecule Name": _test_df["Molecule Name"].to_list(),
            "pEC50":         _y_pred.tolist(),
        })

        _PRED_OUT.parent.mkdir(parents=True, exist_ok=True)
        _submission.write_csv(_PRED_OUT)
        print(f"Saved {len(_submission)} predictions → {_PRED_OUT}")
    return


@app.cell
def _(Iterable, Optional, Path, mo, np, pd):
    """
    Validate the submission file using the rules from the OpenADMET activity_validation.py file:
      - Required columns: SMILES, Molecule Name, pEC50
      - No missing identifiers or duplicate Molecule Names
      - pEC50 must be numeric and finite
      - Exactly 513 rows
    """

    _PRED_OUT = Path("../predictions/2_ml_baseline_chemeleon_test_submission.csv")

    ACTIVITY_DATASET_SIZE = 513


    def _as_set(values: Iterable[str]) -> set[str]:
        return {str(v) for v in values}


    def validate_activity_submission(
        activity_predictions_file: Path,
        expected_ids: Optional[set[str]] = None,
        required_id_columns: tuple[str, ...] = ("SMILES", "Molecule Name"),
        required_value_columns: tuple[str, ...] = ("pEC50",),
    ) -> tuple[bool, list[str]]:
        errors: list[str] = []

        path = Path(activity_predictions_file)
        if not path.exists():
            return False, [f"File does not exist: {path}"]

        try:
            activity_predictions = pd.read_csv(path)
        except Exception as exc:
            return False, [f"Error reading CSV file: {exc}"]

        required_columns = (*required_id_columns, *required_value_columns)
        missing_columns = [col for col in required_columns if col not in activity_predictions.columns]
        if missing_columns:
            errors.append(f"Missing required column(s): {missing_columns}")
            return False, errors

        if activity_predictions.empty:
            errors.append("Submission is empty.")
            return False, errors

        null_id_rows = activity_predictions[list(required_id_columns)].isna().any(axis=1).sum()
        if null_id_rows:
            errors.append(f"Found {null_id_rows} row(s) with missing identifier values.")

        if "Molecule Name" in activity_predictions.columns:
            duplicate_ids = activity_predictions["Molecule Name"].duplicated().sum()
            if duplicate_ids:
                errors.append(f"Found {duplicate_ids} duplicated 'Molecule Name' value(s).")

        for col in required_value_columns:
            numeric_col = pd.to_numeric(activity_predictions[col], errors="coerce")
            invalid_numeric = numeric_col.isna().sum()
            if invalid_numeric:
                errors.append(f"Column '{col}' contains {invalid_numeric} non-numeric or missing value(s).")
                continue

            non_finite = (~np.isfinite(numeric_col.to_numpy())).sum()
            if non_finite:
                errors.append(f"Column '{col}' contains {non_finite} non-finite value(s) (inf or -inf).")

        submitted_ids = _as_set(activity_predictions["Molecule Name"])
        if expected_ids is not None:
            expected_ids = _as_set(expected_ids)
            missing = sorted(expected_ids - submitted_ids)
            extra = sorted(submitted_ids - expected_ids)
            if missing:
                errors.append(f"Missing {len(missing)} expected molecule(s): {missing[:20]}")
            if extra:
                errors.append(f"Found {len(extra)} unexpected molecule(s): {extra[:20]}")
        elif len(activity_predictions) != ACTIVITY_DATASET_SIZE:
            errors.append(
                f"Submission contains {len(activity_predictions)} rows, expected {ACTIVITY_DATASET_SIZE}."
            )

        return len(errors) == 0, errors

    _ok, _errs = validate_activity_submission(_PRED_OUT)
    if _ok:
        _out = mo.md(f"**Validation passed.** `{_PRED_OUT.name}` is ready for submission.")
    else:
        _out = mo.md("**Validation failed:**\n" + "\n".join(f"- {e}" for e in _errs))
    _out
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    I submitted the file using the Web GUI. At the time of submission, I was rank 57 of 84, just above the baseline provided by OpenADMET using LGBM.
    The provided results are:

    | Data | MAE | R2 | rho |
    | ---  | --- | --- | --- |
    | Test | 0.5738 | 0.3355 | 0.7084 |
    | CV  | 0.50 |  0.62 | 0.75 |


    Comparing these results to the CV estimates, we see a large drop in R², while rho and MAE degrade more moderately.
    This suggests meaningful differences in the distribution of activity values between the CV and test sets, either in the predicted or observed distributions.
    """)
    return


if __name__ == "__main__":
    app.run()
