import marimo

__generated_with = "0.23.2"
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
    return (
        AnovaRM,
        AtomPairFingerprint,
        AvalonFingerprint,
        BaseKFold,
        BatchMolGraph,
        ConformerGenerator,
        E3FPFingerprint,
        ECFPFingerprint,
        Iterable,
        Iterator,
        MACCSFingerprint,
        MPNN,
        MQNsFingerprint,
        MolFromSmiles,
        MolFromSmilesTransformer,
        MordredFingerprint,
        Optional,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        RandomForestClassifier,
        RandomForestRegressor,
        RegressionFFN,
        TopologicalTorsionFingerprint,
        accuracy_score,
        balanced_accuracy_score,
        chemnn,
        f1_score,
        featurizers,
        gc,
        gzip,
        math,
        matthews_corrcoef,
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
        urlretrieve,
        warnings,
        xgb,
    )


@app.cell
def _(
    BatchMolGraph,
    MPNN,
    MolFromSmiles,
    Path,
    RegressionFFN,
    chemnn,
    featurizers,
    torch,
    urlretrieve,
):
    class CheMeleonFingerprint:
        """
        Learned molecular fingerprints from the CheMeleon pretrained D-MPNN backbone.

        Wraps the message-passing encoder of CheMeleon and extracts the aggregated
        graph-level embedding (2048-d) for each molecule without running the
        task-specific FFN head. Weights are downloaded once from Zenodo and cached
        at ~/.chemprop/chemeleon_mp.pt.

        Usage::

            fp_gen = CheMeleonFingerprint()
            embeddings = fp_gen(["CCO", "c1ccccc1"])  # np.ndarray (2, 2048)

        The instance can also be passed SMILES strings directly, matching the
        calling convention of scikit-fingerprints transformers.
        """

        def __init__(self, device: str | torch.device | None = None) -> None:
            """
            Args:
                device: PyTorch device to run inference on. If None, uses CPU.
                    Pass "mps" or "cuda" to accelerate on Apple Silicon / NVIDIA.
            """
            self.featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()

            ckpt_dir = Path.home() / ".chemprop"
            ckpt_dir.mkdir(exist_ok=True)
            mp_path = ckpt_dir / "chemeleon_mp.pt"

            # Download the backbone weights on first use (~30 MB, cached afterwards)
            if not mp_path.exists():
                print("Downloading CheMeleon backbone weights (~30 MB)…")
                urlretrieve(
                    "https://zenodo.org/records/15460715/files/chemeleon_mp.pt",
                    mp_path,
                )

            chemeleon_mp = torch.load(mp_path, weights_only=True)
            mp = chemnn.BondMessagePassing(**chemeleon_mp["hyper_parameters"])
            mp.load_state_dict(chemeleon_mp["state_dict"])

            self.model = MPNN(
                message_passing=mp,
                agg=chemnn.MeanAggregation(),
                predictor=RegressionFFN(input_dim=mp.output_dim),
            )
            self.model.eval()
            if device is not None:
                self.model.to(device=device)

        def transform(self, smiles: list[str]) -> "np.ndarray":
            """
            Generate CheMeleon embeddings for a list of SMILES strings.

            Mirrors the scikit-fingerprints `.transform()` API so this class
            can be used as a drop-in inside `generate_fingerprint`.

            Args:
                smiles: List of SMILES strings.

            Returns:
                Float32 array of shape (n_molecules, 2048).
            """
            bmg = BatchMolGraph(
                [self.featurizer(MolFromSmiles(s)) for s in smiles]
            )
            bmg.to(device=self.model.device)
            with torch.no_grad():
                return self.model.fingerprint(bmg).numpy(force=True)

        # Expose requires_conformers so generate_fingerprint treats it like skfp classes
        requires_conformers: bool = False

    return (CheMeleonFingerprint,)


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



    return


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
                model_dir: Directory where the CLI writes model checkpoints.
                    Reused (overwritten) on every train() call.
                epochs: Maximum number of training epochs.
                message_hidden_dim: Hidden dimension of the MPNN message-passing
                    layers (--message-hidden-dim).
                depth: Number of message-passing steps, i.e. the radius of the
                    molecular graph neighbourhood considered (--depth).
                dropout: Dropout probability applied after each message-passing
                    and FFN layer (--dropout).
                ffn_hidden_dim: Hidden dimension of the feed-forward network on
                    top of the MPNN (--ffn-hidden-dim).
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
            self.pred_type         = pred_type
            self.model_dir         = model_dir
            self.epochs            = epochs
            self.message_hidden_dim = message_hidden_dim
            self.depth             = depth
            self.dropout           = dropout
            self.ffn_hidden_dim    = ffn_hidden_dim
            self.ffn_num_layers    = ffn_num_layers
            self.batch_size        = batch_size
            self.init_lr           = init_lr
            self.max_lr            = max_lr
            self.final_lr          = final_lr
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
                "--message-hidden-dim", str(self.message_hidden_dim),
                "--depth", str(self.depth),
                "--dropout", str(self.dropout),
                "--ffn-hidden-dim", str(self.ffn_hidden_dim),
                "--ffn-num-layers", str(self.ffn_num_layers),
                "--batch-size", str(self.batch_size),
                "--init-lr", str(self.init_lr),
                "--max-lr", str(self.max_lr),
                "--final-lr", str(self.final_lr),
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
                model_dir: Directory where the CLI writes model checkpoints.
                    Distinct from ChempropModel's default to avoid collisions.
                epochs: Maximum number of training epochs.
                dropout: Dropout probability applied after each FFN layer. Note
                    that message-passing architecture is fixed by CheMeleon
                    (2048 d_h, depth 6) and cannot be changed during fine-tuning.
                ffn_hidden_dim: Hidden dimension of the task-specific feed-forward
                    head added on top of the frozen backbone (--ffn-hidden-dim).
                ffn_num_layers: Number of layers in the feed-forward head
                    (--ffn-num-layers).
                batch_size: Mini-batch size for fine-tuning (--batch-size).
                init_lr: Initial learning rate for the one-cycle scheduler
                    (--init-lr).
                max_lr: Peak learning rate for the one-cycle scheduler (--max-lr).
                final_lr: Final learning rate for the one-cycle scheduler
                    (--final-lr).
            """
            if pred_type not in ("regression", "classification"):
                raise ValueError("pred_type must be 'regression' or 'classification'")
            self.pred_type      = pred_type
            self.model_dir      = model_dir
            self.epochs         = epochs
            self.dropout        = dropout
            self.ffn_hidden_dim = ffn_hidden_dim
            self.ffn_num_layers = ffn_num_layers
            self.batch_size     = batch_size
            self.init_lr        = init_lr
            self.max_lr         = max_lr
            self.final_lr       = final_lr
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
                "--dropout", str(self.dropout),
                "--ffn-hidden-dim", str(self.ffn_hidden_dim),
                "--ffn-num-layers", str(self.ffn_num_layers),
                "--batch-size", str(self.batch_size),
                "--init-lr", str(self.init_lr),
                "--max-lr", str(self.max_lr),
                "--final-lr", str(self.final_lr),
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

    return (ChempropModel,)


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
    CheMeleonFingerprint,
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
        # CheMeleon is handled separately below — it does not follow the skfp
        # constructor / transform pattern for conformers, so we mark it as a
        # sentinel value and branch on it inside generate_fingerprint.
        "chemeleon": CheMeleonFingerprint,
    }

    def generate_fingerprint(df: pl.DataFrame, fingerprint_type: str, **kwargs) -> pl.DataFrame:
        """
        Generate molecular fingerprints and add them as a new column to the DataFrame.

        Dispatches to the appropriate fingerprint class based on fingerprint_type.
        All skfp types use their standard constructor / transform pipeline.
        The special "chemeleon" type uses the CheMeleonFingerprint class, which
        runs the pretrained CheMeleon D-MPNN backbone and returns 2048-d embeddings.
        For fingerprint types that require 3D conformers (e.g., E3FP), conformers are
        generated automatically via RDKit ETKDGv3.

        Args:
            df: Polars DataFrame containing a "smiles" column.
            fingerprint_type: One of the supported types: "ecfp"/"morgan", "maccs",
                "torsion", "rdkit", "atompair", "avalon", "e3fp", "mordred", "mqn",
                "pubchem", "chemeleon".
            **kwargs: Additional keyword arguments forwarded to the fingerprint class
                constructor (e.g., radius=3, n_bits=1024 for ECFP; device="mps"
                for chemeleon).

        Returns:
            DataFrame with an added column named after fingerprint_type containing
            the computed fingerprint arrays.

        Raises:
            ValueError: If fingerprint_type is not a recognized key.
        """
        if fingerprint_type not in _fp_dict:
            raise ValueError(
                f"Fingerprint type not recognized: {fingerprint_type!r}. "
                f"Valid values: {list(_fp_dict.keys())}"
            )

        smiles_list = df.get_column("smiles").to_list()

        # CheMeleon has a different API: instantiate once, call .transform(smiles)
        if fingerprint_type == "chemeleon":
            fp_func = CheMeleonFingerprint(**kwargs)
            fps = fp_func.transform(smiles_list)
            return df.with_columns(pl.Series(values=fps, name=fingerprint_type))

        # All other fingerprint types follow the skfp constructor / transform pattern.
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

    return (make_boxplots_nonparametric,)


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
        # Use a 1-column grid for a single stat, 2 columns for 4 stats, else 3.
        ncol = 1 if len(stats) == 1 else (2 if len(stats) == 4 else 3)
        nrow = math.ceil(len(stats) / ncol)
        fig, ax = plt.subplots(nrow, ncol, figsize=figsize, squeeze=False)

        # Set defaults
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

    return (make_mcs_plot_grid,)


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


    def ci_plot(
        result_tab,
        ax_in,
        name: str,
        show_ylabel: bool = True,
        xlim: tuple[float, float] = (-0.2, 0.2),
    ) -> None:
        """
        Create a confidence interval plot for the given result table.

        result_tab is a pandas DataFrame produced by rm_tukey_hsd — seaborn's
        pointplot and errorbar require pandas Series for its index labels.

        Args:
            result_tab: pandas DataFrame with columns ['meandiff', 'lower', 'upper'].
            ax_in: Matplotlib Axes on which to draw the plot.
            name: Title string for the subplot.
            show_ylabel: Whether to show y-axis tick labels. Set False for right-column axes.
            xlim: X-axis limits as (min, max). Default is (-0.2, 0.2).
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
        ax.set_xlim(*xlim)
        if not show_ylabel:
            ax.set_yticklabels([])


    def make_ci_plot_grid(
        df_in: pl.DataFrame,
        metric_list: list[str],
        group_col: str = "method",
        figsize: tuple[int, int] = (14, 12),
        xlim: tuple[float, float] = (-0.2, 0.2),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create a grid of confidence interval plots for multiple metrics using Tukey HSD test results.

        Args:
            df_in: Polars DataFrame passed through to rm_tukey_hsd (converted internally).
            metric_list: List of metric column names to create confidence interval plots for.
            group_col: Column indicating the comparison groups. Default is "method".
            figsize: Figure size as (width, height). Default is (14, 12).
            xlim: X-axis limits passed to each ci_plot as (min, max). Default is (-0.2, 0.2).
            save_path: If provided, the figure is saved to this path before returning.

        Returns:
            Matplotlib Figure.
        """
        figure, axes = plt.subplots(2, 2, figsize=figsize, sharex=False)
        for i, metric in enumerate(metric_list):
            row, col = i // 2, i % 2
            df_tukey, _, _, _ = rm_tukey_hsd(df_in, metric, group_col=group_col)
            ci_plot(df_tukey, ax_in=axes[row, col], name=metric, show_ylabel=(col == 0), xlim=xlim)
        for ax in axes.flatten()[len(metric_list):]:
            ax.set_visible(False)
        figure.suptitle("Multiple Comparison of Means\nTukey HSD, FWER=0.05")
        figure.tight_layout()
        if save_path is not None:
            figure.savefig(save_path, dpi=300, bbox_inches="tight")
        return figure

    return (make_ci_plot_grid,)


@app.cell
def _(mo):
    mo.md(r"""
    # Read train dataset and test different data splits
    """)
    return


@app.cell
def _(gc, pl):
    all_compounds = pl.read_csv("../data/processed/all_compounds_activity_data.csv")
    single_task_train = all_compounds.filter(pl.col("pEC50_dr").is_not_null()).\
                                            select(["smiles","inchikey", "molecule_names", "pEC50_dr"])
    single_task_train

    del all_compounds
    gc.collect()
    return (single_task_train,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare larger Chemprop to CheMeleon

    Before we saw Chemeleon being significantly better than Chemprop.
    But Chemeleon, in addition to its pretraining, is a larger model than Chemprop (2048 vs 300 d_h and 3 vs 6 depth).
    I want to check if chemprop as a larger model but no pretraining comes close to chemeleon
    """)
    return


@app.cell
def _(
    ChempropModel,
    Path,
    calc_regression_metrics,
    gc,
    generate_cv_splits_random,
    gzip,
    pl,
    single_task_train,
    tqdm,
):
    # ── constants ──────────────────────────────────────────────────────────────
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/3_larger_chemprop_size_test.csv.gz")
    _N_OUTER    = 5
    _N_INNER    = 5
    _SEED       = 42
    _P_VAL      = 0.1          # fraction of train kept as validation (XGBoost / Chemprop early stopping)

    _MODEL_NAMES = {
        "chemprop_base": {},
        "chemprop_depth6": {"depth": 6},
        "chemprop_1kwidth": {"message_hidden_dim": 1024},
        "chemprop_2kwidth": {"message_hidden_dim": 2048},
        "chemprop_2kwidth_depth6": {"depth": 6, "message_hidden_dim": 2048}
    }

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
                single_task_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL
            ),
            total=_n_folds,
            desc="CV folds",
            unit="fold",
        )

        for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
            # Extract SMILES lists used by Chemprop-based models
            _smi_train = _train_raw["smiles"].to_list()
            _smi_val   = _val_raw["smiles"].to_list()
            _smi_test  = _test_raw["smiles"].to_list()

            _y_train = _train_raw[_TARGET_COL].to_numpy()
            _y_val   = _val_raw[_TARGET_COL].to_numpy()
            _y_true  = _test_raw[_TARGET_COL].to_numpy()

            # ── train & predict each model ────────────────────────────────────
            for _model_name in _MODEL_NAMES.keys():
                _params = _MODEL_NAMES[_model_name]
                _pbar.set_postfix({"fold": _fold, "o": _outer, "i": _inner, "model": _model_name}, refresh=False)

                _model = ChempropModel(pred_type="regression", **_params)
                _model.train(_smi_train, _y_train, _smi_val, _y_val, target_col=_TARGET_COL)
                _y_pred = _model.predict(_smi_test)

                # Free model memory before accumulating results
                del _model
                gc.collect()

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



        # ── write predictions (gzip-compressed directly) ──────────────────────
        _pred_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _pred_df.write_csv(_f)
        print(f"\nSaved {len(_pred_df):,} prediction rows → {_PRED_PATH_GZ}")

    def _summarise(df):
        return (
            calc_regression_metrics(
                df.rename({"model": "method", "fold": "cv_cycle"})
                  .with_columns(pl.lit("random").alias("split")),
                cycle_col="cv_cycle",
                val_col="y_true",
                pred_col="y_pred",
                thresh=4.0,
            )
            .group_by("method")
            .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
            .sort("rho", descending=True)
        )

    _summarise(_pred_df)
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl, rm_tukey_hsd):
    """
    Control check: verify that chemprop_base in the size-test run reproduces the
    chemprop model from the baseline run. Both were trained with the same
    architecture and the same 5×5 random CV splits, so they should be
    statistically indistinguishable. A non-significant Tukey HSD result
    confirms the two runs are consistent before we compare model sizes.
    """
    _BASELINE    = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _SIZE_TEST   = Path("../predictions/3_larger_chemprop_size_test.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]

    _ctrl_df = pl.concat([
        pl.read_csv(_BASELINE)
            .filter(pl.col("model") == "chemprop")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
        pl.read_csv(_SIZE_TEST)
            .filter(pl.col("model") == "chemprop_base")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
    ], how="diagonal")

    _metrics = calc_regression_metrics(
        _ctrl_df, cycle_col="cv_cycle", val_col="y_true",
        pred_col="y_pred", thresh=4.0,
    )

    _tukey_tables = []
    for _metric in _METRIC_LIST:
        _result_tab, _df_means, _, _ = rm_tukey_hsd(_metrics, _metric, group_col="method")
        _tukey_tables.append(
            mo.vstack([
                mo.md(f"**{_metric.upper()}** — mean chemprop: "
                      f"`{_df_means.loc['chemprop', _metric]:.4f}`, "
                      f"mean chemprop_base: "
                      f"`{_df_means.loc['chemprop_base', _metric]:.4f}`"),
                mo.plain_text(
                    _result_tab[["meandiff", "lower", "upper", "p-adj"]]
                    .to_string()
                ),
            ])
        )

    mo.vstack([
        mo.md("## Control: `chemprop` (baseline) vs `chemprop_base` (size test)"),
        mo.md("Expecting **no significant difference** (p-adj > 0.05) for all metrics."),
        *_tukey_tables,
    ])
    return


@app.cell
def _(Path, mo, pl):
    """
    Fold membership check: verify that the exact same set of compounds
    appears in each fold across both prediction files. A mismatch would
    mean the CV splits differ between the two runs and all metric
    comparisons would be invalid.
    """
    _BASELINE  = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _SIZE_TEST = Path("../predictions/3_larger_chemprop_size_test.csv.gz")

    _base = (
        pl.read_csv(_BASELINE)
        .filter(pl.col("model") == "chemprop")
        .select(["fold", "inchikey"])
        .unique()
    )
    _size = (
        pl.read_csv(_SIZE_TEST)
        .filter(pl.col("model") == "chemprop_base")
        .select(["fold", "inchikey"])
        .unique()
    )

    # Anti-join: compounds in baseline not in size-test for the same fold
    _only_in_base = _base.join(_size, on=["fold", "inchikey"], how="anti")
    # And the reverse
    _only_in_size = _size.join(_base, on=["fold", "inchikey"], how="anti")

    _n_folds_base = _base["fold"].n_unique()
    _n_folds_size = _size["fold"].n_unique()
    _n_compounds_base = _base.group_by("fold").len().sort("fold")
    _n_compounds_size = _size.group_by("fold").len().sort("fold")

    if len(_only_in_base) == 0 and len(_only_in_size) == 0:
        _status = mo.md("✅ **All folds match exactly** — same compounds in every fold across both files.")
    else:
        _status = mo.vstack([
            mo.md(f"❌ **Mismatch detected!**"),
            mo.md(f"Compounds in baseline but not size-test: {len(_only_in_base)}"),
            mo.plain_text(str(_only_in_base)),
            mo.md(f"Compounds in size-test but not baseline: {len(_only_in_size)}"),
            mo.plain_text(str(_only_in_size)),
        ])

    mo.vstack([
        mo.md("## Fold membership check: baseline vs size-test"),
        mo.md(f"Baseline folds: {_n_folds_base} | Size-test folds: {_n_folds_size}"),
        _status,
    ])
    return


@app.cell
def _(
    Path,
    calc_regression_metrics,
    make_boxplots_nonparametric,
    make_mcs_plot_grid,
    mo,
    pl,
):
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]
    _PLOT_DIR = Path("../plots/3_ml_optimization")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
    _metrics = calc_regression_metrics(_cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0)

    mo.vstack([
        mo.as_html(make_boxplots_nonparametric(_metrics, _METRIC_LIST,
            save_path=_PLOT_DIR / "chemprop_size_boxplots.png")),
        mo.as_html(make_mcs_plot_grid(
            _metrics, stats=_METRIC_LIST, group_col="method",
            figsize=(13, 12),
            show_diff=True, sort_axes=True,
            save_path=_PLOT_DIR / "chemprop_size_mcs.png",
        )),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This took around 12 hours in a M4 mac mini
    """)
    return


@app.cell
def _():
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare tree models with all different fingerprints
    """)
    return


@app.cell
def _(
    Path,
    RandomForestModel,
    calc_regression_metrics,
    extract_fp_matrix,
    gc,
    generate_cv_splits_random,
    generate_fingerprint,
    gzip,
    pl,
    single_task_train,
    tqdm,
):
    # ── output path ────────────────────────────────────────────────────────────
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/3_rf_fingerprint_comparison.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1          # fraction of train kept as validation — unused by RF but ensures identical splits to other cells

    # Allow multiple OpenMP runtimes (PyTorch libkmp + sklearn libkmp) to coexist
    # in the same process. Without this, the two runtimes collide on kmp barrier
    # synchronisation and segfault (EXC_BAD_ACCESS in __kmp_fork_barrier).
    # This has no effect on RF parallelism — trees are parallelised by joblib's
    # loky process pool, which is completely independent of OpenMP.
    import os as _os
    _os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    # ── fingerprint variants ───────────────────────────────────────────────────
    # Each entry: (fingerprint_type passed to generate_fingerprint, kwargs dict,
    #              column name used to retrieve the feature matrix)
    # The column name equals the fingerprint_type string (set by generate_fingerprint).
    # For CheMeleon we bypass generate_fingerprint and call the class directly.
    #
    # Design rationale per fingerprint family:
    #   ECFP  — vary radius (2=ECFP4, 3=ECFP6) × bit length (1k/2k/4k)
    #            + count variant captures repetitive substructures better
    #            + chirality and FCFP pharmacophoric invariants as alternatives
    #   MACCS — fixed 167 structural keys; test bits vs counts
    #   TopologicalTorsion — vary bit length; counts encode frequency
    #   RDKit path FP — vary path length and bit length; counts
    #   AtomPair — vary bit length; counts capture pair frequencies
    #   Avalon    — vary bit length (native default is 512, much smaller than others)
    #   Mordred   — 2D (1613 descriptors) and 3D (1826 descriptors). The 3D variant
    #               sets requires_conformers=True internally but generates its own
    #               conformers via RDKit ETKDGv3 — no pre-computed conformers needed.
    #   MQNs      — 42 integer counts, no size parameter; single variant
    #   PubChem   — 881-bit CACTVS structural keys; test bits vs counts
    #   CheMeleon — 2048-d learned embeddings; single variant (no tunable params)
    #
    # Each family always includes a base entry with empty kwargs (skfp defaults).
    # count+chirality combinations test whether frequency information and stereo
    # awareness interact constructively across different bit-vector sizes.

    _FP_VARIANTS: list[tuple[str, dict, str]] = [
        # ── ECFP / FCFP ───────────────────────────────────────────────────────
        # Base (skfp defaults: radius=2, fp_size=2048, bits)
        ("ecfp", {},                                                           "ecfp_base"),
        # Radius × size
        ("ecfp", {"radius": 2, "fp_size": 1024},                              "ecfp_r2_1k"),
        ("ecfp", {"radius": 2, "fp_size": 2048},                              "ecfp_r2_2k"),
        ("ecfp", {"radius": 2, "fp_size": 4096},                              "ecfp_r2_4k"),
        ("ecfp", {"radius": 3, "fp_size": 1024},                              "ecfp_r3_1k"),
        ("ecfp", {"radius": 3, "fp_size": 2048},                              "ecfp_r3_2k"),
        ("ecfp", {"radius": 3, "fp_size": 4096},                              "ecfp_r3_4k"),
        # Count only
        ("ecfp", {"radius": 2, "fp_size": 1024, "count": True},               "ecfp_r2_1k_count"),
        ("ecfp", {"radius": 2, "fp_size": 2048, "count": True},               "ecfp_r2_2k_count"),
        ("ecfp", {"radius": 2, "fp_size": 4096, "count": True},               "ecfp_r2_4k_count"),
        ("ecfp", {"radius": 3, "fp_size": 1024, "count": True},               "ecfp_r3_1k_count"),
        ("ecfp", {"radius": 3, "fp_size": 2048, "count": True},               "ecfp_r3_2k_count"),
        ("ecfp", {"radius": 3, "fp_size": 4096, "count": True},               "ecfp_r3_4k_count"),
        # Chirality only
        ("ecfp", {"radius": 2, "fp_size": 1024, "include_chirality": True},   "ecfp_r2_1k_chiral"),
        ("ecfp", {"radius": 2, "fp_size": 2048, "include_chirality": True},   "ecfp_r2_2k_chiral"),
        ("ecfp", {"radius": 2, "fp_size": 4096, "include_chirality": True},   "ecfp_r2_4k_chiral"),
        ("ecfp", {"radius": 3, "fp_size": 2048, "include_chirality": True},   "ecfp_r3_2k_chiral"),
        # Count + chirality
        ("ecfp", {"radius": 2, "fp_size": 1024, "count": True, "include_chirality": True}, "ecfp_r2_1k_count_chiral"),
        ("ecfp", {"radius": 2, "fp_size": 2048, "count": True, "include_chirality": True}, "ecfp_r2_2k_count_chiral"),
        ("ecfp", {"radius": 2, "fp_size": 4096, "count": True, "include_chirality": True}, "ecfp_r2_4k_count_chiral"),
        ("ecfp", {"radius": 3, "fp_size": 2048, "count": True, "include_chirality": True}, "ecfp_r3_2k_count_chiral"),
        # Pharmacophoric invariants (FCFP)
        ("ecfp", {"radius": 2, "fp_size": 2048, "use_fcfp": True},            "fcfp_r2_2k"),
        ("ecfp", {"radius": 3, "fp_size": 2048, "use_fcfp": True},            "fcfp_r3_2k"),
        ("ecfp", {"radius": 2, "fp_size": 2048, "use_fcfp": True, "count": True}, "fcfp_r2_2k_count"),
        # ── MACCS structural keys (fixed 167 bits) ────────────────────────────
        ("maccs", {},                                                           "maccs_base"),
        ("maccs", {"count": True},                                             "maccs_count"),
        # ── Topological Torsion ───────────────────────────────────────────────
        ("torsion", {},                                                         "torsion_base"),
        ("torsion", {"fp_size": 1024},                                         "torsion_1k"),
        ("torsion", {"fp_size": 2048},                                         "torsion_2k"),
        ("torsion", {"fp_size": 4096},                                         "torsion_4k"),
        ("torsion", {"fp_size": 1024, "count": True},                          "torsion_1k_count"),
        ("torsion", {"fp_size": 2048, "count": True},                          "torsion_2k_count"),
        ("torsion", {"fp_size": 4096, "count": True},                          "torsion_4k_count"),
        ("torsion", {"fp_size": 1024, "include_chirality": True},              "torsion_1k_chiral"),
        ("torsion", {"fp_size": 2048, "include_chirality": True},              "torsion_2k_chiral"),
        ("torsion", {"fp_size": 1024, "count": True, "include_chirality": True}, "torsion_1k_count_chiral"),
        ("torsion", {"fp_size": 2048, "count": True, "include_chirality": True}, "torsion_2k_count_chiral"),
        # ── RDKit path fingerprint ─────────────────────────────────────────────
        ("rdkit", {},                                                           "rdkit_base"),
        ("rdkit", {"fp_size": 1024, "max_path": 7},                            "rdkit_1k_p7"),
        ("rdkit", {"fp_size": 2048, "max_path": 5},                            "rdkit_2k_p5"),
        ("rdkit", {"fp_size": 2048, "max_path": 7},                            "rdkit_2k_p7"),
        ("rdkit", {"fp_size": 2048, "max_path": 10},                           "rdkit_2k_p10"),
        ("rdkit", {"fp_size": 4096, "max_path": 7},                            "rdkit_4k_p7"),
        ("rdkit", {"fp_size": 1024, "max_path": 7, "count": True},             "rdkit_1k_p7_count"),
        ("rdkit", {"fp_size": 2048, "max_path": 7, "count": True},             "rdkit_2k_p7_count"),
        ("rdkit", {"fp_size": 4096, "max_path": 7, "count": True},             "rdkit_4k_p7_count"),
        # ── Atom Pair ─────────────────────────────────────────────────────────
        ("atompair", {},                                                        "atompair_base"),
        ("atompair", {"fp_size": 1024},                                        "atompair_1k"),
        ("atompair", {"fp_size": 2048},                                        "atompair_2k"),
        ("atompair", {"fp_size": 4096},                                        "atompair_4k"),
        ("atompair", {"fp_size": 1024, "count": True},                         "atompair_1k_count"),
        ("atompair", {"fp_size": 2048, "count": True},                         "atompair_2k_count"),
        ("atompair", {"fp_size": 4096, "count": True},                         "atompair_4k_count"),
        ("atompair", {"fp_size": 1024, "include_chirality": True},             "atompair_1k_chiral"),
        ("atompair", {"fp_size": 2048, "include_chirality": True},             "atompair_2k_chiral"),
        ("atompair", {"fp_size": 1024, "count": True, "include_chirality": True}, "atompair_1k_count_chiral"),
        ("atompair", {"fp_size": 2048, "count": True, "include_chirality": True}, "atompair_2k_count_chiral"),
        # ── Avalon ────────────────────────────────────────────────────────────
        ("avalon", {},                                                          "avalon_base"),
        ("avalon", {"fp_size": 512},                                           "avalon_512"),
        ("avalon", {"fp_size": 1024},                                          "avalon_1k"),
        ("avalon", {"fp_size": 2048},                                          "avalon_2k"),
        ("avalon", {"fp_size": 512,  "count": True},                           "avalon_512_count"),
        ("avalon", {"fp_size": 1024, "count": True},                           "avalon_1k_count"),
        ("avalon", {"fp_size": 2048, "count": True},                           "avalon_2k_count"),
        # ── Mordred descriptors ───────────────────────────────────────────────
        # 2D: 1613 descriptors, no conformers needed
        # 3D: 1826 descriptors, conformers generated internally by skfp
        ("mordred", {},                                                         "mordred_base"),
        ("mordred", {"use_3D": True},                                          "mordred_3d"),
        # ── MQNs (42 integer counts, no tunable params) ───────────────────────
        ("mqn", {},                                                             "mqn"),
        # ── PubChem CACTVS structural keys ────────────────────────────────────
        ("pubchem", {},                                                         "pubchem_base"),
        ("pubchem", {"count": True},                                           "pubchem_count"),
        # ── CheMeleon learned embedding (2048-d, single variant) ──────────────
        ("chemeleon", {},                                                       "chemeleon"),
    ]

    if _PRED_PATH_GZ.exists():
        print(f"Predictions already exist at {_PRED_PATH_GZ} — skipping training.")
        _pred_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        _all_records: list[dict] = []
        _n_folds = _N_OUTER * _N_INNER

        _VARIANTS_SKL       = [(t, k, n) for t, k, n in _FP_VARIANTS if t != "chemeleon"]
        _VARIANTS_CHEMELEON = [(t, k, n) for t, k, n in _FP_VARIANTS if t == "chemeleon"]

        # ── pass 1: chemeleon embeddings in an isolated subprocess ────────────
        # PyTorch's libkmp and sklearn's libkmp corrupt each other's global
        # scheduler state when both are loaded in the same process, even with
        # KMP_DUPLICATE_LIB_OK=TRUE. multiprocessing.Process can't be used from
        # a marimo cell because cell-local functions aren't picklable by spawn.
        # Solution: subprocess.run with a self-contained Python script string.
        # SMILES are written to a temp JSON file; embeddings are written back as
        # .npy files by the child, read by the parent, then temp files deleted.
        import json as _json
        import subprocess as _subprocess
        import sys as _sys
        import tempfile as _tempfile
        import numpy as _np_sub

        # Self-contained script written to a temp file and executed as a child
        # process — no imports from the parent's namespace, kmp never shared.
        # Written to a .py file (not -c) so indentation in the source is exact.
        _SCRIPT_LINES = [
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
        ]

        _SCRIPT_PATH = Path(_tempfile.gettempdir()) / "chemeleon_embed.py"
        _SCRIPT_PATH.write_text("\n".join(_SCRIPT_LINES))

        def _chemeleon_embed_subprocess(smiles_train, smiles_test):
            """Run CheMeleon inference in an isolated subprocess."""
            tmp = Path(_tempfile.gettempdir())
            smi_file   = tmp / "ch_smiles.json"
            train_file = tmp / "ch_train"
            test_file  = tmp / "ch_test"
            smi_file.write_text(_json.dumps({"train": smiles_train, "test": smiles_test}))
            result = _subprocess.run(
                [_sys.executable, str(_SCRIPT_PATH),
                 str(smi_file), str(train_file), str(test_file)],
                capture_output=True, text=True,
            )
            if result.returncode != 0:
                raise RuntimeError(f"CheMeleon subprocess failed:\n{result.stderr}")
            X_train = _np_sub.load(str(train_file) + ".npy")
            X_test  = _np_sub.load(str(test_file)  + ".npy")
            smi_file.unlink(missing_ok=True)
            Path(str(train_file) + ".npy").unlink(missing_ok=True)
            Path(str(test_file)  + ".npy").unlink(missing_ok=True)
            return X_train, X_test

        _pbar1 = tqdm(
            generate_cv_splits_random(
                single_task_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL,
            ),
            total=_n_folds,
            desc="CV folds (CheMeleon)",
            unit="fold",
        )

        for _fold, _outer, _inner, _train_raw, _, _test_raw in _pbar1:
            _y_train = _train_raw[_TARGET_COL].to_numpy()
            _y_true  = _test_raw[_TARGET_COL].to_numpy()

            for _fp_type, _fp_kwargs, _fp_col in _VARIANTS_CHEMELEON:
                _pbar1.set_postfix({"fold": _fold, "fp": _fp_col}, refresh=False)

                _X_train, _X_test = _chemeleon_embed_subprocess(
                    _train_raw["smiles"].to_list(),
                    _test_raw["smiles"].to_list(),
                )

                _model = RandomForestModel(
                    pred_type="regression",
                    n_estimators=500,
                    random_state=_SEED,
                )
                _model.train(_X_train, _y_train)
                _y_pred = _model.predict(_X_test)
                del _model, _X_train, _X_test
                gc.collect()

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
                        "model":          _fp_col,
                        "y_true":         _yt,
                        "y_pred":         _yp,
                    })

        # ── pass 2: all sklearn fingerprints (RF with n_jobs=-1) ──────────────
        _pbar2 = tqdm(
            generate_cv_splits_random(
                single_task_train, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL,
            ),
            total=_n_folds,
            desc="CV folds (sklearn FPs)",
            unit="fold",
        )

        from concurrent.futures import ThreadPoolExecutor as _TPE

        for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar2:
            _y_train = _train_raw[_TARGET_COL].to_numpy()
            _y_true  = _test_raw[_TARGET_COL].to_numpy()

            for _fp_type, _fp_kwargs, _fp_col in _VARIANTS_SKL:
                _pbar2.set_postfix({"fold": _fold, "fp": _fp_col}, refresh=False)

                # Compute train and test fingerprints concurrently (2 threads).
                # skfp's RDKit C++ code releases the GIL so this gives real
                # parallelism without holding more than one variant in memory.
                with _TPE(max_workers=2) as _pool:
                    _ft = _pool.submit(generate_fingerprint, _train_raw, _fp_type, **_fp_kwargs)
                    _fs = _pool.submit(generate_fingerprint, _test_raw,  _fp_type, **_fp_kwargs)
                    _train_fp, _test_fp = _ft.result(), _fs.result()

                _X_train = extract_fp_matrix(_train_fp, _fp_type)
                _X_test  = extract_fp_matrix(_test_fp,  _fp_type)
                del _train_fp, _test_fp

                _model = RandomForestModel(
                    pred_type="regression",
                    n_estimators=500,
                    random_state=_SEED,
                )
                _model.train(_X_train, _y_train)
                _y_pred = _model.predict(_X_test)
                del _model, _X_train, _X_test
                gc.collect()

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
                        "model":          _fp_col,
                        "y_true":         _yt,
                        "y_pred":         _yp,
                    })

        # ── write compressed predictions ───────────────────────────────────────
        _pred_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _pred_df.write_csv(_f)
        print(f"\nSaved {len(_pred_df):,} prediction rows → {_PRED_PATH_GZ}")

    def _summarise(df):
        return (
            calc_regression_metrics(
                df.rename({"model": "method", "fold": "cv_cycle"})
                  .with_columns(pl.lit("random").alias("split")),
                cycle_col="cv_cycle",
                val_col="y_true",
                pred_col="y_pred",
                thresh=4.0,
            )
            .group_by("method")
            .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
            .sort("rho", descending=True)
        )

    _summarise(_pred_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This takes very long (~8hours) probably could have optimized and paralelized the code better
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl, rm_tukey_hsd):
    _BASELINE    = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _RF_FP_PATH  = Path("../predictions/3_rf_fingerprint_comparison.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]
    _ALPHA       = 0.05  # Tukey HSD significance threshold

    # ── load and reshape predictions ───────────────────────────────────────────
    chemeleon_ref_df = (
        pl.read_csv(_BASELINE)
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    rf_fp_df = (
        pl.read_csv(_RF_FP_PATH)
        .with_columns(
            pl.when(pl.col("model") == "chemeleon")
            .then(pl.lit("chemeleon_fp"))
            .otherwise(pl.col("model"))
            .alias("model")
        )
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )

    # ── fingerprint family membership ─────────────────────────────────────────
    # Multi-variant families get an intra-family boxplot + best-model selection.
    # Small families (≤2 variants) skip the boxplot and select best by mean rho.
    _MULTI_FAMILIES: dict[str, list[str]] = {
        "ECFP/FCFP": [
            "ecfp_base",
            "ecfp_r2_1k", "ecfp_r2_2k", "ecfp_r2_4k",
            "ecfp_r3_1k", "ecfp_r3_2k", "ecfp_r3_4k",
            "ecfp_r2_1k_count", "ecfp_r2_2k_count", "ecfp_r2_4k_count",
            "ecfp_r3_1k_count", "ecfp_r3_2k_count", "ecfp_r3_4k_count",
            "ecfp_r2_1k_chiral", "ecfp_r2_2k_chiral", "ecfp_r2_4k_chiral", "ecfp_r3_2k_chiral",
            "ecfp_r2_1k_count_chiral", "ecfp_r2_2k_count_chiral",
            "ecfp_r2_4k_count_chiral", "ecfp_r3_2k_count_chiral",
            "fcfp_r2_2k", "fcfp_r3_2k", "fcfp_r2_2k_count",
        ],
        "Torsion": [
            "torsion_base",
            "torsion_1k", "torsion_2k", "torsion_4k",
            "torsion_1k_count", "torsion_2k_count", "torsion_4k_count",
            "torsion_1k_chiral", "torsion_2k_chiral",
            "torsion_1k_count_chiral", "torsion_2k_count_chiral",
        ],
        "RDKit": [
            "rdkit_base",
            "rdkit_1k_p7", "rdkit_2k_p5", "rdkit_2k_p7", "rdkit_2k_p10", "rdkit_4k_p7",
            "rdkit_1k_p7_count", "rdkit_2k_p7_count", "rdkit_4k_p7_count",
        ],
        "AtomPair": [
            "atompair_base",
            "atompair_1k", "atompair_2k", "atompair_4k",
            "atompair_1k_count", "atompair_2k_count", "atompair_4k_count",
            "atompair_1k_chiral", "atompair_2k_chiral",
            "atompair_1k_count_chiral", "atompair_2k_count_chiral",
        ],
        "Avalon": [
            "avalon_base",
            "avalon_512", "avalon_1k", "avalon_2k",
            "avalon_512_count", "avalon_1k_count", "avalon_2k_count",
        ],
    }
    _SMALL_FAMILIES: dict[str, list[str]] = {
        "MACCS":   ["maccs_base", "maccs_count"],
        "Mordred": ["mordred_base", "mordred_3d"],
        "MQN":     ["mqn"],
        "PubChem": ["pubchem_base", "pubchem_count"],
        "Chemeleon_fp": ["chemeleon_fp"]
    }

    def _best_by_mae(metrics_df: pl.DataFrame) -> str:
        """Return the method with the lowest mean MAE across folds."""
        return (
            metrics_df.group_by("method")
            .agg(pl.col("mae").mean())
            .sort("mae")
            .row(0)[0]
        )

    def _non_sig_diff_from_best(metrics_df: pl.DataFrame, best: str) -> set:
        """Return methods not significantly different from best on MAE (Tukey HSD).

        Returns an empty set if the test cannot be computed (e.g. all models
        produce identical predictions, causing zero MSresidual in the RM-ANOVA).
        """
        try:
            result_tab, _, _, _ = rm_tukey_hsd(
                metrics_df, "mae", group_col="method", alpha=_ALPHA
            )
        except Exception:
            return set()
        mask = (result_tab["group1"] == best) | (result_tab["group2"] == best)
        not_sig = result_tab[mask & (result_tab["p-adj"] > _ALPHA)]
        others = set()
        for _, row in not_sig.iterrows():
            others.add(row["group2"] if row["group1"] == best else row["group1"])
        return others

    # ── stage 1: intra-family comparison ──────────────────────────────────────
    best_models_rf_fp: dict[str, str] = {}
    _family_plots: list = []

    for _family, _variants in _MULTI_FAMILIES.items():
        _family_df = rf_fp_df.filter(pl.col("method").is_in(_variants))
        _family_metrics = calc_regression_metrics(
            _family_df, cycle_col="cv_cycle", val_col="y_true",
            pred_col="y_pred", thresh=4.0,
        )
        _best = _best_by_mae(_family_metrics)
        best_models_rf_fp[_family] = _best

        # Keep only the best + variants not significantly different from it on MAE
        _show = {_best} | _non_sig_diff_from_best(_family_metrics, _best)
        _n_total = _family_metrics["method"].n_unique()
        _n_show  = len(_show)
        _display_metrics = _family_metrics.filter(pl.col("method").is_in(_show))

        _equiv = sorted(_show - {_best})
        _equiv_str = ", ".join(f"`{m}`" for m in _equiv) if _equiv else "none"
        _plots = [
            mo.md(
                f"### {_family}\n\n"
                f"**Best (lowest MAE):** `{_best}`\n\n"
                f"**Not significantly different (p > {_ALPHA}):** {_equiv_str}\n\n"
                f"*{_n_total - _n_show} of {_n_total} variants excluded as significantly worse.*"
            ),
        ]
        _family_plots.append(mo.vstack(_plots))

    # For small families pick best by MAE directly (too few variants for Tukey)
    for _family, _variants in _SMALL_FAMILIES.items():
        _fam_df = rf_fp_df.filter(pl.col("method").is_in(_variants))
        _fam_metrics = calc_regression_metrics(
            _fam_df, cycle_col="cv_cycle", val_col="y_true",
            pred_col="y_pred", thresh=4.0,
        )
        best_models_rf_fp[_family] = _best_by_mae(_fam_metrics)

    mo.vstack(_family_plots)
    return best_models_rf_fp, chemeleon_ref_df, rf_fp_df


@app.cell
def _(
    Path,
    best_models_rf_fp: dict[str, str],
    calc_regression_metrics,
    chemeleon_ref_df,
    make_mcs_plot_grid,
    mo,
    pl,
    rf_fp_df,
):
    _METRIC_LIST = ["mae"]
    _PLOT_DIR = Path("../plots/3_ml_optimization")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

    # ── stage 2: cross-family comparison (best per family + chemeleon) ─────────
    _best_names = list(best_models_rf_fp.values())
    _cross_df = pl.concat([
        rf_fp_df.filter(pl.col("method").is_in(_best_names)),
        chemeleon_ref_df,
    ])
    _cross_metrics = calc_regression_metrics(
        _cross_df, cycle_col="cv_cycle", val_col="y_true",
        pred_col="y_pred", thresh=4.0,
    )

    mo.vstack([
        mo.md("### Cross-family comparison — best per FP type + CheMeleon"),
        mo.md("Best variants selected: " + ", ".join(
            f"**{fam}** → `{mod}`" for fam, mod in best_models_rf_fp.items()
        )),
        mo.as_html(make_mcs_plot_grid(
            _cross_metrics, stats=_METRIC_LIST, group_col="method",
            figsize=(18, 16), show_diff=True, sort_axes=True, effect_dict={"mae": 0.1},
            save_path=_PLOT_DIR / "rf_fp_cross_family_mcs.png",
        )),
    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Compare scoring ensembles
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, gzip, np, pl):
    # ── paths ──────────────────────────────────────────────────────────────────
    _SRC_PATH  = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _OUT_PATH  = Path("../predictions/3_prediction_ensemble_test.csv.gz")

    # ── the four models we want to ensemble ───────────────────────────────────
    _ENSEMBLE_MODELS = ["rf", "gbm", "chemprop", "chemeleon"]

    # ── enumerate valid weight combinations ───────────────────────────────────
    # Each weight ∈ {0, 1, 2}. Two filters are applied:
    #   • not all weights == 2   (degenerate: equal to plain mean, name collision)
    #   • fewer than 3 weights == 0  (requires at least 2 active models)
    # This yields 71 distinct ensembles out of 3^4 = 81 raw combinations.
    # The chemeleon-only control (0,0,0,1) is appended explicitly — it is
    # excluded by the filters above but useful as a single-model reference.
    _all_weights: list[tuple[int, ...]] = [
        w
        for w in np.ndindex(3, 3, 3, 3)       # iterates (0..2)^4
        if not all(x == 2 for x in w)
        and sum(x == 0 for x in w) < 3
    ] + [(0, 0, 0, 1)] #added chemeleon only as control

    def _weight_label(weights: tuple[int, ...]) -> str:
        """Encode weights as a compact model string, e.g. 'ens_rf1_gbm2_cp0_ch1'."""
        tags = zip(["rf", "gbm", "cp", "ch"], weights)
        return "ens_" + "_".join(f"{tag}{w}" for tag, w in tags)

    if _OUT_PATH.exists():
        print(f"Ensemble predictions already exist at {_OUT_PATH} — skipping.")
        _ens_df = pl.read_csv(_OUT_PATH)
    else:
        # ── load source predictions and pivot to wide format ───────────────────
        # Keep only the four target models; the id columns identify each
        # (compound, fold) pair uniquely.
        _id_cols = ["inchikey", "molecule_names", "smiles", "fold", "outer_fold", "inner_fold"]

        _src = (
            pl.read_csv(_SRC_PATH)
            .filter(pl.col("model").is_in(_ENSEMBLE_MODELS))
        )

        # Wide table: one row per (compound × fold), one column per model's y_pred
        _wide = (
            _src
            .pivot(on="model", index=_id_cols + ["y_true"], values="y_pred")
            # Rename model columns to avoid collisions when building weighted sums
            .rename({m: f"_p_{m}" for m in _ENSEMBLE_MODELS})
        )

        # ── build all ensemble predictions ────────────────────────────────────
        _records = []

        for _w in _all_weights:
            _w_rf, _w_gbm, _w_cp, _w_ch = _w
            _total = float(_w_rf + _w_gbm + _w_cp + _w_ch)

            # Weighted mean as a Polars expression — avoids Python loops over rows
            _y_pred_expr = (
                pl.col("_p_rf")       * _w_rf
                + pl.col("_p_gbm")    * _w_gbm
                + pl.col("_p_chemprop") * _w_cp
                + pl.col("_p_chemeleon") * _w_ch
            ) / _total

            _records.append(
                _wide.select(
                    *_id_cols,
                    pl.col("y_true"),
                    _y_pred_expr.alias("y_pred"),
                    pl.lit(_weight_label(_w)).alias("model"),
                )
            )

        _ens_df = pl.concat(_records)

        _OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_OUT_PATH, "wb") as _f:
            _ens_df.write_csv(_f)
        print(
            f"Saved {len(_ens_df):,} rows "
            f"({len(_all_weights)} ensembles × {len(_wide):,} compound-folds) "
            f"→ {_OUT_PATH}"
        )

    def _summarise(df):
        return (
            calc_regression_metrics(
                df.rename({"model": "method", "fold": "cv_cycle"})
                  .with_columns(pl.lit("random").alias("split")),
                cycle_col="cv_cycle",
                val_col="y_true",
                pred_col="y_pred",
                thresh=4.0,
            )
            .group_by("method")
            .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
            .sort("rho", descending=True)
        )

    _summarise(_ens_df)
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl, rm_tukey_hsd):
    """
    Control check: verify that the chemeleon-only ensemble weight (0,0,0,1),
    labelled ens_rf0_gbm0_cp0_ch1, is statistically identical to the chemeleon
    model from the baseline run. Both are pure chemeleon predictions on the same
    CV folds, so any difference would indicate a data-pipeline bug.
    """
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _ENS_PATH = Path("../predictions/3_prediction_ensemble_test.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]

    _ctrl_df = pl.concat([
        pl.read_csv(_BASELINE)
            .filter(pl.col("model") == "chemeleon")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
        pl.read_csv(_ENS_PATH)
            .filter(pl.col("model") == "ens_rf0_gbm0_cp0_ch1")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
    ], how="diagonal")

    _metrics = calc_regression_metrics(
        _ctrl_df, cycle_col="cv_cycle", val_col="y_true",
        pred_col="y_pred", thresh=4.0,
    )

    _tukey_tables = []
    for _metric in _METRIC_LIST:
        _result_tab, _df_means, _, _ = rm_tukey_hsd(_metrics, _metric, group_col="method")
        _tukey_tables.append(
            mo.vstack([
                mo.md(f"**{_metric.upper()}** — mean chemeleon: "
                      f"`{_df_means.loc['chemeleon', _metric]:.4f}`, "
                      f"mean ens_rf0_gbm0_cp0_ch1: "
                      f"`{_df_means.loc['ens_rf0_gbm0_cp0_ch1', _metric]:.4f}`"),
                mo.plain_text(
                    _result_tab[["meandiff", "lower", "upper", "p-adj"]]
                    .to_string()
                ),
            ])
        )

    mo.vstack([
        mo.md("## Control: `chemeleon` (baseline) vs `ens_rf0_gbm0_cp0_ch1` (ensemble test)"),
        mo.md("Expecting **no significant difference** (p-adj > 0.05) for all metrics."),
        *_tukey_tables,
    ])
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl):
    _BASELINE    = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _ENS_PATH    = Path("../predictions/3_prediction_ensemble_test.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]
    _TOP_N       = 4   # models to show in plots alongside chemeleon

    _chemeleon_ref = (
        pl.read_csv(_BASELINE)
        .filter(pl.col("model") == "chemeleon")
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )
    _ens_df = (
        pl.read_csv(_ENS_PATH)
        .rename({"model": "method", "fold": "cv_cycle"})
        .with_columns(pl.lit("random").alias("split"))
    )

    # Compute metrics for all models + chemeleon via calc_regression_metrics
    _all_df = pl.concat([_ens_df, _chemeleon_ref], how="diagonal")
    _all_metrics = calc_regression_metrics(
        _all_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0,
    )

    # Mean MAE per model from the Tukey-consistent per-fold metrics
    _mean_mae = (
        _all_metrics
        .group_by("method").agg(pl.col("mae").mean())
        .sort("mae")
    )
    _chemeleon_mae = _mean_mae.filter(pl.col("method") == "chemeleon")["mae"][0]

    # All ensemble models that beat chemeleon on mean MAE, ranked best first
    _better_df = _mean_mae.filter(
        (pl.col("method") != "chemeleon") & (pl.col("mae") < _chemeleon_mae)
    )
    _n_better = len(_better_df)
    _better_list = _better_df["method"].to_list()   # already sorted by mae asc

    # Select top N for plots
    _plot_models = _better_list[:_TOP_N]
    _plot_df = pl.concat([
        _ens_df.filter(pl.col("method").is_in(_plot_models)),
        _chemeleon_ref,
    ], how="diagonal")
    _plot_metrics = calc_regression_metrics(
        _plot_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0,
    )

    # Build the text summary of all improving models
    _better_rows = "\n".join(
        f"- `{row[0]}` — MAE {row[1]:.4f}"
        for row in _better_df.iter_rows()
    )
    _summary = (
        f"**CheMeleon baseline MAE:** {_chemeleon_mae:.4f}\n\n"
        f"**{_n_better} ensemble(s) with lower MAE** (ranked best first):\n\n"
        + (_better_rows if _n_better else "*none*")

    )

    mo.vstack([
        mo.md("### Ensemble vs CheMeleon"),
        mo.md(_summary),

    ])
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Multitask test
    """)
    return


@app.cell
def _(
    Path,
    calc_regression_metrics,
    gc,
    generate_cv_splits_random,
    gzip,
    pl,
    shutil,
    single_task_train,
    subprocess,
    sys,
    tempfile,
    torch,
    tqdm,
):
    # ── multitask chemprop helpers ─────────────────────────────────────────────
    # We re-use _get_device and _run_chemprop_cli patterns from ChempropModel but
    # inline them here to keep the cell self-contained (those are cell-private).

    _MT_CHEMPROP_BIN = Path(sys.executable).parent / "chemprop"
    _MT_MODEL_DIR    = Path(tempfile.gettempdir()) / "chemprop_multitask_model"
    _MT_LOG          = Path(tempfile.gettempdir()) / "chemprop_multitask_cli.log"

    def _mt_device() -> str:
        return (
            "cuda" if torch.cuda.is_available()
            else "mps" if torch.backends.mps.is_available()
            else "cpu"
        )

    def _mt_run_cli(args: list[str]) -> None:
        cmd = [str(_MT_CHEMPROP_BIN)] + args
        with open(_MT_LOG, "a") as _log:
            _log.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\n{'='*60}\n")
            result = subprocess.run(cmd, stdout=_log, stderr=_log, text=True)
        if result.returncode != 0:
            lines = _MT_LOG.read_text().splitlines()
            print("\n".join(lines[-30:]))
            raise RuntimeError(
                f"chemprop CLI failed (exit {result.returncode}). "
                f"Full log: {_MT_LOG}"
            )

    def _write_mt_csv(df, path, target_cols) -> None:
        """
        Write a multitask CSV with a smiles column and one column per target.

        Missing target values are written as empty cells, which chemprop reads
        as NaN and automatically masks during loss computation. Boolean is_hit
        columns are cast to 0/1 float so chemprop treats them as regression.

        Args:
            df: Polars DataFrame containing a "smiles" column and target columns.
            path: Output CSV path.
            target_cols: Names of the target columns to include.
        """
        out = df.select([
            pl.col("smiles"),
            *[
                # Cast boolean hit columns to 0/1 float; pass others through
                pl.col(c).cast(pl.Float64).alias(c)
                if df[c].dtype == pl.Boolean
                else pl.col(c).cast(pl.Float64).alias(c)
                for c in target_cols
            ],
        ])
        out.write_csv(path, null_value="")

    # ── multitask scenarios ────────────────────────────────────────────────────
    # Each scenario is a tuple of:
    #   (name, list_of_target_columns)
    # pEC50_dr is always the first target — predict() reads only that column.
    # Additional targets are auxiliary; NaN rows are masked automatically.
    # is_hit columns are treated as 0/1 regression targets.
    #
    # Training pool: all non-test rows that have at least one of the scenario's
    # targets present. This expands the training set for scenarios with hit data
    # (single-dose compounds without dose-response measurements).
    _SCENARIOS: list[tuple[str, list[str]]] = [
        ("st_pec50",      ["pEC50_dr"]),
        ("mt_emax_dr",    ["pEC50_dr", "Emax_dr"]),
        ("mt_counter",    ["pEC50_dr", "pEC50_counter", "Emax_counter"]),
        ("mt_hit10",      ["pEC50_dr", "10.0_is_hit"]),
        ("mt_hit30",      ["pEC50_dr", "30.0_is_hit"]),
        ("mt_hits",       ["pEC50_dr", "10.0_is_hit", "30.0_is_hit"]),
        ("mt_all_counter",["pEC50_dr", "pEC50_counter", "Emax_dr", "Emax_counter"]),
        ("mt_all",        ["pEC50_dr", "pEC50_counter", "Emax_dr", "Emax_counter",
                           "10.0_is_hit", "30.0_is_hit"]),
    ]

    # ── constants ──────────────────────────────────────────────────────────────
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/3_multitask_test.csv.gz")
    _N_OUTER = 5
    _N_INNER = 5
    _SEED    = 42
    _P_VAL   = 0.1      # kept identical to other cells; val split used for early stopping

    # CV splits are driven by the same single_task_train as all other cells.
    # Aux columns (Emax_dr, pEC50_counter, etc.) are loaded separately and
    # joined onto each fold by inchikey — single_task_train only carries the
    # four columns needed for CV splitting, not the full feature set.
    _cv_base = single_task_train

    # All aux target columns that any scenario may need
    _AUX_COLS = ["Emax_dr", "pEC50_counter", "Emax_counter", "10.0_is_hit", "30.0_is_hit"]
    _aux_data = (
        pl.read_csv("../data/processed/all_compounds_activity_data.csv")
        .filter(pl.col("in_test").not_() & pl.col("pEC50_dr").is_not_null())
        .select(["inchikey"] + _AUX_COLS)
    )

    if _PRED_PATH_GZ.exists():
        print(f"Predictions already exist at {_PRED_PATH_GZ} — skipping training.")
        _mt_pred_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        _all_records: list[dict] = []
        _n_folds = _N_OUTER * _N_INNER

        _pbar = tqdm(
            generate_cv_splits_random(
                _cv_base, n_outer=_N_OUTER, n_inner=_N_INNER, seed=_SEED, p_val=_P_VAL,
            ),
            total=_n_folds,
            desc="MT folds",
            unit="fold",
        )

        for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _pbar:
            # Attach aux columns — join is by inchikey, nulls stay null for masking
            _train_raw = _train_raw.join(_aux_data, on="inchikey", how="left")
            _val_raw   = _val_raw.join(_aux_data, on="inchikey", how="left")

            _smi_test = _test_raw["smiles"].to_list()
            _y_true   = _test_raw[_TARGET_COL].to_numpy()

            tmp = Path(tempfile.gettempdir())
            train_csv = tmp / "mt_train.csv"
            val_csv   = tmp / "mt_val.csv"
            test_csv  = tmp / "mt_test.csv"
            pred_csv  = tmp / "mt_preds.csv"

            for _scenario_name, _target_cols in _SCENARIOS:
                _pbar.set_postfix(
                    {"fold": _fold, "scenario": _scenario_name}, refresh=False
                )

                # Training and validation pools contain only compounds with
                # pEC50_dr, sliced to the same fold as all other CV cells.
                # Aux target columns are included as extra columns; rows where
                # those targets are null get an empty cell and are masked by
                # chemprop during loss computation.
                _train_pool = _train_raw
                _val_pool   = _val_raw

                _write_mt_csv(_train_pool, train_csv, _target_cols)
                _write_mt_csv(_val_pool,   val_csv,   _target_cols)

                # test CSV needs only smiles (no targets needed for prediction)
                pl.DataFrame({"smiles": _smi_test}).write_csv(test_csv)

                if _MT_MODEL_DIR.exists():
                    shutil.rmtree(_MT_MODEL_DIR)

                # All targets treated as regression; is_hit encoded as 0/1 float
                _mt_run_cli([
                    "train",
                    "--data-path", str(train_csv), str(val_csv), str(val_csv),
                    "--smiles-columns", "smiles",
                    "--target-columns", *_target_cols,
                    "--task-type", "regression",
                    "--accelerator", _mt_device(),
                    "--epochs", "50",
                    "--save-dir", str(_MT_MODEL_DIR),
                ])

                model_pt = _MT_MODEL_DIR / "model_0" / "best.pt"
                _mt_run_cli([
                    "predict",
                    "--test-path",  str(test_csv),
                    "--model-path", str(model_pt),
                    "--preds-path", str(pred_csv),
                ])

                # Read only pEC50_dr column from the predictions
                _y_pred = pl.read_csv(pred_csv)[_TARGET_COL].to_numpy().flatten()

                train_csv.unlink(missing_ok=True)
                val_csv.unlink(missing_ok=True)
                test_csv.unlink(missing_ok=True)
                pred_csv.unlink(missing_ok=True)
                gc.collect()

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
                        "model":          _scenario_name,
                        "y_true":         _yt,
                        "y_pred":         _yp,
                    })

        # ── write compressed output ────────────────────────────────────────────
        _mt_pred_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _mt_pred_df.write_csv(_f)
        print(f"\nSaved {len(_mt_pred_df):,} prediction rows → {_PRED_PATH_GZ}")

    def _summarise(df):
        return (
            calc_regression_metrics(
                df.rename({"model": "method", "fold": "cv_cycle"})
                  .with_columns(pl.lit("random").alias("split")),
                cycle_col="cv_cycle",
                val_col="y_true",
                pred_col="y_pred",
                thresh=4.0,
            )
            .group_by("method")
            .agg(pl.col(["mae", "mse", "r2", "rho"]).mean())
            .sort("rho", descending=True)
        )

    _summarise(_mt_pred_df)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    This cell took 5h to run on a Mac Mini
    """)
    return


@app.cell
def _(Path, calc_regression_metrics, mo, pl, rm_tukey_hsd):
    """
    Control check: verify that st_pec50 (single-task chemprop trained in the
    multitask cell) is statistically identical to chemprop from the baseline run.
    Both use the same architecture, the same CV splits, and the same single target,
    so any difference would indicate a data-pipeline inconsistency.
    """
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _MT_PATH  = Path("../predictions/3_multitask_test.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]

    _ctrl_df = pl.concat([
        pl.read_csv(_BASELINE)
            .filter(pl.col("model") == "chemprop")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
        pl.read_csv(_MT_PATH)
            .filter(pl.col("model") == "st_pec50")
            .rename({"model": "method", "fold": "cv_cycle"})
            .with_columns(pl.lit("random").alias("split")),
    ], how="diagonal")

    _metrics = calc_regression_metrics(
        _ctrl_df, cycle_col="cv_cycle", val_col="y_true",
        pred_col="y_pred", thresh=4.0,
    )

    _tukey_tables = []
    for _metric in _METRIC_LIST:
        _result_tab, _df_means, _, _ = rm_tukey_hsd(_metrics, _metric, group_col="method")
        _tukey_tables.append(
            mo.vstack([
                mo.md(f"**{_metric.upper()}** — mean chemprop: "
                      f"`{_df_means.loc['chemprop', _metric]:.4f}`, "
                      f"mean st_pec50: "
                      f"`{_df_means.loc['st_pec50', _metric]:.4f}`"),
                mo.plain_text(
                    _result_tab[["meandiff", "lower", "upper", "p-adj"]]
                    .to_string()
                ),
            ])
        )

    mo.vstack([
        mo.md("## Control: `chemprop` (baseline) vs `st_pec50` (multitask test)"),
        mo.md("Expecting **no significant difference** (p-adj > 0.05) for all metrics."),
        *_tukey_tables,
    ])
    return


@app.cell
def _(
    Path,
    calc_regression_metrics,
    make_boxplots_nonparametric,
    make_ci_plot_grid,
    make_mcs_plot_grid,
    mo,
    pl,
):
    _BASELINE = Path("../predictions/2_ml_baseline_5x5cv_random_predictions.csv.gz")
    _METRIC_LIST = ["mae", "mse", "r2", "rho"]
    _PLOT_DIR = Path("../plots/3_ml_optimization")
    _PLOT_DIR.mkdir(parents=True, exist_ok=True)

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
    _metrics = calc_regression_metrics(_cmp_df, cycle_col="cv_cycle", val_col="y_true", pred_col="y_pred", thresh=4.0)

    mo.vstack([
        mo.as_html(make_boxplots_nonparametric(_metrics, _METRIC_LIST,
            save_path=_PLOT_DIR / "multitask_boxplots.png")),
        mo.as_html(make_mcs_plot_grid(
            _metrics, stats=_METRIC_LIST, group_col="method",
            figsize=(13, 12),
            show_diff=True, sort_axes=True,
            save_path=_PLOT_DIR / "multitask_mcs.png",
        )),
        mo.as_html(make_ci_plot_grid(
            _metrics, metric_list=["mae"], group_col="method",
            figsize=(6, 14), xlim=(-0.05, 0.05),
            save_path=_PLOT_DIR / "multitask_ci_mae.png",
        )),
    ])
    return


@app.cell
def _(mo):
    mo.md(r"""
    # Train final models on the full training set and generate test-set submissions

    We train two new models on the full `single_task_train` dataset (all compounds
    with a measured pEC50\_dr) and generate predictions for the 513 held-out test
    compounds:

    1. **`chemprop_depth6`** — Chemprop D-MPNN from scratch with `depth=6` and
       default `message_hidden_dim=300` (matching the CV study configuration).
    2. **`rf_mordred3d`** — Random Forest trained on Mordred 3-D descriptors
       (1826 descriptors computed from RDKit ETKDGv3 conformers).

    These predictions are then combined with the existing CheMeleon baseline
    (`predictions/2_ml_baseline_chemeleon_test_submission.csv`) to build four
    weighted-mean ensemble submission files.
    """)
    return


@app.cell
def _(ChempropModel, Path, np, pl, single_task_train):
    """
    Train a Chemprop depth-6 model (depth=6, default message_hidden_dim=300) on the
    entire training set and generate predictions for the 513 held-out test
    compounds.

    A 10 % internal validation split is drawn for early stopping — this is NOT
    the competition test set.

    Output CSV columns: SMILES | Molecule Name | pEC50
    """

    _TARGET_COL = "pEC50_dr"
    _SEED       = 42
    _PRED_OUT   = Path("../predictions/3_chemprop_depth6_test_submission.csv")

    # ── load test set ──────────────────────────────────────────────────────────
    _test_df = pl.read_csv("../data/raw/20260409/dose_response_test.csv")

    # ── 10 % validation split for early stopping ──────────────────────────────
    _rng      = np.random.default_rng(_SEED)
    _n        = single_task_train.shape[0]
    _val_idx  = _rng.choice(_n, size=int(_n * 0.1), replace=False)
    _train_idx = np.setdiff1d(np.arange(_n), _val_idx)

    _train_sub = single_task_train[_train_idx]
    _val_sub   = single_task_train[_val_idx]

    _X_train = _train_sub["smiles"].to_list()
    _y_train = _train_sub[_TARGET_COL].to_numpy()
    _X_val   = _val_sub["smiles"].to_list()
    _y_val   = _val_sub[_TARGET_COL].to_numpy()
    _X_test  = _test_df["SMILES"].to_list()

    if _PRED_OUT.exists():
        print(f"Submission file already exists at {_PRED_OUT} — skipping training.")
    else:
        # depth=6 matches the chemprop_depth6 CV study configuration (message_hidden_dim=300 default)
        _model = ChempropModel(
            pred_type="regression",
            epochs=50,
            depth=6,
        )
        _model.train(_X_train, _y_train, _X_val, _y_val, target_col=_TARGET_COL)

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
def _(
    Path,
    RandomForestModel,
    extract_fp_matrix,
    generate_fingerprint,
    pl,
    single_task_train,
):
    """
    Train a Random Forest on Mordred 3-D descriptors on the entire training set
    and generate predictions for the 513 held-out test compounds.

    Mordred 3-D uses 1826 molecular descriptors; conformers are generated
    internally by scikit-fingerprints via RDKit ETKDGv3 (no pre-computed
    conformers needed).

    Output CSV columns: SMILES | Molecule Name | pEC50
    """
    #import os as _os_rf
    # Allow multiple OpenMP runtimes (PyTorch + sklearn) to coexist without crash
    #_os_rf.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

    _TARGET_COL = "pEC50_dr"
    _SEED       = 42
    _PRED_OUT   = Path("../predictions/3_rf_mordred3d_test_submission.csv")

    # ── load test set ──────────────────────────────────────────────────────────
    _test_df = pl.read_csv("../data/raw/20260409/dose_response_test.csv")
    # generate_fingerprint expects a "smiles" column (lowercase)
    _test_for_fp = _test_df.rename({"SMILES": "smiles"})

    if _PRED_OUT.exists():
        print(f"Submission file already exists at {_PRED_OUT} — skipping training.")
    else:
        # ── generate Mordred 3-D descriptors for train and test ────────────────
        _train_fp = generate_fingerprint(single_task_train, "mordred", use_3D=True)
        _test_fp  = generate_fingerprint(_test_for_fp, "mordred", use_3D=True)

        _X_train = extract_fp_matrix(_train_fp, "mordred")
        _y_train = single_task_train[_TARGET_COL].to_numpy()
        _X_test  = extract_fp_matrix(_test_fp, "mordred")

        # ── train Random Forest ────────────────────────────────────────────────
        _model = RandomForestModel(pred_type="regression", random_state=_SEED)
        _model.train(_X_train, _y_train)

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
def _(Path, np, pl):
    """
    Build four weighted-mean ensemble submission files by combining:
        • cp  — chemprop_depth6  (predictions/3_chemprop_depth6_test_submission.csv)
        • rf  — RF + Mordred 3-D (predictions/3_rf_mordred3d_test_submission.csv)
        • ch  — CheMeleon        (predictions/2_ml_baseline_chemeleon_test_submission.csv)

    The four ensembles requested, expressed as (rf, gbm, cp, ch) integer weights:
        ens_rf0_gbm0_cp1_ch2  →  (0, 0, 1, 2)
        ens_rf0_gbm0_cp1_ch1  →  (0, 0, 1, 1)
        ens_rf1_gbm0_cp2_ch2  →  (1, 0, 2, 2)
        ens_rf1_gbm0_cp1_ch2  →  (1, 0, 1, 2)

    Output files:
        predictions/3_ens_rf0_gbm0_cp1_ch2_submission.csv
        predictions/3_ens_rf0_gbm0_cp1_ch1_submission.csv
        predictions/3_ens_rf1_gbm0_cp2_ch2_submission.csv
        predictions/3_ens_rf1_gbm0_cp1_ch2_submission.csv
    """

    _CP_PATH  = Path("../predictions/3_chemprop_depth6_test_submission.csv")
    _RF_PATH  = Path("../predictions/3_rf_mordred3d_test_submission.csv")
    _CH_PATH  = Path("../predictions/2_ml_baseline_chemeleon_test_submission.csv")
    _OUT_DIR  = Path("../predictions")

    # ── load predictions from the three component models ──────────────────────
    _cp = pl.read_csv(_CP_PATH).rename({"pEC50": "pEC50_cp"})
    _rf = pl.read_csv(_RF_PATH).rename({"pEC50": "pEC50_rf"})
    _ch = pl.read_csv(_CH_PATH).rename({"pEC50": "pEC50_ch"})

    # Join on Molecule Name to align rows (test set order may differ across files)
    _merged = (
        _cp
        .join(_rf.select(["Molecule Name", "pEC50_rf"]), on="Molecule Name", how="left")
        .join(_ch.select(["Molecule Name", "pEC50_ch"]), on="Molecule Name", how="left")
    )

    # Weights: (w_rf, w_gbm, w_cp, w_ch) — gbm is absent, so w_gbm=0 always
    _ensembles: list[tuple[str, int, int, int, int]] = [
        ("ens_rf0_gbm0_cp1_ch2", 0, 0, 1, 2),
        ("ens_rf0_gbm0_cp1_ch1", 0, 0, 1, 1),
        ("ens_rf1_gbm0_cp2_ch2", 1, 0, 2, 2),
        ("ens_rf1_gbm0_cp1_ch2", 1, 0, 1, 2),
    ]

    for _name, _w_rf, _w_gbm, _w_cp, _w_ch in _ensembles:
        _out_path = _OUT_DIR / f"3_{_name}_submission.csv"
        if _out_path.exists():
            print(f"Already exists — skipping {_out_path.name}")
            continue

        _total = float(_w_rf + _w_gbm + _w_cp + _w_ch)
        _pec50_ens = (
            _merged["pEC50_rf"].to_numpy() * _w_rf
            + np.zeros(len(_merged)) * _w_gbm   # gbm weight is 0; placeholder
            + _merged["pEC50_cp"].to_numpy() * _w_cp
            + _merged["pEC50_ch"].to_numpy() * _w_ch
        ) / _total

        _sub = pl.DataFrame({
            "SMILES":        _merged["SMILES"].to_list(),
            "Molecule Name": _merged["Molecule Name"].to_list(),
            "pEC50":         _pec50_ens.tolist(),
        })
        _sub.write_csv(_out_path)
        print(f"Saved {len(_sub)} predictions → {_out_path.name}")
    return


@app.cell
def _(Iterable, Optional, Path, mo, np, pd):
    """
    Validate all five submission files produced in this section:
        • 3_chemprop_depth6_test_submission.csv
        • 3_rf_mordred3d_test_submission.csv
        • 3_ens_rf0_gbm0_cp1_ch2_submission.csv
        • 3_ens_rf0_gbm0_cp1_ch1_submission.csv
        • 3_ens_rf1_gbm0_cp2_ch2_submission.csv
        • 3_ens_rf1_gbm0_cp1_ch2_submission.csv

    Rules applied (matching the OpenADMET activity_validation.py spec):
        - Required columns: SMILES, Molecule Name, pEC50
        - No missing identifiers or duplicate Molecule Names
        - pEC50 must be numeric and finite
        - Exactly 513 rows
    """

    _ACTIVITY_DATASET_SIZE = 513

    _SUBMISSION_FILES = [
        Path("../predictions/3_chemprop_depth6_test_submission.csv"),
        Path("../predictions/3_rf_mordred3d_test_submission.csv"),
        Path("../predictions/3_ens_rf0_gbm0_cp1_ch2_submission.csv"),
        Path("../predictions/3_ens_rf0_gbm0_cp1_ch1_submission.csv"),
        Path("../predictions/3_ens_rf1_gbm0_cp2_ch2_submission.csv"),
        Path("../predictions/3_ens_rf1_gbm0_cp1_ch2_submission.csv"),
    ]

    def _as_set(values: Iterable[str]) -> set[str]:
        return {str(v) for v in values}

    def validate_activity_submission_3(
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
        elif len(activity_predictions) != _ACTIVITY_DATASET_SIZE:
            errors.append(
                f"Submission contains {len(activity_predictions)} rows, expected {_ACTIVITY_DATASET_SIZE}."
            )

        return len(errors) == 0, errors

    _results = []
    for _path in _SUBMISSION_FILES:
        _ok, _errs = validate_activity_submission_3(_path)
        if _ok:
            _results.append(mo.md(f"✓ **{_path.name}** — validation passed."))
        else:
            _results.append(mo.md(
                f"✗ **{_path.name}** — validation FAILED:\n" + "\n".join(f"- {e}" for e in _errs)
            ))

    mo.vstack(_results)
    return


if __name__ == "__main__":
    app.run()
