import marimo

__generated_with = "0.23.1"
app = marimo.App()


@app.cell
def _(mo):
    mo.md(r"""
    # Import libraries and import functions
    """)
    return


@app.cell
def _():
    import logging
    import joblib
    from abc import ABC, abstractmethod
    from pathlib import Path
    from typing import Literal
    from urllib.request import urlretrieve

    import polars as pl
    import marimo as mo
    import altair as alt
    import numpy as np

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

    return (
        ABC,
        AtomPairFingerprint,
        AvalonFingerprint,
        BaseKFold,
        Chem,
        ConformerGenerator,
        E3FPFingerprint,
        ECFPFingerprint,
        Literal,
        MACCSFingerprint,
        MQNsFingerprint,
        MolFromSmilesTransformer,
        MordredFingerprint,
        MurckoScaffold,
        Path,
        PubChemFingerprint,
        RDKitFingerprint,
        TopologicalTorsionFingerprint,
        abstractmethod,
        accuracy_score,
        alt,
        balanced_accuracy_score,
        f1_score,
        joblib,
        logging,
        matthews_corrcoef,
        mean_absolute_error,
        mean_squared_error,
        mo,
        np,
        pl,
        precision_score,
        r2_score,
        recall_score,
        roc_auc_score,
        spearmanr,
        urlretrieve,
    )


@app.cell
def _(mo):
    mo.md(r"""
    ## Model classes

    Four model types share a stable API:

    ```
    model.train(train_df, target_col, task, **kwargs)
    model.predict(df)          -> np.ndarray
    model.save(path)
    ModelClass.load(path)      -> ModelClass instance
    ```

    | Class | Backend | Input features |
    |---|---|---|
    | `RandomForestModel` | sklearn RF | fingerprint column |
    | `GradientBoostingModel` | LightGBM | fingerprint column |
    | `ChempropModel` | Chemprop v2 MPNN from scratch | SMILES column |
    | `ChempropChemeleonModel` | Chemprop v2 fine-tuned from [CheMeleon](https://github.com/JacksonBurns/chemeleon) backbone | SMILES column |

    `task` is either `"regression"` or `"classification"`.
    Classification `predict()` returns the probability of the positive class.
    """)
    return


@app.cell
def _(ABC, Literal, Path, abstractmethod, logging, np, pl, urlretrieve):
    """Shared infrastructure: abstract base class and helpers."""
    _logger = logging.getLogger("pxr_models")

    # Type alias exposed to downstream cells
    Task = Literal["regression", "classification"]

    class BaseModel(ABC):
        """
        Abstract base class that defines the stable API shared by all model types.

        Subclasses must implement train, predict, save, and load.
        """
        task: str | None = None

        @abstractmethod
        def train(self, train_df: pl.DataFrame, target_col: str, task: str, **kwargs) -> None:
            """
            Fit the model on training data.

            Args:
                train_df: Polars DataFrame containing features and the target column.
                target_col: Name of the column to predict.
                task: "regression" or "classification".
                **kwargs: Model-specific hyperparameters.
            """

        @abstractmethod
        def predict(self, df: pl.DataFrame) -> np.ndarray:
            """
            Generate predictions for the compounds in df.

            Returns:
                1-D numpy array. For classification, probability of the positive class.
            """

        @abstractmethod
        def save(self, path: str | Path) -> None:
            """Persist the trained model to disk."""

        @classmethod
        @abstractmethod
        def load(cls, path: str | Path) -> "BaseModel":
            """Load a previously saved model and return a ready-to-use instance."""

    def _extract_fp_matrix(df: pl.DataFrame, fp_col: str) -> np.ndarray:
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

    def _build_chemprop_dataset(df: pl.DataFrame, target_col: str | None):
        """
        Convert a Polars DataFrame into a chemprop MoleculeDataset.

        Args:
            df: DataFrame with a "smiles" column and optionally a target column.
            target_col: Column name for the prediction target, or None for inference.

        Returns:
            Tuple of (MoleculeDataset, featurizer).
        """
        from chemprop import data, featurizers

        featurizer = featurizers.SimpleMoleculeMolGraphFeaturizer()
        smiles = df["smiles"].to_list()

        if target_col is not None:
            targets = df[target_col].to_numpy().reshape(-1, 1).astype(float)
            datapoints = [
                data.MoleculeDatapoint.from_smi(smi, y)
                for smi, y in zip(smiles, targets)
            ]
        else:
            datapoints = [data.MoleculeDatapoint.from_smi(smi) for smi in smiles]

        return data.MoleculeDataset(datapoints, featurizer), featurizer

    def _build_chemprop_model(task: str, mp, agg, output_transform=None, input_dim: int | None = None):
        """
        Assemble a chemprop MPNN with the appropriate FFN head for the task.

        Args:
            task: "regression" or "classification".
            mp: Message-passing module.
            agg: Aggregation module.
            output_transform: Optional unscale transform (regression only).
            input_dim: FFN input size; defaults to mp.output_dim.
        """
        from chemprop import models, nn

        dim = input_dim if input_dim is not None else mp.output_dim
        if task == "classification":
            ffn = nn.BinaryClassificationFFN(input_dim=dim)
            metrics = [nn.metrics.BinaryAUROC(), nn.metrics.BinaryAUPRC()]
        else:
            ffn = nn.RegressionFFN(input_dim=dim, output_transform=output_transform)
            metrics = [nn.metrics.RMSE(), nn.metrics.MAE()]
        return models.MPNN(mp, agg, ffn, batch_norm=False, metrics=metrics)

    _CHEMELEON_URL      = "https://zenodo.org/records/15460715/files/chemeleon_mp.pt"
    _CHEMELEON_CACHE    = Path.home() / ".chemprop"
    _CHEMELEON_MP_PATH  = _CHEMELEON_CACHE / "chemeleon_mp.pt"

    def _load_chemeleon_mp():
        """
        Download (once) and load the CheMeleon pretrained message-passing weights.

        Weights are cached at ~/.chemprop/chemeleon_mp.pt and reused on subsequent
        calls without re-downloading.

        Returns:
            BondMessagePassing module with CheMeleon weights loaded.
        """
        import torch
        from chemprop import nn

        _CHEMELEON_CACHE.mkdir(exist_ok=True)
        if not _CHEMELEON_MP_PATH.exists():
            _logger.info("Downloading CheMeleon weights from Zenodo …")
            urlretrieve(_CHEMELEON_URL, _CHEMELEON_MP_PATH)

        ckpt = torch.load(_CHEMELEON_MP_PATH, weights_only=True)
        mp = nn.BondMessagePassing(**ckpt["hyper_parameters"])
        mp.load_state_dict(ckpt["state_dict"])
        return mp

    return (
        BaseModel,
        Task,
        _build_chemprop_dataset,
        _build_chemprop_model,
        _extract_fp_matrix,
        _load_chemeleon_mp,
        _logger,
    )


@app.cell
def _(BaseModel, Path, _extract_fp_matrix, _logger, joblib, np, pl):
    class RandomForestModel(BaseModel):
        """
        Random forest model trained on pre-computed molecular fingerprints.

        The fingerprint column is specified via fp_col at train time (default: "ecfp").
        The same column must be present in DataFrames passed to predict().

        Examples
        --------
        >>> rf = RandomForestModel()
        >>> rf.train(train_df, target_col="pEC50_dr", task="regression", fp_col="ecfp")
        >>> preds = rf.predict(test_df)
        >>> rf.save("rf_model.joblib")
        >>> rf2 = RandomForestModel.load("rf_model.joblib")
        """

        def __init__(self) -> None:
            self._model = None
            self._fp_col: str = "ecfp"

        def train(
            self,
            train_df: pl.DataFrame,
            target_col: str,
            task: str,
            fp_col: str = "ecfp",
            n_estimators: int = 500,
            n_jobs: int = -1,
            random_state: int = 42,
            **kwargs,
        ) -> None:
            """
            Args:
                fp_col: Fingerprint column in train_df.
                n_estimators: Number of trees.
                n_jobs: Parallel jobs (-1 = all CPUs).
                random_state: Random seed.
                **kwargs: Forwarded to RandomForestClassifier / RandomForestRegressor.
            """
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

            self.task = task
            self._fp_col = fp_col
            X = _extract_fp_matrix(train_df, fp_col)
            y = train_df[target_col].to_numpy()

            Cls = RandomForestClassifier if task == "classification" else RandomForestRegressor
            self._model = Cls(
                n_estimators=n_estimators, n_jobs=n_jobs, random_state=random_state, **kwargs
            )
            self._model.fit(X, y)
            _logger.info("RandomForestModel trained on %d compounds.", len(y))

        def predict(self, df: pl.DataFrame) -> np.ndarray:
            if self._model is None:
                raise RuntimeError("Call train() before predict().")
            X = _extract_fp_matrix(df, self._fp_col)
            if self.task == "classification":
                return self._model.predict_proba(X)[:, 1]
            return self._model.predict(X)

        def save(self, path: str | Path) -> None:
            joblib.dump({"model": self._model, "fp_col": self._fp_col, "task": self.task}, path)
            _logger.info("RandomForestModel saved to %s.", path)

        @classmethod
        def load(cls, path: str | Path) -> "RandomForestModel":
            payload = joblib.load(path)
            obj = cls()
            obj._model  = payload["model"]
            obj._fp_col = payload["fp_col"]
            obj.task    = payload["task"]
            return obj

    return (RandomForestModel,)


@app.cell
def _(BaseModel, Path, _extract_fp_matrix, _logger, joblib, np, pl):
    class GradientBoostingModel(BaseModel):
        """
        LightGBM gradient boosted tree model trained on pre-computed fingerprints.

        Uses LGBMClassifier for classification and LGBMRegressor for regression.

        Examples
        --------
        >>> gbm = GradientBoostingModel()
        >>> gbm.train(train_df, target_col="pEC50_dr", task="regression", fp_col="ecfp")
        >>> preds = gbm.predict(test_df)
        >>> gbm.save("gbm_model.joblib")
        >>> gbm2 = GradientBoostingModel.load("gbm_model.joblib")
        """

        def __init__(self) -> None:
            self._model = None
            self._fp_col: str = "ecfp"

        def train(
            self,
            train_df: pl.DataFrame,
            target_col: str,
            task: str,
            fp_col: str = "ecfp",
            n_estimators: int = 500,
            learning_rate: float = 0.05,
            num_leaves: int = 31,
            n_jobs: int = -1,
            random_state: int = 42,
            verbose: int = -1,
            **kwargs,
        ) -> None:
            """
            Args:
                fp_col: Fingerprint column in train_df.
                n_estimators: Number of boosting rounds.
                learning_rate: Step size shrinkage.
                num_leaves: Maximum leaves per tree.
                n_jobs: Parallel threads (-1 = all CPUs).
                random_state: Random seed.
                verbose: LightGBM verbosity (-1 = silent).
                **kwargs: Forwarded to LGBMClassifier / LGBMRegressor.
            """
            from lightgbm import LGBMClassifier, LGBMRegressor

            self.task = task
            self._fp_col = fp_col
            X = _extract_fp_matrix(train_df, fp_col)
            y = train_df[target_col].to_numpy()

            shared = dict(
                n_estimators=n_estimators, learning_rate=learning_rate,
                num_leaves=num_leaves, n_jobs=n_jobs,
                random_state=random_state, verbose=verbose, **kwargs,
            )
            Cls = LGBMClassifier if task == "classification" else LGBMRegressor
            self._model = Cls(**shared)
            self._model.fit(X, y)
            _logger.info("GradientBoostingModel trained on %d compounds.", len(y))

        def predict(self, df: pl.DataFrame) -> np.ndarray:
            if self._model is None:
                raise RuntimeError("Call train() before predict().")
            X = _extract_fp_matrix(df, self._fp_col)
            if self.task == "classification":
                return self._model.predict_proba(X)[:, 1]
            return self._model.predict(X)

        def save(self, path: str | Path2) -> None:
            joblib.dump({"model": self._model, "fp_col": self._fp_col, "task": self.task}, path)
            _logger.info("GradientBoostingModel saved to %s.", path)

        @classmethod
        def load(cls, path: str | Path2) -> "GradientBoostingModel":
            payload = joblib.load(path)
            obj = cls()
            obj._model  = payload["model"]
            obj._fp_col = payload["fp_col"]
            obj.task    = payload["task"]
            return obj

    return (GradientBoostingModel,)


@app.cell
def _(BaseModel, Path, _build_chemprop_dataset, _build_chemprop_model, _logger, joblib, pl):
    class ChempropModel(BaseModel):
        """
        Chemprop v2 MPNN trained from scratch using bond-based message passing.

        Operates directly on SMILES strings via graph featurization — no
        pre-computed fingerprint column is needed.

        Regression targets are normalised internally; predictions are returned
        on the original scale.

        Examples
        --------
        >>> cp = ChempropModel()
        >>> cp.train(train_df, target_col="pEC50_dr", task="regression", max_epochs=50)
        >>> preds = cp.predict(test_df)
        >>> cp.save("chemprop_model.pt")
        >>> cp2 = ChempropModel.load("chemprop_model.pt")
        """

        def __init__(self) -> None:
            self._mpnn   = None
            self._scaler = None

        def train(
            self,
            train_df: pl.DataFrame,
            target_col: str,
            task: str,
            val_df: pl.DataFrame | None = None,
            max_epochs: int = 50,
            batch_size: int = 64,
            accelerator: str = "auto",
            **kwargs,
        ) -> None:
            """
            Args:
                val_df: Validation set. If None, 20% of train_df is held out automatically.
                max_epochs: Maximum training epochs.
                batch_size: Mini-batch size.
                accelerator: PyTorch Lightning accelerator ("auto", "cpu", "gpu").
                **kwargs: Forwarded to BondMessagePassing constructor.
            """
            import lightning.pytorch as pl_lightning
            from chemprop import data, nn

            self.task = task

            if val_df is None:
                dset, featurizer = _build_chemprop_dataset(train_df, target_col)
                mols = [d.mol for d in dset]
                train_idx, val_idx, _ = data.make_split_indices(mols, "random", (0.8, 0.2, 0.0))
                train_data, val_data, _ = data.split_data_by_indices(dset, train_idx, val_idx, [])
                train_dset = data.MoleculeDataset(train_data[0], featurizer)
                val_dset   = data.MoleculeDataset(val_data[0],   featurizer)
            else:
                train_dset, featurizer = _build_chemprop_dataset(train_df, target_col)
                val_dset, _            = _build_chemprop_dataset(val_df,   target_col)

            output_transform = None
            if task == "regression":
                self._scaler = train_dset.normalize_targets()
                val_dset.normalize_targets(self._scaler)
                output_transform = nn.UnscaleTransform.from_standard_scaler(self._scaler)

            train_loader = data.build_dataloader(train_dset, batch_size=batch_size)
            val_loader   = data.build_dataloader(val_dset,   batch_size=batch_size, shuffle=False)

            mp  = nn.BondMessagePassing(**kwargs)
            agg = nn.MeanAggregation()
            self._mpnn = _build_chemprop_model(task, mp, agg, output_transform)

            trainer = pl_lightning.Trainer(
                logger=False, enable_checkpointing=False, enable_progress_bar=True,
                accelerator=accelerator, devices=1, max_epochs=max_epochs,
            )
            trainer.fit(self._mpnn, train_loader, val_loader)
            _logger.info("ChempropModel trained for %d epochs.", max_epochs)

        def predict(self, df: pl.DataFrame):
            if self._mpnn is None:
                raise RuntimeError("Call train() before predict().")
            import torch
            import lightning.pytorch as pl_lightning
            from chemprop import data

            dset, _ = _build_chemprop_dataset(df, target_col=None)
            loader  = data.build_dataloader(dset, shuffle=False)
            trainer = pl_lightning.Trainer(
                logger=False, enable_progress_bar=False, accelerator="auto", devices=1
            )
            preds = trainer.predict(self._mpnn, loader)
            return torch.cat(preds).numpy(force=True).flatten()

        def save(self, path: str | Path3) -> None:
            from chemprop.models import save_model
            save_model(path, self._mpnn)
            joblib.dump({"task": self.task, "scaler": self._scaler},
                          Path(path).with_suffix(".meta.joblib"))
            _logger.info("ChempropModel saved to %s.", path)

        @classmethod
        def load(cls, path: str | Path3) -> "ChempropModel":
            from chemprop.models import load_model
            obj = cls()
            obj._mpnn = load_model(path)
            meta_path = Path(path).with_suffix(".meta.joblib")
            if meta_path.exists():
                meta = joblib.load(meta_path)
                obj.task    = meta["task"]
                obj._scaler = meta["scaler"]
            return obj

    return (ChempropModel,)


@app.cell
def _(BaseModel, Path, _build_chemprop_dataset, _build_chemprop_model, _load_chemeleon_mp, _logger, joblib, pl):
    class ChempropChemeleonModel(BaseModel):
        """
        Chemprop v2 MPNN fine-tuned from the CheMeleon pretrained backbone.

        CheMeleon provides a BondMessagePassing module pretrained on ~1M PubChem
        molecules. Fine-tuning only replaces the prediction head, keeping the
        pretrained weights as the starting point — this typically improves
        performance on small datasets.

        Pretrained weights are downloaded automatically on the first call to
        train() and cached at ~/.chemprop/chemeleon_mp.pt.

        Reference: https://github.com/JacksonBurns/chemeleon

        Examples
        --------
        >>> ccm = ChempropChemeleonModel()
        >>> ccm.train(train_df, target_col="pEC50_dr", task="regression", max_epochs=30)
        >>> preds = ccm.predict(test_df)
        >>> ccm.save("chemeleon_model.pt")
        >>> ccm2 = ChempropChemeleonModel.load("chemeleon_model.pt")
        """

        def __init__(self) -> None:
            self._mpnn   = None
            self._scaler = None

        def train(
            self,
            train_df: pl.DataFrame,
            target_col: str,
            task: str,
            val_df: pl.DataFrame | None = None,
            max_epochs: int = 30,
            batch_size: int = 64,
            accelerator: str = "auto",
        ) -> None:
            """
            Args:
                val_df: Validation set. If None, 20% of train_df is held out automatically.
                max_epochs: Fine-tuning epochs (fewer than training from scratch).
                batch_size: Mini-batch size.
                accelerator: PyTorch Lightning accelerator ("auto", "cpu", "gpu").
            """
            import lightning.pytorch as pl_lightning
            from chemprop import data, nn

            self.task = task

            if val_df is None:
                dset, featurizer = _build_chemprop_dataset(train_df, target_col)
                mols = [d.mol for d in dset]
                train_idx, val_idx, _ = data.make_split_indices(mols, "random", (0.8, 0.2, 0.0))
                train_data, val_data, _ = data.split_data_by_indices(dset, train_idx, val_idx, [])
                train_dset = data.MoleculeDataset(train_data[0], featurizer)
                val_dset   = data.MoleculeDataset(val_data[0],   featurizer)
            else:
                train_dset, featurizer = _build_chemprop_dataset(train_df, target_col)
                val_dset, _            = _build_chemprop_dataset(val_df,   target_col)

            output_transform = None
            if task == "regression":
                self._scaler = train_dset.normalize_targets()
                val_dset.normalize_targets(self._scaler)
                output_transform = nn.UnscaleTransform.from_standard_scaler(self._scaler)

            train_loader = data.build_dataloader(train_dset, batch_size=batch_size)
            val_loader   = data.build_dataloader(val_dset,   batch_size=batch_size, shuffle=False)

            # Load CheMeleon backbone; input_dim must match its hidden size (2048)
            mp  = _load_chemeleon_mp()
            agg = nn.MeanAggregation()
            self._mpnn = _build_chemprop_model(
                task, mp, agg, output_transform, input_dim=mp.output_dim
            )

            trainer = pl_lightning.Trainer(
                logger=False, enable_checkpointing=False, enable_progress_bar=True,
                accelerator=accelerator, devices=1, max_epochs=max_epochs,
            )
            trainer.fit(self._mpnn, train_loader, val_loader)
            _logger.info("ChempropChemeleonModel fine-tuned for %d epochs.", max_epochs)

        def predict(self, df: pl.DataFrame):
            if self._mpnn is None:
                raise RuntimeError("Call train() before predict().")
            import torch
            import lightning.pytorch as pl_lightning
            from chemprop import data

            dset, _ = _build_chemprop_dataset(df, target_col=None)
            loader  = data.build_dataloader(dset, shuffle=False)
            trainer = pl_lightning.Trainer(
                logger=False, enable_progress_bar=False, accelerator="auto", devices=1
            )
            preds = trainer.predict(self._mpnn, loader)
            return torch.cat(preds).numpy(force=True).flatten()

        def save(self, path: str | Path4) -> None:
            from chemprop.models import save_model
            save_model(path, self._mpnn)
            joblib.dump({"task": self.task, "scaler": self._scaler},
                          Path(path).with_suffix(".meta.joblib"))
            _logger.info("ChempropChemeleonModel saved to %s.", path)

        @classmethod
        def load(cls, path: str | Path4) -> "ChempropChemeleonModel":
            from chemprop.models import load_model
            obj = cls()
            obj._mpnn = load_model(path)
            meta_path = Path(path).with_suffix(".meta.joblib")
            if meta_path.exists():
                meta = joblib.load(meta_path)
                obj.task    = meta["task"]
                obj._scaler = meta["scaler"]
            return obj

    return (ChempropChemeleonModel,)


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
):
    _classification_metrics = {
        "accuracy": accuracy_score,
        "balanced_accuracy": balanced_accuracy_score,
        "precision": precision_score,
        "recall": recall_score,
        "f1": f1_score,
        "mcc": matthews_corrcoef,
    }

    _regression_metrics = {
        "r2": r2_score,
        "rho": lambda y_true, y_pred: spearmanr(y_true, y_pred).correlation,
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

    return


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
    | **Temporal** | Prospective generalisation — molecules are ordered by their numeric ID (a proxy for acquisition time) and test compounds always come *after* training compounds |

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
    generate_cv_splits_random,
    generate_cv_splits_scaffold,
    generate_cv_splits_temporal,
    generate_fingerprint,
    np,
    pl,
    whole_train,
):
    from rdkit import DataStructs
    from rdkit.DataStructs import ExplicitBitVect

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
    sim_df
    return (sim_df,)


@app.cell
def _(mo, sim_df):
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

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

    mo.as_html(_fig)
    return


if __name__ == "__main__":
    app.run()
