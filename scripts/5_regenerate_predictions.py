"""
Regenerate the individual component predictions for the best ensemble submission
    4_ens_cp5_ch5_rf0_xg13_mc1_tf5_submission.csv

and write each component to its own CSV, then recompute the ensemble and compare
it with the original submission.

Run from the project root:
    .venv/bin/python scripts/5_regenerate_predictions.py

Determinism notes
-----------------
| Component | Method          | Seed | Reproducible? |
|-----------|-----------------|------|---------------|
| cp        | Chemprop D-MPNN | none | No  — NN      |
| ch        | CheMeleon FFN   | none | No  — NN      |
| xg        | XGBoost Mordred | none | No  — random subsample |
| mc        | Macau MCMC      | 42   | Yes           |
| tf        | TabPFN          | 0    | Yes           |

Only mc and tf are expected to reproduce the original predictions closely.
cp, ch, and xg will differ due to stochastic training.
"""

import gc
import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import optuna
import polars as pl
import scipy.sparse as sp
import smurff
import xgboost as xgb
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_absolute_error
from skfp.fingerprints import MordredFingerprint
from skfp.preprocessing import MolFromSmilesTransformer

# ── Paths ──────────────────────────────────────────────────────────────────────
ROOT       = Path(__file__).resolve().parent.parent
PRED_DIR   = ROOT / "predictions"
SUB_DIR    = ROOT / "submissions"
DATA_DIR   = ROOT / "data"
LOG_DIR    = ROOT / "logs"
TMP        = Path(tempfile.gettempdir())

PRED_DIR.mkdir(exist_ok=True)
LOG_DIR.mkdir(exist_ok=True)

TARGET_COL = "pEC50_dr"
SEED       = 42

# ── HPO best params (loaded from Optuna DBs) ──────────────────────────────────
optuna.logging.set_verbosity(optuna.logging.WARNING)

_cp_study   = optuna.load_study(
    study_name="chemprop_hpo_1x5cv",
    storage=f"sqlite:///{ROOT}/predictions/4_hpo_chemprop.db",
)
CP_PARAMS: dict = {**_cp_study.best_params, "epochs": 50}
# CheMeleon uses the same HPO params but without message_hidden_dim / depth
# (those are fixed by the CheMeleon backbone at 2048-d, depth-6).
CH_PARAMS: dict = {
    k: v for k, v in CP_PARAMS.items()
    if k not in ("message_hidden_dim", "depth")
}

_mac_study  = optuna.load_study(
    study_name="macau_chemeleon_hpo",
    storage=f"sqlite:///{ROOT}/predictions/4_hpo_macau_chemeleon.db",
)
MAC_PARAMS: dict = _mac_study.best_params

_xgb_study  = optuna.load_study(
    study_name="xgb_mordred_hpo",
    storage=f"sqlite:///{ROOT}/predictions/4_hpo_xgb_mordred.db",
)
XGB_PARAMS: dict = _xgb_study.best_params

print("Loaded HPO params:")
print(f"  cp : {CP_PARAMS}")
print(f"  ch : {CH_PARAMS}")
print(f"  mc : {MAC_PARAMS}")
print(f"  xg : {XGB_PARAMS}")

# ── Ensemble weights for cp5_ch5_rf0_xg13_mc1_tf5 ────────────────────────────
# Weight tokens: "0"=0, "15"=1/5, "14"=1/4, "13"=1/3, "12"=1/2,
#                "1"=1, "2"=2, "3"=3, "4"=4, "5"=5
_W_MAP  = {"0": 0.0, "15": 1/5, "14": 1/4, "13": 1/3, "12": 1/2,
           "1": 1.0, "2": 2.0, "3": 3.0, "4": 4.0, "5": 5.0}
_TAGS   = ["cp", "ch", "rf", "xg", "mc", "tf"]
_LABEL  = "cp5_ch5_rf0_xg13_mc1_tf5"
_PARTS  = _LABEL.split("_")
ENS_WEIGHTS: dict[str, float] = {
    tag: _W_MAP[part[len(tag):]]
    for tag, part in zip(_TAGS, _PARTS)
}
print(f"\nEnsemble weights: {ENS_WEIGHTS}")

# ── Data loading ───────────────────────────────────────────────────────────────
train_full: pl.DataFrame = (
    pl.read_csv(DATA_DIR / "processed/all_compounds_activity_data.csv")
    .filter(pl.col(TARGET_COL).is_not_null())
    .select(["smiles", "inchikey", "molecule_names", TARGET_COL])
)
test_df: pl.DataFrame    = pl.read_csv(DATA_DIR / "raw/20260409/dose_response_test.csv")
test_smiles: list[str]   = test_df["SMILES"].to_list()
test_names:  list[str]   = test_df["Molecule Name"].to_list()

# 10 % validation split (same seed as notebook 4)
rng      = np.random.default_rng(SEED)
n        = len(train_full)
val_idx  = rng.choice(n, size=int(n * 0.1), replace=False)
tr_idx   = np.setdiff1d(np.arange(n), val_idx)
train_sub: pl.DataFrame = train_full[tr_idx.tolist()]
val_sub:   pl.DataFrame = train_full[val_idx.tolist()]

print(f"\nData: {n} train total | {len(tr_idx)} train_sub | {len(val_idx)} val_sub | "
      f"{len(test_smiles)} test")

# ── Helpers ────────────────────────────────────────────────────────────────────

def save_component(tag: str, preds: np.ndarray) -> Path:
    """Save component predictions as a CSV and return the path."""
    path = PRED_DIR / f"5_regen_{tag}_preds.csv"
    pl.DataFrame({
        "SMILES":          test_smiles,
        "Molecule Name":   test_names,
        f"pEC50_{tag}":    preds.tolist(),
    }).write_csv(path)
    print(f"  → saved {path.name}")
    return path


def save_submission(label: str, preds: np.ndarray) -> Path:
    path = SUB_DIR / f"5_regen_{label}_submission.csv"
    pl.DataFrame({
        "SMILES":        test_smiles,
        "Molecule Name": test_names,
        "pEC50":         preds.tolist(),
    }).write_csv(path)
    print(f"  → saved {path.name}")
    return path


# ── CheMeleon embedding helper ─────────────────────────────────────────────────

CHEMELEON_SCRIPT_PATH = TMP / "chemeleon_embed.py"
CHEMELEON_SCRIPT_PATH.write_text("\n".join([
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
]))


def chemeleon_embed(
    smiles_train: list[str],
    smiles_test: list[str],
    prefix: str = "regen",
) -> tuple[np.ndarray, np.ndarray]:
    """Run CheMeleon embedding in an isolated subprocess, return (X_train, X_test)."""
    smi_file   = TMP / f"{prefix}_smiles.json"
    train_file = TMP / f"{prefix}_train"
    test_file  = TMP / f"{prefix}_test"
    smi_file.write_text(json.dumps({"train": smiles_train, "test": smiles_test}))
    result = subprocess.run(
        [sys.executable, str(CHEMELEON_SCRIPT_PATH),
         str(smi_file), str(train_file), str(test_file)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if result.returncode != 0:
        raise RuntimeError(f"CheMeleon subprocess failed:\n{result.stderr}")
    X_train = np.load(str(train_file) + ".npy")
    X_test  = np.load(str(test_file)  + ".npy")
    for p in [smi_file, Path(str(train_file) + ".npy"), Path(str(test_file) + ".npy")]:
        p.unlink(missing_ok=True)
    return X_train, X_test


# ── Chemprop CLI helpers ───────────────────────────────────────────────────────

CHEMPROP_BIN = Path(sys.executable).parent / "chemprop"
CHEMPROP_LOG = LOG_DIR / "chemprop_regen.log"

def _write_smiles_csv(smiles: list[str], targets, path: Path, col: str) -> None:
    if targets is not None:
        pl.DataFrame({"smiles": smiles, col: targets.flatten().tolist()}).write_csv(path)
    else:
        pl.DataFrame({"smiles": smiles}).write_csv(path)


def _run_chemprop_cli(args: list[str]) -> None:
    cmd = [str(CHEMPROP_BIN)] + args
    env = {**os.environ, "PYTORCH_MPS_HIGH_WATERMARK_RATIO": "0.0"}
    with open(CHEMPROP_LOG, "a") as log:
        log.write(f"\n{'='*60}\nCMD: {' '.join(cmd)}\n{'='*60}\n")
        result = subprocess.run(cmd, stdout=log, stderr=log, text=True, env=env)
    if result.returncode != 0:
        lines = CHEMPROP_LOG.read_text().splitlines()
        print("\n".join(lines[-30:]))
        raise RuntimeError(f"chemprop CLI failed (exit {result.returncode})")


def _get_device() -> str:
    import torch
    return (
        "cuda" if torch.cuda.is_available()
        else "mps" if torch.backends.mps.is_available()
        else "cpu"
    )


def train_predict_chemprop(
    model_dir: Path,
    params: dict,
    from_foundation: str | None,
    smiles_tr: list[str], y_tr: np.ndarray,
    smiles_va: list[str], y_va: np.ndarray,
    smiles_te: list[str],
) -> np.ndarray:
    """Train a Chemprop (or CheMeleon) model and return test predictions."""
    if model_dir.exists():
        shutil.rmtree(model_dir)

    train_csv = TMP / "regen_train.csv"
    val_csv   = TMP / "regen_val.csv"
    test_csv  = TMP / "regen_test.csv"
    pred_csv  = TMP / "regen_preds.csv"

    _write_smiles_csv(smiles_tr, y_tr, train_csv, TARGET_COL)
    _write_smiles_csv(smiles_va, y_va, val_csv,   TARGET_COL)
    _write_smiles_csv(smiles_te, None, test_csv,  TARGET_COL)

    train_args = [
        "train",
        "--data-path", str(train_csv), str(val_csv), str(val_csv),
        "--smiles-columns", "smiles",
        "--target-columns", TARGET_COL,
        "--task-type", "regression",
        "--accelerator", _get_device(),
        "--dropout",        str(params.get("dropout", 0.0)),
        "--ffn-hidden-dim", str(params.get("ffn_hidden_dim", 300)),
        "--ffn-num-layers", str(params.get("ffn_num_layers", 2)),
        "--batch-size",     str(params.get("batch_size", 64)),
        "--max-lr",         str(params.get("max_lr", 1e-3)),
        "--epochs",         str(params.get("epochs", 50)),
        "--save-dir",       str(model_dir),
    ]
    if from_foundation:
        train_args += ["--from-foundation", from_foundation]
    if "message_hidden_dim" in params:
        train_args += ["--message-hidden-dim", str(params["message_hidden_dim"])]
    if "depth" in params:
        train_args += ["--depth", str(params["depth"])]

    _run_chemprop_cli(train_args)

    model_pt = model_dir / "model_0" / "best.pt"
    _run_chemprop_cli([
        "predict",
        "--test-path",  str(test_csv),
        "--model-path", str(model_pt),
        "--preds-path", str(pred_csv),
    ])

    preds = pl.read_csv(pred_csv)[TARGET_COL].to_numpy().flatten()
    for p in [train_csv, val_csv, test_csv, pred_csv]:
        p.unlink(missing_ok=True)
    return preds


# ── Macau model ────────────────────────────────────────────────────────────────

def train_predict_macau(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
    num_latent: int, nsamples: int, burnin: int, seed: int = 42,
) -> np.ndarray:
    """Fit a Macau MCMC model and return posterior-mean predictions."""
    n     = len(y_tr)
    vals  = y_tr.flatten().astype(np.float64)
    mask  = ~np.isnan(vals)
    Y_tr  = sp.coo_matrix(
        (vals[mask], (np.where(mask)[0], np.zeros(mask.sum(), dtype=int))),
        shape=(n, 1),
    )
    smurff_logger = logging.getLogger("smurff")
    root_logger   = logging.getLogger()
    prev_s = smurff_logger.level; prev_r = root_logger.level
    smurff_logger.setLevel(logging.ERROR)
    root_logger.setLevel(logging.ERROR)
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            session = smurff.MacauSession(
                Ytrain=Y_tr,
                side_info=[X_tr.astype(np.float64), None],
                num_latent=num_latent,
                burnin=burnin,
                nsamples=nsamples,
                direct=True,
                seed=seed,
                verbose=0,
                save_name=os.path.join(tmpdir, "smurff.hdf5"),
                save_freq=1,
            )
            session.init()
            while session.step():
                pass
            pred_session = session.makePredictSession()
    finally:
        smurff_logger.setLevel(prev_s)
        root_logger.setLevel(prev_r)

    samples = pred_session.predict((X_te.astype(np.float64), slice(None)))
    return np.mean(np.array(samples), axis=0).flatten()


# ── TabPFN (subprocess, same as notebook 4) ───────────────────────────────────

TABPFN_SCRIPT_PATH = TMP / "regen_tabpfn.py"
TABPFN_SCRIPT_PATH.write_text("\n".join([
    "import os, sys, numpy as np",
    "os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'",
    "import torch",
    "torch.set_num_threads(max(1, (os.cpu_count() or 1) - 1))",
    "from tabpfn import TabPFNRegressor",
    "X_train = np.load(sys.argv[1])",
    "y_train = np.load(sys.argv[2])",
    "X_test  = np.load(sys.argv[3])",
    "out     = sys.argv[4]",
    # random_state=0 matches the class default and makes this deterministic
    "model = TabPFNRegressor(n_estimators=8, ignore_pretraining_limits=True, device='cpu')",
    "model.fit(X_train, y_train)",
    "np.save(out, model.predict(X_test))",
]))


def train_predict_tabpfn(
    X_tr: np.ndarray, y_tr: np.ndarray,
    X_te: np.ndarray,
) -> np.ndarray:
    """Run TabPFN in an isolated subprocess and return predictions."""
    f_Xtr = TMP / "regen_tf_Xtr.npy";  f_ytr = TMP / "regen_tf_ytr.npy"
    f_Xte = TMP / "regen_tf_Xte.npy";  f_out = TMP / "regen_tf_preds"
    np.save(str(f_Xtr), X_tr); np.save(str(f_ytr), y_tr)
    np.save(str(f_Xte), X_te)
    res = subprocess.run(
        [sys.executable, str(TABPFN_SCRIPT_PATH),
         str(f_Xtr), str(f_ytr), str(f_Xte), str(f_out)],
        capture_output=True, text=True, cwd=str(ROOT),
    )
    if res.returncode != 0:
        raise RuntimeError(f"TabPFN failed:\n{res.stderr}")
    preds = np.load(str(f_out) + ".npy")
    for p in [f_Xtr, f_ytr, f_Xte, Path(str(f_out) + ".npy")]:
        p.unlink(missing_ok=True)
    return preds.flatten()


# ── XGBoost (Mordred fingerprints) ────────────────────────────────────────────

def train_predict_xgboost(
    smiles_tr: list[str], y_tr: np.ndarray,
    smiles_va: list[str], y_va: np.ndarray,
    smiles_te: list[str],
    xgb_params: dict,
) -> np.ndarray:
    """Compute Mordred FPs, fit XGBoost with early stopping, return predictions."""
    def mordred_matrix(smiles_list: list[str]) -> np.ndarray:
        fp = MordredFingerprint()
        return np.stack(fp.transform(smiles_list)).astype(np.float32)

    print("  computing Mordred fingerprints …")
    X_tr = mordred_matrix(smiles_tr)
    X_va = mordred_matrix(smiles_va)
    X_te = mordred_matrix(smiles_te)

    # Drop any feature column with NaN in the training set
    valid = ~np.isnan(X_tr).any(axis=0)
    X_tr, X_va, X_te = X_tr[:, valid], X_va[:, valid], X_te[:, valid]

    model = xgb.XGBRegressor(
        **xgb_params,
        early_stopping_rounds=10,
        tree_method="hist",
        n_jobs=-1,
    )
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    return model.predict(X_te).flatten()


# ── Main: regenerate all components ───────────────────────────────────────────

def compare_preds(
    label: str,
    new_preds: np.ndarray,
    orig_path: Path | None = None,
) -> None:
    """Print correlation stats between new predictions and an original submission."""
    if orig_path is None or not orig_path.exists():
        print(f"  [{label}] no original file to compare against")
        return
    orig = pl.read_csv(orig_path).sort("Molecule Name")
    new  = pl.DataFrame({
        "Molecule Name": test_names,
        "pEC50_new":     new_preds.tolist(),
    }).sort("Molecule Name")
    merged = orig.join(new, on="Molecule Name", how="inner")
    y_orig = merged["pEC50"].to_numpy()
    y_new  = merged["pEC50_new"].to_numpy()
    r, _   = pearsonr(y_orig, y_new)
    rho, _ = spearmanr(y_orig, y_new)
    mae    = mean_absolute_error(y_orig, y_new)
    print(f"  [{label}] vs original → Pearson r={r:.4f}  Spearman ρ={rho:.4f}  MAE={mae:.4f}")


def main() -> None:
    component_preds: dict[str, np.ndarray] = {}

    # ── cp: Chemprop HPO ──────────────────────────────────────────────────────
    print("\n[1/5] Chemprop HPO (cp) — non-deterministic NN")
    cp_dir = TMP / "regen_cp_model"
    component_preds["cp"] = train_predict_chemprop(
        model_dir=cp_dir,
        params=CP_PARAMS,
        from_foundation=None,
        smiles_tr=train_sub["smiles"].to_list(),
        y_tr=train_sub[TARGET_COL].to_numpy(),
        smiles_va=val_sub["smiles"].to_list(),
        y_va=val_sub[TARGET_COL].to_numpy(),
        smiles_te=test_smiles,
    )
    save_component("cp", component_preds["cp"])
    compare_preds("cp", component_preds["cp"], orig_path=None)

    # ── ch: CheMeleon HPO ─────────────────────────────────────────────────────
    print("\n[2/5] CheMeleon HPO (ch) — non-deterministic NN")
    ch_dir = TMP / "regen_ch_model"
    component_preds["ch"] = train_predict_chemprop(
        model_dir=ch_dir,
        params=CH_PARAMS,
        from_foundation="CHEMELEON",
        smiles_tr=train_sub["smiles"].to_list(),
        y_tr=train_sub[TARGET_COL].to_numpy(),
        smiles_va=val_sub["smiles"].to_list(),
        y_va=val_sub[TARGET_COL].to_numpy(),
        smiles_te=test_smiles,
    )
    save_component("ch", component_preds["ch"])
    compare_preds("ch", component_preds["ch"],
                  orig_path=SUB_DIR / "4_chemeleon_hpo_submission.csv")

    # ── xg: XGBoost HPO on Mordred ───────────────────────────────────────────
    print("\n[3/5] XGBoost HPO / Mordred (xg) — random subsampling, may differ")
    component_preds["xg"] = train_predict_xgboost(
        smiles_tr=train_sub["smiles"].to_list(),
        y_tr=train_sub[TARGET_COL].to_numpy(),
        smiles_va=val_sub["smiles"].to_list(),
        y_va=val_sub[TARGET_COL].to_numpy(),
        smiles_te=test_smiles,
        xgb_params=XGB_PARAMS,
    )
    save_component("xg", component_preds["xg"])
    compare_preds("xg", component_preds["xg"], orig_path=None)

    # ── mc: Macau HPO on CheMeleon FP ────────────────────────────────────────
    print("\n[4/5] Macau HPO / CheMeleon (mc) — deterministic (seed=42)")
    print("  computing CheMeleon embeddings …")
    X_tr_mc, X_te_mc = chemeleon_embed(
        train_full["smiles"].to_list(), test_smiles, prefix="regen_mc"
    )
    component_preds["mc"] = train_predict_macau(
        X_tr=X_tr_mc, y_tr=train_full[TARGET_COL].to_numpy(),
        X_te=X_te_mc,
        **MAC_PARAMS, seed=SEED,
    )
    save_component("mc", component_preds["mc"])
    compare_preds("mc", component_preds["mc"],
                  orig_path=SUB_DIR / "4_macau_che_hpo_submission.csv")
    del X_tr_mc, X_te_mc
    gc.collect()

    # ── tf: TabPFN on CheMeleon FP ────────────────────────────────────────────
    print("\n[5/5] TabPFN / CheMeleon (tf) — deterministic (no random_state needed)")
    print("  computing CheMeleon embeddings …")
    X_tr_tf, X_te_tf = chemeleon_embed(
        train_full["smiles"].to_list(), test_smiles, prefix="regen_tf"
    )
    component_preds["tf"] = train_predict_tabpfn(
        X_tr=X_tr_tf, y_tr=train_full[TARGET_COL].to_numpy(),
        X_te=X_te_tf,
    )
    save_component("tf", component_preds["tf"])
    compare_preds("tf", component_preds["tf"],
                  orig_path=SUB_DIR / "4_tabpfn_chemeleon_submission.csv")
    del X_tr_tf, X_te_tf
    gc.collect()

    # ── Ensemble: weighted average ─────────────────────────────────────────────
    print("\n[ensemble] Computing weighted average …")
    total_w = sum(w for w in ENS_WEIGHTS.values() if w > 0)
    ens_preds = sum(
        component_preds[tag] * (w / total_w)
        for tag, w in ENS_WEIGHTS.items()
        if w > 0
    )
    save_submission(_LABEL, ens_preds)
    compare_preds(
        "ensemble", ens_preds,
        orig_path=SUB_DIR / f"4_ens_{_LABEL}_submission.csv",
    )

    print("\nDone.  Component CSVs: predictions/5_regen_*_preds.csv")
    print(f"Ensemble CSV:        submissions/5_regen_{_LABEL}_submission.csv")


if __name__ == "__main__":
    main()
