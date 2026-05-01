import marimo

__generated_with = "0.23.2"
app = marimo.App()


@app.cell
def _():
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
    # CheMeleon hyperparameter optimisation (1×5 CV)

    We use a base Chemprop model as a fast proxy to find optimal FFN head
    and training schedule hyperparameters, which are then transferred to
    CheMeleon fine-tuning. The proxy is configured with `message_hidden_dim=2048`
    and `depth=6` to match CheMeleon's frozen backbone dimensions, so the FFN
    head sees the same input size. `message_hidden_dim` and `depth` are excluded
    from the search space since they are fixed by the CheMeleon architecture.
    A 1×5 CV is used for speed — one outer split produces five folds, each
    with a 10 % internal val set for early stopping.

    **Search space:**

    | Parameter | Range | Chemprop default | CheMeleon default |
    |---|---|---|---|
    | `dropout` | 0.0, 0.1, 0.2, 0.3, 0.4 | 0.0 | 0.0 |
    | `ffn_hidden_dim` | 300, 600, 900, 1200 | 300 | 900 |
    | `ffn_num_layers` | 1, 2, 3 | 2 | 2 |
    | `batch_size` | 32, 64, 128 | 64 | 64 |
    | `max_lr` | 1e-4 – 1e-2 (log) | 1e-3 | 1e-3 |
    | `epochs` | 20 – 70 (int) | 50 | 50 |

    **Objective:** minimise mean MAE across the 5 folds.
    """)
    return


@app.cell
def _(
    ChempropModel,
    Path,
    gc,
    generate_cv_splits_random,
    gzip,
    np,
    optuna,
    pl,
    single_task_train,
    tqdm,
):
    _TARGET_COL   = "pEC50_dr"
    _PRED_PATH_GZ = Path("../predictions/3_chemeleon_hpo_results.csv.gz")
    _N_TRIALS     = 15
    _N_FOLDS      = 5
    _SEED         = 42
    _P_VAL        = 0.1

    # ── CV splits (1 outer × 5 inner = 5 folds) ───────────────────────────────
    _cv_splits = list(generate_cv_splits_random(
        single_task_train, n_outer=1, n_inner=_N_FOLDS, seed=_SEED, p_val=_P_VAL,
    ))

    if _PRED_PATH_GZ.exists():
        print(f"HPO results already exist at {_PRED_PATH_GZ} — skipping.")
        _results_df = pl.read_csv(_PRED_PATH_GZ)
    else:
        _all_records: list[dict] = []

        def _objective(trial: optuna.Trial) -> float:
            dropout       = trial.suggest_categorical("dropout", [0.0, 0.1, 0.2, 0.3, 0.4])
            ffn_hidden    = trial.suggest_categorical("ffn_hidden_dim", [300, 600, 900, 1200])
            ffn_layers    = trial.suggest_int("ffn_num_layers", 1, 3)
            batch_size    = trial.suggest_categorical("batch_size", [32, 64, 128])
            max_lr        = trial.suggest_float("max_lr", 1e-4, 1e-2, log=True)
            epochs        = trial.suggest_int("epochs", 20, 70)

            fold_maes: list[float] = []

            for _fold, _outer, _inner, _train_raw, _val_raw, _test_raw in _cv_splits:
                # Use ChempropModel as a fast proxy for CheMeleon HPO.
                # message_hidden_dim=2048 and depth=6 match the CheMeleon backbone
                # so the FFN head sees the same input dimension as in fine-tuning.
                # message_hidden_dim and depth are NOT tuned — they are fixed by
                # CheMeleon and excluded from the search space intentionally.
                _model = ChempropModel(
                    pred_type="regression",
                    message_hidden_dim=2048,
                    depth=6,
                    dropout=dropout,
                    ffn_hidden_dim=ffn_hidden,
                    ffn_num_layers=ffn_layers,
                    batch_size=batch_size,
                    max_lr=max_lr,
                    epochs=epochs,
                )
                _model.train(
                    _train_raw["smiles"].to_list(),
                    _train_raw[_TARGET_COL].to_numpy(),
                    _val_raw["smiles"].to_list(),
                    _val_raw[_TARGET_COL].to_numpy(),
                    target_col=_TARGET_COL,
                )
                _y_pred = _model.predict(_test_raw["smiles"].to_list())
                _y_true = _test_raw[_TARGET_COL].to_numpy()
                fold_mae = float(np.mean(np.abs(_y_pred - _y_true)))
                fold_maes.append(fold_mae)

                # Store per-compound predictions for later analysis
                for _ik, _mn, _smi, _yt, _yp in zip(
                    _test_raw["inchikey"].to_list(),
                    _test_raw["molecule_names"].to_list(),
                    _test_raw["smiles"].to_list(),
                    _y_true.tolist(),
                    _y_pred.tolist(),
                ):
                    _all_records.append({
                        "trial":          trial.number,
                        "inchikey":       _ik,
                        "molecule_names": _mn,
                        "smiles":         _smi,
                        "fold":           _fold,
                        "inner_fold":     _inner,
                        "y_true":         _yt,
                        "y_pred":         _yp,
                        "dropout":        dropout,
                        "ffn_hidden_dim": ffn_hidden,
                        "ffn_num_layers": ffn_layers,
                        "batch_size":     batch_size,
                        "max_lr":         max_lr,
                        "epochs":         epochs,
                    })

                del _model
                gc.collect()

            return float(np.mean(fold_maes))

        _study = optuna.create_study(
            direction="minimize",
            sampler=optuna.samplers.TPESampler(seed=_SEED),
            study_name="chemeleon_hpo",
        )

        _pbar = tqdm(total=_N_TRIALS, desc="Optuna trials")

        def _callback(study, trial):
            _pbar.set_postfix({
                "trial": trial.number,
                "mae": f"{trial.value:.4f}",
                "best": f"{study.best_value:.4f}",
            })
            _pbar.update(1)

        _study.optimize(_objective, n_trials=_N_TRIALS, callbacks=[_callback])
        _pbar.close()

        print(f"\nBest trial: {_study.best_trial.number}")
        print(f"Best MAE:   {_study.best_value:.4f}")
        print(f"Best params: {_study.best_params}")

        _results_df = pl.DataFrame(_all_records)
        _PRED_PATH_GZ.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(_PRED_PATH_GZ, "wb") as _f:
            _results_df.write_csv(_f)
        print(f"Saved {len(_results_df):,} rows → {_PRED_PATH_GZ}")

    # ── summarise best trial ────────────────────────────────────────────────────
    _trial_mae = (
        _results_df
        .group_by(["trial", "dropout", "ffn_hidden_dim", "ffn_num_layers",
                   "batch_size", "max_lr", "epochs"])
        .agg(((pl.col("y_pred") - pl.col("y_true")).abs()).mean().alias("mae"))
        .sort("mae")
    )
    _trial_mae
    return


if __name__ == "__main__":
    app.run()
