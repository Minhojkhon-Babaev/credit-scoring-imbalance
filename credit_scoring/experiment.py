from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import optuna
import pandas as pd
from sklearn.metrics import average_precision_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from credit_scoring.data import load_dataset
from credit_scoring.generators import fit_ctgan_pool
from credit_scoring.metrics import best_f1_threshold, classification_metrics
from credit_scoring.models import build_model, suggest_params
from credit_scoring.preprocess import FoldPreprocessor
from credit_scoring.resample import (
    LEVELS,
    RESAMPLE_METHODS,
    _distance_resample,
    ctgan_resample,
    gmm_resample,
    noise_resample,
    smote_resample,
)
from credit_scoring.synth_quality import quality_row

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
ALL_METHODS = ("raw", "class_weight", *RESAMPLE_METHODS)
ALL_MODELS = ("logreg", "xgboost", "catboost")


def log(message: str) -> None:
    print(message, flush=True)


def apply_resample(
    method: str,
    X: np.ndarray,
    y: np.ndarray,
    level: str | None,
    prep: FoldPreprocessor,
    seed: int,
    noise_scale: float,
    gmm_components: int,
    pool: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray]:
    if method in {"raw", "class_weight"} or level is None:
        return X, y
    rng = np.random.default_rng(seed)
    if method == "smote":
        return smote_resample(X, y, level, prep.cat_cardinalities, prep.n_num, seed)
    if method in {"borderline_smote", "adasyn"}:
        return _distance_resample(method, X, y, level, prep.cat_cardinalities, prep.n_num, seed)
    if method == "noise":
        return noise_resample(X, y, level, prep.n_num, noise_scale, rng, prep.cat_cardinalities)
    if method == "gmm":
        return gmm_resample(X, y, level, prep.n_num, gmm_components, rng, prep.cat_cardinalities)
    if method == "ctgan":
        if pool is None:
            raise RuntimeError("Для CTGAN не подготовлен пул синтетических дефолтов")
        return ctgan_resample(X, y, level, pool, rng, prep.cat_cardinalities, prep.n_num)
    raise ValueError(method)


def _generator_kwargs(params: dict) -> tuple[float, int]:
    return float(params.get("noise_scale", 0.1)), int(params.get("gmm_components", 3))


def _fold_score(
    model_name: str,
    method: str,
    params: dict,
    level: str | None,
    fold: dict,
    pool: np.ndarray | None,
    seed: int,
) -> float:
    noise_scale, gmm_components = _generator_kwargs(params)
    Xb, yb = apply_resample(
        method,
        fold["X"],
        fold["y"],
        level,
        fold["prep"],
        seed,
        noise_scale,
        gmm_components,
        pool,
    )
    model = build_model(model_name, params, method, fold["y"], fold["prep"], seed)
    model.fit(Xb, yb)
    scores = model.predict_proba(fold["X_val"])
    return float(average_precision_score(fold["y_val"], scores))


def _oof_scores(
    model_name: str,
    method: str,
    params: dict,
    level: str | None,
    folds: list[dict],
    pools: list[np.ndarray | None],
    seed: int,
    n_train: int,
) -> np.ndarray:
    oof = np.zeros(n_train, dtype=float)
    noise_scale, gmm_components = _generator_kwargs(params)
    for fold_id, fold in enumerate(folds):
        Xb, yb = apply_resample(
            method,
            fold["X"],
            fold["y"],
            level,
            fold["prep"],
            seed + fold_id,
            noise_scale,
            gmm_components,
            pools[fold_id],
        )
        model = build_model(model_name, params, method, fold["y"], fold["prep"], seed)
        model.fit(Xb, yb)
        oof[fold["val_index"]] = model.predict_proba(fold["X_val"])
    return oof


def run_experiment(
    datasets: list[str],
    models: list[str],
    methods: list[str],
    trials: int,
    cv: int,
    ctgan_epochs: int,
    giveme_rows: int,
    home_rows: int,
    seed: int,
    test_size: float,
    tag: str,
) -> pd.DataFrame:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    RESULTS.mkdir(parents=True, exist_ok=True)
    metrics_path = RESULTS / ("metrics.csv" if tag == "main" else f"{tag}_metrics.csv")
    sweep_path = RESULTS / ("level_sweep.csv" if tag == "main" else f"{tag}_level_sweep.csv")
    quality_path = RESULTS / ("synth_quality.csv" if tag == "main" else f"{tag}_synth_quality.csv")
    profile_path = RESULTS / ("data_profile.csv" if tag == "main" else f"{tag}_data_profile.csv")
    db_path = RESULTS / ("optuna.db" if tag == "main" else f"{tag}_optuna.db")
    done = _load_done(metrics_path)
    rows: list[dict] = []
    if metrics_path.exists() and tag == "main":
        rows = pd.read_csv(metrics_path).to_dict("records")

    for dataset_name in datasets:
        log(f"\n=== Датасет {dataset_name} ===")
        X, y = load_dataset(dataset_name, giveme_rows, home_rows, seed)
        y_np = y.to_numpy(dtype=int)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_np, test_size=test_size, stratify=y_np, random_state=seed
        )
        X_train = X_train.reset_index(drop=True)
        X_test = X_test.reset_index(drop=True)
        profile = {
            "dataset": dataset_name,
            "n_rows": int(len(X)),
            "n_features": int(X.shape[1]),
            "positive_rate": float(y_np.mean()),
            "n_train": int(len(X_train)),
            "n_test": int(len(X_test)),
            "giveme_rows": giveme_rows if dataset_name == "giveme" else "",
            "home_rows": home_rows if dataset_name == "home" else "",
        }
        log(
            f"строк={profile['n_rows']}, признаков={profile['n_features']}, "
            f"доля дефолтов={profile['positive_rate']:.3f}, test={profile['n_test']}"
        )

        splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=seed)
        folds = []
        for train_index, val_index in splitter.split(X_train, y_train):
            prep = FoldPreprocessor().fit(X_train.iloc[train_index])
            folds.append(
                {
                    "prep": prep,
                    "X": prep.transform(X_train.iloc[train_index]),
                    "y": y_train[train_index],
                    "X_val": prep.transform(X_train.iloc[val_index]),
                    "y_val": y_train[val_index],
                    "val_index": val_index,
                }
            )

        full_prep = FoldPreprocessor().fit(X_train)
        X_full = full_prep.transform(X_train)
        X_holdout = full_prep.transform(X_test)
        profile["n_numeric"] = full_prep.n_num
        profile["n_categorical"] = len(full_prep.cat_cardinalities)
        if not _has_rows(profile_path, dataset_name):
            _append_csv(profile_path, profile)
        log(f"числовых={full_prep.n_num}, категориальных={len(full_prep.cat_cardinalities)}")

        ctgan_pools: list[np.ndarray | None] = [None] * len(folds)
        ctgan_full: np.ndarray | None = None
        if "ctgan" in methods:
            log(f"Обучение CTGAN, epochs={ctgan_epochs}")
            for fold_id, fold in enumerate(folds):
                started = time.time()
                ctgan_pools[fold_id] = fit_ctgan_pool(fold["X"], fold["y"], fold["prep"], ctgan_epochs, seed + fold_id)
                log(f"  фолд {fold_id + 1}/{len(folds)}: пул {len(ctgan_pools[fold_id])} за {time.time() - started:.1f}с")
            started = time.time()
            ctgan_full = fit_ctgan_pool(X_full, y_train, full_prep, ctgan_epochs, seed + 100)
            log(f"  полный train: пул {len(ctgan_full)} за {time.time() - started:.1f}с")
            for method in ("smote", "borderline_smote", "adasyn", "noise", "gmm", "ctgan"):
                if method not in methods or _has_rows(quality_path, dataset_name, method):
                    continue
                row = quality_row(method, X_full, y_train, full_prep, ctgan_full, seed)
                row["dataset"] = dataset_name
                _append_csv(quality_path, row)
                log(
                    f"  качество синтетики {method}: "
                    f"W={row['wasserstein_num']:.3f}, KS={row['ks_num']:.3f}, "
                    f"corr={row['corr_frobenius']:.3f}"
                )

        for model_name in models:
            for method in methods:
                key = (dataset_name, model_name, method)
                if key in done:
                    log(f"пропуск {dataset_name} / {model_name} / {method}: уже есть в {metrics_path.name}")
                    continue
                log(f"\n--- {dataset_name} | {model_name} | {method} ---")
                started = time.time()
                study_name = f"{tag}__{dataset_name}__{model_name}__{method}"
                study = optuna.create_study(
                    study_name=study_name,
                    storage=f"sqlite:///{db_path}",
                    load_if_exists=True,
                    direction="maximize",
                    sampler=optuna.samplers.TPESampler(seed=seed),
                )

                def objective(trial: optuna.Trial) -> float:
                    params = suggest_params(trial, model_name, method)
                    level = trial.suggest_categorical("level", list(LEVELS)) if method in RESAMPLE_METHODS else None
                    if level is not None:
                        params["level"] = level
                    scores = [
                        _fold_score(
                            model_name,
                            method,
                            params,
                            level,
                            fold,
                            ctgan_pools[fold_id],
                            seed + trial.number * 17 + fold_id,
                        )
                        for fold_id, fold in enumerate(folds)
                    ]
                    return float(np.mean(scores))

                finished = sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
                if finished < trials:
                    study.optimize(objective, n_trials=trials - finished, show_progress_bar=False)
                best = study.best_params
                params = dict(best)
                noise_scale, gmm_components = _generator_kwargs(params)

                chosen_level = None
                if method in RESAMPLE_METHODS:
                    best_level_score = -1.0
                    chosen_level = LEVELS[0]
                    for level in LEVELS:
                        level_scores = [
                            _fold_score(
                                model_name,
                                method,
                                params,
                                level,
                                fold,
                                ctgan_pools[fold_id],
                                seed + 500 + fold_id,
                            )
                            for fold_id, fold in enumerate(folds)
                        ]
                        mean_score = float(np.mean(level_scores))
                        _append_csv(
                            sweep_path,
                            {
                                "dataset": dataset_name,
                                "model": model_name,
                                "method": method,
                                "level": level,
                                "cv_pr_auc": mean_score,
                            },
                        )
                        log(f"  объём {level}: CV PR-AUC={mean_score:.4f}")
                        if mean_score > best_level_score:
                            best_level_score = mean_score
                            chosen_level = level
                else:
                    best_level_score = float(study.best_value)

                oof = _oof_scores(
                    model_name,
                    method,
                    params,
                    chosen_level,
                    folds,
                    ctgan_pools,
                    seed + 900,
                    len(y_train),
                )
                threshold = best_f1_threshold(y_train, oof)
                Xb, yb = apply_resample(
                    method,
                    X_full,
                    y_train,
                    chosen_level,
                    full_prep,
                    seed + 1200,
                    noise_scale,
                    gmm_components,
                    ctgan_full,
                )
                final_model = build_model(model_name, params, method, y_train, full_prep, seed)
                final_model.fit(Xb, yb)
                test_scores = final_model.predict_proba(X_holdout)
                metrics = classification_metrics(y_test, test_scores, threshold)
                elapsed = time.time() - started
                row = {
                    "dataset": dataset_name,
                    "model": model_name,
                    "method": method,
                    "level": chosen_level or "none",
                    "cv_pr_auc": best_level_score if method in RESAMPLE_METHODS else float(study.best_value),
                    "optuna_best_cv_pr_auc": float(study.best_value),
                    **metrics,
                    "n_train_augmented": int(len(yb)),
                    "seconds": round(elapsed, 1),
                    "params": json.dumps(best, ensure_ascii=False),
                }
                rows.append(row)
                _append_csv(metrics_path, row)
                done.add(key)
                log(
                    f"TEST PR-AUC={metrics['pr_auc']:.4f} ROC-AUC={metrics['roc_auc']:.4f} "
                    f"F1={metrics['f1']:.4f} Recall={metrics['recall']:.4f} "
                    f"level={row['level']} за {elapsed:.1f}с"
                )
    return pd.DataFrame(rows)


def _has_rows(path: Path, dataset: str, method: str | None = None) -> bool:
    if not path.exists():
        return False
    frame = pd.read_csv(path)
    mask = frame["dataset"] == dataset
    if method is not None:
        mask &= frame["method"] == method
    return bool(mask.any())


def _load_done(path: Path) -> set[tuple[str, str, str]]:
    if not path.exists():
        return set()
    frame = pd.read_csv(path)
    return set(zip(frame["dataset"], frame["model"], frame["method"]))


def _append_csv(path: Path, row: dict) -> None:
    frame = pd.DataFrame([row])
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, mode="a", header=not path.exists(), index=False)
