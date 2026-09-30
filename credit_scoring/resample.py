from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC

from credit_scoring.preprocess import snap_categories

LEVELS = ("plus_25", "plus_50", "plus_100", "balance")
LEVEL_MULTIPLIER = {"plus_25": 1.25, "plus_50": 1.5, "plus_100": 2.0}
RESAMPLE_METHODS = ("smote", "adasyn", "ctgan")


def target_ratio(level: str, n_pos: int, n_neg: int) -> float:
    if level == "balance":
        return 1.0
    return (n_pos * LEVEL_MULTIPLIER[level]) / max(n_neg, 1)


def n_synthetic(level: str, n_pos: int, n_neg: int) -> int:
    desired = target_ratio(level, n_pos, n_neg)
    return max(int(round(desired * n_neg - n_pos)), 0)


def _neighbor_k(n_pos: int) -> int:
    return max(1, min(5, n_pos - 1))


def _concat(X: np.ndarray, y: np.ndarray, synth: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    # Оригинальные строки остаются префиксом: хвост — только синтетика.
    del rng
    stacked = np.vstack([X, synth])
    labels = np.concatenate([y, np.ones(len(synth), dtype=int)])
    return stacked, labels


def _sampling_strategy(level: str, y: np.ndarray) -> float | None:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    if n_pos < 2 or n_neg < 2:
        return None
    ratio = target_ratio(level, n_pos, n_neg)
    if ratio <= n_pos / n_neg + 1e-8:
        return None
    return float(ratio)


def _scale_categories(X: np.ndarray, n_num: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scaled = np.array(X, copy=True)
    cats = scaled[:, n_num:]
    means = cats.mean(axis=0)
    stds = cats.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    scaled[:, n_num:] = (cats - means) / stds
    return scaled, means, stds


def _unscale_categories(
    X: np.ndarray,
    n_num: int,
    means: np.ndarray,
    stds: np.ndarray,
    cardinalities: list[int],
) -> np.ndarray:
    restored = np.array(X, copy=True)
    restored[:, n_num:] = restored[:, n_num:] * stds + means
    return snap_categories(restored, cardinalities, n_num)


def smote_resample(
    X: np.ndarray,
    y: np.ndarray,
    level: str,
    cardinalities: list[int],
    n_num: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    strategy = _sampling_strategy(level, y)
    if strategy is None:
        return X, y
    k = _neighbor_k(int((y == 1).sum()))
    if cardinalities and n_num > 0:
        sampler = SMOTENC(
            categorical_features=list(range(n_num, X.shape[1])),
            sampling_strategy=strategy,
            k_neighbors=k,
            random_state=seed,
        )
    elif cardinalities and n_num == 0:
        return X, y
    else:
        sampler = SMOTE(sampling_strategy=strategy, k_neighbors=k, random_state=seed)
    X_res, y_res = sampler.fit_resample(X, y)
    return snap_categories(np.asarray(X_res, dtype=float), cardinalities, n_num), np.asarray(y_res, dtype=int)


def _distance_resample(
    X: np.ndarray,
    y: np.ndarray,
    level: str,
    cardinalities: list[int],
    n_num: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    strategy = _sampling_strategy(level, y)
    if strategy is None:
        return X, y
    n_pos = int((y == 1).sum())
    k = _neighbor_k(n_pos)
    if cardinalities:
        scaled, means, stds = _scale_categories(X, n_num)
    else:
        scaled, means, stds = X, None, None
    sampler = ADASYN(
        sampling_strategy=strategy,
        n_neighbors=k,
        random_state=seed,
    )
    try:
        X_res, y_res = sampler.fit_resample(scaled, y)
    except ValueError:
        return smote_resample(X, y, level, cardinalities, n_num, seed)
    X_res = np.asarray(X_res, dtype=float)
    if means is not None and stds is not None:
        X_res = _unscale_categories(X_res, n_num, means, stds, cardinalities)
    else:
        X_res = snap_categories(X_res, cardinalities, n_num)
    return X_res, np.asarray(y_res, dtype=int)


def ctgan_resample(
    X: np.ndarray,
    y: np.ndarray,
    level: str,
    pool: np.ndarray,
    rng: np.random.Generator,
    cardinalities: list[int],
    n_num: int,
) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_new = n_synthetic(level, n_pos, n_neg)
    if n_new <= 0 or len(pool) == 0:
        return X, y
    if n_new <= len(pool):
        synth = pool[:n_new]
    else:
        extra = pool[rng.integers(0, len(pool), size=n_new - len(pool))]
        synth = np.vstack([pool, extra])
    synth = snap_categories(np.asarray(synth, dtype=float), cardinalities, n_num)
    return _concat(X, y, synth, rng)
