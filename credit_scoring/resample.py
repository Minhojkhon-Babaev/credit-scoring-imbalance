from __future__ import annotations

import numpy as np
from imblearn.over_sampling import ADASYN, SMOTE, SMOTENC, BorderlineSMOTE
from sklearn.mixture import GaussianMixture
from sklearn.neighbors import NearestNeighbors

from credit_scoring.preprocess import snap_categories

LEVELS = ("plus_25", "plus_50", "plus_100", "balance")
LEVEL_MULTIPLIER = {"plus_25": 1.25, "plus_50": 1.5, "plus_100": 2.0}
RESAMPLE_METHODS = ("smote", "borderline_smote", "adasyn", "noise", "gmm", "ctgan")


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
    sampler_name: str,
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
    n_neg = int((y == 0).sum())
    k = _neighbor_k(n_pos)
    if cardinalities:
        scaled, means, stds = _scale_categories(X, n_num)
    else:
        scaled, means, stds = X, None, None
    if sampler_name == "borderline_smote":
        sampler = BorderlineSMOTE(
            sampling_strategy=strategy,
            k_neighbors=k,
            m_neighbors=max(1, min(10, n_neg - 1)),
            random_state=seed,
        )
    else:
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


def noise_resample(
    X: np.ndarray,
    y: np.ndarray,
    level: str,
    n_num: int,
    noise_scale: float,
    rng: np.random.Generator,
    cardinalities: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_new = n_synthetic(level, n_pos, n_neg)
    if n_new <= 0 or n_pos == 0:
        return X, y
    minority = X[y == 1]
    picked = minority[rng.integers(0, len(minority), size=n_new)].copy()
    if n_num > 0:
        picked[:, :n_num] += rng.normal(0.0, noise_scale, size=(n_new, n_num))
    picked = snap_categories(picked, cardinalities, n_num)
    return _concat(X, y, picked, rng)


def gmm_resample(
    X: np.ndarray,
    y: np.ndarray,
    level: str,
    n_num: int,
    n_components: int,
    rng: np.random.Generator,
    cardinalities: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """Генерация миноритарного класса смесью гауссиан по числовым признакам.

    Категории копируются у ближайшего реального дефолта: смесь не выдумывает
    несуществующие коды категорий.
    """
    n_pos = int((y == 1).sum())
    n_neg = int((y == 0).sum())
    n_new = n_synthetic(level, n_pos, n_neg)
    if n_new <= 0 or n_pos < 2 or n_num == 0:
        if n_new > 0 and n_pos > 0 and n_num == 0:
            minority = X[y == 1]
            picked = minority[rng.integers(0, len(minority), size=n_new)]
            return _concat(X, y, picked, rng)
        return X, y
    minority = X[y == 1]
    numeric = minority[:, :n_num]
    components = int(max(1, min(n_components, n_pos, numeric.shape[1] + 1)))
    mixture = GaussianMixture(
        n_components=components,
        covariance_type="diag",
        reg_covar=1e-4,
        random_state=int(rng.integers(0, 1_000_000)),
    )
    mixture.fit(numeric)
    synth_num, _ = mixture.sample(n_new)
    neighbors = NearestNeighbors(n_neighbors=1).fit(numeric)
    _, indices = neighbors.kneighbors(synth_num)
    if cardinalities:
        synth = np.hstack([synth_num, minority[indices[:, 0], n_num:]])
    else:
        synth = synth_num
    synth = snap_categories(np.asarray(synth, dtype=float), cardinalities, n_num)
    return _concat(X, y, synth, rng)


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
