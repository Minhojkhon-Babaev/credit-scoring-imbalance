from __future__ import annotations

import numpy as np
from scipy.stats import entropy, ks_2samp, wasserstein_distance

from credit_scoring.resample import ctgan_resample, gmm_resample, noise_resample, smote_resample, _distance_resample


def _mean_distance(real: np.ndarray, synth: np.ndarray, kind: str) -> float:
    if real.shape[1] == 0 or len(synth) == 0:
        return float("nan")
    values = []
    for column in range(real.shape[1]):
        if kind == "wasserstein":
            values.append(wasserstein_distance(real[:, column], synth[:, column]))
        else:
            values.append(ks_2samp(real[:, column], synth[:, column]).statistic)
    return float(np.mean(values))


def _correlation_gap(real: np.ndarray, synth: np.ndarray) -> float:
    if real.shape[1] < 2 or len(synth) < 2:
        return float("nan")
    real_corr = np.corrcoef(real, rowvar=False)
    synth_corr = np.corrcoef(synth, rowvar=False)
    real_corr = np.nan_to_num(real_corr, nan=0.0)
    synth_corr = np.nan_to_num(synth_corr, nan=0.0)
    return float(np.linalg.norm(real_corr - synth_corr, ord="fro"))


def _js_categories(real: np.ndarray, synth: np.ndarray, cardinalities: list[int], n_num: int) -> float:
    if not cardinalities or len(synth) == 0:
        return float("nan")
    divergences = []
    for offset, width in enumerate(cardinalities):
        real_codes = np.clip(np.rint(real[:, n_num + offset]).astype(int), 0, width - 1)
        synth_codes = np.clip(np.rint(synth[:, n_num + offset]).astype(int), 0, width - 1)
        p = np.bincount(real_codes, minlength=width).astype(float)
        q = np.bincount(synth_codes, minlength=width).astype(float)
        p /= p.sum()
        q /= q.sum()
        mixture = 0.5 * (p + q)
        divergences.append(0.5 * (entropy(p, mixture, base=2) + entropy(q, mixture, base=2)))
    return float(np.mean(divergences))


def synthetic_block(method: str, X: np.ndarray, y: np.ndarray, prep, pool: np.ndarray, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    level = "balance"
    if method == "smote":
        augmented, labels = smote_resample(X, y, level, prep.cat_cardinalities, prep.n_num, seed)
    elif method in {"borderline_smote", "adasyn"}:
        augmented, labels = _distance_resample(method, X, y, level, prep.cat_cardinalities, prep.n_num, seed)
    elif method == "noise":
        augmented, labels = noise_resample(X, y, level, prep.n_num, 0.1, rng, prep.cat_cardinalities)
    elif method == "gmm":
        augmented, labels = gmm_resample(X, y, level, prep.n_num, 3, rng, prep.cat_cardinalities)
    elif method == "ctgan":
        augmented, labels = ctgan_resample(X, y, level, pool, rng, prep.cat_cardinalities, prep.n_num)
    else:
        raise ValueError(method)
    if len(augmented) <= len(X):
        return augmented[0:0]
    return augmented[len(X) :]


def quality_row(method: str, X: np.ndarray, y: np.ndarray, prep, pool: np.ndarray, seed: int) -> dict:
    real = X[y == 1]
    synth = synthetic_block(method, X, y, prep, pool, seed)
    return {
        "method": method,
        "n_synthetic": int(len(synth)),
        "wasserstein_num": _mean_distance(real[:, : prep.n_num], synth[:, : prep.n_num], "wasserstein"),
        "ks_num": _mean_distance(real[:, : prep.n_num], synth[:, : prep.n_num], "ks"),
        "corr_frobenius": _correlation_gap(real[:, : prep.n_num], synth[:, : prep.n_num]),
        "js_categorical": _js_categories(real, synth, prep.cat_cardinalities, prep.n_num),
    }
