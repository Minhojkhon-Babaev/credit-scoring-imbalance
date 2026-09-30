from __future__ import annotations

import inspect
import random

import numpy as np
import pandas as pd

from credit_scoring.preprocess import FoldPreprocessor, snap_categories


def fit_ctgan_pool(
    X: np.ndarray,
    y: np.ndarray,
    prep: FoldPreprocessor,
    epochs: int,
    seed: int,
) -> np.ndarray:
    """Обучает CTGAN только на дефолтах и возвращает пул до баланса 1:1.

    Пул кэшируется на фолд: Optuna потом лишь выбирает, сколько строк из него взять.
    Так генератор не переобучается внутри каждого trial.
    """
    from ctgan import CTGAN

    minority = X[y == 1]
    n_pos = int(len(minority))
    n_neg = int((y == 0).sum())
    n_new = max(n_neg - n_pos, 1)
    if n_pos < 20:
        raise RuntimeError(f"Слишком мало дефолтов для CTGAN: {n_pos}")

    columns = [f"f{i}" for i in range(X.shape[1])]
    frame = pd.DataFrame(minority, columns=columns)
    discrete = columns[prep.n_num :]
    for column in discrete:
        frame[column] = np.rint(frame[column]).astype(int)

    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch

        torch.manual_seed(seed)
    except Exception:
        pass

    pac = 10
    capped = min(n_pos, 200)
    batch = max(pac, (capped // pac) * pac)
    if batch > n_pos:
        batch = max(pac, (n_pos // pac) * pac)
    signature = inspect.signature(CTGAN.__init__)
    kwargs: dict = {"epochs": epochs, "batch_size": int(batch), "verbose": False}
    if "pac" in signature.parameters:
        kwargs["pac"] = pac
    if "enable_gpu" in signature.parameters:
        kwargs["enable_gpu"] = False
    elif "cuda" in signature.parameters:
        kwargs["cuda"] = False
    model = CTGAN(**kwargs)
    model.fit(frame, discrete_columns=discrete)
    sampled = model.sample(n_new)
    sampled = sampled.reindex(columns=columns)
    pool = np.nan_to_num(sampled.to_numpy(dtype=float), nan=0.0)
    return snap_categories(pool, prep.cat_cardinalities, prep.n_num)
