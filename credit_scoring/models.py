from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression


def suggest_params(trial, model_name: str, method: str) -> dict:
    if model_name == "logreg":
        params = {"C": trial.suggest_float("C", 1e-3, 100.0, log=True)}
    elif model_name == "xgboost":
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "n_estimators": trial.suggest_int("n_estimators", 80, 280),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 8),
            "subsample": trial.suggest_float("subsample", 0.7, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.7, 1.0),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-2, 10.0, log=True),
        }
    elif model_name == "catboost":
        params = {
            "depth": trial.suggest_int("depth", 3, 6),
            "learning_rate": trial.suggest_float("learning_rate", 0.03, 0.3, log=True),
            "iterations": trial.suggest_int("iterations", 80, 280),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 10.0),
        }
    else:
        raise ValueError(model_name)

    if method == "class_weight":
        params["weight_multiplier"] = trial.suggest_float("weight_multiplier", 0.5, 3.0)
    return params


class LogisticModel:
    def __init__(self, C: float, class_weight, n_num: int, cardinalities: list[int], seed: int) -> None:
        self.C = C
        self.class_weight = class_weight
        self.n_num = n_num
        self.cardinalities = cardinalities
        self.seed = seed
        self.clf: LogisticRegression | None = None

    def _design(self, X: np.ndarray) -> np.ndarray:
        numeric = X[:, : self.n_num] if self.n_num else np.zeros((len(X), 0))
        parts = [numeric] if self.n_num else []
        for offset, width in enumerate(self.cardinalities):
            codes = np.clip(np.rint(X[:, self.n_num + offset]).astype(int), 0, width - 1)
            parts.append(np.eye(width, dtype=float)[codes])
        if not parts:
            raise ValueError("Пустая матрица признаков")
        return np.hstack(parts)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LogisticModel":
        self.clf = LogisticRegression(
            C=self.C,
            class_weight=self.class_weight,
            solver="lbfgs",
            max_iter=1000,
            random_state=self.seed,
        )
        self.clf.fit(self._design(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Модель не обучена")
        return self.clf.predict_proba(self._design(X))[:, 1]


class XGBoostModel:
    def __init__(self, params: dict, scale_pos_weight: float, seed: int) -> None:
        self.params = params
        self.scale_pos_weight = scale_pos_weight
        self.seed = seed
        self.clf = None

    def fit(self, X: np.ndarray, y: np.ndarray) -> "XGBoostModel":
        from xgboost import XGBClassifier

        self.clf = XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            tree_method="hist",
            random_state=self.seed,
            n_jobs=1,
            scale_pos_weight=self.scale_pos_weight,
            **self.params,
        )
        self.clf.fit(X, y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Модель не обучена")
        return self.clf.predict_proba(X)[:, 1]


class CatBoostModel:
    def __init__(self, params: dict, positive_class_weight: float | None, cat_indices: list[int], seed: int) -> None:
        self.params = params
        self.positive_class_weight = positive_class_weight
        self.cat_indices = cat_indices
        self.seed = seed
        self.clf = None

    def _frame(self, X: np.ndarray):
        import pandas as pd

        frame = pd.DataFrame(X)
        for index in self.cat_indices:
            frame[index] = np.rint(frame[index]).astype(int)
        return frame

    def fit(self, X: np.ndarray, y: np.ndarray) -> "CatBoostModel":
        from catboost import CatBoostClassifier

        kwargs = dict(
            loss_function="Logloss",
            random_seed=self.seed,
            verbose=False,
            allow_writing_files=False,
            thread_count=4,
            **self.params,
        )
        if self.positive_class_weight is not None:
            kwargs["class_weights"] = [1.0, self.positive_class_weight]
        if self.cat_indices:
            kwargs["cat_features"] = self.cat_indices
        self.clf = CatBoostClassifier(**kwargs)
        self.clf.fit(self._frame(X), y)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        if self.clf is None:
            raise RuntimeError("Модель не обучена")
        return self.clf.predict_proba(self._frame(X))[:, 1]


def positive_weight(y: np.ndarray) -> float:
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = int((y == 0).sum())
    return n_neg / n_pos


def build_model(model_name: str, params: dict, method: str, y_for_weight: np.ndarray, prep, seed: int):
    model_params = {
        key: value
        for key, value in params.items()
        if key not in {"level", "weight_multiplier"}
    }
    balanced = method == "class_weight"
    multiplier = float(params.get("weight_multiplier", 1.0)) if balanced else 1.0
    positive = positive_weight(y_for_weight) * multiplier
    if model_name == "logreg":
        return LogisticModel(
            C=float(model_params["C"]),
            class_weight={0: 1.0, 1: positive} if balanced else None,
            n_num=prep.n_num,
            cardinalities=prep.cat_cardinalities,
            seed=seed,
        )
    if model_name == "xgboost":
        return XGBoostModel(
            params=model_params,
            scale_pos_weight=positive if balanced else 1.0,
            seed=seed,
        )
    if model_name == "catboost":
        return CatBoostModel(
            params=model_params,
            positive_class_weight=positive if balanced else None,
            cat_indices=prep.cat_indices,
            seed=seed,
        )
    raise ValueError(model_name)
