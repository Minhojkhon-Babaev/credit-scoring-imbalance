from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OrdinalEncoder, StandardScaler


class FoldPreprocessor:
    """Импутация, ordinal для категорий и StandardScaler для чисел.

    Категории остаются целочисленными кодами: так SMOTENC и CatBoost
    не интерполируют их как непрерывные величины. Числа масштабируются,
    чтобы дистанционные методы (SMOTE, ADASYN) не зависели от единиц измерения.
    """

    def __init__(self) -> None:
        self.num_cols: list[str] = []
        self.cat_cols: list[str] = []
        self.n_num: int = 0
        self.cat_cardinalities: list[int] = []
        self.num_imputer: SimpleImputer | None = None
        self.cat_imputer: SimpleImputer | None = None
        self.encoder: OrdinalEncoder | None = None
        self.scaler: StandardScaler | None = None

    def fit(self, X: pd.DataFrame) -> "FoldPreprocessor":
        self.num_cols = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
        self.cat_cols = [c for c in X.columns if c not in self.num_cols]
        self.n_num = len(self.num_cols)
        if self.num_cols:
            self.num_imputer = SimpleImputer(strategy="median")
            self.scaler = StandardScaler()
            num = self.num_imputer.fit_transform(X[self.num_cols])
            self.scaler.fit(num)
        if self.cat_cols:
            self.cat_imputer = SimpleImputer(strategy="most_frequent")
            self.encoder = OrdinalEncoder(
                handle_unknown="use_encoded_value",
                unknown_value=-1,
                encoded_missing_value=-1,
            )
            cat = self.cat_imputer.fit_transform(X[self.cat_cols].astype(object))
            self.encoder.fit(cat)
            self.cat_cardinalities = [len(categories) for categories in self.encoder.categories_]
        return self

    def transform(self, X: pd.DataFrame) -> np.ndarray:
        parts: list[np.ndarray] = []
        if self.num_cols and self.num_imputer is not None and self.scaler is not None:
            num = self.scaler.transform(self.num_imputer.transform(X[self.num_cols]))
            parts.append(np.asarray(num, dtype=float))
        if self.cat_cols and self.cat_imputer is not None and self.encoder is not None:
            cat = self.encoder.transform(self.cat_imputer.transform(X[self.cat_cols].astype(object)))
            parts.append(np.asarray(cat, dtype=float))
        if not parts:
            raise ValueError("В выборке нет признаков")
        matrix = np.hstack(parts)
        return snap_categories(np.nan_to_num(matrix, nan=0.0), self.cat_cardinalities, self.n_num)

    @property
    def cat_indices(self) -> list[int]:
        return list(range(self.n_num, self.n_num + len(self.cat_cardinalities)))


def snap_categories(X: np.ndarray, cardinalities: list[int], n_num: int) -> np.ndarray:
    if not cardinalities:
        return X
    snapped = np.array(X, copy=True, dtype=float)
    for offset, width in enumerate(cardinalities):
        column = np.rint(snapped[:, n_num + offset])
        snapped[:, n_num + offset] = np.clip(column, 0, max(width - 1, 0))
    return snapped
