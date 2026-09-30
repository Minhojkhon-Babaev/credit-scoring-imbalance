from __future__ import annotations

import urllib.request
from pathlib import Path

import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
GIVEME_URL = (
    "https://raw.githubusercontent.com/ethen8181/programming/master/"
    "kaggle/give_me_some_credit/data/cs-training.csv"
)


def load_german() -> tuple[pd.DataFrame, pd.Series]:
    """Statlog German Credit (OpenML credit-g): 1000 строк, класс bad — дефолт."""
    bunch = fetch_openml("credit-g", version=1, as_frame=True, parser="auto")
    frame = bunch.frame.copy() if bunch.frame is not None else None
    if frame is not None and "class" in frame.columns:
        X = frame.drop(columns=["class"])
        y_raw = frame["class"].astype(str)
    else:
        X = bunch.data.copy()
        y_raw = pd.Series(bunch.target).astype(str)
    y = y_raw.str.lower().str.contains("bad").astype(int)
    y.name = "default"
    return X.reset_index(drop=True), y.reset_index(drop=True)


def load_giveme(max_rows: int | None, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    """Give Me Some Credit. max_rows=None или 0 — вся обучающая выборка."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    path = DATA_DIR / "giveme_training.csv"
    if not path.exists() or path.stat().st_size < 10_000:
        urllib.request.urlretrieve(GIVEME_URL, path)
    df = pd.read_csv(path)
    drop_cols = [c for c in df.columns if str(c).startswith("Unnamed") or str(c).strip() == ""]
    df = df.drop(columns=drop_cols)
    target = "SeriousDlqin2yrs"
    if target not in df.columns:
        raise ValueError(f"В {path} нет колонки {target}. Колонки: {list(df.columns)}")
    y = df[target].astype(int)
    X = df.drop(columns=[target])
    if max_rows and 0 < max_rows < len(df):
        X, _, y, _ = train_test_split(
            X, y, train_size=max_rows, stratify=y, random_state=seed
        )
    y = y.astype(int)
    y.name = "default"
    return X.reset_index(drop=True), y.reset_index(drop=True)


def recover_numeric(X: pd.DataFrame) -> pd.DataFrame:
    """Возвращает числовой тип колонкам, которые OpenML отдал как категории из цифр."""
    recovered = X.copy()
    for column in recovered.columns:
        if pd.api.types.is_numeric_dtype(recovered[column]):
            continue
        numeric = pd.to_numeric(recovered[column], errors="coerce")
        if numeric.notna().mean() > 0.98:
            recovered[column] = numeric
    return recovered


def load_dataset(name: str, giveme_rows: int, seed: int) -> tuple[pd.DataFrame, pd.Series]:
    if name == "german":
        X, y = load_german()
    elif name == "giveme":
        X, y = load_giveme(giveme_rows or None, seed)
    else:
        raise ValueError(f"Неизвестный датасет: {name}")
    return recover_numeric(X), y
