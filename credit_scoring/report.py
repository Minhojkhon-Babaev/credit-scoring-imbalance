from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

RESULTS = Path(__file__).resolve().parents[1] / "results"

METHOD_ORDER = [
    "raw",
    "class_weight",
    "smote",
    "borderline_smote",
    "adasyn",
    "noise",
    "gmm",
    "ctgan",
]
METHOD_LABELS = {
    "raw": "Без баланса",
    "class_weight": "Веса классов",
    "smote": "SMOTE",
    "borderline_smote": "Borderline-SMOTE",
    "adasyn": "ADASYN",
    "noise": "Шум",
    "gmm": "GMM",
    "ctgan": "CTGAN",
}
MODEL_LABELS = {"logreg": "LogReg", "xgboost": "XGBoost", "catboost": "CatBoost"}


def write_report(frame: pd.DataFrame) -> None:
    RESULTS.mkdir(parents=True, exist_ok=True)
    ordered = frame.copy()
    ordered["method"] = pd.Categorical(ordered["method"], categories=METHOD_ORDER, ordered=True)
    ordered = ordered.sort_values(["dataset", "model", "method"])
    columns = [
        "dataset",
        "model",
        "method",
        "level",
        "pr_auc",
        "roc_auc",
        "f1",
        "recall",
        "precision",
        "cv_pr_auc",
    ]
    table = ordered[columns].copy()
    for column in ("pr_auc", "roc_auc", "f1", "recall", "precision", "cv_pr_auc"):
        table[column] = table[column].map(lambda value: f"{float(value):.3f}")
    table.to_csv(RESULTS / "metrics_rounded.csv", index=False)

    lines = [
        "# Результаты эксперимента",
        "",
        "Optuna подбирает гиперпараметры отдельно для каждого метода. "
        "Метрики ниже посчитаны один раз на отложенном тесте. "
        "Порог F1 выбран по out-of-fold предсказаниям обучающей части.",
        "",
        "| Датасет | Модель | Метод | Объём | PR-AUC | ROC-AUC | F1 | Recall |",
        "|---|---|---|---|---:|---:|---:|---:|",
    ]
    for _, row in ordered.iterrows():
        lines.append(
            f"| {row['dataset']} | {MODEL_LABELS.get(row['model'], row['model'])} | "
            f"{METHOD_LABELS.get(row['method'], row['method'])} | {row['level']} | "
            f"{row['pr_auc']:.3f} | {row['roc_auc']:.3f} | {row['f1']:.3f} | {row['recall']:.3f} |"
        )
    lines.append("")
    (RESULTS / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    _plot_metric(ordered, "pr_auc", "PR-AUC на тесте", RESULTS / "pr_auc.png")
    _plot_metric(ordered, "recall", "Recall дефолта на тесте", RESULTS / "recall.png")


def _plot_metric(frame: pd.DataFrame, metric: str, title: str, path: Path) -> None:
    datasets = list(frame["dataset"].unique())
    fig, axes = plt.subplots(1, len(datasets), figsize=(6.2 * len(datasets), 4.4), squeeze=False)
    colors = {"logreg": "#1F4E79", "xgboost": "#0F6E6E", "catboost": "#B08D2A"}
    for axis, dataset in zip(axes[0], datasets):
        subset = frame[frame["dataset"] == dataset]
        methods = [method for method in METHOD_ORDER if method in set(subset["method"])]
        models = [model for model in ("logreg", "xgboost", "catboost") if model in set(subset["model"])]
        x = range(len(methods))
        width = 0.24
        for offset, model in enumerate(models):
            values = []
            for method in methods:
                match = subset[(subset["model"] == model) & (subset["method"] == method)]
                values.append(float(match[metric].iloc[0]) if not match.empty else 0.0)
            positions = [item + (offset - (len(models) - 1) / 2) * width for item in x]
            axis.bar(positions, values, width=width, label=MODEL_LABELS[model], color=colors[model])
        axis.set_xticks(list(x), [METHOD_LABELS[method] for method in methods], rotation=35, ha="right")
        axis.set_title(dataset)
        axis.set_ylabel(metric)
        axis.legend(frameon=False)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
