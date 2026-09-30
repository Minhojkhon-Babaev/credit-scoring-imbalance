#!/usr/bin/env python3
"""Полный эксперимент: baseline, лёгкие oversampling-методы, CTGAN, Optuna."""

from __future__ import annotations

import os

# Torch (CTGAN) и XGBoost оба тянут OpenMP. Без этого флага процесс падает на macOS.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("OMP_NUM_THREADS", "1")

import argparse

from credit_scoring.experiment import ALL_METHODS, ALL_MODELS, run_experiment
from credit_scoring.report import write_report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Кредитный скоринг при дисбалансе классов")
    parser.add_argument("--datasets", nargs="+", default=["german", "giveme"], choices=["german", "giveme"])
    parser.add_argument("--models", nargs="+", default=list(ALL_MODELS), choices=list(ALL_MODELS))
    parser.add_argument("--methods", nargs="+", default=list(ALL_METHODS), choices=list(ALL_METHODS))
    parser.add_argument("--trials", type=int, default=12, help="Число trials Optuna на каждую пару модель×метод")
    parser.add_argument("--cv", type=int, default=3)
    parser.add_argument("--ctgan-epochs", type=int, default=300)
    parser.add_argument(
        "--giveme-rows",
        type=int,
        default=8000,
        help="Стратифицированная подвыборка Give Me Some Credit. 0 — все 150000 строк",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--tag", default="main", help="Префикс файлов результата. smoke не затирает основной прогон")
    parser.add_argument("--skip-report", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = run_experiment(
        datasets=args.datasets,
        models=args.models,
        methods=args.methods,
        trials=args.trials,
        cv=args.cv,
        ctgan_epochs=args.ctgan_epochs,
        giveme_rows=args.giveme_rows,
        seed=args.seed,
        test_size=args.test_size,
        tag=args.tag,
    )
    if not args.skip_report and args.tag == "main" and not frame.empty:
        write_report(frame)


if __name__ == "__main__":
    main()
