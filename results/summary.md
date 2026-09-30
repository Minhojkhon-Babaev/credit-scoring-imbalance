# Результаты эксперимента

Optuna подбирает гиперпараметры отдельно для каждого метода. Метрики ниже посчитаны один раз на отложенном тесте. Порог F1 выбран по out-of-fold предсказаниям обучающей части.

| Датасет | Модель | Метод | Объём | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---:|---:|---:|---:|
| german | CatBoost | Без баланса | none | 0.663 | 0.796 | 0.575 | 0.833 |
| german | CatBoost | Веса классов | none | 0.644 | 0.804 | 0.594 | 0.817 |
| german | CatBoost | SMOTE | plus_25 | 0.662 | 0.800 | 0.583 | 0.850 |
| german | CatBoost | ADASYN | plus_25 | 0.686 | 0.808 | 0.574 | 0.900 |
| german | CatBoost | CTGAN | balance | 0.686 | 0.811 | 0.623 | 0.717 |
| german | LogReg | Без баланса | none | 0.663 | 0.809 | 0.623 | 0.800 |
| german | LogReg | Веса классов | none | 0.651 | 0.810 | 0.620 | 0.817 |
| german | LogReg | SMOTE | plus_50 | 0.668 | 0.808 | 0.644 | 0.800 |
| german | LogReg | ADASYN | plus_25 | 0.674 | 0.811 | 0.599 | 0.833 |
| german | LogReg | CTGAN | plus_25 | 0.647 | 0.809 | 0.636 | 0.817 |
| german | XGBoost | Без баланса | none | 0.661 | 0.792 | 0.559 | 0.633 |
| german | XGBoost | Веса классов | none | 0.631 | 0.790 | 0.619 | 0.717 |
| german | XGBoost | SMOTE | plus_50 | 0.644 | 0.782 | 0.577 | 0.717 |
| german | XGBoost | ADASYN | plus_25 | 0.633 | 0.778 | 0.557 | 0.650 |
| german | XGBoost | CTGAN | plus_50 | 0.634 | 0.786 | 0.590 | 0.817 |
| giveme | CatBoost | Без баланса | none | 0.411 | 0.869 | 0.448 | 0.494 |
| giveme | CatBoost | Веса классов | none | 0.409 | 0.869 | 0.448 | 0.523 |
| giveme | CatBoost | SMOTE | plus_25 | 0.411 | 0.869 | 0.449 | 0.493 |
| giveme | CatBoost | ADASYN | plus_25 | 0.411 | 0.869 | 0.449 | 0.493 |
| giveme | CatBoost | CTGAN | plus_25 | 0.413 | 0.869 | 0.447 | 0.494 |
| giveme | LogReg | Без баланса | none | 0.245 | 0.714 | 0.331 | 0.336 |
| giveme | LogReg | Веса классов | none | 0.336 | 0.811 | 0.404 | 0.414 |
| giveme | LogReg | SMOTE | balance | 0.312 | 0.796 | 0.387 | 0.399 |
| giveme | LogReg | ADASYN | balance | 0.289 | 0.784 | 0.366 | 0.397 |
| giveme | LogReg | CTGAN | balance | 0.305 | 0.797 | 0.380 | 0.412 |
| giveme | XGBoost | Без баланса | none | 0.412 | 0.869 | 0.449 | 0.491 |
| giveme | XGBoost | Веса классов | none | 0.408 | 0.869 | 0.447 | 0.525 |
| giveme | XGBoost | SMOTE | plus_25 | 0.410 | 0.869 | 0.444 | 0.492 |
| giveme | XGBoost | ADASYN | plus_25 | 0.410 | 0.869 | 0.444 | 0.492 |
| giveme | XGBoost | CTGAN | plus_100 | 0.414 | 0.869 | 0.448 | 0.527 |
