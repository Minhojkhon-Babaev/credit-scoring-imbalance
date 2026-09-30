# Результаты эксперимента

Optuna подбирает гиперпараметры отдельно для каждого метода. Метрики ниже посчитаны один раз на отложенном тесте. Порог F1 выбран по out-of-fold предсказаниям обучающей части.

| Датасет | Модель | Метод | Объём | PR-AUC | ROC-AUC | F1 | Recall |
|---|---|---|---|---:|---:|---:|---:|
| german | CatBoost | Без баланса | none | 0.663 | 0.796 | 0.575 | 0.833 |
| german | CatBoost | Веса классов | none | 0.644 | 0.804 | 0.594 | 0.817 |
| german | CatBoost | SMOTE | plus_25 | 0.662 | 0.800 | 0.583 | 0.850 |
| german | CatBoost | Borderline-SMOTE | plus_50 | 0.545 | 0.750 | 0.571 | 0.733 |
| german | CatBoost | ADASYN | plus_25 | 0.686 | 0.808 | 0.574 | 0.900 |
| german | CatBoost | Шум | balance | 0.661 | 0.796 | 0.555 | 0.800 |
| german | CatBoost | GMM | plus_25 | 0.708 | 0.819 | 0.644 | 0.800 |
| german | CatBoost | CTGAN | balance | 0.686 | 0.811 | 0.623 | 0.717 |
| german | LogReg | Без баланса | none | 0.663 | 0.809 | 0.623 | 0.800 |
| german | LogReg | Веса классов | none | 0.651 | 0.810 | 0.620 | 0.817 |
| german | LogReg | SMOTE | plus_50 | 0.668 | 0.808 | 0.644 | 0.800 |
| german | LogReg | Borderline-SMOTE | plus_25 | 0.640 | 0.809 | 0.624 | 0.817 |
| german | LogReg | ADASYN | plus_25 | 0.674 | 0.811 | 0.599 | 0.833 |
| german | LogReg | Шум | balance | 0.645 | 0.807 | 0.667 | 0.767 |
| german | LogReg | GMM | plus_50 | 0.642 | 0.806 | 0.590 | 0.850 |
| german | LogReg | CTGAN | plus_25 | 0.647 | 0.809 | 0.636 | 0.817 |
| german | XGBoost | Без баланса | none | 0.661 | 0.792 | 0.559 | 0.633 |
| german | XGBoost | Веса классов | none | 0.631 | 0.790 | 0.619 | 0.717 |
| german | XGBoost | SMOTE | plus_50 | 0.644 | 0.782 | 0.577 | 0.717 |
| german | XGBoost | Borderline-SMOTE | plus_50 | 0.638 | 0.794 | 0.586 | 0.650 |
| german | XGBoost | ADASYN | plus_25 | 0.633 | 0.778 | 0.557 | 0.650 |
| german | XGBoost | Шум | plus_50 | 0.663 | 0.800 | 0.589 | 0.633 |
| german | XGBoost | GMM | plus_25 | 0.661 | 0.812 | 0.614 | 0.783 |
| german | XGBoost | CTGAN | plus_50 | 0.634 | 0.786 | 0.590 | 0.817 |
| giveme | CatBoost | Без баланса | none | 0.360 | 0.832 | 0.452 | 0.486 |
| giveme | CatBoost | Веса классов | none | 0.337 | 0.823 | 0.427 | 0.477 |
| giveme | CatBoost | SMOTE | plus_25 | 0.350 | 0.831 | 0.462 | 0.514 |
| giveme | CatBoost | Borderline-SMOTE | plus_100 | 0.343 | 0.828 | 0.446 | 0.561 |
| giveme | CatBoost | ADASYN | plus_25 | 0.358 | 0.828 | 0.457 | 0.589 |
| giveme | CatBoost | Шум | plus_50 | 0.373 | 0.829 | 0.435 | 0.514 |
| giveme | CatBoost | GMM | plus_100 | 0.367 | 0.828 | 0.419 | 0.533 |
| giveme | CatBoost | CTGAN | plus_25 | 0.349 | 0.829 | 0.448 | 0.505 |
| giveme | LogReg | Без баланса | none | 0.166 | 0.614 | 0.252 | 0.308 |
| giveme | LogReg | Веса классов | none | 0.274 | 0.766 | 0.347 | 0.327 |
| giveme | LogReg | SMOTE | balance | 0.256 | 0.750 | 0.326 | 0.280 |
| giveme | LogReg | Borderline-SMOTE | balance | 0.241 | 0.736 | 0.323 | 0.299 |
| giveme | LogReg | ADASYN | balance | 0.253 | 0.748 | 0.337 | 0.308 |
| giveme | LogReg | Шум | balance | 0.243 | 0.738 | 0.340 | 0.318 |
| giveme | LogReg | GMM | balance | 0.262 | 0.750 | 0.353 | 0.383 |
| giveme | LogReg | CTGAN | plus_25 | 0.125 | 0.579 | 0.128 | 0.178 |
| giveme | XGBoost | Без баланса | none | 0.352 | 0.823 | 0.433 | 0.495 |
| giveme | XGBoost | Веса классов | none | 0.339 | 0.827 | 0.438 | 0.561 |
| giveme | XGBoost | SMOTE | plus_100 | 0.340 | 0.827 | 0.438 | 0.579 |
| giveme | XGBoost | Borderline-SMOTE | plus_100 | 0.339 | 0.819 | 0.421 | 0.458 |
| giveme | XGBoost | ADASYN | plus_100 | 0.350 | 0.824 | 0.436 | 0.589 |
| giveme | XGBoost | Шум | balance | 0.367 | 0.821 | 0.447 | 0.495 |
| giveme | XGBoost | GMM | plus_50 | 0.353 | 0.823 | 0.422 | 0.430 |
| giveme | XGBoost | CTGAN | plus_100 | 0.349 | 0.825 | 0.444 | 0.551 |
