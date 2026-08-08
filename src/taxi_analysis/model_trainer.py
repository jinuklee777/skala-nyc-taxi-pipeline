"""Scikit-learn preprocessing, model training, evaluation, and persistence."""

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    mean_absolute_error,
    r2_score,
    root_mean_squared_error,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from .config import RANDOM_SEED
from .utils import write_json


# 주요 기능
# 1. 수치형/범주형 전처리와 회귀 모델을 하나의 sklearn Pipeline으로 묶는다.
# 2. 시간 순서 기반 train/test 분할로 total_amount 예측 성능을 평가한다.
# 3. MAE, RMSE, R²와 기준선 성능을 저장하고 학습 모델을 joblib으로 저장한다.
class ModelTrainer:
    """Train a leakage-controlled total-amount regression pipeline."""

    def __init__(
        self,
        processed_path: Path,
        artifacts_dir: Path,
        sample_size: int,
    ) -> None:
        self.processed_path = processed_path
        self.artifacts_dir = artifacts_dir
        self.sample_size = sample_size

    def train_and_save(self) -> dict[str, Any]:
        """Train a chronological holdout model and persist it with joblib."""
        # total_amount 구성요소는 target leakage를 일으키므로 입력 Feature에서 제외한다.
        numeric_features = [
            "trip_distance",
            "trip_duration_minutes",
            "passenger_count",
            "pickup_hour",
            "pickup_day_of_week",
            "is_weekend",
            "average_speed_mph",
            "is_airport_trip",
        ]
        categorical_features = [
            "VendorID",
            "RatecodeID",
            "payment_type",
            "store_and_fwd_flag",
            "time_of_day",
        ]
        target = "total_amount"
        selection = [
            "tpep_pickup_datetime",
            *numeric_features,
            *categorical_features,
            target,
        ]
        data = pl.scan_parquet(self.processed_path).select(selection).collect()
        if data.height > self.sample_size:
            data = data.sample(n=self.sample_size, seed=RANDOM_SEED)
        frame = data.sort("tpep_pickup_datetime").to_pandas()

        # 과거 80%로 학습하고 시간상 뒤쪽 20%로 평가해 실제 미래 예측에 가깝게 만든다.
        split_index = int(len(frame) * 0.8)
        train = frame.iloc[:split_index]
        test = frame.iloc[split_index:]
        x_train = train[numeric_features + categorical_features]
        y_train = train[target]
        x_test = test[numeric_features + categorical_features]
        y_test = test[target]

        # 수치형과 범주형에 서로 다른 결측 처리/인코딩을 적용한다.
        numeric_pipeline = Pipeline(
            [("imputer", SimpleImputer(strategy="median"))]
        )
        categorical_pipeline = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="most_frequent")),
                (
                    "encoder",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                ),
            ]
        )
        preprocessor = ColumnTransformer(
            [
                ("numeric", numeric_pipeline, numeric_features),
                ("categorical", categorical_pipeline, categorical_features),
            ],
            remainder="drop",
            verbose_feature_names_out=False,
        )
        categorical_mask = [False] * len(numeric_features) + [True] * len(
            categorical_features
        )
        estimator = HistGradientBoostingRegressor(
            learning_rate=0.01,
            max_iter=500,
            # max_leaf_nodes=31,
            l2_regularization=1.0,
            categorical_features=categorical_mask,
            random_state=RANDOM_SEED,
        )
        model = Pipeline(
            [("preprocessor", preprocessor), ("regressor", estimator)]
        )

        # 학습 시간과 예측 성능을 측정하고 단순 중앙값 예측보다 나은지 비교한다.
        started = time.perf_counter()
        model.fit(x_train, y_train)
        fit_seconds = time.perf_counter() - started
        predictions = model.predict(x_test)
        baseline_predictions = np.full(len(y_test), y_train.median())

        metrics = {
            "target": target,
            "sample_rows": int(len(frame)),
            "train_rows": int(len(train)),
            "test_rows": int(len(test)),
            "split_strategy": (
                "pickup_datetime 기준 정렬 후 앞 80% 학습 / 뒤 20% 평가"
            ),
            "features": numeric_features + categorical_features,
            "excluded_leakage_features": [
                "fare_amount",
                "tip_amount",
                "tolls_amount",
                "extra",
                "mta_tax",
                "improvement_surcharge",
                "congestion_surcharge",
                "Airport_fee",
                "cbd_congestion_fee",
            ],
            "mae": float(mean_absolute_error(y_test, predictions)),
            "rmse": float(root_mean_squared_error(y_test, predictions)),
            "r2": float(r2_score(y_test, predictions)),
            "baseline_median_mae": float(
                mean_absolute_error(y_test, baseline_predictions)
            ),
            "fit_seconds": float(fit_seconds),
        }
        metrics["mae_improvement_vs_baseline_pct"] = float(
            (metrics["baseline_median_mae"] - metrics["mae"])
            / metrics["baseline_median_mae"]
            * 100
        )

        # 전처리기와 모델을 함께 저장하므로 새 데이터도 같은 방식으로 바로 예측할 수 있다.
        model_path = self.artifacts_dir / "total_amount_model.joblib"
        joblib.dump(
            {
                "pipeline": model,
                "metadata": metrics,
                "created_at": datetime.now().isoformat(timespec="seconds"),
            },
            model_path,
        )
        metrics["model_path"] = str(model_path)
        write_json(self.artifacts_dir / "model_metrics.json", metrics)

        prediction_sample = pd.DataFrame(
            {
                "actual_total_amount": y_test.to_numpy()[:1000],
                "predicted_total_amount": predictions[:1000],
            }
        )
        prediction_sample["absolute_error"] = (
            prediction_sample["actual_total_amount"]
            - prediction_sample["predicted_total_amount"]
        ).abs()
        prediction_sample.to_csv(
            self.artifacts_dir / "prediction_sample.csv", index=False
        )
        return metrics
