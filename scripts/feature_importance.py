"""저장된 모델이 어떤 Feature에 근거해 예측하는지 측정한다.

ModelTrainer와 동일한 방식(고정 seed 표본 → 시간순 정렬 → 뒤 20%)으로 평가 셋을
재구성한 뒤 Permutation Importance를 계산해 artifacts/feature_importance.json에 저장한다.

실행: .venv/bin/python scripts/feature_importance.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.inspection import permutation_importance

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "artifacts/total_amount_model.joblib"
PROCESSED = ROOT / "data/processed/yellow_taxi_2026-05_cleaned.parquet"
OUT = ROOT / "artifacts/feature_importance.json"

RANDOM_SEED = 42
SAMPLE_SIZE = 150_000


def main() -> None:
    bundle = joblib.load(MODEL)
    model, meta = bundle["pipeline"], bundle["metadata"]

    numeric = [
        "trip_distance", "trip_duration_minutes", "passenger_count",
        "pickup_hour", "pickup_day_of_week", "is_weekend",
        "average_speed_mph", "is_airport_trip",
    ]
    categorical = [
        "VendorID", "RatecodeID", "payment_type",
        "store_and_fwd_flag", "time_of_day",
    ]
    features = numeric + categorical

    # ModelTrainer.train_and_save()와 동일한 분할을 재현한다.
    data = (
        pl.scan_parquet(PROCESSED)
        .select(["tpep_pickup_datetime", *features, "total_amount"])
        .collect()
    )
    if data.height > SAMPLE_SIZE:
        data = data.sample(n=SAMPLE_SIZE, seed=RANDOM_SEED)
    frame = data.sort("tpep_pickup_datetime").to_pandas()
    test = frame.iloc[int(len(frame) * 0.8):]
    x_test, y_test = test[features], test["total_amount"]

    r2 = float(model.score(x_test, y_test))
    assert len(test) == meta["test_rows"], "평가 셋 재구성이 학습 시점과 다르다"

    # Feature를 섞었을 때 MAE가 얼마나 나빠지는지로 기여도를 측정한다.
    result = permutation_importance(
        model, x_test, y_test,
        n_repeats=5, random_state=RANDOM_SEED,
        scoring="neg_mean_absolute_error", n_jobs=1,
    )
    ranking = sorted(
        (
            {
                "feature": name,
                "mae_increase": float(result.importances_mean[i]),
                "std": float(result.importances_std[i]),
            }
            for i, name in enumerate(features)
        ),
        key=lambda row: row["mae_increase"],
        reverse=True,
    )

    OUT.write_text(
        json.dumps(
            {
                "test_rows": int(len(test)),
                "r2_recomputed": r2,
                "r2_stored": meta["r2"],
                "n_repeats": 5,
                "scoring": "neg_mean_absolute_error",
                "ranking": ranking,
            },
            ensure_ascii=False, indent=2,
        ),
        encoding="utf-8",
    )

    print(f"평가 셋 {len(test):,}행 | R² 재계산 {r2:.4f} (저장값 {meta['r2']:.4f})")
    print("\n=== Permutation Importance (MAE 증가분) ===")
    for row in ranking:
        value = row["mae_increase"]
        shown = f"+${value:6.3f}" if value > 0.001 else "    ~0"
        print(f"  {row['feature']:<24} {shown}")
    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
