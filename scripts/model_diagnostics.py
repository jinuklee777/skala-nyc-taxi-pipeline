"""저장된 모델을 읽어 해석상의 함정 세 가지를 점검한다.

파이프라인과 모델을 변경하지 않고 읽기만 한다.

1. 그룹 순열 중요도
   is_weekend는 pickup_day_of_week에서, time_of_day는 pickup_hour에서,
   average_speed_mph는 거리·시간에서 유도된다. 이런 중복 변수를 개별로 섞으면
   모델이 남은 쌍둥이 변수에서 같은 정보를 복원하므로 중요도가 0에 가깝게 나온다.
   중복 변수를 묶어서 함께 섞어야 그 정보의 실제 기여도를 알 수 있다.

2. 거리 구간을 통제한 주중·주말 비교
   마일당 요금 = 총액÷거리, 속도 = 거리÷시간으로 거리를 공유하므로,
   거리 구성이 다르면 속도와 무관하게 마일당 요금이 달라 보일 수 있다.

3. 구간별 모델 성능
   전체 평균 MAE는 특정 집단의 큰 오차를 가린다.

실행: .venv/bin/python scripts/model_diagnostics.py
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.metrics import mean_absolute_error

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts/model_diagnostics.json"
RANDOM_SEED = 42
SAMPLE_SIZE = 150_000

NUMERIC = [
    "trip_distance", "trip_duration_minutes", "passenger_count",
    "pickup_hour", "pickup_day_of_week", "is_weekend",
    "average_speed_mph", "is_airport_trip",
]
CATEGORICAL = [
    "VendorID", "RatecodeID", "payment_type",
    "store_and_fwd_flag", "time_of_day",
]
FEATURES = NUMERIC + CATEGORICAL

# 서로를 결정론적으로 유도할 수 있는 변수는 한 그룹으로 묶는다.
GROUPS = {
    "거리·시간·속도": ["trip_distance", "trip_duration_minutes", "average_speed_mph"],
    "요일·주말여부": ["pickup_day_of_week", "is_weekend"],
    "시각·시간대구분": ["pickup_hour", "time_of_day"],
    "RatecodeID": ["RatecodeID"],
    "payment_type": ["payment_type"],
    "is_airport_trip": ["is_airport_trip"],
    "VendorID": ["VendorID"],
    "passenger_count": ["passenger_count"],
    "store_and_fwd_flag": ["store_and_fwd_flag"],
}


def build_test_set():
    """ModelTrainer와 동일한 분할로 평가 셋을 재구성한다."""
    data = (
        pl.scan_parquet(ROOT / "data/processed/yellow_taxi_2026-05_cleaned.parquet")
        .select(["tpep_pickup_datetime", *FEATURES, "total_amount", "tip_amount"])
        .collect()
    )
    if data.height > SAMPLE_SIZE:
        data = data.sample(n=SAMPLE_SIZE, seed=RANDOM_SEED)
    frame = data.sort("tpep_pickup_datetime").to_pandas()
    return frame.iloc[int(len(frame) * 0.8):].copy()


def grouped_importance(model, x, y, n_repeats: int = 5) -> tuple[float, list[dict]]:
    """그룹 단위로 함께 섞어 중복 변수가 서로를 가리지 못하게 한다."""
    rng = np.random.default_rng(RANDOM_SEED)
    base = mean_absolute_error(y, model.predict(x))
    rows = []
    for name, columns in GROUPS.items():
        deltas = []
        for _ in range(n_repeats):
            shuffled = x.copy()
            # 그룹 내 변수는 동일한 순열로 섞어 변수 간 관계는 유지한다.
            index = rng.permutation(len(shuffled))
            for column in columns:
                shuffled[column] = x[column].to_numpy()[index]
            deltas.append(mean_absolute_error(y, model.predict(shuffled)) - base)
        rows.append({
            "group": name,
            "columns": columns,
            "mae_increase": float(np.mean(deltas)),
            "std": float(np.std(deltas)),
        })
    rows.sort(key=lambda r: r["mae_increase"], reverse=True)
    return float(base), rows


def weekend_within_distance_band() -> list[dict]:
    """거리 구간을 고정한 뒤에도 주중·주말 차이가 남는지 확인한다."""
    return (
        pl.scan_parquet(ROOT / "data/processed/yellow_taxi_2026-05_cleaned.parquet")
        .group_by(["distance_band", "is_weekend"])
        .agg(
            pl.len().alias("trips"),
            pl.col("trip_distance").mean().alias("mean_distance"),
            pl.col("cost_per_mile").median().alias("median_cost_per_mile"),
            pl.col("average_speed_mph").mean().alias("mean_speed"),
            pl.col("total_amount").mean().alias("mean_total"),
        )
        .sort(["distance_band", "is_weekend"])
        .collect()
        .to_dicts()
    )


def segment_performance(test) -> dict:
    """결제 유형과 금액 구간별로 오차를 나눠 본다."""
    test = test.copy()
    test["pred"] = None  # 자리만 확보, 아래에서 채운다
    return test


def main() -> None:
    bundle = joblib.load(ROOT / "artifacts/total_amount_model.joblib")
    model, meta = bundle["pipeline"], bundle["metadata"]

    test = build_test_set()
    assert len(test) == meta["test_rows"], "평가 셋 재구성이 학습 시점과 다르다"
    x, y = test[FEATURES], test["total_amount"]
    test["pred"] = model.predict(x)
    test["error"] = test["pred"] - test["total_amount"]
    test["abs_error"] = test["error"].abs()

    base_mae, groups = grouped_importance(model, x, y)

    # 결제 유형별
    by_payment = [
        {
            "payment_type": int(pt),
            "trips": int(len(g)),
            "mean_tip": float(g["tip_amount"].mean()),
            "mae": float(g["abs_error"].mean()),
            "bias": float(g["error"].mean()),
        }
        for pt, g in test.groupby("payment_type") if len(g) >= 50
    ]

    # 금액 10분위별
    test["decile"] = test["total_amount"].rank(pct=True).mul(10).clip(upper=9.999).astype(int) + 1
    by_decile = [
        {
            "decile": int(d),
            "trips": int(len(g)),
            "mean_total": float(g["total_amount"].mean()),
            "mae": float(g["abs_error"].mean()),
            "bias": float(g["error"].mean()),
        }
        for d, g in test.groupby("decile")
    ]

    diagnostics = {
        "base_mae": base_mae,
        "grouped_importance": groups,
        "weekend_within_distance_band": weekend_within_distance_band(),
        "by_payment_type": by_payment,
        "by_amount_decile": by_decile,
    }
    OUT.write_text(json.dumps(diagnostics, ensure_ascii=False, indent=2, default=str),
                   encoding="utf-8")

    print(f"기준 MAE: ${base_mae:.3f}\n")
    print("=== 1. 그룹 순열 중요도 (중복 변수를 묶어 측정) ===")
    for row in groups:
        print(f"  {row['group']:<18} +${row['mae_increase']:6.3f}  (±{row['std']:.3f})")

    print("\n=== 2. 거리 구간을 통제한 주중·주말 비교 ===")
    print(f"  {'구간':<11}{'주말':<5}{'건수':>10}{'평균거리':>10}{'마일당':>9}{'속도':>8}")
    for r in diagnostics["weekend_within_distance_band"]:
        print(f"  {r['distance_band']:<11}{int(r['is_weekend']):<5}{r['trips']:>10,}"
              f"{r['mean_distance']:>10.2f}{r['median_cost_per_mile']:>9.2f}{r['mean_speed']:>8.2f}")

    print("\n=== 3-a. 결제 유형별 오차 ===")
    for r in by_payment:
        print(f"  payment_type={r['payment_type']}  n={r['trips']:>6,}  "
              f"MAE ${r['mae']:>5.2f}  편향 ${r['bias']:>+6.2f}")

    print("\n=== 3-b. 금액 10분위별 오차 ===")
    for r in by_decile:
        print(f"  {r['decile']:>2}분위  평균 ${r['mean_total']:>7.2f}  "
              f"MAE ${r['mae']:>5.2f}  편향 ${r['bias']:>+6.2f}")

    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
