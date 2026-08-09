"""정제 데이터에서 운행 패턴을 분석한다.

파이프라인(run_pipeline.py)과 독립적으로 실행되며, 정제 결과를 입력으로 받아
보고서에 인용하는 운행 분석 수치를 artifacts/domain_analysis.json에 저장한다.

실행: .venv/bin/python scripts/domain_analysis.py
"""

from __future__ import annotations

import json
from pathlib import Path

import polars as pl

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data/processed/yellow_taxi_2026-05_cleaned.parquet"
RAW = ROOT / "data/raw/yellow_tripdata_2026-05.parquet"
OUT = ROOT / "artifacts/domain_analysis.json"


def weekend_gap(lazy: pl.LazyFrame) -> dict:
    """주말 총액이 낮은 원인을 거리·단가·구성으로 분해한다."""
    result = (
        lazy.group_by("is_weekend")
        .agg(
            pl.len().alias("trips"),
            pl.col("total_amount").mean().alias("mean_total"),
            pl.col("trip_distance").mean().alias("mean_distance"),
            pl.col("cost_per_mile").median().alias("median_cost_per_mile"),
            # cost_per_mile은 total_amount 기준이라 팁·통행료·할증이 섞여 있다.
            # 주말은 팁 비율이 낮으므로 팁을 뺀 값으로도 격차가 남는지 확인한다.
            ((pl.col("total_amount") - pl.col("tip_amount")) / pl.col("trip_distance"))
            .median()
            .alias("median_cost_per_mile_ex_tip"),
            pl.col("trip_duration_minutes").mean().alias("mean_duration"),
            pl.col("average_speed_mph").mean().alias("mean_speed"),
            pl.col("is_airport_trip").mean().alias("airport_share"),
            pl.col("tip_percentage").mean().alias("mean_tip_pct"),
        )
        .sort("is_weekend")
        .collect()
    )
    rows = {int(r["is_weekend"]): r for r in result.to_dicts()}
    weekday, weekend = rows[0], rows[1]
    return {
        "weekday": weekday,
        "weekend": weekend,
        "total_gap": weekend["mean_total"] - weekday["mean_total"],
        "distance_gap": weekend["mean_distance"] - weekday["mean_distance"],
        "distance_gap_pct": (
            (weekend["mean_distance"] - weekday["mean_distance"])
            / weekday["mean_distance"] * 100
        ),
        "airport_share_gap_pct": (
            (weekend["airport_share"] - weekday["airport_share"]) * 100
        ),
        "cost_per_mile_gap_pct": (
            (weekend["median_cost_per_mile"] - weekday["median_cost_per_mile"])
            / weekday["median_cost_per_mile"] * 100
        ),
        "cost_per_mile_ex_tip_gap_pct": (
            (weekend["median_cost_per_mile_ex_tip"] - weekday["median_cost_per_mile_ex_tip"])
            / weekday["median_cost_per_mile_ex_tip"] * 100
        ),
    }


def distance_band_profile(lazy: pl.LazyFrame) -> list[dict]:
    """거리 구간별 요금 효율을 비교한다."""
    return (
        lazy.group_by("distance_band")
        .agg(
            pl.len().alias("trips"),
            pl.col("trip_distance").median().alias("median_distance"),
            pl.col("total_amount").median().alias("median_total"),
            pl.col("cost_per_mile").median().alias("median_cost_per_mile"),
            pl.col("tip_percentage").mean().alias("mean_tip_pct"),
        )
        .sort("median_distance")
        .collect()
        .to_dicts()
    )


def hourly_profile(lazy: pl.LazyFrame) -> list[dict]:
    """시간대별 수요·속도·단가를 함께 본다."""
    return (
        lazy.group_by("pickup_hour")
        .agg(
            pl.len().alias("trips"),
            pl.col("average_speed_mph").mean().alias("mean_speed"),
            pl.col("cost_per_mile").median().alias("median_cost_per_mile"),
            pl.col("trip_duration_minutes").mean().alias("mean_duration"),
        )
        .sort("pickup_hour")
        .collect()
        .to_dicts()
    )


def standard_ratecode_gap(lazy: pl.LazyFrame) -> dict:
    """요금제 혼입을 통제하고 주말 격차를 다시 본다.

    Flex Fare는 미터를 쓰지 않는 사전 확정 요금이라 요금 산정 방식 자체가 다르다.
    표준 미터 요금(RatecodeID=1)만 남겨도 같은 방향의 격차가 남는지 확인한다.
    """
    result = (
        lazy.filter(pl.col("RatecodeID") == 1)
        .group_by("is_weekend")
        .agg(
            pl.len().alias("trips"),
            pl.col("total_amount").mean().alias("mean_total"),
            pl.col("trip_distance").mean().alias("mean_distance"),
            pl.col("cost_per_mile").median().alias("median_cost_per_mile"),
            pl.col("average_speed_mph").mean().alias("mean_speed"),
        )
        .sort("is_weekend")
        .collect()
    )
    rows = {int(r["is_weekend"]): r for r in result.to_dicts()}
    weekday, weekend = rows[0], rows[1]

    def pct(key: str) -> float:
        return (weekend[key] - weekday[key]) / weekday[key] * 100

    return {
        "weekday": weekday,
        "weekend": weekend,
        "total_gap": weekend["mean_total"] - weekday["mean_total"],
        "distance_gap_pct": pct("mean_distance"),
        "cost_per_mile_gap_pct": pct("median_cost_per_mile"),
        "speed_gap_pct": pct("mean_speed"),
    }


def unrecorded_group(raw: pl.LazyFrame) -> dict:
    """payment_type=0(결제 미기록) 집단이 나머지와 다른지 원본에서 비교한다."""
    result = (
        raw.with_columns((pl.col("payment_type") == 0).alias("unrecorded"))
        .group_by("unrecorded")
        .agg(
            pl.len().alias("trips"),
            pl.col("total_amount").mean().alias("mean_total"),
            pl.col("trip_distance").median().alias("median_distance"),
            pl.col("tip_amount").mean().alias("mean_tip"),
            (pl.col("tip_amount") > 0).mean().alias("tip_rate"),
        )
        .sort("unrecorded")
        .collect()
    )
    rows = {bool(r["unrecorded"]): r for r in result.to_dicts()}
    return {"recorded": rows[False], "unrecorded": rows[True]}


def main() -> None:
    lazy = pl.scan_parquet(PROCESSED)
    raw = pl.scan_parquet(RAW)

    analysis = {
        "weekend_gap": weekend_gap(lazy),
        "standard_ratecode_gap": standard_ratecode_gap(lazy),
        "distance_bands": distance_band_profile(lazy),
        "hourly": hourly_profile(lazy),
        "unrecorded_group": unrecorded_group(raw),
    }
    OUT.write_text(
        json.dumps(analysis, ensure_ascii=False, indent=2, default=str),
        encoding="utf-8",
    )

    g = analysis["weekend_gap"]
    print("=== 주말 격차 분해 ===")
    print(f"  총액   주중 ${g['weekday']['mean_total']:.2f} → 주말 ${g['weekend']['mean_total']:.2f}"
          f"  (차이 ${g['total_gap']:+.2f})")
    print(f"  거리   주중 {g['weekday']['mean_distance']:.2f} → 주말 {g['weekend']['mean_distance']:.2f} mi"
          f"  ({g['distance_gap_pct']:+.1f}%)")
    print(f"  마일당 총액 주중 ${g['weekday']['median_cost_per_mile']:.2f} → 주말 ${g['weekend']['median_cost_per_mile']:.2f}"
          f"  ({g['cost_per_mile_gap_pct']:+.1f}%)")
    print(f"  팁 제외    주중 ${g['weekday']['median_cost_per_mile_ex_tip']:.2f} → 주말 ${g['weekend']['median_cost_per_mile_ex_tip']:.2f}"
          f"  ({g['cost_per_mile_ex_tip_gap_pct']:+.1f}%)")
    print(f"  공항   주중 {g['weekday']['airport_share']*100:.2f}% → 주말 {g['weekend']['airport_share']*100:.2f}%")
    print(f"  속도   주중 {g['weekday']['mean_speed']:.2f} → 주말 {g['weekend']['mean_speed']:.2f} mph")
    print(f"  팁률   주중 {g['weekday']['mean_tip_pct']:.2f}% → 주말 {g['weekend']['mean_tip_pct']:.2f}%")

    s = analysis["standard_ratecode_gap"]
    print("\n=== 표준 미터 요금(RatecodeID=1)만 — 요금제 통제 ===")
    print(f"  건수   주중 {s['weekday']['trips']:,} / 주말 {s['weekend']['trips']:,}")
    print(f"  총액   주중 ${s['weekday']['mean_total']:.2f} → 주말 ${s['weekend']['mean_total']:.2f}"
          f"  (차이 ${s['total_gap']:+.2f})")
    print(f"  거리   주중 {s['weekday']['mean_distance']:.2f} → 주말 {s['weekend']['mean_distance']:.2f} mi"
          f"  ({s['distance_gap_pct']:+.1f}%)")
    print(f"  마일당 주중 ${s['weekday']['median_cost_per_mile']:.2f} → 주말 ${s['weekend']['median_cost_per_mile']:.2f}"
          f"  ({s['cost_per_mile_gap_pct']:+.1f}%)")
    print(f"  속도   주중 {s['weekday']['mean_speed']:.2f} → 주말 {s['weekend']['mean_speed']:.2f} mph"
          f"  ({s['speed_gap_pct']:+.1f}%)")

    print("\n=== 거리 구간별 ===")
    for b in analysis["distance_bands"]:
        print(f"  {b['distance_band']:<10} {b['trips']:>9,}건  "
              f"중앙거리 {b['median_distance']:>5.2f}mi  "
              f"중앙총액 ${b['median_total']:>6.2f}  "
              f"마일당 ${b['median_cost_per_mile']:>6.2f}  "
              f"팁 {b['mean_tip_pct']:.1f}%")

    print("\n=== 시간대별 (속도·단가) ===")
    for h in analysis["hourly"]:
        print(f"  {h['pickup_hour']:>2}시  {h['trips']:>8,}건  "
              f"{h['mean_speed']:>5.2f}mph  마일당 ${h['median_cost_per_mile']:>6.2f}  "
              f"{h['mean_duration']:>5.1f}분")

    u = analysis["unrecorded_group"]
    print("\n=== payment_type=0 집단 ===")
    for label, r in (("기록됨", u["recorded"]), ("미기록", u["unrecorded"])):
        print(f"  {label}  {r['trips']:>9,}건  평균총액 ${r['mean_total']:.2f}  "
              f"중앙거리 {r['median_distance']:.2f}mi  "
              f"평균팁 ${r['mean_tip']:.2f}  팁발생률 {r['tip_rate']*100:.1f}%")

    print(f"\n저장: {OUT}")


if __name__ == "__main__":
    main()
