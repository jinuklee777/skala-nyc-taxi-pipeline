"""Analysis-driven data cleaning and feature engineering."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .config import MONTH_END, MONTH_START
from .utils import write_json


# 주요 기능
# 1. Pandas 기술통계와 Polars EDA, 원본 통계검정 결과를 입력으로 받는다.
# 2. 관찰값에 따라 결측 대체·이상치 필터 적용 여부와 상한값을 분기한다.
# 3. 선택된 전처리 방법과 실제 근거를 preprocessing_decisions.json에 기록한다.
# 4. 동일한 결정으로 정제와 Feature Engineering을 수행해 Parquet로 저장한다.
class DataPreprocessor:
    """Build and execute a preprocessing plan from observed analysis results."""

    def __init__(
        self,
        raw_path: Path,
        processed_path: Path,
        artifacts_dir: Path | None = None,
    ) -> None:
        self.raw_path = raw_path
        self.processed_path = processed_path
        self.artifacts_dir = artifacts_dir

    @staticmethod
    def add_base_features(lazy_frame: pl.LazyFrame) -> pl.LazyFrame:
        """Add duration and speed needed by both EDA and cleaning."""
        return lazy_frame.with_columns(
            (
                (
                    pl.col("tpep_dropoff_datetime")
                    - pl.col("tpep_pickup_datetime")
                ).dt.total_seconds()
                / 60.0
            ).alias("trip_duration_minutes")
        ).with_columns(
            pl.when(pl.col("trip_duration_minutes") > 0)
            .then(
                pl.col("trip_distance")
                / (pl.col("trip_duration_minutes") / 60.0)
            )
            .otherwise(None)
            .alias("average_speed_mph")
        )

    @staticmethod
    def _default_plan() -> dict[str, Any]:
        """Keep tests/backward usage stable when analysis results are not supplied."""
        return {
            "rules": {
                "filter_target_month": True,
                "filter_invalid_amounts": True,
                "fare_cap": 500.0,
                "total_cap": 500.0,
                "distance_cap": 100.0,
                "duration_cap_minutes": 240.0,
                "speed_cap_mph": 80.0,
                "passenger_min": 1,
                "passenger_max": 6,
                "filter_location_ids": True,
            },
            "imputation": {
                "passenger_count": {"strategy": "median", "value": 1},
                "RatecodeID": {"strategy": "unknown", "value": 99},
                "store_and_fwd_flag": {
                    "strategy": "unknown",
                    "value": "Unknown",
                },
                "congestion_surcharge": {"strategy": "zero", "value": 0.0},
                "Airport_fee": {"strategy": "zero", "value": 0.0},
            },
            "statistical_evidence": {
                "distance_fare_pearson": None,
                "weekend_ttest_p_value": None,
                "weekend_difference": None,
            },
            "decisions": [],
        }

    def build_preprocessing_plan(
        self,
        pandas_analysis: dict[str, Any],
        eda_profile: dict[str, Any],
        raw_statistical_tests: list[dict[str, Any]],
    ) -> dict[str, Any]:
        """Translate observed Pandas/EDA/statistical results into executable rules."""
        quality = eda_profile["quality_counts"]
        numeric = eda_profile["numeric_summary"]
        missing = pandas_analysis["missing"]
        pandas_numeric = pandas_analysis["numeric_summary"]
        missing_pattern = pandas_analysis["missing_pattern"]
        tests = {result["test"]: result for result in raw_statistical_tests}

        pearson = tests["Pearson correlation"]
        spearman = tests["Spearman correlation"]
        weekend_test = tests["Welch t-test"]
        passenger_median = int(round(pandas_numeric["passenger_count"]["50%"]))
        pandas_distance_fare_corr = float(
            pandas_analysis["correlations"]["fare_amount"]["trip_distance"]
        )
        structured_missing = bool(
            missing_pattern["all_five_missing_are_payment_zero"]
            and missing_pattern["all_payment_zero_have_five_missing"]
        )
        structured_missing_label = "예" if structured_missing else "아니오"
        congestion_imputation = (
            0.0
            if structured_missing
            else float(pandas_numeric["congestion_surcharge"]["50%"])
        )
        airport_fee_imputation = (
            0.0
            if structured_missing
            else float(pandas_numeric["Airport_fee"]["50%"])
        )

        # p99보다 최대값이 10배 이상 큰 경우에만 보수적인 도메인 상한을 활성화한다.
        fare_extreme = numeric["fare_amount"]["max"] > numeric["fare_amount"]["p99"] * 10
        total_extreme = numeric["total_amount"]["max"] > numeric["total_amount"]["p99"] * 10
        distance_extreme = numeric["trip_distance"]["max"] > numeric["trip_distance"]["p99"] * 10
        duration_extreme = (
            numeric["trip_duration_minutes"]["max"]
            > numeric["trip_duration_minutes"]["p99"] * 10
        )

        high_missing_rate = 5.0
        ratecode_unknown = missing["RatecodeID"]["rate_pct"] >= high_missing_rate
        flag_unknown = (
            missing["store_and_fwd_flag"]["rate_pct"] >= high_missing_rate
        )

        fare_action = (
            "fare/total $500 상한 및 음수 제거"
            if fare_extreme or total_extreme
            else "관찰 최댓값 유지 및 음수 제거"
        )
        distance_action = (
            "거리 100mile·시간 240분·속도 80mph 상한"
            if distance_extreme or duration_extreme
            else "관찰 최댓값 유지, 비양수·비현실적 속도만 제거"
        )
        categorical_action = (
            "명시적 Unknown 범주 대체"
            if ratecode_unknown or flag_unknown
            else "관찰 최빈 범주 대체"
        )
        categorical_reason = (
            "5%를 넘고 payment_type=0과 일치하는 구조적 결측을 최빈값으로 숨기지 않는다."
            if (ratecode_unknown or flag_unknown) and structured_missing
            else "5%를 넘는 결측을 최빈값으로 숨기지 않는다."
            if ratecode_unknown or flag_unknown
            else "결측률이 낮아 대표 범주로 대체해 정보 손실을 줄인다."
        )
        fee_action = (
            "가산 요금 결측을 0으로 대체"
            if structured_missing
            else "가산 요금 결측을 Pandas 중앙값으로 대체"
        )
        fee_reason = (
            "payment_type=0에서 다섯 컬럼이 함께 기록되지 않은 구조이므로 행을 "
            "삭제하지 않고, 금액 가산 여부에는 중립적인 0을 사용한다."
            if structured_missing
            else "결측 패턴이 특정 범주와 일치하지 않아 이상치에 강건한 중앙값을 사용한다."
        )
        correlation_distorted = (
            abs(spearman["statistic"]) - abs(pearson["statistic"]) >= 0.2
        )

        decisions = [
            {
                "stage": "결측치",
                "observation": (
                    f"passenger_count 결측 {missing['passenger_count']['rate_pct']:.4f}%, "
                    f"Pandas 중앙값 {passenger_median}"
                ),
                "action": f"중앙값 {passenger_median} 대체",
                "reason": "삭제 시 표본 손실이 크고 중앙값이 이상치에 강건하다.",
            },
            {
                "stage": "결측치",
                "observation": (
                    f"RatecodeID/store_and_fwd_flag 결측률 "
                    f"{missing['RatecodeID']['rate_pct']:.4f}%; 5개 컬럼 동시 결측 "
                    f"{missing_pattern['all_five_missing_count']:,}건, "
                    f"payment_type=0과 완전 일치={structured_missing_label}"
                ),
                "action": categorical_action,
                "reason": categorical_reason,
            },
            {
                "stage": "결측치",
                "observation": (
                    f"congestion_surcharge/Airport_fee 결측률 "
                    f"{missing['congestion_surcharge']['rate_pct']:.4f}%; "
                    "위 구조적 결측 패턴과 동일"
                ),
                "action": fee_action,
                "reason": fee_reason,
            },
            {
                "stage": "금액 이상치",
                "observation": (
                    f"fare p99=${numeric['fare_amount']['p99']:.2f}, "
                    f"max=${numeric['fare_amount']['max']:.2f}; "
                    f"total p99=${numeric['total_amount']['p99']:.2f}, "
                    f"max=${numeric['total_amount']['max']:.2f}"
                ),
                "action": fare_action,
                "reason": (
                    "최대값이 p99의 10배를 넘어 입력 오류로 판단했다."
                    if fare_extreme or total_extreme
                    else "p99 대비 비정상적으로 큰 값이 없어 관찰 범위를 유지한다."
                ),
            },
            {
                "stage": "거리·시간 이상치",
                "observation": (
                    f"거리 p99={numeric['trip_distance']['p99']:.2f}, "
                    f"max={numeric['trip_distance']['max']:.2f}; "
                    f"시간 p99={numeric['trip_duration_minutes']['p99']:.2f}, "
                    f"max={numeric['trip_duration_minutes']['max']:.2f}"
                ),
                "action": distance_action,
                "reason": (
                    "최댓값이 p99에서 크게 벗어났으며, IQR 단독 적용으로 정상 공항 운행을 "
                    "제거하지 않도록 보수적인 택시 도메인 상한을 사용했다."
                    if distance_extreme or duration_extreme
                    else "분포의 정상 범위를 유지하고 명백한 도메인 위반만 제거한다."
                ),
            },
            {
                "stage": "Feature Engineering",
                "observation": (
                    f"Pandas 거리-운임 r={pandas_distance_fare_corr:.4f}, "
                    f"원본 Pearson={pearson['statistic']:.4f}, "
                    f"Spearman={spearman['statistic']:.4f}, "
                    f"주말 t-test p={weekend_test['p_value']:.3e}"
                ),
                "action": "운행시간·속도·시간대·주말 여부 Feature 생성",
                "reason": (
                    "원본 극단값이 Pearson 상관을 왜곡해도 순위 상관이 관계를 지지하며, "
                    "유의한 주중·주말 차이를 모델과 시각화에 반영한다."
                    if correlation_distorted and weekend_test["p_value"] < 0.05
                    else "관찰된 거리/운임 관계와 시간 집단 차이를 모델과 시각화에 반영한다."
                ),
            },
        ]

        plan = {
            "rules": {
                "filter_target_month": quality["pickup_outside_may"] > 0,
                "filter_invalid_amounts": any(
                    quality[name] > 0
                    for name in ("negative_fare", "negative_total", "negative_tip")
                ),
                "fare_cap": 500.0 if fare_extreme else float(numeric["fare_amount"]["max"]),
                "total_cap": 500.0 if total_extreme else float(numeric["total_amount"]["max"]),
                "distance_cap": 100.0 if distance_extreme else float(numeric["trip_distance"]["max"]),
                "duration_cap_minutes": 240.0 if duration_extreme else float(numeric["trip_duration_minutes"]["max"]),
                "speed_cap_mph": 80.0 if quality["speed_gt_80_mph"] > 0 else float(numeric["average_speed_mph"]["max"]),
                "passenger_min": 1,
                "passenger_max": 6 if quality["passenger_gt_6"] > 0 else int(numeric["passenger_count"]["max"]),
                "filter_location_ids": quality["invalid_location_id"] > 0,
            },
            "imputation": {
                "passenger_count": {
                    "strategy": "median",
                    "value": passenger_median,
                },
                "RatecodeID": {
                    "strategy": "unknown" if ratecode_unknown else "mode",
                    "value": 99 if ratecode_unknown else 1,
                },
                "store_and_fwd_flag": {
                    "strategy": "unknown" if flag_unknown else "mode",
                    "value": "Unknown" if flag_unknown else "N",
                },
                "congestion_surcharge": {
                    "strategy": "zero" if structured_missing else "median",
                    "value": congestion_imputation,
                },
                "Airport_fee": {
                    "strategy": "zero" if structured_missing else "median",
                    "value": airport_fee_imputation,
                },
            },
            "statistical_evidence": {
                "distance_fare_pearson": float(pearson["statistic"]),
                "distance_fare_p_value": float(pearson["p_value"]),
                "distance_fare_spearman": float(spearman["statistic"]),
                "pandas_distance_fare_correlation": pandas_distance_fare_corr,
                "correlation_outlier_distortion": correlation_distorted,
                "weekend_ttest_p_value": float(weekend_test["p_value"]),
                "weekend_difference": float(weekend_test["effect_or_difference"]),
                "weekend_difference_significant": weekend_test["p_value"] < 0.05,
            },
            "decisions": decisions,
        }
        if self.artifacts_dir is not None:
            write_json(
                self.artifacts_dir / "preprocessing_decisions.json", plan
            )
        return plan

    def apply_quality_filters(
        self,
        lazy_frame: pl.LazyFrame,
        plan: dict[str, Any] | None = None,
    ) -> pl.LazyFrame:
        """Apply only the branches activated by the preprocessing plan."""
        plan = plan or self._default_plan()
        rules = plan["rules"]
        frame = self.add_base_features(lazy_frame)

        if rules["filter_target_month"]:
            frame = frame.filter(
                pl.col("tpep_pickup_datetime").is_between(
                    MONTH_START, MONTH_END, closed="left"
                )
            )

        amount_expression = (
            (pl.col("trip_distance") > 0)
            & pl.col("fare_amount").is_between(0, rules["fare_cap"])
            & pl.col("total_amount").is_between(0, rules["total_cap"])
            & (pl.col("tip_amount") >= 0)
        )
        frame = frame.filter(amount_expression)
        frame = frame.filter(
            pl.col("trip_duration_minutes").is_between(
                0, rules["duration_cap_minutes"], closed="right"
            )
        )
        frame = frame.filter(
            (pl.col("trip_distance") <= rules["distance_cap"])
            & (pl.col("average_speed_mph") <= rules["speed_cap_mph"])
        )
        frame = frame.filter(
            pl.col("passenger_count").is_null()
            | pl.col("passenger_count").is_between(
                rules["passenger_min"], rules["passenger_max"]
            )
        )
        if rules["filter_location_ids"]:
            frame = frame.filter(
                pl.col("PULocationID").is_between(1, 265)
                & pl.col("DOLocationID").is_between(1, 265)
            )
        return frame

    def engineer_features(
        self,
        lazy_frame: pl.LazyFrame,
        plan: dict[str, Any] | None = None,
    ) -> pl.LazyFrame:
        """Use the selected imputation values and create supported features."""
        plan = plan or self._default_plan()
        imputation = plan["imputation"]
        frame = lazy_frame.with_columns(
            pl.col("passenger_count").fill_null(
                imputation["passenger_count"]["value"]
            ),
            pl.col("RatecodeID").fill_null(imputation["RatecodeID"]["value"]),
            pl.col("store_and_fwd_flag").fill_null(
                imputation["store_and_fwd_flag"]["value"]
            ),
            pl.col("congestion_surcharge").fill_null(
                imputation["congestion_surcharge"]["value"]
            ),
            pl.col("Airport_fee").fill_null(
                imputation["Airport_fee"]["value"]
            ),
        )

        frame = frame.with_columns(
            pl.col("tpep_pickup_datetime").dt.hour().alias("pickup_hour"),
            (pl.col("tpep_pickup_datetime").dt.weekday() - 1).alias(
                "pickup_day_of_week"
            ),
            pl.col("tpep_pickup_datetime").dt.date().alias("pickup_date"),
        ).with_columns(
            (pl.col("pickup_day_of_week") >= 5)
            .cast(pl.Int8)
            .alias("is_weekend"),
            pl.when(pl.col("pickup_hour").is_between(5, 10))
            .then(pl.lit("morning"))
            .when(pl.col("pickup_hour").is_between(11, 16))
            .then(pl.lit("afternoon"))
            .when(pl.col("pickup_hour").is_between(17, 21))
            .then(pl.lit("evening"))
            .otherwise(pl.lit("night"))
            .alias("time_of_day"),
        )

        return frame.with_columns(
            pl.when(pl.col("fare_amount") > 0)
            .then(pl.col("tip_amount") / pl.col("fare_amount") * 100)
            .otherwise(0.0)
            .alias("tip_percentage"),
            (pl.col("total_amount") / pl.col("trip_distance")).alias(
                "cost_per_mile"
            ),
            ((pl.col("RatecodeID") == 2) | (pl.col("Airport_fee") > 0))
            .cast(pl.Int8)
            .alias("is_airport_trip"),
            pl.when(pl.col("trip_distance") < 2)
            .then(pl.lit("short"))
            .when(pl.col("trip_distance") < 5)
            .then(pl.lit("medium"))
            .when(pl.col("trip_distance") < 15)
            .then(pl.lit("long"))
            .otherwise(pl.lit("very_long"))
            .alias("distance_band"),
        )

    def clean_and_save(
        self,
        pandas_analysis: dict[str, Any] | None = None,
        eda_profile: dict[str, Any] | None = None,
        raw_statistical_tests: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Build the analysis-driven plan, execute it, and save the result."""
        if (
            pandas_analysis is not None
            and eda_profile is not None
            and raw_statistical_tests is not None
        ):
            plan = self.build_preprocessing_plan(
                pandas_analysis, eda_profile, raw_statistical_tests
            )
        else:
            plan = self._default_plan()

        raw = pl.scan_parquet(self.raw_path)
        before_rows = raw.select(pl.len()).collect().item()
        unique = raw.unique(maintain_order=True)
        filtered = self.apply_quality_filters(unique, plan)
        engineered = self.engineer_features(filtered, plan)
        engineered.sink_parquet(self.processed_path, compression="zstd")

        clean = pl.scan_parquet(self.processed_path)
        after_rows = clean.select(pl.len()).collect().item()
        schema = clean.collect_schema()
        remaining_nulls = clean.select(
            [pl.col(column).null_count().alias(column) for column in schema.names()]
        ).collect().to_dicts()[0]
        remaining_nulls = {
            column: int(count)
            for column, count in remaining_nulls.items()
            if count
        }
        return {
            "before_rows": int(before_rows),
            "after_rows": int(after_rows),
            "removed_rows": int(before_rows - after_rows),
            "removed_rate_pct": float(
                (before_rows - after_rows) / before_rows * 100
            ),
            "output_columns": len(schema),
            "remaining_nulls": remaining_nulls,
            "processed_size_mb": self.processed_path.stat().st_size / (1024**2),
            "preprocessing_plan": plan,
        }
