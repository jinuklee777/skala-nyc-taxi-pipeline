"""Exploratory data analysis and data-quality profiling."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import polars as pl

from .config import MONTH_END, MONTH_START, RANDOM_SEED
from .data_preprocessor import DataPreprocessor
from .utils import write_json


# 주요 기능
# 1. 전체 행/열, 스키마, 결측치, 중복 데이터를 확인한다.
# 2. 수치형 분포와 범주형 빈도, 날짜 범위, 이상치 건수를 계산한다.
# 3. 변수 간 상관관계를 계산하고 전처리 판단 근거를 JSON/CSV로 저장한다.
class EDAAnalyzer:
    """Inspect distributions, missingness, correlation, and quality issues."""

    def __init__(
        self,
        data_path: Path,
        artifacts_dir: Path,
        sample_size: int,
    ) -> None:
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir
        self.sample_size = sample_size

    def build_profile(self) -> dict[str, Any]:
        """Profile the complete source data and persist machine-readable results."""
        # Polars LazyFrame으로 필요한 연산을 구성해 대용량 원본을 효율적으로 분석한다.
        raw = pl.scan_parquet(self.data_path)
        schema = raw.collect_schema()
        columns = schema.names()
        total_rows = raw.select(pl.len()).collect().item()
        total_columns = len(columns)

        null_row = raw.select(
            [pl.col(column).null_count().alias(column) for column in columns]
        ).collect().to_dicts()[0]
        missing = {
            column: {
                "count": int(count),
                "rate_pct": float(count / total_rows * 100),
            }
            for column, count in null_row.items()
        }

        unique_rows = raw.unique().select(pl.len()).collect().item()
        duplicate_rows = int(total_rows - unique_rows)
        featured = DataPreprocessor.add_base_features(raw)

        # 주요 수치형 변수의 중앙값, 사분위수, p99, 최솟값/최댓값을 계산한다.
        numeric_columns = [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "trip_duration_minutes",
            "average_speed_mph",
        ]
        quantile_expressions: list[pl.Expr] = []
        for column in numeric_columns:
            quantile_expressions.extend(
                [
                    pl.col(column).min().alias(f"{column}__min"),
                    pl.col(column).quantile(0.25).alias(f"{column}__q1"),
                    pl.col(column).median().alias(f"{column}__median"),
                    pl.col(column).quantile(0.75).alias(f"{column}__q3"),
                    pl.col(column).quantile(0.99).alias(f"{column}__p99"),
                    pl.col(column).max().alias(f"{column}__max"),
                ]
            )
        raw_quantiles = (
            featured.select(quantile_expressions).collect().to_dicts()[0]
        )
        numeric_summary: dict[str, dict[str, float | None]] = {}
        for column in numeric_columns:
            values = {
                statistic: raw_quantiles[f"{column}__{statistic}"]
                for statistic in ("min", "q1", "median", "q3", "p99", "max")
            }
            values["iqr"] = (
                values["q3"] - values["q1"]
                if values["q1"] is not None and values["q3"] is not None
                else None
            )
            numeric_summary[column] = values

        # 코드형/범주형 변수별 빈도를 확인해 편중과 Unknown 범주를 파악한다.
        categorical_distributions: dict[str, list[dict[str, Any]]] = {}
        for column in (
            "VendorID",
            "RatecodeID",
            "store_and_fwd_flag",
            "payment_type",
        ):
            categorical_distributions[column] = (
                raw.group_by(column)
                .agg(pl.len().alias("count"))
                .sort("count", descending=True)
                .collect()
                .to_dicts()
            )

        date_range = featured.select(
            pl.col("tpep_pickup_datetime").min().alias("pickup_min"),
            pl.col("tpep_pickup_datetime").max().alias("pickup_max"),
            pl.col("tpep_dropoff_datetime").min().alias("dropoff_min"),
            pl.col("tpep_dropoff_datetime").max().alias("dropoff_max"),
        ).collect().to_dicts()[0]

        # 전처리에서 사용할 품질 규칙별 위반 행 수를 먼저 측정한다.
        quality_rules = {
            "nonpositive_distance": pl.col("trip_distance") <= 0,
            "negative_fare": pl.col("fare_amount") < 0,
            "negative_total": pl.col("total_amount") < 0,
            "negative_tip": pl.col("tip_amount") < 0,
            "fare_gt_500": pl.col("fare_amount") > 500,
            "total_gt_500": pl.col("total_amount") > 500,
            "passenger_lt_1_nonnull": (
                pl.col("passenger_count").is_not_null()
                & (pl.col("passenger_count") < 1)
            ),
            "passenger_gt_6": pl.col("passenger_count") > 6,
            "nonpositive_duration": pl.col("trip_duration_minutes") <= 0,
            "duration_gt_240_minutes": pl.col("trip_duration_minutes") > 240,
            "distance_gt_100_miles": pl.col("trip_distance") > 100,
            "speed_gt_80_mph": pl.col("average_speed_mph") > 80,
            "pickup_outside_may": (
                (pl.col("tpep_pickup_datetime") < MONTH_START)
                | (pl.col("tpep_pickup_datetime") >= MONTH_END)
            ),
            "invalid_location_id": (
                ~pl.col("PULocationID").is_between(1, 265)
                | ~pl.col("DOLocationID").is_between(1, 265)
            ),
        }
        quality_counts = featured.select(
            [
                expression.cast(pl.Int64).sum().alias(name)
                for name, expression in quality_rules.items()
            ]
        ).collect().to_dicts()[0]
        quality_counts = {
            key: int(value) for key, value in quality_counts.items()
        }

        # 전체에서 고정 seed 표본을 뽑아 핵심 연속형 변수의 상관계수를 계산한다.
        correlation_columns = [
            "trip_distance",
            "trip_duration_minutes",
            "fare_amount",
            "tip_amount",
            "total_amount",
        ]
        correlation_data = featured.select(correlation_columns).collect()
        if correlation_data.height > self.sample_size:
            correlation_data = correlation_data.sample(
                n=self.sample_size, seed=RANDOM_SEED
            )
        correlation = correlation_data.to_pandas().corr(numeric_only=True).round(4)
        correlation.to_csv(self.artifacts_dir / "correlation_matrix_raw.csv")

        profile = {
            "row_count": int(total_rows),
            "column_count": int(total_columns),
            "columns": columns,
            "schema": {name: str(dtype) for name, dtype in schema.items()},
            "missing": missing,
            "duplicate_rows": duplicate_rows,
            "date_range": date_range,
            "numeric_summary": numeric_summary,
            "categorical_distributions": categorical_distributions,
            "quality_counts": quality_counts,
            "correlation_matrix": correlation.to_dict(),
        }
        write_json(self.artifacts_dir / "eda_profile.json", profile)
        return profile
