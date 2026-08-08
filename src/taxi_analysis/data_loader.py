"""Dataset acquisition and Pandas/Polars loading benchmark."""

from __future__ import annotations

import gc
import io
import time
import urllib.request
from pathlib import Path
from typing import Any

import pandas as pd
import polars as pl

from .utils import write_json


# 주요 기능
# 1. NYC Taxi 원본 Parquet 파일이 없으면 공식 URL에서 다운로드한다.
# 2. 같은 파일을 Pandas와 Polars Lazy API로 각각 불러온다.
# 3. 로딩 시간, 메모리 사용량, GroupBy 처리 시간을 비교해 CSV로 저장한다.
class DataLoader:
    """Download the source data and compare Pandas with Polars Lazy API."""

    def __init__(self, data_url: str, data_path: Path, artifacts_dir: Path) -> None:
        self.data_url = data_url
        self.data_path = data_path
        self.artifacts_dir = artifacts_dir

    def download(self, force: bool = False) -> None:
        """Download the canonical Parquet only when absent or forced."""
        # 이미 정상 파일이 있으면 재다운로드하지 않아 반복 실행 시간을 줄인다.
        if self.data_path.exists() and self.data_path.stat().st_size > 0 and not force:
            print(
                f"[data] 기존 원본 사용: {self.data_path} "
                f"({self.data_path.stat().st_size:,} bytes)"
            )
            return

        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.data_path.with_suffix(".download")
        print(f"[data] 다운로드: {self.data_url}")
        urllib.request.urlretrieve(self.data_url, temporary)
        temporary.replace(self.data_path)
        print(
            f"[data] 저장 완료: {self.data_path} "
            f"({self.data_path.stat().st_size:,} bytes)"
        )

    def benchmark(self) -> dict[str, Any]:
        """Load and aggregate the same file with both data engines."""
        # 두 엔진의 결과를 같은 구조로 모아 마지막에 비교표를 만든다.
        rows: list[dict[str, Any]] = []

        # Pandas: 전체 데이터 로드 시간과 DataFrame의 deep memory를 측정한다.
        started = time.perf_counter()
        pandas_df = pd.read_parquet(self.data_path)
        pandas_load_seconds = time.perf_counter() - started
        pandas_shape = [int(pandas_df.shape[0]), int(pandas_df.shape[1])]
        pandas_memory_mb = pandas_df.memory_usage(index=True, deep=True).sum() / (
            1024**2
        )

        info_buffer = io.StringIO()
        pandas_df.info(buf=info_buffer, memory_usage="deep", show_counts=True)
        missing_count = pandas_df.isna().sum()
        missing_rate = (missing_count / len(pandas_df) * 100).round(4)
        duplicate_count = int(pandas_df.duplicated().sum())

        summary_parts = [
            "=== PANDAS DATA INSPECTION ===",
            f"shape: {pandas_df.shape}",
            f"columns: {pandas_df.columns.tolist()}",
            "\n--- info() ---",
            info_buffer.getvalue(),
            "\n--- describe(include='all') ---",
            pandas_df.describe(include="all").to_string(),
            "\n--- dtypes ---",
            pandas_df.dtypes.to_string(),
            "\n--- missing count ---",
            missing_count.to_string(),
            "\n--- missing rate (%) ---",
            missing_rate.to_string(),
            f"\n--- duplicate rows ---\n{duplicate_count:,}",
            f"\n--- deep memory usage ---\n{pandas_memory_mb:,.2f} MB",
        ]
        pandas_summary = "\n".join(summary_parts)
        print(pandas_summary)
        (self.artifacts_dir / "pandas_data_inspection.txt").write_text(
            pandas_summary, encoding="utf-8"
        )

        # DataPreprocessor가 실제 Pandas 관찰값을 기준으로 분기할 수 있도록
        # 결측률·중앙값·사분위수·상관계수를 구조화해 전달한다.
        analysis_columns = [
            "passenger_count",
            "trip_distance",
            "fare_amount",
            "tip_amount",
            "total_amount",
            "congestion_surcharge",
            "Airport_fee",
        ]
        pandas_numeric_summary = (
            pandas_df[analysis_columns]
            .describe(percentiles=[0.25, 0.5, 0.75, 0.99])
            .T[["mean", "std", "25%", "50%", "75%", "99%", "max"]]
            .round(6)
        )
        pandas_numeric_summary.to_csv(
            self.artifacts_dir / "pandas_numeric_summary.csv"
        )
        pandas_correlation = (
            pandas_df[["trip_distance", "fare_amount", "total_amount"]]
            .corr()
            .round(6)
        )
        pandas_correlation.to_csv(
            self.artifacts_dir / "pandas_correlation_matrix.csv"
        )
        structured_missing_columns = [
            "passenger_count",
            "RatecodeID",
            "store_and_fwd_flag",
            "congestion_surcharge",
            "Airport_fee",
        ]
        all_five_missing = pandas_df[structured_missing_columns].isna().all(axis=1)
        payment_type_zero = pandas_df["payment_type"].eq(0)
        pandas_analysis = {
            "row_count": int(len(pandas_df)),
            "column_count": int(pandas_df.shape[1]),
            "duplicate_count": duplicate_count,
            "missing": {
                column: {
                    "count": int(missing_count[column]),
                    "rate_pct": float(missing_rate[column]),
                }
                for column in pandas_df.columns
            },
            "numeric_summary": pandas_numeric_summary.to_dict(orient="index"),
            "correlations": pandas_correlation.to_dict(),
            "missing_pattern": {
                "all_five_missing_count": int(all_five_missing.sum()),
                "payment_type_zero_count": int(payment_type_zero.sum()),
                "all_five_missing_and_payment_zero_count": int(
                    (all_five_missing & payment_type_zero).sum()
                ),
                "all_five_missing_are_payment_zero": bool(
                    (all_five_missing & ~payment_type_zero).sum() == 0
                ),
                "all_payment_zero_have_five_missing": bool(
                    (payment_type_zero & ~all_five_missing).sum() == 0
                ),
            },
        }
        write_json(self.artifacts_dir / "pandas_analysis.json", pandas_analysis)

        # EDA에서 완전 중복이 0건으로 관찰됐지만, 요구사항을 재현 가능하게 보장하기 위해
        # Pandas의 drop_duplicates()를 명시적으로 적용한다.
        pandas_df = pandas_df.drop_duplicates()

        started = time.perf_counter()
        pandas_grouped = pandas_df.groupby("payment_type", dropna=False).agg(
            trip_count=("total_amount", "size"),
            mean_total_amount=("total_amount", "mean"),
        )
        pandas_process_seconds = time.perf_counter() - started
        pandas_grouped.to_csv(self.artifacts_dir / "pandas_groupby_result.csv")

        rows.append(
            {
                "engine": "Pandas",
                "lazy_scan_setup_seconds": None,
                "load_seconds": pandas_load_seconds,
                "dataframe_memory_mb": pandas_memory_mb,
                "groupby_seconds": pandas_process_seconds,
            }
        )
        del pandas_df, pandas_grouped
        gc.collect()

        # Polars: Lazy API의 scan 생성 시간과 실제 collect 시간을 구분한다.
        started = time.perf_counter()
        polars_lazy = pl.scan_parquet(self.data_path)
        lazy_setup_seconds = time.perf_counter() - started

        started = time.perf_counter()
        polars_df = polars_lazy.collect(engine="streaming")
        polars_load_seconds = time.perf_counter() - started
        polars_memory_mb = polars_df.estimated_size("mb")

        started = time.perf_counter()
        polars_grouped = polars_df.group_by("payment_type").agg(
            pl.len().alias("trip_count"),
            pl.col("total_amount").mean().alias("mean_total_amount"),
        )
        polars_process_seconds = time.perf_counter() - started
        polars_grouped.write_csv(self.artifacts_dir / "polars_groupby_result.csv")

        polars_schema = {name: str(dtype) for name, dtype in polars_df.schema.items()}
        write_json(self.artifacts_dir / "polars_schema.json", polars_schema)
        rows.append(
            {
                "engine": "Polars Lazy",
                "lazy_scan_setup_seconds": lazy_setup_seconds,
                "load_seconds": polars_load_seconds,
                "dataframe_memory_mb": polars_memory_mb,
                "groupby_seconds": polars_process_seconds,
            }
        )
        del polars_df, polars_grouped, polars_lazy
        gc.collect()

        benchmark_df = pd.DataFrame(rows)
        benchmark_df.to_csv(self.artifacts_dir / "engine_benchmark.csv", index=False)
        print("\n=== PANDAS vs POLARS BENCHMARK ===")
        print(benchmark_df.to_string(index=False))

        pandas_row = rows[0]
        polars_row = rows[1]
        return {
            "rows": rows,
            "pandas_shape": pandas_shape,
            "pandas_analysis": pandas_analysis,
            "duplicate_count": duplicate_count,
            "load_speedup_polars_vs_pandas": (
                pandas_row["load_seconds"] / polars_row["load_seconds"]
            ),
            "groupby_speedup_polars_vs_pandas": (
                pandas_row["groupby_seconds"] / polars_row["groupby_seconds"]
            ),
            "memory_ratio_pandas_to_polars": (
                pandas_row["dataframe_memory_mb"]
                / polars_row["dataframe_memory_mb"]
            ),
        }
