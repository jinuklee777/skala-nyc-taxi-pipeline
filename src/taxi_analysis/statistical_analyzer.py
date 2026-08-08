"""Descriptive and inferential statistical analysis."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

from .config import RANDOM_SEED


# 주요 기능
# 1. 원본 데이터 통계검정을 전처리 전에 수행해 전처리 분기 근거를 만든다.
# 2. 정제 데이터에 같은 검정을 반복해 전처리 후에도 관계가 유지되는지 확인한다.
# 3. 상관계수, Welch t-test, 카이제곱 검정과 p-value 해석을 함께 저장한다.
class StatisticalAnalyzer:
    """Run the same statistical tests before and after preprocessing."""

    def __init__(
        self,
        processed_path: Path,
        artifacts_dir: Path,
        sample_size: int,
    ) -> None:
        self.processed_path = processed_path
        self.artifacts_dir = artifacts_dir
        self.sample_size = sample_size

    def run_raw(self, raw_path: Path) -> list[dict[str, Any]]:
        """Analyze raw rows before DataPreprocessor makes any decision."""
        columns = [
            "tpep_pickup_datetime",
            "trip_distance",
            "fare_amount",
            "total_amount",
            "tip_amount",
            "payment_type",
        ]
        data = pl.scan_parquet(raw_path).select(columns).collect()
        if data.height > self.sample_size:
            data = data.sample(n=self.sample_size, seed=RANDOM_SEED)
        frame = data.to_pandas()
        frame["is_weekend"] = (
            frame["tpep_pickup_datetime"].dt.dayofweek >= 5
        ).astype("int8")
        frame["tip_percentage"] = np.where(
            frame["fare_amount"] > 0,
            frame["tip_amount"] / frame["fare_amount"] * 100,
            0.0,
        )
        tests = self._calculate_tests(frame)
        pd.DataFrame(tests).to_csv(
            self.artifacts_dir / "raw_statistical_tests.csv", index=False
        )
        return tests

    def run(self) -> list[dict[str, Any]]:
        """Analyze the processed dataset using the same statistical tests."""
        columns = [
            "trip_distance",
            "fare_amount",
            "total_amount",
            "tip_percentage",
            "is_weekend",
            "payment_type",
        ]
        data = pl.scan_parquet(self.processed_path).select(columns).collect()
        if data.height > self.sample_size:
            data = data.sample(n=self.sample_size, seed=RANDOM_SEED)
        tests = self._calculate_tests(data.to_pandas())
        pd.DataFrame(tests).to_csv(
            self.artifacts_dir / "statistical_tests.csv", index=False
        )
        return tests

    @staticmethod
    def _calculate_tests(frame: pd.DataFrame) -> list[dict[str, Any]]:
        """Calculate a common test set so raw/processed results are comparable."""
        pearson_r, pearson_p = stats.pearsonr(
            frame["trip_distance"], frame["fare_amount"]
        )
        spearman_r, spearman_p = stats.spearmanr(
            frame["trip_distance"], frame["fare_amount"]
        )

        weekday = frame.loc[frame["is_weekend"] == 0, "total_amount"]
        weekend = frame.loc[frame["is_weekend"] == 1, "total_amount"]
        welch_t, welch_p = stats.ttest_ind(weekday, weekend, equal_var=False)
        weekend_difference = float(weekend.mean() - weekday.mean())

        high_tip = frame["tip_percentage"] >= 20
        contingency = pd.crosstab(frame["payment_type"], high_tip)
        chi2, chi_p, _, _ = stats.chi2_contingency(contingency)
        sample_count = contingency.to_numpy().sum()
        min_dimension = min(contingency.shape) - 1
        cramers_v = (
            math.sqrt(chi2 / (sample_count * min_dimension))
            if min_dimension
            else 0.0
        )

        ttest_interpretation = (
            "유의수준 0.05에서 귀무가설을 기각한다. "
            + (
                "주말 평균 total_amount가 주중보다 낮다."
                if weekend_difference < 0
                else "주말 평균 total_amount가 주중보다 높다."
            )
            if welch_p < 0.05
            else "유의수준 0.05에서 주중·주말 평균 차이가 유의하지 않다."
        )
        chi_interpretation = (
            "결제유형과 20% 이상 팁 여부는 통계적으로 연관되어 있다."
            if chi_p < 0.05
            else "결제유형과 20% 이상 팁 여부의 연관성은 유의하지 않다."
        )

        # 계수 절댓값을 구간으로 나눠 0에 가까운 상관을 "중간"으로 잘못 표현하지 않는다.
        def correlation_interpretation(value: float, relation_type: str) -> str:
            absolute = abs(value)
            if absolute >= 0.7:
                strength = "강한"
            elif absolute >= 0.4:
                strength = "중간"
            elif absolute >= 0.2:
                strength = "약한"
            else:
                strength = "매우 약한"
            direction = "양의" if value >= 0 else "음의"
            return (
                f"거리와 운임은 {strength} {direction} {relation_type} 관계를 보인다."
            )

        return [
            {
                "test": "Pearson correlation",
                "question": "trip_distance와 fare_amount의 선형 관계",
                "statistic": float(pearson_r),
                "p_value": float(pearson_p),
                "effect_or_difference": float(pearson_r),
                "sample_size": int(len(frame)),
                "interpretation": correlation_interpretation(
                    pearson_r, "선형"
                ),
            },
            {
                "test": "Spearman correlation",
                "question": "trip_distance와 fare_amount의 단조 관계",
                "statistic": float(spearman_r),
                "p_value": float(spearman_p),
                "effect_or_difference": float(spearman_r),
                "sample_size": int(len(frame)),
                "interpretation": correlation_interpretation(
                    spearman_r, "단조"
                ),
            },
            {
                "test": "Welch t-test",
                "question": "주중과 주말의 평균 total_amount 차이",
                "statistic": float(welch_t),
                "p_value": float(welch_p),
                "effect_or_difference": weekend_difference,
                "weekday_mean": float(weekday.mean()),
                "weekend_mean": float(weekend.mean()),
                "sample_size": int(len(frame)),
                "interpretation": ttest_interpretation,
            },
            {
                "test": "Chi-square independence",
                "question": "payment_type과 20% 이상 팁 여부의 연관성",
                "statistic": float(chi2),
                "p_value": float(chi_p),
                "effect_or_difference": float(cramers_v),
                "sample_size": int(sample_count),
                "interpretation": chi_interpretation,
            },
        ]
