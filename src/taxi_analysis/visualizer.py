"""Static Seaborn and interactive Plotly visualization stage."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import plotly.express as px
import polars as pl
import seaborn as sns

from .config import RANDOM_SEED


# 주요 기능
# 1. Seaborn으로 거리 분포, 결제 유형, 시간대 수요, 상관관계를 2×2로 표현한다.
# 2. 원본과 정제 데이터의 결측·행 수·분포를 비교해 전처리 근거 차트를 만든다.
# 3. Plotly로 주중/주말 시간대별 수요를 탐색할 수 있는 HTML 차트를 만든다.
class TaxiVisualizer:
    """Create the required 2×2 dashboard and interactive demand chart."""

    def __init__(
        self,
        processed_path: Path,
        figures_dir: Path,
        sample_size: int,
        raw_path: Path | None = None,
    ) -> None:
        self.processed_path = processed_path
        self.figures_dir = figures_dir
        self.sample_size = sample_size
        self.raw_path = raw_path

    def create(
        self,
        pandas_analysis: dict[str, Any] | None = None,
        cleaning: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        # 그래프에 필요한 컬럼만 읽고, 행 단위 그래프는 고정 seed 표본을 사용한다.
        lazy = pl.scan_parquet(self.processed_path)
        plot_columns = [
            "trip_distance",
            "trip_duration_minutes",
            "fare_amount",
            "tip_amount",
            "total_amount",
        ]
        sample = lazy.select(plot_columns).collect()
        if sample.height > self.sample_size:
            sample = sample.sample(n=self.sample_size, seed=RANDOM_SEED)
        sample_pd = sample.to_pandas()

        hourly = (
            lazy.group_by("pickup_hour")
            .agg(
                pl.len().alias("trip_count"),
                pl.col("total_amount").mean().alias("mean_total_amount"),
            )
            .sort("pickup_hour")
            .collect()
            .to_pandas()
        )
        payment = (
            lazy.group_by("payment_type")
            .agg(pl.len().alias("trip_count"))
            .sort("trip_count", descending=True)
            .collect()
            .to_pandas()
        )

        # 하나의 PNG에서 핵심 EDA 결과 네 가지를 바로 비교할 수 있게 배치한다.
        sns.set_theme(style="whitegrid", context="notebook")
        figure, axes = plt.subplots(2, 2, figsize=(16, 11))

        distance_clip = sample_pd["trip_distance"].quantile(0.99)
        sns.histplot(
            sample_pd.loc[
                sample_pd["trip_distance"] <= distance_clip, "trip_distance"
            ],
            bins=50,
            color="#1f77b4",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Trip Distance Distribution (<= sample p99)")
        axes[0, 0].set_xlabel("Miles")

        sns.barplot(
            data=payment,
            x="payment_type",
            y="trip_count",
            color="#4c78a8",
            ax=axes[0, 1],
        )
        axes[0, 1].set_title("Trips by Payment Type")
        axes[0, 1].set_xlabel("Payment Type Code")
        axes[0, 1].ticklabel_format(style="plain", axis="y")

        sns.lineplot(
            data=hourly,
            x="pickup_hour",
            y="trip_count",
            marker="o",
            color="#f58518",
            ax=axes[1, 0],
        )
        axes[1, 0].set_title("Hourly Trip Demand")
        axes[1, 0].set_xlabel("Pickup Hour")
        axes[1, 0].set_xticks(range(0, 24, 2))
        axes[1, 0].ticklabel_format(style="plain", axis="y")

        correlation = sample_pd.corr(numeric_only=True)
        sns.heatmap(
            correlation,
            annot=True,
            fmt=".2f",
            cmap="vlag",
            center=0,
            square=True,
            ax=axes[1, 1],
        )
        axes[1, 1].set_title("Numeric Feature Correlation")

        figure.suptitle("NYC Yellow Taxi EDA — May 2026", fontsize=18, y=1.01)
        figure.tight_layout()
        static_path = self.figures_dir / "eda_overview_2x2.png"
        figure.savefig(static_path, dpi=180, bbox_inches="tight")
        plt.close(figure)

        # 주중/주말 패턴을 마우스 hover로 비교할 수 있는 독립 실행형 HTML을 생성한다.
        interactive_data = (
            lazy.group_by(["pickup_hour", "is_weekend"])
            .agg(
                pl.len().alias("trip_count"),
                pl.col("total_amount").mean().alias("mean_total_amount"),
            )
            .sort(["is_weekend", "pickup_hour"])
            .collect()
            .to_pandas()
        )
        interactive_data["day_type"] = interactive_data["is_weekend"].map(
            {0: "Weekday", 1: "Weekend"}
        )
        interactive = px.line(
            interactive_data,
            x="pickup_hour",
            y="trip_count",
            color="day_type",
            markers=True,
            hover_data={"mean_total_amount": ":.2f", "is_weekend": False},
            labels={
                "pickup_hour": "Pickup Hour",
                "trip_count": "Trip Count",
                "day_type": "Day Type",
                "mean_total_amount": "Mean Total Amount ($)",
            },
            title="Interactive Hourly Demand: Weekday vs Weekend",
        )
        interactive.update_layout(hovermode="x unified", template="plotly_white")
        interactive_path = self.figures_dir / "hourly_demand_interactive.html"
        interactive.write_html(interactive_path, include_plotlyjs=True)

        outputs = {
            "static_2x2": str(static_path),
            "interactive": str(interactive_path),
        }
        if (
            self.raw_path is not None
            and pandas_analysis is not None
            and cleaning is not None
        ):
            evidence_path = self._create_preprocessing_evidence(
                pandas_analysis, cleaning
            )
            outputs["preprocessing_evidence"] = str(evidence_path)
        return outputs

    def _create_preprocessing_evidence(
        self,
        pandas_analysis: dict[str, Any],
        cleaning: dict[str, Any],
    ) -> Path:
        """Visualize the observations that led to the cleaning decisions."""
        # 분포 비교에 필요한 두 컬럼만 읽고 동일한 seed로 표본화해 비교 조건을 맞춘다.
        compare_columns = ["trip_distance", "total_amount"]
        raw_sample = pl.scan_parquet(self.raw_path).select(compare_columns).collect()
        clean_sample = (
            pl.scan_parquet(self.processed_path)
            .select(compare_columns)
            .collect()
        )
        if raw_sample.height > self.sample_size:
            raw_sample = raw_sample.sample(n=self.sample_size, seed=RANDOM_SEED)
        if clean_sample.height > self.sample_size:
            clean_sample = clean_sample.sample(n=self.sample_size, seed=RANDOM_SEED)
        raw_pd = raw_sample.to_pandas()
        clean_pd = clean_sample.to_pandas()

        figure, axes = plt.subplots(2, 2, figsize=(16, 11))

        # Pandas에서 실제 관찰한 결측률 가운데 0보다 큰 컬럼만 표시한다.
        missing_rates = pd.DataFrame(
            [
                {"column": column, "missing_rate_pct": values["rate_pct"]}
                for column, values in pandas_analysis["missing"].items()
                if values["rate_pct"] > 0
            ]
        ).sort_values("missing_rate_pct", ascending=False)
        sns.barplot(
            data=missing_rates,
            x="missing_rate_pct",
            y="column",
            color="#e45756",
            ax=axes[0, 0],
        )
        axes[0, 0].set_title("Raw Missing Rates Observed by Pandas")
        axes[0, 0].set_xlabel("Missing rate (%)")
        axes[0, 0].set_ylabel("")

        row_counts = pd.DataFrame(
            {
                "stage": ["Raw", "Processed"],
                "rows": [cleaning["before_rows"], cleaning["after_rows"]],
            }
        )
        sns.barplot(
            data=row_counts,
            x="stage",
            y="rows",
            hue="stage",
            palette=["#bab0ac", "#59a14f"],
            legend=False,
            ax=axes[0, 1],
        )
        axes[0, 1].set_title(
            f"Rows Before vs After ({cleaning['removed_rate_pct']:.2f}% removed)"
        )
        axes[0, 1].set_xlabel("")
        axes[0, 1].set_ylabel("Rows")
        axes[0, 1].ticklabel_format(style="plain", axis="y")

        # 긴 꼬리가 본체를 가리지 않도록 원본 p99 안에서 전처리 전후 분포를 겹쳐 본다.
        for axis, column, color_pair, unit in (
            (axes[1, 0], "trip_distance", ("#bab0ac", "#4c78a8"), "Miles"),
            (axes[1, 1], "total_amount", ("#bab0ac", "#f28e2b"), "USD"),
        ):
            raw_values = raw_pd[column].dropna()
            upper = raw_values.quantile(0.99)
            lower = max(raw_values.quantile(0.001), -20.0)
            comparison = pd.concat(
                [
                    pd.DataFrame(
                        {column: raw_values, "dataset": "Raw"}
                    ),
                    pd.DataFrame(
                        {
                            column: clean_pd[column].dropna(),
                            "dataset": "Processed",
                        }
                    ),
                ],
                ignore_index=True,
            )
            comparison = comparison.loc[
                comparison[column].between(lower, upper)
            ]
            sns.histplot(
                data=comparison,
                x=column,
                hue="dataset",
                hue_order=["Raw", "Processed"],
                palette=color_pair,
                bins=50,
                stat="density",
                common_norm=False,
                element="step",
                ax=axis,
            )
            axis.set_title(
                f"{column}: Raw vs Processed (raw p0.1–p99)"
            )
            axis.set_xlabel(unit)

        figure.suptitle(
            "Evidence-Based Preprocessing: Observations and Outcomes",
            fontsize=18,
            y=1.01,
        )
        figure.tight_layout()
        evidence_path = self.figures_dir / "preprocessing_evidence.png"
        figure.savefig(evidence_path, dpi=180, bbox_inches="tight")
        plt.close(figure)
        return evidence_path
