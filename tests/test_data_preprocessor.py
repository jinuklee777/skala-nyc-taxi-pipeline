from datetime import datetime
from pathlib import Path

import polars as pl

from taxi_analysis.data_preprocessor import DataPreprocessor


PROCESSOR = DataPreprocessor(Path("unused_raw"), Path("unused_processed"))


def _toy_lazyframe() -> pl.LazyFrame:
    return pl.DataFrame(
        {
            "VendorID": [2, 2, 1],
            "tpep_pickup_datetime": [
                datetime(2026, 5, 5, 8, 0),
                datetime(2026, 5, 5, 9, 0),
                datetime(2026, 4, 30, 23, 0),
            ],
            "tpep_dropoff_datetime": [
                datetime(2026, 5, 5, 8, 20),
                datetime(2026, 5, 5, 9, 20),
                datetime(2026, 4, 30, 23, 10),
            ],
            "passenger_count": [None, 1, 1],
            "trip_distance": [4.0, 0.0, 2.0],
            "RatecodeID": [None, 1, 1],
            "store_and_fwd_flag": [None, "N", "N"],
            "PULocationID": [10, 10, 10],
            "DOLocationID": [20, 20, 20],
            "payment_type": [0, 1, 1],
            "fare_amount": [18.0, 0.0, 10.0],
            "tip_amount": [3.6, 0.0, 2.0],
            "total_amount": [24.0, 0.0, 14.0],
            "congestion_surcharge": [None, 0.0, 0.0],
            "Airport_fee": [None, 0.0, 0.0],
        }
    ).lazy()


def test_quality_filters_keep_only_valid_may_trip() -> None:
    cleaned = PROCESSOR.apply_quality_filters(_toy_lazyframe()).collect()
    assert cleaned.height == 1
    assert cleaned["trip_distance"].item() == 4.0


def test_engineering_imputes_and_creates_features() -> None:
    filtered = PROCESSOR.apply_quality_filters(_toy_lazyframe())
    engineered = PROCESSOR.engineer_features(filtered).collect()
    assert engineered["passenger_count"].item() == 1
    assert engineered["RatecodeID"].item() == 99
    assert engineered["store_and_fwd_flag"].item() == "Unknown"
    assert engineered["trip_duration_minutes"].item() == 20.0
    assert engineered["average_speed_mph"].item() == 12.0
    assert engineered["time_of_day"].item() == "morning"
    assert engineered["distance_band"].item() == "medium"


def test_preprocessing_plan_branches_from_observed_results() -> None:
    pandas_analysis = {
        "missing": {
            "passenger_count": {"rate_pct": 23.0},
            "RatecodeID": {"rate_pct": 23.0},
            "store_and_fwd_flag": {"rate_pct": 23.0},
            "congestion_surcharge": {"rate_pct": 23.0},
            "Airport_fee": {"rate_pct": 23.0},
        },
        "numeric_summary": {
            "passenger_count": {"50%": 1.0},
            "congestion_surcharge": {"50%": 2.5},
            "Airport_fee": {"50%": 0.0},
        },
        "correlations": {
            "fare_amount": {"trip_distance": 0.01},
        },
        "missing_pattern": {
            "all_five_missing_count": 950_000,
            "all_five_missing_are_payment_zero": True,
            "all_payment_zero_have_five_missing": True,
        },
    }
    eda_profile = {
        "quality_counts": {
            "pickup_outside_may": 1,
            "negative_fare": 1,
            "negative_total": 1,
            "negative_tip": 1,
            "speed_gt_80_mph": 1,
            "passenger_gt_6": 1,
            "invalid_location_id": 0,
        },
        "numeric_summary": {
            "fare_amount": {"p99": 80.0, "max": 5_000.0},
            "total_amount": {"p99": 100.0, "max": 5_000.0},
            "trip_distance": {"p99": 20.0, "max": 300_000.0},
            "trip_duration_minutes": {"p99": 80.0, "max": 9_000.0},
            "average_speed_mph": {"max": 1_000.0},
            "passenger_count": {"max": 9.0},
        },
    }
    raw_tests = [
        {"test": "Pearson correlation", "statistic": 0.01, "p_value": 0.01},
        {"test": "Spearman correlation", "statistic": 0.80, "p_value": 0.0},
        {
            "test": "Welch t-test",
            "statistic": 10.0,
            "p_value": 0.001,
            "effect_or_difference": -2.0,
        },
    ]

    plan = PROCESSOR.build_preprocessing_plan(
        pandas_analysis, eda_profile, raw_tests
    )

    assert plan["imputation"]["passenger_count"]["value"] == 1
    assert plan["imputation"]["RatecodeID"]["strategy"] == "unknown"
    assert plan["rules"]["distance_cap"] == 100.0
    assert plan["rules"]["fare_cap"] == 500.0
    assert plan["statistical_evidence"]["correlation_outlier_distortion"]
