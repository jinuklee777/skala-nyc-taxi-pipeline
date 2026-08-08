from pathlib import Path

from taxi_analysis.pipeline import TaxiAnalysisPipeline


def test_pipeline_initializes_project_paths(tmp_path: Path) -> None:
    pipeline = TaxiAnalysisPipeline(
        project_root=tmp_path,
        eda_sample_size=100,
        model_sample_size=100,
    )

    assert pipeline.paths.root == tmp_path.resolve()
    assert pipeline.paths.raw_data.name == "yellow_tripdata_2026-05.parquet"
    assert pipeline.paths.processed_data.name.endswith("_cleaned.parquet")

