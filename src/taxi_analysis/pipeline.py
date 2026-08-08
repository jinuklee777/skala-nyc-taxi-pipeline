"""Thin orchestration layer for the class-based end-to-end pipeline."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

from .config import DATA_URL, ProjectPaths
from .data_loader import DataLoader
from .data_preprocessor import DataPreprocessor
from .eda_analyzer import EDAAnalyzer
from .model_trainer import ModelTrainer
from .report_generator import ReportGenerator
from .statistical_analyzer import StatisticalAnalyzer
from .utils import write_json
from .visualizer import TaxiVisualizer


# 주요 기능
# 각 분석 클래스가 담당 업무에만 집중하도록 하고, 이 클래스는 실행 순서와 결과 전달만 관리한다.
# 데이터 로드 → EDA/원본 통계 → 근거 기반 전처리 → 시각화 → 정제 통계 → 모델 → 보고서 순서로 연결한다.
class TaxiAnalysisPipeline:
    """Coordinate each independent analysis class in execution order."""

    def __init__(
        self,
        project_root: Path,
        force_download: bool = False,
        eda_sample_size: int = 200_000,
        model_sample_size: int = 150_000,
    ) -> None:
        self.paths = ProjectPaths(project_root.resolve())
        self.force_download = force_download
        self.eda_sample_size = eda_sample_size
        self.model_sample_size = model_sample_size

    def run(self) -> dict[str, Any]:
        """Execute data preparation through automatic report generation."""
        self.paths.create_directories()
        pipeline_started = time.perf_counter()

        # 1단계: 원본 다운로드 및 Pandas/Polars 성능 비교
        data_loader = DataLoader(
            data_url=DATA_URL,
            data_path=self.paths.raw_data,
            artifacts_dir=self.paths.artifacts,
        )
        data_loader.download(force=self.force_download)
        benchmark = data_loader.benchmark()
        pandas_analysis = benchmark["pandas_analysis"]

        # 2단계: 전처리 전에 원본 데이터의 분포와 품질 문제 확인
        eda_analyzer = EDAAnalyzer(
            data_path=self.paths.raw_data,
            artifacts_dir=self.paths.artifacts,
            sample_size=self.eda_sample_size,
        )
        eda = eda_analyzer.build_profile()

        # 3단계: 전처리 전 원본에 통계검정을 수행해 관계와 집단 차이를 확인
        statistical_analyzer = StatisticalAnalyzer(
            processed_path=self.paths.processed_data,
            artifacts_dir=self.paths.artifacts,
            sample_size=self.eda_sample_size,
        )
        raw_statistical_tests = statistical_analyzer.run_raw(
            self.paths.raw_data
        )

        # 4단계: Pandas·Polars EDA·원본 통계 결과를 근거로 전처리 규칙을 분기
        data_preprocessor = DataPreprocessor(
            raw_path=self.paths.raw_data,
            processed_path=self.paths.processed_data,
            artifacts_dir=self.paths.artifacts,
        )
        cleaning = data_preprocessor.clean_and_save(
            pandas_analysis=pandas_analysis,
            eda_profile=eda,
            raw_statistical_tests=raw_statistical_tests,
        )

        # 5단계: EDA 차트, 전처리 전후 근거 차트, 인터랙티브 차트 생성
        visualizer = TaxiVisualizer(
            raw_path=self.paths.raw_data,
            processed_path=self.paths.processed_data,
            figures_dir=self.paths.figures,
            sample_size=self.eda_sample_size,
        )
        visualizations = visualizer.create(
            pandas_analysis=pandas_analysis,
            cleaning=cleaning,
        )

        # 6단계: 정제 데이터에 같은 검정을 반복해 전처리 전후 결과 비교
        statistical_tests = statistical_analyzer.run()

        # 7단계: sklearn Pipeline 학습, 평가, joblib 모델 저장
        model_trainer = ModelTrainer(
            processed_path=self.paths.processed_data,
            artifacts_dir=self.paths.artifacts,
            sample_size=self.model_sample_size,
        )
        model_metrics = model_trainer.train_and_save()

        # 8단계: 분석 근거와 전처리 전후 통계를 Markdown 보고서로 자동 생성
        report_generator = ReportGenerator(self.paths)
        report_path = report_generator.render(
            benchmark=benchmark,
            eda=eda,
            cleaning=cleaning,
            raw_statistical_tests=raw_statistical_tests,
            statistical_tests=statistical_tests,
            model_metrics=model_metrics,
        )

        run_summary = {
            "status": "success",
            "pipeline_seconds": time.perf_counter() - pipeline_started,
            "report": str(report_path),
            "processed_data": str(self.paths.processed_data),
            "visualizations": visualizations,
            "model": model_metrics["model_path"],
        }
        write_json(self.paths.artifacts / "run_summary.json", run_summary)
        print("\n=== PIPELINE COMPLETE ===")
        print(json.dumps(run_summary, ensure_ascii=False, indent=2))
        return run_summary


def run_pipeline(
    project_root: Path,
    force_download: bool = False,
    eda_sample_size: int = 200_000,
    model_sample_size: int = 150_000,
) -> dict[str, Any]:
    """Backward-compatible functional entry point around the orchestrator class."""
    return TaxiAnalysisPipeline(
        project_root=project_root,
        force_download=force_download,
        eda_sample_size=eda_sample_size,
        model_sample_size=model_sample_size,
    ).run()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="NYC Yellow Taxi 2026-05 end-to-end analysis pipeline"
    )
    parser.add_argument(
        "--project-root",
        type=Path,
        default=Path(__file__).resolve().parents[2],
    )
    parser.add_argument("--force-download", action="store_true")
    parser.add_argument("--eda-sample-size", type=int, default=200_000)
    parser.add_argument("--model-sample-size", type=int, default=150_000)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    run_pipeline(
        project_root=args.project_root,
        force_download=args.force_download,
        eda_sample_size=args.eda_sample_size,
        model_sample_size=args.model_sample_size,
    )


if __name__ == "__main__":
    main()
