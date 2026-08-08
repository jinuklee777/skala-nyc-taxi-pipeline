"""Jinja2-based Markdown report generation."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Any

from jinja2 import Environment, FileSystemLoader, StrictUndefined

from .config import DATA_URL, ProjectPaths


# 주요 기능
# EDA, 벤치마크, 전처리, 통계, 모델 결과를 Jinja2 템플릿에 전달해
# 사람이 바로 읽을 수 있는 최종 Markdown 보고서(report.md)를 자동 생성한다.
class ReportGenerator:
    """Render all pipeline outputs into the final report.md artifact."""

    def __init__(self, paths: ProjectPaths) -> None:
        self.paths = paths

    def render(
        self,
        benchmark: dict[str, Any],
        eda: dict[str, Any],
        cleaning: dict[str, Any],
        raw_statistical_tests: list[dict[str, Any]],
        statistical_tests: list[dict[str, Any]],
        model_metrics: dict[str, Any],
    ) -> Path:
        # 숫자와 p-value 표시 형식을 공통으로 정의해 보고서 표기를 일관되게 유지한다.
        environment = Environment(
            loader=FileSystemLoader(self.paths.templates),
            undefined=StrictUndefined,
            autoescape=False,
            trim_blocks=True,
            lstrip_blocks=True,
        )
        environment.filters["number"] = (
            lambda value, digits=0: f"{value:,.{digits}f}"
        )
        environment.filters["pvalue"] = (
            lambda value: "< 1e-300" if value == 0 else f"{value:.3e}"
        )
        # 템플릿에 각 단계 결과를 주입한 뒤 reports/report.md로 저장한다.
        template = environment.get_template("report.md.j2")
        report = template.render(
            generated_at=datetime.now().isoformat(timespec="seconds"),
            data_url=DATA_URL,
            benchmark=benchmark,
            eda=eda,
            cleaning=cleaning,
            preprocessing_plan=cleaning["preprocessing_plan"],
            raw_statistical_tests=raw_statistical_tests,
            statistical_tests=statistical_tests,
            model=model_metrics,
        )
        report_path = self.paths.reports / "report.md"
        report_path.write_text(report, encoding="utf-8")
        return report_path
