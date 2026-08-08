"""Shared constants and filesystem configuration for the taxi project."""

from __future__ import annotations

import os
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

# 홈 디렉터리 권한에 의존하지 않도록 프로젝트 내부 캐시를 사용한다.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".matplotlib-cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

DATA_URL = (
    "https://d37ci6vzurychx.cloudfront.net/trip-data/"
    "yellow_tripdata_2026-05.parquet"
)
MONTH_START = datetime(2026, 5, 1)
MONTH_END = datetime(2026, 6, 1)
RANDOM_SEED = 42


# 프로젝트에서 사용하는 원본·정제 데이터, 분석 결과, 시각화, 보고서 경로를
# 한곳에서 관리한다. 경로가 바뀌어도 이 클래스만 수정하면 나머지 코드는 그대로 동작한다.
@dataclass(frozen=True)
class ProjectPaths:
    """All filesystem locations used by the pipeline."""

    root: Path

    @property
    def raw_data(self) -> Path:
        return self.root / "data/raw/yellow_tripdata_2026-05.parquet"

    @property
    def processed_data(self) -> Path:
        return self.root / "data/processed/yellow_taxi_2026-05_cleaned.parquet"

    @property
    def artifacts(self) -> Path:
        return self.root / "artifacts"

    @property
    def figures(self) -> Path:
        return self.root / "figures"

    @property
    def reports(self) -> Path:
        return self.root / "reports"

    @property
    def templates(self) -> Path:
        return self.root / "templates"

    def create_directories(self) -> None:
        # 파이프라인 실행 전에 필요한 출력 폴더를 자동으로 생성한다.
        for directory in (
            self.raw_data.parent,
            self.processed_data.parent,
            self.artifacts,
            self.figures,
            self.reports,
        ):
            directory.mkdir(parents=True, exist_ok=True)
