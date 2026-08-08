"""Command-line entry point for the NYC Yellow Taxi analysis pipeline."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
SRC_DIR = PROJECT_ROOT / "src"
# 패키지를 별도 설치하지 않아도 프로젝트의 src 폴더를 import할 수 있게 한다.
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from taxi_analysis.pipeline import main  # noqa: E402


if __name__ == "__main__":
    # 실제 분석 로직은 TaxiAnalysisPipeline에 있고, 이 파일은 CLI 시작점만 담당한다.
    main()
