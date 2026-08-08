# NYC Yellow Taxi End-to-End Data Analysis

NYC Yellow Taxi Trip Data **2026-05** 전체(4,090,836행)를 대상으로 한 재현 가능한 분석 프로젝트입니다.

## 설계 원칙

전처리 규칙을 임의로 정하지 않습니다. EDA와 통계 검정을 먼저 수행하고, 그 관찰값이 전처리 규칙을 선택하도록 구성합니다. 선택된 규칙과 근거는 `artifacts/preprocessing_decisions.json`에 관찰값·처리·이유 형태로 기록됩니다.

## 빠른 실행

Python 3.11과 [uv](https://docs.astral.sh/uv/)를 권장합니다.

```bash
make setup
make run
make test
```

## 프로젝트 구조

```text
data_analysis/
├── data/                      # 원본/정제 데이터 (git 제외)
├── artifacts/                 # EDA JSON, 벤치마크, 통계, 모델, 지표
├── figures/                   # Seaborn PNG, Plotly HTML
├── reports/report.md          # 자동 생성 최종 보고서
├── src/taxi_analysis/
│   ├── config.py              # 공통 상수와 프로젝트 경로
│   └── utils.py               # 공통 직렬화 유틸리티
├── templates/                 # 보고서 템플릿
├── tests/
├── run_pipeline.py
└── Makefile
```

각 분석 단계는 한 클래스와 한 파일로 분리합니다.
