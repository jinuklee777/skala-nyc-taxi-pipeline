# NYC Yellow Taxi End-to-End Data Analysis

NYC Yellow Taxi Trip Data **2026-05** 전체(4,090,836행)를 대상으로 한 재현 가능한 분석 프로젝트입니다.

> 분석 결과와 판단 근거는 **[PROJECT_REPORT.md](PROJECT_REPORT.md)**, 파이프라인이 자동 생성한 상세 수치표는 **[reports/report.md](reports/report.md)**에 있습니다.

## 구현 범위

`데이터 다운로드 → Pandas/Polars 비교 → EDA·원본 통계 검정 → 분석 결과 기반 전처리 분기 → Feature Engineering → 전처리 전후 시각화 → 정제 후 통계 검정 → sklearn Pipeline → 모델 저장 → report.md 생성`

핵심 원칙은 전처리 규칙을 임의로 정하지 않는 것입니다. `DataPreprocessor`가 Pandas의 결측률·중앙값·상관계수, Polars EDA의 분포·품질 위반 건수, 원본 Pearson/Spearman/t-test 결과를 전달받아 처리 전략을 분기하고 `artifacts/preprocessing_decisions.json`에 관찰값·처리·이유를 기록합니다.

## 빠른 실행

Python 3.11과 [uv](https://docs.astral.sh/uv/)를 권장합니다.

```bash
git clone https://github.com/jinuklee777/skala-nyc-taxi-pipeline.git
cd skala-nyc-taxi-pipeline
make setup
make run
make test
```

이미 `.venv`가 준비되어 있다면 다음만 실행하면 됩니다.

```bash
.venv/bin/python run_pipeline.py
```

주요 옵션:

```bash
.venv/bin/python run_pipeline.py \
  --eda-sample-size 200000 \
  --model-sample-size 150000
```

원본 파일이 없으면 공식 CloudFront URL에서 자동 다운로드합니다. `--force-download`로 다시 받을 수 있습니다.

## 프로젝트 구조

```text
skala-nyc-taxi-pipeline/
├── data/                      # git 제외 (실행 시 자동 생성)
│   ├── raw/                   # 공식 원본 Parquet
│   └── processed/             # 정제 + Feature 데이터
├── artifacts/                 # EDA JSON, 벤치마크, 통계, 모델, 지표
├── figures/                   # Seaborn PNG, Plotly HTML
├── reports/report.md          # 자동 생성 최종 보고서
├── src/taxi_analysis/
│   ├── config.py              # 공통 상수와 프로젝트 경로
│   ├── data_loader.py         # DataLoader: 다운로드, Pandas/Polars 비교
│   ├── eda_analyzer.py        # EDAAnalyzer: 분포, 결측, 상관, 품질 진단
│   ├── data_preprocessor.py   # DataPreprocessor: 분석 결과 기반 전처리 분기
│   ├── visualizer.py          # TaxiVisualizer: EDA, 전처리 근거, Plotly
│   ├── statistical_analyzer.py # StatisticalAnalyzer: 전처리 전후 통계 비교
│   ├── model_trainer.py       # ModelTrainer: sklearn Pipeline, 평가, 저장
│   ├── report_generator.py    # ReportGenerator: report.md 생성
│   ├── pipeline.py            # TaxiAnalysisPipeline: 실행 순서만 조정
│   └── utils.py               # 공통 직렬화 유틸리티
├── templates/report.md.j2     # 보고서 템플릿
├── tests/                     # 핵심 정제/Feature 단위 테스트
├── run_pipeline.py
├── pyproject.toml
└── Makefile
```

각 분석 단계는 한 클래스와 한 파일로 분리되어 있습니다. `pipeline.py`는 분석
로직을 직접 구현하지 않고 위 클래스들의 생성 및 실행 순서만 담당합니다.

## 모델링 설계

`total_amount`를 예측하되 `fare_amount`, `tip_amount`, 세금·할증·통행료처럼 target을 직접 구성하는 컬럼은 모두 제외합니다. 무작위 분할 대신 pickup 시각 순으로 정렬한 앞 80%/뒤 20%를 사용해 미래 레코드에 대한 평가에 가깝게 구성했습니다. 모델과 전처리기는 하나의 sklearn `Pipeline`으로 저장됩니다.

## 결과 확인

실행 후 `reports/report.md`에서 Pandas 분석, 원본/정제 통계검정, 전처리 결정 근거와 전후 그래프를 함께 확인할 수 있습니다. 세부 원자료는 `artifacts/`의 CSV/JSON/TXT에 남아 있어 보고서 수치를 추적할 수 있습니다.
