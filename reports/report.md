# NYC Yellow Taxi Trip Data (2026-05) 분석 보고서

- 생성 시각: 2026-08-08T13:47:11
- 원본 데이터: https://d37ci6vzurychx.cloudfront.net/trip-data/yellow_tripdata_2026-05.parquet
- 분석 범위: NYC Yellow Taxi, 2026년 5월
- 파이프라인: 데이터 로드 → Pandas/Polars EDA → 원본 통계 분석 → 근거 기반 전처리 → Feature Engineering → 시각화 → 정제 후 통계 분석 → 머신러닝 → 모델 저장 → 보고서 생성

## 1. Executive Summary

원본은 **4,090,836행 × 20열**이다. 완전 중복은 **0건**이었으며, Pandas 기술통계·Polars EDA·원본 통계검정을 먼저 수행한 뒤 그 결과로 전처리 규칙을 선택했다. 정제 후에는 **3,899,516행**이 남아 원본의 **4.68%**가 제거됐다.

Pandas에서 `passenger_count`, `RatecodeID`, `store_and_fwd_flag`, `congestion_surcharge`, `Airport_fee`의 결측률이 각각 **23.3539%**로 관찰됐다. 다섯 컬럼이 동시에 결측인 **955,371건**은 모두 `payment_type=0`이며 그 역도 성립해 무작위 결측보다 구조적 미기록 패턴으로 판단했다. 승객 수는 이상치에 강건한 관찰 중앙값 **1**로 대체하고, 코드형 변수는 명시적 `Unknown`, 가산 요금은 중립값 0으로 처리했다.

원본 통계검정과 분포에서 확인한 관계 및 극단값을 전처리·Feature Engineering 근거로 사용했다. 이후 같은 통계검정을 정제 데이터에 다시 적용해 전처리 전후 관계가 어떻게 달라졌는지 비교했다.

ML 모델은 요금 구성요소를 Feature에서 제외해 target leakage를 방지한 상태에서 `total_amount`를 예측했다. 시간 순서 holdout 기준 MAE는 **$3.24**, R²는 **0.916**이며, 단순 중앙값 기준선보다 MAE가 **74.8%** 개선됐다.

## 2. 데이터 로드 및 Pandas·Polars 비교

두 엔진 모두 동일한 Parquet 전체 파일을 사용했다. Polars는 `scan_parquet()` Lazy API 생성 시간과 `collect()` 구체화 시간을 구분했다. 메모리는 각 DataFrame 자체 크기, 처리 속도는 같은 `payment_type`별 건수·평균 총액 집계로 비교했다.

| Engine | Lazy scan setup (s) | Load/materialize (s) | DataFrame memory (MB) | GroupBy (s) |
|---|---:|---:|---:|---:|
| Pandas | 0.0000 | 0.3354 | 580.62 | 0.0877 |
| Polars Lazy | 0.0039 | 0.0947 | 550.13 | 0.0217 |

- Polars 로딩 속도 배수(Pandas/Polars): **3.54×**
- Polars GroupBy 속도 배수(Pandas/Polars): **4.04×**
- DataFrame 메모리 비율(Pandas/Polars): **1.06×**
- Pandas의 `shape`, `columns`, `info()`, `describe()`, dtype, 결측 수/비율, 중복 수, 메모리 상세는 [`../artifacts/pandas_data_inspection.txt`](../artifacts/pandas_data_inspection.txt)에 저장했다.

> 실행 시간은 하드웨어·OS 캐시·실행 순서의 영향을 받으므로 같은 환경에서 여러 번 반복 측정할 때 더 안정적이다.

### Pandas 분석 결과

아래 값은 전처리 전에 Pandas가 원본 전체 데이터에서 계산한 결과다. 이 표의 중앙값과 결측률은 결측 대체 분기에, p99와 최댓값의 차이는 이상치 처리 필요성을 판단하는 교차 근거로 사용했다.

| Variable | Mean | Std | Q1 | Median | Q3 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|---:|
| `passenger_count` | 1.24 | 0.64 | 1.00 | 1.00 | 1.00 | 4.00 | 9.00 |
| `trip_distance` | 4.95 | 483.62 | 1.04 | 1.87 | 3.81 | 19.40 | 307,491.47 |
| `fare_amount` | 21.51 | 19.01 | 10.00 | 16.30 | 26.80 | 81.98 | 5,525.99 |
| `tip_amount` | 2.98 | 4.02 | 0.00 | 2.24 | 4.10 | 17.69 | 239.00 |
| `total_amount` | 30.49 | 22.97 | 17.64 | 23.94 | 34.95 | 106.31 | 5,530.74 |
| `congestion_surcharge` | 2.20 | 0.84 | 2.50 | 2.50 | 2.50 | 2.50 | 2.50 |
| `Airport_fee` | 0.16 | 0.59 | 0.00 | 0.00 | 0.00 | 2.00 | 27.00 |

Pandas 원본 상관계수는 `trip_distance–fare_amount` **0.0078**, `trip_distance–total_amount` **0.0073**였다. 전체 행 기준 계산은 극단값의 영향을 크게 받을 수 있으므로 순위 기반 Spearman 검정과 함께 해석했다.

## 3. Polars Lazy EDA 결과

### 결측치

| Column | Missing count | Missing rate |
|---|---:|---:|
| `passenger_count` | 955,371 | 23.3539% |
| `RatecodeID` | 955,371 | 23.3539% |
| `store_and_fwd_flag` | 955,371 | 23.3539% |
| `congestion_surcharge` | 955,371 | 23.3539% |
| `Airport_fee` | 955,371 | 23.3539% |

### 수치형 분포와 이상치

| Variable | Min | Q1 | Median | Q3 | P99 | Max |
|---|---:|---:|---:|---:|---:|---:|
| `passenger_count` | 0.00 | 1.00 | 1.00 | 1.00 | 4.00 | 9.00 |
| `trip_distance` | 0.00 | 1.04 | 1.87 | 3.81 | 19.40 | 307,491.47 |
| `fare_amount` | -950.00 | 10.00 | 16.30 | 26.80 | 81.98 | 5,525.99 |
| `tip_amount` | -47.10 | 0.00 | 2.24 | 4.10 | 17.69 | 239.00 |
| `total_amount` | -951.00 | 17.64 | 23.94 | 34.95 | 106.31 | 5,530.74 |
| `trip_duration_minutes` | 0.00 | 8.58 | 14.48 | 23.27 | 79.62 | 9,960.15 |
| `average_speed_mph` | 0.00 | 6.31 | 8.80 | 12.26 | 32.94 | 1,511,821.46 |

IQR만으로 상단 꼬리를 일괄 제거하면 정상적인 공항·장거리 운행도 함께 제거할 수 있다. 따라서 p99와 최댓값의 격차가 10배 이상인지 먼저 판단하고, 그때만 보수적인 택시 도메인 상한을 활성화했다.

### 범주형 분포

`VendorID=2`가 가장 많고, `payment_type=1`이 주된 결제 유형이다. 범주별 전체 건수는 [`../artifacts/eda_profile.json`](../artifacts/eda_profile.json)에 저장했다.

### 데이터 품질 문제

| Check | Count |
|---|---:|
| `nonpositive_distance` | 113,031 |
| `negative_fare` | 14,231 |
| `negative_total` | 14,877 |
| `negative_tip` | 44 |
| `fare_gt_500` | 74 |
| `total_gt_500` | 105 |
| `passenger_lt_1_nonnull` | 12,533 |
| `passenger_gt_6` | 4 |
| `nonpositive_duration` | 52,063 |
| `duration_gt_240_minutes` | 1,129 |
| `distance_gt_100_miles` | 136 |
| `speed_gt_80_mph` | 1,020 |
| `pickup_outside_may` | 14 |
| `invalid_location_id` | 0 |

픽업 날짜 범위는 2008-12-31 23:05:53부터 2026-06-01 00:20:35까지로, 분석 월 밖 레코드가 포함돼 있었다.

## 4. 분석 결과에 따른 전처리 분기와 근거

`DataPreprocessor`는 Pandas 결측률·중앙값·상관계수, Polars EDA의 p99·최댓값·품질 위반 건수, 원본 통계검정의 Pearson·Spearman·Welch t-test 결과를 입력받아 아래 계획을 만들었다. 즉, 고정된 설명을 붙인 것이 아니라 현재 데이터에서 관찰된 값으로 조건을 평가한 결과다.

| 단계 | 실제 관찰 | 선택한 처리 | 선택 이유 |
|---|---|---|---|
| 결측치 | passenger_count 결측 23.3539%, Pandas 중앙값 1 | 중앙값 1 대체 | 삭제 시 표본 손실이 크고 중앙값이 이상치에 강건하다. |
| 결측치 | RatecodeID/store_and_fwd_flag 결측률 23.3539%; 5개 컬럼 동시 결측 955,371건, payment_type=0과 완전 일치=예 | 명시적 Unknown 범주 대체 | 5%를 넘고 payment_type=0과 일치하는 구조적 결측을 최빈값으로 숨기지 않는다. |
| 결측치 | congestion_surcharge/Airport_fee 결측률 23.3539%; 위 구조적 결측 패턴과 동일 | 가산 요금 결측을 0으로 대체 | payment_type=0에서 다섯 컬럼이 함께 기록되지 않은 구조이므로 행을 삭제하지 않고, 금액 가산 여부에는 중립적인 0을 사용한다. |
| 금액 이상치 | fare p99=$81.98, max=$5525.99; total p99=$106.31, max=$5530.74 | fare/total $500 상한 및 음수 제거 | 최대값이 p99의 10배를 넘어 입력 오류로 판단했다. |
| 거리·시간 이상치 | 거리 p99=19.40, max=307491.47; 시간 p99=79.62, max=9960.15 | 거리 100mile·시간 240분·속도 80mph 상한 | 최댓값이 p99에서 크게 벗어났으며, IQR 단독 적용으로 정상 공항 운행을 제거하지 않도록 보수적인 택시 도메인 상한을 사용했다. |
| Feature Engineering | Pandas 거리-운임 r=0.0078, 원본 Pearson=0.0104, Spearman=0.7971, 주말 t-test p=3.797e-79 | 운행시간·속도·시간대·주말 여부 Feature 생성 | 원본 극단값이 Pearson 상관을 왜곡해도 순위 상관이 관계를 지지하며, 유의한 주중·주말 차이를 모델과 시각화에 반영한다. |

### 이번 실행에서 선택된 규칙

| 항목 | 실행값 |
|---|---|
| 대상 월 필터 | 적용 |
| 운임 범위 | `0 <= fare_amount <= 500.00` |
| 총액 범위 | `0 <= total_amount <= 500.00` |
| 거리 범위 | `0 < trip_distance <= 100.00` |
| 운행시간 범위 | `0 < trip_duration_minutes <= 240.00` |
| 평균속도 상한 | `80.00 mph` |
| 승객 수 | 결측을 `1`로 대체, `1–6`명 유지 |
| RatecodeID 결측 | `unknown` 전략, 값 `99` |
| store_and_fwd_flag 결측 | `unknown` 전략, 값 `Unknown` |
| congestion_surcharge 결측 | `zero` 전략, 값 `0.0` |
| Airport_fee 결측 | `zero` 전략, 값 `0.0` |

적용 순서는 완전 중복 제거 → 대상 월 필터 → 금액·거리·운행시간 규칙 → 속도·승객 수·Location ID 규칙 → 결측 대체 → Feature 생성이다. 생성 Feature는 `trip_duration_minutes`, `average_speed_mph`, `pickup_hour`, `pickup_day_of_week`, `pickup_date`, `is_weekend`, `time_of_day`, `tip_percentage`, `cost_per_mile`, `is_airport_trip`, `distance_band`다.

정제 결과는 **3,899,516행 × 31열**, 파일 크기는 **128.08MB**다. 남은 결측 컬럼: 없음.

![전처리 근거 및 전후 비교](../figures/preprocessing_evidence.png)

위 그래프는 Pandas에서 관찰한 원본 결측률, 전처리 전후 행 수, 원본 p99 구간에서의 거리·총액 분포 변화를 함께 보여준다. 분포 그래프는 모양 비교를 위해 밀도로 정규화했으며, 전체 행 수 차이는 우측 상단 막대에서 별도로 표시했다.

## 5. EDA 시각화

![2×2 EDA dashboard](../figures/eda_overview_2x2.png)

- Seaborn 2×2: 거리 분포, 결제 유형 분포, 시간대별 수요, 수치형 상관관계
- Plotly 인터랙티브 차트: [`hourly_demand_interactive.html`](../figures/hourly_demand_interactive.html)

## 6. 통계 분석: 전처리 전후 비교

계산 비용과 극단적으로 작은 p-value의 과해석을 줄이기 위해 원본과 정제 데이터에서 각각 고정 seed로 **200,000건**을 추출했다. p-value만 보지 않고 상관계수·평균 차이·Cramér's V와 문장 해석을 함께 제시한다.

### 6.1 원본 데이터 통계검정 — 전처리 근거

| Test | Question | Statistic | p-value | Effect / difference | N | 해석 |
|---|---|---:|---:|---:|---:|---|
| Pearson correlation | trip_distance와 fare_amount의 선형 관계 | 0.0104 | 3.634e-06 | 0.0104 | 200,000 | 거리와 운임은 매우 약한 양의 선형 관계를 보인다. |
| Spearman correlation | trip_distance와 fare_amount의 단조 관계 | 0.7971 | < 1e-300 | 0.7971 | 200,000 | 거리와 운임은 강한 양의 단조 관계를 보인다. |
| Welch t-test | 주중과 주말의 평균 total_amount 차이 | 18.8498 | 3.797e-79 | -2.0321 | 200,000 | 유의수준 0.05에서 귀무가설을 기각한다. 주말 평균 total_amount가 주중보다 낮다. |
| Chi-square independence | payment_type과 20% 이상 팁 여부의 연관성 | 85,226.7487 | < 1e-300 | 0.6528 | 200,000 | 결제유형과 20% 이상 팁 여부는 통계적으로 연관되어 있다. |

### 6.2 정제 데이터 통계검정 — 처리 후 재확인

| Test | Question | Statistic | p-value | Effect / difference | N | 해석 |
|---|---|---:|---:|---:|---:|---|
| Pearson correlation | trip_distance와 fare_amount의 선형 관계 | 0.8712 | < 1e-300 | 0.8712 | 200,000 | 거리와 운임은 강한 양의 선형 관계를 보인다. |
| Spearman correlation | trip_distance와 fare_amount의 단조 관계 | 0.8701 | < 1e-300 | 0.8701 | 200,000 | 거리와 운임은 강한 양의 단조 관계를 보인다. |
| Welch t-test | 주중과 주말의 평균 total_amount 차이 | 17.5577 | 6.254e-69 | -1.7877 | 200,000 | 유의수준 0.05에서 귀무가설을 기각한다. 주말 평균 total_amount가 주중보다 낮다. |
| Chi-square independence | payment_type과 20% 이상 팁 여부의 연관성 | 83,098.6727 | < 1e-300 | 0.6446 | 200,000 | 결제유형과 20% 이상 팁 여부는 통계적으로 연관되어 있다. |

원본 Welch t-test에서 주중 평균은 **$31.05**, 주말 평균은 **$29.02**, p-value는 **3.797e-79**였다. 유의수준 0.05에서 귀무가설을 기각한다. 주말 평균 total_amount가 주중보다 낮다.
정제 후에는 주중 평균 **$31.06**, 주말 평균 **$29.28**, p-value **6.254e-69**로 관찰됐다. 유의수준 0.05에서 귀무가설을 기각한다. 주말 평균 total_amount가 주중보다 낮다.

전처리 전후의 계수 차이는 전처리 효과를 의미하지만 인과효과는 아니다. 특히 표본 수가 크면 작은 차이도 유의해질 수 있으므로, 통계적 유의성과 효과 크기를 함께 해석해야 한다.

## 7. 머신러닝 Pipeline

- 목표: `total_amount` 회귀 예측
- 모델: `HistGradientBoostingRegressor`
- 전처리: 수치형 중앙값 대체 + 범주형 최빈값 대체/Ordinal Encoding
- 분할: pickup_datetime 기준 정렬 후 앞 80% 학습 / 뒤 20% 평가
- 표본: 150,000행 (train 120,000, test 30,000)
- 누수 방지: `fare_amount`, `tip_amount`, 세금·할증·통행료 등 총액 구성요소는 Feature에서 제외

| Metric | Value |
|---|---:|
| MAE | $3.238 |
| RMSE | $6.085 |
| R² | 0.9157 |
| Median baseline MAE | $12.864 |
| MAE improvement | 74.83% |
| Fit time | 6.01s |

모델은 `../artifacts/total_amount_model.joblib`에 전처리기와 함께 저장했다. 이는 분석용 baseline이며 실제 배포 전에는 비용 민감도, 시간대별 drift, 예측구간, 외부 월 데이터 검증이 추가로 필요하다.

## 8. 재현 방법 및 산출물

```bash
cd data_analysis
make setup
make run
make test
```

- Pandas 분석: `artifacts/pandas_analysis.json`, `artifacts/pandas_numeric_summary.csv`, `artifacts/pandas_correlation_matrix.csv`
- 정제 데이터: `data/processed/yellow_taxi_2026-05_cleaned.parquet`
- EDA 프로파일: `artifacts/eda_profile.json`
- 전처리 결정: `artifacts/preprocessing_decisions.json`
- 벤치마크: `artifacts/engine_benchmark.csv`
- 원본/정제 통계 검정: `artifacts/raw_statistical_tests.csv`, `artifacts/statistical_tests.csv`
- 모델/평가지표: `artifacts/total_amount_model.joblib`, `artifacts/model_metrics.json`
- 시각화: `figures/eda_overview_2x2.png`, `figures/preprocessing_evidence.png`, `figures/hourly_demand_interactive.html`