# Plan

> Source: GitHub Wiki — `shin584/Demand_Forecasting_Program` (pages: Home, plan)
> Local copy for offline reference. Last synced: 2026-09-16.

## English Summary

**Goal:** Prevent pharmacy stockouts and automate drug reordering via a two-track demand forecast:
1. **Chronic patients (ML-based):** `visit probability × last prescription quantity` = deterministic expected demand.
2. **Acute patients (statistics-based):** day-of-week / seasonal average consumption = statistical demand.

**Pipeline:** Extract 3 years of raw MSSQL data (no pre-aggregation) → build Pandas data marts → train tree-based models (LightGBM/XGBoost) → compute inventory expected values → apply safety stock → feed automated ordering system.

---

## 약국 내방 및 재고 예측을 위한 AI 모델링 계획서

### 프로젝트 개요
* **목표:** 약국 품절 사태 방지 및 효율적인 재고 발주 자동화
* **핵심 전략 (Two-Track):**
    1. **만성질환자 (ML 기반):** 방문 확률 × 최근 처방 수량으로 확정적 수요 예측
    2. **급성질환자 (통계 기반):** 요일/계절별 전체 평균 소모량으로 통계적 수요 예측

### 1단계: DB 원본 데이터 추출 (Data Extraction)
복잡한 전처리는 파이썬(Pandas)에서 수행하므로, DBMS(MSSQL)에서는 최근 3년간의 로우 데이터(Raw Data)를 조건 없이 통째로 추출합니다.

#### 대상 테이블 및 핵심 칼럼
1. **`dbo.tbl매출`**: `조제판매ID`, `고객ID`, `내방일`, `조제일수`, `다음내방일`, `처방전발행기관ID`
2. **`dbo.tbl고객`**: `고객ID`, `가족ID`, `성별`, `생년월일`
3. **`dbo.tbl가족총매출`**: `가족ID`, `총내방`, `총매출`
4. **`dbo.tbl처방전`**: `조제판매ID`, `보험구분`
5. **`dbo.tbl처방병명`**: `조제판매ID`, `병명코드`, `분류기호`
6. **`dbo.tbl조제약품`**: `조제판매ID`, `고객ID`, `약품ID`, `총투약일수`, `일회투약량`, `일일투여회수`, `소모량`
7. **`dbo.TBLLib`**: `ID`(약품ID), `약품명`, `속명`, `제약사`
8. **`dbo.TBL매출의료급여자격조회`**: `조제판매ID`, `중증암/산정특례/산모 대상자 여부`
9. **`dbo.TBL진료과LIB`**: `ID`, `진료과코드1/2`

> ⚠️ 주의: 위 계획 당시 예상 칼럼명 중 일부(`총투약일수`, 중증/산정특례 단일 칼럼 등)는 실제 DB와 달랐습니다. 실제 검증 결과는 [[Test-Log]] 참고.

### 2단계: 데이터셋 마트 작성 (Data Mart Creation)
파이썬(Pandas)을 활용하여 원본 데이터를 만성/급성으로 분리하고, 예측 및 집계에 최적화된 3개의 마트를 구축합니다.

#### 시계열 분할 (Temporal Split)
미래 참조(Data Leakage)를 방지하기 위해 전체 데이터를 시간순으로 3분할 합니다.
* **Train (학습용):** 과거 2년 치
* **Validation (검증용):** 그다음 6개월 치 (과적합 방지)
* **Test (평가용):** 가장 최근 6개월 치 (최종 성능 검증)

#### [Mart 1] 내방 확률 예측 마트 (만성질환자용 ML 학습 셋)
* **필터링 대상:** `조제일수 >= 14일` 또는 `만성/중증 질환 코드 포함` 환자
* **샘플링 전략:** 클래스 불균형 해소를 위해 방문하지 않은 날(Target=0) 중 예약일 전후의 유의미한 날짜만 네거티브 샘플링하여 1:3 비율 유지.

| 데이터 구분 | 변수명 | 설명 (데이터 타입) |
| :--- | :--- | :--- |
| **입력 (X)** | 나이, 성별 | 환자 기본 프로필 (Num/Cat) |
| | 가족_총내방, 가족_총매출 | 단골 여부 지표 (Num) |
| | 마지막방문_경과일 | 최근 방문일 기준 경과 일수 (Num) |
| | 남은_약_일수 | 처방 약품 소진까지 남은 일수 (Num) |
| | 내일이_예약일 | 다음내방일과 일치 여부 (Binary) |
| | 만성질환여부 | 만성/장기 복약 여부 (Binary) |
| | 장기투약_일수 | 최근 처방 최대 투약 일수 (Num) |
| | 주요_진료과, 주요_약품속명 | 처방 특성 (Cat - 인코딩) |
| | 중증/산정특례, 보험구분 | 의료급여/중증도 상태 (Binary/Cat) |
| **정답 (Y)** | **내일_방문 (Target)** | **다음날 실제 방문 여부 (Binary: 1/0)** |
| **가중치 (Weight)** | **학습_가중치** | **중증(3.0), 만성(2.0) 등 오답 페널티 배점** |

#### [Mart 2] 고객별 최근 처방 프로필 마트 (재고 연산용)
* **목적:** 고객이 방문 시 확정적으로 타가는 약품 수량 매핑. (평균의 함정 방지)
* **구조:** `고객ID`, `약품ID`, `약품명`, **`최근 방문 시 소모량(Latest)`**

#### [Mart 3] 급성 약품 통계 마트 (시계열 연산용)
* **필터링 대상:** Mart 1에서 제외된 단기/급성 질환 처방 내역.
* **구조:** `조제일자`, `요일`, `계절`, `약품ID`, `일일 총 소모량`

### 3단계: 모델 구축 및 재고 연산 (Model Building & Inference)

#### Track 1: 머신러닝 기반 확정 수요 예측 (LightGBM/XGBoost)
1. **학습(Training):** Mart 1의 입력(X)과 정답(Y)을 넣고, 학습 가중치(Sample Weight)를 주입하여 만성/VIP 환자의 패턴을 민감하게 학습.
2. **방문 명단 생성 (Dynamic Threshold):**
   * 만성 환자는 예측 확률 `0.3` 이상, 일반 환자는 `0.7` 이상일 경우 '방문(1)'으로 판정하여 약사 확인용 사전 명단 제공.
3. **만성약 재고 산출 (기댓값 연산):**
   * 원칙: **`환자별 예측 확률(소수점) × Mart 2의 최근 소모량`**
   * 예외(품절 방지): 복용 환자 수가 극소수(예: 5명 미만)인 특수 약품은 확률이 임계값 이상이면 100% 확정 수량으로 할당.

#### Track 2: 통계 기반 급성 수요 예측
1. **급성약 재고 산출 (시계열 통계):**
   * Mart 3 데이터를 기반으로 3년 치 동월/동일 요일의 특정 약품 평균 소모량을 계산하여 예측 재고로 산출.

#### 최종 발주량 통합
* **최종 약품별 필요 수량 =** `[Track 1 만성 기댓값] + [Track 2 급성 통계 예측값]`
* 품절 방지를 위해 약국 비즈니스 환경에 맞는 **안전재고(Safety Stock, 예: +20% 또는 박스 단위 올림)**를 최종 반영하여 자동 발주 시스템으로 연계.

---

## 데이터 추출 계획서

**1. 추출 목적 및 환경**
* **목적:** 다음날 내방 예정 고객 및 약품 수요 예측 모델링을 위한 파이썬(Python) 분석용 원천 데이터(Raw Data) 확보
* **환경:** MSSQL
* **대상 기간:** 기준일(현재)로부터 최근 3년 (`내방일` 기준)
* **집계 수준:** 별도의 요약(Aggregation) 없이 조제판매 건별 원천 데이터 추출

**2. 단계별 진행 계획**

* **[1단계] 대상 테이블 및 칼럼 유효성 검증**
  * **목표:** 사전에 정의된 테이블 목록(9개)과 주요 칼럼명이 실제 MSSQL DB에 존재하는지 확인.
  * **방법:** MSSQL의 시스템 뷰(`INFORMATION_SCHEMA.COLUMNS`)를 조회하여 대상 테이블들의 전체 칼럼 명세서를 추출하고, 누락되거나 이름이 다른 칼럼 식별.

* **[2단계] 테이블 간 연결 고리(Join Key) 탐색 및 확정**
  * **목표:** `dbo.TBL진료과LIB`를 포함하여 모호한 테이블 간의 조인(Join) 키를 명확히 규명.
  * **방법:**
    1. 외래키(FK) 등 제약조건 시스템 뷰 조회.
    2. 주요 테이블(`tbl매출`, `tbl처방전` 등)과 `TBL진료과LIB`의 샘플 데이터(Top 10)를 추출하여 `ID`, `처방전발행기관ID`, `진료과코드` 등의 데이터 패턴을 직접 눈으로 비교 분석.
    3. 최종적으로 추출 쿼리에 사용할 조인 다이어그램(ERD 논리) 확정.

* **[3단계] 시계열 원천 데이터 최종 추출**
  * **목표:** 검증된 칼럼과 조인 키를 바탕으로 분석용 최종 원천 데이터 추출.
  * **방법:** `dbo.tbl매출`의 `내방일`을 기준으로 최근 3년 치 데이터를 필터링(`DATEADD(YEAR, -3, GETDATE())` 활용)하고, 확정된 관계도를 바탕으로 `LEFT JOIN`을 수행하여 하나의 큰 Raw Data 테이블 형태로 추출.

See [[Research-Log]] for modeling design details (feature engineering, class imbalance handling, weighting) and [[Test-Log]] for actual schema verification / extraction execution results.
