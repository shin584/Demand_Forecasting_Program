//# Test Log

> Source: GitHub Wiki — `shin584/Demand_Forecasting_Program` (page: Test‐Log)
> Local copy for offline reference. Last synced: 2026-09-16.

## English Summary

Execution log of actually extracting and validating the data against the [[Plan]]:
- Pulled full column specs for the 9 planned tables from MSSQL and found two planned columns didn't exist as named (`총투약일수` → actual `투약일수`; the single "severe/special-eligibility" flag is really several separate columns).
- Found the join key linking `dbo.tbl매출.처방전발행기관ID` to `dbo.TBL진료과LIB.ID` via the shared 8-digit national institution code (요양기관번호 / 발행기관기호); dropped the 진료과 (department) fields since only ~3k of 220k rows had them populated.
- First raw extraction (~220k rows) had two bugs, both fixed: an Excel date-display illusion (fixed via `CAST(... AS DATE)` in SQL) and a Cartesian-product row explosion from a 1:N join on diagnosis codes (fixed via `STRING_AGG` to merge multiple codes into one cell per visit).
- A second re-extraction fixed a drug-master join error (wrong table produced nonsensical drug-to-diagnosis mappings); switching to the `CD약품정보` table produced clinically sensible prescriptions.
- Final validated dataset: 133,832 rows / 5,087 unique customers / 36,124 prescription IDs / 905 unique drugs, spanning 2023-08-28 to 2026-01-16, with 45,493 missing diagnosis codes (out of the total) and near-perfect visit-unit integrity (only 0.54% of customer+visit-date combos had multiple prescription IDs).
- Ran a drug-frequency analysis (top drug appears in 11.9% of visits, top 10 drugs cover 77%) and an 8-criteria × 3-threshold revisit-definition sensitivity analysis (revisit rate range: ~28%–75%), used to decide how "repeat visit" should be defined for labeling.

---

# 데이터 추출

## Step 1 : 주요 테이블들의 모든 칼럼명 확인

### 1. dbo.tbl가족총매출
| 컬럼명 | 데이터 타입 | 길이 | Null 허용 |
|---|---|---|---|
| 가족ID | int | NULL | NO |
| 총내방 | int | NULL | YES |
| 총매출 | money | NULL | YES |
| 총미수 | money | NULL | YES |
| 총환불 | money | NULL | YES |
| 당월내방 | int | NULL | YES |
| 당월매출 | money | NULL | YES |
| 메모 | text | 2147483647 | YES |
| 최종수정일 | datetime | NULL | YES |
| 최초내방일 | datetime | NULL | YES |
| 최종내방일 | datetime | NULL | YES |
| 장기고객관리 | tinyint | NULL | YES |

### 2. dbo.tbl고객 (주요 칼럼)
고객ID, 가족ID, 이름, 성별, 생년월일, 주소 관련 필드 다수, 만성질환, 총내방/총매출/총미수/총환불, 당월내방/당월매출, 최초내방일/최종내방일, 각종 산정특례 등록/종료일 필드 다수(중증환자, 희귀난치, 산정특례 결핵/화상/중증난치/기타염색체/잠복결핵 등), 차상위 관련 필드, 실손 관련 필드 등 총 100개 이상 칼럼 보유.
*(전체 칼럼 목록은 GitHub Wiki 원본 참고: 개인정보 관련 칼럼 다수 포함되어 있으므로 추출 시 최소 컬럼만 선별 사용)*

### 3. dbo.tbl매출 (주요 칼럼)
조제판매ID, 고객ID, 내방일, 구분, 보험, 사용자ID, 조제일수, 다음내방일, 장기조제, 교부번호, 처방전일련번호, 처방전발행기관ID, 요양급여일수, 의료급여자격조회ID 등 (총 60여개 칼럼, 대부분 조제 업무 처리용 메타데이터)

### 4. dbo.TBL매출의료급여자격조회 (주요 칼럼)
의료급여자격조회ID, 고객ID, 조제판매ID, 수진자성명, 자격여부, 진료일자, 희귀난치대상자, 산전산모대상자, 차상위대상자, 중증암등록대상자, 산정특례화상등록대상자, 동일성분의약품제한자, 산정특례결핵등록대상자, 산정특례극희귀등록대상자, 산정특례상세불명희귀등록대상자, 조산아등록대상자, 산정특례중증치매등록대상자, 산정특례중증난치등록대상자, 산정특례기타염색체등록대상자, 산정특례잠복결핵등록대상자 등 — **중증/특례 여부가 단일 칼럼이 아니라 여러 개로 세분화되어 있음이 확인됨.**

### 5. dbo.tbl조제약품 (주요 칼럼)
조제판매ID, 내방일, 고객ID, 약품ID, 약품코드, 투약일수, 판매단위, 소모량, 판매량, 일회투약량, 일일투여회수, 약품순서, 제약사, 거래처 등 — **`총투약일수`가 아니라 `투약일수`가 실제 칼럼명임.**

### 6. dbo.TBL진료과LIB
ID, 요양기관번호, 종합병원명, 의사명, 의사면허번호, 진료과코드1, 진료과코드2, 의사명2

### 7. dbo.tbl처방병명
조제판매ID, 병명코드, 분류기호, 한글병명, 영문병명, 과목, 주상병Ck, 의증Ck, 병명순서, 최종수정일

### 8. dbo.tbl처방전 (주요 칼럼)
처방전ID, 조제판매ID, 내방일, 조제일수, 보험구분, 발행일, 발행기관, 진료의명, 진료의면허번호, 질병분류기호1~5, 처방전상태 등

### 9. dbo.TBLLib
ID, 등록, 약품명, 속명, 제약사, 코드구분, 항목구분, 바코드, DI코드, 최종수정일, sUBDibCode, 향정마약구분ID

## Step 1 검증 결과
1. **완벽하게 일치하는 테이블 (수정 불필요):** dbo.tbl매출, dbo.tbl고객, dbo.tbl가족총매출, dbo.tbl처방전, dbo.tbl처방병명, dbo.TBLLib
2. **수정 및 구체화가 필요한 칼럼:**
   * `dbo.tbl조제약품`: 총투약일수 → **투약일수** (실제 칼럼명)
   * `dbo.TBL매출의료급여자격조회`: 단일 칼럼이 아니라 세분화됨 → 중증암등록대상자, 산전산모대상자, 산정특례화상등록대상자, 희귀난치대상자 등 주요 4~5개 칼럼 선별 추출로 계획 수정
3. **2단계 조인을 위한 결정적 단서 발견:** `dbo.TBL진료과LIB`에 요양기관번호, 의사면허번호 등 칼럼 확인 → `dbo.tbl처방전.진료의면허번호`/`발행기관기호` 또는 `dbo.tbl매출.처방전발행기관ID`와의 연결 키 후보로 식별.

## Step 2 : TBL진료과LIB, tbl매출 조인키 확인

**단서:** `TBL진료과LIB.요양기관번호`(8자리)와 `tbl매출.발행기관기호`(8자리)가 동일한 건강보험심사평가원 요양기관 기호 체계임을 확인.

* **관계 구조:** `TBL진료과LIB`는 전국 병원 마스터/사전 테이블(ID는 등록 순번), `tbl매출`은 거래 내역 테이블로 `처방전발행기관ID`에 마스터의 순번(ID)만 저장하는 정규화 구조.
* **결론:** `tbl매출.처방전발행기관ID` = `TBL진료과LIB.ID` (조인 확정), 매출 테이블에는 약국이 자주 쓰는 로컬 병원이 낮은 ID 번호(예: 33, 43, 46)로, 진료과LIB에는 최근 일괄 등록된 전국 병원이 높은 ID 번호(예: 18만 번대)로 존재.
* **진료과 데이터 배제 결정:** 전체 22만 건 중 진료과코드가 채워진 건 약 3천여 건뿐 → 학습 데이터로 사용 불가 판단, **진료과 테이블 제외**.

### raw data 추출 완료
* 약 22만 개의 raw data를 CSV로 추출 완료 (칼럼: 조제판매ID, 고객ID, 내방일, 처방조제일수, 다음내방일, 처방전발행기관ID, 가족ID, 성별, 생년월일, 가족총내방, 가족총매출, 보험구분, 발행기관기호, 병명코드, 분류기호, 약품ID, 투약일수, 일회투약량, 일일투여회수, 소모량, 약품명, 속명, 제약사, 중증암등록대상자, 산전산모대상자, 희귀난치대상자, 차상위대상자 등)

## raw 데이터 재추출 (버그 수정)

### 1. 날짜 데이터 서식 착시 및 노이즈 제거
* **현상:** `내방일`과 `다음내방일`이 엑셀 화면상 동일 값(예: 26:11.6)으로 표기됨.
* **원인:** DB엔 정상 `YYYY-MM-DD HH:MM:SS.sss` 값이 있었으나 엑셀 자동 포맷팅이 연/월/일을 숨기고 분:초만 노출 (뷰어 착시).
* **해결:** 일 단위 예측엔 시분초가 불필요한 노이즈 → SQL에서 `CAST(내방일 AS DATE)` 적용해 연-월-일만 남김.

### 2. 다중 병명 조인에 따른 소모량 뻥튀기(Cartesian Product) 해결
* **현상:** 한 환자가 한 방문에 여러 병명·여러 약품을 동시에 처방받을 경우 행이 기하급수적으로 증식, 소모량이 몇 배로 부풀려짐.
* **원인:** `tbl매출`(1) ↔ `tbl처방병명`(N)을 병합 함수 없이 단순 `LEFT JOIN`한 전형적인 1:N 조인 문제.
* **해결:** `STRING_AGG` 함수로 한 환자의 모든 병명을 하나의 셀에 쉼표로 병합(예: `I109,M1315`) → 정보 손실 없이 중복 제거.

### 3. 결측치(NULL) 검증
* **현상:** 과거 기록 없는 신규 환자는 `가족총내방`, `가족총매출`이 NULL로 정상 추출됨을 확인.
* **처리 방향:** 파이썬(Pandas) 전처리 단계에서 '전체 평균'이 아닌 **'성별 및 연령대별 그룹 평균'**으로 대치 예정.

## RAW 데이터 재추출 (2) — 약품 마스터 조인 오류 해결
* **문제 발견:** 단순 급성 편도염/두통(J0390, R51) 환자에게 향정신성 수면마취제(포폴주사), 외용 연고, 근이완제 주사 등 처방 상황과 무관한 이상 약품이 매핑됨.
* **원인:** 기존 마스터 테이블(`TBLLib`)의 조인 키 불일치로 엉뚱한 매핑 발생.
* **조치:** 조인 대상을 더 정교한 **`CD약품정보`** 테이블로 변경, 재추출.
* **정합성 검증 결과:** 동일 질병 환자에게 코대원에스시럽(진해거담제), 팬스타정(소염진통제), 펜잘(해열진통제) 등 호흡기 질환에 부합하는 정상 처방 세트가 정확히 출력됨을 확인.

---

# 데이터 분석

## 1단계 : 데이터 구조 및 방문 단위 검증 + 약품별 등장 빈도 분석
* **일자:** 2026-08-29
* **스크립트:** `test/validation_data/data_structure_validation.py`

### [1단계] 데이터 구조 및 방문 단위 검증
* 전체 행 수: **133,832**
* 고유 고객 수: **5,087**
* 고유 조제판매ID 수: **36,124**
* 내방일 범위: **2023-08-28 ~ 2026-01-16**
* 조제판매ID 1개당 평균 행 수: 3.70개
* 고객ID + 내방일 1개당 평균 조제판매ID 수: 1.01개

**결측치 현황**
* 고객ID: 0개 / 조제판매ID: 0개 / 내방일: 0개 / 다음내방일: 0개 / 약품명: 0개
* 병명코드: **45,493개**
* 처방조제일수: 0개

**방문 단위 무결성 진단**
* 1개 조제판매ID에 여러 내방일 존재: 0건 (0.00%)
* 1개 고객ID+내방일에 여러 조제판매ID 존재: 193건 (0.54%)

### [1단계] 약품별 등장 빈도 분석
* 전체 고유 약품 수: **905개**
* 최고 빈도 약품: 펜잘8시간이알서방정(아세트아미노펜)_(0.65g/1정) — 등장 비율 11.9%
* 상위 10개 약품 비율 합계: 77.0%
* 상위 20개 약품 비율 합계: 122.4% (중복 방문 포함 누적치)

**상위 등장 약품 (요약, 상위 10개)**
1. 펜잘8시간이알서방정(아세트아미노펜) — 11.9% (4,292건)
2. 휴메틴정(시메티딘) — 11.8% (4,263건)
3. 스락신정25밀리그램(오르페나드린염산염) — 8.4% (3,043건)
4. 트라노펜세미정 — 7.8% (2,800건)
5. 록스펜정(록소프로펜나트륨) — 7.0% (2,538건)
6. 알테렌정 — 6.4% (2,295건)
7. 렉사프로정5밀리그람(에스시탈로프람옥살산염) — 6.3% (2,280건)
8. 레일라디에스정 — 6.3% (2,262건)
9. 아빌리파이정2밀리그램(아리피프라졸) — 5.8% (2,112건)
10. 코대원포르테시럽 — 5.3% (1,928건)

*(전체 상위 30개 목록은 GitHub Wiki 원본 참고)*

**등장 비율 10% 이상:** 2개 (펜잘8시간이알서방정, 휴메틴정)
**등장 비율 20% 이상:** 0개

## 2단계 : 재방문 조건별 민감도 분석
* **일자:** 2026-08-29
* **스크립트:** `test/validation_data/Sensitivity analysis.py`
* **결과 파일:** `test/validation_data/sensitivity_analysis.csv`

**검증: 동일 방문 내 '다음내방일' 불일치 현황**
* 불일치 발생 방문 건수: 162건 (0.45%)
* 현업 DB 기입 오류 또는 분할 결제 시 발생 가능성
* **처리 방침:** 이후 로직에서는 가장 늦은 날짜(max)를 채택해 보수적으로 재방문 추적

**고빈도(10% 이상) 약품 수:** 2개 (직접 연산으로 처리)

**참고 (실행 로그):** `pandas` `DataFrameGroupBy.apply` 관련 `FutureWarning` 발생 — 그룹핑 컬럼이 향후 버전에서 연산에서 제외될 예정이라는 경고 (`include_groups=False` 권장). 결과에는 영향 없음.

### 민감도 분석 요약 (기준별 재방문율 변화, 전체 구간)

| 기준 | T1 | T2 | T3 |
| :--- | ---: | ---: | ---: |
| A: 공통 약품 1개 이상 | 74.5% | 74.5% | 68.4% |
| B: 공통 약품 2개 이상 | 60.2% | 60.2% | 54.9% |
| C: 약품 Jaccard 0.3 이상 | 71.1% | 71.0% | 64.8% |
| D: 약품 Jaccard 0.5 이상 | 66.9% | 66.9% | 60.8% |
| E: 병명 코드 교집합 | 34.9% | 34.8% | 30.5% |
| F: 병명 교집합 OR 약품 2개 이상 | 63.6% | 63.5% | 57.9% |
| G: 병명 교집합 AND Jaccard 0.3 이상 | 32.4% | 32.3% | 28.1% |
| H: 고빈도 제외 + 공통 약품 1개 이상 | 73.9% | 73.9% | 67.8% |

* 전체 상세 분석 결과는 `sensitivity_analysis.csv` 참고.
* **시사점:** 재방문(=만성/충성 고객) 정의 기준에 따라 라벨링 결과가 28%~75%까지 크게 달라지므로, [[Plan]]의 만성질환 분류 로직 확정 시 이 민감도를 반영해 기준(A~H)을 신중히 선택해야 함.
