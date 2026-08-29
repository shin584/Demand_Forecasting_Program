import pandas as pd
import numpy as np

# 1. 데이터 로드 및 기본 전처리
df = pd.read_csv('pharmacy_raw_data_v0.2.csv')
df['내방일'] = pd.to_datetime(df['내방일'])

# 2. 데이터 기본 구조 및 방문 단위 검증 산출
total_rows = len(df)
unique_customers = df['고객ID'].nunique()
unique_tx = df['조제판매ID'].nunique()
date_min = df['내방일'].min().strftime('%Y-%m-%d')
date_max = df['내방일'].max().strftime('%Y-%m-%d')

missing_values = df[['고객ID', '조제판매ID', '내방일', '다음내방일', '약품명', '병명코드', '처방조제일수']].isnull().sum()

rows_per_tx_mean = df.groupby('조제판매ID').size().mean()
rows_per_cust_date_mean = df.groupby(['고객ID', '내방일']).size().mean()

# 조제판매ID 1개당 내방일 개수 검증
dates_per_tx = df.groupby('조제판매ID')['내방일'].nunique()
tx_multiple_dates_count = (dates_per_tx > 1).sum()
tx_multiple_dates_ratio = (tx_multiple_dates_count / unique_tx) * 100

# 고객ID+내방일 1개당 조제판매ID 개수 검증
tx_per_cust_date = df.groupby(['고객ID', '내방일'])['조제판매ID'].nunique()
cust_date_total_combinations = len(tx_per_cust_date)
cust_date_multiple_tx_count = (tx_per_cust_date > 1).sum()
cust_date_multiple_tx_ratio = (cust_date_multiple_tx_count / cust_date_total_combinations) * 100

# 3. 약품명 전처리
df['약품명'] = df['약품명'].astype(str).str.strip()
df['약품명'] = df['약품명'].replace(['', 'nan', 'None', 'NaN'], np.nan)

# 4 & 5. 약품별 등장 빈도 분석
valid_drugs = df.dropna(subset=['약품명'])
df_drug_frequency = valid_drugs.groupby('약품명').agg(
    등장행수=('조제판매ID', 'count'),
    등장조제판매ID수=('조제판매ID', 'nunique')
).reset_index()

df_drug_frequency['등장비율(%)'] = (df_drug_frequency['등장조제판매ID수'] / unique_tx) * 100
df_drug_frequency = df_drug_frequency.sort_values(by='등장비율(%)', ascending=False).reset_index(drop=True)

unique_drug_count = len(df_drug_frequency)
top1_drug = df_drug_frequency.iloc[0]['약품명'] if unique_drug_count > 0 else "없음"
top1_drug_ratio = df_drug_frequency.iloc[0]['등장비율(%)'] if unique_drug_count > 0 else 0.0

top10_ratio_sum = df_drug_frequency.head(10)['등장비율(%)'].sum()
top20_ratio_sum = df_drug_frequency.head(20)['등장비율(%)'].sum()

drugs_over_10 = df_drug_frequency[df_drug_frequency['등장비율(%)'] >= 10]
drugs_over_20 = df_drug_frequency[df_drug_frequency['등장비율(%)'] >= 20]

# 6. 결과 저장
df_drug_frequency.to_csv('drug_frequency.csv', index=False)

# 7. 출력 형식
print("=" * 60)
print("[1단계] 데이터 구조 및 방문 단위 검증")
print("=" * 60)
print(f"- 전체 행 수: {total_rows:,}")
print(f"- 고유 고객 수: {unique_customers:,}")
print(f"- 고유 조제판매ID 수: {unique_tx:,}")
print(f"- 내방일 범위: {date_min} ~ {date_max}")
print(f"- 조제판매ID 1개당 평균 행 수: {rows_per_tx_mean:.2f}개")
print(f"- 고객ID + 내방일 1개당 평균 조제판매ID 수: {tx_per_cust_date.mean():.2f}개\n")

print("[결측치 현황]")
for col, val in missing_values.items():
    print(f"  * {col}: {val:,}개")

print("\n[방문 단위 무결성 진단]")
print(f"- 1개 조제판매ID에 여러 내방일 존재 건수: {tx_multiple_dates_count:,}건 ({tx_multiple_dates_ratio:.2f}%)")
print(f"- 1개 고객ID+내방일에 여러 조제판매ID 존재 건수: {cust_date_multiple_tx_count:,}건 ({cust_date_multiple_tx_ratio:.2f}%)")

print("\n" + "=" * 60)
print("[1단계] 약품별 등장 빈도 분석")
print("=" * 60)
print(f"- 전체 고유 약품 수: {unique_drug_count:,}개")
print(f"- 최고 빈도 약품: {top1_drug}")
print(f"- 최고 빈도 약품 등장 비율: {top1_drug_ratio:.1f}%")
print(f"- 상위 10개 약품 비율 합계: {top10_ratio_sum:.1f}%")
print(f"- 상위 20개 약품 비율 합계: {top20_ratio_sum:.1f}%\n")

print("상위 30개 약품")
for i, row in df_drug_frequency.head(30).iterrows():
    print(f"{i+1}. {row['약품명']} (비율: {row['등장비율(%)']:.1f}% | 고유방문: {row['등장조제판매ID수']:,}건)")

print(f"\n[등장 비율 10% 이상] - 총 {len(drugs_over_10)}개")
for i, row in drugs_over_10.iterrows():
    print(f"- {row['약품명']}: {row['등장비율(%)']:.1f}%")

print(f"\n[등장 비율 20% 이상] - 총 {len(drugs_over_20)}개")
for i, row in drugs_over_20.iterrows():
    print(f"- {row['약품명']}: {row['등장비율(%)']:.1f}%")