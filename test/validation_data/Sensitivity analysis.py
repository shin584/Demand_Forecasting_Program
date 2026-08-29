import pandas as pd
import numpy as np

print("데이터를 불러오고 전처리를 시작합니다...")
df = pd.read_csv('pharmacy_raw_data_v0.2.csv')
df['내방일'] = pd.to_datetime(df['내방일'])
df['다음내방일'] = pd.to_datetime(df['다음내방일'])
max_date = df['내방일'].max()

def parse_disease_codes(value):
    if pd.isna(value):
        return set()
    return {code.strip() for code in str(value).split(',') if code.strip()}

df['병명집합'] = df['병명코드'].apply(parse_disease_codes)

# [수정 3] 동일 방문 내 '다음내방일' 불일치 사전 검증
date_conflicts = df.groupby(['고객ID', '내방일'])['다음내방일'].nunique()
conflict_count = (date_conflicts > 1).sum()
conflict_ratio = (conflict_count / len(date_conflicts)) * 100

print("\n" + "="*60)
print("📌 [검증] 동일 방문 내 '다음내방일' 불일치 현황")
print("="*60)
print(f"- 불일치 발생 방문 건수: {conflict_count:,}건 ({conflict_ratio:.2f}%)")
if conflict_count > 0:
    print("- 💡 현업 데이터베이스 기입 오류 또는 분할 결제 시 발생한 문제일 수 있습니다.")
    print("- 💡 이후 로직에서는 가장 늦은 날짜(max)를 채택하여 보수적으로 재방문을 추적합니다.")
print("="*60 + "\n")

# [수정 2 & 1] 외부 파일(csv) 의존 없이 현재 데이터로 직접 빈도 계산 (기준 10% 하향)
unique_tx = df['조제판매ID'].nunique()
valid_drugs = df.dropna(subset=['약품명']).copy()
valid_drugs['약품명'] = valid_drugs['약품명'].astype(str).str.strip()

df_freq = valid_drugs.groupby('약품명').agg(등장조제판매ID수=('조제판매ID', 'nunique')).reset_index()
df_freq['등장비율(%)'] = (df_freq['등장조제판매ID수'] / unique_tx) * 100

high_freq_drugs = set(df_freq[df_freq['등장비율(%)'] >= 10.0]['약품명'])
print(f"고빈도(10% 이상) 약품 수: {len(high_freq_drugs)}개 추출 완료 (직접 연산)\n")

# 방문 단위 구성 (추적가능여부 설정)
df['약품명'] = df['약품명'].astype(str).str.strip().replace(['', 'nan', 'None', 'NaN'], np.nan)

df_visit = df.groupby(['고객ID', '내방일']).agg(
    다음내방일=('다음내방일', 'max'), # 불일치 건에 대해 max 채택
    처방조제일수=('처방조제일수', 'max'),
    약품목록=('약품명', lambda x: set(x.dropna())),
    병명목록=('병명집합', lambda x: set().union(*x))
).reset_index()

df_visit = df_visit.sort_values(['고객ID', '내방일']).reset_index(drop=True)
df_visit['추적가능여부'] = (df_visit['다음내방일'] + pd.Timedelta(days=30)) <= max_date

# 다중 조건 동시 평가 (기본 재방문 탐색)
criteria_m = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']
criteria_t = ['T1', 'T2', 'T3']
combinations = [f"{m}_{t}" for m in criteria_m for t in criteria_t]

def find_revisits_all_criteria(patient_df):
    n = len(patient_df)
    records = []
    
    for i in range(n):
        curr = patient_df.iloc[i]
        res = {comb: pd.NaT for comb in combinations}
        found = {comb: False for comb in combinations}
        
        curr_drug = curr['약품목록']
        curr_dis = curr['병명목록']
        curr_clean_drug = curr_drug - high_freq_drugs
        
        for j in range(i + 1, n):
            if all(found.values()):
                break
                
            cand = patient_df.iloc[j]
            if cand['내방일'] <= curr['내방일']:
                continue
                
            cand_drug = cand['약품목록']
            cand_dis = cand['병명목록']
            cand_clean_drug = cand_drug - high_freq_drugs
            
            inter_drug = curr_drug & cand_drug
            union_drug = curr_drug | cand_drug
            jaccard = len(inter_drug) / len(union_drug) if union_drug else 0
            
            inter_dis = curr_dis & cand_dis
            inter_clean = curr_clean_drug & cand_clean_drug
            
            m_eval = {
                'A': len(inter_drug) >= 1,
                'B': len(inter_drug) >= 2,
                'C': jaccard >= 0.3,
                'D': jaccard >= 0.5,
                'E': len(inter_dis) >= 1,
                'F': len(inter_dis) >= 1 or len(inter_drug) >= 2,
                'G': len(inter_dis) >= 1 and jaccard >= 0.3,
                'H': len(inter_clean) >= 1
            }
            
            err_days = (cand['내방일'] - curr['다음내방일']).days
            
            t_eval = {
                'T1': err_days <= 30,
                'T2': -30 <= err_days <= 30,
                'T3': 0 <= err_days <= 30
            }
            
            for m, m_val in m_eval.items():
                if m_val:
                    for t, t_val in t_eval.items():
                        comb = f"{m}_{t}"
                        if t_val and not found[comb]:
                            res[comb] = cand['내방일']
                            found[comb] = True
                            
        records.append(res)
    return pd.DataFrame(records, index=patient_df.index)

print("전체 조건에 대한 민감도 교차 분석을 진행 중입니다 (시간이 소요될 수 있습니다)...")
tracked = df_visit.groupby('고객ID').apply(find_revisits_all_criteria).reset_index(level=0, drop=True)
df_visit = pd.concat([df_visit, tracked], axis=1)

df_valid = df_visit[df_visit['추적가능여부'] == True].copy()
bins_days = [0, 7, 14, 30, 60, float('inf')]
labels_days = ['~7일', '8~14일', '15~30일', '31~60일', '61일 이상']
df_valid['투약일수구간'] = pd.cut(df_valid['처방조제일수'], bins=bins_days, labels=labels_days)

# 통계 집계
results_list = []
total_visits = len(df_valid)

criteria_desc = {
    'A': '공통 약품 1개 이상',
    'B': '공통 약품 2개 이상',
    'C': '약품 Jaccard 0.3 이상',
    'D': '약품 Jaccard 0.5 이상',
    'E': '병명 코드 교집합',
    'F': '병명 교집합 OR 약품 2개 이상',
    'G': '병명 교집합 AND Jaccard 0.3 이상',
    'H': '고빈도 제외 + 공통 약품 1개 이상'
}

for m in criteria_m:
    for t in criteria_t:
        comb = f"{m}_{t}"
        df_valid[f'{comb}_오차'] = (df_valid[comb] - df_valid['다음내방일']).dt.days
        df_valid[f'{comb}_소요일'] = (df_valid[comb] - df_valid['내방일']).dt.days
        
        ret_count = df_valid[comb].notnull().sum()
        ret_rate = (ret_count / total_visits) * 100 if total_visits > 0 else 0
        
        results_list.append({
            '조건(영문)': m,
            '매칭조건설명': criteria_desc[m],
            '시간조건': t,
            '투약일수구간': '전체',
            '총방문': total_visits,
            '재방문수': ret_count,
            '재방문율(%)': round(ret_rate, 1),
            '평균오차(일)': round(df_valid[f'{comb}_오차'].mean(), 1),
            '중앙값(일)': round(df_valid[f'{comb}_오차'].median(), 1),
            '분산': round(df_valid[f'{comb}_오차'].var(), 1),
            '표준편차': round(df_valid[f'{comb}_오차'].std(), 1),
            '평균재방문소요일': round(df_valid[f'{comb}_소요일'].mean(), 1)
        })
        
        for group in labels_days:
            df_g = df_valid[df_valid['투약일수구간'] == group]
            g_tot = len(df_g)
            if g_tot == 0: continue
            
            g_ret_count = df_g[comb].notnull().sum()
            g_ret_rate = (g_ret_count / g_tot) * 100
            
            results_list.append({
                '조건(영문)': m,
                '매칭조건설명': criteria_desc[m],
                '시간조건': t,
                '투약일수구간': group,
                '총방문': g_tot,
                '재방문수': g_ret_count,
                '재방문율(%)': round(g_ret_rate, 1),
                '평균오차(일)': round(df_g[f'{comb}_오차'].mean(), 1),
                '중앙값(일)': round(df_g[f'{comb}_오차'].median(), 1),
                '분산': round(df_g[f'{comb}_오차'].var(), 1),
                '표준편차': round(df_g[f'{comb}_오차'].std(), 1),
                '평균재방문소요일': round(df_g[f'{comb}_소요일'].mean(), 1)
            })

df_results = pd.DataFrame(results_list)
df_results.to_csv('sensitivity_analysis.csv', index=False, encoding='utf-8-sig')

print("\n" + "="*60)
print("[민감도 분석 요약] 기준별 재방문율 변화 (전체 구간)")
print("="*60)

df_overall = df_results[df_results['투약일수구간'] == '전체']
for _, row in df_overall.iterrows():
    print(f"기준 {row['조건(영문)']} ({row['매칭조건설명']}) + {row['시간조건']} → {row['재방문율(%)']:.1f}%")

print("\n... 전체 분석 결과는 'sensitivity_analysis.csv'에 저장되었습니다.")