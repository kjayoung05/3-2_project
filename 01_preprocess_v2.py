import pandas as pd
import numpy as np

pd.set_option('display.max_columns', 50)

# -----------------------------
# 1. 원자료 로드
# -----------------------------
w09 = pd.read_excel('w09_20260413.xlsx', sheet_name='Data', engine='calamine')
w10 = pd.read_excel('w10_20260413.xlsx', sheet_name='Data', engine='calamine')

print('9차 응답자:', len(w09), ' / 10차 응답자:', len(w10))

# -----------------------------
# 2. 9차: X변수 구성 (기존 7개 + 신규 4개 = 11개)
# -----------------------------
x9 = pd.DataFrame()
x9['pid'] = w09['pid']

# --- IADL (기존과 동일) ---
iadl_cols = [f'w09C2{n:02d}' for n in range(8, 18)]
recode_iadl = {1: 0, 3: 1, 5: 2}
iadl_recoded = w09[iadl_cols].apply(lambda s: s.map(recode_iadl))
x9['IADL_total'] = iadl_recoded.sum(axis=1, min_count=10)

# --- 우울 CES-D (기존과 동일) ---
dep_cols = ['w09C142','w09C143','w09C144','w09C145','w09C146',
            'w09C147','w09C148','w09C149','w09C150','w09C151']
positive_items = ['w09C146', 'w09C149']
dep = w09[dep_cols].replace([-8, -9], np.nan).copy()
for c in positive_items:
    dep[c] = 5 - dep[c]
x9['CESD_total'] = dep.sum(axis=1, min_count=10)

# --- 사회적 교류 (기존과 동일) ---
x9['social_contact'] = 11 - w09['w09A032']

# --- 소득 (기존과 동일) ---
x9['income'] = w09['w09E147'].replace(-9, np.nan)

# --- 문화여가활동 (기존과 동일) ---
x9['leisure_count'] = w09['w09G035']

# --- 연령/성별 (기존과 동일) ---
x9['age'] = 2022 - w09['w09A002y']
x9['male'] = (w09['w09gender1'] == 1).astype('Int64')

# =============================================================
# ★ 신규 1: 학력 (w09edu, 1=초졸이하~4=대졸이상, 순서형 그대로 투입)
# =============================================================
x9['education'] = w09['w09edu']

# =============================================================
# ★ 신규 2: 만성질환 개수
#   "지난 조사 이후 새로 진단" 문항은 이미 진단받은 사람에게는 재질문하지
#   않는 skip 구조라 결측이 매우 커서(45%+), 대신 "현재 치료중 여부"
#   문항 8개를 합산 -> 결측은 "해당 질환 없음/치료 안함(0)"으로 처리
# =============================================================
chronic_cols = ['w09C009','w09C014','w09C021','w09C026','w09C031','w09C036','w09C041','w09C051']
chronic = w09[chronic_cols].replace({1: 1, 5: 0, -8: np.nan, -9: np.nan})
x9['chronic_disease_count'] = chronic.fillna(0).sum(axis=1)

# =============================================================
# ★ 신규 3: 흡연 (현재 흡연 여부, 1=예->1, 5=아니오->0)
# =============================================================
x9['smoking'] = (w09['w09C117'] == 1).astype('Int64')

# =============================================================
# ★ 신규 4: 음주 (평소 음주 여부, 1=예->1, 5=아니오->0)
# =============================================================
x9['drinking'] = (w09['w09C122'] == 1).astype('Int64')

print('\n[X변수 결측 현황]')
print(x9.isna().sum())

# -----------------------------
# 3. 10차: Y변수(MMSE) 구성 (기존과 동일)
# -----------------------------
mmse_cols = [c for c in w10.columns if c.startswith('w10C4') and c != 'w10C420']
y10 = pd.DataFrame()
y10['pid'] = w10['pid']
mmse_raw = w10[mmse_cols].replace(-8, np.nan)
mmse_scored = mmse_raw.replace(5, 0)
y10['MMSE_total'] = mmse_scored.sum(axis=1, min_count=len(mmse_cols))

print('\n[MMSE_total 분포]')
print(y10['MMSE_total'].describe())

# -----------------------------
# 4. 9차-10차 병합
# -----------------------------
merged = pd.merge(x9, y10, on='pid', how='inner')
print('\n최종 병합 표본 수:', len(merged))

# -----------------------------
# 5. 결측치 처리
# -----------------------------
print('\n[병합 후 결측치 현황]')
print(merged.isna().sum())

before = len(merged)
required_cols = ['IADL_total','CESD_total','social_contact','leisure_count','age','male',
                  'education','chronic_disease_count','smoking','drinking','MMSE_total']
merged_complete = merged.dropna(subset=required_cols)
print(f'\n완전사례(소득 제외 필수변수 기준): {len(merged_complete)} / {before}')

merged_complete.to_csv('analysis_dataset_v2.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료: analysis_dataset_v2.csv')
print(merged_complete.head())
