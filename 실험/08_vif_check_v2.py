import pandas as pd
import numpy as np
from statsmodels.stats.outliers_influence import variance_inflation_factor
import statsmodels.api as sm

df = pd.read_csv('analysis_dataset_v2.csv')

feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_count','age','male',
             'education','chronic_disease_count','smoking','drinking']
df = df.dropna(subset=feat_cols)
df['income_log'] = np.log1p(df['income'].clip(lower=0))
feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_count','age','male',
                    'education','chronic_disease_count','smoking','drinking']

X = df[feat_cols_model].reset_index(drop=True)
X_const = sm.add_constant(X)

vif_data = pd.DataFrame()
vif_data['변수'] = X_const.columns
vif_data['VIF'] = [variance_inflation_factor(X_const.values, i) for i in range(X_const.shape[1])]

# 상수항 제외하고 출력 (상수항 VIF는 해석 대상 아님)
vif_result = vif_data[vif_data['변수'] != 'const'].sort_values('VIF', ascending=False)
print('=' * 50)
print('[다중공선성 진단: VIF]')
print('=' * 50)
print(vif_result.round(3).to_string(index=False))

print('\n판정 기준: VIF > 10 심각한 다중공선성 / VIF > 5 주의 필요 / VIF < 5 문제없음')
max_vif = vif_result['VIF'].max()
if max_vif > 10:
    print(f'\n>>> 최대 VIF={max_vif:.2f} : 다중공선성 문제 있음, 변수 재검토 필요')
elif max_vif > 5:
    print(f'\n>>> 최대 VIF={max_vif:.2f} : 경미한 다중공선성, 참고만 하면 됨')
else:
    print(f'\n>>> 최대 VIF={max_vif:.2f} : 다중공선성 문제 없음')

# 참고용: 변수 간 상관계수 행렬도 같이 출력
print('\n' + '=' * 50)
print('[참고: 변수 간 상관계수 행렬]')
print('=' * 50)
print(X.corr().round(2))

vif_result.to_csv('vif_result.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료: vif_result.csv')
