import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb

# -----------------------------
# 1. 가중치 변수 불러와서 병합
#    w09wgt_ap = 9차조사 "통합 종단가중치"
#    (KLoSA는 원표본+신규표본이 섞여있어 '통합' 가중치를 사용해야
#     신규표본(refresh cohort)도 함께 대표성 있게 반영됨)
# -----------------------------
w09_wgt = pd.read_excel('w09_20260413.xlsx', sheet_name='Data',
                          usecols=['pid', 'w09oldnew', 'w09wgt_ap'], engine='calamine')

df = pd.read_csv('analysis_dataset_v2.csv')
df = df.merge(w09_wgt, on='pid', how='left')

print('가중치 결측(신규표본 등 종단가중치 미부여) 인원:', df['w09wgt_ap'].isna().sum(), '/', len(df))

df['cog_impair'] = (df['MMSE_total'] <= 23).astype(int)

feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_count','age','male',
             'education','chronic_disease_count','smoking','drinking']
df['income_log'] = np.log1p(df['income'].clip(lower=0))
feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_count','age','male',
                    'education','chronic_disease_count','smoking','drinking']

df_complete = df.dropna(subset=feat_cols_model + ['w09wgt_ap'])
print('가중분석 최종 표본 수:', len(df_complete), '(가중치 결측 제외)')

X = df_complete[feat_cols_model].reset_index(drop=True)
y = df_complete['cog_impair'].reset_index(drop=True)
w = df_complete['w09wgt_ap'].reset_index(drop=True)

# =====================================================================
# 2. 가중 로지스틱 회귀 (survey weight 반영, GLM + freq 근사)
# =====================================================================
print('\n' + '=' * 60)
print('2. 가중 로지스틱 회귀 (statsmodels GLM, 조사가중치 반영)')
print('=' * 60)

scaler = StandardScaler()
X_scaled = pd.DataFrame(scaler.fit_transform(X), columns=feat_cols_model)
X_sm = sm.add_constant(X_scaled)

# 무가중 (비교 기준)
logit_unweighted = sm.Logit(y, X_sm).fit(disp=0)

# 가중 (조사가중치를 var_weights로 반영)
logit_weighted = sm.GLM(y, X_sm, family=sm.families.Binomial(),
                          var_weights=w / w.mean()).fit()

compare = pd.DataFrame({
    '무가중_계수': logit_unweighted.params,
    '무가중_p': logit_unweighted.pvalues,
    '가중_계수': logit_weighted.params,
    '가중_p': logit_weighted.pvalues,
})
print(compare.round(4))
compare.to_csv('weighted_vs_unweighted_logit.csv', encoding='utf-8-sig')

# =====================================================================
# 3. 가중 vs 무가중 모델 성능/변수중요도 비교 (RF, XGBoost)
# =====================================================================
print('\n' + '=' * 60)
print('3. 가중치 반영 모델 성능 비교 (RF, XGBoost)')
print('=' * 60)

X_train, X_test, y_train, y_test, w_train, w_test = train_test_split(
    X, y, w, test_size=0.2, random_state=42, stratify=y
)

results = {}

# 무가중 RF
rf_unw = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
rf_unw.fit(X_train, y_train)
p = rf_unw.predict_proba(X_test)[:, 1]
results['RF_무가중'] = {'AUC': roc_auc_score(y_test, p), 'F1': f1_score(y_test, rf_unw.predict(X_test))}

# 가중 RF (표본가중치를 sample_weight로 반영)
rf_w = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
rf_w.fit(X_train, y_train, sample_weight=w_train)
p = rf_w.predict_proba(X_test)[:, 1]
results['RF_가중'] = {'AUC': roc_auc_score(y_test, p), 'F1': f1_score(y_test, rf_w.predict(X_test))}

# 무가중 XGBoost
spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_unw = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                              scale_pos_weight=spw, eval_metric='logloss', random_state=42)
xgb_unw.fit(X_train, y_train)
p = xgb_unw.predict_proba(X_test)[:, 1]
results['XGB_무가중'] = {'AUC': roc_auc_score(y_test, p), 'F1': f1_score(y_test, xgb_unw.predict(X_test))}

# 가중 XGBoost
xgb_w = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            scale_pos_weight=spw, eval_metric='logloss', random_state=42)
xgb_w.fit(X_train, y_train, sample_weight=w_train)
p = xgb_w.predict_proba(X_test)[:, 1]
results['XGB_가중'] = {'AUC': roc_auc_score(y_test, p), 'F1': f1_score(y_test, xgb_w.predict(X_test))}

res_df = pd.DataFrame(results).T.round(4)
print(res_df)
res_df.to_csv('weighted_model_performance.csv', encoding='utf-8-sig')

# 가중 RF 변수중요도
rf_w_importance = pd.Series(rf_w.feature_importances_, index=feat_cols_model).sort_values(ascending=False)
print('\n[가중 RF 변수중요도]')
print(rf_w_importance.round(4))

print('\n저장 완료: weighted_vs_unweighted_logit.csv / weighted_model_performance.csv')
