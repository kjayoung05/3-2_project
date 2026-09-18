import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import shap

df = pd.read_csv('analysis_dataset_v2.csv')

# -----------------------------
# 1. 결과변수(Y) 정의 (02번과 동일)
# -----------------------------
df['cog_impair'] = (df['MMSE_total'] <= 23).astype(int)

# -----------------------------
# 2. ★ 변경점: leisure_count(횟수) -> leisure_binary(참여여부)
# -----------------------------
df['leisure_binary'] = (df['leisure_count'] > 0).astype(int)
print('[문화여가활동 참여여부 분포]')
print(df['leisure_binary'].value_counts(normalize=True).round(3))

# -----------------------------
# 3. 나머지는 02번과 동일 (income만 로그변환, leisure는 count->binary로 교체)
# -----------------------------
feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_binary','age','male',
             'education','chronic_disease_count','smoking','drinking']
df = df.dropna(subset=['IADL_total','CESD_total','social_contact','income','leisure_binary','age','male',
                        'education','chronic_disease_count','smoking','drinking'])
df['income_log'] = np.log1p(df['income'].clip(lower=0))
feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_binary','age','male',
                    'education','chronic_disease_count','smoking','drinking']

X = df[feat_cols_model]
y = df['cog_impair']
print('\n최종 모델링 표본 수:', len(df))

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# 로지스틱
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)
logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_sc, y_train)
proba_logit = logit.predict_proba(X_test_sc)[:, 1]
pred_logit = logit.predict(X_test_sc)
results['Logistic'] = {
    'Accuracy': accuracy_score(y_test, pred_logit),
    'AUC': roc_auc_score(y_test, proba_logit),
    'F1': f1_score(y_test, pred_logit)
}

# 랜덤포레스트
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
results['RandomForest'] = {
    'Accuracy': accuracy_score(y_test, pred_rf),
    'AUC': roc_auc_score(y_test, proba_rf),
    'F1': f1_score(y_test, pred_rf)
}

# XGBoost
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(
    n_estimators=300, max_depth=4, learning_rate=0.05,
    scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42
)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {
    'Accuracy': accuracy_score(y_test, pred_xgb),
    'AUC': roc_auc_score(y_test, proba_xgb),
    'F1': f1_score(y_test, pred_xgb)
}

res_df = pd.DataFrame(results).T.round(3)
print('\n[모델 성능 비교 - leisure_binary 버전]')
print(res_df)

# SHAP (XGBoost)
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feat_cols_model).sort_values(ascending=False)
print('\n[SHAP 변수중요도 - leisure_binary 버전]')
print(shap_importance.round(4))

# RandomForest 변수중요도
rf_importance = pd.Series(rf.feature_importances_, index=feat_cols_model).sort_values(ascending=False)
print('\n[RandomForest 변수중요도 - leisure_binary 버전]')
print(rf_importance.round(4))
