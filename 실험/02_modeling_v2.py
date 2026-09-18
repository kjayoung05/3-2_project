import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb
import shap
import statsmodels.api as sm

df = pd.read_csv('analysis_dataset_v2.csv')
print('분석 데이터셋 크기:', df.shape)

# -----------------------------
# 1. 결과변수(Y) 정의
# -----------------------------
df['cog_impair'] = (df['MMSE_total'] <= 23).astype(int)

# -----------------------------
# 2. 결측 제거 + 소득 로그변환
# -----------------------------
feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_count','age','male',
             'education','chronic_disease_count','smoking','drinking']
df = df.dropna(subset=feat_cols)
df['income_log'] = np.log1p(df['income'].clip(lower=0))

feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_count','age','male',
                    'education','chronic_disease_count','smoking','drinking']

X = df[feat_cols_model].reset_index(drop=True)
y = df['cog_impair'].reset_index(drop=True)
print('최종 모델링 표본 수:', len(df), ' / 저하 비율:', y.mean().round(3))

# -----------------------------
# 3. train/test 분할
# -----------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

results = {}

# -----------------------------
# 4. 로지스틱 회귀 (statsmodels, p-value/OR/CI 포함)
# -----------------------------
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

X_sm = sm.add_constant(pd.DataFrame(scaler.fit_transform(X), columns=feat_cols_model))
logit_sm = sm.Logit(y, X_sm).fit(disp=0)
summary_df = pd.DataFrame({
    'coef': logit_sm.params, 'std_err': logit_sm.bse, 'z': logit_sm.tvalues,
    'p_value': logit_sm.pvalues, 'OR': np.exp(logit_sm.params),
    'CI_2.5%': np.exp(logit_sm.conf_int()[0]), 'CI_97.5%': np.exp(logit_sm.conf_int()[1]),
})
print('\n[로지스틱 회귀 전체결과 (11개 변수)]')
print(summary_df.round(4))
summary_df.to_csv('logit_inference_table_v2.csv', encoding='utf-8-sig')

logit = LogisticRegression(max_iter=1000)
logit.fit(X_train_sc, y_train)
pred_logit = logit.predict(X_test_sc)
proba_logit = logit.predict_proba(X_test_sc)[:, 1]
results['Logistic'] = {'Accuracy': accuracy_score(y_test, pred_logit),
                        'AUC': roc_auc_score(y_test, proba_logit),
                        'F1': f1_score(y_test, pred_logit)}

# -----------------------------
# 5. 랜덤포레스트
# -----------------------------
rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
rf.fit(X_train, y_train)
pred_rf = rf.predict(X_test)
proba_rf = rf.predict_proba(X_test)[:, 1]
results['RandomForest'] = {'Accuracy': accuracy_score(y_test, pred_rf),
                            'AUC': roc_auc_score(y_test, proba_rf),
                            'F1': f1_score(y_test, pred_rf)}

# -----------------------------
# 6. XGBoost
# -----------------------------
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
xgb_model = xgb.XGBClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                                scale_pos_weight=scale_pos_weight, eval_metric='logloss', random_state=42)
xgb_model.fit(X_train, y_train)
pred_xgb = xgb_model.predict(X_test)
proba_xgb = xgb_model.predict_proba(X_test)[:, 1]
results['XGBoost'] = {'Accuracy': accuracy_score(y_test, pred_xgb),
                       'AUC': roc_auc_score(y_test, proba_xgb),
                       'F1': f1_score(y_test, pred_xgb)}

res_df = pd.DataFrame(results).T.round(3)
print('\n[모델 성능 비교 (11개 변수 버전)]')
print(res_df)
res_df.to_csv('model_performance_v2.csv', encoding='utf-8-sig')

# -----------------------------
# 7. DeLong's test
# -----------------------------
def compute_midrank(x):
    J = np.argsort(x); Z = x[J]; N = len(x); T = np.zeros(N, dtype=float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]: j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, dtype=float); T2[J] = T
    return T2

def fast_delong(preds_sorted_transposed, label_1_count):
    m = label_1_count; n = preds_sorted_transposed.shape[1] - m
    positive_examples = preds_sorted_transposed[:, :m]; negative_examples = preds_sorted_transposed[:, m:]
    k = preds_sorted_transposed.shape[0]
    tx = np.empty([k, m], dtype=float); ty = np.empty([k, n], dtype=float); tz = np.empty([k, m + n], dtype=float)
    for r in range(k):
        tx[r, :] = compute_midrank(positive_examples[r, :])
        ty[r, :] = compute_midrank(negative_examples[r, :])
        tz[r, :] = compute_midrank(preds_sorted_transposed[r, :])
    aucs = tz[:, :m].sum(axis=1) / m / n - float(m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx[:, :]) / n; v10 = 1.0 - (tz[:, m:] - ty[:, :]) / m
    sx = np.cov(v01); sy = np.cov(v10); delongcov = sx / m + sy / n
    return aucs, delongcov

def delong_roc_test(y_true, prob_a, prob_b):
    order = np.argsort(-y_true, kind='stable'); y_sorted = y_true[order]
    m = int(np.sum(y_sorted == 1)); preds = np.vstack([prob_a[order], prob_b[order]])
    aucs, delongcov = fast_delong(preds, m)
    diff = aucs[0] - aucs[1]; var = delongcov[0, 0] + delongcov[1, 1] - 2 * delongcov[0, 1]
    z = diff / np.sqrt(var)
    from scipy.stats import norm
    p = 2 * (1 - norm.cdf(abs(z)))
    return aucs[0], aucs[1], z, p

y_test_arr = y_test.values
pairs = [('Logistic', proba_logit, 'RandomForest', proba_rf),
         ('Logistic', proba_logit, 'XGBoost', proba_xgb),
         ('RandomForest', proba_rf, 'XGBoost', proba_xgb)]
print('\n[DeLong test 결과 (11개 변수 버전)]')
for name_a, pa, name_b, pb in pairs:
    auc_a, auc_b, z, p = delong_roc_test(y_test_arr, pa, pb)
    print(f'{name_a}(AUC={auc_a:.3f}) vs {name_b}(AUC={auc_b:.3f}) -> z={z:.3f}, p={p:.4f}')

# -----------------------------
# 8. SHAP 변수중요도
# -----------------------------
explainer = shap.TreeExplainer(xgb_model)
shap_values = explainer.shap_values(X_test)
mean_abs_shap = np.abs(shap_values).mean(axis=0)
shap_importance = pd.Series(mean_abs_shap, index=feat_cols_model).sort_values(ascending=False)
print('\n[SHAP 변수중요도 (11개 변수 버전)]')
print(shap_importance.round(4))
shap_importance.to_csv('shap_importance_v2.csv', header=['mean_abs_shap'], encoding='utf-8-sig')

rf_importance = pd.Series(rf.feature_importances_, index=feat_cols_model).sort_values(ascending=False)
print('\n[RandomForest 변수중요도 (11개 변수 버전)]')
print(rf_importance.round(4))

print('\n저장 완료: logit_inference_table_v2.csv / model_performance_v2.csv / shap_importance_v2.csv')
