import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import xgboost as xgb

df = pd.read_csv('analysis_dataset_v2.csv')
df['cog_impair'] = (df['MMSE_total'] <= 23).astype(int)

feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_count','age','male',
             'education','chronic_disease_count','smoking','drinking']
df = df.dropna(subset=feat_cols)
df['income_log'] = np.log1p(df['income'].clip(lower=0))
feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_count','age','male',
                    'education','chronic_disease_count','smoking','drinking']

X = df[feat_cols_model].reset_index(drop=True)
y = df['cog_impair'].reset_index(drop=True)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# =====================================================================
# 1. 랜덤포레스트 튜닝
# =====================================================================
print('=' * 60)
print('1. 랜덤포레스트 RandomizedSearchCV (50회 탐색, 5-fold CV, scoring=AUC)')
print('=' * 60)

rf_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [4, 6, 8, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'max_features': ['sqrt', 'log2', None],
}

rf_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42, class_weight='balanced'),
    param_distributions=rf_param_dist,
    n_iter=50, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1
)
rf_search.fit(X_train, y_train)
print('최적 파라미터:', rf_search.best_params_)
print('CV 최고 AUC:', round(rf_search.best_score_, 4))

rf_best = rf_search.best_estimator_
pred = rf_best.predict(X_test)
proba = rf_best.predict_proba(X_test)[:, 1]
rf_tuned_perf = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}
print('Test셋 성능(튜닝 후):', {k: round(v, 3) for k, v in rf_tuned_perf.items()})

# =====================================================================
# 2. XGBoost 튜닝
# =====================================================================
print('\n' + '=' * 60)
print('2. XGBoost RandomizedSearchCV (50회 탐색, 5-fold CV, scoring=AUC)')
print('=' * 60)

spw = (y_train == 0).sum() / (y_train == 1).sum()
xgb_param_dist = {
    'n_estimators': [100, 200, 300, 500],
    'max_depth': [3, 4, 5, 6],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.7, 0.8, 0.9, 1.0],
}

xgb_search = RandomizedSearchCV(
    xgb.XGBClassifier(scale_pos_weight=spw, eval_metric='logloss', random_state=42),
    param_distributions=xgb_param_dist,
    n_iter=50, scoring='roc_auc', cv=cv, random_state=42, n_jobs=-1
)
xgb_search.fit(X_train, y_train)
print('최적 파라미터:', xgb_search.best_params_)
print('CV 최고 AUC:', round(xgb_search.best_score_, 4))

xgb_best = xgb_search.best_estimator_
pred = xgb_best.predict(X_test)
proba = xgb_best.predict_proba(X_test)[:, 1]
xgb_tuned_perf = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}
print('Test셋 성능(튜닝 후):', {k: round(v, 3) for k, v in xgb_tuned_perf.items()})

# =====================================================================
# 3. 튜닝 전 vs 후 비교표
# =====================================================================
print('\n' + '=' * 60)
print('3. 튜닝 전(기본값) vs 튜닝 후 성능 비교')
print('=' * 60)

default_perf = {
    'RandomForest(기본값)': {'Accuracy': 0.782, 'AUC': 0.855, 'F1': 0.699},
    'XGBoost(기본값)': {'Accuracy': 0.784, 'AUC': 0.854, 'F1': 0.694},
}
tuned_perf = {
    'RandomForest(튜닝후)': rf_tuned_perf,
    'XGBoost(튜닝후)': xgb_tuned_perf,
}
comp = pd.DataFrame({**default_perf, **tuned_perf}).T.round(3)
print(comp)
comp.to_csv('tuning_comparison.csv', encoding='utf-8-sig')

print('\n저장 완료: tuning_comparison.csv')
