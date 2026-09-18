import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, f1_score
import statsmodels.api as sm

df = pd.read_csv('analysis_dataset_v2.csv')

feat_cols = ['IADL_total','CESD_total','social_contact','income','leisure_count','age','male',
             'education','chronic_disease_count','smoking','drinking']
df = df.dropna(subset=feat_cols)
df['income_log'] = np.log1p(df['income'].clip(lower=0))
feat_cols_model = ['IADL_total','CESD_total','social_contact','income_log','leisure_count','age','male',
                    'education','chronic_disease_count','smoking','drinking']

X_full = df[feat_cols_model].reset_index(drop=True)

# -----------------------------
# 여러 절단점(cutoff) 후보에 대해 반복 분석
# -----------------------------
cutoffs = [20, 21, 22, 23, 24, 25]
rows = []

for cutoff in cutoffs:
    y = (df['MMSE_total'] <= cutoff).astype(int).reset_index(drop=True)
    n_impair = y.sum()
    pct_impair = y.mean()

    X_train, X_test, y_train, y_test = train_test_split(
        X_full, y, test_size=0.2, random_state=42, stratify=y
    )

    # 로지스틱 (문화여가활동 계수 유의성 확인용)
    scaler = StandardScaler()
    X_train_sc = scaler.fit_transform(X_train)
    X_sm_train = sm.add_constant(pd.DataFrame(X_train_sc, columns=feat_cols_model))
    logit = sm.Logit(y_train.reset_index(drop=True), X_sm_train).fit(disp=0)
    leisure_p = logit.pvalues['leisure_count']
    leisure_coef = logit.params['leisure_count']

    # 랜덤포레스트 (성능 + 변수중요도 순위 확인용)
    rf = RandomForestClassifier(n_estimators=300, max_depth=6, random_state=42, class_weight='balanced')
    rf.fit(X_train, y_train)
    proba = rf.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    f1 = f1_score(y_test, rf.predict(X_test))

    importance = pd.Series(rf.feature_importances_, index=feat_cols_model).sort_values(ascending=False)
    leisure_rank = list(importance.index).index('leisure_count') + 1  # 1위=가장 중요

    rows.append({
        'cutoff': cutoff,
        'n_impair': n_impair,
        'pct_impair': round(pct_impair, 3),
        'RF_AUC': round(auc, 3),
        'RF_F1': round(f1, 3),
        'leisure_RF순위(11개중)': leisure_rank,
        'leisure_logit_계수': round(leisure_coef, 3),
        'leisure_logit_p': round(leisure_p, 4),
    })
    print(f'cutoff<={cutoff}: 저하비율={pct_impair:.1%}, AUC={auc:.3f}, '
          f'문화여가활동 RF순위={leisure_rank}/11, logit p={leisure_p:.4f}')

result_df = pd.DataFrame(rows)
print('\n' + '=' * 70)
print('[MMSE 절단점별 민감도분석 종합]')
print('=' * 70)
print(result_df.to_string(index=False))
result_df.to_csv('mmse_cutoff_sensitivity.csv', index=False, encoding='utf-8-sig')
print('\n저장 완료: mmse_cutoff_sensitivity.csv')
