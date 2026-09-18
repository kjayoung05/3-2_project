import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

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

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc = scaler.transform(X_test)

results = {}

# -----------------------------
# 1. SVM (RBF kernel)
# -----------------------------
svm = SVC(kernel='rbf', probability=True, class_weight='balanced', random_state=42)
svm.fit(X_train_sc, y_train)
proba = svm.predict_proba(X_test_sc)[:, 1]
pred = svm.predict(X_test_sc)
results['SVM(RBF)'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 2. LightGBM
# -----------------------------
lgbm = lgb.LGBMClassifier(n_estimators=300, max_depth=4, learning_rate=0.05,
                            class_weight='balanced', random_state=42, verbose=-1)
lgbm.fit(X_train, y_train)
proba = lgbm.predict_proba(X_test)[:, 1]
pred = lgbm.predict(X_test)
results['LightGBM'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 3. 신경망 (MLP, 은닉층 2개)
# -----------------------------
mlp = MLPClassifier(hidden_layer_sizes=(32, 16), max_iter=1000, random_state=42, early_stopping=True)
mlp.fit(X_train_sc, y_train)
proba = mlp.predict_proba(X_test_sc)[:, 1]
pred = mlp.predict(X_test_sc)
results['신경망(MLP)'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 4. 단일 의사결정나무
# -----------------------------
dt = DecisionTreeClassifier(max_depth=5, class_weight='balanced', random_state=42)
dt.fit(X_train, y_train)
proba = dt.predict_proba(X_test)[:, 1]
pred = dt.predict(X_test)
results['단일의사결정나무'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 5. 정규화 로지스틱 (L1/Lasso, L2/Ridge)
# -----------------------------
lasso = LogisticRegression(penalty='l1', solver='liblinear', C=0.5, max_iter=1000)
lasso.fit(X_train_sc, y_train)
proba = lasso.predict_proba(X_test_sc)[:, 1]
pred = lasso.predict(X_test_sc)
results['로지스틱(L1/Lasso)'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

ridge = LogisticRegression(penalty='l2', C=0.5, max_iter=1000)
ridge.fit(X_train_sc, y_train)
proba = ridge.predict_proba(X_test_sc)[:, 1]
pred = ridge.predict(X_test_sc)
results['로지스틱(L2/Ridge)'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 6. 나이브베이즈
# -----------------------------
nb = GaussianNB()
nb.fit(X_train_sc, y_train)
proba = nb.predict_proba(X_test_sc)[:, 1]
pred = nb.predict(X_test_sc)
results['나이브베이즈'] = {'Accuracy': accuracy_score(y_test, pred), 'AUC': roc_auc_score(y_test, proba), 'F1': f1_score(y_test, pred)}

# -----------------------------
# 기존 3개 모델 결과(참고용, 기존과 동일 split)
# -----------------------------
results['로지스틱(기존, 무규제)'] = {'Accuracy': 0.788, 'AUC': 0.841, 'F1': 0.638}
results['랜덤포레스트(기존)'] = {'Accuracy': 0.782, 'AUC': 0.855, 'F1': 0.699}
results['XGBoost(기존)'] = {'Accuracy': 0.784, 'AUC': 0.854, 'F1': 0.694}

res_df = pd.DataFrame(results).T.round(3)
res_df = res_df.sort_values('AUC', ascending=False)
print('[전체 9개 모델 성능 비교, AUC 내림차순]')
print(res_df)
res_df.to_csv('additional_models_comparison.csv', encoding='utf-8-sig')
print('\n저장 완료: additional_models_comparison.csv')

# Lasso 계수로 어떤 변수가 0으로 수렴했는지 확인 (변수선택 결과)
print('\n[L1/Lasso 로지스틱 계수 - 0에 가까운 변수는 자동 제외된 것]')
lasso_coef = pd.Series(lasso.coef_[0], index=feat_cols_model).sort_values(key=abs, ascending=False)
print(lasso_coef.round(4))
