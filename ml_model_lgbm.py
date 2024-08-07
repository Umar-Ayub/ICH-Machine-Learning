import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import warnings
warnings.filterwarnings('ignore')

# Load the data
df = pd.read_csv('../ich_data_w_scores_modified.csv')

# Drop specified score columns and reset index
score_cols = ["oICH_score", "mICH_score", "ICH_GS_score", "LSICH_score", "ICH_FOS_score", "Max_ICH_score"]
df = df.drop(columns=score_cols).reset_index(drop=True)

# Features and targets
X = df.drop(columns=['MORT90', 'MRS90'])
y_mort90 = df['MORT90']
y_mrs90 = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)  # Binarizing MRS90

# Splitting the dataset into training and testing sets
X_train, X_test, y_train_mort90, y_test_mort90 = train_test_split(X, y_mort90, test_size=0.2, random_state=42)
X_train, X_test, y_train_mrs90, y_test_mrs90 = train_test_split(X, y_mrs90, test_size=0.2, random_state=42)

# Define the LightGBM classifier
lgbm = lgb.LGBMClassifier()

# Parameter grid for LightGBM
lgbm_param_grid = {
    'num_leaves': [31, 62],
    'reg_alpha': [0.1, 0.5],
    'min_data_in_leaf': [30, 50, 100],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200]
}

# Grid search for LightGBM - MORT90
grid_search_lgbm_mort90 = GridSearchCV(lgbm, lgbm_param_grid, cv=5, scoring='roc_auc')
grid_search_lgbm_mort90.fit(X_train, y_train_mort90)
print(f'Best AUC for MORT90 with LightGBM: {grid_search_lgbm_mort90.best_score_}')

# Grid search for LightGBM - MRS90
grid_search_lgbm_mrs90 = GridSearchCV(lgbm, lgbm_param_grid, cv=5, scoring='roc_auc')
grid_search_lgbm_mrs90.fit(X_train, y_train_mrs90)
print(f'Best AUC for MRS90 with LightGBM: {grid_search_lgbm_mrs90.best_score_}')

# Predictions
y_pred_proba_lgbm_mort90 = grid_search_lgbm_mort90.predict_proba(X_test)[:, 1]
y_pred_proba_lgbm_mrs90 = grid_search_lgbm_mrs90.predict_proba(X_test)[:, 1]

# AUC Scores for LightGBM
auc_lgbm_mort90 = roc_auc_score(y_test_mort90, y_pred_proba_lgbm_mort90)
auc_lgbm_mrs90 = roc_auc_score(y_test_mrs90, y_pred_proba_lgbm_mrs90)

print(f'LightGBM: AUC for MORT90: {auc_lgbm_mort90}')
print(f'LightGBM: AUC for MRS90: {auc_lgbm_mrs90}')
