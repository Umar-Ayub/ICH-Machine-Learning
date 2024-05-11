import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import numpy as np 
import warnings
warnings.filterwarnings('ignore')


df = pd.read_csv('../ich_data_w_scores_modified.csv')
# Assuming `df` is your DataFrame
score_cols = ["oICH_score", "mICH_score", "ICH_GS_score", "LSICH_score" , "ICH_FOS_score", "Max_ICH_score"]
df = df.drop(columns=score_cols).reset_index(drop=True)
# Features and Targets
X = df.drop(columns=['MORT90', 'MRS90'])
y_mort90 = df['MORT90']
y_mrs90 = df['MRS90'].apply(lambda x: 0 if x <= 3 else 1)  # Binarizing MRS90 as done previously

# Splitting the dataset into training and testing sets
X_train, X_test, y_train_mort90, y_test_mort90 = train_test_split(X, y_mort90, test_size=0.2, random_state=42)
X_train, X_test, y_train_mrs90, y_test_mrs90 = train_test_split(X, y_mrs90, test_size=0.2, random_state=42)


# Logistic Regression model
model_mort90 = LogisticRegression(max_iter=1000)  # Increasing max_iter for convergence
model_mrs90 = LogisticRegression(max_iter=1000)

# Parameter grid
param_grid = {
    'C': [0.001, 0.01, 0.1, 1, 10],  # Regularization strength
    'solver': ['liblinear', 'lbfgs']  # Algorithm to use for optimization
}

# Grid search for MORT90
grid_search_mort90 = GridSearchCV(model_mort90, param_grid, cv=5, scoring='roc_auc')
grid_search_mort90.fit(X_train, y_train_mort90)

# Grid search for MRS90
grid_search_mrs90 = GridSearchCV(model_mrs90, param_grid, cv=5, scoring='roc_auc')
grid_search_mrs90.fit(X_train, y_train_mrs90)



# Predictions
y_pred_proba_mort90 = grid_search_mort90.predict_proba(X_test)[:, 1]
y_pred_proba_mrs90 = grid_search_mrs90.predict_proba(X_test)[:, 1]

# AUC Scores
auc_mort90 = roc_auc_score(y_test_mort90, y_pred_proba_mort90)
auc_mrs90 = roc_auc_score(y_test_mrs90, y_pred_proba_mrs90)

print(f'LR: AUC for MORT90: {auc_mort90}')
print(f'LR: AUC for MRS90: {auc_mrs90}')


# Define the XGBoost classifier
xgb_clf = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Perform cross-validation for MORT90
scores_mort90 = cross_val_score(xgb_clf, X, y_mort90, cv=5, scoring='roc_auc')
print(f'MORT90 - XGBoost Cross-Validated AUC: {scores_mort90.mean()}')

# Perform cross-validation for MRS90
scores_mrs90 = cross_val_score(xgb_clf, X, y_mrs90, cv=5, scoring='roc_auc')
print(f'MRS90 - XGBoost Cross-Validated AUC: {scores_mrs90.mean()}')


# Define the Random Forest classifier
rf_clf = RandomForestClassifier()

# Perform cross-validation for MORT90
scores_mort90 = cross_val_score(rf_clf, X, y_mort90, cv=5, scoring='roc_auc')
print(f'MORT90 - Random Forest Cross-Validated AUC: {scores_mort90.mean()}')

# Perform cross-validation for MRS90
scores_mrs90 = cross_val_score(rf_clf, X, y_mrs90, cv=5, scoring='roc_auc')
print(f'MRS90 - Random Forest Cross-Validated AUC: {scores_mrs90.mean()}')


# Parameter grid for XGBoost
xgb_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    # Add more parameters as needed
}

# Parameter grid for Random Forest
rf_param_grid = {
    'n_estimators': [100, 200],
    'max_depth': [3, 5],
    # Add more parameters as needed
}

# Grid search for XGBoost - MORT90
grid_search_xgb_mort90 = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=5, scoring='roc_auc')
grid_search_xgb_mort90.fit(X, y_mort90)
print(f'Best AUC for MORT90 with XGBoost: {grid_search_xgb_mort90.best_score_}')

# Grid search for XGBoost - MRS90
grid_search_xgb_mrs90 = GridSearchCV(XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgb_param_grid, cv=5, scoring='roc_auc')
grid_search_xgb_mrs90.fit(X, y_mrs90)
print(f'Best AUC for MRS90 with XGBoost: {grid_search_xgb_mrs90.best_score_}')

# Grid search for Random Forest - MORT90
grid_search_rf_mort90 = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='roc_auc')
grid_search_rf_mort90.fit(X, y_mort90)
print(f'Best AUC for MORT90 with Random Forest: {grid_search_rf_mort90.best_score_}')

# Grid search for Random Forest - MRS90
grid_search_rf_mrs90 = GridSearchCV(RandomForestClassifier(), rf_param_grid, cv=5, scoring='roc_auc')
grid_search_rf_mrs90.fit(X, y_mrs90)
print(f'Best AUC for MRS90 with Random Forest: {grid_search_rf_mrs90.best_score_}')


# Permutation feature importance for Random Forest - MORT90
result_mort90 = permutation_importance(grid_search_rf_mort90.best_estimator_, X_test, y_test_mort90, n_repeats=10, random_state=42, n_jobs=-1)
# Permutation feature importance for Random Forest - MRS90
result_mrs90 = permutation_importance(grid_search_rf_mrs90.best_estimator_, X_test, y_test_mrs90, n_repeats=10, random_state=42, n_jobs=-1)

# Modified plot_feature_importances function to save plot
def plot_feature_importances(importances, feature_names, title, filename):
    indices = np.argsort(importances.mean(axis=1))[::-1]
    names = [feature_names[i] for i in indices]
    plt.figure(figsize=(10, 8))
    plt.title(title)
    plt.barh(range(X.shape[1]), importances.mean(axis=1)[indices])
    plt.yticks(range(X.shape[1]), names, rotation=0)
    plt.xlabel("Mean decrease in accuracy")
    plt.savefig(filename)  # Save the figure
    plt.close()

# Example usage of the modified plot_feature_importances function
# Assume result_mort90 and result_mrs90 are obtained from permutation_importance as in your script

plot_feature_importances(result_mort90.importances, X.columns, "Permutation Feature Importance (Random Forest) - MORT90", "../images/feature_importance_mort90.png")
plot_feature_importances(result_mrs90.importances, X.columns, "Permutation Feature Importance (Random Forest) - MRS90", "../images/feature_importance_mrs90.png")